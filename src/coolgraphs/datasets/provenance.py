"""An append-only ledger recording how a dataset was built.

Every generation run appends one entry: seeds, oracle calls, CPU time,
per-topology hit rates and the code version that ran.

Lives next to the dataset as ``<dataset>_provenance.json``:

    {
      "dataset": "data/natural.pkl",
      "created_at": "...",
      "totals": {...},          # cumulative across every run
      "current": {...},         # the dataset as it stands now
      "runs": [ {...}, {...} ]  # append-only, one per invocation
    }
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time


def provenance_path(dataset_path: str) -> str:
    return os.path.splitext(dataset_path)[0] + "_provenance.json"


def _git_commit() -> str | None:
    """Commit of the repo this code lives in, if it is a git checkout."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(["git", "-C", here, "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=5)
            return sha + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        pass
    return None


def environment() -> dict:
    """What ran this, so a future reader can tell whether it would run the same."""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    for mod in ("numpy", "scipy", "torch"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    return {k: v for k, v in env.items() if v is not None}


def load(dataset_path: str) -> dict:
    """Existing ledger, or a fresh skeleton if this is the first run."""
    p = provenance_path(dataset_path)
    if os.path.exists(p):
        try:
            with open(p) as f:
                led = json.load(f)
            led.setdefault("runs", [])
            return led
        except Exception as e:
            print(f"  [warn] unreadable provenance {p} ({e}); starting a new one")
    return {"dataset": dataset_path,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runs": []}


def append_run(dataset_path: str, run: dict, current: dict) -> str:
    """Append one run entry, refresh the totals, and write atomically."""
    led = load(dataset_path)
    led["dataset"] = dataset_path
    run = {"run": len(led["runs"]) + 1, **run}
    led["runs"].append(run)

    led["totals"] = {
        "runs": len(led["runs"]),
        "graphs_generated": sum(r.get("graphs_added", 0) for r in led["runs"]),
        "oracle_calls": sum(r.get("oracle_calls", 0) for r in led["runs"]),
        "oracle_failures": sum(r.get("oracle_failures", 0) for r in led["runs"]),
        "wall_seconds": round(sum(r.get("wall_seconds", 0) for r in led["runs"]), 1),
        "wall_hours": round(sum(r.get("wall_seconds", 0)
                                for r in led["runs"]) / 3600, 3),
        "cpu_seconds": round(sum(r.get("cpu_seconds", 0) for r in led["runs"]), 1),
        "cpu_hours": round(sum(r.get("cpu_seconds", 0)
                               for r in led["runs"]) / 3600, 3),
    }
    led["current"] = current
    led["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    p = provenance_path(dataset_path)
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(led, f, indent=2, default=str)
    os.replace(tmp, p)
    return p
