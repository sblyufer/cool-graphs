"""Append-only JSON record of training runs (config, dataset, timings,
environment), written next to the run outputs.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time


def ledger_path(root: str, name: str = "training_ledger.json") -> str:
    return os.path.join(root, name)


def _git_commit() -> str | None:
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


def gpu_info() -> dict:
    """Which GPU this ran on."""
    info: dict = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
    except Exception:
        pass
    return info


def environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    for mod in ("numpy", "scipy", "sklearn", "torch", "pytorch_lightning", "optuna"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    env.update(gpu_info())
    return {k: v for k, v in env.items() if v is not None}


def dataset_fingerprint(dataset_path: str) -> dict:
    """Identify a dataset by its provenance ledger (cheap; no hashing)."""
    fp: dict = {"path": dataset_path}
    prov = os.path.splitext(dataset_path)[0] + "_provenance.json"
    if not os.path.exists(prov):
        fp["warning"] = ("no provenance ledger beside this dataset; its origin "
                         "cannot be established from this record")
        return fp
    try:
        with open(prov) as f:
            led = json.load(f)
        cur = led.get("current", {})
        tot = led.get("totals", {})
        fp.update({
            "n_graphs": cur.get("n_graphs"),
            "n_positives": cur.get("n_positives"),
            "positive_rate": cur.get("positive_rate"),
            "generation_runs": tot.get("runs"),
            "oracle_calls": tot.get("oracle_calls"),
            "oracle_cpu_hours": tot.get("cpu_hours"),
            "seeds": [r.get("seed") for r in led.get("runs", [])],
            "updated_at": led.get("updated_at"),
        })
    except Exception as e:
        fp["warning"] = f"could not read {prov}: {e}"
    return fp


def load(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                led = json.load(f)
            led.setdefault("runs", [])
            return led
        except Exception:
            # Never lose a ledger to a truncated write; move it aside.
            os.replace(path, path + f".corrupt.{int(time.time())}")
    return {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}


def append_run(path: str, run: dict) -> dict:
    """Add one run and rewrite the ledger atomically."""
    led = load(path)
    entry = dict(run)
    entry["run"] = len(led["runs"]) + 1
    entry.setdefault("recorded_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    entry.setdefault("environment", environment())
    led["runs"].append(entry)

    gpu_s = sum(r.get("gpu_seconds", 0) or 0 for r in led["runs"])
    led["totals"] = {
        "runs": len(led["runs"]),
        "gpu_seconds": round(gpu_s, 1),
        "gpu_hours": round(gpu_s / 3600, 3),
        "wall_seconds": round(sum(r.get("wall_seconds", 0) or 0
                                  for r in led["runs"]), 1),
        "epochs": sum(r.get("epochs_run", 0) or 0 for r in led["runs"]),
    }
    led["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(led, f, indent=2, default=str)
    os.replace(tmp, path)
    return led


class Timer:
    """Wall-clock timing; ``gpu_seconds`` is wall time when a GPU is in use."""

    def __init__(self):
        self.wall_seconds = 0.0
        self.gpu_seconds = 0.0
        self._t0 = None

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.wall_seconds = round(time.time() - self._t0, 1)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.gpu_seconds = self.wall_seconds
        except Exception:
            pass
        return False
