"""Dataset generation by i.i.d. sampling from the parameter prior.

Each topology is filled to an exact quota, so every topology carries equal
weight. Work is split into chunks, each with its own deterministic task seed
(``seed + 100_000 * topology_index + 1000 * chunk_index``); a dataset is
therefore reproducible from its base seed, and an interrupted or topped-up run
continues without redrawing parameters it has already used.

Chunks are written to ``<out>.shards/`` as they finish and merged at the end,
which makes a killed run resumable and keeps tensors out of the
multiprocessing pipes.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Iterable

import numpy as np

from ..utils.topology import topo_name, topology_of


def _sampler_and_runner():
    """Import the physics lazily (it pulls in scipy and torch)."""
    from .sampling import ParameterSampler, SimulationRunner
    return ParameterSampler(), SimulationRunner()


def generate_random(size: tuple[int, int], n_want: int, seed: int,
                    max_attempts: int | None = None) -> tuple[list, int, int]:
    """Produce exactly ``n_want`` valid samples of ONE topology.

    Returns (entries, oracle_calls, oracle_failures). Failed oracle calls
    (e.g. a singular rate matrix) are skipped and redrawn, so the count is
    exact unless ``max_attempts`` is exhausted.
    """
    np.random.seed(seed)
    random.seed(seed)
    sampler, runner = _sampler_and_runner()

    ni, nj = size
    n_sites = ni * nj
    max_attempts = max_attempts or n_want * 200

    out, attempts, failures = [], 0, 0
    while len(out) < n_want and attempts < max_attempts:
        attempts += 1
        params = sampler.sample_continuous()
        params["W"] = sampler.sample_W(ni, nj)
        params["alpha_vec"] = sampler.sample_alpha_base(n_sites, params["beta0"])
        params["ni"], params["nj"] = ni, nj
        try:
            graph, _flux = runner.run(params)
        except Exception:
            failures += 1
            continue
        out.append({"graph": graph, "ni": ni, "nj": nj})
    return out, attempts, failures


def shard_dir_for(out_path: str) -> str:
    """Where a run's per-task shards live."""
    return os.path.splitext(out_path)[0] + ".shards"


def shard_path(out_path: str, size: tuple[int, int], task_seed: int) -> str:
    return os.path.join(shard_dir_for(out_path),
                        f"{size[0]}x{size[1]}_{task_seed}.pkl")


def generate_shard(task) -> tuple:
    """Generate one chunk and write it to disk; return only small values.

    Returning the graphs would push torch tensors through the multiprocessing
    pipe (one file descriptor per tensor), which exhausts descriptors on long
    runs. Writing a shard avoids that and doubles as the resume mechanism.
    """
    task_seed, size, n_want, out_path = task
    path = shard_path(out_path, size, task_seed)
    if os.path.exists(path):
        # Written by an earlier, killed run; its cost was already recorded then.
        return {"path": path, "size": size, "n": 0, "calls": 0,
                "failures": 0, "seconds": 0.0, "reused": True}

    t0 = time.time()
    entries, calls, failures = generate_random(size, n_want, task_seed)
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    import pickle as pkl
    with open(tmp, "wb") as f:
        pkl.dump(entries, f)
    os.replace(tmp, path)            # atomic: a partial shard is never visible
    # Per-shard cost, so a resumed run still totals correctly.
    with open(os.path.splitext(path)[0] + ".stats.json", "w") as f:
        json.dump({"topology": f"{size[0]}x{size[1]}", "seed": task_seed,
                   "n": len(entries), "calls": calls, "failures": failures,
                   "seconds": round(elapsed, 3)}, f)
    return {"path": path, "size": size, "n": len(entries), "calls": calls,
            "failures": failures, "seconds": elapsed, "reused": False}


def load_shards(out_path: str) -> list:
    """Every completed shard for this run, concatenated."""
    import glob
    import pickle as pkl

    d = shard_dir_for(out_path)
    if not os.path.isdir(d):
        return []
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.pkl"))):
        if p.endswith(".stats.json"):
            continue
        try:
            with open(p, "rb") as f:
                out.extend(pkl.load(f))
        except Exception as e:
            print(f"  [warn] unreadable shard {p}: {e}")
    return out


def shard_stats(out_path: str) -> dict:
    """Total cost recorded by shards currently on disk (including a killed run's)."""
    import glob
    d = shard_dir_for(out_path)
    tot = {"n": 0, "calls": 0, "failures": 0, "seconds": 0.0, "shards": 0}
    for p in glob.glob(os.path.join(d, "*.stats.json")):
        try:
            with open(p) as f:
                s = json.load(f)
            tot["n"] += s.get("n", 0)
            tot["calls"] += s.get("calls", 0)
            tot["failures"] += s.get("failures", 0)
            tot["seconds"] += s.get("seconds", 0.0)
            tot["shards"] += 1
        except Exception:
            pass
    return tot


def used_seeds_path(out_path: str) -> str:
    return os.path.splitext(out_path)[0] + "_seeds.json"


def load_used_seeds(out_path: str) -> set:
    """Task seeds this dataset has already been generated from.

    Tasks are shuffled before dispatch and a run stopped by its time budget
    merges whatever finished, so the completed chunks are a scattered subset.
    Recording the seeds (rather than inferring them from counts) is what stops
    a resumed run from drawing the same parameters twice.
    """
    p = used_seeds_path(out_path)
    if not os.path.exists(p):
        return set()
    try:
        with open(p) as f:
            return set(int(s) for s in json.load(f).get("used", []))
    except Exception as e:
        raise SystemExit(
            f"cannot read {p} ({e}).\nIt records which task seeds this "
            f"dataset was generated from; without it a top-up can repeat "
            f"them. Fix or delete the file deliberately, then rerun.")


def record_used_seeds(out_path: str, seeds) -> str:
    """Add task seeds to the record, atomically. Never removes any."""
    p = used_seeds_path(out_path)
    have = load_used_seeds(out_path)
    merged = sorted(have | set(int(s) for s in seeds))
    tmp = p + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump({"dataset": out_path, "n_used": len(merged),
                   "note": ("every task seed this dataset has been generated "
                            "from; plan_tasks skips these so a resumed or "
                            "repeated top-up cannot redraw the same "
                            "parameters"),
                   "used": merged}, f, indent=2)
    os.replace(tmp, p)
    return p


def shard_seeds(out_path: str) -> set:
    """Task seeds of the shards currently on disk."""
    import glob
    out = set()
    for p in glob.glob(os.path.join(shard_dir_for(out_path), "*.stats.json")):
        try:
            with open(p) as f:
                s = json.load(f)
            if "seed" in s:
                out.add(int(s["seed"]))
        except Exception:
            # fall back to the filename: <ni>x<nj>_<seed>.stats.json
            base = os.path.basename(p).split(".")[0]
            if "_" in base and base.rsplit("_", 1)[1].isdigit():
                out.add(int(base.rsplit("_", 1)[1]))
    return out


def clear_shards(out_path: str) -> None:
    import shutil
    d = shard_dir_for(out_path)
    if os.path.isdir(d):
        shutil.rmtree(d)


def plan_tasks(sizes: Iterable[tuple[int, int]], per_topology: int,
               existing: list, chunk: int, seed: int,
               out_path: str | None = None,
               used: set | None = None) -> tuple[list, dict, int]:
    """Work out what still needs generating, per topology.

    Returns (tasks, have, total_todo) where each task is
    (task_seed, size, n_want). Chunk numbering continues from the quota
    already filled (and skips recorded seeds), so a top-up never repeats the
    parameter draws of an earlier run.
    """
    sizes = list(sizes)
    have = {s: 0 for s in sizes}
    for e in existing:
        t = topology_of(e)
        if t in have:
            have[t] += 1

    tasks, total = [], 0
    for i, size in enumerate(sizes):
        deficit = per_topology - have[size]
        if deficit <= 0:
            continue
        total += deficit
        n_chunks = (deficit + chunk - 1) // chunk
        c0 = have[size] // chunk
        # Topologies are spaced 100_000 apart and chunks 1_000 apart, so chunk
        # index 100 would collide with the next topology's seed range.
        if c0 + n_chunks > 100:
            raise SystemExit(
                f"{topo_name(size)}: chunk indices would reach "
                f"{c0 + n_chunks}, past the 100-chunk stride between "
                f"topologies -- two topologies would draw identical "
                f"parameters.\nRaise --chunk so the quota needs fewer chunks "
                f"(--chunk {max(chunk * 2, (per_topology + 99) // 100)} or more)."
            )
        made, c = 0, 0
        while made < n_chunks:
            if c0 + c >= 100:
                raise SystemExit(
                    f"{topo_name(size)}: ran out of chunk indices below the "
                    f"100-chunk stride between topologies while looking for "
                    f"{n_chunks} unused ones.\nRaise --chunk so the quota "
                    f"needs fewer chunks, or generate into a new file with a "
                    f"fresh --seed.")
            # Distinct seed per (topology, chunk): no two tasks draw the same
            # parameters, and the whole run is reproducible from `seed`.
            task_seed = seed + 100_000 * i + 1000 * (c0 + c)
            c += 1
            if used and task_seed in used:
                continue
            # Skip an index whose shard is still on disk from a killed run; it
            # will be merged rather than regenerated.
            if out_path and os.path.exists(shard_path(out_path, size, task_seed)):
                continue
            tasks.append((task_seed, size, min(chunk, deficit - made * chunk)))
            made += 1
    return tasks, have, total
