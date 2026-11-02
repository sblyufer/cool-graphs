#!/usr/bin/env python
"""Generate a labelled graph dataset by i.i.d. sampling from the prior.

Every topology is filled to the same exact count. Generation is resumable:
rerun the same command after an interruption, or with a larger
--per-topology to top a dataset up. A <out>_provenance.json ledger records
seeds, oracle calls, CPU time and per-topology hit rates.

    # smoke test (seconds)
    python scripts/generate_dataset.py --per-topology 25 --out data/smoke.pkl

    # training set: 8 topologies x 5,000 = 40,000 graphs
    python scripts/generate_dataset.py --per-topology 5000 --out data/natural.pkl

    # extrapolation set: unseen topologies
    python scripts/generate_dataset.py --topologies 1x5,4x2 --per-topology 6000 \
        --seed 54321 --out data/extrapolation.pkl
    python scripts/generate_dataset.py --topologies 3x4 --per-topology 3000 \
        --seed 54322 --chunk 50 --out data/extrapolation.pkl
"""
# Pin BLAS threads before numpy is imported: one thread per worker process.
import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

from coolgraphs.datasets import provenance  # noqa: E402
from coolgraphs.datasets.generate import (clear_shards, generate_shard,  # noqa: E402
                                          load_shards, load_used_seeds,
                                          plan_tasks, record_used_seeds,
                                          shard_dir_for, shard_seeds,
                                          shard_stats)
from coolgraphs.utils.io import label_of, load_entries, save_entries  # noqa: E402
from coolgraphs.utils.topology import describe, parse_topologies, topo_name  # noqa: E402

# The eight training topologies.
DEFAULT_TOPOLOGIES = "1x2,1x3,1x4,2x2,2x3,3x2,3x3,2x4"


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--out", required=True, help="output .pkl path")
    ap.add_argument("--per-topology", type=int, required=True,
                    help="exact number of samples per topology")
    ap.add_argument("--topologies", default=DEFAULT_TOPOLOGIES,
                    help=f"comma-separated list (default: {DEFAULT_TOPOLOGIES})")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--chunk", type=int, default=100, help="samples per task")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--time-budget-hours", type=float, default=None,
                    help="stop cleanly after this long; rerun to continue")
    args = ap.parse_args()

    sizes = sorted(parse_topologies(args.topologies.split(",")))

    existing = load_entries(args.out) if os.path.exists(args.out) else []
    if existing:
        print(f"Resuming: {len(existing)} entries already in {args.out}")
    leftover = load_shards(args.out)          # from an interrupted run
    if leftover:
        print(f"Resuming: {len(leftover)} entries in unmerged shards "
              f"({shard_dir_for(args.out)})")
    accounted = existing + leftover

    used_seeds = load_used_seeds(args.out)
    tasks, have, todo = plan_tasks(sizes, args.per_topology, accounted,
                                   args.chunk, args.seed, args.out,
                                   used=used_seeds)

    print(f"target {args.per_topology} samples x {len(sizes)} topologies")
    for s in sizes:
        if have[s]:
            print(f"    {topo_name(s)}: have {have[s]}, "
                  f"need {max(0, args.per_topology - have[s])}")
    if not tasks:
        print("Every topology already at quota.")
        if leftover:
            save_entries(accounted, args.out)
            clear_shards(args.out)
            print(f"Merged {len(leftover)} shard entries into {args.out}")
        describe(load_entries(args.out), args.out)
        return 0

    tasks = [(seed, size, want, args.out) for seed, size, want in tasks]
    # Spread the expensive topologies across workers.
    random.Random(args.seed).shuffle(tasks)
    print(f"Generating {todo} samples over {len(tasks)} tasks "
          f"on {args.workers} workers\n")

    t0 = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    deadline = t0 + args.time_budget_hours * 3600 if args.time_budget_hours else None

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **_):
            return it

    def out_of_time():
        return deadline is not None and time.time() > deadline

    n_done, stopped_early = 0, False
    if args.workers == 1:
        for t in tqdm(tasks, desc="chunks"):
            generate_shard(t)
            n_done += 1
            if out_of_time():
                stopped_early = True
                break
    else:
        with mp.Pool(args.workers, maxtasksperchild=20) as pool:
            for _ in tqdm(pool.imap_unordered(generate_shard, tasks),
                          total=len(tasks), desc="chunks"):
                n_done += 1
                if out_of_time():
                    stopped_early = True
                    pool.terminate()
                    break

    if stopped_early:
        print("\nTime budget reached; rerun the same command to continue.")

    cost = shard_stats(args.out)        # read before the shards are removed
    n_pos_before = sum(1 for e in existing if label_of(e) == 1)

    print(f"\n{n_done}/{len(tasks)} chunks complete; merging shards...")
    dataset = existing + load_shards(args.out)
    save_entries(dataset, args.out)
    record_used_seeds(args.out, shard_seeds(args.out))
    clear_shards(args.out)

    wall = time.time() - t0
    print(f"Done in {wall / 60:.1f} min -> {args.out}\n")
    rates = describe(dataset, args.out)

    n_pos = sum(r["pos"] for r in rates.values())
    status = ("stopped_early" if stopped_early
              else "complete" if n_done == len(tasks) else "partial")
    led = provenance.append_run(
        args.out,
        run={
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_seconds": round(wall, 1),
            "cpu_seconds": round(cost["seconds"], 1),
            "status": status,
            "seed": args.seed,
            "per_topology_target": args.per_topology,
            "topologies": args.topologies,
            "chunk": args.chunk,
            "workers": args.workers,
            "chunks_planned": len(tasks),
            "chunks_done": n_done,
            "graphs_before": len(existing),
            "graphs_added": len(dataset) - len(existing),
            "graphs_after": len(dataset),
            "positives_before": n_pos_before,
            "positives_after": n_pos,
            "oracle_calls": cost["calls"],
            "oracle_failures": cost["failures"],
            "command": " ".join(sys.argv),
            "environment": provenance.environment(),
        },
        current={
            "n_graphs": len(dataset),
            "n_positives": n_pos,
            "positive_rate": n_pos / len(dataset) if dataset else 0.0,
            "per_topology": {topo_name(t): {"n": r["n"], "pos": r["pos"],
                                            "rate": round(r["rate"], 5)}
                             for t, r in rates.items()},
        },
    )
    print(f"\nprovenance -> {led}")

    # 0: done; 75: out of time but made progress (rerun to continue)
    if status == "complete":
        return 0
    return 75 if len(dataset) > len(existing) else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
