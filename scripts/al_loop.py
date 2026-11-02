#!/usr/bin/env python
"""One active-learning trajectory: one strategy, one model, one seed.

    python scripts/al_loop.py --strategy mc_dropout    --seed 0
    python scripts/al_loop.py --strategy random        --seed 0
    python scripts/al_loop.py --strategy deep_ensemble --members 5 --seed 0
    python scripts/al_loop.py --strategy mc_dropout --random-period 11 --seed 0
    python scripts/al_loop.py --strategy mc_dropout --seed 0 --smoke   # plumbing check

Each cycle retrains the surrogate from scratch on the current pool, draws
--candidates fresh parameter sets from the prior, selects the --acquire with
the highest predictive variance (or uniformly at random), labels them with
the oracle and adds them to the pool. The starting pool is the 28,000-graph
training split of data/natural.pkl; validation, test and extrapolation sets
are the same held-out sets used by scripts/train.py. Any acquisition that
coincides with a held-out graph is dropped (and still counted as an oracle
call).

Cycle c is seeded with seed*1000 + c, so a resumed run (--resume) uses the
same seeds as an uninterrupted one. Floating-point non-determinism in
training means trajectories are statistically, not bitwise, reproducible.

Writes to <out-root>/<run-id>/:
    config.json                  inputs of the run
    cycles.csv                   one row per cycle: acquisitions, positives,
                                 cumulative oracle calls, held-out metrics
    discovery_by_topology.csv    one row per (cycle, topology)
    acquired/cycle_###.json      every acquired point with its parameters,
                                 uncertainty, flux and label
    pool.pkl                     the grown pool
    ckpt/                        per-cycle checkpoints (with --keep-checkpoints)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import sys
import time

import numpy as np

CYCLE_FIELDS = [
    "cycle", "strategy_this_cycle",
    "n_candidates", "n_scored", "n_acquired", "n_positive",
    "n_oracle_failures", "n_dropped_heldout_collision",
    "acquired_positive_rate",
    "cum_oracle_calls", "cum_positives", "cum_positive_rate",
    "calls_per_positive",
    "pool_size", "pool_positives", "pool_positive_rate",
    "epochs_run", "hit_max_epochs",
    "val_pr_auc", "val_roc_auc",
    "test_pr_auc", "test_roc_auc",
    "ext_pr_auc", "ext_roc_auc",
    "train_seconds", "score_seconds", "oracle_seconds", "cycle_seconds",
]


def atomic_pickle(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def read_cycles(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_cycle(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CYCLE_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", required=True,
                   choices=["mc_dropout", "deep_ensemble", "random"])
    p.add_argument("--model-type", default="vae",
                   choices=["vae", "ae", "gat_classifier", "gcn_classifier"])
    p.add_argument("--seed", type=int, required=True, help="0-999")
    p.add_argument("--cycles", type=int, default=32)
    p.add_argument("--candidates", type=int, default=10000)
    p.add_argument("--acquire", type=int, default=500)
    p.add_argument("--mc-passes", type=int, default=20)
    p.add_argument("--members", type=int, default=5,
                   help="ensemble size (deep_ensemble only)")
    p.add_argument("--random-period", type=int, default=0,
                   help="P > 0 makes every P-th cycle a random cycle "
                        "(the exploration ablation uses 11); 0 = off")
    p.add_argument("--hparams", default="configs/gvae.json")
    p.add_argument("--block", default="auto")
    p.add_argument("--dataset", default="data/natural.pkl")
    p.add_argument("--ext", default="data/extrapolation.pkl")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--stage0-epochs", type=int, default=None)
    p.add_argument("--score-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-root", default="results/al")
    p.add_argument("--run-id", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--keep-checkpoints", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="2 cycles, 200 candidates, 20 acquired, 3 MC passes, "
                        "6 epochs, into results/al_smoke")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if not (0 <= args.seed <= 999):
        raise SystemExit("--seed must be in 0-999 (cycle seeds are "
                         "seed*1000 + cycle).")
    if args.smoke:
        args.cycles, args.candidates, args.acquire = 2, 200, 20
        args.mc_passes, args.max_epochs, args.patience = 3, 6, 3
        args.members = min(args.members, 2)
        args.out_root = "results/al_smoke"

    import torch

    from coolgraphs.activelearning.loop import (check_no_leakage,
                                                default_stage0, evaluate,
                                                heldout_sets, jsonable,
                                                sample_candidates,
                                                score_deep_ensemble,
                                                score_mc_dropout,
                                                train_on_pool)
    from coolgraphs.datasets.normalise import normalize_graphs
    from coolgraphs.datasets.sampling import ParameterSampler, SimulationRunner
    from coolgraphs.physics.graphs import create_graph
    from coolgraphs.training import data as data_mod
    from coolgraphs.training import ledger as L
    from coolgraphs.training.hparams import check_required, load_hparams
    from coolgraphs.utils.fingerprint import fingerprint
    from coolgraphs.utils.io import label_of
    from coolgraphs.utils.topology import topology_of

    params, hp_meta = load_hparams(args.hparams, args.block)
    check_required(params, args.hparams)

    run_id = args.run_id or ("al_%s_%s_s%d%s" % (
        args.strategy, args.model_type, args.seed,
        "" if not args.random_period else "_rp%d" % args.random_period))
    run_dir = os.path.join(args.out_root, run_id)
    acq_dir = os.path.join(run_dir, "acquired")
    os.makedirs(acq_dir, exist_ok=True)
    cycles_csv = os.path.join(run_dir, "cycles.csv")
    topo_csv = os.path.join(run_dir, "discovery_by_topology.csv")
    pool_path = os.path.join(run_dir, "pool.pkl")
    ckpt_dir = os.path.join(run_dir, "ckpt")

    print("=== %s ===" % run_id)
    print("strategy %s | model %s | seed %d | cycles %d | candidates %d | "
          "acquire %d" % (args.strategy, args.model_type, args.seed,
                          args.cycles, args.candidates, args.acquire))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- held-out sets (shared with scripts/train.py) and starting pool ---
    ext = args.ext if os.path.exists(args.ext) else None
    base = data_mod.prepare(args.dataset, batch_size=64,
                            split_seed=args.split_seed, ext_path=ext,
                            num_workers=args.num_workers, verbose=True)
    check_no_leakage(base)
    heldout = heldout_sets(base)
    heldout_fp = set()
    for es in heldout.values():
        heldout_fp |= {fingerprint(e) for e in es}

    initial = list(base.entries("train"))
    done = read_cycles(cycles_csv) if args.resume else []
    start_cycle = len(done)

    if start_cycle and os.path.exists(pool_path):
        with open(pool_path, "rb") as f:
            pool = pickle.load(f)
        # the history is authoritative if a kill left the pool ahead of it
        expect = len(initial) + sum(int(r["n_acquired"]) for r in done)
        pool = pool[:expect]
        print("resuming at cycle %d with a pool of %d" % (start_cycle, len(pool)))
    else:
        if start_cycle:
            raise SystemExit("cycles.csv exists but %s is missing; cannot "
                             "resume." % pool_path)
        pool = list(initial)
        n0 = sum(label_of(e) for e in pool)
        print("starting pool: %d graphs, %d positives (%.2f%%)"
              % (len(pool), n0, 100.0 * n0 / max(1, len(pool))))

    if start_cycle >= args.cycles:
        print("all %d cycles already recorded." % args.cycles)
        return 0

    # Every oracle call counts toward the budget, including failures and
    # dropped held-out collisions.
    cum_calls = sum(int(r["n_acquired"]) + int(r["n_oracle_failures"])
                    + int(r.get("n_dropped_heldout_collision") or 0)
                    for r in done)
    cum_pos = sum(int(r["n_positive"]) for r in done)

    stage0 = (args.stage0_epochs if args.stage0_epochs is not None
              else default_stage0(args.model_type))

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "run_id": run_id, "strategy": args.strategy,
            "model_type": args.model_type, "seed": args.seed,
            "cycles": args.cycles, "candidates_per_cycle": args.candidates,
            "acquire_per_cycle": args.acquire, "mc_passes": args.mc_passes,
            "members": args.members if args.strategy == "deep_ensemble" else None,
            "random_period": args.random_period,
            "hparams_source": hp_meta, "params": params,
            "dataset": args.dataset, "ext": ext,
            "split_seed": args.split_seed, "max_epochs": args.max_epochs,
            "patience": args.patience, "stage0_epochs": stage0,
            "initial_pool_size": len(initial),
            "initial_pool_positives": int(sum(label_of(e) for e in initial)),
            "heldout_sizes": {k: len(v) for k, v in heldout.items()},
            "environment": L.environment(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2, default=str)

    sampler = ParameterSampler()
    runner = SimulationRunner()

    for cycle in range(start_cycle, args.cycles):
        t_cycle = time.time()
        cseed = args.seed * 1000 + cycle
        rng = np.random.default_rng(cseed)
        random.seed(cseed)

        this_strategy = args.strategy
        if args.random_period and (cycle % args.random_period
                                   == args.random_period - 1):
            this_strategy = "random"

        print("\n" + "=" * 70)
        print("cycle %d/%d  (seed %d, strategy %s, pool %d)"
              % (cycle + 1, args.cycles, cseed, this_strategy, len(pool)))
        print("=" * 70)

        # --- train on the current pool --------------------------------------
        t0 = time.time()
        if this_strategy == "deep_ensemble":
            models, infos = [], []
            for m in range(args.members):
                mdl, stats, info = train_on_pool(
                    pool, heldout["val"], params, args.model_type,
                    cseed * 10 + m, args.max_epochs, args.patience, stage0,
                    args.num_workers, ckpt_dir=ckpt_dir)
                models.append(mdl.to(device))
                infos.append(info)
            model = models[0]
            info = {"epochs_run": int(np.mean([i["epochs_run"] for i in infos])),
                    "hit_max_epochs": any(i["hit_max_epochs"] for i in infos),
                    "best_ckpts": [i["best_ckpt"] for i in infos]}
        else:
            model, stats, info = train_on_pool(
                pool, heldout["val"], params, args.model_type, cseed,
                args.max_epochs, args.patience, stage0, args.num_workers,
                ckpt_dir=ckpt_dir)
            model = model.to(device)
            models = [model]
        train_s = time.time() - t0

        # --- held-out metrics of this cycle's model -------------------------
        metrics = evaluate(model, heldout, stats, device, args.score_batch_size)
        for name in sorted(metrics):
            m = metrics[name]
            print("  %-5s PR-AUC %s  ROC-AUC %s" % (
                name,
                "--" if m["pr_auc"] is None else "%.4f" % m["pr_auc"],
                "--" if m["roc_auc"] is None else "%.4f" % m["roc_auc"]))

        # --- candidates, scoring, selection ---------------------------------
        cands = sample_candidates(sampler, args.candidates, rng)
        t0 = time.time()
        if this_strategy == "random":
            order = rng.permutation(len(cands))[:args.acquire]
            scores = np.zeros(len(cands))
            means = np.full(len(cands), np.nan)
            chosen = sorted(int(i) for i in order)
            n_scored = 0
        else:
            graphs, keep = [], []
            for i, p in enumerate(cands):
                try:
                    graphs.append(create_graph(**p))
                    keep.append(i)
                except Exception:
                    continue
            if not graphs:
                raise SystemExit("no candidate graph could be built")
            if this_strategy == "deep_ensemble":
                v, mu = score_deep_ensemble(models, graphs, stats, device,
                                            args.score_batch_size,
                                            normalize_graphs)
            else:
                v, mu = score_mc_dropout(model, graphs, stats, device,
                                         args.mc_passes, args.score_batch_size,
                                         normalize_graphs)
            scores = np.full(len(cands), -np.inf)
            means = np.full(len(cands), np.nan)
            scores[keep] = v
            means[keep] = mu
            n_scored = len(keep)
            chosen = [int(i) for i in np.argsort(-scores)[:args.acquire]
                      if np.isfinite(scores[i])]
        score_s = time.time() - t0

        # --- oracle ---------------------------------------------------------
        t0 = time.time()
        added, record = [], []
        n_fail = n_collide = 0
        for i in chosen:
            p = cands[i]
            base_rec = {"candidate_index": int(i),
                        "uncertainty": float(scores[i]),
                        "mean_prediction": (None if np.isnan(means[i])
                                            else float(means[i])),
                        "params": jsonable(p)}
            try:
                g, flux = runner.run(p)
            except Exception as e:
                n_fail += 1
                record.append({**base_rec, "oracle_failed": True,
                               "error": str(e)[:200]})
                continue
            entry = {"graph": g, "ni": p["ni"], "nj": p["nj"]}
            if fingerprint(entry) in heldout_fp:
                n_collide += 1
                record.append({**base_rec, "dropped_heldout_collision": True})
                continue
            added.append(entry)
            record.append({**base_rec, "topology": topology_of(entry),
                           "heat_flux": float(flux),
                           "label": int(label_of(entry))})
        oracle_s = time.time() - t0

        n_pos = sum(r.get("label", 0) for r in record)
        cum_calls += len(added) + n_fail + n_collide
        cum_pos += n_pos
        pool = pool + added
        atomic_pickle(pool, pool_path)

        with open(os.path.join(acq_dir, "cycle_%03d.json" % cycle), "w") as f:
            json.dump({"cycle": cycle, "seed": cseed, "strategy": this_strategy,
                       "n_candidates": args.candidates, "n_scored": n_scored,
                       "acquired": record}, f, indent=2)

        pool_pos = int(sum(label_of(e) for e in pool))
        row = {
            "cycle": cycle, "strategy_this_cycle": this_strategy,
            "n_candidates": args.candidates, "n_scored": n_scored,
            "n_acquired": len(added), "n_positive": n_pos,
            "n_oracle_failures": n_fail,
            "n_dropped_heldout_collision": n_collide,
            "acquired_positive_rate": (n_pos / len(added)) if added else 0.0,
            "cum_oracle_calls": cum_calls, "cum_positives": cum_pos,
            "cum_positive_rate": (cum_pos / cum_calls) if cum_calls else 0.0,
            "calls_per_positive": (cum_calls / cum_pos) if cum_pos else None,
            "pool_size": len(pool), "pool_positives": pool_pos,
            "pool_positive_rate": pool_pos / max(1, len(pool)),
            "epochs_run": info["epochs_run"],
            "hit_max_epochs": info["hit_max_epochs"],
            "train_seconds": round(train_s, 1),
            "score_seconds": round(score_s, 1),
            "oracle_seconds": round(oracle_s, 1),
            "cycle_seconds": round(time.time() - t_cycle, 1),
        }
        for name in ("val", "test", "ext"):
            m = metrics.get(name, {})
            row["%s_pr_auc" % name] = m.get("pr_auc")
            row["%s_roc_auc" % name] = m.get("roc_auc")
        append_cycle(cycles_csv, row)

        by_t = {}
        for r in record:
            if "topology" in r:
                s = by_t.setdefault(r["topology"], [0, 0])
                s[0] += 1
                s[1] += r["label"]
        new = not os.path.exists(topo_csv)
        with open(topo_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["cycle", "strategy_this_cycle", "topology",
                            "n_acquired", "n_positive", "positive_rate"])
            for t in sorted(by_t):
                n, k = by_t[t]
                w.writerow([cycle, this_strategy, t, n, k, (k / n) if n else 0.0])

        print("  acquired %d, positives %d (%.1f%%), failures %d, "
              "held-out collisions %d"
              % (len(added), n_pos, 100.0 * n_pos / max(1, len(added)),
                 n_fail, n_collide))
        print("  cumulative: %d calls, %d positives" % (cum_calls, cum_pos))

        if not args.keep_checkpoints:
            for ck in info.get("best_ckpts") or [info.get("best_ckpt")]:
                if ck and os.path.exists(ck):
                    try:
                        os.remove(ck)
                    except OSError:
                        pass

    print("\n%d cycles done -> %s" % (args.cycles, run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
