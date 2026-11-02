#!/usr/bin/env python
"""Retrain a GVAE from scratch on the initial pool plus one search
trajectory's acquisitions: the held-out rows of the search-results table.

    # an active-learning trajectory (its final pool.pkl: 28,000 + 16,000)
    python scripts/retrain_on_pool.py --pool results/al/al_mc_dropout_vae_s0/pool.pkl

    # a GP-BO trajectory (acquisitions only; the initial pool is added)
    python scripts/retrain_on_pool.py --pool results/bo/bo_pool_ucb_seed0.pkl

Every strategy ends with a 44,000-graph pool, so the rows differ only in
which points were acquired. For an AL pool this reproduces the model the loop
itself would train at the start of a 33rd cycle (same pool, same seed). The held-out sets, the initial 28,000-graph pool
and the two-stage training protocol are those of scripts/al_loop.py.

The pool type is detected automatically: an AL pool.pkl starts with the
initial pool (verified by fingerprint), a GP-BO pool holds acquisitions only.
Acquisitions that coincide with a held-out graph are dropped. GP-BO splits its
budget evenly across topologies while the AL loop reallocates; the topology
mix of the acquisitions is recorded in run.json.

Writes <out-root>/<run-id>/run.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter


def default_seed(path: str) -> int:
    """The training seed the paper used for this pool (see --seed)."""
    import csv
    import re
    run_dir = os.path.dirname(os.path.abspath(path))
    if os.path.basename(path) == "pool.pkl":
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
        rows = list(csv.DictReader(open(os.path.join(run_dir, "cycles.csv"))))
        # the seed al_loop.py gives the model trained on this final pool,
        # i.e. the model of the next cycle: seed*1000 + cycles completed
        cseed = int(cfg["seed"]) * 1000 + len(rows)
        return cseed * 10 if cfg["strategy"] == "deep_ensemble" else cseed
    m = re.search(r"seed(\d+)", os.path.basename(path))
    if not m:
        raise SystemExit("cannot infer a seed from %s; pass --seed" % path)
    return int(m.group(1))


def default_run_id(path: str, seed: int) -> str:
    """al_mc_dropout_vae_s0/pool.pkl -> retrain_al_mc_dropout_vae_s0;
    bo_pool_ucb_seed0.pkl -> retrain_bo_ucb_seed0."""
    name = os.path.basename(path)
    if name == "pool.pkl":
        return "retrain_" + os.path.basename(os.path.dirname(os.path.abspath(path)))
    return "retrain_" + name.replace("bo_pool_", "bo_").replace(".pkl", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True,
                    help="an AL run's pool.pkl, or bo_pool_<acq>_seed<k>.pkl")
    ap.add_argument("--seed", type=int, default=None,
                    help="training seed. Default: for an AL pool, the seed "
                         "al_loop.py would use for a model trained on the final pool "
                         "(seed*1000 + cycles completed, x10 for deep_ensemble, "
                         "i.e. ensemble member 0); for a GP-BO pool, its BO seed")
    ap.add_argument("--model-type", default="vae",
                    choices=["vae", "ae", "gat_classifier", "gcn_classifier"])
    ap.add_argument("--hparams", default="configs/gvae.json")
    ap.add_argument("--block", default="auto")
    ap.add_argument("--dataset", default="data/natural.pkl")
    ap.add_argument("--ext", default="data/extrapolation.pkl")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--stage0-epochs", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out-root", default="results/retrained")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    import torch

    from coolgraphs.activelearning.loop import (check_no_leakage,
                                                default_stage0, evaluate,
                                                heldout_sets, train_on_pool)
    from coolgraphs.training import data as data_mod
    from coolgraphs.training.hparams import check_required, load_hparams
    from coolgraphs.utils.fingerprint import fingerprint
    from coolgraphs.utils.io import label_of
    from coolgraphs.utils.topology import topo_name, topology_of

    if args.seed is None:
        args.seed = default_seed(args.pool)
    print("training seed %d" % args.seed)
    run_id = args.run_id or default_run_id(args.pool, args.seed)
    run_dir = os.path.join(args.out_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    params, hp_meta = load_hparams(args.hparams, args.block)
    check_required(params, args.hparams)

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
    initial_fp = [fingerprint(e) for e in initial]

    with open(args.pool, "rb") as f:
        raw = pickle.load(f)

    # An AL pool.pkl is initial + acquisitions, in that order.
    n0 = len(initial)
    if len(raw) >= n0 and [fingerprint(e) for e in raw[:n0]] == initial_fp:
        source, acquired = "al", raw[n0:]
    else:
        source, acquired = "bo", raw
    print("\n%s: %d graphs -> %s pool, %d acquisitions"
          % (args.pool, len(raw), source.upper(), len(acquired)))

    # GP-BO entries carry the label the oracle returned; check it agrees.
    for e in acquired[:200]:
        if "is_positive" in e and int(label_of(e)) != int(e["is_positive"]):
            raise SystemExit("graph label disagrees with the recorded "
                             "is_positive flag; refusing to train.")

    kept, dropped = [], 0
    for e in acquired:
        if fingerprint(e) in heldout_fp:
            dropped += 1
            continue
        kept.append(e)
    n_pos_acq = sum(int(label_of(e)) for e in kept)
    mix = Counter(topo_name(topology_of(e)) for e in kept)
    print("  dropped %d colliding with a held-out graph; kept %d, %d positive"
          % (dropped, len(kept), n_pos_acq))

    pool = initial + kept
    n_pos = sum(int(label_of(e)) for e in pool)
    print("training pool: %d initial + %d acquired = %d graphs, %d positive "
          "(%.2f%%)" % (len(initial), len(kept), len(pool), n_pos,
                        100.0 * n_pos / len(pool)))

    stage0 = (args.stage0_epochs if args.stage0_epochs is not None
              else default_stage0(args.model_type))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, stats, info = train_on_pool(
        pool, heldout["val"], params, args.model_type, args.seed,
        args.max_epochs, args.patience, stage0, args.num_workers,
        ckpt_dir=os.path.join(run_dir, "ckpt"))
    model = model.to(device)
    print("trained: %d epochs, val PR-AUC %s"
          % (info["epochs_run"], info.get("val_pr_auc_logged")))

    metrics = evaluate(model, {k: v for k, v in heldout.items() if k != "val"},
                       stats, device, args.batch_size)
    print("\n%-6s %10s %10s" % ("split", "ROC-AUC", "PR-AUC"))
    for name, m in metrics.items():
        if m["roc_auc"] is not None:
            print("%-6s %10.4f %10.4f" % (name, m["roc_auc"], m["pr_auc"]))

    doc = {
        "run_id": run_id, "pool_file": os.path.abspath(args.pool),
        "source": source, "seed": args.seed, "model_type": args.model_type,
        "split_seed": args.split_seed, "params": params,
        "hparams_source": hp_meta,
        "n_initial": len(initial), "n_acquired_loaded": len(acquired),
        "n_acquired_kept": len(kept),
        "n_dropped_heldout_collision": dropped,
        "n_acquired_positives": int(n_pos_acq),
        "n_pool": len(pool), "n_pool_positives": int(n_pos),
        "acquired_topology_mix": dict(sorted(mix.items())),
        "epochs_run": info["epochs_run"],
        "hit_max_epochs": info.get("hit_max_epochs"),
        "metrics": metrics,
    }
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(doc, f, indent=2, default=str)
    print("\nwrote %s" % os.path.join(run_dir, "run.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
