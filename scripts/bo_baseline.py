#!/usr/bin/env python
"""GP-BO baseline at the same oracle budget as the AL loop (32 x 500 = 16,000).

    python scripts/bo_baseline.py --acq EI  --seed 0
    python scripts/bo_baseline.py --acq UCB --seed 0

One independent GP per training topology, the budget split evenly; each GP
starts from --n-init random points, scores --n-cand fresh candidates per step
and simulates the best one. Kernel hyperparameters are re-optimised every
--refit-every steps (the posterior is updated every step) and each GP is fit
on at most --max-gp points.

Writes, under --out-dir:
    bo_<acq>_seed<k>.csv        one row per oracle call (cumulative positives)
    bo_pool_<acq>_seed<k>.pkl   every labelled graph, for scripts/retrain_on_pool.py
"""
import argparse
import csv
import os
import pickle
import random

import numpy as np

from coolgraphs.activelearning.bo import SIZE_LIST, physics_oracle, run_topology


def main(oracle_fn=physics_oracle):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acq", default="EI", choices=["EI", "UCB"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=16000,
                    help="total oracle calls across all topologies")
    ap.add_argument("--sizes", nargs="+", default=None,
                    help="subset of topologies, e.g. 1x2 2x2 (default: all 8)")
    ap.add_argument("--alpha-lo", type=float, default=-0.1)
    ap.add_argument("--alpha-hi", type=float, default=1.2)
    ap.add_argument("--n-init", type=int, default=12,
                    help="random initial points per topology")
    ap.add_argument("--n-cand", type=int, default=1200,
                    help="candidate pool scored per step")
    ap.add_argument("--kappa", type=float, default=2.0, help="UCB exploration weight")
    ap.add_argument("--xi", type=float, default=0.01, help="EI exploration margin")
    ap.add_argument("--max-gp", type=int, default=300,
                    help="cap on the GP training-set size")
    ap.add_argument("--refit-every", type=int, default=25,
                    help="re-optimise kernel hyperparameters every K steps")
    ap.add_argument("--out-dir", default="results/bo")
    args = ap.parse_args()

    random.seed(args.seed)          # ParameterSampler draws from the global streams
    np.random.seed(args.seed)
    sizes = (SIZE_LIST if not args.sizes
             else [tuple(map(int, s.split("x"))) for s in args.sizes])
    rng = np.random.default_rng(args.seed)
    per = args.budget // len(sizes)
    extra = args.budget - per * len(sizes)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.acq.lower()}_seed{args.seed}"
    out = os.path.join(args.out_dir, f"bo_{tag}.csv")
    counters = {"calls": 0, "pos": 0}
    pool = []
    fields = ["n_oracle_calls", "cum_positives", "positive_ratio", "is_positive",
              "objective", "topology", "model_type", "acq", "seed", "space"]

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for k, (ni, nj) in enumerate(sizes):
            budget = per + (1 if k < extra else 0)
            print(f"[bo_{args.acq.lower()}] topology {ni}x{nj}: {budget} calls",
                  flush=True)
            run_topology(ni, nj, budget, args.acq, oracle_fn, rng,
                         (args.alpha_lo, args.alpha_hi), args.n_init, args.n_cand,
                         args.kappa, args.xi, args.max_gp, counters, writer,
                         args.seed, refit_every=args.refit_every, pool=pool)

    pkl = os.path.join(args.out_dir, f"bo_pool_{tag}.pkl")
    with open(pkl, "wb") as fh:
        pickle.dump(pool, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {len(pool)} graphs -> {pkl}")
    print(f"Done. {counters['pos']}/{counters['calls']} positive -> {out}")


if __name__ == "__main__":
    main()
