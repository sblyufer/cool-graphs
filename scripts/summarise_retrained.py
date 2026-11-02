#!/usr/bin/env python3
"""Held-out rows of the search-results table: mean +/- s.d. over trajectories.

    python scripts/summarise_retrained.py      # reads results/retrained/*/run.json

Groups the runs written by scripts/retrain_on_pool.py by strategy (the run id
without its seed) and reports the pool size and test / extrapolation metrics.
Standard library only.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict


def fmt(v):
    v = [x for x in v if x is not None]
    if not v:
        return "--"
    return "%.3f +- %.3f" % (st.mean(v), st.stdev(v)) if len(v) > 1 else "%.3f" % v[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/retrained")
    args = ap.parse_args()

    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(args.root, "*", "run.json"))):
        doc = json.load(open(p))
        # strip the seed: ..._s0 (AL) or ..._seed0 (GP-BO)
        key = re.sub(r"_(s|seed)\d+(?=(_|$))", "", doc["run_id"], count=1)
        groups[key].append(doc)
    if not groups:
        raise SystemExit("no run.json under %s" % args.root)

    cols = ("test_roc_auc", "test_pr_auc", "ext_roc_auc", "ext_pr_auc")
    print("%-34s %3s %7s %8s  %s" % ("strategy", "n", "pool", "pos %",
                                    "  ".join("%-15s" % c for c in cols)))
    for key, docs in sorted(groups.items()):
        pools = sorted({d["n_pool"] for d in docs})
        rate = st.mean(d["n_pool_positives"] / d["n_pool"] for d in docs)
        vals = []
        for c in cols:
            split, metric = c.split("_", 1)
            vals.append(fmt([d["metrics"].get(split, {}).get(metric) for d in docs]))
        print("%-34s %3d %7s %7.2f%%  %s" % (
            key, len(docs), "/".join(map(str, pools)), 100 * rate,
            "  ".join("%-15s" % v for v in vals)))


if __name__ == "__main__":
    main()
