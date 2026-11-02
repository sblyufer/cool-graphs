#!/usr/bin/env python3
"""Summarise GP-BO runs and compare them with the AL loop at matched budget.

    python scripts/summarise_bo.py          # after scripts/summarise_al.py

Reads results/bo/*.csv (from scripts/bo_baseline.py) and, if present,
results/al_summary/ (from scripts/summarise_al.py). Writes
results/bo_summary.csv and results/bo_summary_by_topology.csv.

Compared: total positives at the matched call budget, and the per-topology
hit rate. The pooled cumulative curve is not compared, because GP-BO runs the
topologies one after another while the AL loop interleaves them. Note that
GP-BO splits its budget evenly across topologies (a GP needs a fixed input
dimension) while the AL loop reallocates. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import statistics as st
from collections import defaultdict


def f(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return ("%." + str(nd) + "f") % x


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    if len(v) == 1:
        return v[0], None
    return st.mean(v), st.stdev(v)


def topo_label(v):
    d = re.findall(r"\d+", str(v) if v is not None else "")
    return "x".join(d) if len(d) == 2 else str(v)


def topo_sort(t):
    d = re.findall(r"\d+", t)
    return (int(d[0]) * int(d[1]), t) if len(d) == 2 else (10 ** 6, t)


def read_bo(path):
    """-> (meta, rows). rows are dicts with the numeric fields coerced."""
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({
                    "calls": int(float(r["n_oracle_calls"])),
                    "cum_pos": int(float(r["cum_positives"])),
                    "is_pos": int(float(r["is_positive"])),
                    "topology": topo_label(r.get("topology")),
                    "acq": (r.get("acq") or "").strip(),
                    "seed": int(float(r["seed"])) if r.get("seed") else None,
                })
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        return None, None
    return {"acq": rows[-1]["acq"], "seed": rows[-1]["seed"],
            "calls": rows[-1]["calls"], "cum_pos": rows[-1]["cum_pos"],
            "n_rows": len(rows)}, rows


def read_al_cost(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = (r.get("strategy"), r.get("model"), r.get("tag") or "")
            try:
                out[key] = {
                    "positives": float(r["positives_mean"]),
                    "positives_sd": float(r["positives_sd"] or 0) or None,
                    "cpp": float(r["calls_per_positive_mean"]),
                    "calls": float(r["cum_oracle_calls"]),
                    "n_seeds": int(float(r["n_seeds"])),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return out


def read_al_topo(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = (r.get("strategy"), r.get("model"), r.get("tag") or "",
                   topo_label(r.get("topology")))
            try:
                out[key] = {"al_rate": float(r["al_rate"]),
                            "random_rate": (float(r["random_rate"])
                                            if r.get("random_rate") else None)}
            except (KeyError, TypeError, ValueError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bo-dir", default="results/bo")
    ap.add_argument("--al", default="results/al_summary",
                    help="directory holding cost_table.csv and "
                         "by_topology_enrichment.csv from summarise_al.py")
    ap.add_argument("--expect-budget", type=int, default=16000,
                    help="the matched budget; a run short of this is flagged")
    ap.add_argument("--out", default="results/bo_summary.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.bo_dir, "*.csv")))
    paths = [p for p in paths if "summary" not in os.path.basename(p)]
    if not paths:
        print("no CSVs in %s" % args.bo_dir)
        return 1

    runs = {}
    for p in paths:
        meta, rows = read_bo(p)
        if meta is None:
            print("[skip] %s: no readable rows" % os.path.basename(p))
            continue
        runs[p] = (meta, rows)

    # ---- completeness --------------------------------------------------
    print("=" * 82)
    print("== BO RUNS")
    print("=" * 82)
    print("%-38s %5s %6s %9s %11s" % ("file", "acq", "seed", "calls", "positives"))
    print("-" * 82)
    short = []
    for p, (m, _) in sorted(runs.items()):
        mark = ""
        if m["calls"] < args.expect_budget:
            mark = "  <- SHORT"
            short.append(os.path.basename(p))
        print("%-38s %5s %6s %9d %11d%s" %
              (os.path.basename(p), m["acq"], m["seed"], m["calls"],
               m["cum_pos"], mark))
    if short:
        print()
        print("  %d run(s) did not reach %d calls. A partial budget is not a"
              % (len(short), args.expect_budget))
        print("  matched comparison.")

    # ---- aggregate per acquisition -------------------------------------
    by_acq = defaultdict(list)
    for p, (m, _) in runs.items():
        if m["calls"] >= args.expect_budget:
            by_acq[m["acq"]].append(m)

    print("\n" + "=" * 82)
    print("== TOTALS AT MATCHED BUDGET (%d calls)" % args.expect_budget)
    print("=" * 82)
    al = read_al_cost(os.path.join(args.al, "cost_table.csv"))
    print("%-22s %6s %16s %16s" %
          ("method", "seeds", "positives", "calls/positive"))
    print("-" * 82)

    rows_out = []
    for acq in sorted(by_acq):
        pos = [m["cum_pos"] for m in by_acq[acq]]
        pm, psd = ms(pos)
        cpp = [args.expect_budget / x for x in pos if x]
        cm, csd = ms(cpp)
        ptxt = ("%s +- %s" % (f(pm, 0), f(psd, 0)) if psd is not None
                else "%s (n=1)" % f(pm, 0))
        ctxt = ("%s +- %s" % (f(cm, 2), f(csd, 2)) if csd is not None
                else "%s (n=1)" % f(cm, 2))
        print("%-22s %6d %16s %16s" % ("GP-BO " + acq, len(pos), ptxt, ctxt))
        rows_out.append({"method": "bo_" + acq.lower(), "n_seeds": len(pos),
                         "calls": args.expect_budget,
                         "positives_mean": pm, "positives_sd": psd,
                         "calls_per_positive_mean": cm,
                         "calls_per_positive_sd": csd})

    for key, lab in ((("mc_dropout", "vae", ""), "AL MC-dropout GVAE"),
                     (("deep_ensemble", "vae", ""), "AL deep ensemble GVAE"),
                     (("random", "vae", ""), "Random (matched)")):
        a = al.get(key)
        if not a:
            continue
        if a["calls"] < args.expect_budget:
            print("%-22s %6d %16s %16s   <- only %d calls, not comparable" %
                  (lab, a["n_seeds"], f(a["positives"], 0), f(a["cpp"], 2),
                   int(a["calls"])))
            continue
        ptxt = ("%s +- %s" % (f(a["positives"], 0), f(a["positives_sd"], 0))
                if a["positives_sd"] else "%s (n=1)" % f(a["positives"], 0))
        print("%-22s %6d %16s %16s" %
              (lab, a["n_seeds"], ptxt, f(a["cpp"], 2)))
        rows_out.append({"method": lab, "n_seeds": a["n_seeds"],
                         "calls": a["calls"],
                         "positives_mean": a["positives"],
                         "positives_sd": a["positives_sd"],
                         "calls_per_positive_mean": a["cpp"],
                         "calls_per_positive_sd": None})

    rnd = al.get(("random", "vae", ""))
    if rnd and by_acq:
        print()
        for acq in sorted(by_acq):
            pm, _ = ms([m["cum_pos"] for m in by_acq[acq]])
            if pm and rnd["positives"]:
                print("  GP-BO %s enrichment over matched random: %.2fx"
                      % (acq, pm / rnd["positives"]))
        a = al.get(("mc_dropout", "vae", ""))
        if a and a["positives"]:
            for acq in sorted(by_acq):
                pm, _ = ms([m["cum_pos"] for m in by_acq[acq]])
                print("  AL / GP-BO %s: %.2fx" % (acq, a["positives"] / pm))

    print()
    print("  GP-BO splits its budget evenly across topologies; the AL loop")
    print("  reallocates. Compare per-topology hit rates below as well.")

    # ---- per topology ---------------------------------------------------
    print("\n" + "=" * 82)
    print("== PER TOPOLOGY, HIT RATE (positives / calls within the topology)")
    print("=" * 82)
    al_topo = read_al_topo(os.path.join(args.al, "by_topology_enrichment.csv"))

    # (acq, topology) -> [calls, positives] pooled over seeds
    acc = defaultdict(lambda: [0, 0])
    for p, (m, rows) in runs.items():
        if m["calls"] < args.expect_budget:
            continue
        for r in rows:
            cell = acc[(m["acq"], r["topology"])]
            cell[0] += 1
            cell[1] += r["is_pos"]

    topos = sorted({t for (_, t) in acc}, key=topo_sort)
    topo_rows = []
    for acq in sorted(by_acq):
        print("-- GP-BO %s" % acq)
        print("   %-8s %9s %9s %10s %10s %9s" %
              ("topology", "calls", "BO rate", "AL rate", "random", "BO/random"))
        for t in topos:
            c, pos = acc.get((acq, t), [0, 0])
            if not c:
                continue
            bo_rate = pos / c
            a = al_topo.get(("mc_dropout", "vae", "", t)) or {}
            al_rate = a.get("al_rate")
            rd = a.get("random_rate")
            enr = (bo_rate / rd) if rd else None
            print("   %-8s %9d %9s %10s %10s %9s" %
                  (t, c, f(bo_rate), f(al_rate), f(rd),
                   ("%.2fx" % enr) if enr else "--"))
            topo_rows.append({"method": "bo_" + acq.lower(), "topology": t,
                              "calls": c, "positives": pos,
                              "bo_rate": bo_rate, "al_rate": al_rate,
                              "random_rate": rd, "bo_over_random": enr})
        below = [t for t in topos
                 if acc.get((acq, t), [0, 0])[0]
                 and (al_topo.get(("mc_dropout", "vae", "", t)) or {}).get("random_rate")
                 and acc[(acq, t)][1] / acc[(acq, t)][0]
                 <= al_topo[("mc_dropout", "vae", "", t)]["random_rate"]]
        print("   at or below the random control: %s"
              % (", ".join(below) if below else "none"))
        print()

    # ---- write -----------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if rows_out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print("wrote %s" % args.out)
    if topo_rows:
        p2 = os.path.splitext(args.out)[0] + "_by_topology.csv"
        with open(p2, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(topo_rows[0].keys()))
            w.writeheader()
            w.writerows(topo_rows)
        print("wrote %s" % p2)

    # ---- per-topology curves, the only comparable curve ------------------
    p3 = os.path.splitext(args.out)[0] + "_curves_by_topology.csv"
    with open(p3, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "seed", "topology", "calls_in_topology",
                    "cum_positives_in_topology"])
        for p, (m, rows) in sorted(runs.items()):
            if m["calls"] < args.expect_budget:
                continue
            seen = defaultdict(lambda: [0, 0])
            for r in rows:
                cell = seen[r["topology"]]
                cell[0] += 1
                cell[1] += r["is_pos"]
                w.writerow(["bo_" + m["acq"].lower(), m["seed"],
                            r["topology"], cell[0], cell[1]])
    print("wrote %s" % p3)
    print()
    print("NOTE: no pooled discovery curve is written. bo_baseline.py runs")
    print("topologies sequentially, so its pooled cumulative curve is a")
    print("staircase shaped by the order of SIZE_LIST rather than by the")
    print("search, and the AL loop interleaves all eight every cycle. Only the")
    print("per-topology curves above share an x-axis with the AL curves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
