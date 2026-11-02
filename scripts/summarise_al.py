#!/usr/bin/env python3
"""Summarise active-learning trajectories (the AL columns of the search table).

    python scripts/summarise_al.py                 # all runs under results/al
    python scripts/summarise_al.py --plot          # also draw the discovery curve

Reads <root>/*/cycles.csv (and discovery_by_topology.csv) and writes, under
--out:
    discovery_curve.csv   per (strategy, model, cycle): mean / s.d. of
                          cumulative positives across seeds
    cost_table.csv        per (strategy, model): positives found, oracle calls
                          per positive, final pool rate, held-out metrics
    by_topology.csv       per (strategy, model, topology): hit rate, enrichment
                          over the random arm, budget share
    discovery.png         with --plot

Runs that did not finish all cycles are excluded unless --include-partial
(which truncates each group to its shortest run). Error bars are across
seeds at a fixed cycle. Standard library only, except matplotlib for --plot.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import statistics as st
from collections import Counter, defaultdict

NUM = ("cycle", "n_acquired", "n_positive", "n_oracle_failures",
       "n_dropped_heldout_collision", "acquired_positive_rate",
       "cum_oracle_calls", "cum_positives", "cum_positive_rate",
       "calls_per_positive", "pool_size", "pool_positives",
       "pool_positive_rate", "epochs_run", "val_pr_auc", "val_roc_auc",
       "test_pr_auc", "test_roc_auc", "ext_pr_auc", "ext_roc_auc",
       "train_seconds", "score_seconds", "oracle_seconds", "cycle_seconds")


def parse_run_id(name):
    """al_<strategy>_<model>_s<seed>[_<tag>] -> (strategy, model, seed, tag).

    Strategy names contain underscores (mc_dropout, deep_ensemble) and so do
    model names (gat_classifier), so this splits on the _s<digits> marker rather
    than counting underscores.
    """
    base = name[3:] if name.startswith("al_") else name
    head, seed, tag = base, None, ""
    parts = base.split("_")
    for i, p in enumerate(parts):
        if len(p) > 1 and p[0] == "s" and p[1:].isdigit():
            head = "_".join(parts[:i])
            seed = int(p[1:])
            tag = "_".join(parts[i + 1:])
            break
    for strat in ("deep_ensemble", "mc_dropout", "random", "bald", "entropy"):
        if head.startswith(strat):
            return strat, head[len(strat):].lstrip("_") or "?", seed, tag
    return head, "?", seed, tag


def topo_label(v):
    """'(1, 2)' / '1x2' / '(1,2)' -> '1x2'.

    al_loop.py writes the topology as a Python tuple, so the CSV carries
    "(1, 2)". That is not a label anyone wants in a paper table, and it also
    makes the column sort as text rather than by size.
    """
    s = (str(v) if v is not None else "?").strip()
    digits = [p for p in re.findall(r"\d+", s)]
    return "x".join(digits) if len(digits) == 2 else s


def read_cycles(d):
    p = os.path.join(d, "cycles.csv")
    if not os.path.exists(p):
        return None
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            o = {}
            for k, v in r.items():
                if k in NUM:
                    try:
                        o[k] = float(v)
                    except (TypeError, ValueError):
                        o[k] = None
                else:
                    o[k] = (v or "").strip()
            if o.get("cycle") is not None:
                rows.append(o)
    rows.sort(key=lambda r: r["cycle"])
    return rows or None


def read_topology(d):
    p = os.path.join(d, "discovery_by_topology.csv")
    if not os.path.exists(p):
        return None, None
    with open(p) as f:
        rd = csv.DictReader(f)
        return (rd.fieldnames or []), list(rd)


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    if len(v) == 1:
        return v[0], 0.0
    return st.mean(v), st.stdev(v)


def se(v):
    return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")


def f(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return ("%." + str(nd) + "f") % x


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/al")
    ap.add_argument("--models", nargs="+", default=None,
                    help="restrict to these surrogates, e.g. vae ae")
    ap.add_argument("--exclude-tagged", action="store_true",
                    help="drop runs with a tag such as _rp11 (random injection)")
    ap.add_argument("--include-partial", action="store_true")
    ap.add_argument("--out", default="results/al_summary")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    runs = {}
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        strat, model, seed, tag = parse_run_id(name)
        if args.models and model not in args.models:
            continue
        if args.exclude_tagged and tag:
            continue
        rows = read_cycles(d)
        if rows is None:
            print("[skip] %s: no readable cycles.csv" % name)
            continue
        runs[name] = {"dir": d, "strategy": strat, "model": model,
                      "seed": seed, "tag": tag, "rows": rows,
                      "n_cycles": len(rows)}
    if not runs:
        print("no runs found under %s" % args.root)
        return 1

    # group on (strategy, model, tag): a tagged variant is a different arm
    groups = defaultdict(list)
    for name, r in runs.items():
        groups[(r["strategy"], r["model"], r["tag"])].append(name)

    # ---- completeness -------------------------------------------------
    print("=" * 84)
    print("== RUNS")
    print("=" * 84)
    print("%-42s %-14s %-16s %6s %7s" %
          ("run", "strategy", "model", "cycles", "calls"))
    print("-" * 84)
    dropped = []
    for key in sorted(groups):
        names = sorted(groups[key])
        counts = Counter(runs[n]["n_cycles"] for n in names)
        target = max(counts)                       # the full-length trajectory
        for n in names:
            r = runs[n]
            mark = "" if r["n_cycles"] == target else "  <- partial"
            print("%-42s %-14s %-16s %6d %7s%s" %
                  (n, r["strategy"], r["model"] + ("/" + r["tag"] if r["tag"] else ""),
                   r["n_cycles"], f(r["rows"][-1].get("cum_oracle_calls"), 0), mark))
        if not args.include_partial:
            keep = [n for n in names if runs[n]["n_cycles"] == target]
            drop = [n for n in names if n not in keep]
            if drop:
                dropped += drop
                groups[key] = keep
        else:
            shortest = min(counts)
            for n in names:
                runs[n]["rows"] = runs[n]["rows"][:shortest]
    if dropped:
        print("\n  excluded as unfinished: %s" % ", ".join(dropped))
        print("  (--include-partial truncates groups to the shortest run instead)")

    fails = [(n, r["rows"][-1].get("n_oracle_failures"),
              r["rows"][-1].get("n_dropped_heldout_collision"))
             for n, r in runs.items()]
    bad = [(n, a, b) for n, a, b in fails if (a or 0) or (b or 0)]
    if bad:
        print("\n  oracle failures / held-out collisions in the FINAL cycle:")
        for n, a, b in bad:
            print("    %-40s failures %s  collisions %s" % (n, f(a, 0), f(b, 0)))
        print("  (both are charged as oracle calls, so the cost figures already")
        print("   account for them -- this is a note, not a problem)")

    # ---- discovery curve ----------------------------------------------
    curve = []
    for key in sorted(groups):
        strat, model, tag = key
        names = groups[key]
        if not names:
            continue
        by_cycle = defaultdict(list)
        calls_at = {}
        for n in names:
            for r in runs[n]["rows"]:
                c = int(r["cycle"])
                if r.get("cum_positives") is not None:
                    by_cycle[c].append(r["cum_positives"])
                if r.get("cum_oracle_calls") is not None:
                    calls_at.setdefault(c, r["cum_oracle_calls"])
        for c in sorted(by_cycle):
            m, s = ms(by_cycle[c])
            curve.append({"strategy": strat, "model": model, "tag": tag,
                          "cycle": c, "cum_oracle_calls": calls_at.get(c),
                          "cum_positives_mean": m, "cum_positives_sd": s,
                          "n_seeds": len(by_cycle[c])})

    # ---- cost table ----------------------------------------------------
    print("\n" + "=" * 84)
    print("== COST AT MATCHED ORACLE BUDGET  (final cycle)")
    print("=" * 84)
    print("%-14s %-16s %5s %8s %16s %14s" %
          ("strategy", "model", "seeds", "calls", "positives", "calls/positive"))
    print("-" * 84)
    cost = []
    for key in sorted(groups):
        strat, model, tag = key
        names = groups[key]
        if not names:
            continue
        pos = [runs[n]["rows"][-1]["cum_positives"] for n in names]
        cpp = [runs[n]["rows"][-1]["calls_per_positive"] for n in names]
        calls = runs[names[0]]["rows"][-1]["cum_oracle_calls"]
        pm, ps = ms(pos)
        cm, cs = ms(cpp)
        label = model + ("/" + tag if tag else "")
        # a single seed has no spread to report; "+- 0.00" reads as a measured
        # zero variance rather than as one observation
        if len(names) < 2:
            pstr, cstr = "%s (n=1)" % f(pm, 0), "%s (n=1)" % f(cm, 2)
        else:
            pstr = "%s +- %s" % (f(pm, 0), f(ps, 0))
            cstr = "%s +- %s" % (f(cm, 2), f(cs, 2))
        print("%-14s %-16s %5d %8s %16s %14s" %
              (strat, label, len(names), f(calls, 0), pstr, cstr))
        cost.append({"strategy": strat, "model": model, "tag": tag,
                     "n_seeds": len(names), "cum_oracle_calls": calls,
                     "positives_mean": pm, "positives_sd": ps,
                     "calls_per_positive_mean": cm, "calls_per_positive_sd": cs})

    # ---- enrichment against the matched random control ----------------
    print("\n" + "=" * 84)
    print("== ENRICHMENT vs THE RANDOM ARM OF THE SAME SURROGATE")
    print("=" * 84)
    print("Each row is compared at the LARGEST CYCLE BOTH ARMS REACHED, not at")
    print("each arm's own final cycle. A trajectory still in flight has spent")
    print("fewer oracle calls, so comparing its total against a finished arm's")
    print("would read as a failure of the method rather than of the budget.")
    print("The call count the comparison was made at is printed, and a row below")
    print("the full budget is marked.")
    print()
    print("No standard errors here: these ratios are 5-7x with 1-3 seeds, so an")
    print("se ratio would be a large number carrying no information. The seed")
    print("range is the honest spread.")
    print()
    # index the curve so positives can be read at any shared cycle
    at = defaultdict(dict)          # (strategy, model, tag) -> cycle -> row
    for r in curve:
        at[(r["strategy"], r["model"], r["tag"])][r["cycle"]] = r
    full = max((r["cycle"] for r in curve), default=0)

    print("%-14s %-16s %7s %12s %12s %8s %14s" %
          ("strategy", "model", "calls", "positives", "random", "enrich", "seed range"))
    print("-" * 84)
    for key in sorted(at):
        strat, model, tag = key
        if strat == "random" and not tag:
            continue
        rk = ("random", model, "")
        if rk not in at:
            continue
        shared = sorted(set(at[key]) & set(at[rk]))
        if not shared:
            continue
        c = shared[-1]
        a, b = at[key][c], at[rk][c]
        if not b["cum_positives_mean"]:
            continue
        ratio = a["cum_positives_mean"] / b["cum_positives_mean"]
        # seed range of the ratio's numerator, as a spread the reader can check
        pos = [runs[n]["rows"][c]["cum_positives"] for n in groups[key]
               if c < len(runs[n]["rows"])]
        rng = ("%d-%d" % (min(pos), max(pos))) if len(pos) > 1 else "n=1"
        mark = "" if c == full else "   <- partial budget"
        print("%-14s %-16s %7s %12s %12s %8s %14s%s" %
              (strat, model + ("/" + tag if tag else ""),
               f(a["cum_oracle_calls"], 0),
               f(a["cum_positives_mean"], 0), f(b["cum_positives_mean"], 0),
               "%.2fx" % ratio, rng, mark))
    print()
    print("A 'partial budget' row is a live comparison at a shared, smaller")
    print("budget, not the full-budget result.")

    # ---- held-out quality at matched pool size ------------------------
    print("\n" + "=" * 84)
    print("== HELD-OUT PERFORMANCE OF THE LAST CYCLE'S MODEL")
    print("=" * 84)
    print("This model was trained before the last acquisition (pool minus one")
    print("batch). For the table, retrain on the final pool with")
    print("scripts/retrain_on_pool.py and summarise with summarise_retrained.py.")
    print()
    print("%-14s %-16s %16s %16s %16s" %
          ("strategy", "model", "test PR", "test ROC", "ext ROC"))
    print("-" * 84)
    for key in sorted(groups):
        strat, model, tag = key
        names = groups[key]
        if not names:
            continue
        g = lambda k: [runs[n]["rows"][-1].get(k) for n in names
                       if runs[n]["rows"][-1].get(k) is not None]
        out = []
        for k in ("test_pr_auc", "test_roc_auc", "ext_roc_auc"):
            m, s = ms(g(k))
            out.append("%s +- %s" % (f(m), f(s)))
        print("%-14s %-16s %16s %16s %16s" %
              (strat, model + ("/" + tag if tag else ""), *out))

    # ---- per topology --------------------------------------------------
    topo_rows, topo_header = [], None
    for key in sorted(groups):
        for n in groups[key]:
            hdr, rows = read_topology(runs[n]["dir"])
            if not rows:
                continue
            topo_header = topo_header or hdr
            for r in rows:
                r = dict(r)
                r.update(run=n, strategy=key[0], model=key[1], tag=key[2])
                topo_rows.append(r)
    topo_agg = []
    if topo_rows:
        print("\n" + "=" * 84)
        print("== PER-TOPOLOGY HIT RATE AND ENRICHMENT")
        print("=" * 84)
        print("Pooled over seeds and cycles: positives acquired in that topology")
        print("divided by points acquired in it. Uncertainty cycles only for the")
        print("AL arms, so an injected random cycle cannot drag a rate down.")
        print("Budget share is what fraction of all acquisitions went to that")
        print("topology, and evenness is the Shannon entropy of those shares")
        print("normalised to [0,1]: 1.0 is a perfectly even split across")
        print("topologies, lower means the method concentrated its budget.")
        print()
        # (strategy, model, tag) -> topology -> [acquired, positive]
        acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for r in topo_rows:
            key = (r["strategy"], r["model"], r["tag"])
            strat_cycle = (r.get("strategy_this_cycle") or "").strip()
            if key[0] != "random" and strat_cycle == "random":
                continue      # injected cycle: not this arm's acquisition rule
            try:
                na = int(float(r["n_acquired"]))
                np_ = int(float(r["n_positive"]))
            except (KeyError, TypeError, ValueError):
                continue
            cell = acc[key][topo_label(r.get("topology"))]
            cell[0] += na
            cell[1] += np_

        def evenness(shares):
            shares = [s for s in shares if s > 0]
            if len(shares) < 2:
                return None
            h = -sum(s * math.log(s) for s in shares)
            return h / math.log(len(shares))

        def topo_sort(t):
            d = re.findall(r"\d+", t)
            return (int(d[0]) * int(d[1]), t) if len(d) == 2 else (10 ** 6, t)

        topos = sorted({t for d in acc.values() for t in d}, key=topo_sort)
        for key in sorted(acc):
            strat, model, tag = key
            if strat == "random" and not tag:
                continue
            rk = ("random", model, "")
            tot = sum(v[0] for v in acc[key].values())
            print("-- %s / %s%s   (random control: %s)" %
                  (strat, model, ("/" + tag) if tag else "",
                   "present" if rk in acc else "MISSING"))
            print("   %-8s %10s %10s %10s %10s" %
                  ("topology", "AL rate", "random", "enrich", "budget %"))
            shares = []
            for t in topos:
                a = acc[key].get(t)
                if not a or not a[0]:
                    continue
                ra = acc.get(rk, {}).get(t)
                al_rate = a[1] / a[0]
                rd_rate = (ra[1] / ra[0]) if ra and ra[0] else None
                share = a[0] / tot if tot else 0.0
                shares.append(share)
                enr = (al_rate / rd_rate) if rd_rate else None
                print("   %-8s %10s %10s %10s %10s" %
                      (t, f(al_rate, 4), f(rd_rate, 4),
                       ("%.2fx" % enr) if enr else "--", f(100 * share, 1)))
                topo_agg.append({
                    "strategy": strat, "model": model, "tag": tag,
                    "topology": t, "n_acquired": a[0], "n_positive": a[1],
                    "al_rate": al_rate, "random_rate": rd_rate,
                    "enrichment": enr, "budget_share": share})
            ev = evenness(shares)
            rev = evenness([v[0] / sum(w[0] for w in acc[rk].values())
                            for v in acc[rk].values()]) if rk in acc else None
            print("   evenness %s   (random %s)" % (f(ev, 3), f(rev, 3)))
            below = [t for t in topos
                     if acc[key].get(t) and acc.get(rk, {}).get(t)
                     and acc[key][t][0] and acc[rk][t][0]
                     and acc[key][t][1] / acc[key][t][0]
                     <= acc[rk][t][1] / acc[rk][t][0]]
            print("   topologies at or below the random control: %s" %
                  (", ".join(below) if below else "none"))
            print()
    else:
        print("\n  no discovery_by_topology.csv found in any run.")

    # ---- write ---------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)

    def dump(path, rows, cols=None):
        if not rows:
            return
        cols = cols or list(rows[0].keys())
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print("wrote %s" % path)

    print()
    dump(os.path.join(args.out, "discovery_curve.csv"), curve)
    dump(os.path.join(args.out, "cost_table.csv"), cost)
    if topo_rows:
        cols = ["run", "strategy", "model", "tag"] + [
            c for c in topo_header if c not in ("run", "strategy", "model", "tag")]
        dump(os.path.join(args.out, "by_topology.csv"), topo_rows, cols)
    dump(os.path.join(args.out, "by_topology_enrichment.csv"), topo_agg)

    # ---- figure --------------------------------------------------------
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib is not installed; plot discovery_curve.csv "
                  "elsewhere.")
            return 0

        series = defaultdict(list)
        for r in curve:
            series[(r["strategy"], r["model"], r["tag"])].append(r)
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        for key in sorted(series):
            rows = sorted(series[key], key=lambda r: r["cycle"])
            x = [r["cum_oracle_calls"] for r in rows]
            y = [r["cum_positives_mean"] for r in rows]
            sd = [r["cum_positives_sd"] or 0.0 for r in rows]
            lab = "%s / %s%s" % (key[0], key[1], ("/" + key[2]) if key[2] else "")
            line, = ax.plot(x, y, label=lab, linewidth=1.8)
            ax.fill_between(x, [a - b for a, b in zip(y, sd)],
                            [a + b for a, b in zip(y, sd)],
                            alpha=0.18, color=line.get_color(), linewidth=0)
        ax.set_xlabel("Cumulative oracle calls")
        ax.set_ylabel("Cumulative positives discovered")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = os.path.join(args.out, "discovery.png")
        fig.savefig(p, dpi=200)
        print("wrote %s" % p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
