"""Classification metrics for a rare-positive discovery task.

    roc_auc       base-rate independent
    pr_auc        average precision; the metric that separates models at a
                  ~4% base rate
    pr_auc_lift   pr_auc / base_rate, comparable across topologies
    p@k, recall@k, enrichment@k
    precision / recall / F1 of the positive class at a threshold tuned on
    validation

Everything is reported both pooled and per topology (base rates range from
~2% to ~9% across topologies).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

DEFAULT_KS = (10, 50, 100, 500)


def base_rate(y) -> float:
    y = np.asarray(y).ravel()
    return float(y.mean()) if y.size else float("nan")


def _degenerate(y) -> bool:
    """True when y holds only one class, so ranking metrics are undefined."""
    y = np.asarray(y).ravel()
    return y.size == 0 or len(np.unique(y)) < 2


def precision_at_k(y, score, k: int) -> float:
    """Fraction of the top-k highest-scoring samples that are positive.

    Returns NaN when k exceeds the sample count, rather than silently scoring
    a shorter list and reporting it as p@k.
    """
    y = np.asarray(y).ravel()
    score = np.asarray(score).ravel()
    if k <= 0 or k > y.size:
        return float("nan")
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top].mean())


def recall_at_k(y, score, k: int) -> float:
    """Fraction of all positives captured in the top k."""
    y = np.asarray(y).ravel()
    score = np.asarray(score).ravel()
    n_pos = int(y.sum())
    if k <= 0 or k > y.size or n_pos == 0:
        return float("nan")
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top].sum() / n_pos)


def best_threshold(y, score, objective: str = "f1") -> float:
    """Threshold maximising F1 (or Youden's J) on the data given.

    ALWAYS call this on validation and apply the result to test. The old code
    used a hardcoded CLASS_THRESHOLD = 0.35, which is what made precision and
    recall collapse to ~0.003 on the natural split while ROC-AUC stayed at
    0.86: the ranking was fine, the cut point was in the wrong place.
    """
    y = np.asarray(y).ravel()
    score = np.asarray(score).ravel()
    if _degenerate(y):
        return 0.5

    if objective == "youden":
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(y, score)
        return float(thr[int(np.argmax(tpr - fpr))])

    precision, recall, thr = precision_recall_curve(y, score)
    # precision_recall_curve returns one more point than thresholds.
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    best = int(np.argmax(f1[:-1])) if thr.size else 0
    return float(thr[best]) if thr.size else 0.5


def at_threshold(y, score, threshold: float) -> dict:
    """Positive-class precision / recall / F1 at a given cut point."""
    y = np.asarray(y).ravel().astype(int)
    pred = (np.asarray(score).ravel() >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"threshold": float(threshold), "precision": precision,
            "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "n_predicted_positive": int(pred.sum())}


def classification_metrics(y, score, threshold: float | None = None,
                           ks=DEFAULT_KS) -> dict:
    """The full metric block for one set of scores."""
    y = np.asarray(y).ravel().astype(int)
    score = np.asarray(score).ravel()
    if y.size != score.size:
        raise ValueError(f"y has {y.size} entries, score has {score.size}")

    br = base_rate(y)
    out = {"n": int(y.size), "n_pos": int(y.sum()), "base_rate": br}

    if _degenerate(y):
        # One class only: every ranking metric is undefined. Say so rather than
        # emitting a 0.5 that reads like a real measurement.
        out.update({"roc_auc": float("nan"), "pr_auc": float("nan"),
                    "pr_auc_lift": float("nan")})
    else:
        pr_auc = float(average_precision_score(y, score))
        out.update({
            "roc_auc": float(roc_auc_score(y, score)),
            "pr_auc": pr_auc,
            "pr_auc_lift": pr_auc / br if br > 0 else float("nan"),
        })

    for k in ks:
        p = precision_at_k(y, score, k)
        out[f"p@{k}"] = p
        out[f"recall@{k}"] = recall_at_k(y, score, k)
        out[f"enrichment@{k}"] = (p / br) if (br > 0 and np.isfinite(p)) else float("nan")

    if threshold is not None:
        out.update(at_threshold(y, score, threshold))
    return out


def per_topology_metrics(y, score, topologies, threshold: float | None = None,
                         ks=DEFAULT_KS) -> dict:
    """Metrics computed independently inside each topology.

    ``topologies`` is a sequence of 'NxM' strings, one per sample. Each
    topology's base rate is its own, so pr_auc_lift here is directly
    comparable between geometries.
    """
    y = np.asarray(y).ravel()
    score = np.asarray(score).ravel()
    topo = np.asarray(topologies).ravel()
    if not (y.size == score.size == topo.size):
        raise ValueError(
            f"length mismatch: y={y.size} score={score.size} topo={topo.size}")

    out = {}
    for t in sorted(set(topo.tolist())):
        m = topo == t
        # k values larger than the topology's own sample count are meaningless
        # here; keep only those that fit so p@k is never silently truncated.
        fitting = tuple(k for k in ks if k <= int(m.sum()))
        out[str(t)] = classification_metrics(y[m], score[m], threshold, fitting)
    return out


def mean_per_topology(per_topo: dict, keys=("roc_auc", "pr_auc", "pr_auc_lift")) -> dict:
    """Unweighted mean of a metric across topologies, ignoring NaNs.

    Unweighted on purpose: every topology carries equal weight in the claim
    "the surrogate works across geometries", regardless of how many samples it
    happens to have.
    """
    out = {}
    for key in keys:
        vals = [m[key] for m in per_topo.values()
                if key in m and np.isfinite(m[key])]
        out[f"mean_{key}"] = float(np.mean(vals)) if vals else float("nan")
        out[f"std_{key}"] = float(np.std(vals)) if len(vals) > 1 else float("nan")
        out[f"n_topologies_{key}"] = len(vals)
    return out


def evaluate(y, score, topologies=None, threshold: float | None = None,
             ks=DEFAULT_KS) -> dict:
    """Pooled metrics, per-topology metrics and their mean, in one dict."""
    out = {"pooled": classification_metrics(y, score, threshold, ks)}
    if topologies is not None:
        out["per_topology"] = per_topology_metrics(y, score, topologies,
                                                   threshold, ks)
        out.update(mean_per_topology(out["per_topology"]))
    return out


def aggregate_seeds(runs: list[dict], keys=None) -> dict:
    """mean / std / n over a list of flat metric dicts, one per seed.

    NaNs are excluded per key, and ``n`` records how many seeds actually
    contributed -- so a metric that was undefined in some seeds cannot quietly
    masquerade as a full-sample mean.
    """
    if not runs:
        return {}
    keys = keys or sorted({k for r in runs for k, v in r.items()
                           if isinstance(v, (int, float))})
    out = {}
    for k in keys:
        vals = [r[k] for r in runs
                if k in r and isinstance(r[k], (int, float)) and np.isfinite(r[k])]
        if not vals:
            out[k] = {"mean": float("nan"), "std": float("nan"), "n": 0}
            continue
        out[k] = {
            "mean": float(np.mean(vals)),
            # ddof=1: these are a sample of training seeds, not the population.
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return out


def bootstrap_ci(y, score, metric: str = "pr_auc", n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0, ks=DEFAULT_KS) -> dict:
    """Percentile bootstrap CI for one metric on one set of scores.

    Resampling is stratified by class (positives among positives, negatives
    among negatives), so the base rate is fixed across replicates. Useful on
    the rare topologies, which have only ~15 test positives.

    Returns the point estimate, the interval, and ``n_valid`` -- the number of
    replicates that actually produced a finite value.
    """
    y = np.asarray(y).ravel().astype(int)
    score = np.asarray(score).ravel()
    point = classification_metrics(y, score, ks=ks).get(metric, float("nan"))

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return {"metric": metric, "point": point, "lo": float("nan"),
                "hi": float("nan"), "n_boot": 0, "n_valid": 0,
                "n_pos": int(pos_idx.size),
                "note": "one class only; a CI is undefined"}

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        take = np.concatenate([
            rng.choice(pos_idx, pos_idx.size, replace=True),
            rng.choice(neg_idx, neg_idx.size, replace=True)])
        v = classification_metrics(y[take], score[take], ks=ks).get(metric)
        if v is not None and np.isfinite(v):
            vals.append(v)

    if not vals:
        return {"metric": metric, "point": point, "lo": float("nan"),
                "hi": float("nan"), "n_boot": n_boot, "n_valid": 0,
                "n_pos": int(pos_idx.size)}
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"metric": metric, "point": float(point), "lo": float(lo),
            "hi": float(hi), "n_boot": n_boot, "n_valid": len(vals),
            "n_pos": int(pos_idx.size),
            "width": float(hi - lo), "confidence": 1 - alpha}


def bootstrap_per_topology(y, score, topologies, metric: str = "pr_auc",
                           n_boot: int = 2000, alpha: float = 0.05,
                           seed: int = 0, min_pos: int = 5) -> dict:
    """``bootstrap_ci`` inside each topology.

    Topologies with fewer than ``min_pos`` positives get a CI too, but are
    marked ``thin``: below about five positives the percentile bootstrap is
    resampling the same handful of points and the interval understates the
    real uncertainty. Report those as counts, not as PR-AUC.
    """
    y = np.asarray(y).ravel()
    score = np.asarray(score).ravel()
    topo = np.asarray(topologies).ravel()

    out = {}
    for i, t in enumerate(sorted(set(topo.tolist()))):
        m = topo == t
        ks = tuple(k for k in DEFAULT_KS if k <= int(m.sum()))
        res = bootstrap_ci(y[m], score[m], metric, n_boot, alpha,
                           seed + 1000 * i, ks)
        res["thin"] = res["n_pos"] < min_pos
        out[str(t)] = res
    return out


def format_ci(res: dict, places: int = 3) -> str:
    """'0.223 [0.118, 0.341]' -- the form to put in a table."""
    if not np.isfinite(res.get("point", float("nan"))):
        return "n/a"
    if not np.isfinite(res.get("lo", float("nan"))):
        return f"{res['point']:.{places}f} [CI undefined]"
    return (f"{res['point']:.{places}f} "
            f"[{res['lo']:.{places}f}, {res['hi']:.{places}f}]")


def format_table(per_topo: dict, metrics=("n", "n_pos", "base_rate", "roc_auc",
                                          "pr_auc", "pr_auc_lift")) -> str:
    """Per-topology metrics as a readable fixed-width table."""
    lines = []
    head = f"{'topology':>9} " + "".join(f"{m:>13}" for m in metrics)
    lines.append(head)
    lines.append("-" * len(head))
    for t in sorted(per_topo):
        row = f"{t:>9} "
        for m in metrics:
            v = per_topo[t].get(m, float("nan"))
            if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and m in ("n", "n_pos")):
                row += f"{int(v):>13d}"
            else:
                row += f"{v:>13.4f}"
        lines.append(row)
    return "\n".join(lines)
