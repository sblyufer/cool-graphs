"""Tests for the torch-free training layer: splits and metrics."""
import numpy as np
import pytest

from coolgraphs.training import metrics as M
from coolgraphs.training.splits import (
    pos_weight_for, stratified_indices, stratum_of)


# --------------------------------------------------------------- fixtures --
class _Scalar:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


class _Graph:
    def __init__(self, label):
        self.fbos_sign = _Scalar(label)


def entry(ni, nj, label):
    return {"graph": _Graph(label), "ni": ni, "nj": nj}


def dataset(spec):
    """spec: {(ni, nj): (n_total, n_positive)} -> list of entries."""
    out = []
    for (ni, nj), (n, pos) in spec.items():
        out += [entry(ni, nj, 1) for _ in range(pos)]
        out += [entry(ni, nj, 0) for _ in range(n - pos)]
    return out


# A miniature of the real natural dataset: equal n, very different hit rates.
NATURAL_LIKE = {(1, 4): (5000, 419), (3, 3): (5000, 93), (2, 2): (5000, 183)}


# ----------------------------------------------------------------- splits --
def test_split_is_exhaustive_and_disjoint():
    e = dataset(NATURAL_LIKE)
    tr, va, te = stratified_indices(e)
    allidx = np.concatenate([tr, va, te])
    assert len(allidx) == len(e)
    assert len(set(allidx.tolist())) == len(e)


def test_positive_counts_match_fractions_per_topology():
    """The whole point: each topology's positives are split 70/15/15, not
    drawn from a binomial that swings by +/-25% on the rare grids."""
    e = dataset(NATURAL_LIKE)
    tr, va, te = stratified_indices(e, val_fraction=0.15, test_fraction=0.15)
    for (ni, nj), (_, pos) in NATURAL_LIKE.items():
        counts = []
        for idx in (tr, va, te):
            counts.append(sum(1 for i in idx
                              if stratum_of(e[i]) == (ni, nj, 1)))
        assert sum(counts) == pos
        assert counts[2] == pytest.approx(round(pos * 0.15), abs=1)
        assert counts[1] == pytest.approx(round(pos * 0.15), abs=1)


def test_3x3_test_positives_are_exact_not_random():
    """93 positives, 15% -> 14 every time. Unstratified this has sd ~3.4."""
    e = dataset({(3, 3): (5000, 93)})
    for seed in (0, 1, 42, 999):
        _, _, te = stratified_indices(e, split_seed=seed)
        n_pos = sum(1 for i in te if stratum_of(e[i])[2] == 1)
        assert n_pos == 14


def test_split_is_deterministic():
    e = dataset(NATURAL_LIKE)
    a = stratified_indices(e, split_seed=7)
    b = stratified_indices(e, split_seed=7)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_different_seed_gives_different_split():
    e = dataset(NATURAL_LIKE)
    a = stratified_indices(e, split_seed=1)[2]
    b = stratified_indices(e, split_seed=2)[2]
    assert not np.array_equal(a, b)


def test_adding_a_topology_does_not_reshuffle_existing_ones():
    """Per-stratum seeding means a later topology cannot disturb the splits of
    the ones already generated -- otherwise topping up a dataset would silently
    move samples between train and test."""
    base = dataset({(1, 4): (500, 40), (3, 3): (500, 9)})
    more = base + dataset({(2, 4): (500, 19)})
    _, _, te_a = stratified_indices(base, split_seed=5)
    _, _, te_b = stratified_indices(more, split_seed=5)
    a = {i for i in te_a if stratum_of(base[i])[:2] == (1, 4)}
    b = {i for i in te_b if stratum_of(more[i])[:2] == (1, 4)}
    assert a == b


def test_tiny_stratum_still_leaves_training_data():
    e = dataset({(3, 3): (3, 1)})
    tr, va, te = stratified_indices(e)
    assert len(tr) >= 1
    assert len(tr) + len(va) + len(te) == 3


def test_rejects_impossible_fractions():
    e = dataset({(1, 2): (100, 10)})
    with pytest.raises(ValueError):
        stratified_indices(e, val_fraction=0.6, test_fraction=0.5)


def test_pos_weight():
    e = dataset({(1, 2): (1000, 100)})
    assert pos_weight_for(e) == pytest.approx(9.0)


def test_pos_weight_natural_scale():
    """The real dataset: 1,698 of 40,000 -> ~22.6."""
    e = dataset({(1, 2): (40000, 1698)})
    assert pos_weight_for(e) == pytest.approx(22.56, abs=0.01)


# ---------------------------------------------------------------- metrics --
def _ranked(n=1000, n_pos=50, quality=1.0, seed=0):
    """Scores correlated with labels by `quality` (1 = perfect, 0 = random)."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_pos, replace=False)] = 1
    noise = rng.random(n)
    return y, quality * y + (1 - quality) * noise + 1e-6 * noise


def test_perfect_ranker():
    y, s = _ranked(quality=1.0)
    m = M.classification_metrics(y, s)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["pr_auc_lift"] == pytest.approx(1 / m["base_rate"])


def test_random_ranker_has_lift_near_one():
    """A random ranker's average precision equals the base rate, so lift ~ 1.
    This is what makes lift the honest headline number."""
    lifts = [M.classification_metrics(*_ranked(quality=0.0, seed=s))["pr_auc_lift"]
             for s in range(20)]
    assert np.mean(lifts) == pytest.approx(1.0, abs=0.15)


def test_precision_at_k_perfect_and_random():
    y, s = _ranked(n=1000, n_pos=50, quality=1.0)
    assert M.precision_at_k(y, s, 50) == pytest.approx(1.0)
    assert M.recall_at_k(y, s, 50) == pytest.approx(1.0)
    assert M.precision_at_k(y, s, 100) == pytest.approx(0.5)


def test_precision_at_k_is_nan_when_k_exceeds_n():
    """Must not silently score a shorter list and call it p@500."""
    y, s = _ranked(n=100, n_pos=10)
    assert np.isnan(M.precision_at_k(y, s, 500))
    assert np.isnan(M.classification_metrics(y, s)["p@500"])


def test_enrichment_is_precision_over_base_rate():
    y, s = _ranked(n=1000, n_pos=50, quality=1.0)
    m = M.classification_metrics(y, s)
    assert m["enrichment@50"] == pytest.approx(m["p@50"] / m["base_rate"])


def test_single_class_gives_nan_not_a_fake_half():
    y = np.zeros(100, dtype=int)
    s = np.random.default_rng(0).random(100)
    m = M.classification_metrics(y, s)
    assert np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"])
    assert m["n_pos"] == 0


def test_threshold_tuned_beats_hardcoded_035():
    """The collapse of precision/recall to ~0.003 on the natural split was a
    cut-point artifact, not a ranking failure. A tuned threshold recovers it
    from the very same scores."""
    rng = np.random.default_rng(1)
    n, n_pos = 20000, 850                      # ~4.25%, the natural rate
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_pos, replace=False)] = 1
    # Well-ranked but poorly calibrated: nearly every score sits below 0.35.
    score = np.clip(rng.beta(2, 30, n) + 0.25 * y, 0, 1)

    hardcoded = M.at_threshold(y, score, 0.35)
    tuned = M.at_threshold(y, score, M.best_threshold(y, score))
    assert M.classification_metrics(y, score)["roc_auc"] > 0.8
    assert tuned["f1"] > hardcoded["f1"]
    assert tuned["recall"] > hardcoded["recall"]


def test_per_topology_and_pooled_disagree_when_base_rates_differ():
    """Ranking perfectly inside each topology still scores poorly pooled,
    because the pooled order interleaves topologies. This is why both numbers
    are reported."""
    ys, ss, ts = [], [], []
    for name, (n, n_pos, offset) in {
        "1x4": (2000, 170, 0.0),     # rich
        "3x3": (2000, 37, 0.5),      # rare, scores shifted up
    }.items():
        y = np.zeros(n, dtype=int)
        y[:n_pos] = 1
        rng = np.random.default_rng(abs(hash(name)) % 2**31)
        s = offset + 0.4 * y + 1e-3 * rng.random(n)   # perfect within topology
        ys.append(y); ss.append(s); ts += [name] * n
    y = np.concatenate(ys); s = np.concatenate(ss)

    res = M.evaluate(y, s, ts)
    for t in ("1x4", "3x3"):
        assert res["per_topology"][t]["pr_auc"] == pytest.approx(1.0, abs=1e-6)
    assert res["mean_pr_auc"] == pytest.approx(1.0, abs=1e-6)
    assert res["pooled"]["pr_auc"] < 0.95


def test_per_topology_requires_matching_lengths():
    with pytest.raises(ValueError):
        M.per_topology_metrics([0, 1], [0.1, 0.9], ["1x2"])


def test_aggregate_seeds_reports_n_and_sample_std():
    runs = [{"pr_auc": 0.10}, {"pr_auc": 0.14}, {"pr_auc": 0.12}]
    agg = M.aggregate_seeds(runs)["pr_auc"]
    assert agg["mean"] == pytest.approx(0.12)
    assert agg["n"] == 3
    assert agg["std"] == pytest.approx(np.std([0.10, 0.14, 0.12], ddof=1))


def test_aggregate_seeds_excludes_nan_and_says_so():
    runs = [{"pr_auc": 0.10}, {"pr_auc": float("nan")}, {"pr_auc": 0.14}]
    agg = M.aggregate_seeds(runs)["pr_auc"]
    assert agg["n"] == 2
    assert agg["mean"] == pytest.approx(0.12)


def test_evaluate_includes_threshold_block_when_given():
    y, s = _ranked(quality=0.9)
    res = M.evaluate(y, s, threshold=0.5)
    assert "precision" in res["pooled"] and "f1" in res["pooled"]


# -------------------------------------------------------------- bootstrap --
def test_bootstrap_ci_brackets_the_point_estimate():
    y, s = _ranked(n=2000, n_pos=100, quality=0.3, seed=3)
    r = M.bootstrap_ci(y, s, "pr_auc", n_boot=300, seed=1)
    assert r["lo"] <= r["point"] <= r["hi"]
    assert r["n_valid"] > 0


def test_bootstrap_is_wider_with_fewer_positives():
    """The whole reason it is needed: 15 positives cannot support a bare
    point estimate the way 150 can."""
    y_many, s_many = _ranked(n=4000, n_pos=300, quality=0.3, seed=4)
    y_few, s_few = _ranked(n=200, n_pos=15, quality=0.3, seed=4)
    w_many = M.bootstrap_ci(y_many, s_many, "pr_auc", n_boot=300, seed=2)["width"]
    w_few = M.bootstrap_ci(y_few, s_few, "pr_auc", n_boot=300, seed=2)["width"]
    assert w_few > w_many


def test_bootstrap_keeps_the_base_rate_fixed():
    """Stratified resampling: every replicate has the same positive count, so
    the interval reflects ranking quality, not split luck."""
    rng = np.random.default_rng(0)
    y = np.zeros(500, dtype=int); y[:25] = 1
    s = rng.random(500)
    r = M.bootstrap_ci(y, s, "base_rate", n_boot=200, seed=5)
    assert r["lo"] == pytest.approx(r["hi"], abs=1e-12)
    assert r["point"] == pytest.approx(0.05)


def test_bootstrap_single_class_is_undefined_not_zero():
    y = np.zeros(50, dtype=int)
    s = np.random.default_rng(0).random(50)
    r = M.bootstrap_ci(y, s, "pr_auc", n_boot=50)
    assert np.isnan(r["lo"]) and r["n_valid"] == 0 and "note" in r


def test_bootstrap_is_deterministic_given_a_seed():
    y, s = _ranked(n=800, n_pos=40, quality=0.3, seed=6)
    a = M.bootstrap_ci(y, s, "pr_auc", n_boot=200, seed=11)
    b = M.bootstrap_ci(y, s, "pr_auc", n_boot=200, seed=11)
    assert a["lo"] == b["lo"] and a["hi"] == b["hi"]


def test_bootstrap_per_topology_flags_thin_topologies():
    ys, ss, ts = [], [], []
    for name, (n, n_pos) in {"1x4": (1000, 80), "3x3": (300, 3)}.items():
        y = np.zeros(n, dtype=int); y[:n_pos] = 1
        rng = np.random.default_rng(abs(hash(name)) % 2**31)
        ys.append(y); ss.append(0.3 * y + rng.random(n)); ts += [name] * n
    res = M.bootstrap_per_topology(np.concatenate(ys), np.concatenate(ss), ts,
                                   n_boot=100, min_pos=5)
    assert res["3x3"]["thin"] is True
    assert res["1x4"]["thin"] is False


def test_format_ci_shape():
    out = M.format_ci({"point": 0.2234, "lo": 0.1181, "hi": 0.3412})
    assert out == "0.223 [0.118, 0.341]"
