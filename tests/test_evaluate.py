"""Tests for the score / verify path of training/evaluate.py (prediction itself
needs a trained model and is exercised by the smoke run instead)."""
import numpy as np
import pytest

from _fake_graphs import entry
from coolgraphs.training import evaluate as E
from coolgraphs.training import metrics as M

TOPOS = [(1, 2), (1, 3), (3, 3)]


def _split(n=300, pos_rate=0.1, quality=0.85, seed=0):
    """A split whose scores rank positives above negatives `quality` of the time."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    score = rng.random(n) * (1 - quality) + y * quality
    entries = [entry(ni=TOPOS[i % 3][0], nj=TOPOS[i % 3][1],
                     beta0=float(i) + 0.5 + seed * 1e6, label=int(y[i]))
               for i in range(n)]
    return {"y": y, "score": score, "z": rng.random((n, 8)),
            "entries": entries, "idx": np.arange(n)}


def test_threshold_comes_from_validation_only():
    """The test split must not choose its own cut point -- that is the
    hardcoded-0.35 bug in the other direction, and it inflates test scores."""
    out = {"val": _split(seed=1), "test": _split(seed=2)}
    _, thr = E.score_splits(out, verbose=False)
    assert thr == pytest.approx(M.best_threshold(out["val"]["y"],
                                                 out["val"]["score"]))


def test_no_validation_split_means_no_threshold():
    out = {"test": _split(seed=3)}
    report, thr = E.score_splits(out, verbose=False)
    assert thr is None
    assert "test" in report


def test_report_has_the_expected_keys():
    out = {"val": _split(seed=4), "test": _split(seed=5)}
    report, _ = E.score_splits(out, bootstrap_on=("test",), n_boot=200,
                               verbose=False)
    for split in ("val", "test"):
        pooled = report[split]["pooled"]
        for k in ("roc_auc", "pr_auc", "pr_auc_lift", "base_rate"):
            assert k in pooled, f"{split}.pooled is missing {k}"
        assert "per_topology" in report[split]
        assert report[split]["per_topology"].keys() >= {"1x2", "1x3", "3x3"}
        assert "mean_pr_auc" in report[split]
    # only the splits asked for carry bootstrap intervals -- they are the
    # expensive part and train does not need them
    assert "bootstrap_pr_auc" in report["test"]
    assert "bootstrap_pr_auc" not in report["val"]


def test_bootstrap_entries_carry_a_positive_count_and_thin_flag():
    out = {"val": _split(seed=6), "test": _split(n=200, pos_rate=0.04, seed=7)}
    report, _ = E.score_splits(out, bootstrap_on=("test",), n_boot=200,
                               verbose=False)
    for topo, ci in report["test"]["bootstrap_pr_auc"].items():
        assert "n_pos" in ci and "thin" in ci
        assert M.format_ci(ci)


# --- the recomputation check ---------------------------------------------

def _report_with_val_pr(value):
    return {"val": {"pooled": {"pr_auc": value}}}


def test_agreement_is_reported_as_agreement():
    block = E.verify_logged(_report_with_val_pr(0.4578), 0.45772,
                            verbose=False)
    assert block["checked"] and block["agree"]
    assert block["abs_delta"] < E.RECOMPUTE_TOLERANCE


def test_a_real_disagreement_is_caught():
    """The failure this exists for: the saved arrays are not what the model
    was selected on. 0.46 logged against 0.31 recomputed is a wrong
    checkpoint or a reordered loader, and the figures would look fine."""
    block = E.verify_logged(_report_with_val_pr(0.31), 0.4578, verbose=False)
    assert block["checked"] and not block["agree"]
    assert block["abs_delta"] == pytest.approx(0.1478, abs=1e-4)


def test_tolerance_boundary_is_not_accidentally_inverted():
    just_in = E.verify_logged(_report_with_val_pr(0.5), 0.5 + 0.9 * E.RECOMPUTE_TOLERANCE,
                              verbose=False)
    just_out = E.verify_logged(_report_with_val_pr(0.5), 0.5 + 1.1 * E.RECOMPUTE_TOLERANCE,
                               verbose=False)
    assert just_in["agree"] and not just_out["agree"]


def test_nothing_logged_means_not_checked_rather_than_passed():
    """Absence of evidence must not read as agreement in the run record."""
    block = E.verify_logged(_report_with_val_pr(0.4), None, verbose=False)
    assert block["checked"] is False
    assert block.get("agree") is None


def test_no_validation_split_means_not_checked():
    block = E.verify_logged({"test": {"pooled": {"pr_auc": 0.4}}}, 0.4,
                            verbose=False)
    assert block["checked"] is False


def test_the_check_runs_end_to_end_on_real_scores():
    """Recomputing from the arrays must land on sklearn's own number."""
    out = {"val": _split(seed=8)}
    report, _ = E.score_splits(out, verbose=False)
    truth = float(M.classification_metrics(out["val"]["y"],
                                           out["val"]["score"])["pr_auc"])
    assert E.verify_logged(report, truth, verbose=False)["agree"]


# --- architecture read off the built model --------------------------------

class _Fake:
    """Duck-typed stand-in: describe_architecture must not need torch."""
    def __init__(self, name, **kw):
        self.__class__ = type(name, (_Fake,), {})
        self.__dict__.update(kw)


def _module(name, **kw):
    obj = _Fake.__new__(type(name, (object,), {}))
    obj.__dict__.update(kw)
    return obj


class _Model:
    def __init__(self, mods, params=(), inner=None):
        self._mods = mods
        self._params = params
        self.model = inner

    def modules(self):
        return self._mods

    def parameters(self):
        return self._params


class _P:
    def __init__(self, n, grad=True):
        self._n, self.requires_grad = n, grad

    def numel(self):
        return self._n


def test_reads_dropout_and_gat_heads_off_the_model():
    mods = [_module("Dropout", p=0.2), _module("Dropout", p=0.2),
            _module("GATConv", heads=2), _module("GATConv", heads=2),
            _module("Linear")]
    inner = _module("GraphVAE", pool_type="attention", encoder_type="GAT",
                    variational=True)
    arch = E.describe_architecture(_Model(mods, [_P(100), _P(20)], inner))
    assert arch["dropout_p"] == [0.2]
    assert arch["gat_heads"] == [2] and arch["gat_layers"] == 2
    assert arch["n_parameters"] == 120
    assert arch["pool_type"] == "attention"
    assert arch["module_counts"]["GATConv"] == 2


def test_disagreeing_dropout_values_are_all_surfaced():
    """If two layers disagree, the table cannot claim a single number."""
    mods = [_module("Dropout", p=0.2), _module("Dropout", p=0.5)]
    arch = E.describe_architecture(_Model(mods))
    assert arch["dropout_p"] == [0.2, 0.5]


def test_missing_pieces_are_reported_not_crashed():
    arch = E.describe_architecture(_Model([_module("Linear")]))
    assert arch["dropout_p"] == "none found"
    assert arch["gat_heads"] == "none found"
    assert arch["gat_layers"] == 0


def test_an_unwalkable_model_does_not_kill_a_finished_run():
    class Broken:
        def modules(self):
            raise RuntimeError("no")
    arch = E.describe_architecture(Broken())
    assert "error" in arch
