"""The parameter fingerprint separates distinct draws and collapses identical
ones."""
import pytest

from coolgraphs.utils.fingerprint import PARAM_KEYS, fingerprint


class _Scalar:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


class _Graph:
    """Minimal stand-in for a PyG Data object."""

    def __init__(self, meta, label=0):
        self.meta = meta
        self.fbos_sign = _Scalar(label)


def _meta(**over):
    base = {
        "temperature": 120.5, "beta0": 0.25,
        "etaL": 0.4, "etaR": 0.6, "lamE": 0.1, "lamM": 0.2,
        "alpha_vec": [0.1, -0.2, 0.35], "W": [[0.5, 0.25, 0.125]],
        "ni": 1, "nj": 3,
    }
    base.update(over)
    return base


def _entry(**over):
    return {"graph": _Graph(_meta(**over)), "ni": 1, "nj": 3}


def test_identical_parameters_collide():
    assert fingerprint(_entry()) == fingerprint(_entry())


def test_label_does_not_affect_fingerprint():
    """The label is an oracle OUTPUT. Two entries differing only in label came
    from the same call and must still be seen as the same sample."""
    a = {"graph": _Graph(_meta(), label=0)}
    b = {"graph": _Graph(_meta(), label=1)}
    assert fingerprint(a) == fingerprint(b)


@pytest.mark.parametrize("key,value", [
    ("temperature", 120.6),
    ("beta0", 0.2500000001),
    ("etaL", 0.41),
    ("etaR", 0.61),
    ("lamE", 0.11),
    ("lamM", 0.21),
    ("alpha_vec", [0.1, -0.2, 0.3500000001]),
    ("W", [[0.5, 0.25, 0.126]]),
    ("ni", 3),
    ("nj", 1),
])
def test_any_parameter_change_separates(key, value):
    """Every sampled parameter must enter the hash, including at the last bit:
    a near-identical draw is still a different oracle call."""
    assert fingerprint(_entry()) != fingerprint(_entry(**{key: value}))


def test_every_param_key_is_exercised_above():
    """Guards against a parameter being added to PARAM_KEYS without a case in
    test_any_parameter_change_separates."""
    covered = {"temperature", "beta0", "etaL", "etaR", "lamE", "lamM",
               "alpha_vec", "W", "ni", "nj"}
    assert set(PARAM_KEYS) == covered, (
        f"PARAM_KEYS changed; add cases for {set(PARAM_KEYS) ^ covered}")


def test_bare_graph_entry_accepted():
    """Datasets store either a dict or a bare Data object."""
    assert fingerprint(_Graph(_meta())) == fingerprint(_entry())


def test_alpha_vec_order_matters():
    a = fingerprint(_entry(alpha_vec=[0.1, -0.2, 0.35]))
    b = fingerprint(_entry(alpha_vec=[-0.2, 0.1, 0.35]))
    assert a != b


def test_nested_sequence_not_flattened():
    """[[1,2],[3]] and [[1],[2,3]] hold the same numbers in the same order but
    are different coupling matrices, so the structure must be in the hash."""
    a = fingerprint(_entry(W=[[1.0, 2.0], [3.0]]))
    b = fingerprint(_entry(W=[[1.0], [2.0, 3.0]]))
    assert a != b


def test_missing_meta_falls_back_without_raising():
    class Bare:
        pass

    g = Bare()
    g.x = [[1.0, 2.0]]
    assert isinstance(fingerprint(g), str)


def test_arrays_and_lists_agree():
    """A tolist()-able array must hash the same as the equivalent list, so a
    dataset written with numpy meta compares against one written with lists."""
    np = pytest.importorskip("numpy")
    as_list = _entry(alpha_vec=[0.1, -0.2, 0.35])
    as_array = _entry(alpha_vec=np.array([0.1, -0.2, 0.35]))
    assert fingerprint(as_list) == fingerprint(as_array)
