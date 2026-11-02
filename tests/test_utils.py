"""Tests for the torch-free utility layer.

These run anywhere -- no GPU, no datasets. They use a stub graph
with the same duck-typing surface as a PyG Data object (`fbos_sign.item()`,
`heat_flux`, `meta`), which is all the utils touch.

    pytest -q
"""
import numpy as np
import pytest

from coolgraphs.utils.io import (graph_of, graphs_of, label_of, load_entries,
                                 save_entries)
from coolgraphs.utils.meta import META_COLS, inverse_signed_log1p, meta_matrix
from coolgraphs.utils.topology import (group_by_topology, hit_rates,
                                       parse_topologies, topo_name, topology_of)


class _Scalar:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


class StubGraph:
    """Minimal stand-in for a PyG Data object as the utils see it."""

    def __init__(self, label, ni, nj, **meta):
        self.fbos_sign = _Scalar(float(label))
        self.heat_flux = np.array([meta.pop("heat_flux", 0.0)])
        self.meta = {"ni": ni, "nj": nj, "temperature": 300.0, "beta0": -0.05,
                     "etaR": 0.1, "etaL": 0.1, "lamE": 500.0, "lamM": 500.0,
                     "alpha_vec": np.array([0.1, 0.4, 0.9]), **meta}


def entry(label, ni, nj, **kw):
    return {"graph": StubGraph(label, ni, nj, **kw), "ni": ni, "nj": nj}


@pytest.fixture
def entries():
    return ([entry(1, 1, 2) for _ in range(3)]
            + [entry(0, 1, 2) for _ in range(7)]
            + [entry(1, 3, 3) for _ in range(1)]
            + [entry(0, 3, 3) for _ in range(9)])


def test_accessors(entries):
    assert isinstance(graph_of(entries[0]), StubGraph)
    assert len(graphs_of(entries)) == len(entries)
    assert label_of(entries[0]) == 1
    # bare graphs, not wrapped in a dict, must work too
    assert label_of(StubGraph(0, 2, 2)) == 0


def test_topology(entries):
    assert topology_of(entries[0]) == (1, 2)
    assert topology_of(StubGraph(1, 2, 3)) == (2, 3)   # falls back to graph.meta
    assert topo_name((3, 2)) == "3x2"
    assert parse_topologies(["1x2", "3X3"]) == {(1, 2), (3, 3)}


def test_hit_rates(entries):
    rates = hit_rates(entries)
    assert rates[(1, 2)] == {"n": 10, "pos": 3, "rate": 0.3}
    assert rates[(3, 3)]["rate"] == pytest.approx(0.1)
    groups = group_by_topology(entries)
    assert len(groups[(1, 2)]["pos"]) == 3
    assert len(groups[(1, 2)]["neg"]) == 7


def test_meta_matrix_shape_and_alignment(entries):
    m = meta_matrix(entries)
    assert m.shape == (len(entries), len(META_COLS))
    # the label column must match the entries, row for row
    assert np.array_equal(m[:, META_COLS.index("label")],
                          np.array([label_of(e) for e in entries], dtype=float))
    # ni/nj identify the topology of each row
    ni, nj = META_COLS.index("ni"), META_COLS.index("nj")
    assert (int(m[0, ni]), int(m[0, nj])) == (1, 2)


def test_meta_alpha_statistics():
    m = meta_matrix([entry(1, 1, 3, alpha_vec=np.array([0.0, 1.0, 2.0]))])
    assert m[0, META_COLS.index("alpha_mean")] == pytest.approx(1.0)
    assert m[0, META_COLS.index("alpha_range")] == pytest.approx(2.0)
    assert m[0, META_COLS.index("n_sites")] == pytest.approx(3.0)


def test_meta_missing_alpha_is_nan():
    m = meta_matrix([entry(0, 1, 2, alpha_vec=np.array([]))])
    assert np.isnan(m[0, META_COLS.index("alpha_mean")])


def test_signed_log1p_roundtrip():
    raw = np.array([-1e8, -1e5, 0.0, 1e5, 1e8])
    stored = np.sign(raw) * np.log1p(np.abs(raw))
    assert np.allclose(inverse_signed_log1p(stored), raw, rtol=1e-9)


def test_save_load_roundtrip(tmp_path, entries):
    p = tmp_path / "d.pkl"
    save_entries(entries, str(p))
    back = load_entries(str(p))
    assert len(back) == len(entries)
    assert [label_of(e) for e in back] == [label_of(e) for e in entries]
    assert not (tmp_path / "d.pkl.tmp").exists()   # atomic write cleans up


def test_load_missing_file_is_readable(tmp_path):
    with pytest.raises(SystemExit) as e:
        load_entries(str(tmp_path / "nope.pkl"))
    assert "Dataset not found" in str(e.value)


def test_meta_cols_are_append_only():
    """META_COLS is an on-disk format. Reordering breaks saved .npz files."""
    assert META_COLS[:8] == ["T", "beta0", "etaR", "etaL",
                             "lamE", "lamM", "ni", "nj"]
    assert META_COLS[-2:] == ["heat_flux", "label"]
    assert len(META_COLS) == len(set(META_COLS))
