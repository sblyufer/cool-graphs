"""Torch-free stand-ins for PyG entries, importable so they can be pickled.

The duplicate machinery only ever touches ``graph.meta`` and
``graph.fbos_sign``, so a dataset of these behaves like a real one for every
test in this directory -- and unlike a real one, it costs nothing to build.
"""
from __future__ import annotations


class Scalar:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


class Graph:
    def __init__(self, meta, label=0):
        self.meta = meta
        self.fbos_sign = Scalar(label)


def meta(ni=1, nj=3, **over):
    base = {
        "temperature": 120.5, "beta0": 0.25,
        "etaL": 0.4, "etaR": 0.6, "lamE": 0.1, "lamM": 0.2,
        "alpha_vec": [0.1, -0.2, 0.35], "W": [[0.5, 0.25, 0.125]],
        "ni": ni, "nj": nj,
    }
    base.update(over)
    return base


def entry(ni=1, nj=3, label=0, **over):
    return {"graph": Graph(meta(ni=ni, nj=nj, **over), label=label),
            "ni": ni, "nj": nj}
