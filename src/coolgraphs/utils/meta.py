"""Per-graph physical parameters, flattened to a fixed-width matrix.

Saving this matrix next to a model's latent vectors is what makes latent-space
figures reproducible after the fact: any colouring (temperature, beta0, etaR,
alpha spread, heat flux, label, topology) becomes a column lookup instead of a
retraining run.

``META_COLS`` is the column order and is part of the on-disk format. Append new
columns at the END; never reorder or remove, or previously saved .npz files
become unreadable.
"""
from __future__ import annotations

import numpy as np

from .io import graph_of

META_COLS = [
    "T",            # temperature (K)
    "beta0",        # inter-site coupling
    "etaR",         # right electrode coupling
    "etaL",         # left electrode coupling
    "lamE",         # electrode lambda
    "lamM",         # mode lambda
    "ni",           # grid rows
    "nj",           # grid cols
    "n_sites",      # ni * nj
    "alpha_mean",   # on-site energy statistics
    "alpha_std",
    "alpha_range",
    "alpha_min",
    "alpha_max",
    "heat_flux",    # signed_log1p(F_bos), as stored on the graph
    "label",        # fbos_sign (0/1)
]

# graph.meta spells temperature out; every other column matches its key.
_META_KEY = {"T": "temperature"}


def _alpha_stat(meta: dict, stat: str) -> float:
    a = np.asarray(meta.get("alpha_vec", []), dtype=float).ravel()
    if a.size == 0:
        return float("nan")
    if stat == "mean":
        return float(a.mean())
    if stat == "std":
        return float(a.std())
    if stat == "min":
        return float(a.min())
    if stat == "max":
        return float(a.max())
    if stat == "range":
        return float(a.max() - a.min())
    return float("nan")


def meta_row(graph) -> list:
    """One row of META_COLS for a single graph."""
    meta = getattr(graph, "meta", None) or {}
    row = []
    for col in META_COLS:
        if col == "n_sites":
            row.append(float(meta.get("ni", np.nan)) * float(meta.get("nj", np.nan)))
        elif col.startswith("alpha_"):
            row.append(_alpha_stat(meta, col.split("_", 1)[1]))
        elif col == "heat_flux":
            hf = getattr(graph, "heat_flux", None)
            row.append(float(np.asarray(hf).ravel()[0]) if hf is not None else np.nan)
        elif col == "label":
            row.append(float(graph.fbos_sign.item()))
        else:
            row.append(float(meta.get(_META_KEY.get(col, col), np.nan)))
    return row


def meta_matrix(entries) -> np.ndarray:
    """[N, len(META_COLS)], row-aligned to ``entries``."""
    return np.asarray([meta_row(graph_of(e)) for e in entries], dtype=float)


def inverse_signed_log1p(y):
    """heat_flux is stored as sign(x)*log1p(|x|); this recovers F_bos in eV/s."""
    y = np.asarray(y, dtype=float)
    return np.sign(y) * np.expm1(np.abs(y))
