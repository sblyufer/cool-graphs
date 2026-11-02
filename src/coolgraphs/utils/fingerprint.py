"""A stable hash of the sampled parameters behind one dataset entry.

Two entries with the same fingerprint came from the same oracle call inputs.
Used to guarantee that no acquired point ever duplicates a held-out graph.
"""
from __future__ import annotations

import hashlib

from .io import graph_of

# The sampled quantities that define one oracle call, in a fixed order.
PARAM_KEYS = ("temperature", "beta0", "etaL", "etaR", "lamE", "lamM",
              "alpha_vec", "W", "ni", "nj")


def _blob(value) -> bytes:
    """Stable bytes for a scalar, sequence or array, at full precision."""
    if value is None:
        return b"None"
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (list, tuple)):
        return b"[" + b",".join(_blob(v) for v in value) + b"]"
    if isinstance(value, bool):
        return b"1" if value else b"0"
    if isinstance(value, int):
        return str(value).encode()
    if isinstance(value, float):
        return repr(value).encode()     # repr round-trips a float exactly
    return repr(value).encode()


def fingerprint(entry) -> str:
    """Hex digest of one entry's sampled parameters (``graph.meta``)."""
    graph = graph_of(entry)
    meta = getattr(graph, "meta", None) or {}
    h = hashlib.blake2b(digest_size=16)
    missing = []
    for key in PARAM_KEYS:
        if key in meta:
            h.update(key.encode())
            h.update(b"=")
            h.update(_blob(meta[key]))
            h.update(b";")
        else:
            missing.append(key)
    if len(missing) == len(PARAM_KEYS):
        # No parameter record: fall back to the graph tensors.
        for attr in ("x", "edge_index", "edge_attr", "fbos"):
            val = getattr(graph, attr, None)
            if val is not None:
                h.update(attr.encode())
                h.update(_blob(val))
    return h.hexdigest()
