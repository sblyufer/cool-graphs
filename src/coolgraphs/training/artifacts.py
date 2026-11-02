"""Per-run arrays, saved so every figure and metric can be recomputed without
retraining.

Each split is written as ``<split>.npz`` with:

    z        [N, latent_dim]  graph-level latent vectors
    y        [N]              true label
    score    [N]              predicted probability (pre-threshold)
    meta     [N, 16]          physical parameters, in META_COLS order
    topo     [N]              'NxM' strings
    idx      [N]              row index into the source dataset

plus a ``run.json`` with the configuration and metrics.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from ..utils.meta import META_COLS, meta_matrix
from ..utils.topology import topo_name, topology_of


def run_dir(root: str, run_id: str) -> str:
    d = os.path.join(root, run_id)
    os.makedirs(d, exist_ok=True)
    return d


def save_split(path: str, *, y, score, z=None, entries=None,
               topologies=None, idx=None, extra: dict | None = None) -> str:
    """Write one split's arrays to ``path`` (.npz), atomically.

    ``entries`` is used to derive ``meta`` and ``topo``. ``idx`` is each row's
    index into the source dataset, so anything ``meta`` does not carry (the
    per-site alpha vector, W, the raw graph) can be recovered later.
    """
    y = np.asarray(y).ravel()
    score = np.asarray(score).ravel()
    if y.size != score.size:
        raise ValueError(f"y has {y.size} entries, score has {score.size}")

    payload: dict[str, np.ndarray] = {"y": y.astype(np.int8),
                                      "score": score.astype(np.float32)}

    if z is not None:
        z = np.asarray(z)
        if z.shape[0] != y.size:
            raise ValueError(
                f"z has {z.shape[0]} rows but y has {y.size}; the latent "
                f"vectors are not aligned with the labels")
        payload["z"] = z.astype(np.float32)

    if entries is not None:
        if len(entries) != y.size:
            raise ValueError(
                f"{len(entries)} entries but {y.size} labels; a shuffled "
                f"DataLoader will do this -- predict with shuffle=False")
        payload["meta"] = meta_matrix(entries).astype(np.float32)
        payload["meta_cols"] = np.array(META_COLS)
        if topologies is None:
            topologies = [topo_name(topology_of(e)) for e in entries]

    if topologies is not None:
        topo = np.asarray(topologies).ravel()
        if topo.size != y.size:
            raise ValueError(f"topologies has {topo.size} entries, y has {y.size}")
        payload["topo"] = topo.astype("U8")

    if idx is not None:
        idx = np.asarray(idx).ravel()
        if idx.size != y.size:
            raise ValueError(
                f"idx has {idx.size} entries, y has {y.size}; these must be "
                f"the dataset indices of exactly the rows saved here")
        payload["idx"] = idx.astype(np.int64)

    for k, v in (extra or {}).items():
        payload[k] = np.asarray(v)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)
    return path


def save_run(root: str, run_id: str, splits: dict, config: dict,
             metrics: dict | None = None) -> str:
    """Save every split of one run plus its config and metrics.

    ``splits`` maps a split name to the kwargs of ``save_split``:

        {"test": {"y": ..., "score": ..., "z": ..., "entries": [...]}, ...}
    """
    d = run_dir(root, run_id)
    written = []
    for name, kw in splits.items():
        written.append(save_split(os.path.join(d, f"{name}.npz"), **kw))

    sidecar = {
        "run_id": run_id,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splits": [os.path.basename(p) for p in written],
        "meta_cols": META_COLS,
        "config": config,
    }
    if metrics is not None:
        sidecar["metrics"] = metrics
    with open(os.path.join(d, "run.json"), "w") as f:
        json.dump(sidecar, f, indent=2, default=str)
    return d


def load_split(path: str) -> dict:
    """Read one split back as a plain dict of arrays."""
    with np.load(path, allow_pickle=False) as f:
        out = {k: f[k] for k in f.files}
    if "meta" in out and "meta_cols" in out:
        out["meta_by_name"] = {c: out["meta"][:, i]
                               for i, c in enumerate(out["meta_cols"].tolist())}
    return out


def recover_entries(split: dict, dataset_path: str) -> list:
    """The original graph entries behind a saved split, via ``idx``.

    Use this when a figure needs something ``meta`` does not carry -- the
    per-site alpha vector, the coupling matrix W, the raw graph. It reloads
    the dataset and selects the saved rows in their saved order, so the result
    is aligned with ``z``, ``y`` and ``score`` row for row.
    """
    if "idx" not in split:
        raise KeyError(
            "this split was saved without idx, so its rows cannot be traced "
            "back to the dataset. Only the 16 meta columns are available.")
    from ..utils.io import load_entries
    entries = load_entries(dataset_path)
    idx = split["idx"]
    if idx.max(initial=-1) >= len(entries):
        raise IndexError(
            f"idx reaches row {int(idx.max())} but {dataset_path} has only "
            f"{len(entries)} entries -- wrong dataset, or it changed since "
            f"the run (check its provenance ledger)")
    return [entries[i] for i in idx]


def meta_column(split: dict, name: str) -> np.ndarray:
    """One physical parameter by name, for colouring a latent plot."""
    if "meta_by_name" not in split:
        raise KeyError("this split was saved without meta; pass entries= to save_split")
    if name not in split["meta_by_name"]:
        raise KeyError(
            f"{name!r} is not a stored column. Available: "
            f"{sorted(split['meta_by_name'])}")
    return split["meta_by_name"][name]
