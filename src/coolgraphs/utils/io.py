"""Reading and writing graph datasets.

A dataset on disk is a pickled list of *entries*. An entry is either a dict
``{"graph": Data, "ni": int, "nj": int}`` or a bare PyG ``Data`` object. Both
forms appear in datasets produced over the life of this project, so everything
here accepts either.
"""
from __future__ import annotations

import glob
import os
import pickle as pkl


def load_entries(path: str) -> list:
    """Load a dataset .pkl as a list of entries.

    Raises a readable error, listing nearby .pkl files, rather than a bare
    FileNotFoundError traceback.
    """
    if not os.path.exists(path):
        here = os.path.dirname(os.path.abspath(path)) or "."
        candidates = []
        for d in dict.fromkeys([here, os.getcwd(), os.path.join(os.getcwd(), "data")]):
            if os.path.isdir(d):
                candidates += sorted(glob.glob(os.path.join(d, "*.pkl")))
        lines = [f"Dataset not found: {path}",
                 f"  (resolved to {os.path.abspath(path)})"]
        uniq = list(dict.fromkeys(candidates))[:25]
        lines.append("\n.pkl files I can see:" if uniq else "\nNo .pkl files found nearby.")
        lines += [f"    {p}" for p in uniq]
        raise SystemExit("\n".join(lines))

    with open(path, "rb") as f:
        data = pkl.load(f)
    if isinstance(data, dict) and "graphs" in data:
        return data["graphs"]
    return data


def save_entries(entries: list, path: str) -> None:
    """Write entries atomically, so a killed job never leaves a truncated .pkl."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pkl.dump(entries, f)
    os.replace(tmp, path)


def graph_of(entry):
    """The graph object for ONE entry."""
    return entry["graph"] if isinstance(entry, dict) else entry


def graphs_of(entries) -> list:
    """The graph objects for a LIST of entries."""
    return [graph_of(e) for e in entries]


def label_of(entry) -> int:
    """Binary cooling label (``fbos_sign``) for one entry."""
    return int(graph_of(entry).fbos_sign.item())
