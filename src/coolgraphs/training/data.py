"""Dataset -> stratified splits -> normalised DataLoaders.

The split is stratified by (topology, label) with a fixed ``split_seed`` (see
``splits.py``); normalisation statistics come from the training split only;
evaluation loaders never shuffle, so saved predictions stay aligned with the
dataset rows they came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from torch_geometric.loader import DataLoader

from ..utils.io import graph_of, label_of, load_entries
from ..utils.topology import topo_name, topology_of
from .splits import describe_splits, pos_weight_for, stratified_indices


@dataclass
class Splits:
    """Loaders, plus the raw entries behind them in the same order."""
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    stats: object
    train_entries: list
    val_entries: list
    test_entries: list
    ext_loader: DataLoader | None = None
    ext_entries: list = field(default_factory=list)
    pos_weight: float = 1.0
    composition: dict = field(default_factory=dict)
    # Row indices into the source dataset, one array per split.
    train_idx: object = None
    val_idx: object = None
    test_idx: object = None
    # Extra held-out sets beyond train/val/test (here: "ext", the unseen
    # topologies), each {"loader": ..., "entries": [...], "idx": array}.
    eval_sets: dict = field(default_factory=dict)

    def names(self) -> list[str]:
        return ["train", "val", "test"] + sorted(self.eval_sets)

    def indices(self, split: str):
        if split in self.eval_sets:
            return self.eval_sets[split]["idx"]
        return getattr(self, f"{split}_idx")

    def entries(self, split: str) -> list:
        if split in self.eval_sets:
            return self.eval_sets[split]["entries"]
        return getattr(self, f"{split}_entries")

    def predict_loader(self, split: str, batch_size: int = 256,
                       num_workers: int = 2) -> DataLoader:
        """An unshuffled loader for `split`, so row i is entry i."""
        loader = (self.eval_sets[split]["loader"] if split in self.eval_sets
                  else getattr(self, f"{split}_loader", None))
        if loader is None:
            raise KeyError(f"no {split} loader on this Splits object")
        return DataLoader(loader.dataset, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)

    def topologies(self, split: str) -> list[str]:
        entries = getattr(self, f"{split}_entries")
        return [topo_name(topology_of(e)) for e in entries]

    def labels(self, split: str) -> list[int]:
        return [label_of(e) for e in getattr(self, f"{split}_entries")]


def prepare(dataset_path: str, *, batch_size: int = 64,
            val_fraction: float = 0.15, test_fraction: float = 0.15,
            split_seed: int = 42, ext_path: str | None = None,
            num_workers: int = 4, verbose: bool = True) -> Splits:
    from ..datasets.normalise import compute_stats, normalize_graphs

    entries = load_entries(dataset_path)
    if verbose:
        print(f"Loaded {len(entries)} entries from {dataset_path}")

    idx = stratified_indices(entries, val_fraction=val_fraction,
                             test_fraction=test_fraction, split_seed=split_seed)
    composition = (describe_splits(entries, idx, dataset_path) if verbose
                   else {})
    train_e, val_e, test_e = ([entries[i] for i in part] for part in idx)

    # Statistics from the training split only.
    stats = compute_stats([graph_of(e) for e in train_e])

    def loader(es, shuffle):
        graphs = normalize_graphs([graph_of(e) for e in es], stats, method="zscore")
        return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers)

    out = Splits(
        train_loader=loader(train_e, True),
        val_loader=loader(val_e, False),
        test_loader=loader(test_e, False),
        stats=stats,
        train_entries=train_e, val_entries=val_e, test_entries=test_e,
        pos_weight=pos_weight_for(train_e),
        composition=composition,
        train_idx=idx[0], val_idx=idx[1], test_idx=idx[2],
    )

    if ext_path:
        ext_e = load_entries(ext_path)
        out.ext_entries = ext_e
        out.ext_loader = loader(ext_e, False)
        out.eval_sets["ext"] = {"loader": out.ext_loader, "entries": ext_e,
                                "idx": np.arange(len(ext_e))}
        if verbose:
            print(f"Extrapolation set: {len(ext_e)} entries from {ext_path}")

    if verbose:
        print(f"pos_weight from the training split: {out.pos_weight:.2f}")
    return out
