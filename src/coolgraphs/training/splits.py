"""Train / validation / test splits, stratified by topology AND label.

Each (topology, label) stratum is shuffled with its own seed derived from
``split_seed`` and cut at the same fractions, so every split has the
dataset's composition up to integer rounding. ``split_seed`` (default 42) is
independent of the training seed: every model and seed is scored on the same
held-out sets, which keeps comparisons paired.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..utils.io import label_of
from ..utils.topology import topo_name, topology_of

DEFAULT_SPLIT_SEED = 42


def stratum_of(entry) -> tuple[int, int, int]:
    """The (ni, nj, label) cell an entry belongs to."""
    ni, nj = topology_of(entry)
    return (ni, nj, label_of(entry))


def stratified_indices(entries, val_fraction: float = 0.15,
                       test_fraction: float = 0.15,
                       split_seed: int = DEFAULT_SPLIT_SEED
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index arrays for train / val / test, stratified by (topology, label).

    Each stratum is shuffled with its own derived seed and cut at the same
    fractions, so the composition of every split matches the whole dataset's
    as closely as integer counts allow.

    Small strata are handled explicitly rather than silently: a stratum with
    too few members to give at least one sample to val and test still yields
    whatever it can, and the shortfall is reported by ``describe_splits``.
    """
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("fractions must be in (0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError(
            f"val_fraction + test_fraction = {val_fraction + test_fraction} "
            f"leaves nothing for training")

    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        buckets[stratum_of(e)].append(i)

    train, val, test = [], [], []
    # Sorted so the assignment does not depend on dict insertion order, which
    # depends on the order entries happen to sit in the .pkl.
    for key in sorted(buckets):
        idx = np.array(buckets[key], dtype=np.int64)
        # A seed per stratum, derived from split_seed: adding a topology to the
        # dataset later does not reshuffle the strata that already existed.
        rng = np.random.default_rng([split_seed, *key])
        rng.shuffle(idx)

        n = len(idx)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        # Guarantee the training split is never starved for a tiny stratum.
        while n_test + n_val >= n and (n_test > 0 or n_val > 0):
            if n_test >= n_val:
                n_test -= 1
            else:
                n_val -= 1

        test.append(idx[:n_test])
        val.append(idx[n_test:n_test + n_val])
        train.append(idx[n_test + n_val:])

    out = tuple(np.sort(np.concatenate(p)) if p else np.array([], dtype=np.int64)
                for p in (train, val, test))
    total = sum(len(p) for p in out)
    if total != len(entries):
        raise AssertionError(
            f"split lost or duplicated entries: {total} != {len(entries)}")
    return out


def split_entries(entries, **kw) -> tuple[list, list, list]:
    """``stratified_indices`` applied, returning the entries themselves."""
    tr, va, te = stratified_indices(entries, **kw)
    return ([entries[i] for i in tr],
            [entries[i] for i in va],
            [entries[i] for i in te])


def describe_splits(entries, splits, name: str = "split") -> dict:
    """Print and return the composition of each split, per topology.

    Worth reading once per dataset: it is the cheapest place to notice that a
    rare topology has single-digit positives in test, which is the difference
    between a PR-AUC you can report and one you cannot.
    """
    train_i, val_i, test_i = splits
    rows: dict[str, dict] = {}
    by_split = {"train": train_i, "val": val_i, "test": test_i}

    topos = sorted({topology_of(e) for e in entries})
    print(f"\n=== {name}: composition ===")
    header = f"{'topology':>9} " + " ".join(f"{s:>16}" for s in by_split)
    print(header)
    for t in topos:
        cells = {}
        line = f"{topo_name(t):>9} "
        for sname, idx in by_split.items():
            n = sum(1 for i in idx if topology_of(entries[i]) == t)
            pos = sum(1 for i in idx
                      if topology_of(entries[i]) == t and label_of(entries[i]) == 1)
            cells[sname] = {"n": n, "pos": pos,
                            "rate": (pos / n) if n else 0.0}
            line += f"{n:>7d}/{pos:<3d}({100 * cells[sname]['rate']:>4.1f}%)"[:17].rjust(17)
        rows[topo_name(t)] = cells
        print(line)

    print(f"{'TOTAL':>9} ", end="")
    for sname, idx in by_split.items():
        n = len(idx)
        pos = sum(1 for i in idx if label_of(entries[i]) == 1)
        rate = (pos / n) if n else 0.0
        print(f"{n:>7d}/{pos:<3d}({100 * rate:>4.1f}%)"[:17].rjust(17), end="")
    print("\n           (cells are n / positives (positive rate))")

    thin = [t for t, c in rows.items() if c["test"]["pos"] < 10]
    if thin:
        print(f"\nWARNING: fewer than 10 test positives for {', '.join(thin)}. "
              f"Per-topology PR-AUC there will have very wide error bars; "
              f"report a bootstrap CI or pool those topologies.")
    return rows


def pos_weight_for(entries) -> float:
    """``n_negative / n_positive`` -- the weight for the positive class.

    This is the standard BCE ``pos_weight``: it makes the positive and negative
    classes contribute equally to the loss in expectation. On the natural
    dataset (1,698 of 40,000) it is about 22.6.
    """
    pos = sum(1 for e in entries if label_of(e) == 1)
    neg = len(entries) - pos
    if pos == 0:
        raise ValueError("no positive samples; pos_weight is undefined")
    return neg / pos
