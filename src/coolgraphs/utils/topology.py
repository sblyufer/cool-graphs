"""Topology identification and per-topology summaries.

A *topology* is the (ni, nj) site grid of a junction. Positive-flux hit rates
vary strongly with it -- roughly 9% for short 1D chains down to under 3% for 2D
grids -- so almost every analysis in this project is stratified by topology.
"""
from __future__ import annotations

from collections import defaultdict

from .io import graph_of, label_of


def topology_of(entry) -> tuple[int, int]:
    """(ni, nj) for an entry, from the dict wrapper or from ``graph.meta``."""
    if isinstance(entry, dict) and "ni" in entry and "nj" in entry:
        return (int(entry["ni"]), int(entry["nj"]))
    meta = getattr(graph_of(entry), "meta", None)
    if meta and "ni" in meta and "nj" in meta:
        return (int(meta["ni"]), int(meta["nj"]))
    raise KeyError("entry has no ni/nj in its dict wrapper or in graph.meta")


def topo_name(topology: tuple[int, int]) -> str:
    """(3, 2) -> '3x2'."""
    return f"{topology[0]}x{topology[1]}"


def parse_topologies(values) -> set:
    """['1x2', '3x3'] -> {(1, 2), (3, 3)}."""
    out = set()
    for v in values or []:
        a, b = v.strip().lower().split("x")
        out.add((int(a), int(b)))
    return out


def group_by_topology(entries) -> dict:
    """-> {(ni, nj): {"pos": [...], "neg": [...]}}"""
    groups = defaultdict(lambda: {"pos": [], "neg": []})
    for e in entries:
        groups[topology_of(e)]["pos" if label_of(e) == 1 else "neg"].append(e)
    return dict(groups)


def hit_rates(entries) -> dict:
    """-> {(ni, nj): {"n": int, "pos": int, "rate": float}}"""
    out = {}
    for t, v in group_by_topology(entries).items():
        n = len(v["pos"]) + len(v["neg"])
        out[t] = {"n": n, "pos": len(v["pos"]), "rate": len(v["pos"]) / n if n else 0.0}
    return out


def describe(entries, name: str = "dataset") -> dict:
    """Print size, positive rate and the per-topology breakdown. Returns hit_rates."""
    n = len(entries)
    if not n:
        print(f"{name}: empty")
        return {}
    rates = hit_rates(entries)
    pos = sum(r["pos"] for r in rates.values())
    print(f"{name}: {n} graphs, {pos} positives ({pos / n:.2%})")
    for t in sorted(rates):
        r = rates[t]
        print(f"    {topo_name(t):<6} {r['n']:>7} graphs  {r['pos']:>5} pos  {r['rate']:>6.2%}")
    return rates
