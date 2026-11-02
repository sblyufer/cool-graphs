"""Feature normalisation.

Statistics are always computed on the training split and applied to every
other split. Node features, edge attributes and the (signed-log) heat flux are
z-scored.
"""
import torch


def compute_stats(graphs):
    # Collect node feature stats per column
    all_nodes = torch.cat([g.x for g in graphs], dim=0)  # shape: [total_nodes, num_features]
    node_means = all_nodes.mean(dim=0)
    node_stds = all_nodes.std(dim=0)
    node_mins = all_nodes.min(dim=0)[0]
    node_maxs = all_nodes.max(dim=0)[0]

    # Collect edge feature stats
    all_edges = torch.cat([g.edge_attr for g in graphs], dim=0)  # shape: [total_edges, 1]
    edge_mean = all_edges.mean()
    edge_std = all_edges.std()
    edge_min = all_edges.min()
    edge_max = all_edges.max()
    
    # Collect heat flux stats
    heat_fluxes = torch.cat([g.heat_flux for g in graphs], dim=0)
    heat_mean = heat_fluxes.mean()
    heat_std = heat_fluxes.std()

    return {
        "node_means": node_means,
        "node_stds": node_stds,
        "node_mins": node_mins,
        "node_maxs": node_maxs,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "edge_min": edge_min,
        "edge_max": edge_max,
        "heat_mean": heat_mean,
        "heat_std": heat_std
    }


def normalize_graphs(graphs, stats, method="zscore"):
    normed_graphs = []
    for g in graphs:
        g = g.clone()

        if method == "zscore":
            g.x = (g.x - stats["node_means"]) / (stats["node_stds"] + 1e-8)
            g.edge_attr = (g.edge_attr - stats["edge_mean"]) / (stats["edge_std"] + 1e-8)
            g.heat_flux = (g.heat_flux - stats["heat_mean"]) / (stats["heat_std"] + 1e-8)

        elif method == "minmax":
            g.x = (g.x - stats["node_mins"]) / (stats["node_maxs"] - stats["node_mins"] + 1e-8)
            g.edge_attr = (g.edge_attr - stats["edge_min"]) / (stats["edge_max"] - stats["edge_min"] + 1e-8)

        normed_graphs.append(g)

    return normed_graphs
