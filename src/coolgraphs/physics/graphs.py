"""Junction parameters -> attributed PyG graph (Fig. 1b of the paper).

Site nodes carry ``[alpha_i, T, log1p(eta_L), log1p(eta_R)]``; bath nodes carry
``[0, T, 0, 0]``. Site-site edges come from the off-diagonal Hamiltonian and
bath-site edges from the coupling matrix ``W``. The sampled parameters are kept
in ``graph.meta`` so any graph can be re-simulated.
"""
import numpy as np
import torch
from torch_geometric.data import Data

from .functions import create_connectivity_vectors, create_Heff, create_R

POS_THRESHOLD = 1e5


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def create_graph(ni, nj, alpha_vec, W, lamE=1000, lamM=0.5, beta0=-0.1, etaR=1, etaL=1, T=298, Fbos_sign=False,
                 heat_flux=0, heff_cutoff=1e-8):
    edge_types = []
    n = ni * nj  # Number of main nodes
    R = create_R(ni, nj)
    Heff = create_Heff(n, R, alphas=alpha_vec, lamM=lamM, beta0=beta0)

    edge_index = []
    edge_attr = []

    # add connections from main sites based on Heff (with cutoff)
    for i in range(n):
        for j in range(i + 1, n):
            if i != j and abs(Heff[i, j]) > heff_cutoff:
                edge_index.append([i, j])
                edge_attr.append(np.log1p(np.abs(Heff[i, j])))
                edge_types.append(0)  # site-site

    # electrode connections: stored as node features instead of edges
    conn_R, conn_L = create_connectivity_vectors(ni, nj, lamE=lamE, etaR=etaR, etaL=etaL)

    # Add bosonic bath nodes and edges using W
    bath_start_idx = n
    for j in range(n):
        bath_node = bath_start_idx + j
        for i in range(n):
            if W[i, j] > (min(etaR, etaL) * 0.0001):  # threshold
                edge_index.append([bath_node, i])  # Bath j to site i
                edge_attr.append(np.log1p(W[i, j]))
                edge_types.append(1)

    # Convert to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.abs(edge_attr), dtype=torch.float).view(-1, 1)

    num_baths = n

    # build node features
    # alpha_values for sites, 0 for baths
    alpha_features = np.append(alpha_vec, [0] * num_baths)

    # electrode features: each site has conn_L/conn_R, baths get 0
    conn_L_features = np.append(np.log1p(conn_L), [0] * num_baths)
    conn_R_features = np.append(np.log1p(conn_R), [0] * num_baths)

    # stack features: [alpha, temperature, conn_L, conn_R]
    x = np.vstack([
        alpha_features,
        [T] * (n + num_baths),
        conn_L_features,
        conn_R_features
    ]).T

    x = torch.tensor(x, dtype=torch.float)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # differentiate between parts of the system:
    node_type = [0] * n + [1] * num_baths  # 0: site, 1: bath
    graph.node_type = torch.tensor(node_type, dtype=torch.long)
    graph.edge_type = torch.tensor(edge_types, dtype=torch.long)

    # labels
    graph.fbos_sign = torch.tensor([int(heat_flux>POS_THRESHOLD)], dtype=torch.float)
    graph.heat_flux = torch.tensor([signed_log1p(float(heat_flux))], dtype=torch.float)

    # sampled parameters, so the oracle call can be reproduced from the graph
    graph.meta = {
        "temperature": float(T),
        "alpha_vec": np.array(alpha_vec, dtype=float),  # no tensor here
        "etaR": float(etaR),
        "etaL": float(etaL),
        "lamE": float(lamE),
        "lamM": float(lamM),
        "beta0": float(beta0),
        "ni": int(ni),
        "nj": int(nj),
        "W": np.array(W, dtype=float),  # no tensor here
    }

    return graph
