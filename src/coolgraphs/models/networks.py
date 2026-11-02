"""Network architectures.

``GraphVAE``      GAT encoder -> (mu, log sigma) per node, auxiliary node /
                  edge decoders, attention pooling and an MLP classifier head.
                  ``variational=False`` gives the deterministic autoencoder (AE).
``GNNClassifier`` the discriminative baselines: the same input embedding,
                  pooling and classifier head on a GAT or GCN encoder, with no
                  decoders and no KL term.

The classifier head contains the p=0.2 dropout layer that MC-dropout
acquisition samples from.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GlobalAttention, VGAE, global_mean_pool


# ========================================
# Encoders
# ========================================
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=32, num_node_types=2, num_edge_types=2):
        super().__init__()
        self.node_embed = nn.Embedding(num_node_types, hidden_dim)
        self.linear_x = nn.Linear(in_channels, hidden_dim)

        self.conv1 = GCNConv(hidden_dim, hidden_dim)

        # Split into two heads for VAE: mu and logstd
        self.conv_mu = GCNConv(hidden_dim, out_channels)
        self.conv_logstd = GCNConv(hidden_dim, out_channels)

    def forward(self, x, edge_index, edge_attr=None, node_type=None):
        x = self.linear_x(x)
        if node_type is not None:
            x = x + self.node_embed(node_type)

        edge_weight = edge_attr.view(-1) if edge_attr is not None else None

        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))

        mu = self.conv_mu(x, edge_index, edge_weight=edge_weight)
        logstd = self.conv_logstd(x, edge_index, edge_weight=edge_weight)
        return mu, logstd


class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=32, num_node_types=2, num_edge_types=2, heads=2):
        super().__init__()
        self.node_embed = nn.Embedding(num_node_types, hidden_dim)
        self.edge_embed = nn.Embedding(num_edge_types, hidden_dim)
        self.linear_x = nn.Linear(in_channels, hidden_dim)

        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=True, edge_dim=1 + hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)

        # Split into two heads for VAE
        self.gat_mu = GATConv(hidden_dim * heads, out_channels, heads=1, concat=False, edge_dim=1 + hidden_dim)
        self.gat_logstd = GATConv(hidden_dim * heads, out_channels, heads=1, concat=False, edge_dim=1 + hidden_dim)

    def forward(self, x, edge_index, edge_attr=None, node_type=None, edge_type=None):
        x = self.linear_x(x)
        if node_type is not None:
            x = x + self.node_embed(node_type)

        if edge_attr is not None and edge_type is not None:
            embedded_edges = self.edge_embed(edge_type)
            edge_features = torch.cat([edge_attr.view(-1, 1), embedded_edges], dim=1)
        else:
            edge_features = edge_attr

        x = self.gat1(x, edge_index, edge_attr=edge_features)
        x = self.norm1(x)
        x = F.elu(x)

        mu = self.gat_mu(x, edge_index, edge_attr=edge_features)
        logstd = self.gat_logstd(x, edge_index, edge_attr=edge_features)
        return mu, logstd


# ========================================
# Edge Decoder
# ========================================
class MLPEdgeDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=32):
        super().__init__()
        # Concatenating two node latents allows for asymmetric or complex logic
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, edge_index, sigmoid=True):
        src, dst = z[edge_index[0]], z[edge_index[1]]
        combined = torch.cat([src, dst], dim=-1) # [E, 2*latent_dim]
        logits = self.net(combined).view(-1)
        return torch.sigmoid(logits) if sigmoid else logits
        
# ========================================
# Node Decoder
# ========================================
class NodeDecoder(nn.Module):
    """Reconstructs alpha, T, etc. from latent z"""

    def __init__(self, latent_dim, hidden_dim, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, z):
        return self.net(z)


# ========================================
# Edge Regression Decoder
# ========================================
class EdgeRegressor(nn.Module):
    """
    Reconstructs continuous edge attributes (beta0, coupling constants)
    from the latent representations of two connected nodes.
    """

    def __init__(self, latent_dim, hidden_dim, out_channels=1):
        super().__init__()
        # We concatenate two node latents, so input_dim is 2 * latent_dim
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, z, edge_index):
        """
        Args:
            z: Latent node representations [N, latent_dim]
            edge_index: Graph connectivity [2, E]
        Returns:
            Predicted edge attributes [E, out_channels]
        """
        # 1. Gather the latent vectors for the source and destination of each edge
        row, col = edge_index
        z_src = z[row] # [E, latent_dim]
        z_dst = z[col] # [E, latent_dim]

        # 2. Concatenate the node representations
        # Using [E, 2 * latent_dim] allows the MLP to learn the interaction
        combined = torch.cat([z_src, z_dst], dim=-1)

        # 3. Predict the continuous physical value (e.g., beta0)
        return self.net(combined).view(-1)


# ========================================
# Graph VAE Model
# ========================================
class GraphVAE(VGAE):
    def __init__(self, in_channels, out_channels, hidden_dim=32, predictor=None, num_node_types=2, num_edge_types=2,
                 encoder_type="GAT", decoder_type="default", pool_type="mean", variational=True):

        # 1. Encoder
        if encoder_type == "GAT":
            encoder = GATEncoder(in_channels, out_channels, hidden_dim, num_node_types, num_edge_types, heads=2)
        else:
            encoder = GCNEncoder(in_channels, out_channels, hidden_dim, num_node_types, num_edge_types)

        # 2. Edge Decoder (Topology/Weights)
        if decoder_type == "custom":
            decoder = MLPEdgeDecoder(out_channels, hidden_dim)
            super().__init__(encoder, decoder)
            
        else:
            super().__init__(encoder)

        # Modular Decoders
        self.node_decoder = NodeDecoder(out_channels, hidden_dim, in_channels)
        self.edge_value_regressor = EdgeRegressor(out_channels, hidden_dim, out_channels=1)

        self._classifier = None
        self._regressor = None
        self._decoder_type = decoder_type
        self.pool_type = pool_type.lower()
        self.latent_dim = out_channels
        self.variational = variational

        # Pooling & Predictors
        if self.pool_type == "attention":
            self.pool_norm = nn.LayerNorm(out_channels)
            self.gate_nn = torch.nn.Sequential(
                torch.nn.Linear(out_channels, out_channels // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(out_channels // 2, 1)
            )
            self.att_pool = GlobalAttention(self.gate_nn)

        if predictor == "Classifier_Head":
            self._classifier = nn.Sequential(
                nn.Linear(out_channels, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        if predictor == "Regressor_Head":
            self._regressor = nn.Sequential(
                nn.Linear(out_channels, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, data):
        # 1. Encode
        z = self.encode(data.x, data.edge_index, data.edge_attr, data.node_type, data.edge_type)
        if not self.variational:
            # AE mode: deterministic latent (no sampling). __mu__ is set by VGAE.encode.
            z = self.__mu__

        # 2. Reconstruct Nodes and Edges
        x_recon = self.node_decoder(z)
        edge_prob = self.decoder(z, data.edge_index)
        edge_recon = self.edge_value_regressor(z, data.edge_index)

        # 3. Pooling & Prediction
        if self.pool_type == "attention":
            z = self.pool_norm(z)
            graph_repr = self.att_pool(z, data.batch)
        else:
            graph_repr = global_mean_pool(z, data.batch)

        pred = None
        if self._regressor:
            pred = self._regressor(graph_repr).view(-1)
        elif self._classifier:
            pred = self._classifier(graph_repr).view(-1)

        return z, edge_prob, x_recon, edge_recon, pred


# ========================================
# Discriminative baseline (no VAE parts)
# ========================================
class GNNClassifier(nn.Module):
    """
    Discriminative baseline: an encoder stack (GAT *or* GCN)
    + attention pooling + classifier head, with NO reparameterization, NO node/edge
    decoders, and NO KL term.

    encoder_type="GAT" -> attention message passing (edge features used).
    encoder_type="GCN" -> plain message passing, scalar edge weights ("regular GNN").

    It deliberately duck-types GraphVAE so the Lightning wrapper and the MC-dropout AL
    loop accept it unchanged: forward() returns the same 5-tuple
    (z, edge_prob, x_recon, edge_recon, pred) with the generative slots set to None,
    and it exposes .att_pool / .pool_type / ._classifier / ._regressor.
    """

    def __init__(self, in_channels, out_channels, hidden_dim=32, predictor="Classifier_Head",
                 num_node_types=2, num_edge_types=2, encoder_type="GAT", decoder_type="default",
                 pool_type="attention", heads=2):
        super().__init__()
        self.latent_dim = out_channels
        self.pool_type = pool_type.lower()
        self.encoder_type = encoder_type.upper()

        # --- Shared input embedding ---
        self.node_embed = nn.Embedding(num_node_types, hidden_dim)
        self.linear_x = nn.Linear(in_channels, hidden_dim)

        # --- Encoder stack (single deterministic head; no mu/logstd split) ---
        if self.encoder_type == "GAT":
            self.edge_embed = nn.Embedding(num_edge_types, hidden_dim)
            self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=True, edge_dim=1 + hidden_dim)
            self.norm1 = nn.LayerNorm(hidden_dim * heads)
            self.conv2 = GATConv(hidden_dim * heads, out_channels, heads=1, concat=False, edge_dim=1 + hidden_dim)
        else:  # GCN: plain message passing, scalar edge weights, no attention
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_channels)

        # --- Same pooling as GraphVAE ---
        if self.pool_type == "attention":
            self.pool_norm = nn.LayerNorm(out_channels)
            self.gate_nn = nn.Sequential(
                nn.Linear(out_channels, out_channels // 2),
                nn.ReLU(),
                nn.Linear(out_channels // 2, 1),
            )
            self.att_pool = GlobalAttention(self.gate_nn)

        # --- Same classifier head (keeps the p=0.2 Dropout that MC-dropout relies on) ---
        self._classifier = None
        self._regressor = None
        if predictor == "Classifier_Head":
            self._classifier = nn.Sequential(
                nn.Linear(out_channels, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        elif predictor == "Regressor_Head":
            self._regressor = nn.Sequential(
                nn.Linear(out_channels, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim, 1),
            )

    def encode(self, x, edge_index, edge_attr=None, node_type=None, edge_type=None):
        x = self.linear_x(x)
        if node_type is not None:
            x = x + self.node_embed(node_type)

        if self.encoder_type == "GAT":
            if edge_attr is not None and edge_type is not None:
                edge_features = torch.cat([edge_attr.view(-1, 1), self.edge_embed(edge_type)], dim=1)
            else:
                edge_features = edge_attr
            x = F.elu(self.norm1(self.conv1(x, edge_index, edge_attr=edge_features)))
            z = self.conv2(x, edge_index, edge_attr=edge_features)
        else:  # GCN
            # GCNConv's degree normalisation needs non-negative weights, but the
            # edge features are z-scored (signed); softplus keeps them positive.
            ew = F.softplus(edge_attr.view(-1)) if edge_attr is not None else None
            x = F.relu(self.norm1(self.conv1(x, edge_index, edge_weight=ew)))
            z = self.conv2(x, edge_index, edge_weight=ew)
        return z

    def forward(self, data):
        z = self.encode(data.x, data.edge_index, data.edge_attr, data.node_type, data.edge_type)

        if self.pool_type == "attention":
            graph_repr = self.att_pool(self.pool_norm(z), data.batch)
        else:
            graph_repr = global_mean_pool(z, data.batch)

        pred = None
        if self._regressor:
            pred = self._regressor(graph_repr).view(-1)
        elif self._classifier:
            pred = self._classifier(graph_repr).view(-1)

        # Match GraphVAE's signature; generative outputs are None for the baseline.
        return z, None, None, None, pred
