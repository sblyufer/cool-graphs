"""Training loss for the generative models.

    L = beta_recon * L_recon + beta_kl * L_KL + beta_pred * L_pred

L_recon is the sum of an edge-weight MSE, a link-reconstruction BCE against
sampled negative edges, and a node-feature MSE. L_pred is binary
cross-entropy on the positive-flux label. For the deterministic autoencoder
(``variational=False``) the KL term is zero. Optional class weighting
(``pos_weight``) rescales per-sample weights to mean 1 so it does not change
the balance between the three terms; all results in the paper are unweighted.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


def bce_with_class_weight(pred, target, pos_weight: float | None = None):
    """Binary cross-entropy on probabilities, optionally class-weighted.

    ``pred`` is already sigmoid-ed by the classifier head, so this uses
    ``binary_cross_entropy`` with per-sample weights rather than
    ``binary_cross_entropy_with_logits`` and its ``pos_weight`` argument.
    """
    target = target.float()
    if not pos_weight or pos_weight == 1.0:
        return F.binary_cross_entropy(pred, target)

    w = torch.where(target > 0.5,
                    torch.as_tensor(float(pos_weight), device=pred.device),
                    torch.as_tensor(1.0, device=pred.device))
    w = w / w.mean().clamp_min(1e-12)        # keep the loss scale unchanged
    return F.binary_cross_entropy(pred, target, weight=w)


def loss_calc(model, z, x_recon, edge_recon, pred, data, betas,
              predictor=None, pos_weight: float | None = None):
    """Returns (total, recon, kl, pred) for one batch."""
    recon_loss_edges = F.mse_loss(edge_recon, data.edge_attr.view(-1))

    pos_pred = model.decoder(z, data.edge_index)
    pos_loss = -torch.log(pos_pred + 1e-15).mean()
    neg_edge_index = negative_sampling(edge_index=data.edge_index,
                                       num_nodes=z.size(0))
    neg_pred = model.decoder(z, neg_edge_index)
    neg_loss = -torch.log(1 - neg_pred + 1e-15).mean()
    recon_loss_structure = pos_loss + neg_loss

    recon_loss_nodes = F.mse_loss(x_recon, data.x)
    total_recon_loss = recon_loss_edges + recon_loss_structure + recon_loss_nodes

    # The deterministic autoencoder (model_type="ae") has no KL term.
    if getattr(model, "variational", True):
        kl_loss = model.kl_loss()
    else:
        kl_loss = torch.zeros((), device=z.device)

    pred_loss = torch.tensor(0.0, device=z.device)
    if predictor == "Regressor_Head":
        target = (data.normalized_heatflux
                  if hasattr(data, "normalized_heatflux") else data.heat_flux)
        pred_loss = F.mse_loss(pred, target)
    elif predictor == "Classifier_Head":
        pred_loss = bce_with_class_weight(pred, data.fbos_sign, pos_weight)

    loss = betas[0] * total_recon_loss + betas[1] * kl_loss + betas[2] * pred_loss
    return loss, total_recon_loss, kl_loss, pred_loss
