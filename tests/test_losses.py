"""The deterministic autoencoder has no KL term; the GVAE does."""
import torch
from torch_geometric.data import Batch, Data

from coolgraphs.models.lightning import LightningGraphVAE


def _batch():
    g = Data(x=torch.randn(6, 4),
             edge_index=torch.tensor([[0, 1, 3, 4], [1, 2, 0, 1]]),
             edge_attr=torch.rand(4, 1), node_type=torch.tensor([0, 0, 0, 1, 1, 1]),
             edge_type=torch.tensor([0, 0, 1, 1]), fbos_sign=torch.tensor([1.0]),
             heat_flux=torch.tensor([0.5]))
    return Batch.from_data_list([g, g.clone()])


def _kl(model_type):
    torch.manual_seed(0)
    m = LightningGraphVAE(in_channels=4, latent_dim=8, hidden_dim=16,
                          betas=(1.0, 1.0, 1.0), predictor="Classifier_Head",
                          pool_type="attention", model_type=model_type)
    _, _, _, _, kl, _ = m._compute_loss(_batch())
    return float(kl)


def test_ae_has_no_kl():
    assert _kl("ae") == 0.0


def test_vae_has_kl():
    assert _kl("vae") > 0.0
