"""Lightning wrapper shared by all four models (vae, ae, gat_classifier,
gcn_classifier).

Logs ROC-AUC (``val_auc``) and PR-AUC (``val_pr_auc``) every epoch; either
can drive checkpointing, early stopping and Optuna pruning. ``predict_step``
returns the raw probability so the decision threshold can be chosen on
validation afterwards. The optimiser is AdamW with ReduceLROnPlateau on the
training loss (factor 0.5, patience 10).
"""
from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
from torch_geometric.nn import (global_add_pool, global_max_pool,
                                global_mean_pool)
from torchmetrics import MeanSquaredError
from torchmetrics.classification import (Accuracy, BinaryAUROC,
                                         BinaryAveragePrecision)

from .losses import bce_with_class_weight, loss_calc
from .networks import GNNClassifier, GraphVAE

CLASSIFIER_TYPES = ("discriminative", "gat_classifier", "gcn_classifier", "gnn")
# Only affects the logged accuracy; reported metrics use a threshold tuned on
# validation (see training/evaluate.py).
DEFAULT_CLASS_THRESHOLD = 0.35


def _make_scheduler(max_beta, start_epoch=0, end_epoch=50, k=1.0):
    mid = (start_epoch + end_epoch) / 2

    def schedule(epoch):
        if epoch < start_epoch:
            return 0.0
        if epoch > end_epoch:
            return max_beta
        progress = 1 / (1 + math.exp(-k * (epoch - mid)))
        start_val = 1 / (1 + math.exp(-k * (start_epoch - mid)))
        end_val = 1 / (1 + math.exp(-k * (end_epoch - mid)))
        return (progress - start_val) / (end_val - start_val) * max_beta

    return schedule


class LightningGraphVAE(pl.LightningModule):
    """model_type routing:
        "vae"                            -> GraphVAE, variational
        "ae"                             -> GraphVAE, deterministic (KL off)
        "gat_classifier"/"discriminative"-> GNNClassifier, GAT encoder
        "gcn_classifier"/"gnn"           -> GNNClassifier, GCN encoder
    """

    def __init__(self, in_channels, latent_dim, hidden_dim=32,
                 betas=(0.5, 0, 0), lr=1e-5, predictor=None,
                 pos_weight: float | None = None, annealing=False,
                 max_epoch=50, encoder_type="GAT", decoder_type="custom",
                 pool_type="mean", trial=None, model_type="vae",
                 class_threshold: float = DEFAULT_CLASS_THRESHOLD,
                 prune_monitor: str = "val_pr_auc"):
        super().__init__()
        self.trial = trial
        self.prune_monitor = prune_monitor
        self.save_hyperparameters(ignore=["trial"])

        self._discriminative = model_type in CLASSIFIER_TYPES
        if self._discriminative:
            enc = "GCN" if model_type in ("gcn_classifier", "gnn") else "GAT"
            self.model = GNNClassifier(
                in_channels=in_channels, out_channels=latent_dim,
                hidden_dim=hidden_dim, predictor=predictor, encoder_type=enc,
                decoder_type=decoder_type, pool_type=pool_type, num_node_types=2)
        else:
            self.model = GraphVAE(
                in_channels=in_channels, out_channels=latent_dim,
                hidden_dim=hidden_dim, predictor=predictor,
                encoder_type=encoder_type, decoder_type=decoder_type,
                pool_type=pool_type, num_node_types=2,
                variational=(model_type != "ae"))

        self.betas = betas
        self.lr = lr
        self._predictor = predictor
        self.pos_weight = pos_weight
        self.class_threshold = class_threshold
        self._annealing = annealing

        self.recon_beta = betas[0]
        self.kl_beta = 0.0 if annealing else betas[1]
        self.pred_beta = 0.0 if annealing else betas[2]
        self._annealing_kl = (_make_scheduler(betas[1], 0, max_epoch, 0.3)
                              if annealing and betas[1] else None)
        self._annealing_pred = (_make_scheduler(betas[2], 0, max_epoch, 0.3)
                                if annealing and betas[2] else None)

        if predictor == "Classifier_Head":
            self.train_acc = Accuracy(task="binary")
            self.val_acc = Accuracy(task="binary")
            self.train_auroc = BinaryAUROC()
            self.val_auroc = BinaryAUROC()
            # Average precision == area under the precision-recall curve.
            self.train_ap = BinaryAveragePrecision()
            self.val_ap = BinaryAveragePrecision()
        elif predictor == "Regressor_Head":
            self.train_acc = MeanSquaredError()
            self.val_acc = MeanSquaredError()
        else:
            self.train_acc = self.val_acc = None

    # ------------------------------------------------------------------ core
    def forward(self, data):
        return self.model(data)

    def encode(self, data):
        """Node-level latents for a batch."""
        out = self.model.encode(data.x, data.edge_index, data.edge_attr,
                                data.node_type, data.edge_type)
        # GraphVAE's encoder returns (mu, logstd); GNNClassifier returns z.
        return out[0] if isinstance(out, tuple) else out

    def pool(self, z, batch):
        p = getattr(self.model, "pool_type", "mean")
        if p == "attention":
            return self.model.att_pool(self.model.pool_norm(z), batch)
        if p == "sum":
            return global_add_pool(z, batch)
        if p == "max":
            return global_max_pool(z, batch)
        return global_mean_pool(z, batch)

    def enable_dropout(self):
        """Put only Dropout layers into train mode, for MC-dropout."""
        n = 0
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()
                n += 1
        return n

    def _compute_loss(self, batch):
        z, _, x_recon, edge_recon, pred = self(batch)
        if self._discriminative:
            zero = torch.tensor(0.0, device=self.device)
            pred_loss = bce_with_class_weight(pred, batch.fbos_sign, self.pos_weight)
            return z, pred, pred_loss, zero, zero, pred_loss
        loss, recon, kl, pred_loss = loss_calc(
            model=self.model, z=z, x_recon=x_recon, edge_recon=edge_recon,
            pred=pred, data=batch,
            betas=(self.recon_beta, self.kl_beta, self.pred_beta),
            predictor=self._predictor, pos_weight=self.pos_weight)
        return z, pred, loss, recon, kl, pred_loss

    def _log_classifier_metrics(self, stage, pred, batch):
        if self._predictor != "Classifier_Head" or pred is None:
            return
        y = batch.fbos_sign.int()
        bs = batch.num_graphs
        acc = (self.train_acc if stage == "train" else self.val_acc)
        auroc = (self.train_auroc if stage == "train" else self.val_auroc)
        ap = (self.train_ap if stage == "train" else self.val_ap)

        self.log(f"{stage}_acc", acc((pred > self.class_threshold).int(), y),
                 on_step=False, on_epoch=True, batch_size=bs)
        auroc(pred, y)
        self.log(f"{stage}_auc", auroc, on_step=False, on_epoch=True, batch_size=bs)
        ap(pred, y)
        self.log(f"{stage}_pr_auc", ap, on_step=False, on_epoch=True, batch_size=bs)

    # ----------------------------------------------------------------- steps
    def training_step(self, batch, batch_idx):
        if self._annealing:
            if self._annealing_kl:
                self.kl_beta = self._annealing_kl(self.current_epoch)
            if self._annealing_pred:
                self.pred_beta = self._annealing_pred(self.current_epoch)

        z, pred, loss, recon, kl, pred_loss = self._compute_loss(batch)
        bs = batch.num_graphs
        for name, val in (("train_loss", loss), ("recon_loss", recon),
                          ("kl_loss", kl), ("pred_loss", pred_loss)):
            self.log(name, val, on_step=False, on_epoch=True, batch_size=bs)
        self._log_classifier_metrics("train", pred, batch)

        if self._predictor == "Regressor_Head" and pred is not None:
            target = (batch.normalized_heatflux
                      if hasattr(batch, "normalized_heatflux") else batch.heat_flux)
            self.log("train_mse", self.train_acc(pred.squeeze(), target.squeeze()),
                     on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        z, pred, loss, recon, kl, pred_loss = self._compute_loss(batch)
        bs = batch.num_graphs
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_recon_loss", recon, on_step=False, on_epoch=True, batch_size=bs)
        self._log_classifier_metrics("val", pred, batch)

        if self._predictor == "Regressor_Head" and pred is not None:
            target = (batch.normalized_heatflux
                      if hasattr(batch, "normalized_heatflux") else batch.heat_flux)
            self.log("val_mse", self.val_acc(pred.squeeze(), target.squeeze()),
                     on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def predict_step(self, batch, batch_idx):
        """Returns (graph_z, y, hard_label, raw_probability).

        The raw probability is the fourth element and is what every downstream
        metric uses: thresholding belongs after the model, not inside it.
        """
        z, _, _, _, pred = self(batch)
        graph_z = self.pool(z, batch.batch)

        if self._predictor == "Classifier_Head":
            return (graph_z, batch.fbos_sign,
                    (pred > self.class_threshold).int(), pred)

        target = (batch.normalized_heatflux
                  if hasattr(batch, "normalized_heatflux") else batch.heat_flux)
        if self._predictor == "Regressor_Head":
            return graph_z, target, pred, pred
        return graph_z, target, None, None

    # ----------------------------------------------------------------- optim
    def configure_optimizers(self):
        if self._discriminative:
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        else:
            groups = [
                {"params": self.model.encoder.parameters(), "lr": self.lr},
                {"params": self.model.decoder.parameters(), "lr": self.lr},
                {"params": self.model.node_decoder.parameters(), "lr": self.lr},
                {"params": self.model.edge_value_regressor.parameters(), "lr": self.lr},
            ]
            if self.model._classifier:
                groups.append({"params": self.model._classifier.parameters(), "lr": self.lr})
            if self.model._regressor:
                groups.append({"params": self.model._regressor.parameters(), "lr": self.lr})
            if hasattr(self.model, "att_pool"):
                groups.append({"params": self.model.att_pool.parameters(), "lr": self.lr})
            if hasattr(self.model, "pool_norm"):
                groups.append({"params": self.model.pool_norm.parameters(), "lr": self.lr})
            opt = torch.optim.AdamW(groups)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=10),
                "monitor": "train_loss",
            },
        }

    def on_validation_epoch_end(self):
        """Report to Optuna and prune. Monitors whatever prune_monitor names,
        so a PR-AUC search prunes on PR-AUC rather than on ROC."""
        if self.trial is None:
            return
        val = self.trainer.callback_metrics.get(self.prune_monitor)
        if val is None:
            return
        import optuna
        self.trial.report(float(val.item()), step=self.current_epoch)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()
