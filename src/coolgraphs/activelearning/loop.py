"""Building blocks of the active-learning loop (``scripts/al_loop.py``).

One cycle: retrain the surrogate from scratch on the current pool, draw fresh
candidates from the prior, score them by predictive variance (MC dropout or a
deep ensemble), label the top-k with the oracle, and add them to the pool.
"""
from __future__ import annotations

import os

import numpy as np

STAGE0_EPOCHS = 30
CLASSIFIER_MODELS = ("gat_classifier", "gcn_classifier")


def default_stage0(model_type: str) -> int:
    """Autoencoder pre-training epochs: 30 for vae/ae, 0 for the classifiers
    (which have no autoencoder to pre-train)."""
    return 0 if model_type in CLASSIFIER_MODELS else STAGE0_EPOCHS


def model_kwargs(params: dict, model_type: str, pos_weight=None) -> dict:
    """Constructor arguments shared by every LightningGraphVAE in the paper."""
    return dict(in_channels=4, latent_dim=int(params["latent_dim"]),
                hidden_dim=int(params["hidden_dim"]), lr=float(params["lr"]),
                predictor="Classifier_Head", pool_type="attention",
                encoder_type="GAT", decoder_type="custom",
                model_type=model_type, pos_weight=pos_weight)


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------

def sample_candidates(sampler, n, rng):
    """``n`` parameter dicts drawn i.i.d. from the prior.

    Topologies are drawn uniformly from ``sampler.SIZE_LIST``; the remaining
    parameters come from the same sampler that generated the training data.
    """
    out = []
    for _ in range(n):
        ni, nj = sampler.SIZE_LIST[rng.integers(len(sampler.SIZE_LIST))]
        p = sampler.sample_continuous()
        p["W"] = sampler.sample_W(ni, nj)
        p["alpha_vec"] = sampler.sample_alpha_base(ni * nj, p["beta0"])
        p["ni"], p["nj"] = int(ni), int(nj)
        out.append(p)
    return out


def jsonable(p):
    """A parameter dict that survives a JSON round-trip (arrays -> lists)."""
    out = {}
    for k, v in p.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _predict(model, loader, device, n):
    import torch
    out = np.empty(n, dtype=np.float64)
    at = 0
    with torch.no_grad():
        for batch in loader:
            p = model.predict_step(batch.to(device), 0)[3]
            p = p.detach().float().cpu().numpy().ravel()
            out[at:at + len(p)] = p
            at += len(p)
    if at != n:
        raise RuntimeError(f"scored {at} of {n} candidates")
    return out


def score_mc_dropout(model, graphs, stats, device, passes, batch_size,
                     normalize_graphs):
    """Variance and mean of the predicted probability over ``passes``
    stochastic forward passes with dropout active (everything else in eval
    mode)."""
    from torch_geometric.loader import DataLoader

    normed = normalize_graphs([g.clone() for g in graphs], stats,
                              method="zscore")
    loader = DataLoader(normed, batch_size=batch_size, shuffle=False)
    model.eval()
    model.enable_dropout()
    preds = np.stack([_predict(model, loader, device, len(normed))
                      for _ in range(passes)])
    return preds.var(axis=0), preds.mean(axis=0)


def score_deep_ensemble(models, graphs, stats, device, batch_size,
                        normalize_graphs):
    """Variance and mean of the predicted probability across ensemble members
    (one deterministic pass each, dropout off)."""
    from torch_geometric.loader import DataLoader

    normed = normalize_graphs([g.clone() for g in graphs], stats,
                              method="zscore")
    loader = DataLoader(normed, batch_size=batch_size, shuffle=False)
    preds = []
    for model in models:
        model.eval()
        preds.append(_predict(model, loader, device, len(normed)))
    preds = np.stack(preds)
    return preds.var(axis=0), preds.mean(axis=0)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train_on_pool(pool_entries, val_entries, params, model_type, seed,
                  max_epochs, patience, stage0_epochs, num_workers,
                  pos_weight=None, ckpt_dir=None):
    """Retrain from scratch on ``pool_entries``; returns (model, stats, info).

    Same two-stage protocol as ``scripts/train.py``: ``stage0_epochs`` of pure
    autoencoder pre-training (betas 1, 0, 0), then the full objective with
    early stopping on validation PR-AUC. The best-PR-AUC checkpoint (written
    to ``ckpt_dir``) is returned, not the last epoch.
    """
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from torch_geometric.loader import DataLoader

    from ..datasets.normalise import compute_stats, normalize_graphs
    from ..models.lightning import LightningGraphVAE
    from ..utils.io import graph_of

    seed_everything(seed, workers=True)

    stats = compute_stats([graph_of(e) for e in pool_entries])
    tr_g = normalize_graphs([graph_of(e) for e in pool_entries], stats,
                            method="zscore")
    va_g = normalize_graphs([graph_of(e) for e in val_entries], stats,
                            method="zscore")
    bs = int(params["batch_size"])
    tr = DataLoader(tr_g, batch_size=bs, shuffle=True, num_workers=num_workers)
    va = DataLoader(va_g, batch_size=bs, shuffle=False, num_workers=num_workers)

    common = model_kwargs(params, model_type, pos_weight)

    stage0_weights = None
    if stage0_epochs > 0:
        m0 = LightningGraphVAE(betas=(1.0, 0.0, 0.0),
                               max_epoch=stage0_epochs, **common)
        Trainer(max_epochs=stage0_epochs, accelerator="auto", devices=1,
                logger=False, enable_checkpointing=False,
                enable_progress_bar=False).fit(m0, tr, va)
        stage0_weights = m0.model.state_dict()

    m1 = LightningGraphVAE(
        betas=(float(params["beta_recon"]), float(params["beta_kl"]),
               float(params["beta_pred"])),
        max_epoch=max_epochs, **common)
    if stage0_weights is not None:
        m1.model.load_state_dict(stage0_weights)

    ck = ModelCheckpoint(monitor="val_pr_auc", mode="max", save_top_k=1,
                         dirpath=ckpt_dir, filename="best-pr-{epoch}")
    stop = EarlyStopping(monitor="val_pr_auc", mode="max", patience=patience)
    trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices=1,
                      logger=False, enable_progress_bar=False,
                      callbacks=[ck, stop])
    trainer.fit(m1, tr, va)

    epochs_run = int(trainer.current_epoch)
    info = {"epochs_run": epochs_run,
            "hit_max_epochs": epochs_run >= max_epochs - 1,
            "val_pr_auc_logged": (float(ck.best_model_score)
                                  if ck.best_model_score is not None else None),
            "best_ckpt": ck.best_model_path}

    if ck.best_model_path and os.path.exists(ck.best_model_path):
        best = LightningGraphVAE.load_from_checkpoint(ck.best_model_path)
    else:
        best = m1
    return best, stats, info


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate(model, entries_by_name, stats, device, batch_size):
    """PR-AUC and ROC-AUC (sklearn) on each named held-out set, dropout off."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch_geometric.loader import DataLoader

    from ..datasets.normalise import normalize_graphs
    from ..utils.io import graph_of, label_of

    out = {}
    model.eval()
    for name, entries in entries_by_name.items():
        if not entries:
            continue
        g = normalize_graphs([graph_of(e) for e in entries], stats,
                             method="zscore")
        loader = DataLoader(g, batch_size=batch_size, shuffle=False)
        s = _predict(model, loader, device, len(g))
        y = np.array([label_of(e) for e in entries], dtype=int)
        if y.sum() == 0 or y.sum() == len(y):
            out[name] = {"pr_auc": None, "roc_auc": None}
            continue
        out[name] = {"pr_auc": float(average_precision_score(y, s)),
                     "roc_auc": float(roc_auc_score(y, s))}
    return out


def heldout_sets(base) -> dict:
    """``{name: entries}`` for every held-out set on a ``Splits`` object."""
    sets = {"val": base.entries("val"), "test": base.entries("test")}
    if "ext" in base.eval_sets:
        sets["ext"] = base.entries("ext")
    return sets


def check_no_leakage(base) -> None:
    """Abort if any held-out graph also appears in the training split."""
    from ..utils.fingerprint import fingerprint

    train_fp = {fingerprint(e) for e in base.entries("train")}
    for name in base.names():
        if name == "train":
            continue
        shared = len(train_fp & {fingerprint(e) for e in base.entries(name)})
        print(f"  leakage check {name:<5} {shared} of "
              f"{len(base.entries(name))} also in train")
        if shared:
            raise SystemExit(f"{name} shares {shared} graphs with the "
                             f"training split.")
