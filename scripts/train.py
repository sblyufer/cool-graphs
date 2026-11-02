#!/usr/bin/env python
"""Train and evaluate one model: one configuration, one seed.

    python scripts/train.py --model-type vae --seed 0 --run-id gvae_s0

    # baselines (same hyperparameters, Appendix "Baseline Comparison")
    python scripts/train.py --model-type ae             --seed 0 --run-id ae_s0
    python scripts/train.py --model-type gat_classifier --seed 0 --run-id gat_s0
    python scripts/train.py --model-type gcn_classifier --seed 0 --run-id gcn_s0

    # ablations: override any hyperparameter
    python scripts/train.py --set beta_kl=0 --seed 0 --run-id abl_kl0_s0

Training is two-stage: --stage0-epochs of pure autoencoder pre-training
(betas 1, 0, 0; skipped for the classifier baselines), then the full
objective with early stopping on validation PR-AUC. The dataset is split
70/15/15, stratified by (topology, label) with --split-seed, so every model
and seed is scored on the same held-out sets.

Writes to <out-root>/<run-id>/:
    ckpt/best-pr-*.ckpt, best-roc-*.ckpt   checkpoints by each selection metric
    curves.json                            per-epoch training history
    {train,val,test,ext}.npz               latents z, labels y, scores, meta
    run.json                               configuration and all metrics
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def history_callback():
    """A Lightning callback recording every logged metric, per epoch."""
    from pytorch_lightning import Callback

    class History(Callback):
        def __init__(self):
            self.rows: list[dict] = []

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            row = {"epoch": int(trainer.current_epoch)}
            for k, v in trainer.callback_metrics.items():
                try:
                    row[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            self.rows.append(row)

    return History()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hparams", default="configs/gvae.json",
                   help="configs/gvae.json, or best_hparams.json from hp_search.py")
    p.add_argument("--block", default="auto",
                   help="which block of a best_hparams.json to use "
                        "(default: best_by_pr_auc)")
    p.add_argument("--set", action="append", default=[], metavar="K=V",
                   help="override one hyperparameter, e.g. --set beta_kl=0")
    p.add_argument("--dataset", default="data/natural.pkl")
    p.add_argument("--ext", default="data/extrapolation.pkl",
                   help="held-out topologies; skipped if the file is missing")
    p.add_argument("--model-type", default="vae",
                   choices=["vae", "ae", "gat_classifier", "gcn_classifier"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-root", default="results/final")
    p.add_argument("--pos-weight", default="off",
                   help="'off' (as in the paper), 'auto' (n_neg / n_pos of the "
                        "training split), or a number")
    p.add_argument("--max-epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--stage0-epochs", type=int, default=None,
                   help="default 30 for vae/ae, 0 for the classifier baselines")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-boot", type=int, default=2000,
                   help="bootstrap replicates for per-topology PR-AUC CIs")
    return p


def main() -> int:
    args = build_parser().parse_args()

    from coolgraphs.activelearning.loop import default_stage0, model_kwargs
    from coolgraphs.training.hparams import (apply_overrides, check_required,
                                             load_hparams)

    params, hp_meta = load_hparams(args.hparams, args.block)
    apply_overrides(params, args.set)
    check_required(params, args.hparams)

    discriminative = args.model_type in ("gat_classifier", "gcn_classifier")
    if args.stage0_epochs is None:
        args.stage0_epochs = default_stage0(args.model_type)

    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from torch_geometric.loader import DataLoader

    from coolgraphs.models.lightning import LightningGraphVAE
    from coolgraphs.training import data as data_mod
    from coolgraphs.training import evaluate as E
    from coolgraphs.training import ledger as L

    run_dir = os.path.join(args.out_root, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"=== {args.run_id} ===")
    print(f"model {args.model_type} | seed {args.seed} | "
          f"split_seed {args.split_seed} | stage0 {args.stage0_epochs} epochs")
    print(f"hparams {params}")
    if discriminative:
        print("(classifier baseline: the loss is the prediction term only, so "
              "the three betas have no effect)")

    ext = args.ext if args.ext and os.path.exists(args.ext) else None
    base = data_mod.prepare(args.dataset, batch_size=64,
                            split_seed=args.split_seed, ext_path=ext,
                            num_workers=args.num_workers, verbose=True)

    if args.pos_weight == "off":
        pos_weight = None
    elif args.pos_weight == "auto":
        pos_weight = float(base.pos_weight)
    else:
        pos_weight = float(args.pos_weight)

    seed_everything(args.seed, workers=True)

    bs = int(params["batch_size"])
    tr = DataLoader(base.train_loader.dataset, batch_size=bs, shuffle=True,
                    num_workers=args.num_workers)
    va = DataLoader(base.val_loader.dataset, batch_size=bs, shuffle=False,
                    num_workers=args.num_workers)

    common = model_kwargs(params, args.model_type, pos_weight)

    timer = L.Timer()
    with timer:
        # stage 0: autoencoder pre-training
        stage0_weights = None
        if args.stage0_epochs > 0:
            print(f"\n--- stage 0: autoencoder pre-training, "
                  f"{args.stage0_epochs} epochs ---")
            m0 = LightningGraphVAE(betas=(1.0, 0.0, 0.0),
                                   max_epoch=args.stage0_epochs, **common)
            Trainer(max_epochs=args.stage0_epochs, accelerator="auto", devices=1,
                    logger=False, enable_checkpointing=False,
                    enable_progress_bar=False).fit(m0, tr, va)
            stage0_weights = m0.model.state_dict()

        # stage 1: full objective
        print("\n--- stage 1: full objective ---")
        m1 = LightningGraphVAE(
            betas=(float(params["beta_recon"]), float(params["beta_kl"]),
                   float(params["beta_pred"])),
            max_epoch=args.max_epochs, **common)
        if stage0_weights is not None:
            m1.model.load_state_dict(stage0_weights)

        ckdir = os.path.join(run_dir, "ckpt")
        ck_pr = ModelCheckpoint(monitor="val_pr_auc", mode="max", save_top_k=1,
                                dirpath=ckdir, filename="best-pr-{epoch}")
        ck_roc = ModelCheckpoint(monitor="val_auc", mode="max", save_top_k=1,
                                 dirpath=ckdir, filename="best-roc-{epoch}")
        stop = EarlyStopping(monitor="val_pr_auc", mode="max",
                             patience=args.patience)
        hist = history_callback()
        trainer = Trainer(max_epochs=args.max_epochs, accelerator="auto",
                          devices=1, logger=False, enable_progress_bar=False,
                          callbacks=[ck_pr, ck_roc, stop, hist])
        trainer.fit(m1, tr, va)

    epochs_run = int(trainer.current_epoch)
    logged_pr = float(ck_pr.best_model_score) if ck_pr.best_model_score else None
    logged_roc = float(ck_roc.best_model_score) if ck_roc.best_model_score else None
    hit_max_epochs = epochs_run >= args.max_epochs - 1
    print(f"\ntrained {epochs_run} epochs in {timer.wall_seconds / 60:.1f} min")
    print(f"best val PR-AUC {logged_pr} | best val ROC-AUC {logged_roc}")
    if hit_max_epochs:
        print(f"NOTE: reached --max-epochs ({args.max_epochs}) before early "
              f"stopping; the run may not have converged.")

    with open(os.path.join(run_dir, "curves.json"), "w") as f:
        json.dump({"run_id": args.run_id, "epochs": hist.rows}, f, indent=2)

    # evaluate the best-PR-AUC checkpoint, not the last epoch
    best_ckpt = ck_pr.best_model_path
    if not best_ckpt or not os.path.exists(best_ckpt):
        raise SystemExit("no best-PR checkpoint was written; nothing to evaluate")
    print(f"\nevaluating {best_ckpt}")
    best = LightningGraphVAE.load_from_checkpoint(best_ckpt)
    arch = E.describe_architecture(best)

    rec = E.evaluate_checkpoint(
        best, base, out_root=args.out_root, run_id=args.run_id,
        config={"run_id": args.run_id, "dataset": args.dataset, "ext": ext,
                "model_type": args.model_type, "seed": args.seed,
                "split_seed": args.split_seed, "params": params,
                "hparams_source": hp_meta, "overrides": args.set,
                "architecture": arch, "pos_weight": pos_weight,
                "epochs_run": epochs_run, "max_epochs": args.max_epochs,
                "patience": args.patience, "stage0_epochs": args.stage0_epochs,
                "hit_max_epochs": hit_max_epochs,
                "checkpoint_pr": best_ckpt,
                "checkpoint_roc": ck_roc.best_model_path,
                "logged_val_pr_auc": logged_pr,
                "logged_val_roc_auc": logged_roc,
                "wall_seconds": timer.wall_seconds,
                "gpu_seconds": timer.gpu_seconds,
                "dataset_fingerprint": L.dataset_fingerprint(args.dataset),
                "environment": L.environment()},
        logged_val_pr_auc=logged_pr, num_workers=args.num_workers,
        n_boot=args.n_boot)

    L.append_run(L.ledger_path(args.out_root, "training_ledger.json"), {
        "kind": "train", "run_id": args.run_id,
        "dataset": args.dataset, "model_type": args.model_type,
        "seed": args.seed, "split_seed": args.split_seed,
        "pos_weight": pos_weight, "params": params,
        "epochs_run": epochs_run,
        "val_pr_auc": logged_pr, "val_roc_auc": logged_roc,
        **{f"{n}_pr_auc": rec["metrics"][n]["pooled"].get("pr_auc")
           for n in rec["metrics"] if n != "train"},
        **{f"{n}_roc_auc": rec["metrics"][n]["pooled"].get("roc_auc")
           for n in rec["metrics"] if n != "train"},
        "wall_seconds": timer.wall_seconds, "gpu_seconds": timer.gpu_seconds,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    if not rec["val_pr_auc_check"].get("agree", True):
        print("\nThe validation PR-AUC recomputed from the saved scores "
              "disagrees with the logged value; see run.json.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
