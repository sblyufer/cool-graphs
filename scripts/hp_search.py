#!/usr/bin/env python
"""Optuna (TPE) hyperparameter search for the GVAE.

    python scripts/hp_search.py --dataset data/natural.pkl --n-trials 40

Maximises validation PR-AUC (validation ROC-AUC is recorded for every trial
too). Each trial runs the same two-stage protocol as scripts/train.py, capped
at --max-epochs and --max-trial-hours, with median pruning on val PR-AUC.
Dropout (0.2), the GAT depth / heads and attention pooling are fixed. The
selected configuration is then reused unchanged for the AE and classifier
baselines.

The study lives in <out-dir>/study.db (SQLite), so several workers can share
it and an interrupted search resumes where it stopped (exit code 75 means
trials remain). Results go to <out-dir>/best_hparams.json, which
scripts/train.py accepts via --hparams.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np  # noqa: E402

EXIT_MORE_TRIALS = 75


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/natural.pkl")
    p.add_argument("--out-dir", default="results/hp")
    p.add_argument("--study-name", default="gvae_natural")
    p.add_argument("--n-trials", type=int, default=40,
                   help="TOTAL completed trials wanted across all jobs")
    p.add_argument("--timeout-hours", type=float, default=11.0,
                   help="stop and exit 75 after this long, for the wall limit")
    p.add_argument("--max-trial-hours", type=float, default=2.0,
                   help="hard cap on the length of one trial")
    p.add_argument("--space", choices=["full", "focused"], default="full",
                   help="'full' is the search reported in the paper; 'focused' "
                        "freezes the architecture and searches lr + betas")
    p.add_argument("--pos-weight-mode", choices=["off", "fixed", "search"],
                   default="off",
                   help="'search' also draws the positive-class weight over "
                        "[1, 45] (1 = unweighted)")
    p.add_argument("--pos-weight-value", type=float, default=22.6,
                   help="used when --pos-weight-mode fixed")
    p.add_argument("--model-type", default="vae")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-seed", type=int, default=0,
                   help="fixed during the search so trials differ only in HPs")
    p.add_argument("--sampler-seed", type=int, default=None,
                   help="TPE seed for this worker (default: derived from the "
                        "process id, so parallel workers differ)")
    p.add_argument("--stage0-epochs", type=int, default=30)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40,
                   help="early stopping on val_pr_auc")
    p.add_argument("--num-workers", type=int, default=4)
    return p


def sampler_seed(args) -> int:
    """A TPE seed unique to this worker (parallel workers must differ)."""
    if getattr(args, "sampler_seed", None) is not None:
        return int(args.sampler_seed)
    return (args.split_seed + os.getpid()) % (2 ** 31 - 1)


def suggest(trial, space: str = "full", pos_weight_mode: str = "off",
            pos_weight_value: float = 22.6) -> dict:
    """The search space.

    "full"    the seven-dimensional space of the paper's appendix table.
    "focused" architecture frozen (latent 32, hidden 128, batch 128); only lr
              and the three betas are searched.

    pos_weight_mode: "off" (unweighted, as in the paper), "fixed", or
    "search" (log-uniform over [1, 45]).
    """
    if space == "focused":
        hp = {
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            # single-choice categoricals so the values are recorded
            "hidden_dim": trial.suggest_categorical("hidden_dim", [128]),
            "latent_dim": trial.suggest_categorical("latent_dim", [32]),
            "batch_size": trial.suggest_categorical("batch_size", [128]),
            "beta_recon": trial.suggest_float("beta_recon", 1.0, 20.0, log=True),
            "beta_kl": trial.suggest_float("beta_kl", 1e-3, 5e-2, log=True),
            "beta_pred": trial.suggest_float("beta_pred", 20.0, 300.0, log=True),
        }
    else:
        hp = {
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "latent_dim": trial.suggest_categorical("latent_dim", [32, 64, 128]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "beta_recon": trial.suggest_float("beta_recon", 1.0, 200.0, log=True),
            "beta_kl": trial.suggest_float("beta_kl", 1e-4, 1e-1, log=True),
            "beta_pred": trial.suggest_float("beta_pred", 1.0, 300.0, log=True),
        }

    if pos_weight_mode == "search":
        hp["pos_weight"] = trial.suggest_float("pos_weight", 1.0, 45.0, log=True)
    elif pos_weight_mode == "fixed":
        hp["pos_weight"] = pos_weight_value
    return hp


def main() -> int:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from datetime import timedelta

    import optuna
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.callbacks import Timer as PLTimer

    from coolgraphs.models.lightning import LightningGraphVAE
    from coolgraphs.training import data as data_mod
    from coolgraphs.training import ledger as L

    t_start = time.time()
    deadline = t_start + args.timeout_hours * 3600

    # The split and the loaders do not depend on any hyperparameter except
    # batch_size, so the data is loaded once and only the loaders are rebuilt.
    print(f"=== HP search: {args.study_name} on {args.dataset} ===")
    base = data_mod.prepare(args.dataset, batch_size=64,
                            split_seed=args.split_seed,
                            num_workers=args.num_workers, verbose=True)
    y_val = np.asarray(base.labels("val"))
    if y_val.sum() == 0:
        raise SystemExit("validation split has no positives; cannot search")
    print(f"val: {len(y_val)} graphs, {int(y_val.sum())} positives "
          f"({100 * y_val.mean():.2f}%)")

    ledger_file = L.ledger_path(args.out_dir, "hp_search_ledger.json")
    dataset_fp = L.dataset_fingerprint(args.dataset)

    def objective(trial):
        if time.time() > deadline:
            raise optuna.exceptions.OptunaError("time budget reached")

        hp = suggest(trial, args.space, args.pos_weight_mode,
                     args.pos_weight_value)
        seed_everything(args.train_seed, workers=True)

        from torch_geometric.loader import DataLoader
        tr = DataLoader(base.train_loader.dataset, batch_size=hp["batch_size"],
                        shuffle=True, num_workers=args.num_workers)
        va = DataLoader(base.val_loader.dataset, batch_size=hp["batch_size"],
                        shuffle=False, num_workers=args.num_workers)

        common = dict(in_channels=4, latent_dim=hp["latent_dim"],
                      hidden_dim=hp["hidden_dim"], lr=hp["lr"],
                      predictor="Classifier_Head", pool_type="attention",
                      encoder_type="GAT", decoder_type="custom",
                      model_type=args.model_type,
                      # None means unweighted. Passed to BOTH stages so the
                      # pretrain and the scored stage see the same objective.
                      pos_weight=hp.get("pos_weight"))

        timer = L.Timer()
        with timer:
            # --- stage 0: pure autoencoder pretrain (betas = 1, 0, 0) -------
            stage0_weights = None
            if args.stage0_epochs > 0:
                m0 = LightningGraphVAE(betas=(1.0, 0.0, 0.0),
                                       max_epoch=args.stage0_epochs, **common)
                # at most a third of the per-trial budget
                s0_left = max(120.0, deadline - time.time())
                Trainer(max_epochs=args.stage0_epochs, accelerator="auto",
                        devices=1, logger=False, enable_checkpointing=False,
                        enable_progress_bar=False,
                        callbacks=[PLTimer(duration=timedelta(
                            seconds=min(args.max_trial_hours * 1200, s0_left)))]
                        ).fit(m0, tr, va)
                stage0_weights = m0.model.state_dict()

            # --- stage 1: full objective, pruned on val_pr_auc --------------
            m1 = LightningGraphVAE(
                betas=(hp["beta_recon"], hp["beta_kl"], hp["beta_pred"]),
                max_epoch=args.max_epochs, trial=trial,
                prune_monitor="val_pr_auc", **common)
            if stage0_weights is not None:
                m1.model.load_state_dict(stage0_weights)

            ck_pr = ModelCheckpoint(monitor="val_pr_auc", mode="max", save_top_k=1,
                                    dirpath=os.path.join(args.out_dir, "ckpt",
                                                         f"trial{trial.number}"),
                                    filename="best-pr-{epoch}")
            ck_roc = ModelCheckpoint(monitor="val_auc", mode="max", save_top_k=1,
                                     dirpath=os.path.join(args.out_dir, "ckpt",
                                                          f"trial{trial.number}"),
                                     filename="best-roc-{epoch}")
            stop = EarlyStopping(monitor="val_pr_auc", mode="max",
                                 patience=args.patience)
            # per-trial wall-clock cap (the best checkpoint is kept)
            left = max(300.0, deadline - time.time())
            trainer = Trainer(
                max_epochs=args.max_epochs, accelerator="auto", devices=1,
                logger=False, enable_progress_bar=False,
                callbacks=[ck_pr, ck_roc, stop,
                           PLTimer(duration=timedelta(
                               seconds=min(args.max_trial_hours * 3600, left)))])
            trainer.fit(m1, tr, va)

        best_pr = float(ck_pr.best_model_score) if ck_pr.best_model_score else 0.0
        best_roc = float(ck_roc.best_model_score) if ck_roc.best_model_score else 0.0

        trial.set_user_attr("val_roc_auc", best_roc)
        trial.set_user_attr("val_pr_auc", best_pr)
        trial.set_user_attr("epochs_run", int(trainer.current_epoch))
        trial.set_user_attr("gpu_seconds", timer.gpu_seconds)
        trial.set_user_attr("pos_weight", hp.get("pos_weight"))
        trial.set_user_attr("space", args.space)

        L.append_run(ledger_file, {
            "kind": "hp_trial", "study": args.study_name,
            "trial": trial.number, "params": hp,
            "model_type": args.model_type,
            "val_pr_auc": best_pr, "val_roc_auc": best_roc,
            "epochs_run": int(trainer.current_epoch),
            "wall_seconds": timer.wall_seconds,
            "gpu_seconds": timer.gpu_seconds,
            "split_seed": args.split_seed, "train_seed": args.train_seed,
            "space": args.space, "pos_weight": hp.get("pos_weight"),
            "dataset": dataset_fp,
        })
        print(f"[trial {trial.number}] PR-AUC {best_pr:.4f} | ROC {best_roc:.4f} "
              f"| {timer.wall_seconds / 60:.1f} min | {hp}")
        return best_pr

    # --- storage shared by every worker -----------------------------------
    # A generous lock timeout lets several workers share the SQLite file; the
    # heartbeat marks a trial FAILED if its worker dies.
    db = os.path.join(args.out_dir, "study.db")

    def open_storage(attempts: int = 8):
        """Attach to the study DB, retrying if several workers create it at
        the same moment."""
        import random as _r
        for i in range(attempts):
            try:
                return optuna.storages.RDBStorage(
                    url=f"sqlite:///{db}",
                    engine_kwargs={"connect_args": {"timeout": 60}},
                    heartbeat_interval=60, grace_period=300)
            except Exception as e:
                if i == attempts - 1:
                    raise
                wait = 0.5 * 2 ** i + _r.random()
                print(f"  storage busy ({type(e).__name__}); "
                      f"retry {i + 1}/{attempts - 1} in {wait:.1f}s")
                time.sleep(wait)

    storage = open_storage()
    print(f"TPE sampler seed for this worker: {sampler_seed(args)}"
          f"   (split_seed {args.split_seed})")

    study = optuna.create_study(
        study_name=args.study_name, storage=storage, direction="maximize",
        load_if_exists=True,
        # constant_liar keeps parallel workers from proposing the same point;
        # median pruning starts after 10 epochs.
        sampler=optuna.samplers.TPESampler(seed=sampler_seed(args),
                                           constant_liar=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                           n_warmup_steps=10))

    done = len([t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"{done}/{args.n_trials} trials complete "
          f"(budget {args.timeout_hours} h, worker sees the shared study)")

    # The trial target is shared across workers: stop when the study has it.
    def stop_when_target_reached(study_, _trial):
        n = len([t for t in study_.trials
                 if t.state == optuna.trial.TrialState.COMPLETE])
        if n >= args.n_trials:
            study_.stop()

    if done < args.n_trials:
        try:
            study.optimize(objective, n_trials=None,
                           timeout=max(1, deadline - time.time()),
                           callbacks=[stop_when_target_reached],
                           catch=(RuntimeError,))
        except optuna.exceptions.OptunaError as e:
            print(f"stopping: {e}")

    done = len([t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE])
    report(study, args, done)
    return 0 if done >= args.n_trials else EXIT_MORE_TRIALS


def report(study, args, done: int) -> None:
    import optuna
    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = len([t for t in study.trials
                  if t.state == optuna.trial.TrialState.PRUNED])
    print(f"\n=== {done} complete, {pruned} pruned ===")
    if not complete:
        return

    best_pr = max(complete, key=lambda t: t.user_attrs.get("val_pr_auc", -1))
    best_roc = max(complete, key=lambda t: t.user_attrs.get("val_roc_auc", -1))

    for tag, t in (("PR-AUC", best_pr), ("ROC-AUC", best_roc)):
        print(f"\nbest by {tag}: trial {t.number}  "
              f"PR={t.user_attrs.get('val_pr_auc', float('nan')):.4f}  "
              f"ROC={t.user_attrs.get('val_roc_auc', float('nan')):.4f}")
        for k, v in sorted(t.params.items()):
            print(f"    {k:12} {v}")

    same = best_pr.number == best_roc.number
    verdict = ("the same configuration." if same else
               "different configurations.")
    print(f"\nThe two selection metrics pick {verdict}")

    gpu_h = sum(t.user_attrs.get("gpu_seconds", 0) for t in study.trials) / 3600
    out = {
        "study": args.study_name,
        "dataset": args.dataset,
        "n_complete": done, "n_pruned": pruned,
        "gpu_hours": round(gpu_h, 2),
        "search_space": "see scripts/hp_search.py:suggest",
        "space": args.space,
        "pos_weight_mode": args.pos_weight_mode,
        "fixed": {"pool_type": "attention", "encoder_type": "GAT",
                  "decoder_type": "custom", "dropout": 0.2,
                  "stage0_epochs": args.stage0_epochs,
                  "max_epochs": args.max_epochs, "patience": args.patience,
                  "split_seed": args.split_seed},
        "best_by_pr_auc": {"trial": best_pr.number, "params": best_pr.params,
                           **{k: best_pr.user_attrs.get(k)
                              for k in ("val_pr_auc", "val_roc_auc", "epochs_run")}},
        "best_by_roc_auc": {"trial": best_roc.number, "params": best_roc.params,
                            **{k: best_roc.user_attrs.get(k)
                               for k in ("val_pr_auc", "val_roc_auc", "epochs_run")}},
        "selection_metrics_agree": same,
    }
    path = os.path.join(args.out_dir, "best_hparams.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}   ({gpu_h:.1f} GPU-hours spent so far)")


if __name__ == "__main__":
    sys.exit(main())
