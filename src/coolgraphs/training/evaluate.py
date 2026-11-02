"""Turn a trained checkpoint into saved arrays and a metric block.

Predicts every split with unshuffled loaders, tunes the decision threshold on
validation, computes pooled and per-topology metrics (with bootstrap CIs on
the held-out sets), and checks that the validation PR-AUC recomputed from the
saved scores matches the value logged during training.
"""
from __future__ import annotations

import numpy as np

from ..utils.topology import topo_name, topology_of
from . import artifacts
from . import metrics as M

# torchmetrics (float32, during training) and sklearn (float64, here) agree to
# within this.
RECOMPUTE_TOLERANCE = 5e-3


def describe_architecture(model) -> dict:
    """Read architecture facts (dropout, GAT heads/layers, parameter count)
    off the instantiated model, for the run record."""
    out: dict = {}
    try:
        modules = list(model.modules())
    except Exception as e:
        return {"error": f"could not walk the module tree: {e!r}"}

    counts: dict[str, int] = {}
    dropout_p, heads, = [], []
    for m in modules:
        name = m.__class__.__name__
        counts[name] = counts.get(name, 0) + 1
        if "Dropout" in name:
            p = getattr(m, "p", None)
            if p is not None:
                dropout_p.append(float(p))
        if "GAT" in name:
            h = getattr(m, "heads", None)
            if h is not None:
                heads.append(int(h))

    out["module_counts"] = {k: v for k, v in sorted(counts.items())}
    out["dropout_p"] = sorted(set(dropout_p)) or "none found"
    out["gat_heads"] = sorted(set(heads)) or "none found"
    out["gat_layers"] = len(heads)
    try:
        out["n_parameters"] = sum(p.numel() for p in model.parameters())
        out["n_trainable"] = sum(p.numel() for p in model.parameters()
                                 if p.requires_grad)
    except Exception:
        pass
    for attr in ("pool_type", "encoder_type", "decoder_type", "variational"):
        v = getattr(getattr(model, "model", model), attr, None)
        if v is not None:
            out[attr] = v
    return out


def predict_splits(model, splits, which, *, batch_size=256, num_workers=2,
                   verbose=True) -> dict:
    """Run the model over each split and collect latents, labels and scores."""
    from pytorch_lightning import Trainer

    trainer = Trainer(accelerator="auto", devices=1, logger=False,
                      enable_progress_bar=False)
    model.eval()

    out: dict[str, dict] = {}
    for name in which:
        # predict_loader() never shuffles, so row i of z is entry i.
        try:
            loader = splits.predict_loader(name, batch_size, num_workers)
            entries = splits.entries(name)
            idx = splits.indices(name)
        except (KeyError, AttributeError):
            if verbose:
                print(f"  {name}: not present on this Splits, skipping")
            continue

        preds = trainer.predict(model, dataloaders=loader)
        if not preds:
            if verbose:
                print(f"  {name}: no predictions")
            continue

        z = np.vstack([p[0].cpu().numpy() for p in preds])
        y = np.hstack([p[1].cpu().numpy() for p in preds])
        score = np.hstack([p[3].cpu().numpy().ravel() for p in preds])

        if not (len(z) == len(y) == len(score) == len(entries)):
            raise SystemExit(
                f"{name}: {len(z)} latents, {len(y)} labels, {len(score)} "
                f"scores, {len(entries)} entries -- these must match. A "
                f"shuffled loader is the usual cause; predict_loader() guards "
                f"against it, so suspect the ext loader if that is this split.")

        out[name] = {"y": y, "score": score, "z": z, "entries": entries,
                     "idx": idx}
        if verbose:
            print(f"  {name:<5} {len(y):>6} graphs  {int(y.sum()):>5} pos  "
                  f"z{z.shape}")
    return out


def score_splits(out: dict, *, bootstrap_on=("test", "ext"),
                 n_boot=2000,
                 verbose=True) -> tuple[dict, float | None]:
    """Metrics for every split, on a threshold tuned once on validation."""
    thr = None
    if "val" in out:
        thr = M.best_threshold(out["val"]["y"], out["val"]["score"])
        if verbose:
            print(f"\nthreshold tuned on validation: {thr:.4f}")

    report = {}
    for name, d in out.items():
        topo = [topo_name(topology_of(e)) for e in d["entries"]]
        res = M.evaluate(d["y"], d["score"], topo, threshold=thr)
        report[name] = res
        if verbose:
            p = res["pooled"]
            print(f"\n--- {name} ---")
            print(f"  pooled   ROC {p['roc_auc']:.4f}  PR {p['pr_auc']:.4f}  "
                  f"lift {p['pr_auc_lift']:.2f}x  (base {p['base_rate']:.4f})")
            print(f"  per-topo mean ROC {res['mean_roc_auc']:.4f}  "
                  f"mean PR {res['mean_pr_auc']:.4f}  "
                  f"mean lift {res['mean_pr_auc_lift']:.2f}x")
            print(M.format_table(res["per_topology"]))

        # Rare topologies have few test positives; report intervals.
        if name in bootstrap_on:
            cis = M.bootstrap_per_topology(d["y"], d["score"], topo,
                                           "pr_auc", n_boot=n_boot)
            report[name]["bootstrap_pr_auc"] = cis
            if verbose:
                print("  PR-AUC with 95% bootstrap CI:")
                for t in sorted(cis):
                    c = cis[t]
                    flag = "  <- thin" if c["thin"] else ""
                    print(f"      {t:>5}  {M.format_ci(c)}  "
                          f"(n_pos {c['n_pos']}){flag}")
    return report, thr


def verify_logged(report: dict, logged_val_pr_auc: float | None,
                  tolerance: float = RECOMPUTE_TOLERANCE,
                  verbose: bool = True) -> dict:
    """Compare the logged val PR-AUC with the one recomputed from saved
    scores. Returns a block for the run record; never raises."""
    if logged_val_pr_auc is None or "val" not in report:
        return {"checked": False,
                "reason": "no logged value or no validation split"}

    recomputed = float(report["val"]["pooled"]["pr_auc"])
    delta = abs(recomputed - float(logged_val_pr_auc))
    ok = delta <= tolerance
    block = {"checked": True, "logged": float(logged_val_pr_auc),
             "recomputed": recomputed, "abs_delta": delta,
             "tolerance": tolerance, "agree": ok}
    if verbose:
        if ok:
            print(f"\nval PR-AUC check: logged {logged_val_pr_auc:.4f} vs "
                  f"recomputed {recomputed:.4f} (delta {delta:.5f}) -- agree")
        else:
            print(f"\n*** val PR-AUC MISMATCH: the training loop logged "
                  f"{logged_val_pr_auc:.4f}, recomputing from the saved "
                  f"scores gives {recomputed:.4f} (delta {delta:.5f}, "
                  f"tolerance {tolerance}).")
    return block


def evaluate_checkpoint(model, splits, *, out_root: str, run_id: str,
                        which=("train", "val", "test"), config: dict,
                        logged_val_pr_auc: float | None = None,
                        batch_size: int = 256, num_workers: int = 2,
                        n_boot: int = 2000, verbose: bool = True) -> dict:
    """Predict, score, verify and save. Returns the run record."""
    want = list(which)
    for name in getattr(splits, "eval_sets", {}):
        if name not in want:
            want.append(name)

    out = predict_splits(model, splits, want, batch_size=batch_size,
                         num_workers=num_workers, verbose=verbose)
    report, thr = score_splits(out, n_boot=n_boot, verbose=verbose)
    check = verify_logged(report, logged_val_pr_auc, verbose=verbose)

    d = artifacts.save_run(out_root, run_id, out,
                           config={**config, "threshold": thr,
                                   "val_pr_auc_check": check},
                           metrics=report)
    if verbose:
        print(f"\nwrote {d}")
    return {"dir": d, "metrics": report, "threshold": thr,
            "val_pr_auc_check": check}
