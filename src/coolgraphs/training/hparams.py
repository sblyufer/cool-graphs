"""Loading hyperparameter files.

Two formats are accepted:

* a flat config such as ``configs/gvae.json``::

      {"params": {"lr": ..., "hidden_dim": ..., ...}}

* the ``best_hparams.json`` written by ``scripts/hp_search.py``, which holds
  ``best_by_pr_auc`` and ``best_by_roc_auc`` blocks (PR-AUC is used unless
  ``block`` says otherwise).
"""
from __future__ import annotations

import json

#: Keys every run needs. A missing key is an error rather than a silent
#: fallback to a constructor default.
REQUIRED = ("lr", "hidden_dim", "latent_dim", "batch_size",
            "beta_recon", "beta_kl", "beta_pred")

BLOCKS = ("params", "chosen", "best_by_pr_auc", "best_by_roc_auc")


def load_hparams(path: str, block: str = "auto") -> tuple[dict, dict]:
    """Return ``(params, meta)`` from a hyperparameter file."""
    with open(path) as f:
        doc = json.load(f)

    if block == "auto":
        if isinstance(doc.get("params"), dict):
            params, used = dict(doc["params"]), "params"
        else:
            for key in BLOCKS[1:]:
                b = doc.get(key)
                if isinstance(b, dict) and isinstance(b.get("params"), dict):
                    params, used = dict(b["params"]), key
                    break
            else:
                raise SystemExit(f"{path}: no hyperparameter block found "
                                 f"(looked for {list(BLOCKS)}).")
    else:
        b = doc.get(block)
        if block == "params" and isinstance(b, dict):
            params = dict(b)
        elif isinstance(b, dict) and isinstance(b.get("params"), dict):
            params = dict(b["params"])
        else:
            raise SystemExit(f"{path}: no block {block!r}; keys are {sorted(doc)}")
        used = block

    src = doc.get(used, {}) if used != "params" else doc
    meta = {"hparams_file": path, "hparams_block": used,
            "source_trial": src.get("trial") if isinstance(src, dict) else None}
    return params, meta


def apply_overrides(params: dict, overrides: list[str]) -> dict:
    """Apply ``K=V`` strings (e.g. from ``--set beta_kl=0``) to ``params``."""
    for kv in overrides or []:
        if "=" not in kv:
            raise SystemExit(f"--set needs K=V, got {kv!r}")
        k, v = kv.split("=", 1)
        try:
            params[k] = int(v) if v.lstrip("-").isdigit() else float(v)
        except ValueError:
            params[k] = v
    return params


def check_required(params: dict, path: str) -> None:
    missing = [k for k in REQUIRED if k not in params]
    if missing:
        raise SystemExit(f"{path} is missing {missing}; supply them with "
                         f"--set KEY=VALUE.")
