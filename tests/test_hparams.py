"""Hyperparameter files: the shipped config and hp_search output both load."""
import json
import os

import pytest

from coolgraphs.training.hparams import (REQUIRED, apply_overrides,
                                         check_required, load_hparams)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = {"lr": 1e-3, "hidden_dim": 128, "latent_dim": 32, "batch_size": 128,
          "beta_recon": 2.7, "beta_kl": 0.006, "beta_pred": 66.0}


def test_shipped_config_is_complete():
    params, meta = load_hparams(os.path.join(ROOT, "configs", "gvae.json"))
    check_required(params, "configs/gvae.json")
    assert meta["hparams_block"] == "params"


def test_search_output_prefers_pr_block(tmp_path):
    p = tmp_path / "best.json"
    p.write_text(json.dumps({
        "best_by_pr_auc": {"trial": 3, "params": PARAMS},
        "best_by_roc_auc": {"trial": 5, "params": {**PARAMS, "lr": 2e-3}}}))
    params, meta = load_hparams(str(p))
    assert params["lr"] == 1e-3 and meta["source_trial"] == 3
    params, _ = load_hparams(str(p), block="best_by_roc_auc")
    assert params["lr"] == 2e-3


def test_overrides_are_typed():
    p = apply_overrides(dict(PARAMS), ["beta_kl=0", "latent_dim=64", "lr=5e-4"])
    assert p["beta_kl"] == 0 and p["latent_dim"] == 64 and p["lr"] == 5e-4


@pytest.mark.parametrize("missing", REQUIRED)
def test_missing_key_is_an_error(missing):
    with pytest.raises(SystemExit):
        check_required({k: v for k, v in PARAMS.items() if k != missing}, "x")
