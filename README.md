# Cool Graphs: Active Property Search Towards Quantum Nano-Refrigerators

Code for the paper *Cool Graphs: Active Property Search Towards Quantum
Nano-Refrigerators* (NeurIPS 2026).

Nanoscale junctions can absorb energy from their phonon bath, which lets them
act as quantum refrigerators. Configurations that do this are rare: under
random sampling only 1.2–9.2% of junctions have positive bosonic energy flux.
Each candidate costs a many-body master-equation solve that scales as
`(2^N)^3`. This repository contains:

1. **The physics oracle.** It solves the steady-state Born–Markov master
   equation for a 2D junction and returns the bosonic energy flux `F_E`.
2. **A Graph Variational Autoencoder (GVAE).** It maps each junction, encoded
   as an attributed graph, to a latent space and predicts whether `F_E > 1e5 eV/s`.
3. **An active-learning loop.** It uses MC-dropout (or deep-ensemble)
   uncertainty to choose which candidates to simulate. The repository also
   includes the random and Gaussian-process Bayesian optimisation (GP-BO)
   baselines.

## Installation

```bash
pip install -e ".[all]"        # core + optuna, matplotlib, pytest
pytest -q                      # ~1 min, no GPU or data needed
```

You need Python ≥ 3.10, PyTorch ≥ 2.0 and PyTorch Geometric ≥ 2.3. To get a
CUDA build, install PyTorch first by following the instructions at
[pytorch.org](https://pytorch.org). The paper used Python 3.12 on NVIDIA A100
GPUs. Everything also runs on CPU, but much more slowly.

## Repository layout

```
src/coolgraphs/
  physics/         rate matrices (functions.py), junction -> graph (graphs.py)
  datasets/        parameter prior + oracle (sampling.py), dataset generation,
                   normalisation, provenance ledger
  models/          GraphVAE / GNNClassifier (networks.py), loss, Lightning wrapper
  training/        stratified splits, data loaders, metrics, evaluation, saved arrays
  activelearning/  AL building blocks (loop.py), GP-BO baseline (bo.py)
  utils/           I/O, topology helpers, per-graph metadata, parameter fingerprint
  plotting/        figure style
scripts/           command-line entry points (below)
configs/gvae.json  the hyperparameters used in the paper
tests/
```

| Script | Purpose |
|---|---|
| `generate_dataset.py` | Sample parameters, run the oracle and write a labelled graph dataset |
| `hp_search.py` | Optuna hyperparameter search |
| `train.py` | Train and evaluate one model (GVAE, AE, GAT or GCN classifier) for one seed |
| `al_loop.py` | Run one active-learning trajectory (MC dropout, deep ensemble or random) |
| `bo_baseline.py` | Run one GP-BO trajectory (EI or UCB) |
| `retrain_on_pool.py` | Retrain the GVAE on the initial pool plus one trajectory's acquisitions (AL or GP-BO) |
| `summarise_al.py`, `summarise_bo.py`, `summarise_retrained.py` | Aggregate trajectories across seeds |
| `plot_latents.py` | Plot the 3D PCA latent-space figure |

Every script documents itself with `--help`. Run all of them from the
repository root.

## Data

The datasets are generated with the oracle; generation is deterministic given
the seed, resumable, and parallel (`--workers`, default: all cores but one).

```bash
# training set: 8 topologies x 5,000 = 40,000 graphs (4.25% positive)
python scripts/generate_dataset.py --per-topology 5000 --out data/natural.pkl

# extrapolation set, topologies unseen in training: 1x5, 4x2 (6,000 each), 3x4 (3,000)
python scripts/generate_dataset.py --topologies 1x5,4x2 --per-topology 6000 \
    --seed 54321 --out data/extrapolation.pkl
python scripts/generate_dataset.py --topologies 3x4 --per-topology 3000 \
    --seed 54322 --chunk 50 --out data/extrapolation.pkl
```

Cost is dominated by the largest systems (per oracle call: 3x3 ~0.65 s,
3x4 ~76 s), so the 3x4 extrapolation slice is by far the most expensive part.
A `<name>_provenance.json` ledger records seeds, oracle calls, CPU time and
per-topology hit rates.

Each dataset is a pickled list of `{"graph": torch_geometric.data.Data, "ni", "nj"}`.
Graphs carry node features `x` = `[alpha, T, log1p(eta_L), log1p(eta_R)]`,
`edge_attr`, `edge_type`, `node_type`, the label `fbos_sign`, `heat_flux`
(signed-log1p of `F_E`) and `meta`, the sampled parameters, so any graph can
be re-simulated.

## Reproducing the paper

All runs use `configs/gvae.json` and the stratified 70/15/15 split of
`natural.pkl` with `split_seed=42`. This gives 28,000 training, 6,000
validation and 6,000 interpolation-test graphs. Every model and every
active-learning arm is scored on the same held-out sets.

**Hyperparameter search (Appendix, Hyperparameter Optimization)**

```bash
python scripts/hp_search.py --dataset data/natural.pkl --n-trials 40 \
    --max-epochs 300 --max-trial-hours 2
```

Several workers can share `results/hp/study.db`. The configuration selected in
the paper is in `configs/gvae.json`. To train with your own search result
instead, pass `--hparams results/hp/best_hparams.json`.

**Predictive performance: GVAE and baselines (Table "Baseline Comparison")**

```bash
for s in 0 1 2 3 4; do
  for m in vae ae gat_classifier gcn_classifier; do
    python scripts/train.py --model-type $m --seed $s --run-id ${m}_s$s
  done
done
```

`ae` is the deterministic autoencoder: same encoder and decoders as the GVAE,
no sampling and no KL term. For the classifier baselines the three loss
weights have no effect and stage-0 pre-training is skipped.

Each run writes `results/final/<run-id>/run.json` with the following metrics
for `test` (interpolation) and `ext` (extrapolation):

- pooled and per-topology ROC-AUC and PR-AUC
- bootstrap confidence intervals
- latent vectors, scores and metadata in `<split>.npz`

**Ablations**

Any hyperparameter can be overridden with `--set`. Compare each ablation with
the baseline seed for seed.

```bash
python scripts/train.py --set beta_kl=0                    --seed $s --run-id abl_kl0_s$s
python scripts/train.py --set beta_kl=0.05901821425589314  --seed $s --run-id abl_kl10x_s$s
python scripts/train.py --set beta_recon=0.26739296064152267 --seed $s --run-id abl_recon_s$s
python scripts/train.py --set latent_dim=64  --seed $s --run-id abl_lat64_s$s
python scripts/train.py --set latent_dim=128 --seed $s --run-id abl_lat128_s$s
python scripts/train.py --set latent_dim=64 --model-type gat_classifier --seed $s --run-id abl_lat64_gat_s$s
```

**Latent space (Fig. 3)**

```bash
python scripts/plot_latents.py --run results/final/vae_s0 --split test --paper
```

**Active-learning search (Table "Search results")**

Each trajectory runs 32 cycles (16,000 oracle calls). In each cycle it
retrains the model from scratch on the current pool, samples 10,000
candidates and simulates 500 of them. Each strategy is run for three seeds.
GP-BO gets the same 16,000 calls, split evenly over the eight training
topologies (2,000 each).

```bash
for s in 0 1 2; do
  python scripts/al_loop.py --strategy mc_dropout    --seed $s
  python scripts/al_loop.py --strategy random        --seed $s
  python scripts/al_loop.py --strategy deep_ensemble --members 5 --seed $s
  python scripts/bo_baseline.py --acq EI  --seed $s
  python scripts/bo_baseline.py --acq UCB --seed $s
done
python scripts/summarise_al.py --plot      # positives found, calls/positive, per topology
python scripts/summarise_bo.py             # GP-BO, matched budget
```

The held-out rows (interpolation / extrapolation ROC-AUC) come from a GVAE
trained from scratch on the initial 28,000 graphs plus each trajectory's
16,000 acquisitions, i.e. a 44,000-graph pool for every strategy:

```bash
for s in 0 1 2; do
  for st in mc_dropout random deep_ensemble; do
    python scripts/retrain_on_pool.py --pool results/al/al_${st}_vae_s$s/pool.pkl
  done
  python scripts/retrain_on_pool.py --pool results/bo/bo_pool_ei_seed$s.pkl
  python scripts/retrain_on_pool.py --pool results/bo/bo_pool_ucb_seed$s.pkl
done
python scripts/summarise_retrained.py
```

By default each retrain uses the seed `al_loop.py` would give a model trained
on that final pool: `seed*1000 + 32` for the AL arms (`x10` for the deep
ensemble, i.e. ensemble member 0), and the BO seed for GP-BO. Pass `--seed`
to override.

- **Long runs.** A trajectory takes about 30 GPU-hours. If a run is
  interrupted, restart it with `--resume`.
- **Smoke test.** `--smoke` checks the whole loop in a few minutes.
- **Random-exploration ablation.** Add `--random-period 11` to the MC-dropout
  command.
- **Other surrogates.** Use `--model-type ae` or `--model-type gat_classifier`.
- **Outputs.** Each trajectory writes `cycles.csv`, which records cumulative
  oracle calls, positives and held-out metrics for every cycle. It also logs
  every acquired point with its parameters in `acquired/cycle_###.json`.

## Citation

```bibtex
@inproceedings{blyufer2026coolgraphs,
  title     = {Cool Graphs: Active Property Search Towards Quantum Nano-Refrigerators},
  author    = {Blyufer, Sofia and Kleiner, Amir and Peskin, Uri},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
