"""Gaussian-process Bayesian optimisation baseline (GP-BO).

One independent GP per topology (a GP needs a fixed-dimensional input) over
the continuous sampled parameters, with a Matern-5/2 + white-noise kernel.
The objective is signed_log1p(F_E); a point counts as positive when
F_E > 1e5 eV/s, matching the dataset label. Each step scores a fresh pool of
candidates drawn from the same ``ParameterSampler`` the AL loop uses and
simulates the argmax of EI or UCB. The budget is counted in oracle calls and
split evenly across topologies. Driven by ``scripts/bo_baseline.py``.
"""
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Same values as coolgraphs.datasets.sampling (not imported, so this module
# loads without the torch stack).
POS_THRESHOLD = 1e5
SIZE_LIST = [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (3, 2), (3, 3), (2, 4)]


def signed_log1p(x):
    return float(np.sign(x) * np.log1p(np.abs(x)))


# ------------------------------------------------------------------ search space
def build_bounds(ni, nj, alpha_range):
    n = ni * nj
    b = [(150.0, 500.0),                                   # T
         (-0.15, -0.01),                                   # beta0
         (5.0, 1000.0),                                    # lamM
         (5.0, 1000.0),                                    # lamE
         (np.log10(0.01), np.log10(0.2)),                  # log10 etaL
         (np.log10(0.01), np.log10(0.2)),                  # log10 etaR
         (np.log10(0.001), np.log10(0.2))]                 # log10 W_scale
    b += [tuple(alpha_range)] * n                          # alpha_1..alpha_n
    return np.array(b, float)


# ------------------------------------------------------------------ oracle
def physics_oracle(p_dict):
    """(objective, is_positive, ok, graph) for one parameter dict."""
    from scipy.linalg import LinAlgError
    from ..datasets.sampling import SimulationRunner
    try:
        graph, f_bos = SimulationRunner().run(p_dict)
        return signed_log1p(f_bos), int(f_bos > POS_THRESHOLD), True, graph
    except (LinAlgError, ValueError, np.linalg.LinAlgError):
        return float("nan"), 0, False, None


# ------------------------------------------------------------------ search space
class SamplerSpace:
    """Candidates from ParameterSampler (the AL loop's distribution), with a
    fixed-length feature vector in ~[0, 1] for the GP.

    GP features (all ~[0,1]): [T, beta0, lamM, lamE, log10 etaL, log10 etaR, log10 W_scale]
    scaled by build_bounds, alpha_vec scaled by the alpha range, then a mode-free
    description of W (every W mode is normalised to sum 1 and scaled by W_scale):
    the diagonal fractions W_ii/sum(W) (how strongly each site couples to its own bath) and the
    off-diagonal fraction (inter-site bath coupling).  dim = 7 + 2n + 1.
    """

    def __init__(self, ni, nj, alpha_range):
        from ..datasets.sampling import ParameterSampler
        self.ps = ParameterSampler()
        self.ni, self.nj = ni, nj
        self.n = ni * nj
        b = build_bounds(ni, nj, alpha_range)[:7]
        self.lo7, self.span7 = b[:, 0], b[:, 1] - b[:, 0]
        self.a_lo = float(alpha_range[0])
        self.a_span = float(alpha_range[1] - alpha_range[0])

    def _one(self):
        ps, n = self.ps, self.n
        p = ps.sample_continuous()
        p["W"] = ps.sample_W(self.ni, self.nj)
        p["alpha_vec"] = ps.sample_alpha_base(n, p["beta0"])
        p["ni"], p["nj"] = self.ni, self.nj
        return p

    def features(self, p):
        W = np.asarray(p["W"], float)
        sw = float(W.sum())
        cont = np.array([p["T"], p["beta0"], p["lamM"], p["lamE"],
                         np.log10(p["etaL"]), np.log10(p["etaR"]), np.log10(sw)])
        f7 = (cont - self.lo7) / self.span7
        alpha = (np.asarray(p["alpha_vec"], float) - self.a_lo) / self.a_span
        diag = np.diag(W) / sw
        off = 1.0 - diag.sum()
        return np.concatenate([f7, alpha, diag, [off]])

    def draw(self, rng, m):  # rng unused: ParameterSampler uses the global numpy/random streams
        cands = [self._one() for _ in range(m)]
        return cands, np.vstack([self.features(c) for c in cands])

    def to_params(self, cand):
        return cand


# ------------------------------------------------------------------ acquisitions
def expected_improvement(mu, sigma, best, xi):
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - best - xi) / sigma
    return (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def upper_confidence_bound(mu, sigma, kappa):
    return mu + kappa * sigma


# ------------------------------------------------------------------ GP helper
def _make_gp(seed):
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(1e-3, (1e-6, 1e1)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=2, random_state=seed)


def _fit_subset(X, y, cap, rng):
    """Cap the GP training set: keep the incumbent, subsample the rest."""
    X, y = np.asarray(X), np.asarray(y)
    if len(X) <= cap:
        return X, y
    best = int(np.argmax(y))
    rest = np.delete(np.arange(len(X)), best)
    keep = np.concatenate([[best], rng.choice(rest, cap - 1, replace=False)])
    return X[keep], y[keep]


# ------------------------------------------------------------------ per-topology BO
def run_topology(ni, nj, budget, acq, oracle_fn, rng, alpha_range,
                 n_init, n_cand, kappa, xi, max_gp, counters, writer, seed,
                 refit_every=1, pool=None):
    sp = SamplerSpace(ni, nj, alpha_range)
    topo = f"{ni}x{nj}"

    F, Y = [], []   # GP features (normalised) and objectives

    def evaluate(raw, feat):
        obj, pos, ok, graph = oracle_fn(sp.to_params(raw))
        counters["calls"] += 1
        counters["pos"] += int(pos)
        if ok and graph is not None and pool is not None:
            pool.append({"graph": graph, "ni": ni, "nj": nj,
                         "is_positive": int(pos), "objective": obj})
        if ok and np.isfinite(obj):
            F.append(feat); Y.append(obj)
        else:  # penalize infeasible region so BO avoids it, still costs a call
            penalty = (min(Y) - 1.0) if Y else -10.0
            F.append(feat); Y.append(penalty)
        writer.writerow({
            "n_oracle_calls": counters["calls"],
            "cum_positives": counters["pos"],
            "positive_ratio": counters["pos"] / counters["calls"],
            "is_positive": int(pos),
            "objective": obj if np.isfinite(obj) else "",
            "topology": topo, "model_type": f"bo_{acq.lower()}",
            "acq": acq, "seed": seed, "space": "sampler",
        })

    n_init = min(n_init, budget)
    raws, feats = sp.draw(rng, n_init)
    for raw, feat in zip(raws, feats):
        evaluate(raw, feat)

    used = n_init
    kernel = None
    while used < budget:
        if kernel is None or (used - n_init) % refit_every == 0:
            gp = _make_gp(seed)          # full hyper-parameter optimisation
        else:                            # cheap: keep last kernel, only update the posterior
            gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
        Fs, ys = _fit_subset(F, Y, max_gp, rng)
        gp.fit(Fs, ys)
        kernel = gp.kernel_

        raws, feats = sp.draw(rng, n_cand)
        mu, sigma = gp.predict(feats, return_std=True)
        if acq == "EI":
            score = expected_improvement(mu, sigma, best=np.max(Y), xi=xi)
        else:  # UCB
            score = upper_confidence_bound(mu, sigma, kappa=kappa)
        k = int(np.argmax(score))
        evaluate(raws[k], feats[k])
        used += 1
