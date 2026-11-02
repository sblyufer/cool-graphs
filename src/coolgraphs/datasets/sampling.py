"""Parameter prior and the physics oracle.

``ParameterSampler`` draws junction parameters from the ranges in Table 1 of
the paper; ``SimulationRunner.run`` solves the steady-state master equation for
one parameter set and returns the attributed graph together with the bosonic
energy flux F_E (eV/s).

Both draw from the global ``numpy.random`` / ``random`` streams, so seed those
before sampling if you need reproducible draws.
"""
import random
import warnings
from itertools import product

import numpy as np
from scipy.linalg import LinAlgError, eigh

from ..physics.functions import (calc_k_bos, calc_k_elec, compute_Es,
                                 create_connectivity_vectors, create_Heff,
                                 create_R, generate_random_matrix)
from ..physics.graphs import create_graph

# The rate expressions overflow harmlessly (exp of large arguments) for some
# draws; the results are handled downstream, so keep the output readable.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Training topologies (ni, nj).
SIZE_LIST = [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (3, 2), (3, 3), (2, 4)]
# Chemical potential bias between the electrodes (eV): mu_L = MU, mu_R = 0.
MU = 4.0
# A configuration is labelled positive when F_E exceeds this value (eV/s).
POS_THRESHOLD = 1e5


class ParameterSampler:
    """Samples junction parameters from the prior used throughout the paper."""

    def __init__(self):
        self.T_range = (150, 500)
        self.beta0_range = (-0.15, -0.01)
        self.lamM_range = (5, 1000)
        self.lamE_range = (5, 1000)
        self.etaL_log_range = (np.log10(0.01), np.log10(0.2))
        self.etaR_log_range = (np.log10(0.01), np.log10(0.2))
        self.W_scale_log_range = (np.log10(0.001), np.log10(0.2))
        self.SIZE_LIST = SIZE_LIST

    def sample_continuous(self):
        return {
            "T": np.random.uniform(*self.T_range),
            "beta0": np.random.uniform(*self.beta0_range),
            "lamM": np.random.uniform(*self.lamM_range),
            "lamE": np.random.uniform(*self.lamE_range),
            "etaL": 10 ** np.random.uniform(*self.etaL_log_range),
            "etaR": 10 ** np.random.uniform(*self.etaR_log_range),
        }

    def sample_alpha_base(self, n, beta0):
        base = np.random.uniform(-0.1, 1.2)
        delta = np.random.uniform(-abs(beta0) * 2.0, abs(beta0) * 2.0)
        vec = np.array([base + i * delta for i in range(n)])
        np.random.shuffle(vec)
        return vec

    def sample_W(self, ni, nj):
        n = ni * nj

        weights_dict = {
            "identity": 0.15,
            "asymm_diag": 0.10,
            "random_diag": 0.10,
            "dominantL": 0.10,
            "nondominantL": 0.10,
            "interL": 0.10,
            "dominantR": 0.10,
            "nondominantR": 0.10,
            "interR": 0.10,
            "uniform": 0.05
        }

        modes = list(weights_dict.keys())
        weights = list(weights_dict.values())

        mode = random.choices(modes, weights=weights, k=1)[0]
        scale = 10 ** np.random.uniform(*self.W_scale_log_range)
        W = generate_random_matrix(ni, nj, mode)

        return scale * W


class SimulationRunner:
    """The oracle: one master-equation solve per call."""

    def run(self, p):
        """Return ``(graph, F_E)`` for a parameter dict from ``ParameterSampler``.

        Raises ``LinAlgError`` when the steady state cannot be found.
        """
        n = p['ni'] * p['nj']

        R = create_R(p['ni'], p['nj'])
        H = create_Heff(n, R, alphas=p['alpha_vec'], lamM=p['lamM'], beta0=p['beta0'])

        epsilons, us = eigh(H)
        epsilons = np.flip(epsilons)
        us = np.flip(us, axis=1)

        states = list(product([0, 1], repeat=(n - 1)))
        Es = compute_Es(epsilons, n)

        etas = create_connectivity_vectors(p['ni'], p['nj'], lamE=p['lamE'], etaR=p['etaR'], etaL=p['etaL'])
        orbital_contributionsR = (etas[0] @ us)
        orbital_contributionsL = (etas[1] @ us)

        k_bos, _, _, _, _, _, _, _ = calc_k_bos(epsilons, us, n, p['W'], p['T'])
        k_fer_L, k_fer_R = calc_k_elec(n, epsilons, p['T'], states, MU, orbital_contributionsL,
                                       orbital_contributionsR)

        K_total = k_fer_R + k_fer_L + k_bos

        u, s, vh = np.linalg.svd(K_total)
        null_space_vec = vh[-1, :].T

        if np.sum(null_space_vec) == 0:
            raise LinAlgError("Null space calculation failed (zero sum in null space).")

        P = null_space_vec
        P /= np.sum(P)
        if np.any(P < 0): P = np.abs(P) / np.sum(np.abs(P))

        F_bos_val = (Es @ k_bos @ P)

        graph = create_graph(
            p['ni'], p['nj'], p['alpha_vec'], p['W'],
            lamE=p['lamE'], lamM=p['lamM'], beta0=p['beta0'],
            etaR=p['etaR'], etaL=p['etaL'], T=p['T'],
            Fbos_sign=(F_bos_val > POS_THRESHOLD),
            heat_flux=float(F_bos_val)
        )

        return graph, float(F_bos_val)
