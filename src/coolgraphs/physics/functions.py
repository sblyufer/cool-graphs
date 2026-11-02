"""Rate matrices for the Born-Markov master equation (Appendices A-C).

Single-particle Hamiltonian, electrode connectivity, many-body state energies,
and the fermionic / bosonic transition-rate matrices whose steady state gives
the bosonic energy flux. Units: energies in eV, rates in 1/s.
"""
import numpy as np
from itertools import product, combinations
from math import comb as comblib

# constants
k_B = 8.617333e-5  # eV / K
h = 4.13566751691 * (10 ** (-15))  # eV * sec
hbar = h / (2 * np.pi)


def fermi_func(epsilon, mu=0, T=298):
    return 1 / (np.exp((epsilon - mu) / (k_B * T)) + 1)


def bose_func(epsilon, T=298):
    return 1 / (np.exp(epsilon / (k_B * T)) - 1)


def J_J(epsilon):
    return 1


def J_nb(epsilon):
    return 1


def create_connectivity_vectors(ni, nj, lamE=0.5, etaR=1, etaL=1):
    d = np.zeros(ni * nj)
    for i in range(ni * nj):
        d[i] = i % nj
    connectivity_vectorL = np.exp(-lamE * 2 * d)
    connectivity_vectorR = np.exp(-lamE * 2 * np.flip(d))

    return etaR * connectivity_vectorR, etaL * connectivity_vectorL


def calc_Em(state, epsilons):
    return np.sum(state * epsilons)


def compute_Es(epsilons, n):
    # Generate all binary states of size n
    states = list(product([0, 1], repeat=n))
    # Calculate Em for each state
    Es = np.array([calc_Em(state, epsilons) for state in states])
    return Es


def calc_k_bos(epsilons, us, n, W, T):
    # Generate all binary vectors of size n
    states = list(product([0, 1], repeat=n))
    pair_count = comblib(n, 2)
    # Group states by Hamming weight (number of 1s)
    states_by_weight = {i: [] for i in range(1, n)}  # Hamming weights from 1 to n-1

    k_ab = np.zeros((pair_count, 1))  # Initialize k_vec
    k_em = np.zeros((pair_count, 1))  # Initialize k_vec
    pair_list = []
    index_list = []

    # Separate contributions for emission and absorption
    Gamma_nb_em_total = np.zeros((2 ** n, 2 ** n))
    Gamma_nb_ab_total = np.zeros((2 ** n, 2 ** n))

    # Group states by Hamming weight
    for state in states:
        weight = sum(state)
        if 1 <= weight < n:
            states_by_weight[weight].append(state)

    pair_counter = 0
    # Iterate over Hamming weights
    for weight in range(1, n):
        for state1, state2 in combinations(states_by_weight[weight], 2):  # Iterate over all pairs
            m = np.array(state1)
            m_tag = np.array(state2)

            # Find positions where the bit flips occur (0->1 in m, 1->0 in m_tag)
            flip_up = np.where((m == 0) & (m_tag == 1))[0]
            flip_down = np.where((m == 1) & (m_tag == 0))[0]

            # Check if the flips occur in the same position
            if len(flip_up) == 1 and len(flip_down) == 1:  # Ensure exactly one flip
                binary_m = int(''.join(map(str, m)), 2)
                binary_m_tag = int(''.join(map(str, m_tag)), 2)

                # Energy difference
                delta_Em = np.abs(calc_Em(m_tag, epsilons) - calc_Em(m, epsilons))

                # --- VECTORIZATION START ---
                idx_up = flip_up[0]
                idx_down = flip_down[0]
                # 1. Compute vector product of the two relevant eigenvectors (size n)
                vec = us[:, idx_up] * us[:, idx_down]

                # 2. Matrix-vector multiply to get all couplings at once (size n)
                #    This replaces the 'for i, W_row in enumerate(W)' loop
                coeffs = W @ vec

                # 3. Sum squared magnitudes
                total_coupling = np.sum(np.abs(coeffs) ** 2)
                # --- VECTORIZATION END ---

                # Calculate Bose factor once
                bose_val = bose_func(delta_Em, T=T)

                # Update total Gamma matrices
                Gamma_nb_em_total[binary_m, binary_m_tag] += J_nb(delta_Em) * (1 + bose_val) * total_coupling
                Gamma_nb_ab_total[binary_m_tag, binary_m] += J_nb(delta_Em) * bose_val * total_coupling

                if weight == (n - 1):
                    pair_list.append((flip_up[0], flip_down[0]))
                    index_list.append((binary_m, binary_m_tag))
                    k_ab[pair_counter, 0] += Gamma_nb_ab_total[binary_m_tag, binary_m] / hbar
                    k_em[pair_counter, 0] += Gamma_nb_em_total[binary_m, binary_m_tag] / hbar
                    pair_counter += 1

    # Final k_bos as the sum over all baths
    k_bos = Gamma_nb_em_total + Gamma_nb_ab_total
    k_bos[np.diag_indices_from(k_bos)] = -np.sum(k_bos, axis=0)
    k_bos = k_bos / hbar

    return k_bos, Gamma_nb_em_total, Gamma_nb_ab_total, states_by_weight, pair_list, index_list, k_ab, k_em


# create a distance matrix for all theoretically available sites
def create_R(ni, nj):
    n = ni * nj
    R = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            R[i][j] = np.sqrt((i % nj - j % nj) ** 2 + ((i // nj - j // nj) ** 2))  # value of j then value of i

    return R


# create a beta matrix for all theoretically available sites
def create_Heff(n, r, alphas, lamM=0.5, beta0=-0.1):
    Heff = [
        [alphas[i] if i == j else (beta0 * np.exp(-lamM * np.abs(r[i][j] - 1))) for j in range(n)]
        for
        i in range(n)]  # columns then rows, this is beta
    return np.matrix(Heff)


def calc_k_elec(n, epsilons, T, states, mu, orbital_contributionsL, orbital_contributionsR):
    k_fer_L = np.zeros((2 ** n, 2 ** n))
    k_fer_R = np.zeros((2 ** n, 2 ** n))

    Gamma_Je_L = np.zeros((2 ** n, 2 ** n))
    Gamma_Jh_L = np.zeros((2 ** n, 2 ** n))

    Gamma_Je_R = np.zeros((2 ** n, 2 ** n))
    Gamma_Jh_R = np.zeros((2 ** n, 2 ** n))

    for comb in states:
        state = list(comb)  # list flips the locations but it's fine with the way it converts to binary, checked
        for i in range(n):
            m = np.insert(state, i, 0)
            m_tag = np.insert(state, i, 1)

            # Binary indices
            binary_m = int(''.join(map(str, m)), 2)
            binary_m_tag = int(''.join(map(str, m_tag)), 2)

            # Energy difference
            delta_Em = calc_Em(m_tag, epsilons) - calc_Em(m, epsilons)

            # Update Gamma matrices for left (L)
            Gamma_Je_L[binary_m_tag, binary_m] = (
                    J_J(delta_Em) * fermi_func(delta_Em, mu=mu, T=T) * np.abs(orbital_contributionsL[i]) ** 2
            )
            Gamma_Jh_L[binary_m, binary_m_tag] = (
                    J_J(delta_Em) * (1 - fermi_func(delta_Em, mu=mu, T=T)) * np.abs(orbital_contributionsL[i]) ** 2
            )

            # Update Gamma matrices for right (R)
            Gamma_Je_R[binary_m_tag, binary_m] = (
                    J_J(delta_Em) * fermi_func(delta_Em, mu=0, T=T) * np.abs(orbital_contributionsR[i]) ** 2
            )
            Gamma_Jh_R[binary_m, binary_m_tag] = (
                    J_J(delta_Em) * (1 - fermi_func(delta_Em, mu=0, T=T)) * np.abs(orbital_contributionsR[i]) ** 2
            )

    # Calculate k_fer for left (L)
    k_fer_L = Gamma_Jh_L + Gamma_Je_L
    k_fer_L[np.diag_indices_from(k_fer_L)] = -np.sum(k_fer_L, axis=0)
    k_fer_L /= hbar

    # Calculate k_fer for right (R)
    k_fer_R = Gamma_Jh_R + Gamma_Je_R
    k_fer_R[np.diag_indices_from(k_fer_R)] = -np.sum(k_fer_R, axis=0)
    k_fer_R /= hbar

    return k_fer_L, k_fer_R


def generate_random_matrix(ni, nj, mode="identity"):
    size = ni * nj

    def normalize(mat):
        s = np.sum(mat)
        if s != 0:
            mat = mat / s
        return mat

    if mode == "identity":
        mat = np.eye(size)
        return normalize(mat)

    if mode == "dominantL":
        mat = np.zeros((size, size))
        for i in range(size):
            col = i % nj
            if col == 0:  # leftmost column
                mat[i, i] = 1
        return normalize(mat)

    if mode == "nondominantL":
        mat = np.zeros((size, size))
        for i in range(size):
            col = i % nj
            if col == 0:
                mat[i, i] = 1
                if i + 1 < size:
                    mat[i + 1, i + 1] = 1
        return normalize(mat)

    if mode == "interL":
        mat = np.zeros((size, size))
        for i in range(size):
            col = i % nj
            if col == 0:
                mat[i, i] = 1
                if i + 1 < size:
                    mat[i + 1, i + 1] = 1
                    mat[i + 1, i] = 1
                    mat[i, i + 1] = 1
        return normalize(mat)

    if mode == "dominantR":
        mat = np.zeros((size, size))
        for i in range(size):
            col = i % nj
            if col == nj - 1:  # rightmost column
                mat[i, i] = 1
        return normalize(mat)

    if mode == "nondominantR":
        mat = np.zeros((size, size))
        for i in range(size):
            col = i % nj
            if col == nj - 1:
                mat[i, i] = 1
                if i - 1 >= 0:
                    mat[i - 1, i - 1] = 1
        return normalize(mat)

    if mode == "interR":
        mat = np.zeros((size, size))
        for i in range(size):
            col = i % nj
            if col == nj - 1:
                mat[i, i] = 1
                if i - 1 >= 0:
                    mat[i - 1, i - 1] = 1
                    mat[i - 1, i] = 1
                    mat[i, i - 1] = 1
        return normalize(mat)

    if mode == "noisy_uniform":
        mat = 1 + 0.05 * np.random.randn(size, size)
        return normalize(mat)

    if mode == "uniform":
        mat = np.ones((size, size))
        return normalize(mat)

    if mode == "asymm_diag":
        mat = np.diag(np.linspace(1.0, 0.5, size))
        return normalize(mat)

    if mode == "random_diag":
        mat = np.diag(np.random.rand(size))
        return normalize(mat)
