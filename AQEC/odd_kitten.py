"""
The odd-parity binomial ('kitten') code of Shirol et al., PRX 16, 021042 (2026), Eq. 1:

    |0_L> = (|1> + |5>) / sqrt(2)       |1_L> = |3>                 (odd parity, nbar = 3)
    |0_E> = (|0> + sqrt(5)|4>) / sqrt(6) |1_E> = |2>                (even parity)

The error words are the normalized single-photon-loss images a|j_L>/||a|j_L>||. States
here are CAVITY-ONLY Fock vectors of length n_c; embedding into the three-mode
(storage x transmon x readout) space is left to whichever simulation needs it.

Unlike EST/kitten_code.py, a|0_L> and a|1_L> have equal norm sqrt(3) (Knill-Laflamme
for {I, a}), but the PReSPA recovery sum_n |2n+1><2n| does NOT invert a on the code:
it sends |0_E> to (|1> + sqrt(5)|5>)/sqrt(6), not |0_L>. See aqec_experiments.ipynb §4.
"""

import numpy as np

MIN_N_C = 7   # |0_L> occupies Fock 5; keep it interior so a' on it is not truncated

# Apple's Accelerate BLAS raises spurious divide/overflow RuntimeWarnings on ordinary
# complex matmuls (see core/grape_core.py's np.seterr); scoped here, not global.
_QUIET = dict(divide="ignore", over="ignore", invalid="ignore")


def _fock(n_c, k):
    v = np.zeros(n_c, dtype=complex)
    v[k] = 1.0
    return v


def annihilation(n_c):
    """(n_c, n_c) cavity lowering operator."""
    return np.diag(np.sqrt(np.arange(1, n_c)), 1).astype(complex)


def logical_basis(n_c):
    """(n_c, 2), columns [|0_L>, |1_L>]; asserts Knill-Laflamme for {I, a}."""
    if n_c < MIN_N_C:
        raise ValueError(f"n_c={n_c} < {MIN_N_C}: |0_L> needs Fock 5 in the interior.")
    ket0 = (_fock(n_c, 1) + _fock(n_c, 5)) / np.sqrt(2.0)
    ket1 = _fock(n_c, 3)
    B = np.stack([ket0, ket1], axis=1)
    _assert_knill_laflamme(B, annihilation(n_c))
    return B


def error_basis(n_c):
    """(n_c, 2), columns [|0_E>, |1_E>] = normalized a|j_L>."""
    B = logical_basis(n_c)
    with np.errstate(**_QUIET):
        img = annihilation(n_c) @ B
    return img / np.linalg.norm(img, axis=0)


def _assert_knill_laflamme(B, a):
    """P E_i' E_j P = c_ij P for E in {I, a}: the only nontrivial block is a'a."""
    with np.errstate(**_QUIET):
        M = B.conj().T @ (a.conj().T @ a) @ B      # 2x2, must be 3 * identity
        M_a = B.conj().T @ a @ B                   # a flips parity: must vanish
    if not np.allclose(M, 3.0 * np.eye(2), atol=1e-12):
        raise AssertionError(f"Knill-Laflamme violated for a'a on the code: {M}")
    if not np.allclose(M_a, 0.0, atol=1e-12):
        raise AssertionError(f"<i_L|a|j_L> != 0: {M_a}")
