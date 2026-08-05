"""
The binomial 'kitten' code and its Bloch-sphere cardinal points.

    |0_L> = (|0> + |4>) / sqrt(2)      |1_L> = |2>
    |0_E> = |3>                         |1_E> = |1>

(Roy, Wetherbee & Fatemi, arXiv:2603.15356, Sec. III.) The error words are the
single-photon-loss images of the code words. All states are embedded in the
joint transmon-cavity space at transmon level |g>, using the repo's index
convention |t,c> = t*n_c + c (core/grape_core.py:34).

Target states for a gate are built by applying the ideal 2x2 logical unitary to
each cardinal point's COEFFICIENT VECTOR and re-embedding -- the same
construction as core.cat_code.get_coherent_state_pairs, and reusing
core.propagator.PAULI_EIGENSTATE_COEFFS -- rather than by hand-permuting states.
Hand-permuting is where sign/phase conventions get silently lost.

On phases: the cost C1 is an incoherent mean of |<target_j|U|psi_j>|^2 over the
six columns, so each column is only pinned up to its own phase. Linearity of U
plus the presence of the +-X and +-Y columns then pins every RELATIVE phase
(if U|0_L> = e^{i a}|1_L> and U|1_L> = e^{i b}|0_L>, demanding |<+X|U|+X>| = 1
forces a = b), leaving only the physically irrelevant global phase free. This is
why no coherent-fidelity variant is needed here, unlike the 2-state training in
core.cat_code that motivated grape_core.coherent_fidelity_multi_state.
"""

import numpy as np

from core.grape_core import basis_state, make_ops
from core.propagator import PAULI_EIGENSTATE_COEFFS

# Fixed column order used by every (n, 6) array this module returns.
CARDINAL_ORDER = ["+Z", "-Z", "+X", "-X", "+Y", "-Y"]

# Ideal logical unitaries, in the {|0_L>, |1_L>} basis.
IDEAL_LOGICAL_U = {
    "X": np.array([[0.0, 1.0],
                   [1.0, 0.0]], dtype=complex),
    "H": np.array([[1.0, 1.0],
                   [1.0, -1.0]], dtype=complex) / np.sqrt(2.0),
    "T": np.array([[1.0, 0.0],
                   [0.0, np.exp(1j * np.pi / 4)]], dtype=complex),
}

# Which drives each gate is allowed to use (App. C): X and H drive cavity and
# qubit simultaneously; T is qubit-drive only. Columns are [C_I, C_Q, T_I, T_Q].
CONTROL_MASK = {
    "X": np.array([1.0, 1.0, 1.0, 1.0]),
    "H": np.array([1.0, 1.0, 1.0, 1.0]),
    "T": np.array([0.0, 0.0, 1.0, 1.0]),
}

MIN_N_C = 6   # |0_L> needs Fock 4; leave headroom so the code words are interior


def _require_n_c(n_c):
    if n_c < MIN_N_C:
        raise ValueError(
            f"n_c={n_c} is too small for the kitten code: |0_L> occupies Fock 4 "
            f"and |0_E> occupies Fock 3, so n_c >= {MIN_N_C} is required."
        )


def logical_basis(n_t, n_c):
    """
    (n, 2) orthonormal basis of the code space, columns [|0_L>, |1_L>] at |g>.

    Orthonormality is exact, not approximate: |0_L> is supported on Fock {0, 4}
    and |1_L> on Fock {2}, so the supports are disjoint. The assertion is a guard
    against a future change to the construction, following the same reasoning as
    core/propagator.py:123-128.
    """
    _require_n_c(n_c)
    ket0 = basis_state(n_t, n_c, 0, 0) + basis_state(n_t, n_c, 0, 4)
    ket0 /= np.linalg.norm(ket0)
    ket1 = basis_state(n_t, n_c, 0, 2)
    B = np.stack([ket0, ket1], axis=1)
    _assert_orthonormal(B, "code", n_c)
    return B


def error_basis(n_t, n_c):
    """
    (n, 2) orthonormal basis of the single-photon-loss error space, columns
    [|0_E>, |1_E>] = [|3>, |1>] at |g>.

    Column ORDER matters and is not arbitrary: it is chosen so that the cavity
    annihilation operator maps code column j to error column j, which is what
    makes `error_cardinals` below a simple re-embedding of the same coefficient
    vectors. See the assertion in `error_cardinals`.
    """
    _require_n_c(n_c)
    B = np.stack([basis_state(n_t, n_c, 0, 3),
                  basis_state(n_t, n_c, 0, 1)], axis=1)
    _assert_orthonormal(B, "error", n_c)
    return B


def _assert_orthonormal(B, label, n_c):
    err = np.linalg.norm(B.conj().T @ B - np.eye(2))
    assert err < 1e-12, (
        f"{label} basis is not orthonormal at n_c={n_c} "
        f"(||B^dag B - I|| = {err:.3e})"
    )


def _coeff_matrix():
    """(2, 6) matrix whose columns are the six cardinal coefficient vectors."""
    return np.stack([PAULI_EIGENSTATE_COEFFS[k] for k in CARDINAL_ORDER], axis=1)


def cardinals(n_t, n_c):
    """(n, 6) code-space cardinal points, columns in CARDINAL_ORDER."""
    return logical_basis(n_t, n_c) @ _coeff_matrix()


def error_cardinals(n_t, n_c):
    """
    (n, 6) error-space cardinal points: a|psi_C> renormalized, column by column.

    These coincide with the cardinal points built directly on {|0_E>, |1_E>}
    ONLY because a|0_L> and a|1_L> carry EQUAL weight for this code
    (a|0_L> = sqrt(2)|3>, a|1_L> = sqrt(2)|1>), so `a` restricted to the code
    space is sqrt(2) times an isometry onto the error space and does not distort
    superpositions. That is a property of the kitten code, not a general fact --
    for a code with unequal weights, a|+X_L> would NOT be the error-space |+X>,
    and this function would have to choose which of the two it means. The
    assertion below pins the coincidence so a future code change cannot quietly
    invalidate the reasoning.
    """
    A, _ = make_ops(n_t, n_c)
    raw = A @ cardinals(n_t, n_c)
    out = raw / np.linalg.norm(raw, axis=0, keepdims=True)

    direct = error_basis(n_t, n_c) @ _coeff_matrix()
    assert np.allclose(out, direct, atol=1e-12), (
        "a|psi_C> renormalized does not equal the error-space cardinal points. "
        "This holds for the kitten code because a|0_L> and a|1_L> have equal "
        "weight; if the code words changed, `error_cardinals` is ambiguous and "
        "the caller must decide which construction it wants."
    )
    return out


def gate_target(gate, n_t, n_c):
    """
    (n, 6) target states: the ideal logical unitary applied to each cardinal
    point, re-embedded in the joint space. Columns in CARDINAL_ORDER.
    """
    if gate not in IDEAL_LOGICAL_U:
        raise KeyError(f"unknown gate {gate!r}; have {sorted(IDEAL_LOGICAL_U)}")
    return logical_basis(n_t, n_c) @ (IDEAL_LOGICAL_U[gate] @ _coeff_matrix())


def control_mask(gate):
    """(4,) float mask over [C_I, C_Q, T_I, T_Q] for a named gate."""
    if gate not in CONTROL_MASK:
        raise KeyError(f"unknown gate {gate!r}; have {sorted(CONTROL_MASK)}")
    return CONTROL_MASK[gate].copy()
