"""
Logical gates on the odd-parity n̄=3 kitten code of Shirol et al. (AQEC/odd_kitten.py):

    |0_L> = (|1> + |5>)/sqrt2,   |1_L> = |3>

embedded in the full storage x transmon x readout space with the transmon in |g> and
the readout in |0>.

PROCESS-FIDELITY OBJECTIVE. For a channel Phi on the full space and a target unitary U
on the code, with the operator basis E_ij = |i_L,g,0><j_L,g,0| and targets
T_ij = (U E_ij U') embedded the same way,

    F_pro = (1/4) sum_ij Tr[T_ij' Phi(E_ij)]
          = <<Phi_U^{-1} o P Phi P>>_entanglement fidelity on the code.

It is linear in Phi, so it is exact for any CPTP map (no pure-state assumption). The
targets carry |g,0>, so leakage out of the code, a transmon left in |e>/|f> or a
readout photon all count as error. For a unitary Phi(rho) = V rho V' it reduces to
|Tr(U' P V P)|^2 / 4.

Phi preserves Hermiticity, so Phi(E_10) = Phi(E_01)' and Tr[T_10' Phi(E_10)] =
conj(Tr[T_01' Phi(E_01)]). Only E_00, E_11, E_01 are propagated, with weights (1, 1, 2):

    F_pro = (1/4) Re{ t_00 + t_11 + 2 t_01 },  t_ij = Tr[T_ij' Phi(E_ij)].
"""

import numpy as np

from AQEC.odd_kitten import MIN_N_C, logical_basis
from DGRAPE.model import G, N_Q, dim

IDEAL_LOGICAL_U = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0),
    "T": np.diag([1.0, np.exp(1j * np.pi / 4)]).astype(complex),
}

# Index pairs (i, j) of the propagated E_ij and their weights in F_pro.
PROCESS_PAIRS = ((0, 0), (1, 1), (0, 1))
PROCESS_WEIGHTS = np.array([1.0, 1.0, 2.0])


def embedded_basis(n_c, n_r=2):
    """(d, 2), columns |0_L, g, 0>, |1_L, g, 0>."""
    B = logical_basis(n_c)                      # (n_c, 2), asserts Knill-Laflamme
    gq = np.zeros(N_Q)
    gq[G] = 1.0
    r0 = np.zeros(n_r)
    r0[0] = 1.0
    return np.stack([np.kron(np.kron(B[:, k], gq), r0) for k in range(2)], axis=1)


def process_targets(gate, n_c, n_r=2):
    """(E, T, w): E, T are (3, d, d) stacks of inputs E_ij and targets T_ij for the pairs
    in PROCESS_PAIRS, w the weights; F_pro = sum_b w_b Re Tr[T_b' Phi(E_b)] / 4."""
    if gate not in IDEAL_LOGICAL_U:
        raise KeyError(f"unknown gate {gate!r}; have {sorted(IDEAL_LOGICAL_U)}")
    if n_c < MIN_N_C:
        raise ValueError(f"n_c={n_c} < {MIN_N_C}")
    B = embedded_basis(n_c, n_r)
    BU = B @ IDEAL_LOGICAL_U[gate]               # column i = embedded U|i_L>
    E = np.stack([np.outer(B[:, i], B[:, j].conj()) for i, j in PROCESS_PAIRS])
    T = np.stack([np.outer(BU[:, i], BU[:, j].conj()) for i, j in PROCESS_PAIRS])
    assert E.shape[1] == dim(n_c, n_r)
    return E, T, PROCESS_WEIGHTS.copy()


def process_fidelity_from_outputs(rho_out, T, w):
    """F_pro from the propagated (3, d, d) stack."""
    t = np.einsum("bij,bij->b", T.conj(), rho_out)
    return float(np.sum(w * t.real) / 4.0)


def average_gate_fidelity(F_pro, d_code=2):
    """F_avg = (d F_pro + 1)/(d + 1). Valid for a trace-preserving channel on the code;
    with leakage it is the standard leakage-inclusive reading."""
    return (d_code * F_pro + 1.0) / (d_code + 1.0)
