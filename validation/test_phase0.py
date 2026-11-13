#!/usr/bin/env python3
"""
test_phase0.py

Acceptance checks for the Phase 0 measuring stick (`core/propagator.py`) and for
the Pedersen correction to `average_gate_fidelity`.

The point of this file is that each check can actually FAIL for a specific,
named reason. Two checks that look plausible were deliberately NOT written the
obvious way, because the obvious way has no teeth:

  * Unitarity of the accumulated propagator does not test the accumulation
    ORDER --- a product of unitaries taken in the wrong order is still unitary
    (||U^dag U - I|| ~ 6e-13 either way). So the order is pinned by comparing
    against the independent state-vector path `propagate_pulse`
    (test_b0_matches_independent_state_propagation).

  * "mean state-transfer fidelity over the 6 Pauli eigenstates == Pedersen
    fidelity" is a TAUTOLOGY when both sides are computed from the same 2x2
    block: <B(U_id c)|U|Bc> = c^dag U_id^dag (B^dag U B) c, so both sides are
    functions of B^dag U B alone and degrade in lockstep. Measured with the
    propagator order reversed: mean6 = 0.162486, Pedersen = 0.162486, agreeing
    to 3e-17 --- it passes on a wrong propagator. It also cannot pin the
    index convention, since Pedersen is transpose-invariant for the entire
    current gate set (Tr(U_id^dag U_log^T) = Tr(conj(U_id) U_log), and I,X,Z,H,T
    are symmetric while Y^T = -Y). So that identity is kept only as the pure
    2-design math it really is (test_b2_...), and the convention is pinned
    separately with a deliberately non-symmetric target (test_b1_...).

Run:
    python3 validation/test_phase0.py [-v]
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.grape_core import make_hamiltonian, make_ops
from core.propagator import (
    PAULI_EIGENSTATE_COEFFS,
    full_propagator,
    logical_basis,
    logical_block,
    pedersen_gate_fidelity,
    leakage_L1,
    error_basis,
    knill_laflamme_residual,
)
from validation.validate_logical_gates import (
    IDEAL_LOGICAL_U,
    average_gate_fidelity,
    propagate_pulse,
    ALPHA,
    DT,
    N_T,
)

PULSE_DIR = os.path.join(REPO_ROOT, "pulses")


def _old_plus_d_fidelity(U_ideal, U_actual):
    """The pre-Phase-0 formula, kept here so the correction can be quantified."""
    d = U_ideal.shape[0]
    overlap = np.trace(U_ideal.conj().T @ U_actual)
    return (np.abs(overlap) ** 2 + d) / (d * (d + 1))


def _load(gate):
    u = np.load(os.path.join(PULSE_DIR, f"u_{gate}_main.npy"))
    assert u.shape == (550, 4), f"expected a 550x4 main pulse for {gate}, got {u.shape}"
    return u


class PropagatorCorrectnessTest(unittest.TestCase):
    """Checks (a) and (b0): the propagator itself."""

    @classmethod
    def setUpClass(cls):
        cls.n_c = 24
        cls.u = _load("X")
        cls.H0, cls.Hc = make_hamiltonian(N_T, cls.n_c)
        cls.U = full_propagator(cls.u, cls.H0, cls.Hc, DT)
        cls.B = logical_basis(N_T, cls.n_c, alpha=ALPHA)

    def test_a_unitarity(self):
        D = self.U.shape[0]
        self.assertEqual(D, N_T * self.n_c)
        err = np.linalg.norm(self.U.conj().T @ self.U - np.eye(D))
        self.assertLess(err, 1e-10, f"||U^dag U - I|| = {err:.3e} (n_c={self.n_c}, dt={DT})")

    def test_b0_matches_independent_state_propagation(self):
        """
        THE correctness gate. `propagate_pulse` never forms a D×D matrix, so
        agreement pins the accumulation order (U = Uk @ U, later steps on the
        left). Reversing it gives a residual of ~1.25 here, not ~1e-15.
        If this fails, stop --- nothing downstream is trustworthy.
        """
        for label, c in PAULI_EIGENSTATE_COEFFS.items():
            psi_in = self.B @ c
            residual = np.linalg.norm(
                self.U @ psi_in - propagate_pulse(self.u, self.H0, self.Hc, psi_in, DT)
            )
            self.assertLess(
                residual, 1e-12,
                f"propagator disagrees with propagate_pulse on |{label}_L>: "
                f"residual {residual:.3e} -- check the accumulation order"
            )


class LogicalBlockConventionTest(unittest.TestCase):
    """Check (b1): logical_block is column-per-input, matching IDEAL_LOGICAL_U."""

    def test_b1_column_per_input_convention(self):
        n_c = 24
        B = logical_basis(N_T, n_c, alpha=ALPHA)
        D = N_T * n_c

        # Deliberately NON-symmetric and non-Hermitian, so that V and V^T are
        # distinguishable by the fidelity too: pedersen(V, V) = 1 but
        # pedersen(V, V^T) = 0.5.
        V = np.array([[1.0, 1.0], [1.0j, -1.0j]], dtype=complex) / np.sqrt(2.0)
        self.assertAlmostEqual(pedersen_gate_fidelity(V, V), 1.0, places=12)
        self.assertAlmostEqual(pedersen_gate_fidelity(V, V.T), 0.5, places=12)

        # A synthetic full propagator that acts as exactly V on the logical
        # subspace and as the identity everywhere else.
        P = B @ B.conj().T
        U_synth = B @ V @ B.conj().T + (np.eye(D) - P)

        extracted = logical_block(U_synth, B)
        np.testing.assert_allclose(extracted, V, atol=1e-12, err_msg=
            "logical_block returned the transpose of the intended block: it must "
            "be column-per-input ([i,j] = <b_i|U|b_j>), which is the convention "
            "IDEAL_LOGICAL_U is written in")
        self.assertAlmostEqual(pedersen_gate_fidelity(V, extracted), 1.0, places=12)
        self.assertAlmostEqual(leakage_L1(extracted), 0.0, places=12)

    def test_b1b_ideal_targets_are_column_per_input(self):
        """
        Cross-check the claim that IDEAL_LOGICAL_U uses column-per-input, by
        rebuilding each ideal from the trained state pairs in core.cat_code.
        Y is the discriminating case (Y^T = -Y).
        """
        from core.cat_code import (
            get_logical_X_state_pairs, get_logical_Y_state_pairs,
            get_logical_Z_state_pairs, get_logical_H_state_pairs,
            get_logical_T_state_pairs, get_identity_state_pairs,
        )
        factories = {
            "I": get_identity_state_pairs, "X": get_logical_X_state_pairs,
            "Y": get_logical_Y_state_pairs, "Z": get_logical_Z_state_pairs,
            "H": get_logical_H_state_pairs, "T": get_logical_T_state_pairs,
        }
        n_c = 24
        B = logical_basis(N_T, n_c, alpha=ALPHA)
        for gate, factory in factories.items():
            pairs = factory(n_c=n_c, n_t=N_T)
            # Column j = the trained target for input basis state j, in logical coords.
            U_trained = np.stack([B.conj().T @ psi_f for _, psi_f in pairs], axis=1)
            np.testing.assert_allclose(
                U_trained, IDEAL_LOGICAL_U[gate], atol=1e-12,
                err_msg=f"IDEAL_LOGICAL_U['{gate}'] does not match the trained targets "
                        "under the column-per-input convention")


class PedersenFormulaTest(unittest.TestCase):
    """Checks (b2) and (c): the fidelity formula itself, no propagator involved."""

    def test_b2_two_design_identity_on_non_unitary_blocks(self):
        """
        The 6 Pauli eigenstates are a spherical 2-design, so the plain mean of
        |<psi|M|psi>|^2 over them equals the Haar average, which is the Pedersen
        expression. Holds for ARBITRARY M, unitary or not -- which is exactly
        why this identity cannot validate a propagator, only the formula.
        """
        rng = np.random.default_rng(20260729)
        coeffs = list(PAULI_EIGENSTATE_COEFFS.values())
        for trial in range(20):
            U_ideal = np.eye(2, dtype=complex)
            # A generic non-unitary 2x2 (contractive, like a leaky logical block)
            M = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            M *= 0.6 / np.linalg.norm(M)
            mean6 = np.mean([np.abs(c.conj() @ (M @ c)) ** 2 for c in coeffs])
            self.assertAlmostEqual(
                mean6, pedersen_gate_fidelity(U_ideal, M), places=12,
                msg=f"2-design identity failed on trial {trial}")

    def test_c_reduces_to_old_formula_when_unitary(self):
        """The '+d' form was right for unitary blocks; the fix must not move those."""
        rng = np.random.default_rng(7)
        for _ in range(10):
            G = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            Q, R = np.linalg.qr(G)
            U = Q * (np.diag(R) / np.abs(np.diag(R)))  # Haar-ish unitary 2x2
            self.assertAlmostEqual(average_gate_fidelity(U, U), 1.0, places=12)
            self.assertAlmostEqual(leakage_L1(U), 0.0, places=12)
            U_ideal = IDEAL_LOGICAL_U["H"]
            self.assertAlmostEqual(
                average_gate_fidelity(U_ideal, U),
                _old_plus_d_fidelity(U_ideal, U), places=12,
                msg="fixed and old formulas must agree when the block is unitary")

    def test_c2_fix_landed_and_delta_is_exactly_L1_over_3(self):
        """
        Pins the actual correction: for d=2, F_old - F_pedersen == L1/3 exactly,
        and validation.average_gate_fidelity must now return the Pedersen value.
        Fails if `average_gate_fidelity` still carries the '+ d' term.
        """
        rng = np.random.default_rng(11)
        for gate, U_ideal in IDEAL_LOGICAL_U.items():
            # A leaky block: shrink an ideal gate and perturb it.
            M = 0.97 * U_ideal + 0.02 * (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
            F_ped = pedersen_gate_fidelity(U_ideal, M)
            self.assertAlmostEqual(
                average_gate_fidelity(U_ideal, M), F_ped, places=12,
                msg=f"average_gate_fidelity is not the Pedersen form for {gate}")
            self.assertAlmostEqual(
                _old_plus_d_fidelity(U_ideal, M) - F_ped,
                leakage_L1(M) / 3.0, places=12,
                msg=f"F_old - F_pedersen != L1/3 for {gate}")


class LogicalBasisTest(unittest.TestCase):
    """Check (d): the reference basis is orthonormal at every truncation used."""

    def test_d_orthonormal_across_truncations(self):
        for n_c in (18, 22, 24, 26, 32):
            B = logical_basis(N_T, n_c, alpha=ALPHA)
            err = np.linalg.norm(B.conj().T @ B - np.eye(2))
            self.assertLess(err, 1e-12, f"||B^dag B - I|| = {err:.3e} at n_c={n_c}")


class ErrorSubspaceTest(unittest.TestCase):
    """
    Checks (e): the single-photon-loss error subspace E = span{a|+Z_L>, a|-Z_L>}.

    Same discipline as the rest of this file — none of these are written the
    obvious way, because the obvious way has no teeth:

      * "error_basis returns an orthonormal basis" cannot fail as a test of
        error_basis, since error_basis asserts it internally. So (e0) rebuilds
        a|j_L> from `make_ops` directly and checks the CLAIM, then checks that
        error_basis agrees with the independent construction.

      * "a @ B[:, j] normalized equals error_basis column j" is a restatement of
        the implementation (`raw = A @ B; E = raw / nrm`) and would pass a
        column swap only if the swap were also made in the test. So (e1) pins
        the column pairing to the physical fact underneath it instead: the mod-4
        Fock support of each column.
    """

    N_C_SCAN = (18, 22, 24, 26, 32)

    def test_e0_error_subspace_is_exactly_orthonormal_and_outside_the_code(self):
        """
        E is orthonormal and orthogonal to C *exactly*, at every truncation.

        Built here from make_ops rather than taken from error_basis, so this is
        a test of the physics (disjoint mod-4 supports, even/odd parity) and not
        a restatement of error_basis's own assertions. It has real teeth: for a
        code whose words did not have disjoint supports one rung down, or for
        any code with odd-parity content, both norms below would be nonzero.
        """
        for n_c in self.N_C_SCAN:
            B = logical_basis(N_T, n_c, alpha=ALPHA)
            A, _ = make_ops(N_T, n_c)
            raw = A @ B
            E_indep = raw / np.linalg.norm(raw, axis=0)

            orth = np.linalg.norm(E_indep.conj().T @ E_indep - np.eye(2))
            self.assertLess(orth, 1e-12,
                            f"||E^dag E - I|| = {orth:.3e} at n_c={n_c}")
            cross = np.linalg.norm(B.conj().T @ E_indep)
            self.assertLess(cross, 1e-12,
                            f"||B^dag E|| = {cross:.3e} at n_c={n_c}: a loss "
                            "would be partly invisible to the parity check")

            E, _nrm = error_basis(N_T, n_c, alpha=ALPHA, B=B)
            self.assertLess(np.linalg.norm(E - E_indep), 1e-14,
                            f"error_basis disagrees with a|j_L> at n_c={n_c}")

    def test_e1_column_pairing_matches_mod4_support(self):
        """
        Column j of E is the loss image of column j of B — pinned by Fock support.

        The cats sit on n = 0 (mod 4) and n = 2 (mod 4); one photon down puts
        their images on n = 3 and n = 1 respectively. Checking the support
        directly means a swapped column order fails here even though it would
        leave `error_basis`'s own orthonormality assertion perfectly happy, and
        it does not merely re-derive `E = A @ B / nrm`.

        This is the pairing `analysis.phase2_leakage_seepage.error_space_metrics`
        relies on when it reads L1_E off the diagonal of E^dag U E.
        """
        n_c = 24
        B = logical_basis(N_T, n_c, alpha=ALPHA)
        E, _nrm = error_basis(N_T, n_c, alpha=ALPHA, B=B)

        for col, (code_mod, err_mod) in enumerate([(0, 3), (2, 1)]):
            b_pop = np.abs(B[:, col].reshape(N_T, n_c)) ** 2
            e_pop = np.abs(E[:, col].reshape(N_T, n_c)) ** 2
            n_idx = np.arange(n_c)

            self.assertGreater(
                b_pop[0, n_idx % 4 == code_mod].sum(), 1.0 - 1e-12,
                f"logical column {col} is not supported on n = {code_mod} (mod 4)")
            self.assertGreater(
                e_pop[0, n_idx % 4 == err_mod].sum(), 1.0 - 1e-12,
                f"error column {col} is not supported on n = {err_mod} (mod 4); "
                "the columns of error_basis and logical_basis are misaligned")

    def test_e2_loss_is_only_approximately_correctable(self):
        """
        The two cats have UNEQUAL photon number, so Knill-Laflamme fails on the
        diagonal and a single loss is only approximately correctable here.

        Asserted as a live property of this code, not as trivia. Two things
        depend on it: there is deliberately no `error_cardinals` analogue in
        core/propagator.py (a|+X_L> renormalized is not the error-space |+X>),
        and the mirror-image assertion in EST/kitten_code.py:error_cardinals
        holds because the binomial code has EQUAL weights. If a future change
        to the cat construction made these equal, this test should fail loudly
        and force both decisions to be revisited.
        """
        n_c = 24
        B = logical_basis(N_T, n_c, alpha=ALPHA)
        _E, nrm = error_basis(N_T, n_c, alpha=ALPHA, B=B)
        kl = knill_laflamme_residual(B, N_T, n_c)

        # The two conditions the even cat code DOES satisfy, identically.
        self.assertLess(kl["a_block_norm"], 1e-12,
                        "<i_L|a|j_L> should vanish by parity")
        self.assertLess(kl["n_offdiag"], 1e-12,
                        "<+Z_L|a^dag a|-Z_L> should vanish by disjoint mod-4 support")

        # The one it does not.
        self.assertGreater(
            abs(nrm[0] / nrm[1] - 1.0), 0.05,
            "||a|+Z_L>|| and ||a|-Z_L>|| are now nearly equal; `a` restricted to "
            "the code space would be an isometry and the reasoning in "
            "core.propagator.error_basis needs revisiting")

        # The two functions must not drift apart: rel_mismatch is built from the
        # same norms error_basis returns.
        expected = (nrm[0] ** 2 - nrm[1] ** 2) / (0.5 * (nrm[0] ** 2 + nrm[1] ** 2))
        self.assertAlmostEqual(kl["rel_mismatch"], expected, places=12)

    def test_e3_drift_conserves_parity_exactly_but_the_pulses_do_not(self):
        """
        The idle claim: with the drive off, a lost photon stays detectable.

        H0 is diagonal in the joint Fock/transmon basis, so drift cannot move
        any amplitude from the odd error subspace back into the even code space
        — seepage is zero to machine precision, and the parity flag survives
        indefinitely.

        The contrast against U_X is what stops this from passing for a trivial
        reason. If L2_(E->C) were identically zero by construction (a bug in the
        metric, or an E that was not really outside C), the drift assertion
        alone would still pass. U_X puts measurable population back into the
        code space, so the metric demonstrably can be nonzero.
        """
        n_c = 24
        H0, Hc = make_hamiltonian(N_T, n_c)
        B = logical_basis(N_T, n_c, alpha=ALPHA)
        E, _nrm = error_basis(N_T, n_c, alpha=ALPHA, B=B)

        def seepage(U):
            return float(np.mean([
                np.linalg.norm(B.conj().T @ (U @ E[:, j])) ** 2 for j in range(2)
            ]))

        U_drift = full_propagator(np.zeros((550, 4)), H0, Hc, DT)
        L2_drift = seepage(U_drift)
        self.assertLess(L2_drift, 1e-14,
                        f"drift moved {L2_drift:.3e} back into the code space; H0 "
                        "is supposed to be parity-diagonal")

        # Drift's action on E is a pure phase per column, i.e. a known frame
        # rotation rather than lost information.
        M_E_drift = logical_block(U_drift, E)
        self.assertLess(max(abs(M_E_drift[0, 1]), abs(M_E_drift[1, 0])), 1e-12,
                        "drift mixed the two error words; it should be diagonal")

        U_X = full_propagator(_load("X"), H0, Hc, DT)
        self.assertGreater(seepage(U_X), 1e-6,
                           "U_X shows no seepage from E either — L2_(E->C) may be "
                           "identically zero for the wrong reason")


if __name__ == "__main__":
    unittest.main(verbosity=2)
