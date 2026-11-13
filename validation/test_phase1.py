#!/usr/bin/env python3
"""
test_phase1.py

Acceptance checks for Phase 1: guard + canonicalize the coherent training
objective.

  a) GUARD FIRES: assert_coherent_inputs raises on 6 states and on 2
     non-orthonormal states; passes on the two logical cats.
  b) CONVENTION: get_coherent_state_pairs targets reconstruct IDEAL_LOGICAL_U
     under the column-per-input convention (relative phase correct by
     construction).
  c) PHASE SENSITIVITY LIVE: coherent_fidelity_multi_state drops well below 1
     when the second target is phase-flipped (not phase-blind).
  d) POST-RETRAIN: skipped when Phase 0 found no relative-phase defect; if
     pulses/phase1/ exists, Pedersen(new) >= Pedersen(old) - 1e-4.
  e) IDENTITY STILL HOLDS: 6-state mean state-transfer fidelity equals
     Pedersen on the extracted logical block of each main pulse.

Run:
    python3 validation/test_phase1.py [-v]
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.grape_core import (
    make_hamiltonian,
    assert_coherent_inputs,
    coherent_fidelity_multi_state,
    fidelity_multi_state,
)
from core.cat_code import get_coherent_state_pairs, make_coherent_gate_factory
from core.propagator import (
    PAULI_EIGENSTATE_COEFFS,
    full_propagator,
    logical_basis,
    logical_block,
    pedersen_gate_fidelity,
)
from validation.validate_logical_gates import IDEAL_LOGICAL_U, ALPHA, DT, N_T

PULSE_DIR = os.path.join(REPO_ROOT, "pulses")
PHASE1_DIR = os.path.join(PULSE_DIR, "phase1")
LOGICAL_GATES = ("I", "X", "Y", "Z", "H", "T")


def _pedersen_for_pulse(gate, u, n_c=24, n_t=N_T, dt=DT, alpha=ALPHA):
    H0, Hc = make_hamiltonian(n_t, n_c)
    U = full_propagator(u, H0, Hc, dt)
    B = logical_basis(n_t, n_c, alpha=alpha)
    U_log = logical_block(U, B)
    return float(pedersen_gate_fidelity(IDEAL_LOGICAL_U[gate], U_log)), U_log


class GuardFiresTest(unittest.TestCase):
    """Check (a): assert_coherent_inputs seatbelt."""

    def test_a_passes_on_two_cats(self):
        n_c = 24
        B = logical_basis(N_T, n_c, alpha=ALPHA)
        psi_i_list = [B[:, 0], B[:, 1]]
        assert_coherent_inputs(psi_i_list)  # must not raise

    def test_a_raises_on_six_states(self):
        n_c = 24
        B = logical_basis(N_T, n_c, alpha=ALPHA)
        # Six non-orthonormal Pauli eigenstates in the joint space
        six = [B @ c for c in PAULI_EIGENSTATE_COEFFS.values()]
        with self.assertRaises(AssertionError) as ctx:
            assert_coherent_inputs(six)
        self.assertIn("exactly d=2", str(ctx.exception))

    def test_a_raises_on_non_orthonormal_pair(self):
        n_c = 24
        B = logical_basis(N_T, n_c, alpha=ALPHA)
        # Same state twice: Gram is not I
        bad = [B[:, 0].copy(), B[:, 0].copy()]
        with self.assertRaises(AssertionError) as ctx:
            assert_coherent_inputs(bad)
        self.assertIn("orthonormal", str(ctx.exception).lower())

    def test_a_hot_path_len_assert(self):
        """Cheap len assert lives inside coherent_fidelity_multi_state itself."""
        n_c = 8  # small for speed
        n_t = 2
        H0, Hc = make_hamiltonian(n_t, n_c)
        u = np.zeros((4, 4))
        psi = np.zeros(n_t * n_c, dtype=complex)
        psi[0] = 1.0
        with self.assertRaises(AssertionError):
            coherent_fidelity_multi_state(
                u, H0, Hc, [psi, psi], [psi], dt=DT, want_grad=False
            )


class ConventionTest(unittest.TestCase):
    """Check (b): targets carry the right relative phase from IDEAL_LOGICAL_U."""

    def test_b_reconstructs_ideal_logical_u(self):
        n_c = 24
        B = logical_basis(N_T, n_c, alpha=ALPHA)
        for gate in LOGICAL_GATES:
            psi_i_list, psi_f_list = get_coherent_state_pairs(
                gate, n_c=n_c, n_t=N_T, alpha=ALPHA
            )
            self.assertEqual(len(psi_i_list), 2)
            self.assertEqual(len(psi_f_list), 2)
            # M2[k, j] = <B[:,k] | psi_f_list[j]>
            M2 = np.zeros((2, 2), dtype=complex)
            for j in range(2):
                for k in range(2):
                    M2[k, j] = np.vdot(B[:, k], psi_f_list[j])
            np.testing.assert_allclose(
                M2, IDEAL_LOGICAL_U[gate], atol=1e-12,
                err_msg=f"get_coherent_state_pairs('{gate}') targets do not "
                        f"reconstruct IDEAL_LOGICAL_U['{gate}']",
            )
            # Inputs are exactly the logical basis columns
            np.testing.assert_allclose(psi_i_list[0], B[:, 0], atol=1e-12)
            np.testing.assert_allclose(psi_i_list[1], B[:, 1], atol=1e-12)

    def test_b_factory_adapter_zip_interface(self):
        factory = make_coherent_gate_factory("Y")
        pairs = factory(n_c=24, n_t=N_T)
        self.assertEqual(len(pairs), 2)
        psi_i_list = [p[0] for p in pairs]
        assert_coherent_inputs(psi_i_list)


class PhaseSensitivityTest(unittest.TestCase):
    """Check (c): coherent objective is not phase-blind."""

    def test_c_phase_flip_drops_f_coh(self):
        n_c = 24
        n_t = N_T
        H0, Hc = make_hamiltonian(n_t, n_c)
        # Zero-length "pulse": identity propagator on the Hilbert space.
        # (A multi-step free evolution under H0 is NOT identity.)
        u_id = np.zeros((0, 4))  # N=0 -> U = I via empty product

        psi_i_list, psi_f_list = get_coherent_state_pairs(
            "I", n_c=n_c, n_t=n_t, alpha=ALPHA
        )
        # Perfect identity transfer: targets == inputs
        F_perfect, _ = coherent_fidelity_multi_state(
            u_id, H0, Hc, psi_i_list, psi_i_list, DT, want_grad=False
        )
        self.assertGreater(F_perfect, 1.0 - 1e-12,
                           f"perfect identity should give F_coh≈1, got {F_perfect}")

        # Phase-scramble the second target only
        psi_f_scrambled = [psi_f_list[0].copy(), np.exp(1j * np.pi) * psi_f_list[1]]
        F_scrambled, _ = coherent_fidelity_multi_state(
            u_id, H0, Hc, psi_i_list, psi_f_scrambled, DT, want_grad=False
        )
        self.assertLess(
            F_scrambled, 0.5,
            f"phase-scrambled targets should drop F_coh well below 1, "
            f"got {F_scrambled} (perfect was {F_perfect})",
        )
        # Incoherent average is blind to this defect
        F_avg, _ = fidelity_multi_state(
            u_id, H0, Hc, psi_i_list, psi_f_scrambled, DT, want_grad=False
        )
        self.assertGreater(
            F_avg, 0.99,
            "incoherent average should still look perfect after a pure phase "
            f"flip on one target; got F_mean={F_avg}",
        )


class PostRetrainTest(unittest.TestCase):
    """
    Check (d): retrain only if Phase-0 found a relative-phase defect.

    Decision (Phase 1): no defect remains on u_*_main.npy (all Pedersen
    ≫ 0.99 at n_c=24). Retrain was skipped; pulses/phase1/ should be absent.
    If it exists (manual retrain), require non-regression.
    """

    def test_d_no_phase0_defect_on_main_pulses(self):
        """Audit: main pulses already pass the Phase-0 relative-phase bar."""
        for gate in LOGICAL_GATES:
            path = os.path.join(PULSE_DIR, f"u_{gate}_main.npy")
            self.assertTrue(os.path.isfile(path), f"missing {path}")
            u = np.load(path)
            F_ped, _ = _pedersen_for_pulse(gate, u, n_c=24)
            self.assertGreaterEqual(
                F_ped, 0.99,
                f"Phase-0 defect still present on U_{gate}: Pedersen={F_ped:.6f}",
            )

    def test_d_phase1_pulses_non_regression_if_present(self):
        if not os.path.isdir(PHASE1_DIR):
            self.skipTest(
                "pulses/phase1/ absent — retrain correctly skipped "
                "(no Phase-0 relative-phase defect)"
            )
        for gate in LOGICAL_GATES:
            new_path = os.path.join(PHASE1_DIR, f"u_{gate}.npy")
            if not os.path.isfile(new_path):
                continue
            old = np.load(os.path.join(PULSE_DIR, f"u_{gate}_main.npy"))
            new = np.load(new_path)
            F_old, _ = _pedersen_for_pulse(gate, old, n_c=24)
            F_new, _ = _pedersen_for_pulse(gate, new, n_c=24)
            self.assertGreaterEqual(
                F_new, F_old - 1e-4,
                f"phase1 U_{gate} regressed: Pedersen {F_new:.6f} < "
                f"{F_old:.6f} - 1e-4",
            )


class SixStatePedersenIdentityTest(unittest.TestCase):
    """Check (e): 6-state mean == Pedersen on main pulses (Phase-0 6b identity)."""

    def test_e_six_state_mean_equals_pedersen_on_main(self):
        coeffs = list(PAULI_EIGENSTATE_COEFFS.values())
        n_c = 24
        for gate in LOGICAL_GATES:
            path = os.path.join(PULSE_DIR, f"u_{gate}_main.npy")
            if not os.path.isfile(path):
                self.skipTest(f"missing {path}")
            u = np.load(path)
            F_ped, U_log = _pedersen_for_pulse(gate, u, n_c=n_c)
            U_ideal = IDEAL_LOGICAL_U[gate]
            # M = U_ideal^dagger U_log; mean |c^dag M c|^2 over Pauli eigenstates
            M = U_ideal.conj().T @ U_log
            mean6 = float(np.mean([np.abs(c.conj() @ (M @ c)) ** 2 for c in coeffs]))
            self.assertAlmostEqual(
                mean6, F_ped, places=10,
                msg=f"6-state mean != Pedersen for {gate}: "
                    f"mean6={mean6:.12f}, F_ped={F_ped:.12f}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
