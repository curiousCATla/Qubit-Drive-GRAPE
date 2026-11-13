#!/usr/bin/env python3
"""
test_phase1b.py — acceptance checks for the cross-subspace (encode/decode)
measuring stick.

The whole phase exists because encode and decode are NOT logical gates. A
logical gate maps the cat subspace to itself, so one basis B serves as both
projector and `logical_block(U, B) = B^dag U B` is the effective map. Encode
maps the TRANSMON subspace {|g,0>, |e,0>} to the CAT subspace
{|+Z_L>, |-Z_L>}; decode maps it back. The effective map is an OFF-DIAGONAL 2x2
block of U_full, with a different basis on each side.

Two traps this file pins down:

  1. Reusing `logical_block` on an encode pulse type-checks and returns a
     plausible-looking 2x2 matrix. It is measuring what the encode pulse does to
     CAT inputs, which is not what encode is for, and reports ~0.07 instead of
     ~0.997 (test d).
  2. `E_ideal = eye(2)` is correct only because the basis column order matches
     the pair order in get_encode/decode_state_pairs. Nothing enforces that
     coupling at runtime, so it is asserted here (test b).

Also pinned: the input and output subspaces are NON-orthogonal
(<g,0|+Z_L> ~ 0.469 -- vacuum overlaps the even cat), which is why no 4x4 block
over the union of the two bases is well defined and none is ever built. That
non-orthogonality is irrelevant to the coherent training objective, which only
requires the INPUT states to be orthonormal (test e).

Usage:
    python3 validation/test_phase1b.py -v
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
    basis_state,
    step_data,
    assert_coherent_inputs,
    coherent_fidelity_multi_state,
    fidelity_multi_state,
)
from core.cat_code import get_encode_state_pairs, get_decode_state_pairs
from core.propagator import (
    full_propagator,
    logical_basis,
    logical_block,
    transmon_basis,
    cross_subspace_block,
    encode_bases,
    decode_bases,
    pedersen_gate_fidelity,
    leakage_L1,
)

N_T = 3
N_C = 24
DT = 0.002
ALPHA = np.sqrt(3.0)
PULSE_DIR = os.path.join(REPO_ROOT, "pulses")
EXPECTED_PULSE_SHAPE = (550, 4)


def _load(gate):
    u = np.load(os.path.join(PULSE_DIR, f"u_{gate}_main.npy"))
    # DT is hardcoded above; a dt-refined pulse (e.g. a 10x refine_pulse_dt
    # output, 5500x4 at dt=2e-4) would be silently scored at the wrong timestep.
    assert u.shape == EXPECTED_PULSE_SHAPE, (
        f"u_{gate}_main.npy has shape {u.shape}, expected {EXPECTED_PULSE_SHAPE}"
    )
    return u


def _propagate(u, H0, Hc, psi0, dt=DT):
    """State-vector propagation, matching validate_logical_gates.propagate_pulse."""
    psi = psi0.copy().astype(complex)
    for uk in u:
        Uk, _, _ = step_data(H0, Hc, uk, dt)
        psi = Uk @ psi
    return psi


class Phase1bSetup(unittest.TestCase):
    """Shared, expensive-once setup: two propagators at n_c=24."""

    @classmethod
    def setUpClass(cls):
        cls.H0, cls.Hc = make_hamiltonian(N_T, N_C)
        cls.u_enc = _load("enc")
        cls.u_dec = _load("dec")
        cls.U_enc = full_propagator(cls.u_enc, cls.H0, cls.Hc, DT)
        cls.U_dec = full_propagator(cls.u_dec, cls.H0, cls.Hc, DT)


class CrossSubspaceBlockTest(Phase1bSetup):

    def test_a_block_equals_direct_four_overlaps(self):
        """(a) The projection is exactly the four hand-computed overlaps."""
        B_in, B_out, _ = encode_bases(N_T, N_C, alpha=ALPHA)
        M = cross_subspace_block(self.U_enc, B_in, B_out)

        M_direct = np.empty((2, 2), dtype=complex)
        for k in range(2):
            for j in range(2):
                # <out_k| U |in_j> -- column per input, same as logical_block
                M_direct[k, j] = np.vdot(B_out[:, k], self.U_enc @ B_in[:, j])

        self.assertLess(np.linalg.norm(M - M_direct), 1e-12)

    def test_a2_input_output_subspaces_are_non_orthogonal(self):
        """
        (a2) Documents WHY there is no 4x4 block: the union of the two bases is
        not an orthonormal set, so no 4x4 'ideal' is definable.
        """
        B_tr = transmon_basis(N_T, N_C)
        B_cat = logical_basis(N_T, N_C, alpha=ALPHA)
        overlap = abs(np.vdot(B_tr[:, 0], B_cat[:, 0]))  # <g,0|+Z_L>
        self.assertGreater(overlap, 0.4)
        self.assertAlmostEqual(overlap, 0.469, places=2)
        # Each basis is individually orthonormal -- that is all the Pedersen
        # formula needs.
        self.assertLess(np.linalg.norm(B_tr.conj().T @ B_tr - np.eye(2)), 1e-12)
        self.assertLess(np.linalg.norm(B_cat.conj().T @ B_cat - np.eye(2)), 1e-12)

    def test_b_ideal_convention_matches_pair_order(self):
        """
        (b) E_ideal = eye(2) is a claim about column order. Verify it against the
        trained pairs instead of trusting it: reconstruct the ideal map from
        get_encode/decode_state_pairs by expanding each target in the output
        basis, and check it reproduces the identity.
        """
        for name, bases, factory in (
            ("encode", encode_bases, get_encode_state_pairs),
            ("decode", decode_bases, get_decode_state_pairs),
        ):
            with self.subTest(name=name):
                B_in, B_out, E_ideal = bases(N_T, N_C, alpha=ALPHA)
                pairs = factory(n_c=N_C, n_t=N_T, alpha=ALPHA)
                self.assertEqual(len(pairs), 2)

                E_from_pairs = np.empty((2, 2), dtype=complex)
                for j, (psi_i, psi_f) in enumerate(pairs):
                    # input j must BE column j of B_in ...
                    self.assertAlmostEqual(
                        abs(np.vdot(B_in[:, j], psi_i)), 1.0, places=12,
                        msg=f"{name}: pair {j} input is not column {j} of B_in",
                    )
                    # ... and its target expands in B_out as column j of the ideal
                    E_from_pairs[:, j] = B_out.conj().T @ psi_f

                self.assertLess(
                    np.linalg.norm(E_from_pairs - np.eye(2)), 1e-12,
                    msg=f"{name}: pair order does not match basis column order; "
                        f"reconstructed ideal =\n{np.round(E_from_pairs, 6)}",
                )
                self.assertLess(np.linalg.norm(E_ideal - np.eye(2)), 1e-15)

    def test_c_cross_subspace_fidelity_is_sane(self):
        """(c) The current pulses score ~0.997 under the correct projectors."""
        for name, U, bases in (
            ("enc", self.U_enc, encode_bases),
            ("dec", self.U_dec, decode_bases),
        ):
            with self.subTest(name=name):
                B_in, B_out, E_ideal = bases(N_T, N_C, alpha=ALPHA)
                M = cross_subspace_block(U, B_in, B_out)
                F = pedersen_gate_fidelity(E_ideal, M)
                L1 = leakage_L1(M)
                self.assertGreaterEqual(F, 0.99)
                self.assertLessEqual(F, 1.0)
                # Pedersen and L1 must be mutually consistent: the '+d' form
                # over-reports by exactly L1/3 at d=2.
                F_old = (abs(np.trace(E_ideal.conj().T @ M)) ** 2 + 2) / 6.0
                self.assertLess(abs((F_old - F) - L1 / 3.0), 1e-12)

        # The block is ~ a global phase times identity: diagonal, equal phases.
        B_in, B_out, _ = encode_bases(N_T, N_C, alpha=ALPHA)
        M = cross_subspace_block(self.U_enc, B_in, B_out)
        self.assertLess(abs(M[0, 1]), 1e-2)
        self.assertLess(abs(M[1, 0]), 1e-2)
        self.assertLess(abs(M[0, 0] - M[1, 1]), 1e-2)

    def test_d_logical_block_is_the_wrong_projector_here(self):
        """
        (d) The trap. logical_block on an encode pulse measures U's action on CAT
        inputs -- meaningless for encode -- and lands far below the true number.
        Asserted as a gap, not a pinned constant: the point is that the two
        differ by ~0.9, not that the wrong answer is any particular value.
        """
        B = logical_basis(N_T, N_C, alpha=ALPHA)
        B_in, B_out, E_ideal = encode_bases(N_T, N_C, alpha=ALPHA)

        F_correct = pedersen_gate_fidelity(
            E_ideal, cross_subspace_block(self.U_enc, B_in, B_out))
        F_wrong = pedersen_gate_fidelity(
            np.eye(2, dtype=complex), logical_block(self.U_enc, B))

        self.assertLess(F_wrong, 0.2)
        self.assertGreater(F_correct - F_wrong, 0.8)


class CoherentGuardTest(unittest.TestCase):

    def test_e_coherent_guard_accepts_both_input_sets(self):
        """
        (e) Both enc and dec INPUT sets are orthonormal, so the coherent
        objective is legitimate for both: {|g,0>, |e,0>} differ in transmon
        level, {|+Z_L>, |-Z_L>} differ in Fock parity mod 4. The
        input/output non-orthogonality (test a2) is irrelevant to this guard.
        """
        enc_in = [p[0] for p in get_encode_state_pairs(n_c=N_C, n_t=N_T)]
        dec_in = [p[0] for p in get_decode_state_pairs(n_c=N_C, n_t=N_T)]
        assert_coherent_inputs(enc_in)   # must not raise
        assert_coherent_inputs(dec_in)   # must not raise

    def test_e2_coherent_guard_rejects_bad_input_sets(self):
        """(e) ... and rejects wrong count / non-orthonormal inputs."""
        enc_in = [p[0] for p in get_encode_state_pairs(n_c=N_C, n_t=N_T)]

        with self.assertRaises(AssertionError):
            assert_coherent_inputs(enc_in + [basis_state(N_T, N_C, 0, 1)])

        # Non-orthogonal pair: vacuum and the even cat overlap at 0.469.
        cat_plus = logical_basis(N_T, N_C, alpha=ALPHA)[:, 0]
        with self.assertRaises(AssertionError):
            assert_coherent_inputs([enc_in[0], cat_plus])


class PhaseSensitivityTest(Phase1bSetup):

    def test_f_only_the_coherent_objective_sees_a_branch_phase_flip(self):
        """
        (f) Flip the sign of the SECOND encode target. Physically this is a
        different gate (encode composed with logical Z), but the incoherent
        per-branch average cannot tell: it lets every branch match its target up
        to an independent global phase, so it barely moves. The coherent
        objective sums complex overlaps before squaring and collapses to ~0.
        """
        pairs = get_encode_state_pairs(n_c=N_C, n_t=N_T, alpha=ALPHA)
        psi_i = [p[0] for p in pairs]
        psi_f = [p[1] for p in pairs]
        psi_f_flipped = [psi_f[0], np.exp(1j * np.pi) * psi_f[1]]

        F_inc, _ = fidelity_multi_state(
            self.u_enc, self.H0, self.Hc, psi_i, psi_f, DT, want_grad=False)
        F_inc_flip, _ = fidelity_multi_state(
            self.u_enc, self.H0, self.Hc, psi_i, psi_f_flipped, DT, want_grad=False)
        F_coh, _ = coherent_fidelity_multi_state(
            self.u_enc, self.H0, self.Hc, psi_i, psi_f, DT, want_grad=False)
        F_coh_flip, _ = coherent_fidelity_multi_state(
            self.u_enc, self.H0, self.Hc, psi_i, psi_f_flipped, DT, want_grad=False)

        self.assertGreater(F_inc, 0.99)
        self.assertLess(abs(F_inc_flip - F_inc), 1e-9)   # blind to the flip
        self.assertGreater(F_coh, 0.99)
        self.assertLess(F_coh_flip, 1e-3)                # sees it

    def test_f2_coherent_matches_pedersen_on_the_current_pulses(self):
        """
        (f2) The training objective and the reported metric agree here, which is
        what says the saved pulses carry no hidden relative-phase error and did
        not need retraining under the coherent objective. F_coh is the objective;
        F_avg (Pedersen) is the reported fidelity -- they are different
        quantities that happen to coincide when leakage is the only defect.
        """
        for name, u, bases, factory in (
            ("enc", self.u_enc, encode_bases, get_encode_state_pairs),
            ("dec", self.u_dec, decode_bases, get_decode_state_pairs),
        ):
            with self.subTest(name=name):
                pairs = factory(n_c=N_C, n_t=N_T, alpha=ALPHA)
                F_coh, _ = coherent_fidelity_multi_state(
                    u, self.H0, self.Hc,
                    [p[0] for p in pairs], [p[1] for p in pairs],
                    DT, want_grad=False)

                B_in, B_out, E_ideal = bases(N_T, N_C, alpha=ALPHA)
                F_avg = pedersen_gate_fidelity(
                    E_ideal, cross_subspace_block(
                        full_propagator(u, self.H0, self.Hc, DT), B_in, B_out))

                self.assertLess(abs(F_coh - F_avg), 5e-3)


class ConsistencyTest(Phase1bSetup):

    def test_g_trace_MMdag_is_the_arrived_population(self):
        """
        (g) Tr(M M^dag) must equal the summed output-subspace population of the
        two propagated inputs. Computed via independent state propagation (not
        the full propagator), so this also cross-checks full_propagator against
        the state-vector path the training objective uses.
        """
        B_in, B_out, _ = encode_bases(N_T, N_C, alpha=ALPHA)
        M = cross_subspace_block(self.U_enc, B_in, B_out)
        tr_MMdag = np.trace(M.conj().T @ M).real

        pop_sum = 0.0
        for j in range(2):
            out = _propagate(self.u_enc, self.H0, self.Hc, B_in[:, j])
            pop_sum += np.linalg.norm(B_out.conj().T @ out) ** 2

        self.assertLess(abs(tr_MMdag - pop_sum), 1e-9)
        self.assertLess(abs(leakage_L1(M) - (1.0 - pop_sum / 2.0)), 1e-9)

    def test_h_round_trip_returns_to_the_transmon_subspace(self):
        """
        (h) U_dec U_enc returns the transmon computational states to the transmon
        subspace. Magnitude-only: this does NOT check that enc and dec share a
        phase convention. Each is trained coherently to identity up to its own
        global phase, so the round trip is identity up to a global phase --
        fine for state encoding, insufficient if a later use needs a
        phase-faithful inverse.
        """
        B_tr = transmon_basis(N_T, N_C)
        U_round = self.U_dec @ self.U_enc
        M = cross_subspace_block(U_round, B_tr, B_tr)
        pop = np.trace(M.conj().T @ M).real / 2.0
        self.assertGreater(pop, 0.99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
