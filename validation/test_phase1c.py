#!/usr/bin/env python3
"""
test_phase1c.py — acceptance checks for consolidating the enc/dec measuring
stick onto one route.

Before Phase 1c there were two implementations of the same number: the tier4
relative-phase checks in validate_logical_gates propagated |g,0>/|e,0> as state
vectors and assembled four overlaps by hand, while Phase 1b built the same 2x2
cross-subspace block by projecting the full propagator. They agreed to ~1e-15
(analysis/check_tier4_agreement.py), so the tier4 functions now delegate.

What must not break:
  - the return keys 'U_log' and 'F_avg_gate' (analysis/retrain_phase_coherent_gates.py,
    QuTip/qutip_validate.py and notebooks/grape_experiments.ipynb all read them);
  - the orientation -- enc is transmon->cat, dec is cat->transmon; swapping them
    silently transposes the block;
  - tier4_enc_gate_dec_pipeline, which measures the composite enc->gate->dec
    trajectory end to end and is NOT a duplicate of the block fidelity.

The pre-refactor algorithm is not re-derived here: it lives frozen in
analysis/check_tier4_agreement.py as `legacy_tier4_enc`/`legacy_tier4_dec` and
is imported, so there is exactly one copy of the reference and it cannot drift
out of sync with the script that reports on it.

Usage:
    python3 validation/test_phase1c.py -v
"""

import inspect
import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.grape_core import make_hamiltonian
from core.propagator import (
    full_propagator,
    cross_subspace_block,
    encode_bases,
    decode_bases,
    pedersen_gate_fidelity,
    leakage_L1,
)
from validation import validate_logical_gates as vlg
from validation.validate_logical_gates import (
    tier4_enc_relative_phase,
    tier4_dec_relative_phase,
    tier4_enc_dec_relative_phase,
    N_T,
    DT,
    ALPHA,
)
from analysis.check_tier4_agreement import legacy_tier4_enc, legacy_tier4_dec

N_C = 24
PULSE_DIR = os.path.join(REPO_ROOT, "pulses")
EXPECTED_PULSE_SHAPE = (550, 4)

CASES = {
    # gate: (live fn, frozen pre-refactor fn, bases builder)
    # enc: in=transmon out=cat.  dec: in=cat out=transmon.  Do not swap.
    "enc": (tier4_enc_relative_phase, legacy_tier4_enc, encode_bases),
    "dec": (tier4_dec_relative_phase, legacy_tier4_dec, decode_bases),
}


def _load(gate):
    u = np.load(os.path.join(PULSE_DIR, f"u_{gate}_main.npy"))
    assert u.shape == EXPECTED_PULSE_SHAPE, (
        f"u_{gate}_main.npy has shape {u.shape}, expected {EXPECTED_PULSE_SHAPE}"
    )
    return u


class Phase1cSetup(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pulses = {g: _load(g) for g in CASES}


class AgreementTest(Phase1cSetup):

    def test_a_legacy_and_block_routes_agree(self):
        """
        (a) The deliverable-1 gate, as a test: the frozen pre-refactor route and
        the canonical block route produce the same 2x2 block and the same
        fidelity. This is what licensed the consolidation.
        """
        for gate, (_, legacy_fn, bases_fn) in CASES.items():
            with self.subTest(gate=gate):
                u = self.pulses[gate]
                old = legacy_fn(u, n_c=N_C, n_t=N_T, dt=DT)

                H0, Hc = make_hamiltonian(N_T, N_C)
                B_in, B_out, E_ideal = bases_fn(N_T, N_C, alpha=ALPHA)
                M = cross_subspace_block(
                    full_propagator(u, H0, Hc, DT), B_in, B_out)

                self.assertLess(abs(old["F_avg_gate"] - pedersen_gate_fidelity(E_ideal, M)), 1e-9)
                self.assertTrue(np.allclose(old["U_log"], M, atol=1e-9))

    def test_a2_orientation_is_not_transposed(self):
        """
        (a) cont. The agreement above would survive a transposed block if the
        block happened to be symmetric. Pin the orientation directly: for enc,
        row index is the CAT state and column index is the TRANSMON state.
        """
        u = self.pulses["enc"]
        H0, Hc = make_hamiltonian(N_T, N_C)
        B_in, B_out, _ = encode_bases(N_T, N_C, alpha=ALPHA)
        U = full_propagator(u, H0, Hc, DT)
        M = tier4_enc_relative_phase(u, n_c=N_C, n_t=N_T, dt=DT)["U_log"]

        # M[k, j] = <cat_k| U |transmon_j>
        for k in range(2):
            for j in range(2):
                self.assertLess(
                    abs(M[k, j] - np.vdot(B_out[:, k], U @ B_in[:, j])), 1e-12)


class ContractTest(Phase1cSetup):

    def test_b_return_contract_preserved_and_extended(self):
        """
        (b) 'U_log' and 'F_avg_gate' still present and numerically unchanged
        versus the pre-refactor implementation; 'L1' newly added.
        """
        for gate, (live_fn, legacy_fn, _) in CASES.items():
            with self.subTest(gate=gate):
                u = self.pulses[gate]
                new = live_fn(u, n_c=N_C, n_t=N_T, dt=DT)
                old = legacy_fn(u, n_c=N_C, n_t=N_T, dt=DT)

                self.assertIn("U_log", new)
                self.assertIn("F_avg_gate", new)
                self.assertIn("L1", new)

                self.assertLess(abs(new["F_avg_gate"] - old["F_avg_gate"]), 1e-9)
                self.assertTrue(np.allclose(new["U_log"], old["U_log"], atol=1e-9))
                self.assertEqual(new["U_log"].shape, (2, 2))

                # L1 must be the leakage of the returned block, not a stray number.
                self.assertLess(abs(new["L1"] - leakage_L1(new["U_log"])), 1e-15)
                self.assertGreater(new["L1"], 0.0)
                self.assertLess(new["L1"], 1e-2)

    def test_b2_signatures_unchanged(self):
        """(b) cont. Call sites pass n_c/n_t/dt by keyword; keep the names."""
        for fn in (tier4_enc_relative_phase, tier4_dec_relative_phase):
            params = list(inspect.signature(fn).parameters)
            self.assertEqual(params[1:], ["n_c", "n_t", "dt"])

    def test_c_single_source_of_truth(self):
        """
        (c) F_avg_gate is bit-for-bit pedersen_gate_fidelity of the block --
        exact equality, not a tolerance. Any independent fidelity arithmetic
        surviving inside these functions would perturb the last ulp and fail.
        """
        for gate, (live_fn, _, bases_fn) in CASES.items():
            with self.subTest(gate=gate):
                u = self.pulses[gate]
                H0, Hc = make_hamiltonian(N_T, N_C)
                B_in, B_out, E_ideal = bases_fn(N_T, N_C, alpha=ALPHA)
                M = cross_subspace_block(
                    full_propagator(u, H0, Hc, DT), B_in, B_out)

                new = live_fn(u, n_c=N_C, n_t=N_T, dt=DT)
                self.assertEqual(new["F_avg_gate"], pedersen_gate_fidelity(E_ideal, M))
                np.testing.assert_array_equal(new["U_log"], M)

    def test_d_no_stale_fidelity_or_hand_extraction(self):
        """
        (d) Source-level: these functions must reach the number only through
        pedersen_gate_fidelity. No average_gate_fidelity, no propagate_pulse,
        no hand-assembled np.vdot overlaps.
        """
        for fn in (tier4_enc_relative_phase, tier4_dec_relative_phase):
            src = inspect.getsource(fn)
            body = src[src.index('"""', src.index('"""') + 3) + 3:]  # strip docstring
            with self.subTest(fn=fn.__name__):
                self.assertNotIn("average_gate_fidelity", body)
                self.assertNotIn("propagate_pulse", body)
                self.assertNotIn("np.vdot", body)
                self.assertIn("pedersen_gate_fidelity", body)
                self.assertIn("cross_subspace_block", body)

    def test_d2_dead_helpers_removed(self):
        """
        (d) cont. The private basis helpers that fed the old extraction are gone;
        leaving them would invite rebuilding the second route.
        """
        self.assertFalse(hasattr(vlg, "_computational_basis_states"))
        self.assertFalse(hasattr(vlg, "_logical_cat_basis_states"))


class WrapperAndPipelineTest(Phase1cSetup):

    def test_wrapper_still_returns_both_branches(self):
        """The combined report keeps its {'enc','dec'} shape and gains L1."""
        res = tier4_enc_dec_relative_phase(
            self.pulses["enc"], self.pulses["dec"], n_c=N_C, n_t=N_T, dt=DT)
        self.assertEqual(set(res), {"enc", "dec"})
        for label in ("enc", "dec"):
            self.assertIn("U_log", res[label])
            self.assertIn("F_avg_gate", res[label])
            self.assertIn("L1", res[label])
            self.assertGreater(res[label]["F_avg_gate"], 0.99)

    def test_f_pipeline_untouched_and_still_independent(self):
        """
        (f) tier4_enc_gate_dec_pipeline measures the composite enc->gate->dec
        trajectory and must not have been folded into the block route. It still
        propagates states itself, and it still does not reference the block API.
        """
        src = inspect.getsource(vlg.tier4_enc_gate_dec_pipeline)
        self.assertIn("propagate_pulse", src)
        self.assertNotIn("cross_subspace_block", src)
        self.assertNotIn("full_propagator", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
