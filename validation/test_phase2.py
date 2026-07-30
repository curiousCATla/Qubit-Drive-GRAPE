#!/usr/bin/env python3
"""
test_phase2.py — acceptance checks for Phase 2 leakage / seepage tooling.

  a) Outside probes (A/B/C) are unit-norm and orthogonal to B at n_c=24
  b) Partition bins sum to 1 on random states and on U_X |g,1>
  c) Group C Gram vectors have P_logical_initial < 1e-12
  d) L1 for X at n_c=24 matches Phase-0/1 table within 1e-9
  e) Truncation destination class stable for X at 22/24/26 (inline)

Run:
    python3 validation/test_phase2.py [-v]
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.grape_core import make_hamiltonian, basis_state
from core.propagator import full_propagator, logical_basis, logical_block, leakage_L1
from validation.outside_inputs import outside_input_list, assert_outside_probes
from validation.resolve_populations import (
    resolve_state,
    resolve_after_pulse,
    partition_sum,
    PARTITION_KEYS,
)
from validation.validate_logical_gates import N_T, DT, ALPHA

PULSE_DIR = os.path.join(REPO_ROOT, "pulses")
PHASE1_CSV = os.path.join(REPO_ROOT, "tables", "phase1_decision_gate.csv")


class OutsideInputsTest(unittest.TestCase):
    def test_a_probes_outside_and_unit(self):
        n_c = 24
        B = logical_basis(N_T, n_c, alpha=ALPHA)
        probes = outside_input_list(n_t=N_T, n_c=n_c, alpha=ALPHA, B=B)
        assert_outside_probes(probes, B, tol=1e-12)
        groups = {g for _, g, _ in probes}
        self.assertIn("A", groups)
        self.assertIn("B", groups)
        self.assertIn("C", groups)
        self.assertNotIn("C_raw", groups)  # production n_c should use Gram path

    def test_c_group_c_true_outside(self):
        n_c = 24
        B = logical_basis(N_T, n_c, alpha=ALPHA)
        probes = outside_input_list(n_t=N_T, n_c=n_c, alpha=ALPHA, B=B)
        c_probes = [(n, k) for n, g, k in probes if g == "C"]
        self.assertGreaterEqual(len(c_probes), 2)
        for name, ket in c_probes:
            p = float(np.linalg.norm(B.conj().T @ ket) ** 2)
            self.assertLess(p, 1e-12, f"{name} P_logical={p}")


class ResolvePopulationsTest(unittest.TestCase):
    def test_b_partition_sums_to_one_random(self):
        n_c = 24
        n_t = N_T
        B = logical_basis(n_t, n_c, alpha=ALPHA)
        rng = np.random.default_rng(0)
        for _ in range(8):
            v = rng.normal(size=n_t * n_c) + 1j * rng.normal(size=n_t * n_c)
            v = v / np.linalg.norm(v)
            bins = resolve_state(v, B, n_t, n_c)
            self.assertAlmostEqual(partition_sum(bins), 1.0, places=10)

    def test_b_partition_on_UX_odd_probe(self):
        n_c = 24
        n_t = N_T
        path = os.path.join(PULSE_DIR, "u_X_main.npy")
        if not os.path.isfile(path):
            self.skipTest("missing u_X_main.npy")
        u = np.load(path)
        H0, Hc = make_hamiltonian(n_t, n_c)
        U = full_propagator(u, H0, Hc, DT)
        B = logical_basis(n_t, n_c, alpha=ALPHA)
        ket = basis_state(n_t, n_c, 0, 1)  # |g,1>
        bins = resolve_after_pulse(U, ket, B, n_t, n_c)
        self.assertAlmostEqual(partition_sum(bins), 1.0, places=10)
        self.assertLess(bins["P_logical_initial"], 1e-12)


class L1RegressionTest(unittest.TestCase):
    def test_d_L1_X_matches_phase1_table(self):
        path = os.path.join(PULSE_DIR, "u_X_main.npy")
        if not os.path.isfile(path):
            self.skipTest("missing u_X_main.npy")
        n_c = 24
        u = np.load(path)
        H0, Hc = make_hamiltonian(N_T, n_c)
        U = full_propagator(u, H0, Hc, DT)
        B = logical_basis(N_T, n_c, alpha=ALPHA)
        L1 = float(leakage_L1(logical_block(U, B)))

        # Prefer phase1 decision table; fall back to phase0 CSV.
        expected = None
        for csv_path, gate_col in (
            (PHASE1_CSV, "gate"),
            (os.path.join(REPO_ROOT, "tables", "phase0_corrected_fidelities.csv"), "gate"),
        ):
            if not os.path.isfile(csv_path):
                continue
            import pandas as pd
            df = pd.read_csv(csv_path)
            row = df[(df[gate_col] == "X") & (df["n_c"] == 24)]
            if len(row):
                expected = float(row.iloc[0]["L1"])
                break
        self.assertIsNotNone(expected, "no Phase-0/1 L1 reference table found")
        self.assertAlmostEqual(L1, expected, places=9)


class TruncationStabilityTest(unittest.TestCase):
    def test_e_X_destination_class_stable(self):
        """Coarse leak destination class for U_X agrees at n_c=22/24/26.

        Fine bins may flip e_odd <-> e_even; both are 'transmon_excited'.
        """
        path = os.path.join(PULSE_DIR, "u_X_main.npy")
        if not os.path.isfile(path):
            self.skipTest("missing u_X_main.npy")
        from analysis.phase2_leakage_seepage import analyze_gate, _coarse_dest_class

        coarse = {}
        for n_c in (22, 24, 26):
            _detail, summary = analyze_gate("X", n_c=n_c)
            coarse[n_c] = _coarse_dest_class(summary["dominant_leak_destination"])
        self.assertEqual(
            len(set(coarse.values())), 1,
            f"U_X coarse destination class not stable across truncations: {coarse}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
