#!/usr/bin/env python3
"""
test_fwm_tail_ge_chip.py

Checks for the chip option of the Gertler-comb tails (DGRAPE/ge_chip.py, chip= / gate_chip= in
DGRAPE/fwm_tail_ge.py; FWM_tails_ge_chips.ipynb).

  * constants    -- ge_chip's Shirol cap and band are fwm_tail_ge's; drift(SHIROL) is DGRAPE.model's drift
  * anchor       -- derived(1.12, 1.13) differs from Shirol's drift only by -chi_ar n_a n_r; with chi_ar = 0
                    it gives the same tail fidelity as chip=None, bit for bit (the apply_chip path equals the
                    built-in one)
  * derived      -- K and chi_q' scale as chi_ge^2, chi_gf as chi_ge, chi_ar = chi_ge chi_qr / (2 alpha)
  * swap         -- on chip (3, 5) one comb-M tooth swaps |n,e,0> -> |n+1,g,1> in pi / (2 g_m |xi| sqrt(n+1)),
                    g_m = sqrt(chi_ge chi_qr)
  * teeth        -- comb-T lines move to about +n chi_ge of the chip
  * gradient     -- W form = augmented-generator directional derivative = central differences on chip (3, 5)
  * gate chip    -- the gate run on SHIROL through gate_output_on_chip equals the track-7 input exactly
  * est gate     -- est_gate.system(chip=None) = system(chip=SHIROL); another chip changes only H0
  * decoding     -- a known phase exp(i [theta n + phi n(n-1)/2]) is found and undone

Run:
    python validation/test_fwm_tail_ge_chip.py [-v]       (under a minute)
"""

import dataclasses
import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from AQEC import device as dev                             # noqa: E402
from DGRAPE import fwm_tail as ft                          # noqa: E402
from DGRAPE import fwm_tail_ge as ge                       # noqa: E402
from DGRAPE import ge_chip as gc                           # noqa: E402
from DGRAPE.code import process_targets                    # noqa: E402
from DGRAPE.model import E, G, drift_hamiltonian, index    # noqa: E402

np.seterr(divide="ignore", over="ignore", invalid="ignore")
NC = 8
CHIP35 = gc.derived(3.0, 5.0)


def _inputs(n_c=NC, n_r=2):
    E3, T3, w = process_targets("X", n_c, n_r)
    return ge.project_ge(E3, n_c, n_r), ge.project_ge(T3, n_c, n_r), w


def _rand_b4(N, eps_amp, xi_amp, seed):
    rng = np.random.default_rng(seed)
    return np.c_[eps_amp * rng.normal(size=(N, 2)), xi_amp * rng.normal(size=(N, 2))]


class Constants(unittest.TestCase):
    def test_shirol_cap_band(self):
        self.assertEqual(gc.CAP_T_SHIROL, ge.CAP_T)
        self.assertEqual(tuple(gc.BAND_SHIROL), tuple(ge.BAND))
        self.assertEqual(gc.SHIROL.g_m, ge.G_M)

    def test_shirol_drift(self):
        for hot in (False, True):
            np.testing.assert_array_equal(gc.drift(gc.SHIROL, NC, 2, hot), drift_hamiltonian(NC, 2, hot))


class Anchor(unittest.TestCase):
    def test_only_chi_ar(self):
        a = gc.derived(1.12, 1.13)
        self.assertGreater(a.chi_ar, 0.0)
        d = np.real(np.diag(gc.drift(a, NC, 2) - gc.drift(gc.SHIROL, NC, 2)))
        for n in range(NC):
            for j in range(3):
                for m in range(2):
                    self.assertAlmostEqual(d[index(n, j, m, NC)], -a.chi_ar * n * m, places=12)

    def test_bit_for_bit_without_chi_ar(self):
        a0 = dataclasses.replace(gc.derived(1.12, 1.13), chi_ar=0.0)
        R0, T, w = _inputs()
        b4 = _rand_b4(40, 2.0, 0.1, 7)
        F_none = ge.fidelity(ge.tail_model(NC, 2, 2), b4, R0, T, w)
        F_a0 = ge.fidelity(ge.tail_model(NC, 2, 2, chip=a0), b4, R0, T, w)
        self.assertEqual(F_none, F_a0)


class Derived(unittest.TestCase):
    def test_scaling(self):
        c = gc.derived(2.24, 3.0)          # r = 2
        self.assertAlmostEqual(c.K / dev.K_a, 4.0, places=12)
        self.assertAlmostEqual(c.chi_q_p / dev.chi_q_p, 4.0, places=12)
        self.assertAlmostEqual(c.chi_gf / dev.chi_gf, 2.0, places=12)
        self.assertAlmostEqual(c.chi_ar, c.chi_ge * c.chi_qr / (2 * dev.alpha_q), places=12)
        self.assertAlmostEqual(c.cap_T / gc.CAP_T_SHIROL, 2.0, places=12)
        np.testing.assert_allclose(c.band, (-17.2, 17.2), rtol=1e-12)
        self.assertEqual(c.tag, "ch224x300")
        self.assertEqual(gc.resolve((2.24, 3.0)), c)


class Swap(unittest.TestCase):
    def test_full_swap_on_new_chip(self):
        tm = ge.tail_model(NC, 2, 2, keep=set(), chip=CHIP35)
        fr = ge.tooth_frequencies(0.0, n_c=NC, chip=CHIP35)
        for k, n in enumerate(ge.ERROR_FOCK):
            xi = 0.1
            T_sw = ge.swap_time(xi, n, chip=CHIP35)
            self.assertAlmostEqual(T_sw, np.pi / (2 * np.sqrt(CHIP35.chi_ge * CHIP35.chi_qr) * xi * np.sqrt(n + 1)))
            Ns = ge.n_steps(T_sw)
            t = (np.arange(Ns) + 0.5) * ge.DT
            b4 = ge.b4_from(np.zeros(Ns), xi * np.exp(1j * fr["M"][k] * t))
            rho = np.zeros((1, tm.model.d, tm.model.d), complex)
            i0 = ge.ge_index(n, E, 0, NC)
            rho[0, i0, i0] = 1.0
            R, _ = ge.propagate(tm, b4, rho, stark=False)
            i1 = ge.ge_index(n + 1, G, 1, NC)
            self.assertGreater(R[0, i1, i1].real, 0.995, f"path {n}")


class Teeth(unittest.TestCase):
    def test_lines_move(self):
        fr = ge.tooth_frequencies(0.0, chip=CHIP35)
        for k, n in enumerate(ge.ERROR_FOCK):
            self.assertLess(abs(fr["T"][k] - n * CHIP35.chi_ge), 2 * np.pi * 0.1)
            self.assertLess(abs(fr["M"][k] + n * CHIP35.chi_ge), 2 * np.pi * 0.6)


class Gradient(unittest.TestCase):
    def test_new_chip(self):
        tm = ge.tail_model(NC, 2, 2, chip=CHIP35)
        R0, T, w = _inputs()
        N = 20
        b4 = _rand_b4(N, 6.0, 0.15, 2)
        db = np.random.default_rng(3).normal(size=(N, 4))
        F, g = ge.fidelity_and_grad(tm, b4, R0, T, w)
        dd = ge.directional_derivative(tm, b4, R0, T, w, db)
        h = 1e-6
        fd = (ge.fidelity(tm, b4 + h * db, R0, T, w) - ge.fidelity(tm, b4 - h * db, R0, T, w)) / (2 * h)
        self.assertAlmostEqual(float(np.sum(g * db)) / dd, 1.0, places=10)
        self.assertAlmostEqual(fd / dd, 1.0, places=5)


class GateChip(unittest.TestCase):
    def test_shirol_gate_equals_track7_input(self):
        n_c = 10
        R_none, c_none = ge.tail_input("EsT", n_c=n_c, gate_n_c=None)
        R_sh, c_sh = ge.tail_input("EsT", n_c=n_c, gate_n_c=None, gate_chip=gc.SHIROL)
        np.testing.assert_array_equal(R_none, R_sh)
        self.assertEqual(c_none, c_sh)

    def test_new_chip_changes_gate(self):
        n_c = 10
        R_none, _ = ge.tail_input("EsT", n_c=n_c, gate_n_c=None)
        R_new, _ = ge.tail_input("EsT", n_c=n_c, gate_n_c=None, gate_chip=CHIP35)
        self.assertGreater(np.abs(R_none - R_new).max(), 1e-3)


class EstGateChip(unittest.TestCase):
    def test_system(self):
        from DGRAPE import est_gate as eg
        s0, s1 = eg.system(10), eg.system(10, chip=gc.SHIROL)
        np.testing.assert_array_equal(s0.H0, s1.H0)
        s2 = eg.system(10, chip=CHIP35)
        self.assertGreater(np.abs(s2.H0 - s0.H0).max(), 1.0)
        np.testing.assert_array_equal(s2.Hc, s0.Hc)
        np.testing.assert_array_equal(s2.target, s0.target)


class Decoding(unittest.TestCase):
    def test_known_phase(self):
        _, T, w = _inputs()
        n = np.real(np.diag(ge.ge_operators(NC, 2, 2)["n_a"]))
        th0, ph0 = 0.37, -0.81
        U = np.exp(1j * (th0 * n + ph0 * 0.5 * n * (n - 1)))
        R = U[None, :, None] * T * U.conj()[None, None, :]
        self.assertLess(ft.fidelity_from_outputs(R, T, w), 0.9)
        F, th, ph = ge.decoded_fidelity(R, T, w, NC)
        self.assertAlmostEqual(F, 1.0, places=9)
        # tracked alone cannot undo the n^2 part
        self.assertLess(ge.frame_tracked_fidelity(R, T, w, NC)[0], 0.99)


if __name__ == "__main__":
    unittest.main()
