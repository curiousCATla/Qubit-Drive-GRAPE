#!/usr/bin/env python3
"""
test_fwm_repeat.py

Checks for DGRAPE/fwm_repeat.py, repeated X gates with FWM tails (FWM_tails.ipynb §9).

  * one gate       -- N = 1 of after_n_gates (with and without the tail) and of pairs equals fwm_tail.score
                      for the same gate and tail (the gate there runs at n_r = 1 and is embedded)
  * frame          -- a zero tail on a random state with |f> content, handed back to the DGRAPE frame,
                      equals the DGRAPE-frame idle of the same length; so does gate + zero tail + gate
  * pairs          -- pairs with a zero eta equals after_n_gates of the gate followed by its idle, step by step
  * teeth          -- teeth_memory over N periods equals one propagation of the whole waveform
  * targets        -- X^N alternates X, I; an idle "gate" targets I
  * kerr decode    -- closed waits decoded with kerr_decode score 1; on the code space kerr_decode is a cavity
                      frame shift by 2.5 K times a logical Z by 2 K t; the plain rows do not change

Run:
    python3 validation/test_fwm_repeat.py [-v]       (about ten seconds)
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from DGRAPE import fwm                                     # noqa: E402
from DGRAPE import fwm_repeat as rp                        # noqa: E402
from DGRAPE import fwm_tail as ft                          # noqa: E402
from DGRAPE import lindblad_np as lnp                      # noqa: E402

np.seterr(divide="ignore", over="ignore", invalid="ignore")
DT = ft.DT
NC = 8


def _rand_eta(N, amp, seed):
    rng = np.random.default_rng(seed)
    return amp * (rng.normal(size=(N, 2)) + 1j * rng.normal(size=(N, 2)))


def _rand_pulse(N, amp, seed):
    return amp * np.random.default_rng(seed).normal(size=(N, 4))


def _rand_states(d, seed, k=3):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(k, d, d)) + 1j * rng.normal(size=(k, d, d))
    R = A @ np.conj(np.swapaxes(A, 1, 2))
    return R / np.trace(R, axis1=1, axis2=2).real[:, None, None]


class OneGate(unittest.TestCase):
    def test_n1_equals_score(self):
        u, eta = _rand_pulse(40, 3.0, 1), _rand_eta(30, 0.05, 2)
        ref_tail = ft.score(eta, n_c=NC, pulse=u)["F_pro"]
        ref_alone = ft.score(np.zeros((0, 2)), n_c=NC, pulse=u)["F_pro"]
        rows = rp.after_n_gates(u, eta, 1, n_c=NC)
        by = {r["tail"]: r["F_pro"] for r in rows}
        self.assertAlmostEqual(by["tail"], ref_tail, delta=1e-11)
        self.assertAlmostEqual(by["none"], ref_alone, delta=1e-11)
        self.assertAlmostEqual(rp.pairs(u, eta, 1, n_c=NC)[0]["F_pro"], ref_tail, delta=1e-11)


class Frame(unittest.TestCase):
    def test_zero_tail_is_dgrape_idle(self):
        K = 37
        gm, tm = rp._models(NC, 2, None, True, False, False)
        R = _rand_states(gm.d, 3)
        a = rp.tail_step(tm, np.zeros((K, 2)), R)
        b = rp.gate_step(gm, np.zeros((K, 4)), R)
        self.assertLess(np.abs(a - b).max(), 1e-11)

    def test_gate_zero_tail_gate(self):
        u = _rand_pulse(30, 4.0, 4)
        gm, tm = rp._models(NC, 2, None, True, False, False)
        R0 = rp.inputs(NC)
        a = rp.gate_step(gm, u, rp.tail_step(tm, np.zeros((25, 2)), rp.gate_step(gm, u, R0)))
        b = rp.gate_step(gm, np.vstack([u, np.zeros((25, 4)), u]), R0)
        self.assertLess(np.abs(a - b).max(), 1e-11)

    def test_frame_matters(self):
        """Without the hand-off the same check fails, so the test above can see a wrong frame."""
        K = 37
        gm, tm = rp._models(NC, 2, None, True, False, False)
        R = _rand_states(gm.d, 5)
        a, _ = ft.propagate(tm, np.zeros((K, 2)), R)
        b = rp.gate_step(gm, np.zeros((K, 4)), R)
        self.assertGreater(np.abs(a - b).max(), 1e-3)


class Pairs(unittest.TestCase):
    def test_zero_eta_pairs(self):
        u = _rand_pulse(30, 3.0, 6)
        rows = rp.pairs(u, np.zeros((20, 2)), 3, n_c=NC)
        ref = rp.after_n_gates(np.vstack([u, np.zeros((20, 4))]), np.zeros((0, 2)), 3, n_c=NC)
        for r, q in zip(rows, ref):
            self.assertAlmostEqual(r["F_pro"], q["F_pro"], delta=1e-11)


class Teeth(unittest.TestCase):
    def test_segments(self):
        rows = rp.teeth_memory(3, 0.04, n_c=NC)
        tm = ft.tail_model(NC, 2)
        eta = ft.eta_from_teeth(fwm.beta_table2(ft.REPAIR_PATHS), 3 * ft.n_steps(0.04), DT, NC)
        R, _ = ft.propagate(tm, eta, rp.inputs(NC))
        self.assertAlmostEqual(rows[-1]["F_pro"], rp.score_state(R, NC, 2, "I")["F_pro"], delta=1e-11)


class KerrDecode(unittest.TestCase):
    def test_closed_idle_decoded_is_perfect(self):
        idle = np.zeros((300, 4))
        rows = rp.after_n_gates(idle, np.zeros((0, 2)), 3, n_c=NC, gate="I", keep=set(), with_kerr_decode=True)
        plain = [r["F_pro"] for r in rows if r["decode"] == "none"]
        kerr = [r["F_pro"] for r in rows if r["decode"] == "kerr"]
        self.assertLess(plain[-1], 1 - 1e-4)                 # the phase is there without the decoding
        for F in kerr:
            self.assertAlmostEqual(F, 1.0, delta=1e-10)

    def test_frame_shift_and_logical_z(self):
        """On the code's Fock levels |1>, |3>, |5> (transmon g, readout empty) the decoding is a cavity frame
        shift exp(-i 2.5 K n t) times a logical Z that adds 2 K t to |1_L> = |3>, up to a global phase."""
        from AQEC import device as dev
        from DGRAPE.model import index
        tau = 7.3
        d = rp.operators(NC, 2)["n_a"].shape[0]
        c = rp.kerr_decode(np.ones((1, d, d), complex), NC, 2, tau)[0][:, 0]   # c_i c_0', c_0 = 1 (vacuum)
        idx = [index(n, 0, 0, NC, 2) for n in (1, 3, 5)]
        n = np.array([1.0, 3.0, 5.0])
        frame_z = np.exp(-2.5j * dev.K_a * n * tau) * np.exp([0.0, 2j * dev.K_a * tau, 0.0])
        ratio = c[idx] / frame_z
        self.assertLess(np.abs(ratio - ratio[0]).max(), 1e-12)
        self.assertAlmostEqual(np.angle(ratio[0]), np.angle(np.exp(2.5j * dev.K_a * tau)), delta=1e-12)

    def test_plain_rows_unchanged(self):
        u, eta = _rand_pulse(30, 2.0, 7), _rand_eta(20, 0.05, 8)
        a = rp.after_n_gates(u, eta, 2, n_c=NC, gate="I")
        b = [r for r in rp.after_n_gates(u, eta, 2, n_c=NC, gate="I", with_kerr_decode=True)
             if r["decode"] == "none"]
        self.assertEqual([r["F_pro"] for r in a], [r["F_pro"] for r in b])
        R = _rand_states(rp.inputs(NC).shape[1], 9)
        self.assertIs(rp.kerr_decode(R, NC, 2, 0.0), R)
        with self.assertRaises(ValueError):
            rp.after_n_gates(u, eta, 1, n_c=NC, gate="X", with_kerr_decode=True)


class Targets(unittest.TestCase):
    def test_targets(self):
        self.assertEqual([rp.target_of(n) for n in range(1, 5)], ["X", "I", "X", "I"])
        self.assertEqual(rp.target_of(3, "I"), "I")


if __name__ == "__main__":
    unittest.main()
