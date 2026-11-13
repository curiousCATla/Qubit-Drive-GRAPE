#!/usr/bin/env python3
"""
test_fwm_tail_ge.py

Checks for DGRAPE/fwm_tail_ge.py, FWM tails with Gertler-style combs on transmon {g, e}
(notebooks/FWM_tails_ge.ipynb).

  * model          -- restrict_model keeps the g, e block of H_d, Hc and the jump operators; |e><f| is dropped
  * idle           -- with both combs off, the {g, e} tail equals the three-level tail projected (exact)
  * weak drive     -- a weak transmon comb and a modest mixing comb give the same F_pro on two and three
                      levels to leakage order
  * swap           -- one comb-M tooth swaps |n,e,0> -> |n+1,g,1> fully in pi / (2 g_m |xi| sqrt(n+1))
  * teeth          -- comb-T carriers at about +n chi_ge, comb-M at about -n chi_ge (exp(i Delta t)
                      convention); Gertler's amplitudes give equal swap rates and teeth that add up to the cap
  * gradient       -- W form = augmented-generator directional derivative = central differences
                      (two and three levels)
  * objective      -- TailGEObjective's pre-image gradient (both cap penalties active) equals central
                      differences, and n_jobs = 3 equals n_jobs = 1
  * gate           -- the gate alone scores the same projected as on three levels; no gate is refused
  * frame tracking -- a known cavity rotation exp(i theta n) of the ideal output is found and undone exactly

Run:
    python3 validation/test_fwm_tail_ge.py [-v]       (under a minute)
"""

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
from DGRAPE.code import process_targets                    # noqa: E402
from DGRAPE.model import E, G, Model, index                # noqa: E402

np.seterr(divide="ignore", over="ignore", invalid="ignore")
NC = 8


def _inputs(n_c=NC, n_r=2, levels=2):
    E3, T3, w = process_targets("X", n_c, n_r)
    if levels == 2:
        return ge.project_ge(E3, n_c, n_r), ge.project_ge(T3, n_c, n_r), w
    return E3, T3, w


def _rand_b4(N, eps_amp, xi_amp, seed):
    rng = np.random.default_rng(seed)
    return np.c_[eps_amp * rng.normal(size=(N, 2)), xi_amp * rng.normal(size=(N, 2))]


class ModelTests(unittest.TestCase):
    def test_restrict(self):
        full = Model(NC, 2, include_higher_order=True)
        m = ge.restrict_model(full)
        k = ge.ge_keep(NC, 2)
        self.assertEqual(m.d, 2 * NC * 2)
        np.testing.assert_array_equal(np.diag(m.H_d), np.diag(full.H_d)[k])
        np.testing.assert_array_equal(m.Hc, full.Hc[:, k][:, :, k])
        self.assertEqual([n for n, _ in m.c_ops], ["a", "r", "s_ge"])
        self.assertEqual(k[ge.ge_index(3, E, 1, NC)], index(3, E, 1, NC))
        self.assertLess(m.half_width_d, 0.2 * full.half_width_d)      # alpha left the drift


class Idle(unittest.TestCase):
    def test_two_equals_three_projected(self):
        N = 50
        b4 = np.zeros((N, 4))
        R2, T2, w = _inputs(levels=2)
        R3, T3, _ = _inputs(levels=3)
        out2, _ = ge.propagate(ge.tail_model(NC, 2, 2), b4, R2)
        out3, _ = ge.propagate(ge.tail_model(NC, 2, 3), b4, R3)
        np.testing.assert_allclose(out2, ge.project_ge(out3, NC), atol=1e-12)
        self.assertAlmostEqual(ft.fidelity_from_outputs(out2, T2, w), ft.fidelity_from_outputs(out3, T3, w),
                               places=12)


class WeakDrive(unittest.TestCase):
    def test_two_vs_three_levels(self):
        N = 100
        b4 = _rand_b4(N, 0.5, 0.03, 1)
        R2, T2, w = _inputs(levels=2)
        R3, T3, _ = _inputs(levels=3)
        F2 = ge.fidelity(ge.tail_model(NC, 2, 2), b4, R2, T2, w)
        F3 = ge.fidelity(ge.tail_model(NC, 2, 3), b4, R3, T3, w)
        # leakage to |f> ~ (2 |eps| / alpha)^2 ~ 1e-4 per unit drive; here a few 1e-5
        self.assertLess(abs(F2 - F3), 2e-4)
        self.assertGreater(abs(F2 - F3), 0.0)


class Swap(unittest.TestCase):
    def test_full_swap(self):
        tm = ge.tail_model(NC, 2, 2, keep=set())
        fr = ge.tooth_frequencies(0.0, n_c=NC)
        for k, n in enumerate(ge.ERROR_FOCK):
            xi = 0.15
            Ns = ge.n_steps(ge.swap_time(xi, n))
            t = (np.arange(Ns) + 0.5) * ge.DT
            b4 = ge.b4_from(np.zeros(Ns), xi * np.exp(1j * fr["M"][k] * t))
            rho = np.zeros((1, tm.model.d, tm.model.d), complex)
            i0 = ge.ge_index(n, E, 0, NC)
            rho[0, i0, i0] = 1.0
            R, _ = ge.propagate(tm, b4, rho, stark=False)
            i1 = ge.ge_index(n + 1, G, 1, NC)
            self.assertGreater(R[0, i1, i1].real, 0.999, f"path {n}")


class Teeth(unittest.TestCase):
    def test_carriers(self):
        fr = ge.tooth_frequencies(0.0)
        for k, n in enumerate(ge.ERROR_FOCK):
            self.assertLess(abs(fr["T"][k] - n * dev.chi_ge), 2 * np.pi * 0.05)
            self.assertLess(abs(fr["M"][k] + n * dev.chi_ge), 2 * np.pi * 0.05)

    def test_gertler_amplitudes(self):
        lam, xi = ge.gertler_amplitudes(0.1)
        rates = ge.G_M * xi * np.sqrt(np.array(ge.ERROR_FOCK) + 1)
        np.testing.assert_allclose(rates, rates[0])
        self.assertAlmostEqual(xi.sum(), 0.1)
        np.testing.assert_allclose(lam, ge.GERTLER_LAMBDA_OVER_OMEGA * rates[0])


class Gradient(unittest.TestCase):
    def _check(self, levels):
        tm = ge.tail_model(NC, 2, levels)
        R0, T, w = _inputs(levels=levels)
        N = 20
        b4 = _rand_b4(N, 3.0, 0.15, 2)
        db = np.random.default_rng(3).normal(size=(N, 4))
        F, g = ge.fidelity_and_grad(tm, b4, R0, T, w)
        dd = ge.directional_derivative(tm, b4, R0, T, w, db)
        h = 1e-6
        fd = (ge.fidelity(tm, b4 + h * db, R0, T, w) - ge.fidelity(tm, b4 - h * db, R0, T, w)) / (2 * h)
        self.assertAlmostEqual(float(np.sum(g * db)) / dd, 1.0, places=10)
        self.assertAlmostEqual(fd / dd, 1.0, places=5)

    def test_two_levels(self):
        self._check(2)

    def test_three_levels(self):
        self._check(3)


class Objective(unittest.TestCase):
    def test_preimage_gradient_and_jobs(self):
        R0, _, _ = _inputs()
        kw = dict(cap_M=0.1, n_c=NC, inputs=R0, dt=ge.DT)
        obj1 = ge.TailGEObjective(None, 0.2, n_jobs=1, **kw)
        x = 22.0 + np.random.default_rng(4).normal(size=(obj1.N, 4))      # |comb| ~ 1.2 cap: both penalties on
        c1, g1, terms = obj1.evaluate(x)
        self.assertGreater(terms["amp"], 0.0)
        dx = np.random.default_rng(5).normal(size=x.shape)
        h = 1e-5
        fd = (obj1.cost(x + h * dx) - obj1.cost(x - h * dx)) / (2 * h)
        self.assertAlmostEqual(float(g1 @ dx.ravel()) / fd, 1.0, places=5)
        obj3 = ge.TailGEObjective(None, 0.2, n_jobs=3, **kw)
        try:
            c3, g3, _ = obj3.evaluate(x)
        finally:
            obj3.close()
        self.assertAlmostEqual(c1, c3, places=13)
        np.testing.assert_allclose(g1, g3, atol=1e-12)


class Gate(unittest.TestCase):
    def test_alone_projected_equals_three_levels(self):
        u = ge.gate_pulse("EsT")
        n_c = 10
        F_ge = ge.score(np.zeros((0, 4)), u, n_c=n_c, gate_n_c=None)["F_pro"]
        F_3 = ft.score(np.zeros((0, 2)), n_c=n_c, pulse=u)["F_pro"]
        self.assertAlmostEqual(F_ge, F_3, places=12)

    def test_no_gate_refused(self):
        with self.assertRaises(ValueError):
            ge.gate_pulse(None)


class FrameTracking(unittest.TestCase):
    def test_known_rotation(self):
        _, T, w = _inputs()
        n = np.real(np.diag(ge.ge_operators(NC, 2, 2)["n_a"]))
        th0 = 0.37
        U = np.exp(1j * th0 * n)
        R = U[None, :, None] * T * U.conj()[None, None, :]
        F_raw = ft.fidelity_from_outputs(R, T, w)
        F, th = ge.frame_tracked_fidelity(R, T, w, NC)
        self.assertLess(F_raw, 0.99)
        self.assertAlmostEqual(F, 1.0, places=10)
        # exp(i theta n) on the odd code is a symmetry only up to theta -> theta + pi (n odd: global sign)
        self.assertAlmostEqual(np.cos(2 * (th + th0)), 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
