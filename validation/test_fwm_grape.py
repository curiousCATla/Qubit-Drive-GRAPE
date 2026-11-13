#!/usr/bin/env python3
"""
test_fwm_grape.py

Checks for DGRAPE/fwm.py, FWM pump tones as GRAPE controls (notebooks/fwm_grape.ipynb).

  * tones off       -- with beta = 0 the sub-stepped propagator equals lindblad_np.fidelity, and the
                       readout-traced target equals DGRAPE's when nothing drives the readout
  * order           -- CF4 converges at 4th order and the midpoint scheme at 2nd, against
                       scipy solve_ivp on the Lindblad equation with the exact carriers
  * gradient        -- the W-form gradient (beta and u) equals the augmented-generator directional
                       derivative to ~1e-13 relative and central differences to ~1e-6, with and
                       without the pump-mixing term, for CF4 and midpoint
  * occupation      -- the forbidden + transmon costate gradient equals central differences
  * penalties       -- modulus, comb and derivative penalty gradients equal central differences
  * chain           -- BetaChain is linear with adjoint to_preimage_grad: <P x, g> = <x, P' g>;
                       the whole Stage 2 objective gradient on x equals central differences
                       (and n_jobs = 3 equals n_jobs = 1)
  * readout         -- n_r = 2 equals n_r = 3 with tones on (one readout photon at a time)
  * physicality     -- trace 1, Hermitian, positive after a long run with strong tones
  * tone sets       -- (notebooks/fwm_grape.ipynb §7) the default set equals paths (1, 3, 5); the 18-tone
                       comb (paths 1..9) has carriers chi_gf apart; its W-form gradient equals the
                       augmented-generator derivative; BetaChain's adjoint holds for 18 tones and
                       for an odd tone count (the padded column pair)

Run:
    python validation/test_fwm_grape.py [-v]       (about two minutes)
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scipy.integrate import solve_ivp                      # noqa: E402

from DGRAPE import fwm                                     # noqa: E402
from DGRAPE import lindblad_np as lnp                      # noqa: E402
from DGRAPE.code import process_targets                    # noqa: E402
from DGRAPE.model import Model, operators                  # noqa: E402

DT = 0.002
NC = 7


def _setup(n_c=NC, n_r=2, N=12, seed=1, mixing=0.0, amp=0.05):
    rng = np.random.default_rng(seed)
    m = Model(n_c, n_r, include_higher_order=True)
    fw = fwm.fwm_terms(m, mixing)
    E, T, w = process_targets("X", n_c, n_r)
    T = fwm.readout_traced(T, n_c, n_r)
    u = rng.normal(0, 8, (N, 4))
    beta = amp * (rng.normal(size=(N, 6)) + 1j * rng.normal(size=(N, 6)))
    return rng, m, fw, E, T, w, u, beta


class TonesOff(unittest.TestCase):
    def test_equals_dgrape(self):
        u = np.load(os.path.join(REPO_ROOT, "pulses", "dgrape", "u_X_open_d2.npy"))[:80]
        m = Model(8, 2, include_higher_order=True)
        fw = fwm.fwm_terms(m)
        E, T, w = process_targets("X", 8)
        F0 = lnp.fidelity(m, u, E, T, w, DT)
        for S in (1, 3):
            for scheme in ("cf4", "midpoint"):
                F = fwm.fidelity(m, fw, u, np.zeros((80, 6)), E, T, w, DT, S, scheme)
                self.assertAlmostEqual(F, F0, delta=1e-12)
        Ft = fwm.fidelity(m, fw, u, np.zeros((80, 6)), E, fwm.readout_traced(T, 8), w, DT, 2)
        self.assertAlmostEqual(Ft, F0, delta=1e-12)


class Order(unittest.TestCase):
    def test_convergence_order(self):
        _, m, fw, E, _, _, u, _ = _setup(N=10, amp=0.0)
        beta = np.tile(0.05 * np.exp(2j * np.pi * np.arange(6) / 6), (10, 1))
        cops = [c for _, c in m.c_ops]
        cdc = sum(c.conj().T @ c for c in cops)

        def rhs(t, y, k):
            R = y.reshape(-1, m.d, m.d)
            H = fwm.h_eff(m, fw, m.H_d + np.tensordot(u[k], m.Hc, axes=1), beta[k], t)
            out = -1j * (H @ R - R @ H) - 0.5 * (cdc @ R + R @ cdc)
            for c in cops:
                out += c @ R @ c.conj().T
            return out.ravel()

        R_ref = E.astype(complex)
        for k in range(10):
            R_ref = solve_ivp(rhs, (k * DT, (k + 1) * DT), R_ref.ravel(), args=(k,), method="DOP853",
                              rtol=1e-12, atol=1e-14).y[:, -1].reshape(R_ref.shape)
        err = {}
        for scheme, Ss in (("midpoint", (4, 8)), ("cf4", (1, 2))):
            for S in Ss:
                R, _ = fwm.propagate(m, fw, u, beta, DT, S, E, scheme)
                err[scheme, S] = np.abs(R - R_ref).max()
        self.assertAlmostEqual(np.log2(err["midpoint", 4] / err["midpoint", 8]), 2.0, delta=0.2)
        self.assertAlmostEqual(np.log2(err["cf4", 1] / err["cf4", 2]), 4.0, delta=0.4)


class Gradient(unittest.TestCase):
    def _check(self, mixing, scheme, S):
        rng, m, fw, E, T, w, u, beta = _setup(mixing=mixing)
        spec = fwm.spec_for(m, fw, u, beta, DT, S, scheme)
        F, gb, gu = fwm.fidelity_and_grad(m, fw, u, beta, E, T, w, DT, S, scheme, spec=spec, want_u=True)
        self.assertAlmostEqual(F, fwm.fidelity(m, fw, u, beta, E, T, w, DT, S, scheme, spec=spec), delta=1e-14)
        db = rng.normal(size=beta.shape) + 1j * rng.normal(size=beta.shape)
        du = rng.normal(size=u.shape)
        lin = np.sum(gb * fwm.real_from_beta(db)) + np.sum(gu * du)
        ref = fwm.directional_derivative(m, fw, u, beta, E, T, w, DT, S, db, du, scheme, spec=spec)
        self.assertLess(abs(lin - ref), 1e-12 * max(1.0, abs(ref)) + 1e-17)
        self.assertLess(abs(lin - ref) / abs(ref), 1e-10)
        h = 1e-5
        fd = (fwm.fidelity(m, fw, u + h * du, beta + h * db, E, T, w, DT, S, scheme, spec=spec)
              - fwm.fidelity(m, fw, u - h * du, beta - h * db, E, T, w, DT, S, scheme, spec=spec)) / (2 * h)
        self.assertLess(abs(lin - fd) / abs(ref), 1e-5)

    def test_cf4(self):
        self._check(0.0, "cf4", 2)

    def test_cf4_mixing(self):
        self._check(1.0, "cf4", 2)

    def test_midpoint_mixing(self):
        self._check(2.0, "midpoint", 3)


class Occupation(unittest.TestCase):
    def test_costate(self):
        rng, m, fw, E, T, w, u, beta = _setup(mixing=1.0)
        S = 2
        spec = fwm.spec_for(m, fw, u, beta, DT, S)
        mk = np.array([lnp.forbidden_mask(m, 5), lnp.transmon_mask(m)])
        wts = np.array([1.0, 0.3])
        out = fwm.fidelity_and_grad(m, fw, u, beta, E[:2], T[:2], w[:2], DT, S, spec=spec,
                                    occ_masks=mk, occ_weights=wts, want_u=True)

        def Cw(uu, bb):
            R, hist = fwm.propagate(m, fw, uu, bb, DT, S, E[:2], store=True, spec=spec)
            return lnp._occupations(hist[1:] + [R], mk, [0, 1])

        np.testing.assert_allclose(out["C"], Cw(u, beta), rtol=1e-12)
        db = rng.normal(size=beta.shape) + 1j * rng.normal(size=beta.shape)
        du = rng.normal(size=u.shape)
        h = 1e-5
        fd = (Cw(u + h * du, beta + h * db) - Cw(u - h * du, beta - h * db)) @ wts / (2 * h)
        lin = np.sum(out["gC"] * fwm.real_from_beta(db)) + np.sum(out["gC_u"] * du)
        self.assertLess(abs(lin - fd) / abs(fd), 1e-6)


class Penalties(unittest.TestCase):
    def test_fd(self):
        rng = np.random.default_rng(3)
        _, m, fw, *_ = _setup()
        b12 = 0.12 * rng.normal(size=(30, 12))
        db = rng.normal(size=b12.shape)
        h = 1e-6
        for name, fn in (("amp", lambda b: fwm.modulus_penalty(b, 0.1)),
                         ("comb", lambda b: fwm.comb_penalty(fw, b, DT, 2, 0.1)),
                         ("deriv", fwm.derivative_penalty)):
            v, g = fn(b12)
            self.assertGreater(v, 0, name)
            fd = (fn(b12 + h * db)[0] - fn(b12 - h * db)[0]) / (2 * h)
            self.assertLess(abs(np.sum(g * db) - fd) / abs(fd), 1e-6, name)


class Chain(unittest.TestCase):
    def test_adjoint(self):
        rng = np.random.default_rng(4)
        ch = fwm.BetaChain(550, DT)
        x, g = rng.normal(size=(550, 12)), rng.normal(size=(550, 12))
        self.assertAlmostEqual(np.sum(ch.to_physical(x) * g), np.sum(x * ch.to_preimage_grad(g)), delta=1e-10)
        np.testing.assert_allclose(ch.to_physical(2 * x), 2 * ch.to_physical(x), atol=1e-14)

    def test_objective_fd(self):
        rng = np.random.default_rng(5)
        N = 40
        u = rng.normal(0, 8, (N, 4))
        x = 25 * rng.normal(size=(N, 12))          # some entries above the cap: penalties live
        kw = dict(n_c=11, S=2, ramp_ns=10.0, band=(-60.0, 60.0),
                  penalties={"forbid": 1e-3, "transmon": 1e-4})
        obj = fwm.Stage2Objective(u, DT, n_jobs=1, **kw)
        c, g, terms = obj.evaluate(x)
        for name in ("amp", "comb", "deriv", "forbid", "transmon"):
            self.assertIn(name, terms)
        dx = rng.normal(size=x.shape)
        h = 1e-4
        fd = (obj.cost(x + h * dx) - obj.cost(x - h * dx)) / (2 * h)
        self.assertLess(abs(np.dot(g, dx.ravel()) - fd) / abs(fd), 1e-5)
        obj3 = fwm.Stage2Objective(u, DT, n_jobs=3, **kw)
        try:
            c3, g3, _ = obj3.evaluate(x)
        finally:
            obj3.close()
        self.assertAlmostEqual(c, c3, delta=1e-13)
        np.testing.assert_allclose(g, g3, atol=1e-12)


class Readout(unittest.TestCase):
    def test_nr2_equals_nr3(self):
        # The d2 gate with Shirol's Table II tones left on: <n_r> stays ~3e-4, so a second
        # readout photon is negligible. (With strong constant tones, |beta| = 0.05 on all six,
        # n_r = 2 vs 3 differ by ~2e-5 in F: held-out condition 'nr3' checks trained pulses.)
        u = np.load(os.path.join(REPO_ROOT, "pulses", "dgrape", "u_X_open_d2.npy"))
        beta = np.tile(fwm.BETA_TABLE2.astype(complex), (u.shape[0], 1))
        F = [fwm.score(u, beta, DT, n_c=NC, n_r=nr, S=2)["F_pro"] for nr in (2, 3)]
        self.assertAlmostEqual(F[0], F[1], delta=1e-6)


class Physicality(unittest.TestCase):
    def test_state(self):
        rng = np.random.default_rng(7)
        N = 200
        m = Model(NC, 2, include_higher_order=True)
        fw = fwm.fwm_terms(m, 1.0)
        u = rng.normal(0, 8, (N, 4))
        beta = np.tile(0.1 * np.exp(2j * np.pi * rng.random(6)), (N, 1))
        psi = rng.normal(size=m.d) + 1j * rng.normal(size=m.d)
        rho = np.outer(psi, psi.conj()) / np.vdot(psi, psi).real
        R, _ = fwm.propagate(m, fw, u, beta, DT, 2, rho[None])
        R = R[0]
        self.assertAlmostEqual(np.trace(R).real, 1.0, delta=1e-12)
        self.assertLess(np.abs(R - R.conj().T).max(), 1e-13)
        self.assertGreater(np.linalg.eigvalsh(0.5 * (R + R.conj().T)).min(), -1e-12)


class ToneSets(unittest.TestCase):
    def test_default_unchanged(self):
        m = Model(NC, 2, include_higher_order=True)
        a, b = fwm.fwm_terms(m), fwm.fwm_terms(m, paths=(1, 3, 5))
        np.testing.assert_array_equal(a.Delta, b.Delta)
        np.testing.assert_array_equal(a.comb, b.comb)
        self.assertEqual(a.n_tones, 6)
        self.assertEqual(a.tones, fwm.TONES)

    def test_carriers_chi_gf_apart(self):
        # Consecutive teeth of each comb differ by chi_gf; Kerr and chi' shrink the step by
        # ~1.5 kHz per photon (2.070 -> 2.040 MHz over the comb).
        m = Model(12, 2, include_higher_order=True)
        fw = fwm.fwm_terms(m, paths=range(1, 10))
        self.assertEqual(fw.n_tones, 18)
        D = fw.Delta / (2 * np.pi)
        chi_gf = fwm.dev.chi_gf / (2 * np.pi)
        np.testing.assert_allclose(np.diff(D[:9]), chi_gf, atol=0.04)
        np.testing.assert_allclose(np.diff(D[9:]), -chi_gf, atol=0.04)
        with self.assertRaises(ValueError):
            fwm.fwm_terms(Model(10, 2), paths=range(1, 10))

    def test_gradient_18_tones(self):
        rng = np.random.default_rng(11)
        n_c, N, S = 12, 4, 1
        m = Model(n_c, 2, include_higher_order=True)
        fw = fwm.fwm_terms(m, 1.0, paths=range(1, 10))
        E, T, w = process_targets("X", n_c, 2)
        T = fwm.readout_traced(T, n_c, 2)
        u = rng.normal(0, 8, (N, 4))
        beta = 0.03 * (rng.normal(size=(N, 18)) + 1j * rng.normal(size=(N, 18)))
        spec = fwm.spec_for(m, fw, u, beta, DT, S)
        F, gF, gU = fwm.fidelity_and_grad(m, fw, u, beta, E[:2], T[:2], w[:2], DT, S, spec=spec,
                                          want_u=True)
        self.assertEqual(gF.shape, (N, 36))
        db = rng.normal(size=beta.shape) + 1j * rng.normal(size=beta.shape)
        du = rng.normal(size=u.shape)
        ref = fwm.directional_derivative(m, fw, u, beta, E[:2], T[:2], w[:2], DT, S, db, du, spec=spec)
        lin = np.sum(gF * fwm.real_from_beta(db)) + np.sum(gU * du)
        self.assertLess(abs(lin - ref) / abs(ref), 1e-12)

    def test_chain_adjoint(self):
        rng = np.random.default_rng(12)
        for K in (18, 5):
            ch = fwm.BetaChain(550, DT, fwm.BETA_BAND_FULL, n_tones=K)
            x, g = rng.normal(size=(550, 2 * K)), rng.normal(size=(550, 2 * K))
            Px = ch.to_physical(x)
            self.assertEqual(Px.shape, (550, 2 * K))
            self.assertAlmostEqual(np.sum(Px * g), np.sum(x * ch.to_preimage_grad(g)), delta=1e-10)


if __name__ == "__main__":
    unittest.main()
