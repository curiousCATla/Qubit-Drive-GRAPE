#!/usr/bin/env python3
"""
test_dgrape.py

Checks for DGRAPE/, exact open-system GRAPE on the Shirol chip (notebooks/dissipation_grape.ipynb).

  * operators       -- at n_r = 2 the model's operators, decay jump operators and drift
                       equal AQEC/master_equation.py's (plus the -alpha|f><f| term)
  * Taylor step     -- the matrix-level Taylor action equals scipy expm of the full
                       d^2 x d^2 Liouvillian (AQEC.master_equation.liouvillian), all channels on
  * adjoint map     -- <X, L rho> = <L'X, rho> in the Hilbert-Schmidt pairing
  * physicality     -- trace, Hermiticity and positivity of a propagated mixed state
  * closed limit    -- with every rate 0, F_pro equals |Tr(U' P V P)|^2/4 from unitary propagation
  * readout         -- decay-only dynamics never populates m = 1; n_r = 2 equals n_r = 1
  * gradient        -- hand-derived adjoint (W form and augmented form) vs JAX autodiff vs
                       central differences
  * constraint      -- the jnp constraint chain equals core.ramp.make_constraint_chain
  * forbidden       -- the forbidden-state cost C = sum_k Tr[P_F rho_b(k)] and its gradient
                       from the shared backward pass: np == JAX == central differences; the
                       F-only path is bit-identical with the mask off; full objective with
                       the term live == central differences; skipped where P_F is empty,
                       refused by train.py
  * transmon        -- the transmon-excitation cost C_T = sum_k Tr[P_T rho_b(k)]: the mask is
                       |e><e| + |f><f|; with P_F in one weighted costate, both values and the
                       weighted gradient are np == JAX == central differences; full objective
                       with both terms == central differences, and n_jobs-independent; off
                       by default
  * parallel        -- n_jobs = 3, 5, 9 (truncations split by process input) equal
                       n_jobs = 1 (cost, gradient, report); the split plan
  * recipe          -- DGRAPE/recipe.py equals main.py's production defaults (except
                       trunc_list [12, 14, 16], pinned as DIFFERENT, and the DGRAPE-only
                       forbid term); the objective defaults to it; bad penalty keys / norms
                       raise; Eq. 24 off at one n_c
  * full cost       -- band + ramp + deriv + amp (above cap) + Eq. 24 + all channels:
                       np == JAX, and both == central differences on the pre-image x
  * smoke           -- smooth seeded cold start; deramp warm start round-trips; infeasible
                       starts refused; 5 L-BFGS-B iterations at the recipe lower the cost
                       (nothing saved)
  * qutip           -- qutip.mesolve with step-function coefficients agrees

Run:
    python3 validation/test_dgrape.py [-v]
"""

import math
import os
import sys
import unittest

import numpy as np
import scipy.linalg as sla

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from AQEC import device as dev                                   # noqa: E402
from AQEC import master_equation as me                           # noqa: E402
from DGRAPE import lindblad_np as lnp                            # noqa: E402
from DGRAPE.code import embedded_basis, process_targets, IDEAL_LOGICAL_U  # noqa: E402
from DGRAPE.model import Model, operators, collapse_ops, drift_hamiltonian  # noqa: E402
from DGRAPE.objective import OpenGrapeObjective                  # noqa: E402

_QUIET = dict(divide="ignore", over="ignore", invalid="ignore")
DT = 0.002
ALL = dict(decay=True, dephasing=True, heating=True)


def random_pulse(N, scale=15.0, seed=0):
    return np.random.default_rng(seed).normal(0.0, scale, (N, 4))


def random_density(d, seed=0, rank=3):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, rank)) + 1j * rng.normal(size=(d, rank))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def superop_propagate(model, u, R0, dt):
    """Reference: full Liouvillian (AQEC.master_equation.liouvillian) and scipy expm."""
    d = model.d
    cs = [c for _, c in model.c_ops]
    vs = [R.reshape(-1, order="F") for R in R0]
    with np.errstate(**_QUIET):
        for k in range(u.shape[0]):
            H = model.H_d + np.tensordot(u[k], model.Hc, axes=1)
            P = sla.expm(me.liouvillian(H, cs) * dt)
            vs = [P @ v for v in vs]
    return np.stack([v.reshape(d, d, order="F") for v in vs])


class OperatorsTest(unittest.TestCase):
    def test_match_aqec(self):
        n_c = 7
        mine, ref = operators(n_c, 2), me.build_operators(n_c)
        for k in ref:
            np.testing.assert_array_equal(mine[k], ref[k], err_msg=k)
        ours = [c for _, c in collapse_ops(n_c, 2)]
        theirs = me.collapse_operators(n_c, "split", heating="none", dephasing=False)
        self.assertEqual(len(ours), len(theirs))
        for a, b in zip(ours, theirs):
            np.testing.assert_allclose(a, b, atol=0)
        H = drift_hamiltonian(n_c, 2)
        H_ref = me.dispersive_hamiltonian(n_c) - dev.alpha_q * ref["Pf"]
        np.testing.assert_allclose(H, H_ref, atol=1e-12)
        H1 = drift_hamiltonian(n_c, 2, include_higher_order=True)
        np.testing.assert_allclose(H1 - H, me.higher_order_shifts(n_c), atol=1e-12)

    def test_rates(self):
        from DGRAPE.model import rates
        R = rates()
        # AQEC notebook Table 5: heating 0.35 ms^-1 (transmon), ~0.04 ms^-1 (storage)
        self.assertAlmostEqual(R["up_q"] * 1e3, 0.35, delta=0.01)
        self.assertAlmostEqual(R["up_a"] * 1e3, 0.044, delta=0.002)
        for k in ("phi_ge", "phi_gf", "phi_a"):
            self.assertGreater(R[k], 0.0, k)


class TaylorStepTest(unittest.TestCase):
    def test_spec_meets_bound(self):
        for theta in (0.01, 0.5, 2.0, 7.3, 40.0):
            s, m = lnp.taylor_spec(theta)
            th = theta / s
            self.assertLessEqual((m + 1) * math.log(th) - math.lgamma(m + 2) + th,
                                 math.log(lnp.TAYLOR_TOL) + 1e-12)

    def test_against_superoperator_expm(self):
        for kw in (dict(), ALL):
            model = Model(7, **kw)
            u = random_pulse(12, seed=1)
            E, _, _ = process_targets("H", 7)
            R = lnp.propagate(model, u, E, DT)
            ref = superop_propagate(model, u, E, DT)
            self.assertLess(np.abs(R - ref).max(), 1e-12, kw)

    def test_adjoint_pairing(self):
        model = Model(7, **ALL)
        rng = np.random.default_rng(3)
        d = model.d
        X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        R = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        Heff = lnp.h_eff(model, rng.normal(0, 10, 4))
        lhs = np.vdot(X, lnp.apply_L(model, Heff, R))
        rhs = np.vdot(lnp.apply_Ldag(model, Heff, X), R)
        self.assertLess(abs(lhs - rhs) / abs(lhs), 1e-13)

    def test_physicality(self):
        model = Model(7, **ALL)
        rho0 = random_density(model.d, seed=4)
        rho = lnp.propagate(model, random_pulse(60, seed=5), rho0[None], DT)[0]
        self.assertAlmostEqual(np.trace(rho).real, 1.0, places=12)
        self.assertLess(abs(np.trace(rho).imag), 1e-13)
        self.assertLess(np.abs(rho - rho.conj().T).max(), 1e-13)
        self.assertGreater(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).min(), -1e-12)


class ClosedLimitTest(unittest.TestCase):
    def test_unitary_process_fidelity(self):
        n_c = 7
        model = Model(n_c, rate_scale=0.0)
        self.assertEqual(len(model.c_ops), 0)
        u = random_pulse(30, scale=8.0, seed=6)
        for gate in IDEAL_LOGICAL_U:
            E, T, w = process_targets(gate, n_c)
            F = lnp.fidelity(model, u, E, T, w, DT)
            V = np.eye(model.d, dtype=complex)
            with np.errstate(**_QUIET):
                for k in range(u.shape[0]):
                    V = sla.expm(-1j * DT * (model.H_d + np.tensordot(u[k], model.Hc, axes=1))) @ V
                B = embedded_basis(n_c)
                M = B.conj().T @ V @ B
            ref = abs(np.trace(IDEAL_LOGICAL_U[gate].conj().T @ M)) ** 2 / 4.0
            self.assertAlmostEqual(F, ref, places=12, msg=gate)

    def test_identity_pulse_on_identity_gate(self):
        # Zero drive: the code words only pick up dispersive phases with the transmon in g
        # (zero), so the closed-system identity is perfect.
        model = Model(7, rate_scale=0.0)
        E, T, w = process_targets("I", 7)
        self.assertAlmostEqual(lnp.fidelity(model, np.zeros((20, 4)), E, T, w, DT), 1.0,
                               places=13)


class ReadoutTest(unittest.TestCase):
    def test_readout_stays_empty_and_matches_n_r_1(self):
        n_c = 7
        u = random_pulse(25, seed=7)
        m2, m1 = Model(n_c, n_r=2), Model(n_c, n_r=1)
        E2, T2, w = process_targets("X", n_c, 2)
        E1, T1, _ = process_targets("X", n_c, 1)
        R2 = lnp.propagate(m2, u, E2, DT)
        idx_m1 = np.arange(m2.d) % 2 == 1
        self.assertEqual(np.abs(R2[:, idx_m1, :]).max(), 0.0)
        self.assertEqual(np.abs(R2[:, :, idx_m1]).max(), 0.0)
        F2 = lnp.fidelity(m2, u, E2, T2, w, DT)
        F1 = lnp.fidelity(m1, u, E1, T1, w, DT)
        self.assertAlmostEqual(F2, F1, places=13)


class GradientTest(unittest.TestCase):
    def test_np_vs_jax_vs_fd(self):
        from DGRAPE import lindblad_jax as ljx
        n_c, N = 7, 15
        for kw in (dict(), ALL):
            model = Model(n_c, **kw)
            E, T, w = process_targets("H", n_c)
            u = random_pulse(N, seed=8)
            spec = lnp.spec_for_pulse(model, u, DT)
            F, g = lnp.fidelity_and_grad(model, u, E, T, w, DT, spec)
            _, fg = ljx.make_fidelity_fns(model, E, T, w, DT)
            Fj, gj = fg(u, *spec)
            self.assertAlmostEqual(F, float(Fj), places=13)
            np.testing.assert_allclose(g, np.asarray(gj), rtol=1e-8,
                                       atol=1e-10 * np.abs(g).max())
            h = 1e-5
            for k, j in [(0, 0), (4, 1), (7, 2), (14, 3), (10, 0)]:
                up, um = u.copy(), u.copy()
                up[k, j] += h
                um[k, j] -= h
                fd = (lnp.fidelity(model, up, E, T, w, DT, spec)
                      - lnp.fidelity(model, um, E, T, w, DT, spec)) / (2 * h)
                self.assertLess(abs(fd - g[k, j]), 1e-8 * max(1.0, abs(g).max()),
                                f"{kw} ({k},{j}): adjoint {g[k, j]} fd {fd}")

    def test_w_form_equals_augmented(self):
        # Two independent exact derivations of the same polynomial's derivative; s = 3
        # forces the multi-substep path of the W form.
        model = Model(7, **ALL)
        E, T, w = process_targets("Y", 7)
        u = random_pulse(12, seed=13)
        for spec in (None, (3, 20)):
            F1, g1 = lnp.fidelity_and_grad(model, u, E, T, w, DT, spec)
            F2, g2 = lnp.fidelity_and_grad_augmented(model, u, E, T, w, DT, spec)
            self.assertEqual(F1, F2)
            np.testing.assert_allclose(g1, g2, rtol=1e-9, atol=1e-11 * np.abs(g2).max())

    def test_objective_backends_agree(self):
        x = random_pulse(40, seed=9)
        kw = dict(cav_band=(-40, 40), tra_band=(-60, 60), ramp_ns=10.0,
                  penalties={"amp_max": 20.0, "amp": 1e-3, "deriv": 1e-5, "disc": 0.5})
        c1, g1, _ = OpenGrapeObjective("T", 40, DT, [7, 8], "np", **kw).evaluate(x)
        c2, g2, _ = OpenGrapeObjective("T", 40, DT, [7, 8], "jax", **kw).evaluate(x)
        self.assertAlmostEqual(c1, c2, places=12)
        np.testing.assert_allclose(g1, g2, rtol=1e-8, atol=1e-12 * np.abs(g1).max())


class RecipeObjectiveTest(unittest.TestCase):
    """The notebooks/grape_experiments.ipynb production recipe, as threaded through DGRAPE."""

    def test_recipe_matches_main_py(self):
        import main
        from DGRAPE.recipe import RECIPE
        a = main.build_arg_parser().parse_args(["--gate", "X"])
        self.assertEqual(RECIPE["cav_band"], main.resolve_band(a.cav_band))
        self.assertEqual(RECIPE["tra_band"], main.resolve_band(a.tra_band))
        self.assertEqual(RECIPE["ramp_ns"], a.ramp_ns)
        from DGRAPE.recipe import HEERES_PENALTY_KEYS
        self.assertEqual({k: RECIPE["penalties"][k] for k in HEERES_PENALTY_KEYS},
                         {"deriv": a.lambda_deriv, "amp": a.lambda_amp,
                          "amp_max": a.amp_max, "disc": a.lambda_disc})
        self.assertEqual(set(RECIPE["penalties"]) - set(HEERES_PENALTY_KEYS),
                         {"forbid", "transmon"})
        self.assertEqual(RECIPE["penalties"]["transmon"], 0.0)   # off: d2 stays reproducible
        self.assertEqual(RECIPE["n_jobs"], a.n_jobs)
        self.assertEqual(RECIPE["amp_norm"], a.amp_norm)
        self.assertEqual(RECIPE["hard_amp_limit"], a.hard_amp_limit)
        self.assertIsNone(a.warm_start)      # => core's smooth cold start, our defaults below
        import inspect
        from core.optimizer import optimize_multi_state_pulse
        sig = inspect.signature(optimize_multi_state_pulse).parameters
        self.assertEqual(RECIPE["warm_start_amp"], sig["warm_start_amp"].default)
        self.assertEqual(RECIPE["warm_start_cutoff_frac"], sig["warm_start_cutoff_frac"].default)

    def test_defaults_are_the_recipe(self):
        from DGRAPE.recipe import RECIPE
        obj = OpenGrapeObjective("X", 60, DT, 7)
        cfg = obj.config()
        for k in ("cav_band", "tra_band", "ramp_ns", "penalties", "amp_norm",
                  "forbid_fock_min"):
            self.assertEqual(cfg[k], RECIPE[k], k)
        self.assertEqual(cfg["n_jobs"], 3)          # one truncation, split by process input
        self.assertEqual(obj.split, {0})
        off = OpenGrapeObjective("X", 60, DT, 7, cav_band=None, tra_band=None, ramp_ns=0)
        x = random_pulse(60, seed=20)
        np.testing.assert_array_equal(off.to_physical(x), x)

    def test_default_truncation_list(self):
        # [12, 14, 16] is chosen for this chip ([10, 12] was falsified, notebooks/dissipation_grape.ipynb
        # §10), NOT the Heeres [22, 24, 26]: it must not be pinned to main.py, the code needs
        # n_c >= MIN_N_C with >= 2 entries for Eq. 24, and every entry must hold P_F.
        import main
        from AQEC.odd_kitten import MIN_N_C
        from DGRAPE.recipe import RECIPE
        from DGRAPE.train import build_arg_parser
        self.assertEqual(RECIPE["trunc_list"], [12, 14, 16])
        self.assertNotEqual(RECIPE["trunc_list"],
                            main.build_arg_parser().parse_args(["--gate", "X"]).trunc_list)
        self.assertGreaterEqual(min(RECIPE["trunc_list"]), MIN_N_C)
        self.assertGreaterEqual(len(RECIPE["trunc_list"]), 2)
        self.assertGreater(min(RECIPE["trunc_list"]), RECIPE["forbid_fock_min"])
        self.assertEqual(OpenGrapeObjective("X", 60, DT).trunc, [12, 14, 16])
        self.assertEqual(OpenGrapeObjective("X", 60, DT, n_c=7).trunc, [7])
        a = build_arg_parser().parse_args(["--gate", "X", "--duration-us", "1.1", "--dt", "0.002"])
        self.assertEqual(a.n_c, [12, 14, 16])
        self.assertEqual(a.lambda_forbid, RECIPE["penalties"]["forbid"])
        self.assertEqual(a.n_jobs, RECIPE["n_jobs"])

    def test_penalty_validation(self):
        with self.assertRaises(ValueError):
            OpenGrapeObjective("X", 60, DT, 7, penalties={"boundary": 1.0})
        with self.assertRaises(ValueError):
            OpenGrapeObjective("X", 60, DT, 7, amp_norm="l2")
        for norm in ("modulus", "quadrature"):
            c, _, terms = OpenGrapeObjective("X", 60, DT, 7, amp_norm=norm,
                                             penalties={"amp_max": 1.0}).evaluate(
                random_pulse(60, seed=21))
            self.assertGreater(terms["amp"], 0.0, norm)

    def test_disc_vanishes_for_one_truncation(self):
        _, _, terms = OpenGrapeObjective("X", 60, DT, 7).evaluate(random_pulse(60, seed=22))
        self.assertNotIn("disc", terms)

    def test_full_cost_gradient_fd(self):
        # Every term live: band, ramp, deriv, modulus amp ABOVE its cap, Eq. 24 over two
        # truncations (weight raised so it is not negligible), all Lindblad channels.
        N = 60
        kw = dict(penalties={"deriv": 1e-3, "amp": 1e-3, "amp_max": 3.0, "disc": 1e3},
                  dephasing=True, heating=True)
        # Identity target at scale 20: F_pro ~ 0.31 and the pulse reaches the n_c = 7 wall,
        # so F(7) - F(8) ~ 1e-3 and the discrepancy term is genuinely exercised.
        x = random_pulse(N, scale=20.0, seed=23)
        objs = {b: OpenGrapeObjective("I", N, DT, [7, 8], b, **kw) for b in ("np", "jax")}
        c, g, terms = objs["np"].evaluate(x)
        for k in ("disc", "deriv", "amp"):
            self.assertGreater(terms[k], 1e-6, k)
        cj, gj, _ = objs["jax"].evaluate(x)
        self.assertAlmostEqual(c, cj, places=12)
        np.testing.assert_allclose(g, gj, rtol=1e-8, atol=1e-11 * np.abs(g).max())
        g = g.reshape(N, 4)
        h = 1e-5
        for k, j in [(0, 0), (11, 1), (30, 2), (45, 3), (59, 1), (24, 0)]:
            xp, xm = x.copy(), x.copy()
            xp[k, j] += h
            xm[k, j] -= h
            fd = (objs["np"].cost(xp) - objs["np"].cost(xm)) / (2 * h)
            self.assertLess(abs(fd - g[k, j]), 1e-7 * np.abs(g).max(),
                            f"({k},{j}): grad {g[k, j]} fd {fd}")


class ForbiddenStateTest(unittest.TestCase):
    """Heeres' forbidden-state cost C = sum_{k=1}^{N} sum_{b=0_L,1_L} Tr[P_F rho_b(k)]."""

    def test_mask_is_storage_fock_projector(self):
        model = Model(12)
        mask = lnp.forbidden_mask(model, 10)
        n_a = np.real(np.diag(operators(12)["n_a"]))
        np.testing.assert_array_equal(mask, n_a >= 10 - 1e-9)
        self.assertEqual(mask.sum(), 2 * 3 * 2)

    def test_gradient_np_jax_fd(self):
        from DGRAPE import lindblad_jax as ljx
        model = Model(12, **ALL)
        E, T, w = process_targets("X", 12)
        u = random_pulse(20, scale=15.0, seed=31)
        mask = lnp.forbidden_mask(model, 6)          # low threshold so C is not ~0
        for spec in (None, (3, 20)):                 # s = 3: multi-substep source handling
            spec = lnp.spec_for_pulse(model, u, DT) if spec is None else spec
            F0, g0 = lnp.fidelity_and_grad(model, u, E, T, w, DT, spec)
            F, g, C, gC = lnp.fidelity_and_grad(model, u, E, T, w, DT, spec, forbid_mask=mask)
            self.assertEqual(F, F0)                  # F part untouched by the extra costate
            np.testing.assert_array_equal(g, g0)
            self.assertGreater(C, 1e-3)
            self.assertAlmostEqual(C, lnp.forbidden_population(model, u, E, DT, spec, mask)[0],
                                   places=13)
            _, fg = ljx.make_fidelity_fns(model, E, T, w, DT, forbid_mask=mask)
            Fj, gj, Cj, gCj = fg(u, *spec)
            self.assertAlmostEqual(C, float(Cj), places=12)
            np.testing.assert_allclose(gC, np.asarray(gCj), rtol=1e-8,
                                       atol=1e-11 * np.abs(gC).max())
            np.testing.assert_allclose(g, np.asarray(gj), rtol=1e-8,
                                       atol=1e-11 * np.abs(g).max())
        h = 1e-5
        for k, j in [(0, 0), (5, 1), (10, 2), (15, 3), (19, 0)]:
            up, um = u.copy(), u.copy()
            up[k, j] += h
            um[k, j] -= h
            fd = (lnp.forbidden_population(model, up, E, DT, spec, mask)[0]
                  - lnp.forbidden_population(model, um, E, DT, spec, mask)[0]) / (2 * h)
            self.assertLess(abs(fd - gC[k, j]), 1e-8 * np.abs(gC).max(),
                            f"({k},{j}): adjoint {gC[k, j]} fd {fd}")

    def test_objective_with_forbid_fd(self):
        # fock_min = 6, so the term is live on both n_c = 7 and 8; its
        # weight is raised so it is a visible share of the cost. np == JAX == central differences.
        N = 40
        kw = dict(penalties={"forbid": 0.1}, forbid_fock_min=6, ramp_ns=10.0)
        x = random_pulse(N, scale=20.0, seed=32)
        objs = {b: OpenGrapeObjective("I", N, DT, [7, 8], b, **kw) for b in ("np", "jax")}
        c, g, terms = objs["np"].evaluate(x)
        self.assertGreater(terms["forbid"], 1e-4)
        self.assertEqual(len(terms["C_forbid"]), 2)
        cj, gj, _ = objs["jax"].evaluate(x)
        self.assertAlmostEqual(c, cj, places=12)
        np.testing.assert_allclose(g, gj, rtol=1e-8, atol=1e-11 * np.abs(g).max())
        g = g.reshape(N, 4)
        h = 1e-5
        for k, j in [(3, 0), (12, 1), (20, 2), (33, 3)]:
            xp, xm = x.copy(), x.copy()
            xp[k, j] += h
            xm[k, j] -= h
            fd = (objs["np"].cost(xp) - objs["np"].cost(xm)) / (2 * h)
            self.assertLess(abs(fd - g[k, j]), 1e-7 * np.abs(g).max(),
                            f"({k},{j}): grad {g[k, j]} fd {fd}")

    def test_skipped_where_projector_is_empty(self):
        # n_c = 7 has no Fock >= 10: C is exactly 0, so the term is skipped, not computed.
        obj = OpenGrapeObjective("X", 30, DT, 7, ramp_ns=10.0)
        self.assertEqual(obj.masks, [None])
        _, _, terms = obj.evaluate(random_pulse(30, seed=33))
        self.assertNotIn("forbid", terms)
        self.assertEqual(obj.report(random_pulse(30, seed=33))[0]["C_forbid"], 0.0)


class TransmonPenaltyTest(unittest.TestCase):
    """Transmon-excitation cost C_T = sum_{k=1}^{N} sum_{b=0_L,1_L} Tr[P_T rho_b(k)], sharing
    the forbidden-state costate as a weighted mask stack."""

    def test_mask_is_transmon_excitation(self):
        for n_r in (1, 2):
            model = Model(7, n_r)
            ops = operators(7, n_r)
            P_T = ops["Pe"] + ops["Pf"]
            self.assertEqual(np.abs(P_T - np.diag(np.diag(P_T))).max(), 0.0)  # diagonal
            mask = lnp.transmon_mask(model)
            np.testing.assert_array_equal(mask, np.real(np.diag(P_T)) > 0.5)
            self.assertEqual(mask.sum(), 7 * 2 * n_r)

    def test_weighted_stack_np_jax_fd(self):
        from DGRAPE import lindblad_jax as ljx
        model = Model(12, **ALL)
        E, T, w = process_targets("X", 12)
        u = random_pulse(20, scale=15.0, seed=41)
        mF, mT = lnp.forbidden_mask(model, 6), lnp.transmon_mask(model)
        masks, wts = np.array([mF, mT]), np.array([0.7, 2.5])
        spec = (3, 20)
        F0, g0 = lnp.fidelity_and_grad(model, u, E, T, w, DT, spec)
        F, g, C, gC = lnp.fidelity_and_grad(model, u, E, T, w, DT, spec, forbid_mask=masks,
                                            occ_weights=wts)
        self.assertEqual(F, F0)
        np.testing.assert_array_equal(g, g0)
        self.assertEqual(C.shape, (2,))
        for k, mk in enumerate((mF, mT)):           # separate values == single-mask runs
            _, _, Ck, gk = lnp.fidelity_and_grad(model, u, E, T, w, DT, spec, forbid_mask=mk)
            self.assertAlmostEqual(C[k], Ck, places=11)
            self.assertAlmostEqual(Ck, lnp.forbidden_population(model, u, E, DT, spec, mk)[0],
                                   places=11)
            if k == 0:
                g_sum = wts[0] * gk
            else:
                g_sum = g_sum + wts[1] * gk
        self.assertGreater(C[1], 1e-3)
        np.testing.assert_allclose(gC, g_sum, rtol=1e-10, atol=1e-13 * np.abs(gC).max())
        _, fg = ljx.make_fidelity_fns(model, E, T, w, DT, forbid_mask=masks, occ_weights=wts)
        _, _, Cj, gCj = fg(u, *spec)
        np.testing.assert_allclose(C, np.asarray(Cj), rtol=1e-12)
        np.testing.assert_allclose(gC, np.asarray(gCj), rtol=1e-8,
                                   atol=1e-11 * np.abs(gC).max())
        h = 1e-5
        tot = lambda v: sum(wt * lnp.forbidden_population(model, v, E, DT, spec, mk)[0]  # noqa
                            for wt, mk in zip(wts, (mF, mT)))
        for k, j in [(0, 0), (6, 1), (11, 2), (17, 3)]:
            up, um = u.copy(), u.copy()
            up[k, j] += h
            um[k, j] -= h
            fd = (tot(up) - tot(um)) / (2 * h)
            self.assertLess(abs(fd - gC[k, j]), 1e-8 * np.abs(gC).max(),
                            f"({k},{j}): adjoint {gC[k, j]} fd {fd}")

    def test_objective_with_both_terms_fd(self):
        N = 40
        kw = dict(penalties={"forbid": 0.1, "transmon": 1e-2}, forbid_fock_min=6, ramp_ns=10.0)
        x = random_pulse(N, scale=20.0, seed=42)
        objs = {b: OpenGrapeObjective("I", N, DT, [7, 8], b, **kw) for b in ("np", "jax")}
        self.assertEqual(objs["np"].occ_names, [["forbid", "transmon"]] * 2)
        c, g, terms = objs["np"].evaluate(x)
        self.assertGreater(terms["transmon"], 1e-4)
        self.assertEqual(len(terms["C_transmon"]), 2)
        self.assertAlmostEqual(terms["transmon"], 1e-2 * np.mean(terms["C_transmon"]), places=14)
        cj, gj, tj = objs["jax"].evaluate(x)
        self.assertAlmostEqual(c, cj, places=12)
        np.testing.assert_allclose(terms["C_transmon"], tj["C_transmon"], rtol=1e-12)
        np.testing.assert_allclose(g, gj, rtol=1e-8, atol=1e-11 * np.abs(g).max())
        g = g.reshape(N, 4)
        h = 1e-5
        for k, j in [(3, 0), (12, 1), (20, 2), (33, 3)]:
            xp, xm = x.copy(), x.copy()
            xp[k, j] += h
            xm[k, j] -= h
            fd = (objs["np"].cost(xp) - objs["np"].cost(xm)) / (2 * h)
            self.assertLess(abs(fd - g[k, j]), 1e-7 * np.abs(g).max(),
                            f"({k},{j}): grad {g[k, j]} fd {fd}")

    def test_transmon_alone_and_parallel(self):
        # Forbid empty at n_c = 7 (fock_min 10), transmon on: only the transmon term, and the
        # split-by-input path (n_jobs 3 on one truncation) equals the serial one.
        x = random_pulse(30, scale=10.0, seed=43)
        kw = dict(penalties={"transmon": 1e-3}, ramp_ns=10.0)
        serial = OpenGrapeObjective("H", 30, DT, 7, n_jobs=1, **kw)
        self.assertEqual(serial.occ_names, [["transmon"]])
        c1, g1, t1 = serial.evaluate(x)
        self.assertNotIn("forbid", t1)
        with OpenGrapeObjective("H", 30, DT, 7, n_jobs=3, **kw) as par:
            self.assertEqual(par.split, {0})
            c3, g3, t3 = par.evaluate(x)
            self.assertAlmostEqual(c1, c3, places=14)
            np.testing.assert_allclose(g1, g3, rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(t1["C_transmon"], t3["C_transmon"], rtol=1e-13)
            u = par.to_physical(x)
            self.assertAlmostEqual(par.report(u)[0]["C_transmon"], t1["C_transmon"][0],
                                   places=10)

    def test_off_by_default(self):
        obj = OpenGrapeObjective("X", 30, DT, 12, ramp_ns=10.0)
        self.assertEqual(obj.occ_names, [["forbid"]])
        _, _, terms = obj.evaluate(random_pulse(30, seed=44))
        self.assertNotIn("transmon", terms)
        self.assertEqual(obj.report(random_pulse(30, seed=44))[0]["C_transmon"], 0.0)


class ParallelTest(unittest.TestCase):
    def test_split_plan(self):
        # Largest truncations split first, three jobs each; capped at 3 per truncation (np).
        plans = {nj: OpenGrapeObjective("X", 30, DT, [7, 8, 9], n_jobs=nj, ramp_ns=10.0)
                 for nj in (1, 3, 5, 7, 9, 12)}
        self.assertEqual({nj: o.split for nj, o in plans.items()},
                         {1: set(), 3: set(), 5: {2}, 7: {1, 2}, 9: {0, 1, 2}, 12: {0, 1, 2}})
        self.assertEqual(plans[12].n_jobs, 9)
        for o in plans.values():
            o.close()
        jx = OpenGrapeObjective("X", 30, DT, [7, 8], "jax", n_jobs=9, ramp_ns=10.0)
        self.assertEqual((jx.n_jobs, jx.split), (2, set()))

    def test_parallel_equals_serial(self):
        # n_jobs = 3 (one job per truncation), 5 and 9 (truncations split by process input).
        x = random_pulse(30, scale=10.0, seed=34)
        kw = dict(penalties={"forbid": 0.1}, forbid_fock_min=6, ramp_ns=10.0)
        serial = OpenGrapeObjective("H", 30, DT, [7, 8, 9], n_jobs=1, **kw)
        c1, g1, t1 = serial.evaluate(x)
        u = serial.to_physical(x)
        rep1 = serial.report(u)
        for nj in (3, 5, 9):
            with OpenGrapeObjective("H", 30, DT, [7, 8, 9], n_jobs=nj, **kw) as par:
                self.assertEqual(par.n_jobs, nj)
                c3, g3, t3 = par.evaluate(x)
                self.assertAlmostEqual(c1, c3, places=14, msg=nj)
                np.testing.assert_allclose(g1, g3, rtol=1e-12, atol=1e-15)
                np.testing.assert_allclose(t1["F_pro"], t3["F_pro"], rtol=1e-13)
                np.testing.assert_allclose(t1["C_forbid"], t3["C_forbid"], rtol=1e-13)
                for r1, r3 in zip(rep1, par.report(u)):
                    self.assertAlmostEqual(r1["F_pro"], r3["F_pro"], places=14)
                    self.assertAlmostEqual(r1["C_forbid"], r3["C_forbid"], places=13)


class ConstraintTest(unittest.TestCase):
    def test_jnp_chain_equals_core(self):
        from core.ramp import make_constraint_chain
        from DGRAPE.lindblad_jax import make_constrainer
        N = 200
        x = random_pulse(N, seed=10)
        for kw in (dict(cav_band=(-27, 27), tra_band=(-33, 33), ramp_ns=48.0),
                   dict(cav_band=(-10, 30), tra_band=(-50, 5), ramp_ns=None),
                   dict(ramp_ns=24.0)):
            ref, _ = make_constraint_chain(N, DT, kw.get("cav_band"), kw.get("tra_band"),
                                           kw.get("ramp_ns"))
            mine = make_constrainer(N, DT, kw.get("cav_band"), kw.get("tra_band"),
                                    kw.get("ramp_ns"))
            np.testing.assert_allclose(np.asarray(mine(x)), ref(x), atol=1e-12)


class SmokeTest(unittest.TestCase):
    """train.py at the recipe defaults. Nothing is saved."""

    def test_cold_start_is_smooth_and_feasible(self):
        from DGRAPE.recipe import RECIPE
        from DGRAPE.train import initial_preimage
        obj = OpenGrapeObjective("X", 200, DT, 7)
        x0, kind, _ = initial_preimage(obj, seed=3)
        self.assertEqual(kind, "cold")
        self.assertLessEqual(np.abs(x0).max(), RECIPE["warm_start_amp"] + 1e-12)
        spec = np.abs(np.fft.rfft(x0, axis=0)) ** 2
        kcut = max(1, int(RECIPE["warm_start_cutoff_frac"] * spec.shape[0]))
        self.assertLess(spec[kcut:].sum() / spec.sum(), 1e-20)

    def test_warm_start_roundtrip_and_refusals(self):
        from DGRAPE.train import initial_preimage, train
        obj = OpenGrapeObjective("X", 60, DT, 7)
        u = obj.to_physical(random_pulse(60, scale=3.0, seed=24))
        x0, kind, err = initial_preimage(obj, 0, warm_start=u)
        self.assertEqual(kind, "pulse")
        np.testing.assert_allclose(obj.to_physical(x0), u, atol=1e-10)
        with self.assertRaises(ValueError):
            initial_preimage(obj, 0, init_x=x0, warm_start=u)
        with self.assertRaises(ValueError):          # outside the 25 box: refused, not clipped
            train("X", 60, DT, 7, maxiter=1, init_x=np.full((60, 4), 30.0), verbose=False,
                  forbid_fock_min=5)
        with self.assertRaises(ValueError):          # P_F empty at n_c = 7 <= 10: refused
            train("X", 60, DT, 7, maxiter=1, verbose=False)

    def test_five_iterations_at_recipe(self):
        from DGRAPE.train import train
        # forbid_fock_min = 5 so the forbidden-state term is live at n_c = 7.
        u, x, info = train("X", 60, DT, 7, maxiter=5, seed=0, verbose=False, forbid_fock_min=5)
        self.assertIn("forbid", info["history"][-1])
        self.assertIn("C_forbid", info["per_trunc"][0])
        _, _, info_t = train("X", 60, DT, 7, maxiter=3, seed=0, verbose=False, forbid_fock_min=5,
                             penalties={"transmon": 1e-3})
        self.assertIn("transmon", info_t["history"][-1])
        self.assertGreater(info_t["per_trunc"][0]["C_transmon"], 0.0)
        self.assertEqual(len(info["history"]), 5)
        self.assertLess(info["history"][-1]["cost"], info["cost0"])
        self.assertEqual(info["warm_start_kind"], "cold")
        self.assertEqual(info["ramp_ns"], 48.0)
        for k in ("out_of_band_cavity", "out_of_band_transmon", "endpoint_rel_to_peak"):
            self.assertIn(k, info["constraints"])
        self.assertLessEqual(info["max_abs_preimage"], 25.0)
        np.testing.assert_allclose(OpenGrapeObjective("X", 60, DT, 7).to_physical(x), u,
                                   atol=1e-12)


class QutipTest(unittest.TestCase):
    def test_mesolve_step_coefficients(self):
        try:
            import qutip
        except ImportError:
            self.skipTest("qutip not installed")
        n_c, N = 7, 10
        model = Model(n_c, **ALL)
        u = random_pulse(N, scale=10.0, seed=11)
        rho0 = random_density(model.d, seed=12)
        R = lnp.propagate(model, u, rho0[None], DT)[0]

        dims = [[n_c, 3, 2], [n_c, 3, 2]]
        Q = lambda M: qutip.Qobj(M, dims=dims)           # noqa: E731
        tlist = np.arange(N + 1) * DT
        H = [Q(model.H_d)]
        for j in range(4):
            coeff = qutip.coefficient(np.append(u[:, j], u[-1, j]), tlist=tlist, order=0)
            H.append([Q(model.Hc[j]), coeff])
        res = qutip.mesolve(qutip.QobjEvo(H), Q(rho0), tlist,
                            c_ops=[Q(c) for _, c in model.c_ops],
                            options={"atol": 1e-12, "rtol": 1e-10, "nsteps": 100000,
                                     "max_step": DT / 20})
        self.assertLess(np.abs(res.states[-1].full() - R).max(), 1e-6)


if __name__ == "__main__":
    unittest.main()
