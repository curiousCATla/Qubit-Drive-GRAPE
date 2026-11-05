#!/usr/bin/env python3
"""
test_est_gate.py

Checks for DGRAPE/est_gate.py, an EsT X gate on the Shirol chip trained in the closed system with the
cost of EST/grape_eigh.py (FWM_tails.ipynb §3).

  * recipe     -- the weight schedule is EST/train_est.py's (est, ord); the drive limits are
                  DGRAPE/recipe.py's (d2's); tiers follow EST's rule
  * code       -- six cardinals and error cardinals on the odd kitten code; the X targets swap +-Z and
                  +-Y and keep +-X
  * frame      -- the trainer's eigh propagation equals lindblad_np.propagate with every rate at 0, and
                  the metrics' independent propagator (diagnostics.propagate_states) agrees with both
  * gradient   -- EstObjective's pre-image gradient (chain, C1-C4, C4 active) = central differences
  * idle       -- u = 0 without Kerr: all three metrics vanish. With Kerr: Delta_QEC stays 0 (Kerr is
                  diagonal) but L and eta do not: for this code the error word (|0> + sqrt5|4>)/sqrt6
                  dephases against a|0_L(t)>, the obstruction App. A of the EsT paper describes

Run:
    python3 validation/test_est_gate.py [-v]       (about ten seconds)
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from DGRAPE import est_gate as eg                          # noqa: E402
from DGRAPE import lindblad_np as lnp                      # noqa: E402
from DGRAPE.model import Model                             # noqa: E402
from DGRAPE.recipe import RECIPE                           # noqa: E402
from EST import diagnostics as dg                          # noqa: E402
from EST.grape_eigh import propagate_trajectory            # noqa: E402

np.seterr(divide="ignore", over="ignore", invalid="ignore")
NC = 8


def _smooth_pulse(N, amp, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / N
    u = np.zeros((N, 4))
    for k in range(1, 4):
        u += np.outer(np.sin(np.pi * k * t), rng.normal(size=4)) / k
    return amp * u


class Recipe(unittest.TestCase):
    def test_schedule_is_ests(self):
        from EST.train_est import SCHEDULES
        for v in ("est", "ord"):
            self.assertEqual(eg.SCHEDULES[v], SCHEDULES[v])
            self.assertEqual(eg.stage_weights(v), [w + (1.0,) for w in SCHEDULES[v]])

    def test_est_err_is_section7s(self):
        from EST.err_two_stage import WEIGHTS
        self.assertEqual(eg.stage_weights("est_err"), WEIGHTS["run"])

    def test_four_tuple_path_unchanged(self):
        """est's weights with zero-padded extras give the same cost: C_err enters only through w_err."""
        N = 40
        u = _smooth_pulse(N, 8.0, 3)
        kw = dict(n_c=NC, N=N, cav_band=None, tra_band=None, ramp_ns=None)
        r4 = eg.EstObjective((1.0, 0.7, 7.0, 1.0), **kw).report(u)
        r7 = eg.EstObjective((1.0, 0.7, 7.0, 1.0, 0.0, 0.0, 0.0), **kw).report(u)
        self.assertNotIn("cerr", r4)
        self.assertIn("cerr", r7)
        self.assertEqual(r4["cost"], r7["cost"])

    def test_est_err2_is_est_err_plus_stage2_cerr(self):
        """est_err2 shares est_err's stage 1 (so a stage-2 resume is its full run) and keeps w_err in stage 2."""
        e1, e2 = eg.stage_weights("est_err"), eg.stage_weights("est_err2")
        self.assertEqual(e2[0], e1[0])
        self.assertEqual(e2[1], e1[1][:4] + (0.3, 0.0, 0.0))
        with self.assertRaises(ValueError):
            eg.train("est_err2", maxiter=1, start_stage=2, verbose=False)

    def test_limits_are_d2s(self):
        self.assertEqual(eg.CAV_BAND, tuple(RECIPE["cav_band"]))
        self.assertEqual(eg.TRA_BAND, tuple(RECIPE["tra_band"]))
        self.assertEqual(eg.RAMP_NS, RECIPE["ramp_ns"])
        self.assertEqual(eg.U_BOX, RECIPE["hard_amp_limit"])
        self.assertEqual(eg.EPS_MAX, RECIPE["penalties"]["amp_max"])
        self.assertEqual((eg.N_STEPS, eg.DT), (550, 0.002))
        self.assertTrue(np.array_equal(eg.start_preimage().shape, (550, 4)))

    def test_tiers(self):
        self.assertEqual(eg.tier(1.0 / 3.0), "parked")
        self.assertEqual(eg.tier(0.99), "open")
        self.assertEqual(eg.tier(0.9995), "closed")


class Code(unittest.TestCase):
    def test_cardinals_and_targets(self):
        s = eg.system(NC)
        for M in (s.C0, s.E0, s.target):
            self.assertLess(np.abs(np.linalg.norm(M, axis=0) - 1).max(), 1e-12)
        # +Z, -Z, +X, -X, +Y, -Y -> X sends them to -Z, +Z, +X, -X, -Y, +Y (up to phases)
        ov = np.abs(s.target.conj().T @ s.C0)
        self.assertLess(np.abs(ov[np.arange(6), [1, 0, 2, 3, 5, 4]] - 1.0).max(), 1e-12)
        # the error cardinals live on Fock 0, 2, 4 with the transmon in g
        pop = (np.abs(s.E0) ** 2).reshape(NC, 3, 6).sum(axis=(1, 2))
        self.assertAlmostEqual(pop[[0, 2, 4]].sum(), 6.0, delta=1e-12)
        # C_err's target: X on the error cardinals, same permutation as the code targets
        self.assertLess(np.abs(np.linalg.norm(s.target_err, axis=0) - 1).max(), 1e-12)
        ov = np.abs(s.target_err.conj().T @ s.E0)
        self.assertLess(np.abs(ov[np.arange(6), [1, 0, 2, 3, 5, 4]] - 1.0).max(), 1e-12)

    def test_h_targets(self):
        """H sends +Z, -Z, +X, -X, +Y, -Y to +X, -X, +Z, -Z, -Y, +Y (up to phases), on both bases."""
        s = eg.system(NC, gate="H")
        perm = [2, 3, 0, 1, 5, 4]
        for target, cards in ((s.target, s.C0), (s.target_err, s.E0)):
            ov = np.abs(target.conj().T @ cards)
            self.assertLess(np.abs(ov[np.arange(6), perm] - 1.0).max(), 1e-12)
        self.assertEqual(eg.tier(1.0 / 3.0, "H"), "parked")


class Seeds(unittest.TestCase):
    """DGRAPE/est_gate_seeds.py: X's file names are pinned; H ranks by peak score and has no reference."""

    def test_x_paths_unchanged(self):
        from DGRAPE import est_gate_seeds as S
        P = S.paths("X")
        self.assertEqual(os.path.basename(P["run_dir"]), "est_err_seeds")
        self.assertEqual([os.path.basename(P[k]) for k in ("runs_csv", "pick_csv", "open_csv", "traces_npz")],
                         ["est_gate_seeds_X_runs.csv", "est_gate_seeds_X_pick.csv", "est_gate_seeds_X_open.csv",
                          "est_gate_seeds_X_traces.npz"])
        self.assertEqual((S.RUN_DIR, S.RUNS_CSV, S.PICK_CSV), (P["run_dir"], P["runs_csv"], P["pick_csv"]))
        self.assertEqual(S.tag(24), "X_est_err_closed_s24")
        self.assertEqual(S.tag(3, "H"), "H_est_err_closed_s3")
        self.assertEqual(os.path.basename(S.paths("H")["run_dir"]), "est_err_seeds_H")
        P2 = S.paths("H", "est_err2")
        self.assertEqual(os.path.basename(P2["run_dir"]), "est_err2_seeds_H")
        self.assertEqual(os.path.basename(P2["pick_csv"]), "est_gate_seeds_H_err2_pick.csv")
        args = S.train_args(5, "H", "est_err2")
        self.assertEqual(args[args.index("--init-x") + 1], S.stage1_preimage(5, "H"))
        self.assertEqual(args[args.index("--start-stage") + 1], "2")
        self.assertNotIn("--init-x", S.train_args(5, "H"))
        self.assertNotIn("--duration-us", S.train_args(5, "H"))

    def test_h_2us_paths(self):
        from DGRAPE import est_gate_seeds as S
        P = S.paths("H", duration=2.0)
        self.assertEqual(os.path.basename(P["run_dir"]), "est_err_seeds_H_T2")
        self.assertEqual(os.path.basename(P["pick_csv"]), "est_gate_seeds_H_T2_pick.csv")
        self.assertEqual(S.tag(3, "H", duration=2.0), "H_est_err_closed_T2_s3")
        args = S.train_args(3, "H", duration=2.0)
        self.assertEqual(args[args.index("--duration-us") + 1], "2.0")
        self.assertEqual(args[args.index("--tag") + 1], "H_est_err_closed_T2_s3")
        self.assertEqual((S.rank_by("X"), S.rank_by("H"), S.rank_by("H", 2.0)), ("score", "peak_score", "score"))
        self.assertEqual(S.peak_max("H", 2.0), {"dqec": 0.4, "L": 0.5, "eta": 0.5})       # user, 2026-11-02
        self.assertEqual((S.peak_max("X"), S.peak_max("H")), (S.PEAK_MAX, S.PEAK_MAX))

    def test_ranking(self):
        import pandas as pd
        from DGRAPE import est_gate_seeds as S
        good = dict(stage="final", kind="run", F1=0.999, tier="closed", trunc_spread=0.0, dqec_tail=0.0,
                    L_tail=0.0, eta_tail=0.0)
        rows = [dict(good, name="a", seed=0, score=0.10, peak_score=0.40, dqec_peak=0.4, L_peak=0.4, eta_peak=0.4),
                dict(good, name="b", seed=1, score=0.20, peak_score=0.20, dqec_peak=0.2, L_peak=0.2, eta_peak=0.2)]
        runs = pd.DataFrame(rows)
        ph = S.pick_table(runs, "H").set_index("name")
        self.assertEqual((ph.loc["b", "rank"], ph.loc["a", "rank"]), (1, 2))
        self.assertNotIn("beats_current", ph.columns)
        p2 = S.pick_table(runs, "H", 2.0).set_index("name")          # H at 2 us ranks by the mean score
        self.assertEqual((p2.loc["a", "rank"], p2.loc["b", "rank"]), (1, 2))
        rows_mid = [dict(rows[0], dqec_peak=0.45), rows[1]]           # passes the 0.5 bar, not 2 us H's 0.4
        p2 = S.pick_table(pd.DataFrame(rows_mid), "H", 2.0).set_index("name")
        self.assertFalse(p2.loc["a", "candidate"])
        self.assertEqual(p2.loc["b", "rank"], 1)
        self.assertTrue(S.pick_table(pd.DataFrame(rows_mid), "H").set_index("name").loc["a", "candidate"])
        ref =dict(good, name=S.CURRENT, kind="reference", seed=42, score=0.15, peak_score=0.3, dqec_peak=0.3,
                   L_peak=0.3, eta_peak=0.3)
        px = S.pick_table(pd.DataFrame(rows + [ref]), "X").set_index("name")
        self.assertEqual((px.loc["a", "rank"], px.loc["b", "rank"]), (1, 2))
        self.assertEqual((bool(px.loc["a", "beats_current"]), bool(px.loc["b", "beats_current"])), (True, False))


class Frame(unittest.TestCase):
    def test_eigh_equals_lindblad_closed(self):
        N = 40
        u = _smooth_pulse(N, 8.0, 0)
        s = eg.system(NC)
        Psi0 = np.concatenate([s.C0, s.E0], axis=1)
        traj = propagate_trajectory(u, s.H0, s.Hc, Psi0, eg.DT)
        rho0 = np.einsum("im,jm->mij", Psi0, Psi0.conj())
        m = Model(NC, 1, include_higher_order=True, rate_scale=0.0)
        R = lnp.propagate(m, u, rho0, eg.DT)
        rho_T = np.einsum("im,jm->mij", traj[-1], traj[-1].conj())
        self.assertLess(np.abs(R - rho_T).max(), 1e-10)
        alt = dg.propagate_states(u, s.H0, list(s.Hc), Psi0, eg.DT)
        self.assertLess(np.abs(alt[-1] - traj[-1]).max(), 1e-10)
        # closed F1 from the metrics' propagator equals the trainer's 1 - C1
        F1_train = eg.EstObjective((1, 0, 0, 0), NC, N, cav_band=None, tra_band=None,
                                   ramp_ns=None).report(u)["F1"]
        self.assertAlmostEqual(eg.metrics(u, NC)["F1"], F1_train, delta=1e-10)


class Gradient(unittest.TestCase):
    def test_preimage_gradient(self):
        N = 40
        rng = np.random.default_rng(1)
        obj = eg.EstObjective((1.0, 0.7, 7.0, 1.0), NC, N, ramp_ns=8.0, cav_band=(-120, 120),
                              tra_band=(-120, 120))
        x = 40.0 * rng.normal(size=(N, 4))
        _, g, terms = obj.evaluate(x)
        self.assertGreater(terms["c4"], 0.0)                     # the cap is active
        h = 1e-4
        for j in rng.choice(N * 4, 6, replace=False):
            e = np.zeros(N * 4)
            e[j] = h
            fd = (obj.cost(x.ravel() + e) - obj.cost(x.ravel() - e)) / (2 * h)
            self.assertLess(abs(fd - g[j]), 1e-6 * max(1.0, np.abs(g).max()))

    def test_preimage_gradient_with_cerr(self):
        N = 40
        rng = np.random.default_rng(2)
        obj = eg.EstObjective((1.0, 0.7, 7.0, 1.0, 0.3, 0.0, 0.0), NC, N, ramp_ns=8.0,
                              cav_band=(-120, 120), tra_band=(-120, 120))
        x = 10.0 * rng.normal(size=(N, 4))
        _, g, terms = obj.evaluate(x)
        self.assertGreater(terms["cerr"], 1e-3)
        h = 1e-4
        for j in rng.choice(N * 4, 6, replace=False):
            e = np.zeros(N * 4)
            e[j] = h
            fd = (obj.cost(x.ravel() + e) - obj.cost(x.ravel() - e)) / (2 * h)
            self.assertLess(abs(fd - g[j]), 1e-6 * max(1.0, np.abs(g).max()))


class Idle(unittest.TestCase):
    def test_idle_metrics(self):
        u = np.zeros((100, 4))
        m0 = eg.metrics(u, NC, include_higher_order=False)
        for k in ("delta_qec", "leakage", "eta_avg"):
            self.assertLess(np.abs(m0[k]).max(), 1e-12, k)
        m1 = eg.metrics(u, NC, include_higher_order=True)
        self.assertLess(np.abs(m1["delta_qec"]).max(), 1e-12)
        self.assertGreater(m1["leakage"][-1], 1e-8)
        self.assertGreater(m1["eta_avg"][-1], 1e-6)


if __name__ == "__main__":
    unittest.main()
