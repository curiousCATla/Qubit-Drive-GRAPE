#!/usr/bin/env python3
"""
test_fwm_tail.py

Checks for DGRAPE/fwm_tail.py, an FWM tail after the d2 X gate with one waveform per comb
(notebooks/FWM_tails.ipynb).

  * idle           -- with eta = 0 the tail equals lnp.propagate's idle evolution in the DGRAPE frame
                      (same F_pro, traced and empty; same transmon-g block)
  * frame          -- a random piecewise-constant eta in the |f>-rotating frame (one exponential per
                      step) equals fwm.propagate in the DGRAPE frame with carriers (+Delta_ref,
                      -Delta_ref), Stark and pump mixing on: CF4 converges to it at 4th order
  * teeth          -- tooth offsets sit at about (n - n_ref) chi_gf (comb 2 mirrored), and a tooth
                      built by eta_from_teeth on path 1 swaps |0,g,0> -> |1,f,0> as its closed form says
  * gradient       -- W form = augmented-generator directional derivative = central differences
  * objective      -- TailObjective's pre-image gradient (cap penalty active) equals central
                      differences, and n_jobs = 3 equals n_jobs = 1
  * input          -- d2_output (n_r = 1, embedded) equals propagating d2 at n_r = 2 directly
  * gate pulse     -- gate_output / score(pulse=...) for a gate other than d2 equal direct propagation
  * ceilings       -- the ideal repair after a loss right after a perfect X reaches the kitten limit
                      (1 + (1+sqrt5)/sqrt12)^2/4; the optimal recovery of a unitary channel is 1
  * readout        -- n_r = 2 equals n_r = 3 with modest tones
  * cut            -- gate_n_c = n_c is the default bit for bit; a gate run at a larger n_c and cut
                      (tail_input) scores within the trace lost of the uncut gate + tail
  * n gates        -- train_fwm_tail's --n-gates m: the tail input is m gates back to back (equal to m
                      fwm_repeat gate steps at n_r = 2), the default target is X^m, m = 1 is unchanged
  * code words     -- code_word_scores (a tail with no gate in front): no tail gives code 1, error 0; the
                      50/50 mixed input scores the average of the two (linearity); the closed 0.8 us EsT
                      tail reproduces tables/fwm_tail_H_code.csv (notebook Table 16)
  * shirol noise   -- noise="shirol" (Shirol's full set) has AQEC.master_equation.collapse_operators'
                      dissipator; keep filters it; the default is unchanged bit for bit; the |f> frame
                      stays exact; TailObjective(noise="shirol") gradient = central differences

Run:
    python3 validation/test_fwm_tail.py [-v]       (about ten seconds)
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scipy.stats import unitary_group                      # noqa: E402

from AQEC import device as dev                             # noqa: E402
from DGRAPE import fwm                                     # noqa: E402
from DGRAPE import fwm_tail as ft                          # noqa: E402
from DGRAPE import lindblad_np as lnp                      # noqa: E402
from DGRAPE.code import embedded_basis, process_targets    # noqa: E402
from DGRAPE.model import F, G, Model, index, operators     # noqa: E402

np.seterr(divide="ignore", over="ignore", invalid="ignore")
DT = ft.DT
NC = 8


def _rand_eta(N, amp, seed):
    rng = np.random.default_rng(seed)
    return amp * (rng.normal(size=(N, 2)) + 1j * rng.normal(size=(N, 2)))


class Idle(unittest.TestCase):
    def test_equals_dgrape_idle(self):
        N = 60
        R0 = ft.d2_output(NC, 2)
        tm = ft.tail_model(NC, 2)
        R_tail, _ = ft.propagate(tm, np.zeros((N, 2)), R0)
        m = fwm.model_with_channels(NC, 2, None, True)
        R_dg = lnp.propagate(m, np.zeros((N, 4)), R0, DT)
        for readout in ("traced", "empty"):
            T, w = ft.tail_targets(NC, 2, readout=readout)
            self.assertAlmostEqual(ft.fidelity_from_outputs(R_tail, T, w),
                                   ft.fidelity_from_outputs(R_dg, T, w), delta=1e-12)
        g = (np.arange(tm.model.d) // 2) % 3 == G
        self.assertLess(np.abs(R_tail[:, g][:, :, g] - R_dg[:, g][:, :, g]).max(), 1e-12)


class Frame(unittest.TestCase):
    def test_equals_dgrape_frame(self):
        N = 30
        eta = _rand_eta(N, 0.06, 0)
        R0 = ft.d2_output(NC, 2)
        tm = ft.tail_model(NC, 2, mixing=1.0)
        R_f, _ = ft.propagate(tm, eta, R0)
        m0 = Model(NC, 2, include_higher_order=True)
        fw0 = fwm.fwm_terms(m0, 1.0, paths=(ft.N_REF,))
        fw0.Delta = np.array([tm.delta_ref, -tm.delta_ref])
        pf = np.real(np.diag(operators(NC, 2)["Pf"]))
        ph = np.exp(1j * tm.delta_ref * N * DT * pf)           # V(T) = exp(i Delta_ref T P_f)
        R_f_dg = ph[:, None] * R_f * ph[None, :].conj()
        err = {}
        for S in (4, 8):
            R_d, _ = fwm.propagate(m0, fw0, np.zeros((N, 4)), eta, DT, S, R0)
            err[S] = np.abs(R_d - R_f_dg).max()
        self.assertGreater(err[4] / err[8], 10.0)              # 4th order: about 16
        self.assertLess(err[8], 1e-6)


class Teeth(unittest.TestCase):
    def test_offsets(self):
        off = ft.tooth_offsets(12, ft.N_REF, (1, 2, 3, 4, 5))
        chi = dev.chi_gf
        for k, n in enumerate(off["paths"]):
            self.assertAlmostEqual(off["comb1"][k] / chi, n - ft.N_REF, delta=0.02)
            self.assertAlmostEqual(off["comb2"][k] / chi, -(n - ft.N_REF), delta=0.02)
        self.assertEqual(off["comb1"][2], 0.0)

    def test_tooth_on_path_1_swaps(self):
        # One comb-1 tooth on path 1 (offset -4.1 MHz inside eta_1), closed system, from |0,g,0>.
        b = 0.01
        Om = fwm.G1 * b * np.sqrt(2)
        N = int(round(np.pi / (2 * Om) / DT))
        teeth = np.zeros(6, dtype=complex)
        teeth[0] = b                                           # (comb 1, n = 1)
        eta = ft.eta_from_teeth(teeth, N, DT, NC)
        tm = ft.tail_model(NC, 2, keep=set())
        i0, i1 = index(0, G, 0, NC), index(1, F, 0, NC)
        rho = np.zeros((1, tm.model.d, tm.model.d), dtype=complex)
        rho[0, i0, i0] = 1.0
        R, _ = ft.propagate(tm, eta, rho)
        P = R[0, i1, i1].real
        fw = fwm.fwm_terms(Model(NC, 2, include_higher_order=True))
        dS = b ** 2 * (fw.S_diag[i1] - fw.S_diag[i0])
        contrast = 4 * Om ** 2 / (4 * Om ** 2 + dS ** 2)
        self.assertGreater(P, 0.9 * contrast)                  # a wrong offset sign gives ~0.01


class Gradient(unittest.TestCase):
    def test_w_form(self):
        n_c, N = 7, 10
        tm = ft.tail_model(n_c, 2, mixing=1.0)
        R0 = ft.d2_output(n_c, 2)
        T, w = ft.tail_targets(n_c, 2)
        eta = _rand_eta(N, 0.05, 1)
        F0, g = ft.fidelity_and_grad(tm, eta, R0, T, w)
        deta = _rand_eta(N, 1.0, 2)
        w_form = np.sum(g * fwm.real_from_beta(deta))
        aug = ft.directional_derivative(tm, eta, R0, T, w, deta)
        self.assertLess(abs(w_form - aug), 1e-12 * max(1.0, abs(aug)) + 1e-14)
        h = 1e-6
        fd = (ft.fidelity(tm, eta + h * deta, R0, T, w) - ft.fidelity(tm, eta - h * deta, R0, T, w)) / (2 * h)
        self.assertLess(abs(fd - w_form), 1e-6 * max(1.0, abs(w_form)))


class Objective(unittest.TestCase):
    def test_fd_and_jobs(self):
        n_c, N = 7, 40
        kw = dict(tail_us=N * DT, n_c=n_c, band=(-30.0, 30.0), ramp_ns=8.0)
        rng = np.random.default_rng(3)
        x = 0.15 * ft.S_ETA * rng.normal(size=(N, 4))         # |eta| well above the cap
        o1 = ft.TailObjective(n_jobs=1, **kw)
        c, gx, terms = o1.evaluate(x)
        self.assertGreater(terms["amp"], 0.0)
        for j in rng.choice(N * 4, 6, replace=False):
            h = 1e-3
            e = np.zeros(N * 4)
            e[j] = h
            fd = (o1.cost(x.ravel() + e) - o1.cost(x.ravel() - e)) / (2 * h)
            self.assertLess(abs(fd - gx[j]), 1e-6 * max(1.0, abs(gx).max()))
        o3 = ft.TailObjective(n_jobs=3, **kw)
        try:
            c3, g3, _ = o3.evaluate(x)
        finally:
            o3.close()
        self.assertAlmostEqual(c3, c, delta=1e-13)
        self.assertLess(np.abs(g3 - gx).max(), 1e-12)


class Input(unittest.TestCase):
    def test_d2_output(self):
        u = ft.d2_pulse()
        m = Model(NC, 2, include_higher_order=True)
        E, _, _ = process_targets("X", NC, 2)
        R = lnp.propagate(m, u, E, DT)
        self.assertLess(np.abs(ft.d2_output(NC, 2) - R).max(), 1e-12)


class GatePulse(unittest.TestCase):
    def test_other_gate(self):
        # gate_output(u) for a pulse other than d2: its own cache entry, equal to direct propagation
        u = 0.5 * ft.d2_pulse()
        R = ft.gate_output(u, NC, 2)
        m = Model(NC, 1, include_higher_order=True)
        E, _, _ = process_targets("X", NC, 1)
        direct = ft.embed_readout(lnp.propagate(m, u, E, DT), 2)
        self.assertLess(np.abs(R - direct).max(), 1e-12)
        self.assertGreater(np.abs(R - ft.d2_output(NC, 2)).max(), 1e-3)
        self.assertTrue(np.array_equal(ft.gate_output(None, NC, 2), ft.d2_output(NC, 2)))
        s = ft.score(np.zeros((0, 2)), n_c=NC, pulse=u)["F_pro"]
        T, w = ft.tail_targets(NC, 2)
        self.assertAlmostEqual(s, ft.fidelity_from_outputs(direct, T, w), delta=1e-14)


class Ceilings(unittest.TestCase):
    def test_kitten_limit(self):
        n_c = 9
        B = embedded_basis(n_c, 1)
        _, T, w = process_targets("X", n_c, 1)
        a = operators(n_c, 1)["a"]
        Xg = np.array([[0, 1], [1, 0]], dtype=complex)
        R_out = []
        for i, j in ((0, 0), (1, 1), (0, 1)):
            psi_i, psi_j = a @ B @ Xg[:, i], a @ B @ Xg[:, j]
            R_out.append(np.outer(psi_i, psi_j.conj()) / 3.0)  # <n> = 3 on both code words
        r = ft.ideal_repair_fidelity(np.array(R_out), T, w, n_c, 1)
        self.assertAlmostEqual(r["F"], ft.kitten_repair_limit(), delta=1e-8)
        self.assertAlmostEqual(ft.kitten_repair_limit(), 0.935256, delta=1e-6)

    def test_optimal_recovery_of_unitary(self):
        n_c = 7
        B = embedded_basis(n_c, 1)
        V = unitary_group.rvs(B.shape[0], random_state=4)
        R_out = np.array([V @ np.outer(B[:, i], B[:, j].conj()) @ V.conj().T
                          for i, j in ((0, 0), (1, 1), (0, 1))])
        r = ft.optimal_recovery_fidelity(R_out, B)
        self.assertAlmostEqual(r["F_opt"], 1.0, delta=1e-9)
        self.assertLess(r["F_none"], 0.5)
        self.assertGreaterEqual(r["F_opt"], r["F_petz"] - 1e-12)


class Readout(unittest.TestCase):
    def test_nr2_equals_nr3(self):
        N = 100
        eta = ft.eta_from_teeth(np.full(6, 0.02, dtype=complex), N, DT, NC)
        F = {}
        for n_r in (2, 3):
            tm = ft.tail_model(NC, n_r)
            T, w = ft.tail_targets(NC, n_r)
            F[n_r] = ft.fidelity(tm, eta, ft.d2_output(NC, n_r), T, w)
        self.assertAlmostEqual(F[2], F[3], delta=1e-6)


class Cut(unittest.TestCase):
    def test_same_n_c_is_default(self):
        self.assertTrue(np.array_equal(ft.tail_input(None, NC, 2, NC), ft.d2_output(NC, 2)))
        o = ft.TailObjective(tail_us=20 * DT, n_c=NC, n_jobs=1, ramp_ns=8.0, gate_n_c=NC)
        self.assertTrue(np.array_equal(o.R0, ft.d2_output(NC, 2)))

    def test_cut_scores_within_trace_lost(self):
        big, small, N = 12, 10, 40
        R = ft.gate_output(None, big, 2)
        Rc = ft.tail_input(None, small, 2, big)
        self.assertEqual(Rc.shape[-1], small * 3 * 2)
        self.assertTrue(np.array_equal(Rc, R[:, :small * 6, :small * 6]))
        o = ft.TailObjective(tail_us=N * DT, n_c=small, n_jobs=1, ramp_ns=8.0, gate_n_c=big)
        lost = 1.0 - np.mean([np.trace(Rc[b]).real for b in (0, 1)])
        self.assertAlmostEqual(o.trace_cut, lost, delta=1e-15)
        eta = ft.eta_from_teeth(np.full(6, 0.02, dtype=complex), N, DT, small)
        F_cut = ft.score(eta, n_c=small, gate_n_c=big)["F_pro"]
        F_full = ft.score(eta, n_c=big)["F_pro"]
        self.assertLess(abs(F_cut - F_full), lost + 1e-5)


class NGates(unittest.TestCase):
    def test_tail_input_after_m_gates(self):
        import tempfile
        from DGRAPE import fwm_repeat as rp
        from DGRAPE import train_fwm_tail as tr
        u = 0.4 * ft.d2_pulse()[:120]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "u.npy")
            np.save(path, u)
            mk = lambda m: tr.make_objective(0.2, NC, 1, ft.READOUT, ft.ETA_BAND, ft.ETA_CAP, 0.0,  # noqa: E731
                                             gate_pulse=path, target=tr.default_target(m), n_gates=m)
            o1, o2 = mk(1), mk(2)
        try:
            self.assertTrue(np.array_equal(o1.R0, ft.tail_input(u, NC, 2)))
            gm = rp.gate_model(NC, 2)
            ref = rp.gate_step(gm, u, rp.gate_step(gm, u, rp.inputs(NC)))
            self.assertLess(np.abs(o2.R0 - ref).max(), 1e-12)
            self.assertEqual((o1.gate, o2.gate), ("X", "I"))
        finally:
            o1.close()
            o2.close()
        self.assertEqual([tr.default_target(m) for m in (1, 2, 3, 8)], ["X", "I", "X", "I"])
        with self.assertRaises(ValueError):
            tr.make_objective(0.2, NC, 1, ft.READOUT, ft.ETA_BAND, ft.ETA_CAP, 0.0, gate_pulse=None, n_gates=2)

    def test_deriv_scale(self):
        """--deriv-scale multiplies only the derivative weight; the default leaves TAIL_PENALTIES as they are."""
        from DGRAPE import train_fwm_tail as tr
        mk = lambda s: tr.make_objective(0.2, NC, 1, ft.READOUT, ft.ETA_BAND, ft.ETA_CAP, 0.0,  # noqa: E731
                                         deriv_scale=s)
        o1, o2 = mk(1.0), mk(0.1)
        try:
            self.assertEqual(o1.pen, ft.TAIL_PENALTIES)
            self.assertAlmostEqual(o2.pen["deriv"], 0.1 * ft.TAIL_PENALTIES["deriv"], places=15)
            self.assertEqual({k: v for k, v in o2.pen.items() if k != "deriv"},
                             {k: v for k, v in ft.TAIL_PENALTIES.items() if k != "deriv"})
            x = ft.eta_preimage(0.05 * np.exp(1j * np.linspace(0, 6, o1.N))[:, None] * np.ones((1, 2)))
            c1, _, t1 = o1.evaluate(x)
            c2, _, t2 = o2.evaluate(x)
            self.assertAlmostEqual(t2["deriv"], 0.1 * t1["deriv"], delta=1e-15 + 1e-12 * t1["deriv"])
            self.assertAlmostEqual(c1 - c2, 0.9 * t1["deriv"], delta=1e-12)
        finally:
            o1.close()
            o2.close()


class CodeWords(unittest.TestCase):
    def test_no_tail(self):
        r = {x["input"]: x for x in ft.code_word_scores(np.zeros((0, 2), dtype=complex), NC)}
        self.assertAlmostEqual(r["code words"]["F_vs_I"], 1.0, delta=1e-14)
        self.assertAlmostEqual(r["one-loss error words"]["F_vs_I"], 0.0, delta=1e-14)

    def test_mixture_is_the_average(self):
        eta = _rand_eta(30, 0.03, 7)
        inp = ft.code_word_inputs(NC)
        (Ec, pc), (Ee, pe) = inp["code words"], inp["one-loss error words"]
        self.assertAlmostEqual(pe, 3.0, delta=1e-12)
        mix = {"mix": (0.5 * Ec / pc + 0.5 * Ee / pe, 1.0)}
        for keep in (frozenset(), None):
            r = {x["input"]: x["F_vs_I"] for x in ft.code_word_scores(eta, NC, keep=keep)}
            F_mix = ft.code_word_scores(eta, NC, keep=keep, inputs=mix)[0]["F_vs_I"]
            self.assertAlmostEqual(F_mix, 0.5 * (r["code words"] + r["one-loss error words"]), delta=1e-12)

    def test_table_14(self):
        path = os.path.join(REPO_ROOT, "tables", "fwm_tail_H_code.csv")
        eta_path = os.path.join(REPO_ROOT, "pulses", "dgrape", "eta_X_tail_estg_T08_t2.npy")
        if not (os.path.exists(path) and os.path.exists(eta_path)):
            self.skipTest("table or tail not on this machine")
        import pandas as pd
        ref = pd.read_csv(path).query("tail == 'X tail'").set_index("input")
        for r in ft.code_word_scores(np.load(eta_path), 16):
            self.assertAlmostEqual(r["F_vs_I"], ref.F_vs_I[r["input"]], delta=1e-10)


class ShirolNoise(unittest.TestCase):
    """noise="shirol" (DGRAPE.model.shirol_collapse_ops, notebooks/tail_experiments.ipynb)."""

    def test_equals_aqec_table5(self):
        # Same dissipator as AQEC.master_equation.collapse_operators(n_c) (every channel, the default).
        from AQEC import master_equation as me
        m = Model(NC, 2, include_higher_order=True, noise="shirol")
        aq = me.collapse_operators(NC)
        rng = np.random.default_rng(5)
        A = rng.normal(size=(m.d, m.d)) + 1j * rng.normal(size=(m.d, m.d))
        rho = A @ A.conj().T
        rho /= np.trace(rho)
        s_dg = sum(c @ rho @ c.conj().T for _, c in m.c_ops)
        s_aq = sum(c @ rho @ c.conj().T for c in aq)
        self.assertLess(np.abs(s_dg - s_aq).max(), 1e-12)
        self.assertLess(np.abs(np.diag(m.Gdiag) - sum(c.conj().T @ c for c in aq)).max(), 1e-12)
        self.assertEqual(len(m.c_ops), 10)

    def test_keep_filters_and_default_unchanged(self):
        m = fwm.model_with_channels(NC, 2, keep={"a", "up_ef"}, noise="shirol")
        self.assertEqual([nm for nm, _ in m.c_ops], ["a", "up_ef"])
        m0, m1 = Model(NC, 2, True), Model(NC, 2, True, noise="dgrape")
        self.assertEqual([nm for nm, _ in m0.c_ops], ["a", "r", "s_ge", "s_ef"])
        self.assertTrue(all(np.array_equal(a[1], b[1]) for a, b in zip(m0.c_ops, m1.c_ops)))
        eta = _rand_eta(20, 0.03, 7)
        self.assertEqual(ft.score(eta, n_c=NC)["F_pro"], ft.score(eta, n_c=NC, noise="dgrape")["F_pro"])
        self.assertLess(ft.score(eta, n_c=NC, noise="shirol")["F_pro"], ft.score(eta, n_c=NC)["F_pro"])

    def test_frame_exact(self):
        # As Frame.test_equals_dgrape_frame, with Shirol's channels (|f><e| heating gains only a phase).
        N = 30
        eta = _rand_eta(N, 0.06, 0)
        R0 = ft.d2_output(NC, 2, noise="shirol")
        tm = ft.tail_model(NC, 2, mixing=1.0, noise="shirol")
        R_f, _ = ft.propagate(tm, eta, R0)
        m0 = Model(NC, 2, include_higher_order=True, noise="shirol")
        fw0 = fwm.fwm_terms(m0, 1.0, paths=(ft.N_REF,))
        fw0.Delta = np.array([tm.delta_ref, -tm.delta_ref])
        pf = np.real(np.diag(operators(NC, 2)["Pf"]))
        ph = np.exp(1j * tm.delta_ref * N * DT * pf)
        R_f_dg = ph[:, None] * R_f * ph[None, :].conj()
        R_d, _ = fwm.propagate(m0, fw0, np.zeros((N, 4)), eta, DT, 8, R0)
        self.assertLess(np.abs(R_d - R_f_dg).max(), 1e-6)

    def test_objective_fd(self):
        n_c, N = 7, 30
        kw = dict(tail_us=N * DT, n_c=n_c, band=(-30.0, 30.0), ramp_ns=8.0, n_jobs=1)
        x = 0.05 * ft.S_ETA * np.random.default_rng(4).normal(size=(N, 4))
        o = ft.TailObjective(noise="shirol", **kw)
        c, gx, terms = o.evaluate(x)
        self.assertLess(terms["F_pro"], ft.TailObjective(**kw).evaluate(x)[2]["F_pro"])
        for j in np.random.default_rng(6).choice(N * 4, 5, replace=False):
            h = 1e-3
            e = np.zeros(N * 4)
            e[j] = h
            fd = (o.cost(x.ravel() + e) - o.cost(x.ravel() - e)) / (2 * h)
            self.assertLess(abs(fd - gx[j]), 1e-6 * max(1.0, abs(gx).max()))


if __name__ == "__main__":
    unittest.main()
