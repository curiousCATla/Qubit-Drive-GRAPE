"""Checks for DGRAPE/readout.py and DGRAPE/aqec_window.py (the realistic readout window of the Cornell AQEC step 2,
notebooks/AQEC_compare.ipynb §6.3). Run from the repo root:
    python3 validation/test_aqec_window.py -v
Small truncations (n_c 6-12), under a minute; the qutip check is skipped without qutip."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.seterr(divide="ignore", over="ignore", invalid="ignore")

from DGRAPE import aqec_ideal as aq            # noqa: E402
from DGRAPE import aqec_pulse as ap            # noqa: E402
from DGRAPE import aqec_window as aw           # noqa: E402
from DGRAPE import fwm                         # noqa: E402
from DGRAPE import fwm_repeat as fr            # noqa: E402
from DGRAPE import fwm_tail as ft              # noqa: E402
from DGRAPE import readout as ro               # noqa: E402
from DGRAPE.model import operators             # noqa: E402

N_C = 10
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mixed_input(n_c=N_C):
    """Code words with some one-loss error words: a stand-in for a gate's output."""
    E = fr.inputs(n_c, 1)
    a = operators(n_c, 1)["a"]
    return 0.9 * E + 0.1 * np.einsum("ij,bjk,lk->bil", a, E, a.conj()) / 3.0


def idle_spec(t_w=aq.TAU2):
    """No drive, perfect decision, an instant pi pulse at the end: the old window."""
    return aw.WindowSpec(drive=ro.DriveSpec("square", 0.0, 0.0), t_int=0.0, latency=t_w, pi=aw.PiSpec("ideal"),
                         t_w=t_w, decision="ideal")


def real_spec(**kw):
    s = aw.WindowSpec(drive=ro.DriveSpec("square", 5.0, 0.5), latency=0.2, pi=aw.PiSpec("gauss", 6.0), eta=0.3,
                      bias=0.4, nres_target=0.05)
    return s.replace(**kw)


class Limit(unittest.TestCase):
    """With zero drive, a perfect decision and an instant pi pulse at t_w, window + ideal reset is the old step."""

    @classmethod
    def setUpClass(cls):
        cls.m = aq.model(N_C, 1, "shirol")
        cls.K = aq.kraus(N_C, 1)
        cls.R = mixed_input()
        cls.W = aw.Window(idle_spec(), cls.m)

    def test_ideal_step(self):
        ref = aq.aqec_step(self.m, self.R, self.K, "forgive")
        out = aq.forgive(aw.real_step(self.W, self.R, self.K, twist=aq.phase(self.m, aq.TAU2)), N_C, 1)
        self.assertLess(np.abs(out - ref).max(), 1e-12)

    def test_calibrated_pretwist(self):
        F = lambda X: fr.score_state(X, N_C, 1, "I")["F_pro"]  # noqa: E731
        ref = aq.aqec_step(self.m, self.R, self.K, "forgive")
        out = aq.forgive(aw.real_step(self.W, self.R, self.K), N_C, 1)
        self.assertAlmostEqual(F(out), F(ref), places=12)

    def test_open_step(self):
        u = np.load(os.path.join(REPO, "pulses/dgrape/u_aqec_open_T15_w.npy"))
        ref = ap.open_step(self.m, self.R, u)
        out = aq.forgive(aw.open_step_real(self.W, self.R, u), N_C, 1)
        self.assertLess(np.abs(out - ref).max(), 1e-12)


class Pointers(unittest.TestCase):
    def test_steady_state_and_ring_down(self):
        eps = ro.midpoint_amplitude(5.0)
        dl = ro.detunings(1)[:2, 0]
        grid, _ = ro.pointers(np.r_[np.full(6000, eps), np.zeros(500)], dl)
        lam = 0.5 * ro.KAPPA_R + 1j * dl
        ss = -1j * eps / lam
        np.testing.assert_allclose(grid[6000], ss * (1 - np.exp(-lam * 6000 * ro.DT)), atol=1e-12)  # ring-up
        np.testing.assert_allclose(np.abs(ss) ** 2, 5.0, rtol=1e-12)
        np.testing.assert_allclose(grid[-1], grid[6000] * np.exp(-lam * 500 * ro.DT), atol=1e-12)

    def test_clear_empties_both_pointers(self):
        for kick in (1.0, 2.0):
            eps = ro.build_drive(ro.DriveSpec("clear", 5.0, 0.6, kick=kick, t_kick=0.1, t_dep1=0.08, t_dep2=0.12))
            grid, _ = ro.pointers(eps, ro.detunings(1)[:, 0])
            self.assertLess(np.abs(grid[-1]).max(), 1e-12)

    def test_snr_formula(self):
        """Matched filter: SNR^2 = 4 eta kappa_r int |alpha_e - alpha_g|^2 dt."""
        eps = ro.build_drive(ro.DriveSpec("kick", 4.0, 0.7, kick=2.0, t_kick=0.1))
        _, mid = ro.pointers(eps, ro.detunings(1)[:, 0])
        rec = ro.record(eps, eps.size, [(1, 0)])
        direct = np.sqrt(4 * 0.3 * ro.KAPPA_R * np.sum(np.abs(mid[:, 1] - mid[:, 0]) ** 2) * ro.DT)
        self.assertAlmostEqual(ro.snr(rec.s1_level[0], rec.s1_level[1], 0.3), direct, places=10)


class Confusion(unittest.TestCase):
    """The binned one-jump decision probability against a Monte-Carlo of jump times and Gaussian record noise."""

    def test_decay_during_readout(self):
        gam, eta, bias, n_int = 2.0, 0.3, 0.3, 600               # an exaggerated T1 of 0.5 us, so jumps matter
        eps = ro.build_drive(ro.DriveSpec("square", 5.0, n_int * ro.DT))
        binned = ro.record(eps, n_int, [(1, 0)], bin_steps=10)
        exact = ro.record(eps, n_int, [(1, 0)], bin_steps=1)     # one bin per ns: s1 of every jump time
        t = binned.bin_centres * ro.DT
        p_an = np.exp(-gam * n_int * ro.DT) * binned.p_e_level(eta, bias)[1] + \
            np.sum(gam * binned.bin_widths * np.exp(-gam * t) * binned.p_e_jump(eta, bias)[0])
        rng = np.random.default_rng(3)
        shots = 200_000
        tj = rng.exponential(1 / gam, shots)
        k = np.floor(tj / ro.DT).astype(int)
        s1 = np.where(k >= n_int, exact.s1_level[1], exact.s1_jump[0, np.minimum(k, n_int - 1)])
        mid = 0.5 * (exact.s1_level[0] + exact.s1_level[1])
        s = np.sqrt(eta) * s1 + rng.normal(size=shots)
        p_mc = np.mean(s > np.sqrt(eta) * mid + bias)
        self.assertLess(abs(p_an - p_mc), 4 * np.sqrt(p_mc * (1 - p_mc) / shots) + 2e-4)


@unittest.skipUnless(__import__("importlib").util.find_spec("qutip"), "qutip not installed")
class ChiArBruteForce(unittest.TestCase):
    """The chi_ar terms (exact linear-readout pointer formula) against a Lindblad simulation with the readout mode
    explicit (storage x readout, transmon held in g or e, no other channels)."""

    def test_readout_factor(self):
        import qutip as qt
        n_c, n_r, chi_ar = 5, 24, 3 * ro.CHI_AR
        spec = aw.WindowSpec(drive=ro.DriveSpec("square", 3.0, 0.5), t_int=0.5, latency=0.0, pi=aw.PiSpec("ideal"),
                             t_w=0.8, decision="ideal", chi_ar=chi_ar)
        m = fwm.model_with_channels(n_c, 1, frozenset(), False)
        for j in (0, 1):
            W = aw.Window(spec, m)
            W.P_level, W.P_jump = np.zeros(3), np.zeros_like(W.P_jump)            # no pi pulse: decision g
            psi = np.zeros(m.d, complex)
            psi[np.arange(n_c) * 3 + j] = 1 / np.sqrt(n_c)
            X = W.sec.to_sector(np.outer(psi, psi.conj())[None])
            on = W.sec.from_sector(W.post_forward("g", W.integrate_forward(X)[0]))[0]
            W.spec = W.spec.replace(readout_on=False)
            off = W.sec.from_sector(W.post_forward("g", W.integrate_forward(X)[0]))[0]
            fac = np.array([on[n * 3 + j, j] / off[n * 3 + j, j] for n in range(n_c)])
            a = qt.tensor(qt.destroy(n_c), qt.qeye(n_r))
            r = qt.tensor(qt.qeye(n_c), qt.destroy(n_r))
            eps = ro.build_drive(spec.drive)
            amp = lambda t: eps[min(int(t / ro.DT), eps.size - 1)] if t < eps.size * ro.DT else 0.0  # noqa: E731
            H = [ro.detunings(1)[j, 0] * r.dag() * r - chi_ar * a.dag() * a * r.dag() * r,
                 [r.dag(), amp], [r, lambda t: np.conj(amp(t))]]
            s0 = qt.tensor(qt.Qobj(np.ones(n_c) / np.sqrt(n_c)), qt.basis(n_r, 0))
            res = qt.mesolve(H, qt.ket2dm(s0), np.arange(W.n_w + 1) * ro.DT, [np.sqrt(ro.KAPPA_R) * r],
                             options={"atol": 1e-10, "rtol": 1e-9, "max_step": 5e-4})
            bf = res.states[-1].ptrace(0).full()[:, 0] * n_c
            self.assertGreater(np.abs(1 - fac).max(), 0.05)                       # the effect is not small
            self.assertLess(np.abs(fac - bf).max(), 2e-5)


class MapProperties(unittest.TestCase):
    """The realistic window: adjoint, re-scoring, trace, bin convergence."""

    @classmethod
    def setUpClass(cls):
        cls.n_c = 8
        cls.m = aq.model(cls.n_c, 1, "shirol")
        cls.W = aw.Window(real_spec(), cls.m)
        rng = np.random.default_rng(0)
        d = cls.m.d

        def rnd():
            X = rng.normal(size=(3, d, d)) + 1j * rng.normal(size=(3, d, d))
            return X + np.conj(np.swapaxes(X, 1, 2))
        cls.rho, cls.T = rnd(), rnd()

    def block_diag(self, X):
        return self.W.sec.from_sector(self.W.sec.to_sector(X))

    def test_adjoint(self):
        lhs = sum(np.vdot(self.T[b], self.W.apply(self.rho)[b]) for b in range(3))
        rhs = sum(np.vdot(self.W.adjoint(self.T)[b], self.rho[b]) for b in range(3))
        self.assertLess(abs(lhs - rhs) / abs(lhs), 1e-12)

    def test_terms_equal_apply(self):
        R = mixed_input(self.n_c)
        T, w = ft.tail_targets(self.n_c, 1, "I", "traced")
        tm = self.W.terms(R, T, w)
        F = ft.fidelity_from_outputs(self.W.apply(R), T, w)
        self.assertAlmostEqual(tm.fidelity(), F, places=12)
        W2 = aw.Window(real_spec(eta=0.6, bias=-0.2), self.m)                    # re-scoring = rebuilding
        self.assertAlmostEqual(tm.fidelity(0.6, -0.2), ft.fidelity_from_outputs(W2.apply(R), T, w), places=12)

    def test_trace_preserved(self):
        rng = np.random.default_rng(1)
        for j in range(3):
            psi = np.zeros(self.m.d, complex)
            psi[np.arange(self.n_c) * 3 + j] = rng.normal(size=self.n_c) + 1j * rng.normal(size=self.n_c)
            psi /= np.linalg.norm(psi)
            out = self.W.apply(np.outer(psi, psi.conj())[None])
            self.assertAlmostEqual(np.real(np.trace(out[0])), 1.0, places=9)

    def test_bins_converged(self):
        R = mixed_input(self.n_c)
        T, w = ft.tail_targets(self.n_c, 1, "I", "traced")
        F10 = self.W.terms(R, T, w).fidelity()
        F5 = aw.Window(real_spec(bin_ns=5), self.m).terms(R, T, w).fidelity()
        self.assertLess(abs(F10 - F5), 1e-5)


class OpenTrainingWindow(unittest.TestCase):
    """aqec_open with a realistic window: the folded target W'(T) equals running the window forward, and the gradient
    matches finite differences."""

    @classmethod
    def setUpClass(cls):
        from DGRAPE import aqec_open as ao
        cls.N = 150
        cls.spec = real_spec(drive=ro.DriveSpec("square", 4.0, 0.3))
        cls.obj = ao.OpenAqecObjective(cls.N, n_c=12, gate_n_c=12, n_jobs=1, lambda_forbid=0.0, channels="shirol",
                                       window=cls.spec)
        cls.x = np.random.default_rng(0).normal(0, 8, (cls.N, 4))

    def test_fold_equals_forward(self):
        u = self.obj.to_physical(self.x)
        F_fold = self.obj.evaluate_physical(u, want_grad=False)[0]
        W = aw.Window(self.spec, self.obj.m)
        F_fwd = fr.score_state(aw.open_step_real(W, self.obj.R0, u), 12, 1, "X")["F_pro"]
        self.assertAlmostEqual(F_fold, F_fwd, places=12)

    def test_gradient(self):
        cost, g, _ = self.obj.evaluate(self.x)
        dx = np.random.default_rng(1).normal(size=self.x.shape)
        h = 1e-4
        fd = (self.obj.cost(self.x + h * dx) - self.obj.cost(self.x - h * dx)) / (2 * h)
        self.assertAlmostEqual(fd / (g @ dx.ravel()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
