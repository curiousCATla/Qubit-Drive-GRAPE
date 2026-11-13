"""Checks for DGRAPE/aqec_ideal.py (the idealised Cornell AQEC on Shirol's chip) and DGRAPE/aqec_open.py (open-system
training of the real AQEC step: folded window target, exact gradient). Run from the repo root:
    python3 validation/test_aqec_ideal.py -v
Small truncation (n_c 10) so it runs in seconds."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.seterr(divide="ignore", over="ignore", invalid="ignore")

from DGRAPE import aqec_ideal as aq            # noqa: E402
from DGRAPE import fwm_repeat as fr            # noqa: E402
from DGRAPE import fwm                         # noqa: E402
from DGRAPE.model import operators             # noqa: E402

N_C = 10


def closed_model(n_c=N_C, n_r=1):
    return fwm.model_with_channels(n_c, n_r, frozenset(), True)


def error_inputs(n_c=N_C, n_r=1):
    """Clean one-loss error words a E a' / 3 (both code words lose a photon with probability weight 3)."""
    E = fr.inputs(n_c, n_r)
    a = operators(n_c, n_r)["a"]
    return np.einsum("ij,bjk,lk->bil", a, E, a.conj()) / 3.0


def F(R, m, N=0):
    return fr.score_state(R, m.n_c, m.n_r, "I" if N == 0 else fr.target_of(N))["F_pro"]


class Kraus(unittest.TestCase):
    def test_complete_on_g_subspace(self):
        for n_r in (1, 2):
            Kg, Ke = aq.kraus(N_C, n_r)
            S = Kg.conj().T @ Kg + Ke.conj().T @ Ke
            Pg = operators(N_C, n_r)["Pg"]
            np.testing.assert_allclose(S, Pg, atol=1e-12)

    def test_instant_map_fixes_code_and_error_words(self):
        m = closed_model()
        K = aq.kraus(N_C, 1)
        for R in (fr.inputs(N_C, 1), error_inputs()):
            out = aq.aqec_step(m, R, K, "forgive", 0.0, 0.0)
            self.assertAlmostEqual(F(out, m), 1.0, places=12)


class Closed(unittest.TestCase):
    def test_twist_cancels_drift_closed(self):
        """With no jumps, the full step (1 us + map + 1.2 us) is exact on code and error words: the undo after step 1
        and the pre-twist before step 2 cancel Kerr and chi on both branches."""
        m = closed_model()
        K = aq.kraus(N_C, 1)
        for R in (fr.inputs(N_C, 1), error_inputs()):
            out = aq.aqec_step(m, R, K, "forgive", aq.TAU1, aq.TAU2)
            self.assertAlmostEqual(F(out, m), 1.0, places=9)

    def test_no_twist_would_fail(self):
        """Control: without the pre-twist, Kerr over 2.2 us costs fidelity, so the test above is not empty."""
        m = closed_model()
        R = aq.idle(m, fr.inputs(N_C, 1), aq.TAU1 + aq.TAU2)
        self.assertLess(F(R, m), 0.9999)


class Modes(unittest.TestCase):
    def test_strict_equals_forgive_when_transmon_in_g(self):
        m = fwm.model_with_channels(N_C, 1, None, True)
        K = aq.kraus(N_C, 1)
        R = fr.inputs(N_C, 1)
        a = aq.aqec_step(m, R, K, "forgive")
        b = aq.aqec_step(m, R, K, "strict")
        np.testing.assert_allclose(a, b, atol=1e-13)

    def test_forgive_traces_transmon(self):
        rng = np.random.default_rng(0)
        d = 3 * N_C * 2
        X = rng.normal(size=(1, d, d)) + 1j * rng.normal(size=(1, d, d))
        R = X @ X.conj().transpose(0, 2, 1)
        R /= np.trace(R[0])
        out = aq.forgive(R, N_C, 2)
        self.assertAlmostEqual(np.real(np.trace(out[0])), 1.0, places=12)
        ops = operators(N_C, 2)
        self.assertAlmostEqual(np.real(np.trace(ops["Pg"] @ out[0])), 1.0, places=12)
        np.testing.assert_allclose(np.real(np.trace(ops["n_a"] @ out[0])), np.real(np.trace(ops["n_a"] @ R[0])),
                                   atol=1e-12)

    def test_zero_tau_is_map_only(self):
        m = fwm.model_with_channels(N_C, 1, None, True)
        K = aq.kraus(N_C, 1)
        R = error_inputs()
        np.testing.assert_allclose(aq.aqec_step(m, R, K, "forgive", 0.0, 0.0),
                                   aq.forgive(aq.apply_map(R, K), N_C, 1), atol=1e-14)


class Idle(unittest.TestCase):
    def test_idle_matches_many_small_steps(self):
        m = fwm.model_with_channels(N_C, 1, None, True)
        R = fr.inputs(N_C, 1)
        big = aq.idle(m, R, 0.2)
        import DGRAPE.lindblad_np as lnp
        small = lnp.propagate(m, np.zeros((100, 4)), R, 0.002)
        np.testing.assert_allclose(big, small, atol=1e-10)

    def test_trace_preserved(self):
        m = fwm.model_with_channels(N_C, 1, None, True)
        K = aq.kraus(N_C, 1)
        out = aq.aqec_step(m, fr.inputs(N_C, 1), K)
        for b in (0, 1):
            self.assertAlmostEqual(np.real(np.trace(out[b])), 1.0, places=6)   # n_c 10 truncation loses a little


class OpenTraining(unittest.TestCase):
    """aqec_open: the folded target Idle'(Reset'(T)) gives the same F as running window and reset forward, and the
    gradient of the whole cost (1 - F + cap + forbidden-state guard) matches finite differences."""

    @classmethod
    def setUpClass(cls):
        from DGRAPE import aqec_open as ao
        cls.ao, cls.fock_min = ao, ao.FORBID_FOCK_MIN
        ao.FORBID_FOCK_MIN = 8                            # below n_c 12, so the guard term is on (n_jobs 1: in-process)
        cls.N = 150                                       # 150 ns: the 48 ns ramp needs a flat top
        cls.obj = ao.OpenAqecObjective(cls.N, n_c=12, gate_n_c=12, n_jobs=1, lambda_forbid=1.0)
        cls.x = np.random.default_rng(0).normal(0, 8, (cls.N, 4))

    @classmethod
    def tearDownClass(cls):
        cls.ao.FORBID_FOCK_MIN = cls.fock_min

    def test_fold_equals_forward_chain(self):
        from DGRAPE import aqec_pulse as ap
        u = self.obj.to_physical(self.x)
        F_fold = self.obj.evaluate_physical(u, want_grad=False)[0]
        F_fwd = fr.score_state(ap.open_step(self.obj.m, self.obj.R0, u), 12, 1, "X")["F_pro"]
        self.assertAlmostEqual(F_fold, F_fwd, places=12)

    def test_gradient_matches_finite_differences(self):
        cost, g, terms = self.obj.evaluate(self.x)
        self.assertGreater(terms["C_forbid"], 0.0)       # the guard term is exercised
        dx = np.random.default_rng(1).normal(size=self.x.shape)
        h = 1e-4
        fd = (self.obj.cost(self.x + h * dx) - self.obj.cost(self.x - h * dx)) / (2 * h)
        self.assertAlmostEqual(fd / (g @ dx.ravel()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
