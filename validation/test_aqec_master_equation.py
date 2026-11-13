#!/usr/bin/env python3
"""
test_aqec_master_equation.py

Checks for AQEC/master_equation.py, the frame-(i) Lindblad model of PReSPA
(Shirol et al., PRX 16, 021042 (2026), Eq. 3 + dissipators). Each check pins
something the Fig. 2 simulation in notebooks/aqec_experiments.ipynb §10 relies on:

  * physicality      -- trace 1, Hermitian, positive along the default trajectory
  * drives off       -- Fock |4> under D[a] alone decays as exp(-4 kappa_a t)
  * Appendix B link  -- with only Omega_1, Omega_2 and D[r], the error-state
                        population equals |<0|exp(-i H_B2 t)|0>|^2 for the 3x3
                        matrix of Eq. (B2). Exact, not perturbative: a readout jump
                        lands in |1,g,0>, which Eq. (3) never drives, so nothing
                        returns to |0,g,0>. This pins the drive matrix elements,
                        the readout operator and the Liouvillian vectorisation.
  * D1 shifts        -- higher_order_shifts reproduces the notebook's §3.2 table
  * qutip            -- an independent qutip.mesolve build agrees (frame (i))
  * heating/dephasing (the default since decision D3) -- rates from Table I and App. D,
                        detailed balance (P_e = 3.7% with drives off, paper 3.8%), g-f
                        coherence at exactly 1/T2gf, the drives-off model reproduces T2a,
                        and the shared rates equal DGRAPE.model.rates()

and the frame-(ii) solver used for Fig. 2(d):

  * chi -> 0         -- with every dispersive shift zero, frame (ii) IS frame (i); the full
                        rho, coherences included, must agree. Pins the drive-term phases.
  * populations      -- frame (ii) populations equal frame (i)'s without storage loss and
                        heating; heating adds ~1e-6, storage loss a second-order (double
                        loss) difference, but the Fig. 2(a)/(c) observables still agree to 1e-3
  * six tones        -- the Eq. (3) operators are the resonant matrix elements of the
                        six-tone operators
  * qutip            -- time-dependent qutip.mesolve on a superposition agrees (first 2 us)
  * Eq. (B2) rate    -- the Fig. 2(b) map reproduces Appendix B and the corner values

Run:
    python3 validation/test_aqec_master_equation.py [-v]
"""

import os
import sys
import unittest

import numpy as np
import scipy.linalg as sla

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from AQEC import device as dev                      # noqa: E402
from AQEC import master_equation as me              # noqa: E402

try:
    import qutip
    HAVE_QUTIP = True
except ImportError:
    HAVE_QUTIP = False

QUIET = dict(divide="ignore", over="ignore", invalid="ignore")
N_C = me.MIN_N_C
TWO_PI = 2 * np.pi


def fock_rho(n, j=me.G, m=0):
    psi = me.ket(n, j, m, N_C)
    return np.outer(psi, psi.conj())


class PhysicalityTest(unittest.TestCase):
    def test_trace_hermitian_positive(self):
        H = me.drive_hamiltonian(N_C) + me.higher_order_shifts(N_C)
        L = me.liouvillian(H, me.collapse_operators(N_C))
        t = np.arange(0.0, 25.0 + 1e-9, 0.25)
        rhos = me.propagate(L, fock_rho(4), t)
        np.testing.assert_allclose(np.trace(rhos, axis1=1, axis2=2), 1.0, atol=1e-10)
        herm = np.max(np.abs(rhos - rhos.conj().transpose(0, 2, 1)))
        self.assertLess(herm, 1e-12)
        with np.errstate(**QUIET):
            min_eig = min(np.linalg.eigvalsh(0.5 * (r + r.conj().T)).min() for r in rhos)
        self.assertGreater(min_eig, -1e-10)

    def test_nonuniform_grid_rejected(self):
        L = me.liouvillian(me.drive_hamiltonian(N_C), [])
        with self.assertRaises(ValueError):
            me.propagate(L, fock_rho(0), [0.0, 0.1, 0.3])


class DrivesOffTest(unittest.TestCase):
    def test_fock_decay_under_storage_loss(self):
        H = np.zeros((me.dim(N_C),) * 2, dtype=complex)
        c_ops = me.collapse_operators(N_C, transmon_decay="none", kappa_r=0.0,
                                      heating="none", dephasing=False)
        self.assertEqual(len(c_ops), 1)                 # only D[a] survives
        t = np.linspace(0.0, 100.0, 51)
        rhos = me.propagate(me.liouvillian(H, c_ops), fock_rho(4), t)
        P4 = me.fig2_observables(rhos, 4, N_C)["Png"]
        np.testing.assert_allclose(P4, np.exp(-4 * dev.kappa_a * t), atol=1e-10)


class AppendixBTest(unittest.TestCase):
    def test_error_state_matches_eq_B2(self):
        H = me.drive_hamiltonian(N_C)
        c_ops = me.collapse_operators(N_C, transmon_decay="none", kappa_a=0.0,
                                      heating="none", dephasing=False)
        t = np.linspace(0.0, 15.0, 61)
        rhos = me.propagate(me.liouvillian(H, c_ops), fock_rho(0), t)
        i0 = me.index(0, me.G, 0, N_C)
        P_me = np.real(rhos[:, i0, i0])

        om1, om2, kr = dev.OMEGA_1, dev.OMEGA_2, dev.kappa_r
        H_B2 = np.array([[0, om1, 0], [om1, 0, om2], [0, om2, -0.5j * kr]])
        with np.errstate(**QUIET):
            P_B2 = np.array([abs(sla.expm(-1j * H_B2 * tk)[0, 0]) ** 2 for tk in t])
        np.testing.assert_allclose(P_me, P_B2, atol=1e-10)


class JumpRatesTest(unittest.TestCase):
    """Heating and dephasing (default since D3 was adopted): rates, detailed balance, T2."""

    def test_rates(self):
        R = me.jump_rates()
        self.assertAlmostEqual(R["up_ge"] * 1e3, 0.35 + 0.42, delta=0.01)      # ms^-1, App. D
        self.assertAlmostEqual(R["up_ef"] / R["up_ge"], dev.T1_GE / dev.T1_EF, places=12)
        self.assertAlmostEqual(R["up_a"] * 1e3, 0.0444, delta=1e-3)
        Rth = me.jump_rates(drive_heating=False)
        self.assertAlmostEqual(Rth["up_ge"] * 1e3, 0.35, delta=0.01)
        self.assertEqual(me.jump_rates(kappa_a=0.0)["up_a"], 0.0)

    def test_rates_match_dgrape(self):
        from DGRAPE.model import rates
        R, D = me.jump_rates(drive_heating=False), rates()
        for k in ("phi_ge", "phi_gf", "up_a"):
            self.assertAlmostEqual(R[k], D[k], places=15, msg=k)
        self.assertAlmostEqual(R["up_ge"], D["up_q"], places=15)
        self.assertAlmostEqual(R["phi_a"] + R["up_ge"], D["phi_a"], places=15)  # heating part removed

    def test_drives_off_model_reproduces_T2a(self):
        """Storage |0>+|1> with drives off, all channels, thermal heating (T2a is an equilibrium
        measurement). The thermal g->e jumps dephase the storage through chi in H_disp; with the
        residual phi_a on top, the total pure dephasing should come back to 1/T2a - kappa_a/2.
        Measured: heating alone 0.360 ms^-1, total 2% above the T2a value."""
        psi = (me.ket(0, me.G, 0, N_C) + me.ket(1, me.G, 0, N_C)) / np.sqrt(2)
        H = me.dispersive_hamiltonian(N_C)
        target = 1 / dev.T2_CAVITY - dev.kappa_a / 2
        t = np.array([0.0, 300.0])
        for dephasing, lo, hi in ((False, 0.55, 0.70), (True, 0.97, 1.05)):
            L = me.liouvillian(H, me.collapse_operators(N_C, heating="thermal", dephasing=dephasing))
            s = me.storage_reduced(me.propagate(L, np.outer(psi, psi.conj()), t)[-1], N_C)
            rate = -np.log(abs(s[0, 1]) / 0.5) / t[-1] - 0.5 * dev.kappa_a
            self.assertTrue(lo < rate / target < hi, msg=f"dephasing={dephasing}: {rate / target:.3f}")

    def _transmon_steady_state(self, heating):
        H = np.zeros((me.dim(N_C),) * 2, dtype=complex)
        c_ops = me.collapse_operators(N_C, kappa_a=0.0, kappa_r=0.0, heating=heating)
        rho = me.propagate(me.liouvillian(H, c_ops), fock_rho(0), [0.0, 5000.0])[-1]
        p = np.real(np.diag(rho))
        return [p[me.index(0, j, 0, N_C)] for j in (me.G, me.E, me.F)]

    def test_detailed_balance(self):
        """Drives off: P_e/P_g = P_f/P_e = r. Thermal: 1.7% equilibrium. Driven: P_e ~ 3.7%,
        against the 3.8% the paper measures under the pumps (App. D)."""
        for heating in ("thermal", "driven"):
            r = me.jump_rates(drive_heating=(heating == "driven"))["r"]
            pg, pe, pf = self._transmon_steady_state(heating)
            self.assertAlmostEqual(pe / pg, r, places=8, msg=heating)
            self.assertAlmostEqual(pf / pe, r, places=8, msg=heating)
        self.assertAlmostEqual(self._transmon_steady_state("thermal")[1] / 0.983, 0.017 / 0.983,
                               delta=5e-4)
        self.assertAlmostEqual(self._transmon_steady_state("driven")[1], 0.037, delta=1e-3)

    def test_gf_coherence_decays_at_T2gf(self):
        H = np.zeros((me.dim(N_C),) * 2, dtype=complex)
        c_ops = me.collapse_operators(N_C, kappa_r=0.0, heating="none")
        psi = (me.ket(0, me.G, 0, N_C) + me.ket(0, me.F, 0, N_C)) / np.sqrt(2)
        t = np.linspace(0.0, 60.0, 7)
        rhos = me.propagate(me.liouvillian(H, c_ops), np.outer(psi, psi.conj()), t)
        coh = np.abs(rhos[:, me.index(0, me.G, 0, N_C), me.index(0, me.F, 0, N_C)])
        np.testing.assert_allclose(coh, 0.5 * np.exp(-t / dev.T2_GF_RAMSEY), rtol=1e-9)


class HigherOrderShiftTest(unittest.TestCase):
    def test_stage1_and_stage2_shifts_match_section_3_2(self):
        Hx = np.real(np.diag(me.higher_order_shifts(N_C)))
        E = lambda n, j, m: Hx[me.index(n, j, m, N_C)]
        stage1 = {n: E(n, me.F, 0) - E(n - 1, me.G, 0) for n in dev.CORRECTED_FOCK}
        stage2 = {n: E(n, me.G, 1) - E(n, me.F, 0) for n in dev.CORRECTED_FOCK}
        # §3.2: stage 1 = -K(n-1) + chi'_q n(n-1), stage 2 = -chi'_q n(n-1)
        for n in dev.CORRECTED_FOCK:
            self.assertAlmostEqual(stage1[n], -dev.K_a * (n - 1) + dev.chi_q_p * n * (n - 1), places=12)
            self.assertAlmostEqual(stage2[n], -dev.chi_q_p * n * (n - 1), places=12)
        kHz = {n: 1e3 * stage1[n] / TWO_PI for n in dev.CORRECTED_FOCK}
        self.assertAlmostEqual(kHz[1], 0.0, places=6)
        self.assertAlmostEqual(kHz[3], 4.8, places=6)
        self.assertAlmostEqual(kHz[5], 24.8, places=6)


def qutip_heating_dephasing(a, sq):
    """Default heating + dephasing operators, rebuilt from Table I and App. D without
    me.jump_rates: one up/down ratio r for g-e and e-f, storage at its own ratio, Ramsey dephasing."""
    r = dev.P_E_THERMAL / (1 - dev.P_E_THERMAL) + dev.DELTA_GAMMA_UP_DRIVE * dev.T1_GE
    up_a = dev.kappa_a * dev.P_CAVITY_THERMAL / (1 - dev.P_CAVITY_THERMAL)
    phi_ge = 1 / dev.T2_RAMSEY - 1 / (2 * dev.T1_GE)
    phi_gf = 1 / dev.T2_GF_RAMSEY - 1 / (2 * dev.T1_EF)
    phi_a = (1 / dev.T2_CAVITY - dev.kappa_a / 2
             - dev.P_E_THERMAL / (1 - dev.P_E_THERMAL) / dev.T1_GE)
    return [np.sqrt(r / dev.T1_GE) * sq(1, 0), np.sqrt(r / dev.T1_EF) * sq(2, 1),
            np.sqrt(up_a) * a.dag(),
            np.sqrt(2 * phi_ge) * sq(1, 1), np.sqrt(2 * phi_gf) * sq(2, 2),
            np.sqrt(2 * phi_a) * a.dag() * a]


def qutip_fig2(t, include_higher_order=True):
    """Fig. 2 curves from qutip.mesolve, built from qutip primitives only (no
    AQEC.master_equation operators). Same return layout as me.simulate_fig2."""
    a = qutip.tensor(qutip.destroy(N_C), qutip.qeye(3), qutip.qeye(2))
    r = qutip.tensor(qutip.qeye(N_C), qutip.qeye(3), qutip.destroy(2))
    b = lambda n, j, m: qutip.tensor(qutip.basis(N_C, n), qutip.basis(3, j), qutip.basis(2, m))
    H = 0
    for n in dev.CORRECTED_FOCK:
        H += dev.OMEGA_1 * b(n, 2, 0) * b(n - 1, 0, 0).dag()
        H += dev.OMEGA_2 * b(n, 0, 1) * b(n, 2, 0).dag()
    H = H + H.dag()
    if include_higher_order:
        for n in range(N_C):
            for j in range(3):
                for m in range(2):
                    H += ((-0.5 * dev.K_a + 0.5 * dev.chi_q_p * j) * n * (n - 1)
                          * b(n, j, m) * b(n, j, m).dag())
    sq = lambda i, k: qutip.tensor(qutip.qeye(N_C), qutip.basis(3, i) * qutip.basis(3, k).dag(),
                                   qutip.qeye(2))
    c_ops = [np.sqrt(dev.kappa_a) * a, np.sqrt(dev.kappa_r) * r,
             np.sqrt(1 / dev.T1_GE) * sq(0, 1), np.sqrt(1 / dev.T1_EF) * sq(1, 2)]
    c_ops += qutip_heating_dephasing(a, sq)
    proj = lambda n, j: sum(b(n, j, m) * b(n, j, m).dag() for m in range(2))
    Pe = sum(proj(n, 1) for n in range(N_C))
    Pf = sum(proj(n, 2) for n in range(N_C))
    out = {}
    for n0 in me.FIG2_INITIAL:
        e_ops = [proj(n0 + 1, 0), proj(n0 + 1, 1), Pe, Pf]
        res = qutip.mesolve(H, b(n0, 0, 0) * b(n0, 0, 0).dag(), t, c_ops, e_ops=e_ops,
                            options={"atol": 1e-12, "rtol": 1e-10, "nsteps": 100000})
        out[n0] = {"t": t, **{k: np.real(ex) for k, ex in zip(("Png", "Pne", "Pe", "Pf"),
                                                                 res.expect)}}
    return out


@unittest.skipUnless(HAVE_QUTIP, "qutip not installed")
class QutipCrossCheckTest(unittest.TestCase):
    def test_fig2_curves_agree_with_mesolve(self):
        sim = me.simulate_fig2(t_max=25.0, dt=0.25, include_higher_order=True)
        ref = qutip_fig2(sim[0]["t"], include_higher_order=True)
        for n0 in me.FIG2_INITIAL:
            for key in ("Png", "Pne", "Pe", "Pf"):
                np.testing.assert_allclose(sim[n0][key], ref[n0][key], atol=1e-6,
                                           err_msg=f"n0={n0} {key}")


class FrameIITest(unittest.TestCase):
    T = np.linspace(0.0, 15.0, 7)

    def test_zero_chi_reduces_to_frame_i(self):
        psi = (me.ket(0, me.G, 0, N_C) + me.ket(4, me.G, 0, N_C)) / np.sqrt(2)
        rho0 = np.outer(psi, psi.conj())
        H = me.drive_hamiltonian(N_C) + me.higher_order_shifts(N_C)
        ref = me.propagate(me.liouvillian(H, me.collapse_operators(N_C)), rho0, self.T)
        got = me.simulate_frame_ii(rho0, self.T, include_higher_order=True,
                                   chi_ge=0.0, chi_gf=0.0, chi_qr=0.0)
        self.assertLess(np.max(np.abs(got - ref)), 1e-8)

    def test_populations_match_frame_i_without_storage_loss(self):
        """§8.2: frame (i) is exact for populations -- but only while no storage photon is lost
        mid-correction (next test), and without heating. Dephasing is diagonal, so it stays
        exact. A heating jump, like a decay jump, carries a photon-number phase that frame (i)
        drops; with heating on the two frames differ by ~1e-6, which the second bound pins."""
        for heating, tol in (("none", 1e-7), ("driven", 1e-5)):
            L = me.liouvillian(me.drive_hamiltonian(N_C),
                               me.collapse_operators(N_C, kappa_a=0.0, heating=heating))
            for n0 in (0, 4):
                ref = me.propagate(L, fock_rho(n0), self.T)
                got = me.simulate_frame_ii(fock_rho(n0), self.T, kappa_a=0.0, heating=heating)
                diff = np.einsum("tii->ti", got) - np.einsum("tii->ti", ref)
                self.assertLess(np.max(np.abs(diff)), tol, msg=f"n0={n0} heating={heating}")

    def test_fig2_observables_match_frame_i(self):
        """With storage loss the frames differ at second order: a loss during path 5's g<->f
        Rabi carries the |4,g,0><5,f,0| coherence down, and a second loss lands it on path 3's
        driven pair |2,g,0><3,f,0|. Time-independent D[a] in frame (i) keeps its phase; in the
        lab (frame ii) it rotates at multiples of chi and averages out. Measured: <= 2.2e-3 on
        |3,g,0>, but <= 3.9e-4 on the Fig. 2(a)/(c) observables, which is what this pins.
        From |0> there is no double-loss path, so the only difference left is the heating phase
        (~2e-6; exact without heating)."""
        L = me.liouvillian(me.drive_hamiltonian(N_C), me.collapse_operators(N_C))
        t = np.arange(0.0, 25.0 + 1e-9, 0.25)
        for n0 in (0, 2, 4):
            ref = me.fig2_observables(me.propagate(L, fock_rho(n0), t), n0 + 1, N_C)
            got = me.fig2_observables(me.simulate_frame_ii(fock_rho(n0), t), n0 + 1, N_C)
            for key in ref:
                self.assertLess(np.max(np.abs(got[key] - ref[key])), 1e-3, msg=f"n0={n0} {key}")
        diff = me.simulate_frame_ii(fock_rho(0), t) - me.propagate(L, fock_rho(0), t)
        self.assertLess(np.max(np.abs(np.einsum("tii->ti", diff))), 1e-5)
        L0 = me.liouvillian(me.drive_hamiltonian(N_C), me.collapse_operators(N_C, heating="none"))
        diff = (me.simulate_frame_ii(fock_rho(0), t, heating="none")
                - me.propagate(L0, fock_rho(0), t))
        self.assertLess(np.max(np.abs(np.einsum("tii->ti", diff))), 1e-7)

    def test_coherence_differs_from_frame_i(self):
        """The reason frame (ii) exists: random-time transmon decays dephase photon numbers."""
        c_i = me.simulate_fig2d(pairs=((0, 2),), frame="i")[(0, 2)]["coherence"]
        c_ii = me.simulate_fig2d(pairs=((0, 2),), frame="ii")[(0, 2)]["coherence"]
        self.assertGreater(c_i - c_ii, 0.01)

    def test_six_tone_resonant_elements_are_eq3(self):
        eq3 = me.frame_ii_drive_terms(N_C)
        six = me.frame_ii_drive_terms(N_C, six_tones=True)
        self.assertEqual([w for _, w in eq3], [w for _, w in six])
        for (A3, _), (A6, _) in zip(eq3, six):
            mask = A3 != 0
            np.testing.assert_allclose(A6[mask], A3[mask], atol=1e-12)
            self.assertTrue(np.any(A6[~mask] != 0))      # the off-resonant pairings exist

    def test_storage_reduced_partial_trace(self):
        rho = fock_rho(3, me.F, 1)
        s = me.storage_reduced(rho, N_C)
        self.assertEqual(s.shape, (N_C, N_C))
        self.assertAlmostEqual(s[3, 3].real, 1.0)
        self.assertAlmostEqual(np.trace(s).real, 1.0)


class CorrectionRateTest(unittest.TestCase):
    def test_eq_b2_values(self):
        kHz = lambda o1, o2: 1e3 * me.b2_correction_rate(TWO_PI * 1e-3 * o1, TWO_PI * 1e-3 * o2)[0] / TWO_PI
        self.assertAlmostEqual(kHz(55, 160), 57.5, places=1)        # the experiment, Appendix B
        self.assertAlmostEqual(kHz(80, 180), 88.5, places=1)
        self.assertAlmostEqual(kHz(40, 180), 15.6, places=1)

    def test_map_marks_oscillating_region(self):
        o1 = TWO_PI * 1e-3 * np.array([50.0, 65.0])
        rate, osc = me.correction_rate_map(o1, [TWO_PI * 0.160])
        self.assertLess(osc[0, 0], 1e-9)              # Omega_1 = 50 kHz: below the split (58.9 kHz)
        self.assertGreater(osc[1, 0], 1e-3)           # Omega_1 = 65 kHz: oscillating


def qutip_frame_ii(rho0, t, include_higher_order=True):
    """Frame-(ii) model from qutip primitives with python coefficient functions."""
    b = lambda n, j, m: qutip.tensor(qutip.basis(N_C, n), qutip.basis(3, j), qutip.basis(2, m))
    a = qutip.tensor(qutip.destroy(N_C), qutip.qeye(3), qutip.qeye(2))
    r = qutip.tensor(qutip.qeye(N_C), qutip.qeye(3), qutip.destroy(2))
    H0 = 0 * a.dag() * a
    chi = (0.0, dev.chi_ge, dev.chi_gf)
    for n in range(N_C):
        for j in range(3):
            for m in range(2):
                e = -n * chi[j] - m * dev.chi_qr * (j == 1)
                if include_higher_order:
                    e += (-0.5 * dev.K_a + 0.5 * dev.chi_q_p * j) * n * (n - 1)
                H0 += e * b(n, j, m) * b(n, j, m).dag()
    H = [H0]
    for n in dev.CORRECTED_FOCK:
        for A, w in ((dev.OMEGA_1 * b(n, 2, 0) * b(n - 1, 0, 0).dag(), n * dev.chi_gf),
                     (dev.OMEGA_2 * b(n, 0, 1) * b(n, 2, 0).dag(), -n * dev.chi_gf)):
            H += [[A, (lambda t, w=w: np.exp(1j * w * t))],
                  [A.dag(), (lambda t, w=w: np.exp(-1j * w * t))]]
    sq = lambda i, k: qutip.tensor(qutip.qeye(N_C), qutip.basis(3, i) * qutip.basis(3, k).dag(),
                                   qutip.qeye(2))
    c_ops = [np.sqrt(dev.kappa_a) * a, np.sqrt(dev.kappa_r) * r,
             np.sqrt(1 / dev.T1_GE) * sq(0, 1), np.sqrt(1 / dev.T1_EF) * sq(1, 2)]
    c_ops += qutip_heating_dephasing(a, sq)
    dense = lambda q: q.to("dense")              # 42x42: dense is faster than qutip's CSR default
    H = [dense(H[0])] + [[dense(A), f] for A, f in H[1:]]
    rho_q = qutip.Qobj(rho0, dims=[[N_C, 3, 2], [N_C, 3, 2]])
    res = qutip.mesolve(H, rho_q, t, [dense(c) for c in c_ops],
                        options={"atol": 1e-11, "rtol": 1e-9, "nsteps": 1000000})
    return np.array([s.full() for s in res.states])


@unittest.skipUnless(HAVE_QUTIP, "qutip not installed")
class QutipFrameIITest(unittest.TestCase):
    def test_superposition_agrees_with_time_dependent_mesolve(self):
        """First 2 us only (the g<->f transient, where the drive phases matter): qutip's python
        coefficient callbacks take ~5 min for the full 15 us. test_zero_chi_reduces_to_frame_i
        covers the whole window."""
        psi = (me.ket(2, me.G, 0, N_C) + me.ket(4, me.G, 0, N_C)) / np.sqrt(2)
        rho0 = np.outer(psi, psi.conj())
        t = np.array([0.0, 2.0])
        got = me.simulate_frame_ii(rho0, t, include_higher_order=True)
        ref = qutip_frame_ii(rho0, t)
        self.assertLess(np.max(np.abs(got[-1] - ref[-1])), 1e-6)


if __name__ == "__main__":
    unittest.main()
