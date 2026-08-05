"""
Correctness suite for the JAX autodiff path.

Run from the repository root:

    python EST/test_grape_jax.py -v

Plain unittest, matching the repo convention (there is no pytest config).

The load-bearing test is `AnalyticGradientCrossCheck`: with only C1 active and
the constraint chain disabled, the JAX gradient must agree with
core.grape_core.fidelity_multi_state's hand-derived adjoint gradient, which is
already validated against QuTiP and against a frozen pre-refactor reference.
That anchors the new path to the old one. The finite-difference tests then cover
C2/C3/C4, which have no analytic counterpart to compare against.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from core import grape_core
from core.fourier_cutoff import out_of_band_energy_fraction
from core.grape_core import make_hamiltonian, make_ops
from EST import kitten_code
from EST.device import (BAND_MHZ, DT, EPS_MAX, RAMP_NS, make_hamiltonian_est,
                        ramp_envelope)
from EST.grape_jax import (amplitude_cost, make_constrainer, make_cost_and_grad,
                           make_cost_terms, propagate_trajectory)


def _random_states(n, M, seed):
    """Mirrors validation/test_grape_core_perf.py:113."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(M):
        v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        out.append(v / np.linalg.norm(v))
    return out


def _jax_args(n_t, n_c, psi_i_list, psi_f_list, est_device=True):
    H0_np, Hc_np = (make_hamiltonian_est(n_t, n_c) if est_device
                    else make_hamiltonian(n_t, n_c))
    A_np, _ = make_ops(n_t, n_c)
    return (jnp.asarray(H0_np, dtype=jnp.complex128),
            jnp.stack([jnp.asarray(h, dtype=jnp.complex128) for h in Hc_np]),
            jnp.asarray(A_np, dtype=jnp.complex128),
            jnp.asarray(np.stack(psi_i_list, axis=1), dtype=jnp.complex128),
            jnp.asarray(np.stack(psi_f_list, axis=1), dtype=jnp.complex128))


class AnalyticGradientCrossCheck(unittest.TestCase):
    """Check 1: JAX autodiff vs the validated hand-derived adjoint."""

    def test_c1_gradient_matches_grape_core(self):
        n_t, n_c, N, M, dt = 3, 6, 8, 4, 0.02
        rng = np.random.default_rng(4242)
        u = rng.uniform(-8.0, 8.0, size=(N, 4))
        psi_i_list = _random_states(n_t * n_c, M, 201)
        psi_f_list = _random_states(n_t * n_c, M, 202)

        # Same device on both sides: this test is about the GRADIENT ALGORITHM
        # (expm+autodiff vs eigh+adjoint), not about the Hamiltonian, so it uses
        # the repo's own make_hamiltonian rather than EST/device.py.
        H0, Hc, A, Psi_i, Psi_f = _jax_args(n_t, n_c, psi_i_list, psi_f_list,
                                            est_device=False)
        F_np, g_np = grape_core.fidelity_multi_state(
            u, *make_hamiltonian(n_t, n_c), psi_i_list, psi_f_list, dt,
            want_grad=True)

        # weights = (1,0,0,0): C1 only. constrain=None: identity chain.
        cg = make_cost_and_grad(H0, Hc, A, Psi_i, Psi_i, Psi_f, dt,
                                (1.0, 0.0, 0.0, 0.0), constrain=None)
        c_jax, g_jax = cg(jnp.asarray(u))

        # The JAX side returns the COST 1-F, the numpy side the FIDELITY F,
        # so the values are complementary and the gradients are sign-flipped.
        self.assertAlmostEqual(float(c_jax), 1.0 - F_np, places=10)
        np.testing.assert_allclose(np.asarray(g_jax), -g_np, rtol=1e-6, atol=1e-9)

    def test_trajectory_matches_eigh_propagation(self):
        """expm and eigh must produce the same states, not just the same gradient."""
        n_t, n_c, N, dt = 3, 6, 12, 0.02
        rng = np.random.default_rng(77)
        u = rng.uniform(-6.0, 6.0, size=(N, 4))
        H0_np, Hc_np = make_hamiltonian(n_t, n_c)
        psi0 = _random_states(n_t * n_c, 2, 5)

        traj = np.asarray(propagate_trajectory(
            jnp.asarray(u), jnp.asarray(H0_np, dtype=jnp.complex128),
            jnp.stack([jnp.asarray(h, dtype=jnp.complex128) for h in Hc_np]),
            jnp.asarray(np.stack(psi0, axis=1), dtype=jnp.complex128), dt))

        # Tolerance is 1e-9, not machine epsilon: expm (scaling-and-squaring
        # Pade) and eigh (V diag(exp) V^dag) are different algorithms for the
        # same matrix function, and their round-off accumulates independently
        # over the trajectory. Measured drift here is ~2e-11 after 12 steps.
        psi = np.stack(psi0, axis=1)
        for k, u_k in enumerate(u):
            Uk, _, _ = grape_core.step_data(H0_np, Hc_np, u_k, dt)
            psi = Uk @ psi
            np.testing.assert_allclose(traj[k + 1], psi, rtol=1e-9, atol=1e-10)


class FiniteDifferenceTest(unittest.TestCase):
    """
    Check 2: central differences against the full C_tot.

    This is what actually validates C2/C3/C4. Pattern (h, delta, 10 random
    samples) follows validation/test_grape_core_perf.py:194.
    """

    def _fd_check(self, weights, constrain, seed, dt=0.02, N=8, n_c=8):
        n_t = 3
        rng = np.random.default_rng(seed)
        u = rng.uniform(-4.0, 4.0, size=(N, 4))

        H0_np, Hc_np = make_hamiltonian_est(n_t, n_c)
        A_np, _ = make_ops(n_t, n_c)
        args = (jnp.asarray(H0_np, dtype=jnp.complex128),
                jnp.stack([jnp.asarray(h, dtype=jnp.complex128) for h in Hc_np]),
                jnp.asarray(A_np, dtype=jnp.complex128),
                jnp.asarray(kitten_code.cardinals(n_t, n_c), dtype=jnp.complex128),
                jnp.asarray(kitten_code.error_cardinals(n_t, n_c), dtype=jnp.complex128),
                jnp.asarray(kitten_code.gate_target("X", n_t, n_c), dtype=jnp.complex128),
                dt)

        cg = make_cost_and_grad(*args, weights, constrain=constrain)
        terms = make_cost_terms(*args, constrain=constrain)
        w = np.asarray(weights)

        def scalar(uu):
            t = terms(jnp.asarray(uu))
            return float(w @ np.array([t["c1"], t["c2"], t["c3"], t["c4"]]))

        _, grad = cg(jnp.asarray(u))
        grad = np.asarray(grad)

        h = 1e-6
        idx = np.random.default_rng(7)
        for k, j in zip(idx.integers(0, N, size=10), idx.integers(0, 4, size=10)):
            up, um = u.copy(), u.copy()
            up[k, j] += h
            um[k, j] -= h
            fd = (scalar(up) - scalar(um)) / (2 * h)
            with self.subTest(k=int(k), j=int(j)):
                self.assertAlmostEqual(fd, grad[k, j], delta=1e-4)

    def test_full_cost_unconstrained(self):
        self._fd_check((1.0, 0.7, 7.0, 1.0), None, seed=11)

    def test_full_cost_with_constraint_chain(self):
        """Differentiates through band-limit + ramp + control mask as well."""
        N, dt = 8, 0.02
        constrain = make_constrainer(N, dt, band=(-8.0, 8.0), ramp_ns=None,
                                     cmask=kitten_code.control_mask("X"))
        self._fd_check((1.0, 0.7, 7.0, 1.0), constrain, seed=12, dt=dt, N=N)

    def test_et_term_alone(self):
        self._fd_check((0.0, 1.0, 0.0, 0.0), None, seed=13)

    def test_velocity_term_alone(self):
        self._fd_check((0.0, 0.0, 1.0, 0.0), None, seed=14)

    def test_amplitude_term_alone(self):
        """Seeded so the pulse actually exceeds eps_max; otherwise C4 == 0
        identically and the test would pass vacuously."""
        rng = np.random.default_rng(15)
        u = rng.uniform(-40.0, 40.0, size=(8, 4))
        mag = np.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2)
        self.assertGreater(mag.max(), EPS_MAX, "test pulse does not exercise the cap")
        self.assertGreater(float(amplitude_cost(jnp.asarray(u))), 0.0)
        self._fd_check((0.0, 0.0, 0.0, 1.0), None, seed=15)


class ConstraintSatisfactionTest(unittest.TestCase):
    """Check 3: the constrained pulse actually obeys App. C's constraints."""

    N = 1000

    def _constrained(self, gate="X", **kw):
        rng = np.random.default_rng(3)
        x = rng.uniform(-30.0, 30.0, size=(self.N, 4))
        kw.setdefault("cmask", kitten_code.control_mask(gate))
        constrain = make_constrainer(self.N, DT, **kw)
        return np.asarray(constrain(jnp.asarray(x)))

    def test_bandlimit_alone_is_exact(self):
        """Without the ramp, out-of-band energy must be at round-off."""
        u = self._constrained(ramp_ns=None)
        # out_of_band_energy_fraction returns {'cavity': ..., 'transmon': ...}
        frac = max(out_of_band_energy_fraction(u, DT, BAND_MHZ, BAND_MHZ).values())
        self.assertLess(frac, 1e-20)

    def test_ramp_reintroduces_little_out_of_band_energy(self):
        """
        Measured, not assumed. Projecting then ramping means the envelope's own
        spectrum leaks back across the 50 MHz edge; this pins how much, so a
        change to RAMP_NS cannot silently break the band constraint.

        Measured value on this white-noise input is 3.4e-3 (0.34%), which is the
        worst case -- a trained pulse is smoother than noise. The alternative
        ordering (ramp then project) drives this to zero but leaves the endpoints
        at ~23% of mid-amplitude; see make_constrainer's docstring.
        """
        u = self._constrained()
        # out_of_band_energy_fraction returns {'cavity': ..., 'transmon': ...}
        frac = max(out_of_band_energy_fraction(u, DT, BAND_MHZ, BAND_MHZ).values())
        self.assertLess(frac, 1e-2, f"out-of-band fraction {frac:.2e}")

    def test_endpoints_ramped_down(self):
        u = self._constrained()
        mid = np.abs(u[self.N // 2 - 20:self.N // 2 + 20]).max()
        self.assertLess(np.abs(u[0]).max(), 0.02 * mid)
        self.assertLess(np.abs(u[-1]).max(), 0.02 * mid)

    def test_t_gate_zeroes_cavity_drive(self):
        u = self._constrained(gate="T")
        np.testing.assert_array_equal(u[:, :2], 0.0)
        self.assertGreater(np.abs(u[:, 2:]).max(), 0.0)

    def test_ramp_envelope_shape(self):
        env = ramp_envelope(self.N, DT, RAMP_NS)
        self.assertTrue(np.all((env >= 0.0) & (env <= 1.0)))
        self.assertAlmostEqual(env.max(), 1.0, places=12)
        n_ramp = int(round(RAMP_NS * 1e-3 / DT))
        self.assertTrue(np.all(env[n_ramp:-n_ramp] > 0.999))
        self.assertLess(env[0], 0.02)


class VelocityNormalizationTest(unittest.TestCase):
    """
    Pins the reason C3 is normalized (see velocity_variance_cost's docstring):
    the raw form is dimensionful and dt-dependent, and at dt = 1 ns it is large
    enough to make C1 and C2 numerically irrelevant under the paper's weights.
    """

    def _c3(self, dt, normalize):
        n_t, n_c = 3, 12
        N = int(round(0.2 / dt))
        H0_np, Hc_np = make_hamiltonian_est(n_t, n_c)
        constrain = make_constrainer(N, dt, cmask=kitten_code.control_mask("X"))
        x = jnp.asarray(np.random.default_rng(0).uniform(-15.0, 15.0, size=(N, 4)))
        traj = propagate_trajectory(
            constrain(x), jnp.asarray(H0_np, dtype=jnp.complex128),
            jnp.stack([jnp.asarray(h, dtype=jnp.complex128) for h in Hc_np]),
            jnp.asarray(kitten_code.cardinals(n_t, n_c), dtype=jnp.complex128), dt)
        from EST.grape_jax import velocity_variance_cost
        return float(velocity_variance_cost(traj, dt, normalize=normalize))

    def test_raw_form_would_swamp_the_fidelity_terms(self):
        """If this ever stops being true, the normalization can be revisited."""
        raw = self._c3(0.001, normalize=False)
        self.assertGreater(raw * 7.0, 50.0,
                           "raw C3 is no longer large; re-check the normalization")

    def test_normalized_form_is_order_one(self):
        norm = self._c3(0.001, normalize=True)
        self.assertGreater(norm, 1e-4)
        self.assertLess(norm * 7.0, 20.0,
                        "normalized C3 should be comparable to C1, C2 in [0,1]")

    def test_normalized_form_is_dt_stable(self):
        """The dimensionless form must not silently reweight when dt changes."""
        coarse, fine = self._c3(0.004, normalize=True), self._c3(0.001, normalize=True)
        self.assertLess(abs(coarse - fine) / max(coarse, fine), 1.0)


class DiagnosticsTest(unittest.TestCase):
    """
    Eqs. 6-8 are reconstructed, so they are pinned against two cases whose
    answers are known from the physics rather than from a reference number.
    """

    n_t, n_c, dt, N = 3, 16, 0.001, 100

    def _metrics(self, H0, Hc, u):
        from EST.diagnostics import (delta_qec, eta_mismatch, leakage_Ej,
                                     propagate_states)
        A, _ = make_ops(self.n_t, self.n_c)
        words = propagate_states(u, H0, Hc, kitten_code.logical_basis(self.n_t, self.n_c), self.dt)
        both = propagate_states(u, H0, Hc, np.concatenate(
            [kitten_code.cardinals(self.n_t, self.n_c),
             kitten_code.error_cardinals(self.n_t, self.n_c)], axis=1), self.dt)
        code, err = both[:, :, :6], both[:, :, 6:]
        return (delta_qec(words, A).max(), leakage_Ej(code, err, A).max(),
                eta_mismatch(code, err, A).max())

    def test_trivially_transparent_case_is_exactly_zero(self):
        """No Hamiltonian, no drive: nothing evolves, so every metric must vanish."""
        H0, Hc = make_hamiltonian_est(self.n_t, self.n_c)
        d, l, e = self._metrics(np.zeros_like(H0), Hc, np.zeros((self.N, 4)))
        for name, val in (("delta_qec", d), ("leakage", l), ("eta", e)):
            self.assertLess(val, 1e-12, f"{name} = {val:.3e} should be zero")

    def test_free_kerr_evolution_violates_only_the_eta_channel(self):
        """
        Under drift alone, Kerr is diagonal in Fock, so it preserves the code's
        photon-number structure: the Knill-Laflamme conditions and the error-space
        containment both still hold. But [K/2 a^dag^2 a^2, a] != 0, so the
        photon-loss image drifts away from the evolved error state -- App. A's
        fundamental obstruction, visible in eta and nowhere else.
        """
        H0, Hc = make_hamiltonian_est(self.n_t, self.n_c)
        d, l, e = self._metrics(H0, Hc, np.zeros((self.N, 4)))
        self.assertLess(d, 1e-12, "Kerr should not violate Knill-Laflamme")
        self.assertLess(l, 1e-10, "Kerr should not cause error-space leakage")
        self.assertGreater(e, 1e-6, "App. A's obstruction should appear in eta")

    def test_propagate_states_matches_grape_core_step_data(self):
        from EST.diagnostics import propagate_states
        H0, Hc = make_hamiltonian_est(self.n_t, self.n_c)
        u = np.random.default_rng(1).uniform(-5, 5, size=(8, 4))
        psi0 = kitten_code.cardinals(self.n_t, self.n_c)
        traj = propagate_states(u, H0, Hc, psi0, self.dt)
        psi = psi0.copy()
        for k, u_k in enumerate(u):
            psi = grape_core.step_data(H0, Hc, u_k, self.dt)[0] @ psi
            np.testing.assert_allclose(traj[k + 1], psi, rtol=1e-12, atol=1e-13)


class KittenCodeTest(unittest.TestCase):
    """The code definition itself -- cheap, and everything downstream assumes it."""

    def test_code_words(self):
        n_t, n_c = 3, 12
        B = kitten_code.logical_basis(n_t, n_c)
        self.assertAlmostEqual(abs(B[0, 0]), 1 / np.sqrt(2), places=12)   # |0>
        self.assertAlmostEqual(abs(B[4, 0]), 1 / np.sqrt(2), places=12)   # |4>
        self.assertAlmostEqual(abs(B[2, 1]), 1.0, places=12)              # |2>

    def test_error_words_are_photon_loss_images(self):
        n_t, n_c = 3, 12
        A, _ = make_ops(n_t, n_c)
        B, BE = kitten_code.logical_basis(n_t, n_c), kitten_code.error_basis(n_t, n_c)
        for j in range(2):
            img = A @ B[:, j]
            self.assertAlmostEqual(np.linalg.norm(img), np.sqrt(2.0), places=12)
            self.assertAlmostEqual(abs(np.vdot(BE[:, j], img / np.linalg.norm(img))),
                                   1.0, places=12)

    def test_error_cardinals_assertion_is_live(self):
        """error_cardinals asserts a code-specific coincidence; confirm it runs."""
        self.assertIsNotNone(kitten_code.error_cardinals(3, 12))

    def test_gate_targets_are_the_ideal_action(self):
        n_t, n_c = 3, 12
        B = kitten_code.logical_basis(n_t, n_c)
        for gate, U2 in kitten_code.IDEAL_LOGICAL_U.items():
            T = kitten_code.gate_target(gate, n_t, n_c)
            C = kitten_code.cardinals(n_t, n_c)
            # Applying U2 in the logical block reproduces the targets exactly.
            np.testing.assert_allclose(T, B @ (U2 @ (B.conj().T @ C)), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
