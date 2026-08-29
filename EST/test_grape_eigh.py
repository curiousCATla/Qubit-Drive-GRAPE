"""
Correctness suite for EST/grape_eigh.py, the pure-numpy eigh + analytic-adjoint
pipeline.

    python EST/test_grape_eigh.py -v

Three independent things are checked, in increasing order of strength:

  1. The C1 gradient against core.grape_core.fidelity_multi_state's own
     hand-derived adjoint. This mirrors EST/test_grape_jax.py's
     AnalyticGradientCrossCheck, but both sides are now numpy adjoints of the
     same form, so it should come out far tighter than that test's 1e-6.
  2. Central finite differences on C1, C2, C3, C4 individually and on C_tot with
     and without the constraint chain. This is what validates the C2 and C3
     costate injections and the reversed-order constraint adjoint, none of which
     has an analytic counterpart anywhere else in the repo.
  3. Agreement with EST/grape_jax.py's JAX autodiff gradient on the full
     objective. An independently hand-derived gradient matching autodiff to
     ~1e-10 rules out a whole class of silent objective bugs in both.

EST/test_grape_jax.py must keep passing unchanged; nothing here modifies it or
anything it imports.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)   # must precede jax.numpy
import jax.numpy as jnp

from core.grape_core import make_hamiltonian, fidelity_multi_state, make_ops, step_data
from EST import grape_eigh, kitten_code
from EST.device import (BAND_MHZ, DT, EPS_MAX, RAMP_NS, make_hamiltonian_est)
from EST.grape_jax import (make_constrainer as jax_constrainer,
                           make_cost_and_grad as jax_cost_and_grad,
                           propagate_trajectory as jax_propagate)

warnings.filterwarnings("ignore")


def _random_states(n, M, seed):
    """M normalized random complex vectors, mirroring test_grape_jax.py:40."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, M)) + 1j * rng.standard_normal((n, M))
    v /= np.linalg.norm(v, axis=0)
    return [v[:, m].copy() for m in range(M)]


def _est_system(n_t, n_c, gate="X"):
    """(H0, Hc, A, Psi0_code, Psi0_err, Psi_target) for the EsT device."""
    H0, Hc_list = make_hamiltonian_est(n_t, n_c)
    A, _ = make_ops(n_t, n_c)
    return (np.asarray(H0, dtype=complex),
            np.stack([np.asarray(h, dtype=complex) for h in Hc_list]),
            np.asarray(A, dtype=complex),
            np.asarray(kitten_code.cardinals(n_t, n_c), dtype=complex),
            np.asarray(kitten_code.error_cardinals(n_t, n_c), dtype=complex),
            np.asarray(kitten_code.gate_target(gate, n_t, n_c), dtype=complex))


def _jax_system(sysargs, dt):
    """The same tuple as JAX arrays, in make_cost_and_grad's argument order."""
    H0, Hc, A, P0c, P0e, Ptg = sysargs
    return (jnp.asarray(H0, dtype=jnp.complex128),
            jnp.asarray(Hc, dtype=jnp.complex128),
            jnp.asarray(A, dtype=jnp.complex128),
            jnp.asarray(P0c, dtype=jnp.complex128),
            jnp.asarray(P0e, dtype=jnp.complex128),
            jnp.asarray(Ptg, dtype=jnp.complex128), dt)


class AnalyticGradientCrossCheck(unittest.TestCase):
    """The anchor: against the OTHER track's hand-derived adjoint."""

    def test_c1_gradient_matches_grape_core(self):
        # Setup copied from EST/test_grape_jax.py:64-88 so the two anchors are
        # directly comparable. Uses core.grape_core.make_hamiltonian, not the EsT
        # device: this test is about the gradient algorithm, not the physics.
        n_t, n_c, N, M, dt = 3, 6, 8, 4, 0.02
        rng = np.random.default_rng(4242)
        u = rng.uniform(-8.0, 8.0, size=(N, 4))
        psi_i_list = _random_states(n_t * n_c, M, 201)
        psi_f_list = _random_states(n_t * n_c, M, 202)

        H0, Hc_list = make_hamiltonian(n_t, n_c)
        F_np, g_np = fidelity_multi_state(u, H0, Hc_list, psi_i_list,
                                          psi_f_list, dt, want_grad=True)

        Psi_i = np.stack(psi_i_list, axis=1)
        Psi_f = np.stack(psi_f_list, axis=1)
        A = np.eye(n_t * n_c, dtype=complex)   # inert: w2 = 0
        cost, grad, _ = grape_eigh.cost_and_grad(
            u, H0, np.stack(Hc_list), A, Psi_i, Psi_i, Psi_f, dt,
            (1.0, 0.0, 0.0, 0.0))

        # grape_eigh returns the COST 1-F, grape_core the FIDELITY F, so the
        # values are complementary and the gradients sign-flipped -- same
        # convention as test_grape_jax.py:85-88.
        self.assertAlmostEqual(cost, 1.0 - F_np, places=12)
        np.testing.assert_allclose(grad, -g_np, rtol=1e-6, atol=1e-9)

    def test_anchor_is_much_tighter_than_the_autodiff_one(self):
        """Both sides are numpy adjoints of the same Phi formula, so agreement
        should be at round-off, not at the 1e-6 the autodiff anchor needs."""
        n_t, n_c, N, M, dt = 3, 6, 8, 4, 0.02
        rng = np.random.default_rng(4242)
        u = rng.uniform(-8.0, 8.0, size=(N, 4))
        psi_i_list = _random_states(n_t * n_c, M, 201)
        psi_f_list = _random_states(n_t * n_c, M, 202)
        H0, Hc_list = make_hamiltonian(n_t, n_c)
        _, g_np = fidelity_multi_state(u, H0, Hc_list, psi_i_list, psi_f_list,
                                       dt, want_grad=True)
        Psi_i, Psi_f = np.stack(psi_i_list, axis=1), np.stack(psi_f_list, axis=1)
        _, grad, _ = grape_eigh.cost_and_grad(
            u, H0, np.stack(Hc_list), np.eye(n_t * n_c, dtype=complex),
            Psi_i, Psi_i, Psi_f, dt, (1.0, 0.0, 0.0, 0.0))
        rel = np.max(np.abs(grad + g_np)) / np.max(np.abs(g_np))
        self.assertLess(rel, 1e-12)


class PropagatorTest(unittest.TestCase):

    def test_matches_grape_core_step_data(self):
        """Same algorithm as core.grape_core, so the tolerance is round-off."""
        n_t, n_c, N, dt = 3, 6, 12, 0.02
        u = np.random.default_rng(77).uniform(-6.0, 6.0, size=(N, 4))
        H0, Hc_list = make_hamiltonian(n_t, n_c)
        Psi0 = np.stack(_random_states(n_t * n_c, 2, 5), axis=1)

        traj = grape_eigh.propagate_trajectory(u, H0, np.stack(Hc_list), Psi0, dt)
        psi = Psi0.copy()
        for k, u_k in enumerate(u):
            Uk, _, _ = step_data(H0, Hc_list, u_k, dt)
            psi = Uk @ psi
            np.testing.assert_allclose(traj[k + 1], psi, rtol=1e-12, atol=1e-13)

    def test_matches_jax_expm_trajectory(self):
        """Different algorithm (eigh vs scaling-and-squaring Pade), so the
        tolerance is 1e-9 -- same reasoning as test_grape_jax.py:100-104."""
        n_t, n_c, N, dt = 3, 6, 12, 0.02
        u = np.random.default_rng(77).uniform(-6.0, 6.0, size=(N, 4))
        H0, Hc_list = make_hamiltonian(n_t, n_c)
        Hc = np.stack(Hc_list)
        Psi0 = np.stack(_random_states(n_t * n_c, 2, 5), axis=1)

        got = grape_eigh.propagate_trajectory(u, H0, Hc, Psi0, dt)
        want = np.asarray(jax_propagate(
            jnp.asarray(u), jnp.asarray(H0, dtype=jnp.complex128),
            jnp.asarray(Hc, dtype=jnp.complex128),
            jnp.asarray(Psi0, dtype=jnp.complex128), dt))
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-10)

    def test_propagation_is_unitary(self):
        n_t, n_c, N, dt = 3, 6, 12, 0.02
        u = np.random.default_rng(5).uniform(-6.0, 6.0, size=(N, 4))
        H0, Hc_list = make_hamiltonian(n_t, n_c)
        Psi0 = np.stack(_random_states(n_t * n_c, 3, 5), axis=1)
        traj = grape_eigh.propagate_trajectory(u, H0, np.stack(Hc_list), Psi0, dt)
        for k in range(traj.shape[0]):
            np.testing.assert_allclose(np.linalg.norm(traj[k], axis=0),
                                       np.ones(3), rtol=1e-12)


class DegeneracyTest(unittest.TestCase):
    """
    The reason a hand-derived gradient is needed at all.

    d/dH of an eigendecomposition is singular where eigenvalues coincide. The
    Phi matrix hits 0/0 there, and core/grape_core.py:174-179 resolves it with
    the L'Hopital limit -i dt e^{-i dt w}. These tests build an EXACTLY
    degenerate H_k and check that branch fires and gives the right answer.
    """

    def _degenerate_system(self, n=8, M=2):
        levels = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0])[:n]
        rng = np.random.default_rng(31)
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        H0 = Q @ np.diag(levels.astype(complex)) @ Q.conj().T
        H0 = 0.5 * (H0 + H0.conj().T)
        B = rng.standard_normal((4, n, n)) + 1j * rng.standard_normal((4, n, n))
        Hc = 0.5 * (B + np.conj(np.transpose(B, (0, 2, 1))))
        Psi = np.stack(_random_states(n, M, 12), axis=1)
        Tgt = np.stack(_random_states(n, M, 13), axis=1)
        return H0, Hc, Psi, Tgt, levels

    def test_degenerate_branch_actually_fires(self):
        """A test that silently never enters the branch proves nothing."""
        H0, _, _, _, levels = self._degenerate_system()
        w = np.linalg.eigvalsh(H0)
        dw = w[:, None] - w[None, :]
        near = np.abs(dw) < grape_eigh.DEGEN_TOL
        # 3+2+3 repeated levels -> 9+4+9 = 22 near-degenerate pairs including
        # the diagonal, i.e. strictly more than the n diagonal entries alone.
        self.assertGreater(int(near.sum()), len(w))

    def test_gradient_is_finite_and_matches_jax_expm(self):
        """At u = 0 the step Hamiltonian IS the exactly degenerate H0."""
        n, M, N, dt = 8, 2, 4, 0.02
        H0, Hc, Psi, Tgt, _ = self._degenerate_system(n, M)
        u = np.zeros((N, 4))
        A = np.eye(n, dtype=complex)

        cost, grad, _ = grape_eigh.cost_and_grad(
            u, H0, Hc, A, Psi, Psi, Tgt, dt, (1.0, 0.0, 0.0, 0.0))
        self.assertTrue(np.all(np.isfinite(grad)),
                        "analytic gradient is not finite at exact degeneracy")

        cj, gj = jax_cost_and_grad(
            jnp.asarray(H0, dtype=jnp.complex128), jnp.asarray(Hc, dtype=jnp.complex128),
            jnp.asarray(A, dtype=jnp.complex128), jnp.asarray(Psi, dtype=jnp.complex128),
            jnp.asarray(Psi, dtype=jnp.complex128), jnp.asarray(Tgt, dtype=jnp.complex128),
            dt, (1.0, 0.0, 0.0, 0.0), constrain=None)(jnp.asarray(u))
        self.assertAlmostEqual(cost, float(cj), places=12)
        np.testing.assert_allclose(grad, np.asarray(gj), rtol=1e-9, atol=1e-12)

    def test_naive_eigendecomposition_derivative_would_divide_by_zero(self):
        """
        The failure the L'Hopital branch avoids, made explicit: without it the
        divided difference is 0/0 at a repeated eigenvalue.

        If this ever stops producing non-finite values, grape_eigh is still
        correct but its justification has changed -- re-derive before removing
        the branch.
        """
        H0, _, _, _, _ = self._degenerate_system()
        w = np.linalg.eigvalsh(H0)
        ew = np.exp(-1j * 0.02 * w)
        with np.errstate(divide="ignore", invalid="ignore"):
            Phi_naive = (ew[:, None] - ew[None, :]) / (w[:, None] - w[None, :])
        self.assertFalse(np.all(np.isfinite(Phi_naive)))


class ConstraintChainTest(unittest.TestCase):
    """
    The band-limit / ramp / mask chain and, critically, its adjoint.

    u = M(R(B(x)))  =>  dC/dx = B(R(M(dC/du))): the SAME operators applied in
    REVERSE order. Applying them forward in the adjoint still yields a
    plausible-looking gradient, which is why this is tested directly and again
    through finite differences below.
    """
    N, dt = 200, DT   # N*dt must exceed 2*RAMP_NS or ramp_envelope refuses

    def _chain(self):
        return grape_eigh.make_constrainer(self.N, self.dt, band=BAND_MHZ,
                                           ramp_ns=RAMP_NS,
                                           cmask=kitten_code.control_mask("X"))

    def test_forward_matches_jax_chain(self):
        constrain, _ = self._chain()
        jc = jax_constrainer(self.N, self.dt, band=BAND_MHZ, ramp_ns=RAMP_NS,
                             cmask=kitten_code.control_mask("X"))
        x = np.random.default_rng(3).uniform(-20.0, 20.0, size=(self.N, 4))
        np.testing.assert_allclose(constrain(x), np.asarray(jc(jnp.asarray(x))),
                                   rtol=1e-12, atol=1e-13)

    def test_adjoint_identity(self):
        """<a, C b> == <C* a, b> for the real inner product, for all a, b."""
        constrain, constrain_adj = self._chain()
        rng = np.random.default_rng(9)
        for _ in range(5):
            a = rng.standard_normal((self.N, 4))
            b = rng.standard_normal((self.N, 4))
            lhs = float(np.sum(a * constrain(b)))
            rhs = float(np.sum(constrain_adj(a) * b))
            self.assertAlmostEqual(lhs, rhs, delta=1e-9 * max(abs(lhs), 1.0))

    def test_adjoint_order_matters(self):
        """
        Guard against the plausible-looking bug: applying the chain in FORWARD
        order as the adjoint. It must not coincidentally agree.
        """
        constrain, constrain_adj = self._chain()
        g = np.random.default_rng(4).standard_normal((self.N, 4))
        wrong = constrain(g)          # forward order used as an 'adjoint'
        right = constrain_adj(g)
        self.assertGreater(np.max(np.abs(wrong - right)), 1e-6)


class FiniteDifferenceTest(unittest.TestCase):
    """
    Central differences against the analytic gradient.

    This is what actually validates the C2 and C3 costate injections and the
    constraint-chain adjoint: neither has an analytic reference anywhere else.
    Pattern (h, delta, 10 sampled entries) follows
    EST/test_grape_jax.py:114-182, which follows
    validation/test_grape_core_perf.py:194.
    """

    def _fd_check(self, weights, seed, dt=0.02, N=8, n_c=8, constrain=None,
                  constrain_adj=None, u_scale=4.0, h=1e-6):
        n_t = 3
        rng = np.random.default_rng(seed)
        x = rng.uniform(-u_scale, u_scale, size=(N, 4))
        H0, Hc, A, P0c, P0e, Ptg = _est_system(n_t, n_c)

        def value(xx):
            u = xx if constrain is None else constrain(xx)
            c, _, _ = grape_eigh.cost_and_grad(u, H0, Hc, A, P0c, P0e, Ptg, dt,
                                               weights, want_grad=False)
            return c

        u = x if constrain is None else constrain(x)
        _, grad, _ = grape_eigh.cost_and_grad(u, H0, Hc, A, P0c, P0e, Ptg, dt,
                                              weights)
        if constrain_adj is not None:
            grad = constrain_adj(grad)

        # Liveness guard. A test scalar that reduces to a norm is exactly
        # conserved under unitary propagation, so its gradient is identically
        # zero and an FD check would "pass" while measuring nothing.
        self.assertGreater(np.abs(grad).max(), 1e-6,
                           "gradient is ~zero; this FD check would be vacuous")

        idx = np.random.default_rng(7)
        for k, j in zip(idx.integers(0, N, size=10), idx.integers(0, 4, size=10)):
            xp, xm = x.copy(), x.copy()
            xp[k, j] += h
            xm[k, j] -= h
            fd = (value(xp) - value(xm)) / (2 * h)
            with self.subTest(k=int(k), j=int(j)):
                self.assertAlmostEqual(fd, grad[k, j], delta=1e-4)

    def test_c1_alone(self):
        self._fd_check((1.0, 0.0, 0.0, 0.0), seed=11)

    def test_c2_alone(self):
        self._fd_check((0.0, 1.0, 0.0, 0.0), seed=13)

    def test_c3_alone(self):
        self._fd_check((0.0, 0.0, 1.0, 0.0), seed=14)

    def test_c4_alone(self):
        """Seeded so the pulse actually exceeds eps_max; otherwise C4 == 0
        identically and the check would pass vacuously."""
        u = np.random.default_rng(15).uniform(-40.0, 40.0, size=(8, 4))
        mag = np.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2)
        self.assertGreater(mag.max(), EPS_MAX, "test pulse does not reach the cap")
        self._fd_check((0.0, 0.0, 0.0, 1.0), seed=15, u_scale=40.0)

    def test_full_cost_stage1(self):
        self._fd_check((1.0, 0.7, 7.0, 1.0), seed=11)

    def test_full_cost_stage2(self):
        self._fd_check((1.0, 0.1, 0.0, 1.0), seed=12)

    def test_full_cost_with_constraint_chain(self):
        """
        Differentiates through band-limit + ramp + control mask as well.

        h = 1e-4 here, not the 1e-6 used above, and the reason is the band-limit
        rather than anything about the adjoint. The projection is global in
        time, so perturbing ONE entry of x moves the whole pulse; at h = 1e-6
        the resulting cost difference is ~1e-10 against O(1) cost values, which
        is float64's cancellation floor and makes the DIFFERENCE QUOTIENT, not
        the gradient, the inaccurate side. Measured at these ten sample points:
        h = 1e-6 scatters by up to 1.6e-4 while h = 1e-4 converges onto the
        analytic value to ~1e-6. The gradient itself is pinned independently and
        far more tightly by CrossBackendGradientTest, which agrees with JAX
        autodiff to 3e-10 relative on this exact configuration.
        """
        N, dt = 200, DT
        constrain, constrain_adj = grape_eigh.make_constrainer(
            N, dt, band=BAND_MHZ, ramp_ns=RAMP_NS,
            cmask=kitten_code.control_mask("X"))
        self._fd_check((1.0, 0.7, 7.0, 1.0), seed=21, dt=dt, N=N, n_c=6,
                       constrain=constrain, constrain_adj=constrain_adj,
                       u_scale=20.0, h=1e-4)


class CrossBackendGradientTest(unittest.TestCase):
    """
    The headline check: a hand-derived adjoint against JAX autodiff, on the real
    objective with the production constraint chain.

    N = 200 because ramp_envelope needs N*dt > 2*RAMP_NS to leave a flat top --
    the same reason test_grape_jax.py's PreimagePersistenceTest uses 200.
    """
    n_t, n_c, N, dt = 3, 10, 200, DT

    def _compare(self, weights, rtol_grad=1e-9):
        sysargs = _est_system(self.n_t, self.n_c)
        x = np.random.default_rng(3).uniform(-20.0, 20.0, size=(self.N, 4))

        jc = jax_constrainer(self.N, self.dt, band=BAND_MHZ, ramp_ns=RAMP_NS,
                             cmask=kitten_code.control_mask("X"))
        cj, gj = jax_cost_and_grad(*_jax_system(sysargs, self.dt), weights,
                                   constrain=jc)(jnp.asarray(x))
        cj, gj = float(cj), np.asarray(gj)

        objective, _, _ = grape_eigh.build_gate_objective(
            "X", self.N, dt=self.dt, n_t=self.n_t, trunc_list=[self.n_c],
            weights=weights)
        cn, gn = objective(x.ravel())
        gn = gn.reshape(self.N, 4)

        self.assertAlmostEqual(cn, cj, delta=1e-9 * max(abs(cj), 1.0))
        rel = np.max(np.abs(gn - gj)) / np.max(np.abs(gj))
        self.assertLess(rel, rtol_grad,
                        f"gradient disagrees with JAX autodiff at rel={rel:.3e}")

    def test_stage1_weights(self):
        # C3 carries w3 = 7 and goes through arccos near 1, which costs a few
        # digits relative to the other terms; 1e-8 still pins every injection.
        self._compare((1.0, 0.7, 7.0, 1.0), rtol_grad=1e-8)

    def test_stage2_weights(self):
        self._compare((1.0, 0.1, 0.0, 1.0))

    def test_ord_weights(self):
        """w2 = w3 = 0: the Ord control comes from the same code path."""
        self._compare((1.0, 0.0, 0.0, 1.0))

    def test_report_terms_agree(self):
        from EST.grape_jax import make_cost_terms
        sysargs = _est_system(self.n_t, self.n_c)
        x = np.random.default_rng(3).uniform(-20.0, 20.0, size=(self.N, 4))
        jc = jax_constrainer(self.N, self.dt, band=BAND_MHZ, ramp_ns=RAMP_NS,
                             cmask=kitten_code.control_mask("X"))
        jt = make_cost_terms(*_jax_system(sysargs, self.dt), constrain=jc)
        want = {k: float(v) for k, v in jt(jnp.asarray(x)).items()}

        _, report, _ = grape_eigh.build_gate_objective(
            "X", self.N, dt=self.dt, n_t=self.n_t, trunc_list=[self.n_c],
            weights=(1.0, 0.7, 7.0, 1.0))
        got = report(x.ravel())[self.n_c]
        for k in ("c1", "c2", "c3", "c4"):
            self.assertAlmostEqual(got[k], want[k], places=10, msg=k)


class ObjectiveInterfaceTest(unittest.TestCase):
    """build_gate_objective must be drop-in for grape_jax's, since train drivers
    and comparison scripts call them interchangeably."""

    def test_returns_scipy_ready_triple(self):
        N = 200
        objective, report, constrain_np = grape_eigh.build_gate_objective(
            "X", N, trunc_list=[6])
        x = np.random.default_rng(1).uniform(-10.0, 10.0, size=N * 4)
        cost, grad = objective(x)
        self.assertIsInstance(cost, float)
        self.assertEqual(grad.shape, (N * 4,))
        self.assertEqual(constrain_np(x).shape, (N, 4))
        self.assertEqual(sorted(report(x)[6]), ["c1", "c2", "c3", "c4"])

    def test_control_mask_is_applied(self):
        """The T gate is transmon-only; its cavity columns must be exactly 0."""
        N = 200
        _, _, constrain_np = grape_eigh.build_gate_objective(
            "T", N, trunc_list=[6])
        u = constrain_np(np.random.default_rng(2).uniform(-10.0, 10.0, size=N * 4))
        np.testing.assert_allclose(u[:, :2], 0.0, atol=1e-14)
        self.assertGreater(np.abs(u[:, 2:]).max(), 1e-3)

    def test_truncation_average(self):
        """Multi-truncation cost is the mean over trunc_list (Heeres Eq. 23)."""
        N = 200
        x = np.random.default_rng(1).uniform(-10.0, 10.0, size=N * 4)
        singles = []
        for n_c in (6, 8):
            obj, _, _ = grape_eigh.build_gate_objective("X", N, trunc_list=[n_c])
            singles.append(obj(x)[0])
        obj_both, _, _ = grape_eigh.build_gate_objective("X", N, trunc_list=[6, 8])
        self.assertAlmostEqual(obj_both(x)[0], np.mean(singles), places=12)


if __name__ == "__main__":
    unittest.main()
