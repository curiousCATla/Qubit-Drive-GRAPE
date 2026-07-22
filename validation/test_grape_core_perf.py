#!/usr/bin/env python3
"""
test_grape_core_perf.py

Regression suite protecting grape_core.py/optimizer.py behavior across the
performance refactor described in README.md's "Performance" section:

  1. grape_core.fidelity_grad / fidelity_multi_state were rewritten around a
     shared, batched `_fidelity_core` (dedup eigendecomposition/gradient-basis
     rotation across state pairs, batch the per-timestep eigh).
  2. optimizer.optimize_multi_state_pulse reuses one joblib Parallel pool
     across the whole L-BFGS-B run instead of recreating it every call.

Strategy (no pytest dependency -- stdlib unittest only):
  - `_RefImplementation` is a verbatim, frozen copy of grape_core.py's
    PRE-REFACTOR step_data/fidelity_grad/fidelity_multi_state, captured
    before any production code was touched. It is the "old behavior" oracle
    that the new grape_core.py must reproduce to near machine precision.
  - `FiniteDifferenceGradientTest` independently checks the analytic
    gradient against central differences, so correctness doesn't rely
    solely on matching the (possibly also-buggy) old implementation.
  - `EndToEndSmokeTest` re-runs small optimize_multi_state_pulse/refine_pulse/
     refine_pulse_dt problems and compares final fidelity against numbers
     captured from the pre-refactor code (see the BASELINE_* constants).

Run: python3 test_grape_core_perf.py [-v]
"""
import os
import sys
import time
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.grape_core import make_hamiltonian, basis_state
from core import grape_core
from core import optimizer


# ============================================================================
# Frozen reference implementation (verbatim copy of grape_core.py as it
# existed BEFORE the batched-_fidelity_core refactor). Do not "fix" this to
# match new behavior -- it exists specifically to detect behavior drift.
# ============================================================================

def _ref_step_data(H0, Hc, u_k, dt):
    Hk = H0 + u_k[0] * Hc[0] + u_k[1] * Hc[1] + u_k[2] * Hc[2] + u_k[3] * Hc[3]
    w, V = np.linalg.eigh(Hk)
    Uk = V @ np.diag(np.exp(-1j * dt * w)) @ V.conj().T
    return Uk, w, V


def _ref_fidelity_grad(u, H0, Hc, psi_i, psi_f, dt, want_grad=True):
    N = u.shape[0]
    phi = [psi_i.copy()]
    Ws, Vs, Us = [], [], []
    psi = psi_i.copy()
    for k in range(N):
        Uk, w, V = _ref_step_data(H0, Hc, u[k], dt)
        psi = Uk @ psi
        phi.append(psi)
        Us.append(Uk); Ws.append(w); Vs.append(V)
    v = np.vdot(psi_f, psi)
    F = np.abs(v) ** 2
    if not want_grad:
        return F, None
    lam = [None] * (N + 1)
    lam[N] = psi_f.copy()
    for k in range(N - 1, -1, -1):
        lam[k] = Us[k].conj().T @ lam[k + 1]

    grad = np.zeros((N, 4))
    for k in range(N):
        w, V = Ws[k], Vs[k]
        p = V.conj().T @ phi[k]
        q = V.conj().T @ lam[k + 1]
        ew = np.exp(-1j * dt * w)
        dw = w[:, None] - w[None, :]
        near = np.abs(dw) < 1e-10
        dw_safe = np.where(near, 1.0, dw)
        Phi = (ew[:, None] - ew[None, :]) / dw_safe
        Phi = np.where(near, (-1j * dt * ew)[:, None], Phi)
        qc = q.conj()
        for j in range(4):
            X = V.conj().T @ Hc[j] @ V
            dv = qc @ ((Phi * X) @ p)
            grad[k, j] = 2.0 * np.real(np.conj(v) * dv)
    return F, grad


def _ref_fidelity_multi_state(u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=True):
    M = len(psi_i_list)
    total_F = 0.0
    total_grad = np.zeros_like(u)
    for psi_i, psi_f in zip(psi_i_list, psi_f_list):
        F_i, grad_i = _ref_fidelity_grad(u, H0, Hc, psi_i, psi_f, dt, want_grad=want_grad)
        total_F += F_i
        if want_grad and grad_i is not None:
            total_grad += grad_i
    F_avg = total_F / M
    grad_avg = total_grad / M if want_grad else None
    return F_avg, grad_avg


# ============================================================================
# Shared random-problem helpers
# ============================================================================

def _random_states(n, M, seed):
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(M):
        v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        v /= np.linalg.norm(v)
        states.append(v)
    return states


# (n_t, n_c, N, M, dt, u_seed, state_seed)
EQUIVALENCE_CONFIGS = [
    (2, 4, 6, 1, 0.05, 1, 101),
    (2, 4, 6, 2, 0.05, 2, 102),
    (3, 5, 8, 2, 0.02, 3, 103),
    (3, 8, 10, 1, 0.01, 4, 104),
    (2, 6, 12, 2, 0.002, 5, 105),
    (3, 4, 3, 2, 0.002, 6, 106),  # tiny N, exercises chunking boundary trivially
]


class FidelityCoreEquivalenceTest(unittest.TestCase):
    """grape_core.fidelity_grad/fidelity_multi_state must match the frozen
    pre-refactor reference implementation to near machine precision -- same
    math, only reordered/batched."""

    def _build(self, n_t, n_c, N, M, dt, u_seed, state_seed):
        H0, Hc = make_hamiltonian(n_t, n_c)
        n = H0.shape[0]
        rng = np.random.default_rng(u_seed)
        u = rng.uniform(-10.0, 10.0, size=(N, 4))
        psi_i_list = _random_states(n, M, state_seed)
        psi_f_list = _random_states(n, M, state_seed + 1)
        return H0, Hc, u, psi_i_list, psi_f_list

    def test_fidelity_multi_state_matches_reference(self):
        for cfg in EQUIVALENCE_CONFIGS:
            n_t, n_c, N, M, dt, u_seed, state_seed = cfg
            with self.subTest(cfg=cfg):
                H0, Hc, u, psi_i_list, psi_f_list = self._build(*cfg)

                F_new, grad_new = grape_core.fidelity_multi_state(
                    u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=True
                )
                F_ref, grad_ref = _ref_fidelity_multi_state(
                    u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=True
                )

                np.testing.assert_allclose(F_new, F_ref, rtol=1e-8, atol=1e-10)
                np.testing.assert_allclose(grad_new, grad_ref, rtol=1e-6, atol=1e-8)

    def test_fidelity_multi_state_matches_reference_no_grad(self):
        for cfg in EQUIVALENCE_CONFIGS:
            n_t, n_c, N, M, dt, u_seed, state_seed = cfg
            with self.subTest(cfg=cfg):
                H0, Hc, u, psi_i_list, psi_f_list = self._build(*cfg)
                F_new, grad_new = grape_core.fidelity_multi_state(
                    u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=False
                )
                F_ref, grad_ref = _ref_fidelity_multi_state(
                    u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=False
                )
                self.assertIsNone(grad_new)
                np.testing.assert_allclose(F_new, F_ref, rtol=1e-8, atol=1e-10)

    def test_fidelity_grad_single_state_matches_reference(self):
        # fidelity_grad(u, H0, Hc, psi_i, psi_f, dt) must equal the M=1 case
        # of fidelity_multi_state's per-state math (unaveraged).
        for cfg in EQUIVALENCE_CONFIGS:
            n_t, n_c, N, M, dt, u_seed, state_seed = cfg
            with self.subTest(cfg=cfg):
                H0, Hc, u, psi_i_list, psi_f_list = self._build(*cfg)
                psi_i, psi_f = psi_i_list[0], psi_f_list[0]

                F_new, grad_new = grape_core.fidelity_grad(u, H0, Hc, psi_i, psi_f, dt, want_grad=True)
                F_ref, grad_ref = _ref_fidelity_grad(u, H0, Hc, psi_i, psi_f, dt, want_grad=True)

                np.testing.assert_allclose(F_new, F_ref, rtol=1e-8, atol=1e-10)
                np.testing.assert_allclose(grad_new, grad_ref, rtol=1e-6, atol=1e-8)


class FiniteDifferenceGradientTest(unittest.TestCase):
    """Independent correctness check: analytic gradient vs central differences,
    doesn't rely on the reference implementation being bug-free."""

    def test_finite_difference(self):
        n_t, n_c, N, M, dt = 3, 6, 8, 2, 0.02
        H0, Hc = make_hamiltonian(n_t, n_c)
        n = H0.shape[0]
        rng = np.random.default_rng(999)
        u = rng.uniform(-8.0, 8.0, size=(N, 4))
        psi_i_list = _random_states(n, M, 201)
        psi_f_list = _random_states(n, M, 202)

        F0, grad = grape_core.fidelity_multi_state(u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=True)

        h = 1e-6
        rng_idx = np.random.default_rng(7)
        sample_k = rng_idx.integers(0, N, size=10)
        sample_j = rng_idx.integers(0, 4, size=10)
        for k, j in zip(sample_k, sample_j):
            u_plus = u.copy(); u_plus[k, j] += h
            u_minus = u.copy(); u_minus[k, j] -= h
            F_plus, _ = grape_core.fidelity_multi_state(u_plus, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=False)
            F_minus, _ = grape_core.fidelity_multi_state(u_minus, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=False)
            fd = (F_plus - F_minus) / (2 * h)
            with self.subTest(k=int(k), j=int(j)):
                self.assertAlmostEqual(fd, grad[k, j], delta=1e-4)


# ============================================================================
# End-to-end smoke tests against pre-refactor baselines
#
# Baselines captured by running today's (pre-refactor) optimizer.py/
# grape_core.py once with these exact configs -- see the git history of this
# file / the plan file for the capture script. Toy problem: n_t=2 transmon,
# Fock |g,0> <-> |g,1> transfer(s), far cheaper than any real gate but
# exercises the same code paths (M=1 and M=2, optimize/refine/refine_dt).
# ============================================================================

def _toy_state_pairs_m1(n_c, n_t=2):
    return [(basis_state(n_t, n_c, 0, 0), basis_state(n_t, n_c, 0, 1))]


def _toy_state_pairs_m2(n_c, n_t=2):
    a = basis_state(n_t, n_c, 0, 0)
    b = basis_state(n_t, n_c, 0, 1)
    return [(a, b), (b, a)]


_TOY_PENALTIES = {'deriv': 0.0001, 'boundary': 0.0002, 'amp': 0.0001, 'amp_max': 40.0}

BASELINE_OPT_M1_F = 0.36741513627087313
BASELINE_OPT_M2_F = 0.36741513103490353
BASELINE_N550_BASE_F = 0.9951110678161428
BASELINE_REFINE_PULSE_F = 0.9989711824612416
BASELINE_REFINE_DT_F = 0.36775971001479013

# End-to-end L-BFGS-B trajectories can pick up tiny floating-point
# differences (parallel reduction order, batched vs per-step eigh) that
# nudge the exact iterate without changing the underlying optimization
# problem -- allow a modest absolute tolerance rather than requiring
# bit-identical convergence.
_E2E_ATOL = 3e-3


class EndToEndSmokeTest(unittest.TestCase):
    def test_optimize_multi_state_pulse_m1(self):
        u_opt, info = optimizer.optimize_multi_state_pulse(
            get_state_pairs=_toy_state_pairs_m1,
            trunc_list=[6, 8], n_t=2, N=25,
            warm_start_amp=6.0, warm_start_cutoff_frac=0.3, warm_start_seed=7,
            penalties=_TOY_PENALTIES, n_jobs=2, maxiter=150, verbose=False,
        )
        self.assertAlmostEqual(info['final_fidelity'], BASELINE_OPT_M1_F, delta=_E2E_ATOL)

    def test_optimize_multi_state_pulse_m2_dedup_path(self):
        u_opt, info = optimizer.optimize_multi_state_pulse(
            get_state_pairs=_toy_state_pairs_m2,
            trunc_list=[6, 8], n_t=2, N=25,
            warm_start_amp=6.0, warm_start_cutoff_frac=0.3, warm_start_seed=7,
            penalties=_TOY_PENALTIES, n_jobs=2, maxiter=150, verbose=False,
        )
        self.assertAlmostEqual(info['final_fidelity'], BASELINE_OPT_M2_F, delta=_E2E_ATOL)

    def test_refine_pulse(self):
        u_base, info_base = optimizer.optimize_multi_state_pulse(
            get_state_pairs=_toy_state_pairs_m1,
            trunc_list=[6, 8], n_t=2, N=550,
            warm_start_amp=6.0, warm_start_cutoff_frac=0.05, warm_start_seed=7,
            penalties=_TOY_PENALTIES, n_jobs=2, maxiter=30, verbose=False,
        )
        self.assertAlmostEqual(info_base['final_fidelity'], BASELINE_N550_BASE_F, delta=_E2E_ATOL)

        u_ref, info_ref = optimizer.refine_pulse(
            get_state_pairs=_toy_state_pairs_m1,
            initial_pulse=u_base,
            trunc_list=[6, 8], n_t=2, extra_maxiter=30,
            penalties=_TOY_PENALTIES, verbose=False,
        )
        self.assertAlmostEqual(info_ref['final_fidelity'], BASELINE_REFINE_PULSE_F, delta=_E2E_ATOL)

    def test_refine_pulse_dt(self):
        u_opt, _ = optimizer.optimize_multi_state_pulse(
            get_state_pairs=_toy_state_pairs_m1,
            trunc_list=[6, 8], n_t=2, N=25,
            warm_start_amp=6.0, warm_start_cutoff_frac=0.3, warm_start_seed=7,
            penalties=_TOY_PENALTIES, n_jobs=2, maxiter=150, verbose=False,
        )
        u_dt, info_dt = optimizer.refine_pulse_dt(
            get_state_pairs=_toy_state_pairs_m1,
            initial_pulse=u_opt, s=2, dt=0.002,
            trunc_list=[6, 8], n_t=2, extra_maxiter=50,
            penalties=_TOY_PENALTIES, verbose=False,
        )
        self.assertEqual(u_dt.shape, (50, 4))
        self.assertAlmostEqual(info_dt['dt'], 0.001)
        self.assertAlmostEqual(info_dt['final_fidelity'], BASELINE_REFINE_DT_F, delta=_E2E_ATOL)


class TimingBenchmarkTest(unittest.TestCase):
    """Informational only (no hard assertion on absolute speed -- machine
    dependent): prints wall-clock time for a refine_dt-shaped workload so
    the speedup from the batched _fidelity_core + persistent joblib pool is
    directly visible when this file is run with -v."""

    def test_refine_dt_shaped_timing(self):
        u0, _ = optimizer.optimize_multi_state_pulse(
            get_state_pairs=_toy_state_pairs_m2,
            trunc_list=[8, 10, 12], n_t=2, N=60,
            warm_start_amp=6.0, warm_start_cutoff_frac=0.1, warm_start_seed=3,
            penalties=_TOY_PENALTIES, n_jobs=3, maxiter=5, verbose=False,
        )
        t0 = time.perf_counter()
        optimizer.refine_pulse_dt(
            get_state_pairs=_toy_state_pairs_m2,
            initial_pulse=u0, s=4, dt=0.002,
            trunc_list=[8, 10, 12], n_t=2, extra_maxiter=15,
            penalties=_TOY_PENALTIES, verbose=False,
        )
        elapsed = time.perf_counter() - t0
        print(f"\n[timing] refine_pulse_dt(N={u0.shape[0]}->{u0.shape[0]*4}, "
              f"trunc_list=[8,10,12], extra_maxiter=15): {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main(verbosity=2)
