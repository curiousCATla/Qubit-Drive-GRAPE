#!/usr/bin/env python3
"""
qutip_grape_optimizer.py
========================

Teaching implementation of the same GRAPE waveform optimization used in
``optimizer.py`` / ``grape_core.py``, but with the *physics* built in QuTiP.

Also the shared home for QuTiP helpers used by ``qutip_validate.py``:
``build_qutip_ops``, ``build_qutip_hamiltonian``, ``qutip_final_state``,
``qutip_multi_state_fidelity`` (sesolve cross-check of a saved pulse).

What this file does (and does not)
----------------------------------
DOES
  * Build the transmon–cavity Hamiltonian with QuTiP ``Qobj`` / ``tensor``
    (independent of ``grape_core.make_ops``; same Hilbert-space convention).
  * Run classical GRAPE: piecewise-constant controls, analytic adjoint
    gradients, L-BFGS-B minimization of -F + penalties — same structure as
    ``optimize_multi_state_pulse`` in ``optimizer.py``.
  * Provide sesolve-based fidelity helpers for independent validation.
  * Optimize a simple multi-state / multi-truncation example (|g,0⟩ → |g,6⟩).

DOES NOT
  * Depend on ``qutip-qtrl`` / ``qutip.control.pulseoptim`` (optional extra
    package in QuTiP 5). We implement GRAPE ourselves so you can see every step.
  * Replace production ``optimizer.py`` (which is faster and more complete:
    band-limit projection, discrepancy penalty, joblib parallelism, refine, …).

How GRAPE works (same algorithm as grape_core)
----------------------------------------------
Controls ``u[k, j]`` are constant on each time slice ``k = 0 … N-1`` of width dt:

    H_k = H0 + Σ_j  u[k,j] * Hc[j]
    U_k = exp(-i * dt * H_k)
    U   = U_{N-1} … U_0

Fidelity for one state transfer:

    F = |⟨f| U |i⟩|²

The gradient ∂F/∂u[k,j] is obtained with the **adjoint method** (not finite
differences):

  1. Forward:  |φ_k⟩ = U_{k-1} … U_0 |i⟩   (store eigendecomposition of each H_k)
  2. Backward: |λ_k⟩ = U_k† … U_{N-1}† |f⟩
  3. At step k, project into the eigenbasis of H_k and apply the Φ matrix that
     differentiates the matrix exponential (handles near-degenerate eigenvalues
     in the rotating frame, e.g. |g,0⟩ ↔ |g,1⟩).

L-BFGS-B then minimizes  cost = -F_avg + λ_d * smooth + λ_b * boundary + λ_a * amp.

Usage
-----
    # Short smoke-test demo (default; finishes in ~1–2 min on a laptop),
    # run from the repo root:
    python QuTip/qutip_grape_optimizer.py

    # Longer run closer to production settings
    python QuTip/qutip_grape_optimizer.py --full

    # Import as a library
    from QuTip.qutip_grape_optimizer import optimize_grape_pulse, get_g6_state_pairs
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import qutip as qt
from numpy.linalg import eigh
from scipy.optimize import minimize

# Apple Accelerate BLAS can emit harmless RuntimeWarnings on complex matmuls
# (same issue as grape_core.py). Silence them so demos stay readable.
np.seterr(divide="ignore", over="ignore", invalid="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

# ---------------------------------------------------------------------------
# 1. Physical parameters  (identical to grape_core.py)
# ---------------------------------------------------------------------------
two_pi = 2 * np.pi
chi = two_pi * (-2.194)    # MHz → rad/μs  dispersive shift χ
Kerr = two_pi * (-0.0037)  # cavity self-Kerr K
chip = two_pi * (-0.019)   # second-order dispersive χ'
alpha = two_pi * (-236.0)  # transmon anharmonicity α
DT_DEFAULT = 0.002         # μs per time step (2 ns)


# ---------------------------------------------------------------------------
# 2. Hamiltonian in QuTiP  (shared with qutip_validate.py)
# ---------------------------------------------------------------------------
def build_qutip_ops(n_t: int, n_c: int):
    """
    QuTiP-native rebuild of grape_core.make_ops — not by calling grape_core,
    so it is a genuine second implementation of the operators.

    Hilbert-space ordering matches grape_core.basis_state:
        index(|t, c⟩) = t * n_c + c
    i.e. transmon is the outer (slow) tensor factor, cavity the inner (fast).
    ``qt.tensor(qeye(n_t), a)`` matches ``np.kron(np.eye(n_t), a)``.
    """
    a = qt.destroy(n_c)  # cavity lowering
    b = qt.destroy(n_t)  # transmon lowering
    A = qt.tensor(qt.qeye(n_t), a)  # cavity op on joint space
    B = qt.tensor(b, qt.qeye(n_c))  # transmon op on joint space
    return A, B


def build_qutip_hamiltonian(n_t: int, n_c: int):
    """
    Drift H0 and four control operators Hc (as QuTiP Qobj).

        H0 = χ nA nB + (K/2) A†²A² + (χ'/2) nB A†²A² + (α/2) B†²B²
        Hc = [A+A†,  i(A-A†),  B+B†,  i(B-B†)]   # cavity I/Q, transmon I/Q
    """
    A, B = build_qutip_ops(n_t, n_c)
    Ad, Bd = A.dag(), B.dag()
    nA, nB = Ad * A, Bd * B

    H0 = chi * (nA * nB) + (Kerr / 2) * (Ad * Ad * A * A) + (chip / 2) * (nB * (Ad * Ad * A * A))
    if n_t >= 3:
        H0 = H0 + (alpha / 2) * (Bd * Bd * B * B)

    Hc = [A + Ad, 1j * (A - Ad), B + Bd, 1j * (B - Bd)]
    return H0, Hc


def qobj_to_dense(H):
    """Extract a dense complex ndarray from a QuTiP Qobj (for eigh / GRAPE)."""
    return np.asarray(H.full(), dtype=complex)


def hamiltonian_as_numpy(n_t: int, n_c: int):
    """
    Build H0, Hc with QuTiP, then return dense numpy matrices.

    GRAPE needs many eigendecompositions per objective call; working with
    plain ndarrays is much faster than calling Qobj methods in a tight loop.
    The *definition* of the operators still comes entirely from QuTiP.
    """
    H0_qt, Hc_qt = build_qutip_hamiltonian(n_t, n_c)
    H0 = qobj_to_dense(H0_qt)
    # Hermitize numerically (eigh expects exact Hermitian)
    H0 = 0.5 * (H0 + H0.conj().T)
    Hc = [qobj_to_dense(Hj) for Hj in Hc_qt]
    for j in range(len(Hc)):
        Hc[j] = 0.5 * (Hc[j] + Hc[j].conj().T)
    return H0, Hc


def basis_state(n_t: int, n_c: int, t_level: int, c_level: int) -> np.ndarray:
    """|t_level, c_level⟩ as a length-(n_t * n_c) complex vector."""
    v = np.zeros(n_t * n_c, dtype=complex)
    v[t_level * n_c + c_level] = 1.0
    return v


def get_g6_state_pairs(n_c: int, n_t: int = 3):
    """Factory: |g,0⟩ → |g,6⟩ (production target in pulse_analysis.py)."""
    return [(basis_state(n_t, n_c, 0, 0), basis_state(n_t, n_c, 0, 6))]


def get_g1_state_pairs(n_c: int, n_t: int = 3):
    """
    Factory: |g,0⟩ → |g,1⟩.

    Easier teaching target: one cavity photon, works with short N so a laptop
    demo can reach F ≳ 0.9 in tens of L-BFGS-B iterations. Same optimizer API
    as get_g6_state_pairs — only the target ket changes.
    """
    return [(basis_state(n_t, n_c, 0, 0), basis_state(n_t, n_c, 0, 1))]


# ---------------------------------------------------------------------------
# 2b. Independent sesolve propagator (used by qutip_validate.py)
# ---------------------------------------------------------------------------
def qutip_final_state(u, n_t, n_c, psi0, dt=DT_DEFAULT):
    """
    Propagate psi0 under the piecewise-constant control sequence u (shape
    [N, 4]) using qutip.sesolve, and return the final state ket.

    The pulse is constant on each interval [k*dt, (k+1)*dt) — a step function —
    so each control channel uses a QuTiP Coefficient with order=0 (zero-order
    hold), which reproduces that convention exactly rather than smoothing it
    with a spline.
    """
    N = u.shape[0]
    H0, Hc = build_qutip_hamiltonian(n_t, n_c)

    tlist = np.arange(N + 1) * dt  # edges: 0, dt, …, N*dt
    terms = [H0]
    for j in range(4):
        vals = np.append(u[:, j], u[-1, j])  # pad to len(tlist); last sample unused
        coeff = qt.coefficient(vals.astype(complex), tlist=tlist, order=0)
        terms.append([Hc[j], coeff])
    H = qt.QobjEvo(terms)

    psi0_qt = qt.Qobj(psi0.reshape(-1, 1), dims=[[n_t, n_c], [1, 1]])
    result = qt.sesolve(H, psi0_qt, tlist)
    return result.states[-1]


def qutip_multi_state_fidelity(u, factory, n_t, n_c, dt=DT_DEFAULT):
    """
    QuTiP-sesolve counterpart of grape_core.fidelity_multi_state.

    Propagate each (psi_i, psi_f) pair with sesolve and average |⟨f|ψ(T)⟩|²,
    matching grape_core's simple-average convention for multi-target gates.
    """
    pairs = factory(n_c=n_c, n_t=n_t)
    fids = []
    for psi_i, psi_f in pairs:
        psi_final = qutip_final_state(u, n_t, n_c, psi_i, dt=dt)
        psi_f_qt = qt.Qobj(psi_f.reshape(-1, 1), dims=[[n_t, n_c], [1, 1]])
        fids.append(np.abs(psi_f_qt.overlap(psi_final)) ** 2)
    return float(np.mean(fids))


# ---------------------------------------------------------------------------
# 3. Piecewise-constant propagator + GRAPE adjoint gradient
#    (same math as grape_core.step_data / fidelity_grad)
# ---------------------------------------------------------------------------
def step_data(H0, Hc, u_k, dt):
    """
    One time step of piecewise-constant control.

    Returns
    -------
    Uk : complex (d, d)  propagator exp(-i dt H_k)
    w  : real (d,)       eigenvalues of H_k
    V  : complex (d, d)  eigenvectors (columns)
    """
    Hk = H0 + u_k[0] * Hc[0] + u_k[1] * Hc[1] + u_k[2] * Hc[2] + u_k[3] * Hc[3]
    w, V = eigh(Hk)
    Uk = V @ np.diag(np.exp(-1j * dt * w)) @ V.conj().T
    return Uk, w, V


def fidelity_grad(u, H0, Hc, psi_i, psi_f, dt, want_grad=True):
    """
    State-transfer fidelity F = |⟨f|U|i⟩|² and optional analytic gradient.

    Parameters
    ----------
    u : (N, 4) real array
        Control waveform (cavity I/Q, transmon I/Q).
    H0, Hc : drift + list of 4 control Hamiltonians (dense numpy)
    psi_i, psi_f : complex vectors
    dt : float
    want_grad : bool

    Returns
    -------
    F : float
    grad : (N, 4) array or None
    """
    N = u.shape[0]
    phi = [psi_i.copy()]
    Ws, Vs, Us = [], [], []
    psi = psi_i.copy()

    # ---- forward pass ----
    for k in range(N):
        Uk, w, V = step_data(H0, Hc, u[k], dt)
        psi = Uk @ psi
        phi.append(psi)
        Us.append(Uk)
        Ws.append(w)
        Vs.append(V)

    v = np.vdot(psi_f, psi)  # ⟨f|U|i⟩
    F = float(np.abs(v) ** 2)
    if not want_grad:
        return F, None

    # ---- backward costates: λ_N = |f⟩, λ_k = U_k† λ_{k+1} ----
    lam = [None] * (N + 1)
    lam[N] = psi_f.copy()
    for k in range(N - 1, -1, -1):
        lam[k] = Us[k].conj().T @ lam[k + 1]

    # ---- ∂U_k/∂u_j via eigenbasis Φ matrix ----
    grad = np.zeros((N, 4))
    for k in range(N):
        w, V = Ws[k], Vs[k]
        p = V.conj().T @ phi[k]       # |φ_{k-1}⟩ in eigenbasis
        q = V.conj().T @ lam[k + 1]   # |λ_k⟩ in eigenbasis

        ew = np.exp(-1j * dt * w)
        dw = w[:, None] - w[None, :]
        near = np.abs(dw) < 1e-10
        dw_safe = np.where(near, 1.0, dw)
        # Φ_mn = (e^{-i dt w_m} - e^{-i dt w_n}) / (w_m - w_n)
        # diagonal / near-degeneracy: -i dt e^{-i dt w}
        Phi = (ew[:, None] - ew[None, :]) / dw_safe
        Phi = np.where(near, (-1j * dt * ew)[:, None], Phi)

        qc = q.conj()
        for j in range(4):
            X = V.conj().T @ Hc[j] @ V
            dv = qc @ ((Phi * X) @ p)  # ⟨λ| ∂U/∂u_j |φ⟩
            # d(|v|²)/du = 2 Re[ v* · dv ]
            grad[k, j] = 2.0 * np.real(np.conj(v) * dv)

    return F, grad


def fidelity_multi_state(u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=True):
    """Average F (and gradient) over several (psi_i, psi_f) pairs."""
    M = len(psi_i_list)
    total_F = 0.0
    total_grad = np.zeros_like(u)
    for psi_i, psi_f in zip(psi_i_list, psi_f_list):
        F_i, g_i = fidelity_grad(u, H0, Hc, psi_i, psi_f, dt, want_grad=want_grad)
        total_F += F_i
        if want_grad and g_i is not None:
            total_grad += g_i
    F_avg = total_F / M
    grad_avg = total_grad / M if want_grad else None
    return F_avg, grad_avg


# ---------------------------------------------------------------------------
# 4. Soft penalties (identical idea to grape_core)
# ---------------------------------------------------------------------------
def derivative_penalty(u):
    """Smoothness: Σ ||u_{k+1} - u_k||²."""
    diff = u[1:] - u[:-1]
    pen = np.sum(diff ** 2)
    grad = np.zeros_like(u)
    grad[1:] += 2 * diff
    grad[:-1] -= 2 * diff
    return pen, grad


def boundary_penalty(u):
    """Prefer controls that start and end near zero."""
    pen = np.sum(u[0] ** 2) + np.sum(u[-1] ** 2)
    grad = np.zeros_like(u)
    grad[0] += 2 * u[0]
    grad[-1] += 2 * u[-1]
    return pen, grad


def amplitude_penalty(u, amp_max=40.0):
    """Soft quadratic excess beyond amp_max (rad/μs)."""
    excess = np.maximum(np.abs(u) - amp_max, 0.0)
    pen = np.sum(excess ** 2)
    grad = np.zeros_like(u)
    mask = np.abs(u) > amp_max
    grad[mask] = 2 * excess[mask] * np.sign(u[mask])
    return pen, grad


def smooth_initial_controls(N, amp=4.0, cutoff_frac=0.04, seed=42):
    """Low-pass random I/Q envelopes (same idea as grape_core)."""
    rng = np.random.default_rng(seed)
    nf = N // 2 + 1
    kcut = max(1, int(cutoff_frac * nf))
    u = np.zeros((N, 4))
    for j in range(4):
        spec = np.zeros(nf, dtype=complex)
        spec[:kcut] = rng.standard_normal(kcut) + 1j * rng.standard_normal(kcut)
        col = np.fft.irfft(spec, n=N)
        u[:, j] = amp * col / (np.std(col) + 1e-12)
    peak = np.max(np.abs(u))
    if peak > amp:
        u *= amp / peak
    return u


# ---------------------------------------------------------------------------
# 5. Optimizer — mirrors optimizer.optimize_multi_state_pulse (core path)
# ---------------------------------------------------------------------------
def optimize_grape_pulse(
    get_state_pairs,
    trunc_list=(16, 18),
    n_t=3,
    N=80,
    dt=DT_DEFAULT,
    penalties=None,
    warm_start=None,
    warm_start_amp=4.0,
    warm_start_cutoff_frac=0.04,
    warm_start_seed=42,
    hard_amp_limit=50.0,
    maxiter=200,
    save_path=None,
    verbose=True,
):
    """
    Multi-truncation GRAPE with L-BFGS-B (teaching subset of optimizer.py).

    At every objective evaluation we:
      1. For each n_c in trunc_list, build H with QuTiP → numpy.
      2. Average fidelity (and analytic gradient) over state pairs & truncations.
      3. Add soft smoothness / boundary / amplitude penalties.
      4. Return (cost, grad) with cost = -F_avg + penalties.

    Parameters largely match ``optimize_multi_state_pulse``; omitted features
    (for clarity): Fourier band-limit projection, discrepancy penalty, joblib
    parallel truncations, best-seen-F bookkeeping.

    Returns
    -------
    u_opt : (N, 4) ndarray
    info  : dict with final_fidelity, message, success, iterations, …
    """
    if penalties is None:
        penalties = {"deriv": 1e-5, "boundary": 4e-5, "amp": 1.2e-4, "amp_max": 40.0}

    if verbose:
        print(f"\n{'=' * 60}")
        print("QuTiP-GRAPE optimize")
        print(f"  trunc_list = {list(trunc_list)}")
        print(f"  n_t={n_t}, N={N}, dt={dt} μs  (T = {N * dt:.3f} μs)")
        print(f"  penalties  = {penalties}")
        print(f"  maxiter    = {maxiter}")
        print(f"{'=' * 60}")

    # Pre-build Hamiltonians once (QuTiP → numpy)
    H0_list, Hc_list = [], []
    for nc in trunc_list:
        H0_k, Hc_k = hamiltonian_as_numpy(n_t, nc)
        H0_list.append(H0_k)
        Hc_list.append(Hc_k)
        if verbose:
            print(f"  Built QuTiP Hamiltonian for n_c={nc}  dim={n_t * nc}")

    # Initial controls
    if isinstance(warm_start, str) and os.path.exists(warm_start):
        u0 = np.load(warm_start)
        if verbose:
            print(f"  Warm start from file: {warm_start}")
    elif isinstance(warm_start, np.ndarray):
        u0 = warm_start.copy()
        if verbose:
            print("  Warm start from array")
    elif warm_start in ("zero", 0, "zeros"):
        u0 = np.zeros((N, 4))
        if verbose:
            print("  Warm start: zeros")
    else:
        u0 = smooth_initial_controls(
            N, amp=warm_start_amp, cutoff_frac=warm_start_cutoff_frac, seed=warm_start_seed
        )
        if verbose:
            print(f"  Warm start: smooth random (peak |u|={np.max(np.abs(u0)):.3f})")

    if u0.shape != (N, 4):
        raise ValueError(f"warm_start shape {u0.shape} != ({N}, 4)")

    x0 = u0.ravel()
    bounds = [(-hard_amp_limit, hard_amp_limit)] * (N * 4)
    n_eval = {"count": 0}

    def objective(x):
        u = x.reshape(N, 4)
        total_F = 0.0
        total_grad = np.zeros_like(u)
        M = len(trunc_list)

        for H0_k, Hc_k, nc in zip(H0_list, Hc_list, trunc_list):
            pairs = get_state_pairs(n_c=nc, n_t=n_t)
            psi_i_list = [p[0] for p in pairs]
            psi_f_list = [p[1] for p in pairs]
            F_k, g_k = fidelity_multi_state(
                u, H0_k, Hc_k, psi_i_list, psi_f_list, dt, want_grad=True
            )
            total_F += F_k
            total_grad += g_k

        F_avg = total_F / M
        grad_avg = total_grad / M

        cost = -F_avg
        g = -grad_avg

        if penalties.get("deriv", 0) > 0:
            p, gp = derivative_penalty(u)
            cost += penalties["deriv"] * p
            g += penalties["deriv"] * gp
        if penalties.get("boundary", 0) > 0:
            p, gp = boundary_penalty(u)
            cost += penalties["boundary"] * p
            g += penalties["boundary"] * gp
        if penalties.get("amp", 0) > 0:
            p, gp = amplitude_penalty(u, amp_max=penalties.get("amp_max", 40.0))
            cost += penalties["amp"] * p
            g += penalties["amp"] * gp

        n_eval["count"] += 1
        if verbose and (n_eval["count"] == 1 or n_eval["count"] % 20 == 0):
            print(f"  eval {n_eval['count']:4d}:  F_avg={F_avg:.6f}  cost={cost:.6f}")

        return cost, g.ravel()

    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
    )
    u_opt = res.x.reshape(N, 4)

    # Final fidelity at largest truncation
    nc_max = max(trunc_list)
    H0_main, Hc_main = hamiltonian_as_numpy(n_t, nc_max)
    pairs = get_state_pairs(n_c=nc_max, n_t=n_t)
    F_final, _ = fidelity_multi_state(
        u_opt,
        H0_main,
        Hc_main,
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        dt,
        want_grad=False,
    )

    if verbose:
        print(f"\nFinished: {res.message}")
        print(f"  iterations = {res.nit},  objective evals = {n_eval['count']}")
        print(f"  Final F (n_c={nc_max}): {F_final:.6f}")
        print(f"  peak |u| = {np.max(np.abs(u_opt)):.4f} rad/μs")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        np.save(save_path, u_opt)
        if verbose:
            print(f"  Saved pulse → {save_path}")

    info = {
        "message": res.message,
        "success": res.success,
        "iterations": res.nit,
        "n_evals": n_eval["count"],
        "final_fidelity": F_final,
        "trunc_list": list(trunc_list),
    }
    return u_opt, info


# ---------------------------------------------------------------------------
# 6. Optional: cross-check one F against grape_core (if importable)
# ---------------------------------------------------------------------------
def crosscheck_vs_grape_core(u, n_t, n_c, dt):
    """Compare this file's F to grape_core.fidelity_grad on the same pulse."""
    try:
        from core.grape_core import make_hamiltonian, fidelity_grad as gc_fid
    except ImportError:
        print("  (grape_core not importable — skip cross-check)")
        return

    H0_qt, Hc_qt = hamiltonian_as_numpy(n_t, n_c)
    H0_gc, Hc_gc = make_hamiltonian(n_t, n_c)
    psi_i = basis_state(n_t, n_c, 0, 0)
    psi_f = basis_state(n_t, n_c, 0, 6)

    F_qt, _ = fidelity_grad(u, H0_qt, Hc_qt, psi_i, psi_f, dt, want_grad=False)
    F_gc, _ = gc_fid(u, H0_gc, Hc_gc, psi_i, psi_f, dt, want_grad=False)
    print(f"  Cross-check n_c={n_c}:  QuTiP-GRAPE F={F_qt:.8f}  grape_core F={F_gc:.8f}  |Δ|={abs(F_qt - F_gc):.2e}")


# ---------------------------------------------------------------------------
# 7. CLI demo
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="QuTiP-based GRAPE teaching optimizer")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Optimize |g,0⟩→|g,6⟩ with larger N/trunc/maxiter (slow; production-like)",
    )
    parser.add_argument(
        "--target",
        choices=("g1", "g6"),
        default=None,
        help="State transfer target. Default: g1 (demo) or g6 with --full.",
    )
    parser.add_argument("--maxiter", type=int, default=None, help="Override L-BFGS-B maxiter")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save optimized pulse .npy",
    )
    args = parser.parse_args(argv)

    if args.full:
        # Closer to pulse_analysis.py (still a bit lighter for a teaching run)
        target = args.target or "g6"
        trunc_list = [20, 22, 24]
        N = 250
        maxiter = 500
        warm_start_amp = 6.0
        penalties = {"deriv": 2e-5, "boundary": 1e-4, "amp": 1e-5, "amp_max": 40.0}
    else:
        # Fast laptop demo: |g,0⟩→|g,1⟩ with short pulse (~0.2 μs)
        target = args.target or "g1"
        trunc_list = [12, 14]
        N = 100
        maxiter = 150
        warm_start_amp = 5.0
        penalties = {"deriv": 1e-5, "boundary": 4e-5, "amp": 1.2e-4, "amp_max": 40.0}

    if args.maxiter is not None:
        maxiter = args.maxiter

    if target == "g6":
        get_pairs = get_g6_state_pairs
        c_target = 6
        default_save = os.path.join("pulses", "u_opt_qutip_grape_g6.npy")
        # |g,6⟩ needs enough total time; bump N if user forced g6 without --full
        if not args.full and N < 200:
            N = 200
            trunc_list = [16, 18]
            print("Note: |g,0⟩→|g,6⟩ needs longer pulses; using N=200, trunc=[16,18].")
    else:
        get_pairs = get_g1_state_pairs
        c_target = 1
        default_save = os.path.join("pulses", "u_opt_qutip_grape_g1.npy")

    save_path = args.save or default_save

    print("QuTiP version:", qt.__version__)
    print(
        "\nThis script teaches the same GRAPE loop as optimizer.py:\n"
        "  QuTiP builds H0/Hc  →  adjoint gradients  →  L-BFGS-B on -F + penalties.\n"
        f"Target: |g,0⟩ → |g,{c_target}⟩.  Use --full for production-like |g,6⟩.\n"
    )

    # ---- 1) Gradient sanity check (cheap, before the long opt) ----
    print("--- Gradient sanity: analytic vs finite difference ---")
    n_c_chk = trunc_list[-1]
    H0, Hc = hamiltonian_as_numpy(3, n_c_chk)
    psi_i = basis_state(3, n_c_chk, 0, 0)
    psi_f = basis_state(3, n_c_chk, 0, c_target)
    rng = np.random.default_rng(0)
    u_test = 0.8 * rng.standard_normal((30, 4))
    F0, g_an = fidelity_grad(u_test, H0, Hc, psi_i, psi_f, DT_DEFAULT, want_grad=True)
    eps = 1e-6
    k, j = 5, 0
    u_p = u_test.copy()
    u_p[k, j] += eps
    Fp, _ = fidelity_grad(u_p, H0, Hc, psi_i, psi_f, DT_DEFAULT, want_grad=False)
    g_fd = (Fp - F0) / eps
    denom = max(abs(g_fd), abs(g_an[k, j]), 1e-30)
    rel = abs(g_an[k, j] - g_fd) / denom
    print(f"  F(u_test) = {F0:.6e}")
    print(
        f"  ∂F/∂u[{k},{j}]: analytic={g_an[k, j]:.6e}  FD={g_fd:.6e}  rel_err={rel:.2e}"
    )
    if rel > 1e-3 and abs(g_fd) > 1e-12:
        print("  WARNING: analytic gradient disagrees with FD — check Φ / hermiticity.")
    else:
        print("  OK: analytic gradient matches finite differences.")

    # ---- 2) Optimize ----
    u_opt, info = optimize_grape_pulse(
        get_state_pairs=get_pairs,
        trunc_list=trunc_list,
        n_t=3,
        N=N,
        dt=DT_DEFAULT,
        penalties=penalties,
        warm_start_amp=warm_start_amp,
        warm_start_seed=1,
        maxiter=maxiter,
        save_path=save_path,
        verbose=True,
    )

    # ---- 3) Cross-check vs grape_core (same F for same u) ----
    print("\n--- Cross-check vs grape_core on optimized pulse ---")
    try:
        from core.grape_core import make_hamiltonian, fidelity_grad as gc_fid

        nc = max(trunc_list)
        H0_qt, Hc_qt = hamiltonian_as_numpy(3, nc)
        H0_gc, Hc_gc = make_hamiltonian(3, nc)
        psi_i = basis_state(3, nc, 0, 0)
        psi_f = basis_state(3, nc, 0, c_target)
        F_qt, _ = fidelity_grad(u_opt, H0_qt, Hc_qt, psi_i, psi_f, DT_DEFAULT, want_grad=False)
        F_gc, _ = gc_fid(u_opt, H0_gc, Hc_gc, psi_i, psi_f, DT_DEFAULT, want_grad=False)
        print(
            f"  n_c={nc}:  QuTiP-GRAPE F={F_qt:.8f}  grape_core F={F_gc:.8f}  "
            f"|Δ|={abs(F_qt - F_gc):.2e}"
        )
    except ImportError:
        print("  (grape_core not importable — skip cross-check)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
