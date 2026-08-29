"""
Pure-numpy EsT pipeline: eigendecomposition propagator + hand-derived adjoint.

This is a SECOND, independent implementation of the same objective that
EST/grape_jax.py computes with JAX autodiff over an `expm` propagator. Nothing
here imports jax. The two exist side by side on purpose: agreement between an
autodiff gradient and an independently hand-derived one rules out a whole class
of silent objective bugs, in the same spirit as QuTip/qutip_validate.py against
core/grape_core.py.

Why eigh is usable here when EST/grape_jax.py says it is not
-----------------------------------------------------------
grape_jax's docstring is right about autodiff: d/dH of an eigendecomposition is
singular at degeneracies, so `jnp.linalg.eigh` under reverse mode reintroduces
the defect core/grape_core.py fixes by hand. That argument applies to AUTODIFF.
It does not apply to a hand-derived gradient, which is exactly what
core/grape_core.py:174-179 does: the divided-difference matrix

    Phi_ab = (e^{-i dt w_a} - e^{-i dt w_b}) / (w_a - w_b)

is evaluated with an explicit near-degenerate branch replaced by its L'Hopital
limit, -i dt e^{-i dt w_a}. That construction is ported verbatim below.

What is new relative to core/grape_core.py
------------------------------------------
_fidelity_core seeds its costate only at k = N, because C1 is a TERMINAL cost.
C2 and C3 are RUNNING costs -- they read the trajectory at every step -- so the
costate recursion picks up a source term at every step:

    Lambda_N = dPhi/dPsi*_N + dL_N/dPsi*_N
    Lambda_k = dL_k/dPsi*_k + U_k^dag Lambda_{k+1}          k = N-1 ... 0

with the gradient then read off exactly as in _fidelity_core:

    dC/du_{k,j} = 2 Re tr( Lambda_{k+1}^dag (dU_k/du_{k,j}) Psi_k )

Conventions
-----------
Wirtinger, with Lambda = dC/dPsi*, under which dC = 2 Re tr(Lambda^dag dPsi)
for real C. The code and error cardinals are stacked into one (n, 2M) batch and
propagated once, matching EST/grape_jax.py:305 -- exact, since propagation acts
column-wise.

C3's variance normalization couples the WHOLE trajectory (mean and variance are
global reductions over k), so no injection can be formed until the forward pass
has finished. This is a strictly two-pass algorithm: forward storing the
trajectory and eigendata, then all injections, then one backward sweep. It is
not a streaming adjoint and cannot be made one.

The cost DEFINITIONS (including C2's N_norm^2 normalization and C3's mean(v)^2
normalization, both deliberate deviations from the paper's printed equations --
see EST/grape_jax.py) are reproduced here in numpy. They must match grape_jax
term for term; EST/test_grape_eigh.py pins that against the JAX implementation.
"""

import numpy as np

from core.fourier_cutoff import project_bandlimit
from core.grape_core import make_ops
from EST.device import (BAND_MHZ, DT, EPS_MAX, N_T, RAMP_NS, TRUNC_LIST,
                        make_hamiltonian_est, ramp_envelope)
from EST import kitten_code

# Bound peak memory of the batched eigh call, as core/grape_core.py:66 does.
_EIGH_CHUNK = 256

# Degeneracy threshold, ported verbatim from core/grape_core.py:176. Below this
# eigenvalue gap the divided difference is replaced by its analytic limit.
DEGEN_TOL = 1e-10

# arccos'(x) diverges at x = 1 and consecutive states in a 1 ns trajectory ARE
# nearly parallel. Same constant and same reasoning as EST/grape_jax.py:77.
_FS_CLIP = 1.0 - 1e-9


# ---------------------------------------------------------------------------
# Constraint chain: raw optimizer variable -> physical pulse, and its adjoint
# ---------------------------------------------------------------------------

def make_constrainer(N, dt=DT, band=BAND_MHZ, ramp_ns=RAMP_NS, cmask=None):
    """
    Return (constrain, constrain_adj), the (N,4) -> (N,4) forward map and its
    adjoint.

    Forward order is band-limit, then ramp, then control mask -- the order set
    and justified on measurement at EST/grape_jax.py:108-119.

    Every factor is linear and SELF-ADJOINT, so the adjoint is the same three
    operations applied in REVERSE order:

        u = M(R(B(x)))      =>      dC/dx = B(R(M(dC/du)))

    R and M are real diagonal scalings, trivially self-adjoint. B is the
    orthogonal band-limit projection (IFFT . mask . FFT on the complex drives
    eps_C = u0 + i u1, eps_T = u2 + i u3); it is self-adjoint and idempotent
    under the complex inner product, and a complex-self-adjoint operator is also
    self-adjoint as a real map under Re<.,.>, which is what makes
    core.fourier_cutoff.project_bandlimit correct on a gradient array as well as
    on a pulse -- see that module's docstring and core/grape_core.py:464-467.

    Getting the ORDER wrong in the adjoint (projecting first) still produces a
    plausible-looking gradient, which is why the constraint-chain
    finite-difference test in EST/test_grape_eigh.py is not optional.

    Returns (None, None) if every constraint is disabled; callers treat that as
    the identity.
    """
    ops = []
    if band is not None:
        ops.append(("band", band))
    if ramp_ns is not None:
        ops.append(("ramp", ramp_envelope(N, dt, ramp_ns)))
    if cmask is not None:
        ops.append(("mask", np.asarray(cmask, dtype=np.float64)))

    if not ops:
        return None, None

    def constrain(u):
        u = np.asarray(u, dtype=np.float64)
        for kind, arr in ops:
            if kind == "band":
                u = project_bandlimit(u, dt, arr, arr)
            elif kind == "ramp":
                u = u * arr[:, None]
            else:
                u = u * arr[None, :]
        return u

    def constrain_adj(g):
        g = np.asarray(g, dtype=np.float64)
        for kind, arr in reversed(ops):
            if kind == "band":
                g = project_bandlimit(g, dt, arr, arr)
            elif kind == "ramp":
                g = g * arr[:, None]
            else:
                g = g * arr[None, :]
        return g

    return constrain, constrain_adj


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def propagate_trajectory(u, H0, Hc, Psi0, dt, want_eig=False):
    """
    Propagate every column of Psi0 through the piecewise-constant pulse u.

    u    : (N,4) real controls [C_I, C_Q, T_I, T_Q]
    H0   : (n,n) complex drift
    Hc   : (4,n,n) complex control operators
    Psi0 : (n,M) complex, M state vectors stacked as columns

    Returns Psi_traj : (N+1,n,M), the state at every step t = 0..N; plus, when
    want_eig is set, the per-step eigendata the adjoint needs.

    The propagator is U_k = V diag(e^{-i dt w}) V^dag with (w, V) from eigh --
    the same construction as core.grape_core.step_data. U_k is never formed as a
    matrix: applying V^dag, scaling, then V costs two (n,n)@(n,M) products,
    against two (n,n)@(n,n) products to build U_k first. That is a real saving
    here because M = 12 << n = 60.
    """
    u = np.asarray(u, dtype=np.float64)
    N = u.shape[0]
    n = H0.shape[0]
    M = Psi0.shape[1]

    Hk = H0[None, :, :] + np.tensordot(u, Hc, axes=([1], [0]))   # (N,n,n)
    w_stack = np.empty((N, n))
    V_stack = np.empty((N, n, n), dtype=complex)
    for start in range(0, N, _EIGH_CHUNK):
        end = min(start + _EIGH_CHUNK, N)
        w_stack[start:end], V_stack[start:end] = np.linalg.eigh(Hk[start:end])
    del Hk
    ew_stack = np.exp(-1j * dt * w_stack)                        # (N,n)

    traj = np.empty((N + 1, n, M), dtype=complex)
    traj[0] = Psi0
    p_stack = np.empty((N, n, M), dtype=complex) if want_eig else None
    for k in range(N):
        Vk = V_stack[k]
        p = Vk.conj().T @ traj[k]          # V^dag Psi_k, reused by the adjoint
        if want_eig:
            p_stack[k] = p
        traj[k + 1] = Vk @ (ew_stack[k][:, None] * p)

    if want_eig:
        return traj, w_stack, V_stack, ew_stack, p_stack
    return traj


# ---------------------------------------------------------------------------
# Cost terms, each returning its value and its costate injection dL/dPsi*
# ---------------------------------------------------------------------------

def fidelity_cost(code_final, Psi_target):
    """
    C1, terminal. Ordinary infidelity, incoherently averaged over the six
    cardinals. Matches EST/grape_jax.fidelity_cost.

    Injection, from d|v|^2/dpsi* = v * tgt with v = <tgt|psi>:

        dC1/dpsi^C*_N = -(1/M) v_m tgt_m          (error columns get nothing)
    """
    M = code_final.shape[1]
    v = np.sum(np.conj(Psi_target) * code_final, axis=0)          # (M,)
    c1 = 1.0 - np.mean(np.abs(v) ** 2)
    inj_code = -(1.0 / M) * Psi_target * v[None, :]
    return c1, inj_code


def et_cost(code_traj, err_traj, A):
    """
    C2, running over every step. Compares the normalized photon-loss image
    a|psi_C(t)> against the independently evolved error cardinal |psi_E(t)>.

    Uses the N_norm^2 normalization (EST/grape_jax.py:204-213), the deliberate
    deviation from the paper's printed Eq. (C2) that makes F_ET a genuine
    fidelity in [0,1]. The adjoint below differentiates THAT form.

    With o = <psi_E|A psi_C>, s = ||A psi_C||^2 + eps, F = |o|^2/s and
    w = -1/(T*M):

        dC2/dpsi^E*_k = w (o*/s) A psi^C_k
        dC2/dpsi^C*_k = w [ (o/s) A^dag psi^E_k - (|o|^2/s^2) A^dag A psi^C_k ]

    The second term of the psi^C injection is the one that is easy to drop: s
    depends on psi^C as well as the overlap does.
    """
    T, n, M = code_traj.shape
    Ad = A.conj().T
    AdA = Ad @ A

    APsi = np.einsum('ij,tjm->tim', A, code_traj)                 # (T,n,M)
    s = np.sum(np.abs(APsi) ** 2, axis=1) + 1e-12                 # (T,M)
    o = np.einsum('tim,tim->tm', np.conj(err_traj), APsi)         # (T,M)
    F_et = np.abs(o) ** 2 / s
    c2 = 1.0 - np.mean(F_et)

    w = -1.0 / (T * M)
    inj_err = w * (np.conj(o) / s)[:, None, :] * APsi
    Ad_err = np.einsum('ij,tjm->tim', Ad, err_traj)
    AdA_code = np.einsum('ij,tjm->tim', AdA, code_traj)
    inj_code = w * ((o / s)[:, None, :] * Ad_err
                    - (np.abs(o) ** 2 / s ** 2)[:, None, :] * AdA_code)
    return c2, inj_code, inj_err


def velocity_variance_cost(code_traj, dt, normalize=True):
    """
    C3, running over consecutive PAIRS, with a global coupling. Penalizes
    non-uniform speed through Hilbert space; matches
    EST/grape_jax.velocity_variance_cost, including the mean(v)^2 normalization
    (a deliberate deviation -- see that function's docstring).

    Backward chain. With n_v = N, mu = mean(v), sig2 = Var(v) (population):

        dg/dv_k    = 2(v_k - mu)/(n_v (mu^2+e)) - 2 sig2 mu/(n_v (mu^2+e)^2)
        dv_k/dmod_k = -(2/dt)/sqrt(1-mod_k^2)   * [mod_k < _FS_CLIP]
        a_k        = (1/M) dg/dv_k dv_k/dmod_k

    and, from mod = |<psi_k|psi_{k+1}>|,

        dC3/dpsi^C*_k     += a_k (ov*_k/(2 mod_k)) psi^C_{k+1}
        dC3/dpsi^C*_{k+1} += a_k (ov_k /(2 mod_k)) psi^C_k

    so every interior state receives injections from BOTH the pair behind it and
    the pair ahead of it.

    The clip must zero the gradient above _FS_CLIP exactly as jnp.clip does, or
    the cross-check against the JAX gradient disagrees precisely on the steps
    where consecutive states are nearly parallel -- which at dt = 1 ns is most
    of them.
    """
    T, n, M = code_traj.shape
    n_v = T - 1

    ov = np.einsum('tim,tim->tm', np.conj(code_traj[:-1]), code_traj[1:])  # (N,M)
    mod = np.sqrt(np.abs(ov) ** 2 + 1e-30)
    inside = mod < _FS_CLIP
    arg = np.minimum(mod, _FS_CLIP)
    v = 2.0 * np.arccos(arg) / dt

    mu = np.mean(v, axis=0)                                       # (M,)
    sig2 = np.var(v, axis=0)                                      # population
    if normalize:
        den = mu ** 2 + 1e-12
        g = sig2 / den
        dg_dv = (2.0 * (v - mu[None, :]) / (n_v * den[None, :])
                 - 2.0 * sig2[None, :] * mu[None, :] / (n_v * den[None, :] ** 2))
    else:
        g = sig2
        dg_dv = 2.0 * (v - mu[None, :]) / n_v
    c3 = float(np.mean(g))

    # dv/dmod, zeroed where the clip is active.
    dv_dmod = np.where(inside,
                       -2.0 / (dt * np.sqrt(np.maximum(1.0 - arg ** 2, 1e-300))),
                       0.0)
    a = (1.0 / M) * dg_dv * dv_dmod                               # (N,M) real

    inj = np.zeros_like(code_traj)
    coef_k = a * np.conj(ov) / (2.0 * mod)                        # (N,M) complex
    coef_k1 = a * ov / (2.0 * mod)
    inj[:-1] += coef_k[:, None, :] * code_traj[1:]
    inj[1:] += coef_k1[:, None, :] * code_traj[:-1]
    return c3, inj


def amplitude_cost(u, eps_max=EPS_MAX):
    """
    C4. Circular cap |eps| <= eps_max on each quadrature pair; a pure function
    of the pulse, so it contributes no costate. Matches
    EST/grape_jax.amplitude_cost.
    """
    u = np.asarray(u, dtype=np.float64)
    N = u.shape[0]
    magC = np.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2 + 1e-30)
    magT = np.sqrt(u[:, 2] ** 2 + u[:, 3] ** 2 + 1e-30)
    overC = np.maximum(magC - eps_max, 0.0)
    overT = np.maximum(magT - eps_max, 0.0)
    c4 = float(np.mean(overC ** 2 + overT ** 2))

    grad = np.zeros_like(u)
    grad[:, 0] = 2.0 * overC * u[:, 0] / magC / N
    grad[:, 1] = 2.0 * overC * u[:, 1] / magC / N
    grad[:, 2] = 2.0 * overT * u[:, 2] / magT / N
    grad[:, 3] = 2.0 * overT * u[:, 3] / magT / N
    return c4, grad


# ---------------------------------------------------------------------------
# Total objective: forward pass, injections, one backward sweep
# ---------------------------------------------------------------------------

def cost_and_grad(u, H0, Hc, A, Psi0_code, Psi0_err, Psi_target, dt, weights,
                  eps_max=EPS_MAX, c3_normalize=True, want_grad=True):
    """
    C_tot = sum_i w_i C_i and its gradient with respect to the PHYSICAL pulse u.

    weights : (w1, w2, w3, w4) for (fidelity, ET, velocity-variance, amplitude).
    Setting w2 = w3 = 0 gives the 'Ord' variant from this same code path, exactly
    as in EST/grape_jax.make_cost_and_grad -- the whole EsT-vs-Ord comparison
    rests on the control differing by weights alone.

    Returns (cost, grad, terms) with grad shaped (N,4), or (cost, None, terms)
    when want_grad is False.
    """
    w1, w2, w3, w4 = (float(x) for x in weights)
    M = Psi0_code.shape[1]
    Psi0 = np.concatenate([Psi0_code, Psi0_err], axis=1)          # (n,2M)

    out = propagate_trajectory(u, H0, Hc, Psi0, dt, want_eig=want_grad)
    if want_grad:
        traj, w_stack, V_stack, ew_stack, p_stack = out
    else:
        traj = out
    code, err = traj[:, :, :M], traj[:, :, M:]
    N = traj.shape[0] - 1
    n = traj.shape[1]

    c1, inj1_code = fidelity_cost(code[-1], Psi_target)
    c2, inj2_code, inj2_err = et_cost(code, err, A)
    c3, inj3_code = velocity_variance_cost(code, dt, normalize=c3_normalize)
    c4, grad4 = amplitude_cost(u, eps_max)
    terms = {"c1": float(c1), "c2": float(c2), "c3": float(c3), "c4": float(c4)}
    cost = w1 * c1 + w2 * c2 + w3 * c3 + w4 * c4

    if not want_grad:
        return float(cost), None, terms

    # Assemble the per-step injection dL_k/dPsi*_k over the stacked (n,2M) batch.
    inj = np.zeros((N + 1, n, 2 * M), dtype=complex)
    inj[:, :, :M] += w2 * inj2_code + w3 * inj3_code
    inj[:, :, M:] += w2 * inj2_err
    inj[N, :, :M] += w1 * inj1_code                # C1 is terminal-only

    # Backward sweep. Lambda_k = inj_k + U_k^dag Lambda_{k+1}; the gradient at
    # step k reads Lambda_{k+1} against Psi_k through dU_k/du.
    grad = np.zeros((N, 4))
    lam = inj[N].copy()
    for k in range(N - 1, -1, -1):
        Vk, wk, ewk = V_stack[k], w_stack[k], ew_stack[k]
        q = Vk.conj().T @ lam                       # V^dag Lambda_{k+1}
        p = p_stack[k]                              # V^dag Psi_k, from forward

        # dU_k/du_j = V (Phi o (V^dag Hc_j V)) V^dag, with the near-degenerate
        # branch replaced by its L'Hopital limit -- core/grape_core.py:174-179.
        dw = wk[:, None] - wk[None, :]
        near = np.abs(dw) < DEGEN_TOL
        dw_safe = np.where(near, 1.0, dw)
        Phi = (ewk[:, None] - ewk[None, :]) / dw_safe
        Phi = np.where(near, (-1j * dt * ewk)[:, None], Phi)

        VH = Vk.conj().T
        X_all = VH[None, :, :] @ (Hc @ Vk)          # (4,n,n), one shot
        PhiX = Phi[None, :, :] * X_all
        # 2 Re tr(Lambda^dag dU Psi) = 2 Re sum_m q*_a (Phi o X)_ab p_b
        grad[k] = 2.0 * np.real(
            np.einsum('am,jab,bm->j', np.conj(q), PhiX, p, optimize=True))

        lam = Vk @ (np.conj(ewk)[:, None] * q) + inj[k]

    grad += w4 * grad4
    return float(cost), grad, terms


# ---------------------------------------------------------------------------
# Multi-truncation objective + scipy bridge
# ---------------------------------------------------------------------------

def build_gate_objective(gate, N, dt=DT, n_t=N_T, trunc_list=TRUNC_LIST,
                         weights=(1.0, 0.7, 7.0, 1.0), band=BAND_MHZ,
                         ramp_ns=RAMP_NS, eps_max=EPS_MAX, use_control_mask=True,
                         c3_normalize=True):
    """
    Assemble the scipy-ready objective for a named gate.

    Same signature, same truncation average (Heeres Eq. 23) and same return
    triple as EST/grape_jax.build_gate_objective, so the two are drop-in
    interchangeable in EST/train_est.py's stage loop and in any comparison
    script.

    Returns
    -------
    objective : x (flat float64) -> (cost float, grad flat float64), for
                scipy.optimize.minimize(..., jac=True)
    report    : x -> dict of per-truncation per-term costs
    constrain_np : x -> (N,4) physical pulse, for saving and constraint checks
    """
    cmask = kitten_code.control_mask(gate) if use_control_mask else None
    constrain, constrain_adj = make_constrainer(N, dt, band=band,
                                                ramp_ns=ramp_ns, cmask=cmask)

    systems = []
    for n_c in trunc_list:
        H0, Hc_list = make_hamiltonian_est(n_t, n_c)
        A, _ = make_ops(n_t, n_c)
        systems.append((
            np.asarray(H0, dtype=complex),
            np.stack([np.asarray(h, dtype=complex) for h in Hc_list]),
            np.asarray(A, dtype=complex),
            np.asarray(kitten_code.cardinals(n_t, n_c), dtype=complex),
            np.asarray(kitten_code.error_cardinals(n_t, n_c), dtype=complex),
            np.asarray(kitten_code.gate_target(gate, n_t, n_c), dtype=complex),
        ))

    K = len(trunc_list)

    def _physical(x):
        u_raw = np.asarray(x, dtype=np.float64).reshape(N, 4)
        return u_raw if constrain is None else constrain(u_raw)

    def objective(x):
        u = _physical(x)
        cost = 0.0
        grad = np.zeros((N, 4))
        for (H0, Hc, A, P0c, P0e, Ptg) in systems:
            c, g, _ = cost_and_grad(u, H0, Hc, A, P0c, P0e, Ptg, dt, weights,
                                    eps_max=eps_max, c3_normalize=c3_normalize)
            cost += c
            grad += g
        grad /= K
        if constrain_adj is not None:
            grad = constrain_adj(grad)
        return cost / K, grad.ravel()

    def report(x):
        u = _physical(x)
        out = {}
        for n_c, (H0, Hc, A, P0c, P0e, Ptg) in zip(trunc_list, systems):
            _, _, terms = cost_and_grad(u, H0, Hc, A, P0c, P0e, Ptg, dt, weights,
                                        eps_max=eps_max,
                                        c3_normalize=c3_normalize,
                                        want_grad=False)
            out[n_c] = terms
        return out

    def constrain_np(x):
        return np.asarray(_physical(x), dtype=np.float64)

    return objective, report, constrain_np
