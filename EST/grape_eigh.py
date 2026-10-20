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
from core.grape_core import derivative_penalty, make_ops
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

# The cost terms exactly as App. C prints them. A copy of
# EST.grape_jax.PAPER_COST_FORM (this module must not import jax);
# EST/test_grape_eigh.py pins the two as equal. Pass as **PAPER_COST_FORM to
# cost_and_grad / build_gate_objective; omitting it gives the repo's form.
#   c2_norm_power=1            Eq. (C2) divides by N_norm, not N_norm^2
#   c2_include_endpoint=False  Eq. (C2) sums i = 0 .. N-1
#   c3_normalize=False         Eqs. (C6)-(C7): raw variance of v
#   c3_dt=DT*1e3               v in rad/ns (the paper's Delta t = 1 ns)
PAPER_COST_FORM = dict(c2_norm_power=1, c2_include_endpoint=False,
                       c3_normalize=False, c3_dt=DT * 1e3)


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


def et_cost(code_traj, err_traj, A, norm_power=2, include_endpoint=True):
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

    The printed form is opt-in, exactly as in EST/grape_jax.et_cost (see
    PAPER_COST_FORM):
      norm_power=1          F = |o|^2 / sqrt(s), with s = ||A psi_C||^2 + 1e-24
                            under the root (grape_jax's eps). Then
                              dC2/dpsi^E*_k = w (o*/sqrt s) A psi^C_k
                              dC2/dpsi^C*_k = w [ (o/sqrt s) A^dag psi^E_k
                                               - (|o|^2/(2 s^{3/2})) A^dag A psi^C_k ]
      include_endpoint=False  average over t = 0 .. N-1 (T = N steps), and the
                            injection at t = N is zero.
    The defaults run the original arithmetic unchanged.
    """
    if norm_power not in (1, 2):
        raise ValueError(f"norm_power must be 1 or 2, got {norm_power!r}")
    T_all = code_traj.shape[0]
    if not include_endpoint:
        code_traj, err_traj = code_traj[:-1], err_traj[:-1]
    T, n, M = code_traj.shape
    Ad = A.conj().T
    AdA = Ad @ A

    APsi = np.einsum('ij,tjm->tim', A, code_traj)                 # (T,n,M)
    w = -1.0 / (T * M)
    Ad_err = np.einsum('ij,tjm->tim', Ad, err_traj)
    AdA_code = np.einsum('ij,tjm->tim', AdA, code_traj)
    o = np.einsum('tim,tim->tm', np.conj(err_traj), APsi)         # (T,M)
    if norm_power == 2:
        s = np.sum(np.abs(APsi) ** 2, axis=1) + 1e-12             # (T,M)
        F_et = np.abs(o) ** 2 / s
        c2 = 1.0 - np.mean(F_et)
        inj_err = w * (np.conj(o) / s)[:, None, :] * APsi
        inj_code = w * ((o / s)[:, None, :] * Ad_err
                        - (np.abs(o) ** 2 / s ** 2)[:, None, :] * AdA_code)
    else:
        s = np.sum(np.abs(APsi) ** 2, axis=1) + 1e-24
        r = np.sqrt(s)
        F_et = np.abs(o) ** 2 / r
        c2 = 1.0 - np.mean(F_et)
        inj_err = w * (np.conj(o) / r)[:, None, :] * APsi
        inj_code = w * ((o / r)[:, None, :] * Ad_err
                        - (np.abs(o) ** 2 / (2.0 * s * r))[:, None, :] * AdA_code)

    if T != T_all:                       # endpoint excluded: zero row at t = N
        pad = np.zeros((1, n, M), dtype=complex)
        inj_code = np.concatenate([inj_code, pad], axis=0)
        inj_err = np.concatenate([inj_err, pad], axis=0)
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


def photon_number_mismatch_cost(code_words, Nop, eps=1e-12):
    """
    C5, running. Eq. (A7)'s codeword photon-number difference,

        Delta_nbar_L(t) = <0_L(t)|a^dag a|0_L(t)> - <1_L(t)|a^dag a|1_L(t)> ,

    normalized per step by n0 + n1 and squared:

        r_t = (n0 - n1) / (n0 + n1 + eps) ,     C5 = mean_{t=0..N} r_t^2 .

    Because a^dag a >= 0, |n0 - n1| <= n0 + n1, so C5 is in [0, 1]. The paper
    prints no normalization; this one is the repo's choice. Squaring keeps it
    smooth at r = 0, and r_0 = 0 exactly for the kitten code (both words carry
    nbar = 2).

    NECESSARY, NOT SUFFICIENT. Delta_nbar_L is twice the Z coefficient of
    Delta_QEC's (a, a) term (App. A, Eq. A9). It leaves the X/Y coefficients
    <0_L|n|1_L> and the (I, a) pairs unconstrained, so minimizing C5 lowers only
    part of Delta_QEC.

    code_words : (T, n, 2) evolved |0_L(t)>, |1_L(t)> -- the +Z, -Z cardinals.

    Injection, with S = n0 + n1 + eps and d<psi|N|psi>/dpsi* = N psi:

        dC5/dpsi0*_t =  (1/T) 2 r_t (2 n1_t + eps)/S_t^2  N psi0_t
        dC5/dpsi1*_t = -(1/T) 2 r_t (2 n0_t + eps)/S_t^2  N psi1_t
    """
    T = code_words.shape[0]
    Npsi = np.einsum('ij,tjm->tim', Nop, code_words)               # (T,n,2)
    nbar = np.real(np.einsum('tim,tim->tm', np.conj(code_words), Npsi))
    n0, n1 = nbar[:, 0], nbar[:, 1]
    S = n0 + n1 + eps
    r = (n0 - n1) / S
    c5 = float(np.mean(r ** 2))

    inj = np.empty_like(code_words)
    inj[:, :, 0] = ((2.0 * r * (2.0 * n1 + eps) / S ** 2) / T)[:, None] * Npsi[:, :, 0]
    inj[:, :, 1] = ((-2.0 * r * (2.0 * n0 + eps) / S ** 2) / T)[:, None] * Npsi[:, :, 1]
    return c5, inj, r


def leakage_cost(code_words, err_words, A):
    """
    C_L, running. Eq. 7's leakage of the error state out of the INSTANTANEOUS
    error space, averaged over the gate:

        C_L = mean_{t=0..N} L(t),    L(t) = 1 - 1/2 Tr[P_E(t) E(t) E(t)^dag]
        P_E = G S^-1 G^dag,          G = A C(t),  S = G^dag G   (2 x 2)

    with C(t), E(t) the two evolved code and error WORDS, (n, 2) each. The
    six-cardinal average of EST/diagnostics.leakage_Ej equals this two-word form
    exactly: the six cardinals sum to 3 x the identity on the error space.
    P_E is built from G itself, not from diagnostics' polar isometry, but both
    span the same space, so the projector -- and L -- are identical.

    Relation to C2 (repo form, N_norm^2). Per cardinal, the normalized image
    a|psi_C>/||a|psi_C>|| lies in E(t), so 1 - F_ET = L + (1 - L)(1 - f), with f
    the overlap of the projected, renormalized error state with that image.
    Hence C2 = C_L + C_mis with C_mis >= 0, and C_L <= C2: this term reweights
    the leakage part of C2 only.

    Injections (Wirtinger, Lambda = dC/dPsi*), with B = G^dag E, Y = S^-1 B and
    f = 1/2 Tr[B^dag Y], so that P_E E = G Y:

        dC_L/dE*_t = -(1/T) 1/2 G Y                      (= -(1/T) 1/2 P_E E)
        dC_L/dC*_t = -(1/T) 1/2 A^dag (E - G Y) Y^dag   (= ... A^dag (1-P_E) E B^dag S^-1)

    The code-word term carries (1 - P_E): moving the space helps only where the
    error states sit outside it. S is Hermitian, so B^dag S^-1 = Y^dag.

    code_words, err_words : (T, n, 2) evolved |0_L>,|1_L> and |0_E>,|1_E>
                            (the +Z, -Z columns of the code and error stacks)
    Returns (c_leak, inj_code (T,n,2), inj_err (T,n,2), L (T,)).
    """
    T = code_words.shape[0]
    G = np.einsum('ij,tjm->tim', A, code_words)                   # (T,n,2)
    Gh = np.conj(np.swapaxes(G, 1, 2))                            # (T,2,n)
    S = Gh @ G                                                    # (T,2,2)
    B = Gh @ err_words                                            # (T,2,2)
    Y = np.linalg.solve(S, B)                                     # S^-1 B
    f = 0.5 * np.real(np.einsum('tkm,tkm->t', np.conj(B), Y))
    L = 1.0 - f
    c_leak = float(np.mean(L))

    GY = G @ Y                                                    # P_E E
    inj_err = (-0.5 / T) * GY
    inj_code = (-0.5 / T) * np.einsum(
        'ij,tjm->tim', A.conj().T, (err_words - GY) @ np.conj(np.swapaxes(Y, 1, 2)))
    return c_leak, inj_code, inj_err, L


def smoothness_cost(u):
    """
    C6. The incumbent cat-code smoothness penalty, sum_k ||u_{k+1} - u_k||^2 on
    the PHYSICAL pulse, reused from core.grape_core.derivative_penalty rather
    than restated. A pure function of u, so -- like C4 -- no costate.

    It is a SUM, not a mean, exactly as in the incumbent; its weight therefore
    does not transfer across N or dt without rescaling.
    """
    pen, grad = derivative_penalty(np.asarray(u, dtype=np.float64))
    return float(pen), grad


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

def _unpack_weights(weights):
    """(w1..w4), (w1..w4, w_err, w5, w6) or (..., w_L) -> 8 floats, extras zero-padded."""
    w = [float(x) for x in weights]
    if len(w) not in (4, 7, 8):
        raise ValueError(f"weights must have 4, 7 or 8 entries, got {len(w)}")
    return w + [0.0] * (8 - len(w))


def cost_and_grad(u, H0, Hc, A, Psi0_code, Psi0_err, Psi_target, dt, weights,
                  eps_max=EPS_MAX, c3_normalize=True, want_grad=True,
                  Psi_target_err=None, Nop=None, c2_norm_power=2,
                  c2_include_endpoint=True, c3_dt=None):
    """
    C_tot = sum_i w_i C_i and its gradient with respect to the PHYSICAL pulse u.

    weights : (w1, w2, w3, w4) for (fidelity, ET, velocity-variance, amplitude),
              or (w1, w2, w3, w4, w_err, w5, w6) adding the error-space
              infidelity `cerr`, the photon-number mismatch C5 and the
              smoothness penalty C6, or that plus w_L for the leakage C_L
              (`cleak`). A 4-tuple is exactly the original objective: the extra
              injections are skipped, not multiplied by 0. A 7-tuple, or an
              8-tuple with w_L = 0, is the 7-term objective bit for bit; `cleak`
              is then only reported.
    Setting w2 = w3 = 0 gives the 'Ord' variant from this same code path, exactly
    as in EST/grape_jax.make_cost_and_grad -- the whole EsT-vs-Ord comparison
    rests on the control differing by weights alone.

    Psi_target_err : (n, 6) error-space targets (kitten_code.error_gate_target);
                     needed for `cerr`, which is C1's form on the error cardinals.
    Nop            : (n, n) cavity photon number a^dag a; needed for C5.
    Either may be None when its weight is zero; the term is then omitted from
    `terms` as well.

    c2_norm_power, c2_include_endpoint : forwarded to `et_cost`.
    c3_dt : time step used for the C3 velocity only (None -> dt); propagation
            always uses dt. Together with c3_normalize these select the cost
            form; pass **PAPER_COST_FORM for App. C as printed.

    These four terms exist on this pipeline only. Their autodiff reference
    lives in EST/test_grape_eigh.py, not in EST/grape_jax.py.

    Returns (cost, grad, terms) with grad shaped (N,4), or (cost, None, terms)
    when want_grad is False.
    """
    w1, w2, w3, w4, w_err, w5, w6, w_leak = _unpack_weights(weights)
    if w_err and Psi_target_err is None:
        raise ValueError("w_err > 0 needs Psi_target_err")
    if w5 and Nop is None:
        raise ValueError("w5 > 0 needs Nop")
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
    c2, inj2_code, inj2_err = et_cost(code, err, A, norm_power=c2_norm_power,
                                      include_endpoint=c2_include_endpoint)
    c3, inj3_code = velocity_variance_cost(code, dt if c3_dt is None else c3_dt,
                                           normalize=c3_normalize)
    c4, grad4 = amplitude_cost(u, eps_max)
    terms = {"c1": float(c1), "c2": float(c2), "c3": float(c3), "c4": float(c4)}
    cost = w1 * c1 + w2 * c2 + w3 * c3 + w4 * c4

    # Extra terms: evaluated whenever their inputs exist (so a report can score
    # any pulse on them), added to the cost only at nonzero weight.
    if Psi_target_err is not None:
        c_err, inj_err_T = fidelity_cost(err[-1], Psi_target_err)
        terms["cerr"] = float(c_err)
        if w_err:
            cost += w_err * c_err
    if Nop is not None:
        # Columns 0, 1 are +Z, -Z, i.e. exactly |0_L(t)>, |1_L(t)>.
        c5, inj5_words, _ = photon_number_mismatch_cost(code[:, :, :2], Nop)
        terms["c5"] = c5
        if w5:
            cost += w5 * c5
    if Psi_target_err is not None or Nop is not None:
        c6, grad6 = smoothness_cost(u)
        terms["c6"] = c6
        if w6:
            cost += w6 * c6
    if Psi_target_err is not None or Nop is not None or w_leak:
        # Columns 0, 1 of both stacks are the +Z, -Z cardinals: the code words
        # |0_L(t)>, |1_L(t)> and the error words |0_E(t)>, |1_E(t)>.
        c_leak, inj_leak_code, inj_leak_err, _ = leakage_cost(
            code[:, :, :2], err[:, :, :2], A)
        terms["cleak"] = c_leak
        if w_leak:
            cost += w_leak * c_leak

    if not want_grad:
        return float(cost), None, terms

    # Assemble the per-step injection dL_k/dPsi*_k over the stacked (n,2M) batch.
    inj = np.zeros((N + 1, n, 2 * M), dtype=complex)
    inj[:, :, :M] += w2 * inj2_code + w3 * inj3_code
    inj[:, :, M:] += w2 * inj2_err
    inj[N, :, :M] += w1 * inj1_code                # C1 is terminal-only
    if w_err:
        inj[N, :, M:] += w_err * inj_err_T         # cerr is terminal, error columns
    if w5:
        inj[:, :, :2] += w5 * inj5_words           # C5 reads the two code words
    if w_leak:
        inj[:, :, :2] += w_leak * inj_leak_code    # C_L reads both pairs of words
        inj[:, :, M:M + 2] += w_leak * inj_leak_err

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
    if w6:
        grad += w6 * grad6
    return float(cost), grad, terms


# ---------------------------------------------------------------------------
# Multi-truncation objective + scipy bridge
# ---------------------------------------------------------------------------

def build_gate_objective(gate, N, dt=DT, n_t=N_T, trunc_list=TRUNC_LIST,
                         weights=(1.0, 0.7, 7.0, 1.0), band=BAND_MHZ,
                         ramp_ns=RAMP_NS, eps_max=EPS_MAX, use_control_mask=True,
                         c3_normalize=True, extra_terms=False, **form):
    """
    Assemble the scipy-ready objective for a named gate.

    `weights` may be a 4-, 7- or 8-tuple (see `cost_and_grad`). The error-space
    target and a^dag a are built when `extra_terms` is set or the weights are a
    7- or 8-tuple; `report` then also returns `cerr`, `c5`, `c6`, `cleak`. With a
    4-tuple and extra_terms=False the objective and report are exactly the
    original ones.

    Same signature, same truncation average (Heeres Eq. 23) and same return
    triple as EST/grape_jax.build_gate_objective, so the two are drop-in
    interchangeable in EST/train_est.py's stage loop and in any comparison
    script.

    `**form` (c2_norm_power, c2_include_endpoint, c3_dt) and `c3_normalize`
    select the cost-term form; pass **PAPER_COST_FORM for App. C as printed.
    Omitted means the repo's form.

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

    extras = bool(extra_terms) or len(tuple(weights)) >= 7
    _unpack_weights(weights)                      # validate length up front

    systems = []
    for n_c in trunc_list:
        H0, Hc_list = make_hamiltonian_est(n_t, n_c)
        A, _ = make_ops(n_t, n_c)
        A = np.asarray(A, dtype=complex)
        if extras:
            kw = {"Psi_target_err": np.asarray(
                      kitten_code.error_gate_target(gate, n_t, n_c), dtype=complex),
                  "Nop": A.conj().T @ A}
        else:
            kw = {}
        systems.append((
            np.asarray(H0, dtype=complex),
            np.stack([np.asarray(h, dtype=complex) for h in Hc_list]),
            A,
            np.asarray(kitten_code.cardinals(n_t, n_c), dtype=complex),
            np.asarray(kitten_code.error_cardinals(n_t, n_c), dtype=complex),
            np.asarray(kitten_code.gate_target(gate, n_t, n_c), dtype=complex),
            kw,
        ))

    K = len(trunc_list)

    def _physical(x):
        u_raw = np.asarray(x, dtype=np.float64).reshape(N, 4)
        return u_raw if constrain is None else constrain(u_raw)

    def objective(x):
        u = _physical(x)
        cost = 0.0
        grad = np.zeros((N, 4))
        for (H0, Hc, A, P0c, P0e, Ptg, kw) in systems:
            c, g, _ = cost_and_grad(u, H0, Hc, A, P0c, P0e, Ptg, dt, weights,
                                    eps_max=eps_max, c3_normalize=c3_normalize,
                                    **kw, **form)
            cost += c
            grad += g
        grad /= K
        if constrain_adj is not None:
            grad = constrain_adj(grad)
        return cost / K, grad.ravel()

    def report(x):
        u = _physical(x)
        out = {}
        for n_c, (H0, Hc, A, P0c, P0e, Ptg, kw) in zip(trunc_list, systems):
            _, _, terms = cost_and_grad(u, H0, Hc, A, P0c, P0e, Ptg, dt, weights,
                                        eps_max=eps_max,
                                        c3_normalize=c3_normalize,
                                        want_grad=False, **kw, **form)
            out[n_c] = terms
        return out

    def constrain_np(x):
        return np.asarray(_physical(x), dtype=np.float64)

    return objective, report, constrain_np
