"""
JAX autodiff layer for the error-semitransparent (EsT) cost functions.

The repo's production GRAPE (core/grape_core.py) computes gradients with a
hand-derived adjoint: forward pass stores the trajectory, backward pass
propagates costates, and each step's gradient uses the eigenbasis Phi-matrix
with an explicit near-degenerate limit branch. That is fast and validated, but
every NEW running-cost term needs its own hand-derived adjoint (done once, for
leakage_grad). The EsT construction adds two running costs (C2, C3), so this
module replaces only propagation + gradient with an autodiff path. The physics
(EST/device.py), the state construction (EST/kitten_code.py), the band-limit
definition (core/fourier_cutoff.py) and the L-BFGS-B driver are all shared.

Design notes
------------
Double precision. JAX defaults to float32/complex64; GRAPE needs ~1e-6 fidelity
precision. `jax_enable_x64` MUST be set before any other jax call.

Propagator: expm, not eigh. grape_core.step_data diagonalizes H_k and hand-
derives the eigenbasis limit for near-degenerate eigenvalues (the `near` mask in
_fidelity_core) specifically because d/dH of an eigendecomposition is singular at
degeneracies. Autodiff through jnp.linalg.eigh hits that SAME singularity, so
naively letting autodiff "handle it" via eigh would silently reintroduce the
exact bug that was fixed by hand. jax.scipy.linalg.expm computes the identical
propagator exp(-i dt H_k) by scaling-and-squaring Pade without ever
diagonalizing H_k, so there is no degeneracy for autodiff to trip over.

One propagation, not two. The code cardinals and the error cardinals see the
same u and the same H, and propagation acts column-wise, so they are stacked
into a single (n, 12) batch and propagated once. This is exact, not an
approximation, and halves the expm work per gradient evaluation.

Constraints live INSIDE the cost. The band-limit, the Gaussian ramp and the
per-gate control mask are applied to the raw optimizer variable at the top of
`total_cost`, so autodiff differentiates the whole chain and there is no manual
chain rule to get wrong. The consequence -- that box bounds constrain the raw
pre-image rather than the physical pulse -- is the same caveat already
documented for the existing pipeline at core/optimizer.py:73-76, and is why the
circular amplitude cap is carried by a penalty (`amplitude_cost`) rather than by
bounds.

Deviation from the paper, deliberate: Eq. (C2) as printed divides the squared
overlap by a single power of N_norm = sqrt(<psi_C|a^dag a|psi_C>), which does not
bound F_ET in [0, 1]. `et_cost` divides by N_norm^2 instead, which does bound it
via Cauchy-Schwarz. See the note in that function.
"""

import warnings

import jax
jax.config.update("jax_enable_x64", True)   # must precede other jax use
import jax.numpy as jnp
from jax.scipy.linalg import expm
import numpy as np

# The controls u are real but the states are complex, so reverse mode produces a
# complex cotangent for u and casts it to real on every backward pass, emitting a
# ComplexWarning from jax/_src/lax/lax.py. The discarded imaginary part is zero by
# construction (the cost is real-valued and u is real), and this is verified, not
# assumed: EST/test_grape_jax.py checks the C1 gradient against
# core.grape_core.fidelity_multi_state's independently-derived analytic adjoint to
# rtol=1e-6, and every component of grad(C_tot) against central differences.
# Silenced narrowly here, in the same spirit as the Accelerate BLAS warnings
# suppressed at core/grape_core.py:7-10.
warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning,
                        module=r"jax\._src\.lax\.lax")

from core.grape_core import make_ops
from EST.device import (BAND_MHZ, DT, EPS_MAX, N_T, RAMP_NS, TRUNC_LIST,
                        band_mask, make_hamiltonian_est, ramp_envelope)
from EST import kitten_code

# arccos'(x) = -1/sqrt(1-x^2) diverges at x = 1, and consecutive states in a 1 ns
# trajectory ARE nearly parallel, so the Fubini-Study distance in C3 is evaluated
# at a clipped argument. 1 - 1e-9 bounds |d arccos| at ~2e4, which is large enough
# to be a real gradient contribution and small enough not to swamp the others.
_FS_CLIP = 1.0 - 1e-9


def _abs2(z):
    """|z|^2 for complex z, without routing the gradient through jnp.abs.

    jnp.abs on a complex array makes reverse-mode cast a complex cotangent to
    real, which is correct here but emits a ComplexWarning on every backward
    pass. Writing the modulus out in real arithmetic gives the identical value
    and gradient with no cast.
    """
    return z.real ** 2 + z.imag ** 2


# ---------------------------------------------------------------------------
# Constraint chain: raw optimizer variable -> physical pulse
# ---------------------------------------------------------------------------

def make_constrainer(N, dt=DT, band=BAND_MHZ, ramp_ns=RAMP_NS, cmask=None):
    """
    Build the (N,4) -> (N,4) map from the raw optimizer variable to the physical
    pulse: band-limit, then Gaussian ramp, then the per-gate control mask.

    The band-limit is the same orthogonal projection as
    core.fourier_cutoff.project_bandlimit -- IFFT . mask . FFT applied to the
    COMPLEX drives eps_C = u0 + i u1 and eps_T = u2 + i u3 -- reimplemented in
    jnp so autodiff can see through it. The mask itself comes from
    core.fourier_cutoff.make_band_mask via EST.device.band_mask, so the
    training-time constraint and the post-hoc check in
    core.fourier_cutoff.out_of_band_energy_fraction cannot drift apart.

    Order matters, and the choice is made on measurement rather than taste.
    On white-noise input at N=1000, dt=1 ns, 48 ns ramp (see
    EST/test_grape_jax.py:ConstraintSatisfactionTest):

        project -> ramp :  0.34% out-of-band,  endpoints at 0.4% of mid-amplitude
        ramp -> project :  0%    out-of-band,  endpoints at 23%  of mid-amplitude

    Projecting last gives exact band-limiting but smears the envelope so badly
    that the endpoints are a quarter of full scale, which defeats the entire
    point of the ramp. Ramping last is therefore used: the residual band
    violation is sub-percent and is measured by the test suite, not assumed.

    Returns None if every constraint is disabled, which callers treat as identity.
    """
    parts = []

    if band is not None:
        mask = jnp.asarray(band_mask(N, dt, band).astype(np.float64))
        parts.append(("band", mask))
    if ramp_ns is not None:
        env = jnp.asarray(ramp_envelope(N, dt, ramp_ns))
        parts.append(("ramp", env))
    if cmask is not None:
        parts.append(("mask", jnp.asarray(np.asarray(cmask, dtype=np.float64))))

    if not parts:
        return None

    def constrain(u):
        for kind, arr in parts:
            if kind == "band":
                zC = jnp.fft.ifft(jnp.fft.fft(u[:, 0] + 1j * u[:, 1]) * arr)
                zT = jnp.fft.ifft(jnp.fft.fft(u[:, 2] + 1j * u[:, 3]) * arr)
                u = jnp.stack([zC.real, zC.imag, zT.real, zT.imag], axis=1)
            elif kind == "ramp":
                u = u * arr[:, None]
            else:
                u = u * arr[None, :]
        return u

    return constrain


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def propagate_trajectory(u, H0, Hc, Psi0, dt):
    """
    u    : (N,4) real controls [C_I, C_Q, T_I, T_Q]
    H0   : (n,n) complex drift
    Hc   : (4,n,n) complex control operators
    Psi0 : (n,M) complex, M state vectors stacked as columns
    dt   : float, step size (us)

    Returns Psi_traj : (N+1,n,M), the state at every step t = 0..N.

    The scan body is wrapped in jax.checkpoint: reverse-mode through N=1000 steps
    of expm on a 72x72 would otherwise tape every Pade intermediate at every step.
    Rematerializing the forward step during the backward pass trades roughly a 2x
    forward cost for a large constant-factor drop in peak memory.
    """
    @jax.checkpoint
    def step(psi, u_k):
        Hk = H0 + jnp.tensordot(u_k, Hc, axes=([0], [0]))
        psi_next = expm(-1j * dt * Hk) @ psi
        return psi_next, psi_next

    _, Psi_rest = jax.lax.scan(step, Psi0, u)                 # (N,n,M)
    return jnp.concatenate([Psi0[None], Psi_rest], axis=0)    # (N+1,n,M)


# ---------------------------------------------------------------------------
# Cost terms -- pure forward-mode math; JAX supplies every gradient
# ---------------------------------------------------------------------------

def fidelity_cost(Psi_final, Psi_target):
    """
    C1, Eq. (C1). Ordinary infidelity, incoherently averaged over the six logical
    cardinal points. Matches grape_core.fidelity_multi_state (up to the 1-F sign).

    Incoherent is correct here and does not need the coherent variant that
    core.cat_code's 2-state training required: with all six cardinal points
    present, linearity of U pins every relative phase and leaves only the global
    phase free. See the note in EST/kitten_code.py.
    """
    overlaps = jnp.sum(jnp.conj(Psi_target) * Psi_final, axis=0)   # (M,)
    return 1.0 - jnp.mean(_abs2(overlaps))


def et_cost(Psi_code_traj, Psi_err_traj, A):
    """
    C2, Eq. (C2). Running error-transparency fidelity: at every step, compare the
    normalized "just lost a photon" state a|psi_C(t)> against the independently
    evolved error cardinal point |psi_E(t)>.

    NORMALIZATION -- deviation from the paper, flagged not hidden. Eq. (C2) as
    printed divides |<psi_E|a|psi_C>|^2 by a single power of
    N_norm = sqrt(<psi_C|a^dag a|psi_C>), which is not bounded by 1. Dividing by
    N_norm^2 is, via Cauchy-Schwarz:
        |<psi_E|a|psi_C>|^2 <= <psi_C|a^dag a|psi_C> * <psi_E|psi_E>
    with psi_E of unit norm. This implements the N_norm^2 version, so F_ET is a
    genuine fidelity in [0,1]. If exact numerical agreement with the paper's
    stage-1 value (~0.83) is ever needed, check against their released code
    (Zenodo DOI in the paper); the EsT-vs-Ord comparison this module targets is
    insensitive to the choice, since both variants are scored the same way.
    """
    APsi = jnp.einsum('ij,tjm->tim', A, Psi_code_traj)
    norm_sq = jnp.sum(_abs2(APsi), axis=1)                                # (T,M)
    overlap = jnp.einsum('tim,tim->tm', jnp.conj(Psi_err_traj), APsi)
    F_et = _abs2(overlap) / (norm_sq + 1e-12)
    return 1.0 - jnp.mean(F_et)    # mean over (time, cardinal points)


def velocity_variance_cost(Psi_code_traj, dt, normalize=True):
    """
    C3, Eqs. (C4)-(C7). Penalizes non-uniform speed through Hilbert space along
    the code trajectory.

    This term exists to stop the optimizer parking the dynamics in order to cheat
    C2: a state that does not move trivially satisfies error transparency. Note
    that it penalizes UNEVEN motion, not slow motion -- a trajectory that sprints
    to the target and then sits still has high velocity variance, while a
    uniformly-paced one does not. It does not itself improve transparency, which
    is why the paper's stage-2 schedule drops it to zero.

    NORMALIZATION -- a deviation from the printed equations, on evidence. The raw
    variance of v = d_FS/dt carries units of (rad/us)^2 and is numerically huge:
    measured on a 1 us X-gate trajectory at dt = 1 ns it is ~100, so with the
    paper's w3 = 5-10 the C3 contribution would be ~700 against C1, C2 in [0, 1].
    The fidelity terms would be numerically invisible and the optimizer would
    minimize C3 alone. Dividing by mean(v)^2 -- the squared coefficient of
    variation -- makes the term dimensionless and O(0.1-1), which is the only
    reading under which the paper's stated weights (w1, w2, w3) = (1, 0.6-0.75,
    5-10) form a sensible schedule. It is also dt-invariant, so refining the time
    step does not silently reweight the objective.

    Pass normalize=False for the literal Eqs. (C6)-(C7) form.
    """
    ov = jnp.einsum('tim,tim->tm',
                    jnp.conj(Psi_code_traj[:-1]), Psi_code_traj[1:])   # (N,M)
    mod = jnp.sqrt(_abs2(ov) + 1e-30)
    fsd = 2.0 * jnp.arccos(jnp.clip(mod, 0.0, _FS_CLIP))               # Eq. (C4)
    v = fsd / dt                                                      # Eq. (C5)
    var = jnp.var(v, axis=0)                                          # Eq. (C6)
    if normalize:
        var = var / (jnp.mean(v, axis=0) ** 2 + 1e-12)
    return jnp.mean(var)                                              # Eq. (C7)


def amplitude_cost(u, eps_max=EPS_MAX):
    """
    C4. Circular cap on the drive amplitude, |eps| <= eps_max.

    NOT expressible with the repo's existing tools, which is why this is new.
    With Hc = [A+Ad, 1j*(A-Ad), ...] the physical cavity drive is
    eps_C = u0 - i u1, so Table I's eps_max/2pi = 4 MHz caps sqrt(u0^2 + u1^2).
    Both grape_core.amplitude_penalty (core/grape_core.py:414) and the L-BFGS-B
    box (core/grape_core.py:583) act per element, and a per-element box at
    eps_max admits |eps| up to sqrt(2)*eps_max ~ 35.5 rad/us.

    Mean rather than sum over time, so the weight w4 does not need rescaling when
    N or the gate duration changes.
    """
    magC = jnp.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2 + 1e-30)
    magT = jnp.sqrt(u[:, 2] ** 2 + u[:, 3] ** 2 + 1e-30)
    over = (jnp.maximum(magC - eps_max, 0.0) ** 2
            + jnp.maximum(magT - eps_max, 0.0) ** 2)
    return jnp.mean(over)


# ---------------------------------------------------------------------------
# Total objective
# ---------------------------------------------------------------------------

def make_cost_terms(H0, Hc, A, Psi0_code, Psi0_err, Psi_target, dt,
                    constrain=None, eps_max=EPS_MAX, c3_normalize=True):
    """
    Return an unjitted `terms(u_raw) -> dict` of the four cost components, so
    both the weighted objective and the reporting path compute them once, the
    same way.

    Psi0_code, Psi0_err, Psi_target : (n, 6) complex
    """
    M = Psi0_code.shape[1]
    Psi0 = jnp.concatenate([Psi0_code, Psi0_err], axis=1)   # (n, 2M), one scan

    def terms(u_raw):
        u = u_raw if constrain is None else constrain(u_raw)
        traj = propagate_trajectory(u, H0, Hc, Psi0, dt)     # (N+1, n, 2M)
        code, err = traj[:, :, :M], traj[:, :, M:]
        return {
            "c1": fidelity_cost(code[-1], Psi_target),
            "c2": et_cost(code, err, A),
            "c3": velocity_variance_cost(code, dt, normalize=c3_normalize),
            "c4": amplitude_cost(u, eps_max),
        }

    return terms


def make_cost_and_grad(H0, Hc, A, Psi0_code, Psi0_err, Psi_target, dt, weights,
                       constrain=None, eps_max=EPS_MAX, c3_normalize=True):
    """
    JIT-compiled `value_and_grad` of C_tot = sum_i w_i C_i (Eq. C8).

    weights : (w1, w2, w3, w4) for (fidelity, ET, velocity-variance, amplitude).
    Setting w2 = w3 = 0 gives the 'Ord' variant from the same code path -- the
    control that the whole EsT-vs-Ord comparison rests on, so it is important
    that it differs from EsT by weights alone and not by a separate implementation.
    """
    w = jnp.asarray(weights, dtype=jnp.float64)
    terms = make_cost_terms(H0, Hc, A, Psi0_code, Psi0_err, Psi_target, dt,
                            constrain=constrain, eps_max=eps_max,
                            c3_normalize=c3_normalize)

    def total_cost(u):
        t = terms(u)
        return w[0] * t["c1"] + w[1] * t["c2"] + w[2] * t["c3"] + w[3] * t["c4"]

    return jax.jit(jax.value_and_grad(total_cost))


def make_report(H0, Hc, A, Psi0_code, Psi0_err, Psi_target, dt,
                constrain=None, eps_max=EPS_MAX, c3_normalize=True):
    """JIT-compiled per-term report, for logging the schedule's progress."""
    return jax.jit(make_cost_terms(H0, Hc, A, Psi0_code, Psi0_err, Psi_target,
                                   dt, constrain=constrain, eps_max=eps_max,
                                   c3_normalize=c3_normalize))


# ---------------------------------------------------------------------------
# Multi-truncation objective + scipy bridge
# ---------------------------------------------------------------------------

def build_gate_objective(gate, N, dt=DT, n_t=N_T, trunc_list=TRUNC_LIST,
                         weights=(1.0, 0.7, 7.0, 1.0), band=BAND_MHZ,
                         ramp_ns=RAMP_NS, eps_max=EPS_MAX, use_control_mask=True,
                         c3_normalize=True):
    """
    Assemble the full scipy-ready objective for a named gate.

    The truncation average follows Heeres Eq. 23, mirroring
    core/optimizer.py:175-178, but as a plain Python loop over separately-jitted
    closures: each n_c has a different Hilbert dimension, so jax.vmap cannot
    batch them. joblib is not used either -- these are XLA calls that already
    release the GIL and saturate the device.

    Returns
    -------
    objective : x (flat float64) -> (cost float, grad flat float64), for
                scipy.optimize.minimize(..., jac=True)
    report    : x -> dict of per-truncation per-term costs
    constrain_np : x -> (N,4) physical pulse as numpy, for saving and for the
                constraint checks
    """
    cmask = kitten_code.control_mask(gate) if use_control_mask else None
    constrain = make_constrainer(N, dt, band=band, ramp_ns=ramp_ns, cmask=cmask)

    cgs, reports = [], []
    for n_c in trunc_list:
        H0_np, Hc_np = make_hamiltonian_est(n_t, n_c)
        A_np, _ = make_ops(n_t, n_c)
        H0 = jnp.asarray(H0_np, dtype=jnp.complex128)
        Hc = jnp.stack([jnp.asarray(h, dtype=jnp.complex128) for h in Hc_np])
        A = jnp.asarray(A_np, dtype=jnp.complex128)

        args = (H0, Hc, A,
                jnp.asarray(kitten_code.cardinals(n_t, n_c), dtype=jnp.complex128),
                jnp.asarray(kitten_code.error_cardinals(n_t, n_c), dtype=jnp.complex128),
                jnp.asarray(kitten_code.gate_target(gate, n_t, n_c), dtype=jnp.complex128),
                dt)
        cgs.append(make_cost_and_grad(*args, weights, constrain=constrain,
                                      eps_max=eps_max, c3_normalize=c3_normalize))
        reports.append(make_report(*args, constrain=constrain, eps_max=eps_max,
                                   c3_normalize=c3_normalize))

    K = len(trunc_list)

    def objective(x):
        u = jnp.asarray(np.asarray(x, dtype=np.float64).reshape(N, 4))
        cost, grad = 0.0, np.zeros((N, 4))
        for cg in cgs:
            c, g = cg(u)
            cost += float(c)
            grad += np.asarray(g, dtype=np.float64)
        return cost / K, (grad / K).ravel()

    def report(x):
        u = jnp.asarray(np.asarray(x, dtype=np.float64).reshape(N, 4))
        return {n_c: {k: float(v) for k, v in r(u).items()}
                for n_c, r in zip(trunc_list, reports)}

    def constrain_np(x):
        u = jnp.asarray(np.asarray(x, dtype=np.float64).reshape(N, 4))
        return np.asarray(u if constrain is None else constrain(u), dtype=np.float64)

    return objective, report, constrain_np
