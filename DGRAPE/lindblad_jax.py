"""
JAX autodiff path for the open-system GRAPE fidelity -- the same Taylor-series step map as
DGRAPE/lindblad_np.py, differentiated by reverse-mode autodiff instead of the hand-derived
adjoint. Both evaluate the identical polynomial for the same (s, m), so their fidelities
and gradients agree to round-off (validation/test_dgrape.py pins it).

Design notes
------------
* Double precision: `jax_enable_x64` is set before any other jax call (as EST/grape_jax.py).
* Steps run in `lax.scan` under `jax.jit`. (s, m) are static arguments, so a new pair
  recompiles once and is then cached.
* `jax.checkpoint` on the step body. Without it reverse mode stores every Taylor term of
  every step (~s*m*N stacks: ~4.5 GB at n_c = 12, N = 500); with it only the per-step
  inputs rho_k are kept and each step's terms are recomputed on the backward pass.
* Jump operators use the flat monomial gather of DGRAPE/model.py (`_flat_sandwich`), not
  dense matmuls -- the same arithmetic as the numpy path.
"""

import warnings
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)   # must precede other jax use
import jax.numpy as jnp
import numpy as np

# u is real and the states complex, so reverse mode casts a complex cotangent of u to real
# and warns. The discarded imaginary part is zero (real cost, real u) -- verified, not
# assumed: validation/test_dgrape.py checks this gradient against the hand-derived adjoint
# and against central differences. Same narrow filter as EST/grape_jax.py.
warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning,
                        module=r"jax\._src\.lax\.lax")

from core.fourier_cutoff import make_band_mask
from core.ramp import ramp_envelope


def model_arrays(model):
    """The jnp arrays the propagator closes over."""
    return dict(
        H_d=jnp.asarray(model.H_d_shifted),
        Hc=jnp.asarray(model.Hc),
        Gdiag=jnp.asarray(model.Gdiag),
        j_src=jnp.asarray(model.jump_flat[0]),
        j_w=jnp.asarray(model.jump_flat[1]),
        j_seg=jnp.asarray(np.repeat(np.arange(model.jump_flat[2].size),
                                    np.diff(np.append(model.jump_flat[3],
                                                      model.jump_flat[0].size)))),
        j_dst=jnp.asarray(model.jump_flat[2]),
    )


def _apply_L(arrs, Heff, R):
    out = -1j * (Heff @ R - R @ Heff.conj().T)
    if arrs["j_src"].size:
        lead, d = R.shape[:-2], R.shape[-1]
        Rf = R.reshape(lead + (d * d,))
        v = jnp.take(Rf, arrs["j_src"], axis=-1) * arrs["j_w"]
        seg = jax.ops.segment_sum(jnp.moveaxis(v, -1, 0), arrs["j_seg"],
                                  num_segments=arrs["j_dst"].size)
        J = jnp.zeros_like(Rf).at[..., arrs["j_dst"]].set(jnp.moveaxis(seg, 0, -1))
        out = out + J.reshape(R.shape)
    return out


def _step(arrs, u_k, R, dt, s, m):
    H = arrs["H_d"] + jnp.tensordot(u_k, arrs["Hc"], axes=1)
    Heff = H - 0.5j * jnp.diag(arrs["Gdiag"])
    tau = dt / s
    for _ in range(s):
        term = R
        acc = R
        for k in range(1, m + 1):
            term = _apply_L(arrs, Heff, term) * (tau / k)
            acc = acc + term
        R = acc
    return R


def propagate(arrs, u, R0, dt, s, m):
    step = jax.checkpoint(lambda R, u_k: (_step(arrs, u_k, R, dt, s, m), None))
    R, _ = jax.lax.scan(step, R0, u)
    return R


def fidelity(arrs, u, E, T, w, dt, s, m):
    R = propagate(arrs, u, E, dt, s, m)
    t = jnp.einsum("bij,bij->b", T.conj(), R)
    return jnp.sum(w * t.real) / 4.0


def fidelity_and_forbidden(arrs, u, E, T, w, dt, s, m, mask, idx):
    """(F, C) with C = sum_{k=1}^{N} sum_{b in idx} Tr[P_F rho_b(k)], P_F = diag(mask)."""
    idx = jnp.asarray(idx)

    def body(R, u_k):
        R = _step(arrs, u_k, R, dt, s, m)
        diag = jnp.real(jnp.diagonal(R[idx], axis1=-2, axis2=-1))
        return R, jnp.sum(diag * mask)

    R, pops = jax.lax.scan(jax.checkpoint(body), E, u)
    t = jnp.einsum("bij,bij->b", T.conj(), R)
    return jnp.sum(w * t.real) / 4.0, jnp.sum(pops)


def make_fidelity_fns(model, E, T, w, dt, forbid_mask=None, forbid_idx=(0, 1)):
    """(fid(u, s, m), fid_and_grad(u, s, m)), jitted, (s, m) static.

    With `forbid_mask`, fid_and_grad instead returns (F, dF/du, C, dC/du), the forbidden-state
    cost of lindblad_np.fidelity_and_grad, from one forward scan and one vjp per output."""
    arrs = model_arrays(model)
    E, T, w = jnp.asarray(E), jnp.asarray(T), jnp.asarray(w)

    @partial(jax.jit, static_argnums=(1, 2))
    def fid(u, s, m):
        return fidelity(arrs, u, E, T, w, dt, s, m)

    if forbid_mask is None:
        @partial(jax.jit, static_argnums=(1, 2))
        def fid_and_grad(u, s, m):
            return jax.value_and_grad(lambda v: fidelity(arrs, v, E, T, w, dt, s, m))(u)
    else:
        mask = jnp.asarray(np.asarray(forbid_mask, dtype=float))
        idx = tuple(forbid_idx)

        @partial(jax.jit, static_argnums=(1, 2))
        def fid_and_grad(u, s, m):
            (F, C), vjp = jax.vjp(
                lambda v: fidelity_and_forbidden(arrs, v, E, T, w, dt, s, m, mask, idx), u)
            gF, = vjp((jnp.ones_like(F), jnp.zeros_like(C)))
            gC, = vjp((jnp.zeros_like(F), jnp.ones_like(C)))
            return F, gF, C, gC

    return fid, fid_and_grad


# ---------------------------------------------------------------------------
# Constraint chain and penalties, mirrored from core/ in jnp
# ---------------------------------------------------------------------------

def make_constrainer(N, dt, cav_band=None, tra_band=None, ramp_ns=None):
    """jnp mirror of core.ramp.make_constraint_chain's to_physical: band-limit (separate
    cavity / transmon bands, core.fourier_cutoff.make_band_mask) then ramp. Identity when
    both are disabled. validation/test_dgrape.py checks it equals the numpy chain."""
    if (cav_band is None) != (tra_band is None):
        raise ValueError("cav_band and tra_band must both be given or both be None")
    mC = mT = env = None
    if cav_band is not None:
        mC = jnp.asarray(make_band_mask(N, dt, *cav_band).astype(np.float64))
        mT = jnp.asarray(make_band_mask(N, dt, *tra_band).astype(np.float64))
    if ramp_ns:
        env = jnp.asarray(ramp_envelope(N, dt, ramp_ns))

    def constrain(x):
        u = x
        if mC is not None:
            zC = jnp.fft.ifft(jnp.fft.fft(u[:, 0] + 1j * u[:, 1]) * mC)
            zT = jnp.fft.ifft(jnp.fft.fft(u[:, 2] + 1j * u[:, 3]) * mT)
            u = jnp.stack([zC.real, zC.imag, zT.real, zT.imag], axis=1)
        if env is not None:
            u = u * env[:, None]
        return u

    return constrain


def amplitude_penalty_modulus(u, amp_max):
    """jnp copy of core.grape_core.amplitude_penalty_modulus (Heeres Supp. Eq. 19)."""
    g = 0.0
    for i, q in ((0, 1), (2, 3)):
        mag = jnp.sqrt(u[:, i] ** 2 + u[:, q] ** 2 + 1e-30)
        g = g + jnp.sum(jnp.maximum(mag - amp_max, 0.0) ** 2)
    return g


def derivative_penalty(u):
    """jnp copy of core.grape_core.derivative_penalty."""
    return jnp.sum((u[1:] - u[:-1]) ** 2)
