"""
Exact open-system propagation and its exact hand-derived gradient (numpy reference).

MASTER EQUATION (GKSL), piecewise-constant controls u_k on [k dt, (k+1) dt):

    d rho/dt = L_k rho,   L_k rho = -i[H_k, rho] + sum_c ( c rho c' - {c'c, rho}/2 ),
    H_k = H_d + sum_j u_kj Hc_j.

Within a step L_k is constant, so rho_{k+1} = exp(L_k dt) rho_k is the exact solution.
Written with H_eff = H_k - (i/2) sum c'c:

    L_k rho = -i (H_eff rho - rho H_eff') + sum_c c rho c'
    L_k' X  = +i (H_eff' X - X H_eff)  + sum_c c' X c       (Hilbert-Schmidt adjoint)

THE STEP MAP. exp(L dt) is never formed: the d^2 x d^2 superoperator costs O(d^6) to
exponentiate. Its ACTION is evaluated at the d x d matrix level by a truncated Taylor
series with s substeps of tau = dt/s and m terms each,

    T_{s,m}(rho) = [ sum_{k=0}^{m} (tau L)^k / k! ]^s rho,

each L application costing two O(d^3) matmuls plus O(d^2) gathers for the (monomial)
jump operators. (s, m) is chosen from an a-priori bound on ||L|| so the truncation error
is below `tol` per substep (`taylor_spec`). L is non-normal, so the eigh propagator of
core/grape_core.py does not apply.

THE GRADIENT (exact). With F = sum_b w_b Re Tr[T_b' rho_b(N)]/4,

    dF/du_kj = Re sum_b Tr[ Lambda_b(k+1)' (d rho_b(k+1)/du_kj) ],
    Lambda(N) = w T / 4,   Lambda(k) = T_{s,m}'(Lambda(k+1))  (adjoint of the SAME polynomial).

d rho(k+1)/du_kj is the derivative of the step polynomial in direction L_j = -i[Hc_j, .].
Two exact constructions are implemented; they agree to round-off (validation/test_dgrape.py).

(1) `fidelity_and_grad` (default): the W-matrix form. With c_k = tau^k/k!, forward terms
a_q = L^q x and backward terms b_p = (L')^p Lambda of one substep,

    <Lambda, dP/du_j x> = sum_{k=1}^{m} c_k sum_{p+q=k-1} <b_p, L_j a_q>
                        = sum_q <Bt_q, L_j a_q>,     Bt_q = sum_{p=0}^{m-1-q} c_{p+q+1} b_p,

and because L_j a = -i[Hc_j, a], <B, L_j a> = Re Tr[Hc_j (-i)(a B' - B' a)], so

    dF/du_kj = Re Tr[Hc_j W_k],   W_k = -i sum_{substeps} sum_b sum_q (a_q Bt_q' - Bt_q' a_q).

One W per step serves all four controls. The a_q are the step's own forward terms and the
b_p are the costate's Taylor terms (their weighted sum IS the next costate), so the
gradient costs about three forward passes plus two GEMMs per substep.

(2) `fidelity_and_grad_augmented`: the augmented block generator acting on (x, y_1..y_4),

    x' = L x,       y_j' = L y_j + L_j x,

run through the SAME Taylor polynomial from (rho_k, 0): the y-block of the k-th Taylor
term of the augmented generator is sum_p L^p L_j L^{k-1-p} rho, which is exactly the
derivative of (tau L)^k rho / k!. Carrying (x, y) across the s substeps reproduces the
chain rule through the product. The result is therefore the exact derivative of the
propagator that the cost uses -- not the first-order dt <Lambda, L_j rho> of approximate
Open-GRAPE -- and it agrees with JAX autodiff of the same polynomial to round-off.
"""

import math

import numpy as np

from DGRAPE.model import _QUIET, N_Q, drive_moduli

TAYLOR_TOL = 1e-15
M_MAX = 60
FORBID_IDX = (0, 1)    # E_00, E_11: the two code basis states, real density matrices


def taylor_spec(theta, tol=TAYLOR_TOL, m_max=M_MAX):
    """(s, m) minimizing s*m subject to the substep remainder bound
    theta_s^{m+1}/(m+1)! * e^{theta_s} <= tol, theta_s = theta/s."""
    best = None
    for s in range(1, 10_000):
        th = theta / s
        if th == 0.0:
            return (1, 1)
        for m in range(1, m_max + 1):
            log_rem = (m + 1) * math.log(th) - math.lgamma(m + 2) + th
            if log_rem <= math.log(tol):
                if best is None or s * m < best[0] * best[1]:
                    best = (s, m)
                break
        if best is not None and s > best[0] * best[1]:
            break
    return best


def spec_for_pulse(model, u, dt, tol=TAYLOR_TOL):
    """Taylor (s, m) valid for every step of pulse u (N, 4)."""
    ec, et = drive_moduli(u)
    return taylor_spec(model.generator_norm_bound(ec, et) * dt, tol)


# ---------------------------------------------------------------------------
# Generator actions on stacks (..., d, d)
# ---------------------------------------------------------------------------

def _sandwich(flat, R):
    """sum_k W_k * R[src_k][:, src_k] via the flat gather of model._flat_sandwich."""
    src, w, dst, starts = flat
    out = np.zeros_like(R)
    if src.size:
        lead = R.shape[:-2]
        n = R.shape[-1] * R.shape[-1]
        v = np.take(R.reshape(lead + (n,)), src, axis=-1) * w
        out.reshape(lead + (n,))[..., dst] = np.add.reduceat(v, starts, axis=-1)
    return out


def _jumps(model, R):
    """sum_c c R c'."""
    return _sandwich(model.jump_flat, R)


def _jumps_adj(model, X):
    """sum_c c' X c, the same gather with the monomial form of c'."""
    return _sandwich(model.jump_flat_adj, X)


def h_eff(model, u_k):
    """Shifted H_eff for one step: H_d - shift + sum u_j Hc_j - (i/2) diag(G)."""
    H = model.H_d_shifted + np.tensordot(u_k, model.Hc, axes=1)
    return H - 0.5j * np.diag(model.Gdiag)


def apply_L(model, Heff, R):
    with np.errstate(**_QUIET):
        return -1j * (Heff @ R - R @ Heff.conj().T) + _jumps(model, R)


def apply_Ldag(model, Heff, X):
    with np.errstate(**_QUIET):
        return 1j * (Heff.conj().T @ X - X @ Heff) + _jumps_adj(model, X)


def taylor_action(apply, R, tau, s, m):
    """[sum_{k<=m} (tau A)^k/k!]^s R for a linear map `apply`."""
    for _ in range(s):
        term = R
        acc = R.copy()
        for k in range(1, m + 1):
            term = apply(term) * (tau / k)
            acc += term
        R = acc
    return R


def step(model, u_k, R, dt, spec):
    s, m = spec
    Heff = h_eff(model, u_k)
    return taylor_action(lambda X: apply_L(model, Heff, X), R, dt / s, s, m)


def step_adjoint(model, u_k, X, dt, spec):
    s, m = spec
    Heff = h_eff(model, u_k)
    return taylor_action(lambda Y: apply_Ldag(model, Heff, Y), X, dt / s, s, m)


def step_with_derivative(model, u_k, R, dt, spec):
    """(rho_{k+1}, D) with D[j] = d rho_{k+1} / d u_kj, shape (4, *R.shape).

    Augmented generator on Z = (x, y_1..y_4) stacked along a leading axis of size 5:
        x' = L x,  y_j' = L y_j - i[Hc_j, x].
    """
    s, m = spec
    tau = dt / s
    Heff = h_eff(model, u_k)
    nc = model.Hc.shape[0]
    Hc = model.Hc.reshape((nc,) + (1,) * (R.ndim - 2) + model.Hc.shape[1:])

    def apply_aug(Z):
        out = apply_L(model, Heff, Z)
        x = Z[0]
        with np.errstate(**_QUIET):
            out[1:] += -1j * (Hc @ x - x @ Hc)
        return out

    Z = np.zeros((nc + 1,) + R.shape, dtype=complex)
    Z[0] = R
    Z = taylor_action(apply_aug, Z, tau, s, m)
    return Z[0], Z[1:]


# ---------------------------------------------------------------------------
# Whole pulse
# ---------------------------------------------------------------------------

def propagate(model, u, R0, dt, spec=None, store=False):
    """Propagate the stack R0 (..., d, d) through pulse u (N, 4).

    Returns rho(N), or (rho(N), [rho(0), ..., rho(N-1)]) if store.
    """
    u = np.asarray(u, dtype=float)
    spec = spec_for_pulse(model, u, dt) if spec is None else spec
    R = np.asarray(R0, dtype=complex)
    hist = [] if store else None
    for k in range(u.shape[0]):
        if store:
            hist.append(R)
        R = step(model, u[k], R, dt, spec)
    return (R, hist) if store else R


def fidelity(model, u, E, T, w, dt, spec=None):
    """F_pro = sum_b w_b Re Tr[T_b' Phi(E_b)] / 4."""
    R = propagate(model, u, E, dt, spec)
    t = np.einsum("bij,bij->b", T.conj(), R)
    return float(np.sum(w * t.real) / 4.0)


def _taylor_terms(apply, R, tau, m):
    """[R, (A)R, (A)^2 R, ..., (A)^m R] stacked, (m+1, *R.shape); A = apply."""
    terms = np.empty((m + 1,) + R.shape, dtype=complex)
    terms[0] = R
    for q in range(1, m + 1):
        terms[q] = apply(terms[q - 1])
    return terms


def forbidden_mask(model, fock_min):
    """Diagonal of P_F = sum_{n_a >= fock_min} |n_a><n_a| (x) 1_q (x) 1_r, as a bool (d,).
    Storage is outermost (DGRAPE/model.py), so the Fock number of index i is i // (3 n_r)."""
    return (np.arange(model.d) // (N_Q * model.n_r)) >= fock_min


def _forbidden_sum(states, mask, idx):
    """sum_{rho in states} sum_{b in idx} Tr[P_F rho_b] for (B, d, d) stacks."""
    diag = np.stack([np.real(np.diagonal(R[list(idx)], axis1=-2, axis2=-1)) for R in states])
    return float(diag[..., mask].sum())


def forbidden_population(model, u, E, dt, spec=None, mask=None, idx=FORBID_IDX):
    """(C, per_step): C = sum_{k=1}^{N} sum_{b in idx} Tr[P_F rho_b(k)] (Heeres' forbidden-state
    cost) and the forbidden population of each step, (N, len(idx)). No gradient."""
    u = np.asarray(u, dtype=float)
    spec = spec_for_pulse(model, u, dt) if spec is None else spec
    R_N, hist = propagate(model, u, E, dt, spec, store=True)
    states = hist[1:] + [R_N]
    per_step = np.array([[np.real(np.diagonal(R[b]))[mask].sum() for b in idx] for R in states])
    return float(per_step.sum()), per_step


def _w_term(a, Bt, d):
    """-i sum_{q,b} (a_qb Bt_qb' - Bt_qb' a_qb) as two single GEMMs."""
    A_wide = np.transpose(a, (2, 0, 1, 3)).reshape(d, -1)          # [a_qb] side by side
    Bd_tall = np.transpose(Bt.conj(), (0, 1, 3, 2)).reshape(-1, d)  # [Bt_qb'] stacked
    Bd_wide = np.transpose(Bt.conj(), (3, 0, 1, 2)).reshape(d, -1)  # [Bt_qb'] side by side
    a_tall = a.reshape(-1, d)                                       # [a_qb] stacked
    return -1j * (A_wide @ Bd_tall - Bd_wide @ a_tall)


def fidelity_and_grad(model, u, E, T, w, dt, spec=None, forbid_mask=None, forbid_idx=FORBID_IDX):
    """(F, dF/du) with dF/du of shape (N, 4), exact for the Taylor propagator (W form).

    With `forbid_mask` (the bool diagonal of P_F, `forbidden_mask`), also the forbidden-state
    cost C = sum_{k=1}^{N} sum_{b in forbid_idx} Tr[P_F rho_b(k)] and its exact gradient, as
    (F, dF/du, C, dC/du). Both share ONE backward pass: C is linear in every rho_b(k), so its
    costate Lambda_C obeys the same adjoint recursion plus a source P_F at every step,

        Lambda_C(N) = P_F,   Lambda_C(k) = T'(Lambda_C(k+1)) + P_F   (k >= 1),

    and is carried as extra entries of the costate stack, paired with the forward terms of
    the inputs in forbid_idx. dF/du stays separate, as Eq. 24 needs it.
    """
    u = np.asarray(u, dtype=float)
    spec = spec_for_pulse(model, u, dt) if spec is None else spec
    s, m = spec
    tau = dt / s
    c = np.array([tau ** k / math.factorial(k) for k in range(m + 1)])
    # Bt_q = sum_p C[q, p] b_p with C[q, p] = c_{p+q+1} for p + q <= m - 1
    C = np.zeros((m, m))
    for q in range(m):
        C[q, : m - q] = c[q + 1: m + 1]

    R_N, hist = propagate(model, u, E, dt, spec, store=True)
    t = np.einsum("bij,bij->b", T.conj(), R_N)
    F = float(np.sum(w * t.real) / 4.0)

    d = model.d
    nb = T.shape[0]
    HcT = np.transpose(model.Hc, (0, 2, 1))      # Re Tr[Hc W] = Re sum(Hc^T * W)
    Lam = (w[:, None, None] / 4.0) * T
    forbid = forbid_mask is not None
    if forbid:
        idx = list(forbid_idx)
        P = np.diag(np.asarray(forbid_mask, dtype=float)).astype(complex)
        Cval = _forbidden_sum(hist[1:] + [R_N], forbid_mask, idx)
        Lam = np.concatenate([Lam, np.broadcast_to(P, (len(idx), d, d))])
        grad_C = np.empty_like(u)
    grad = np.empty_like(u)
    with np.errstate(**_QUIET):
        for k in range(u.shape[0] - 1, -1, -1):
            Heff = h_eff(model, u[k])
            fwd = lambda X: apply_L(model, Heff, X)        # noqa: E731
            bwd = lambda X: apply_Ldag(model, Heff, X)     # noqa: E731
            xs = [hist[k]]                                 # substep inputs
            for _ in range(s - 1):
                xs.append(taylor_action(fwd, xs[-1], tau, 1, m))
            W = np.zeros((d, d), dtype=complex)
            if forbid:
                W_C = np.zeros((d, d), dtype=complex)
            for r in range(s - 1, -1, -1):
                a = _taylor_terms(fwd, xs[r], 1.0, m - 1)      # L^q x, q < m
                b = _taylor_terms(bwd, Lam, 1.0, m)            # (L')^p Lambda, p <= m
                Lam = np.tensordot(c, b, axes=1)               # costate before this substep
                Bt = np.tensordot(C, b[:m], axes=1)            # (m, B, d, d)
                W += _w_term(a, Bt[:, :nb], d)
                if forbid:
                    W_C += _w_term(a[:, idx], Bt[:, nb:], d)
            grad[k] = (HcT.reshape(HcT.shape[0], -1) @ W.ravel()).real
            if forbid:
                grad_C[k] = (HcT.reshape(HcT.shape[0], -1) @ W_C.ravel()).real
                if k >= 1:
                    Lam[nb:] += P                              # source for rho(k)
    if forbid:
        return F, grad, Cval, grad_C
    return F, grad


def fidelity_and_grad_augmented(model, u, E, T, w, dt, spec=None):
    """(F, dF/du), exact for the Taylor propagator (augmented-generator form)."""
    u = np.asarray(u, dtype=float)
    spec = spec_for_pulse(model, u, dt) if spec is None else spec
    R_N, hist = propagate(model, u, E, dt, spec, store=True)
    t = np.einsum("bij,bij->b", T.conj(), R_N)
    F = float(np.sum(w * t.real) / 4.0)

    Lam = (w[:, None, None] / 4.0) * T          # dF/d rho(N) in the Re Tr[Lam' rho] pairing
    grad = np.empty_like(u)
    for k in range(u.shape[0] - 1, -1, -1):
        _, D = step_with_derivative(model, u[k], hist[k], dt, spec)
        grad[k] = np.einsum("bij,cbij->c", Lam.conj(), D).real
        Lam = step_adjoint(model, u[k], Lam, dt, spec)
    return F, grad
