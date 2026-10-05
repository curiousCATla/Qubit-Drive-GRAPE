"""
Open-system GRAPE model on the Shirol chip -- PRX 16, 021042 (2026), Table I, via
AQEC/device.py. Storage (Fock 0..n_c-1) x transmon (g, e, f) x readout (m = 0..n_r-1),
storage outermost: |n, j, m> sits at index (n*3 + j)*n_r + m, the layout of
AQEC/master_equation.py (which fixes n_r = 2; here n_r is a parameter so the readout
embedding can be checked against n_r = 1).

FRAME. Rotate at the LINEAR mode frequencies omega_a, omega_q, omega_r only. This is
frame (ii) of aqec_experiments.ipynb §8.2 except that the anharmonicity stays in the
Hamiltonian, because the transmon drive is referenced to omega_q (so |f> sits at -alpha):

    H_d = -alpha |f><f| - chi_ge |e><e| n_a - chi_gf |f><f| n_a - chi_qr |e><e| n_r
          [ + (-(K/2) a'^2 a^2 + (chi'_q/2) q'q a'^2 a^2)   if include_higher_order ]

On three levels -(alpha/2) q'^2 q^2 = -alpha |f><f| exactly. Every jump operator below
only picks up a global phase in this frame, so the Lindbladian is exact as written.

SIGNS follow AQEC/device.py (Shirol): alpha, chi_* stored POSITIVE, explicit minus signs.

CONTROLS. Four real quadratures, repo convention (core/grape_core.make_ops):
    Hc = [a + a', i(a - a'), q + q', i(q - q')]
so u0 Hc0 + u1 Hc1 = eps a' + eps* a with eps = u0 - i u1, |eps| = sqrt(u0^2 + u1^2).
The readout is never driven.

DISSIPATORS. Default = decay only, AQEC Table 5's 'yes' rows:
    sqrt(kappa_a) a,  sqrt(kappa_r) r,  sqrt(1/T1_ge) |g><e|,  sqrt(1/T1_ef) |e><f|
Opt-in: pure dephasing (transmon from T2*_ge and T2*_gf, storage from T2a) and thermal
heating (transmon g->e, storage a'), rates from Table I.
"""

import numpy as np

from AQEC import device as dev

N_Q = 3
G, E, F = 0, 1, 2
N_CTRL = 4

# Apple's Accelerate BLAS raises spurious divide/overflow RuntimeWarnings on ordinary
# complex matmuls (see core/grape_core.py's np.seterr); scoped here, not global.
_QUIET = dict(divide="ignore", over="ignore", invalid="ignore")


def dim(n_c, n_r=2):
    return n_c * N_Q * n_r


def index(n, j, m, n_c, n_r=2):
    if not (0 <= n < n_c and 0 <= j < N_Q and 0 <= m < n_r):
        raise IndexError(f"|{n},{j},{m}> outside n_c={n_c}, n_r={n_r}")
    return (n * N_Q + j) * n_r + m


def _embed(op_a, op_q, op_r):
    return np.kron(np.kron(op_a, op_q), op_r)


def _lower(n):
    return np.diag(np.sqrt(np.arange(1, n)), 1).astype(complex)


def _unit(n, i, j):
    M = np.zeros((n, n), dtype=complex)
    M[i, j] = 1.0
    return M


def operators(n_c, n_r=2):
    """a, q, r, number operators and transmon projectors on the full space."""
    Ia, Iq, Ir = np.eye(n_c), np.eye(N_Q), np.eye(n_r)
    ops = {
        "a": _embed(_lower(n_c), Iq, Ir),
        "q": _embed(Ia, _lower(N_Q), Ir),
        "r": _embed(Ia, Iq, _lower(n_r)),
        "Pg": _embed(Ia, _unit(N_Q, G, G), Ir),
        "Pe": _embed(Ia, _unit(N_Q, E, E), Ir),
        "Pf": _embed(Ia, _unit(N_Q, F, F), Ir),
        "s_ge": _embed(Ia, _unit(N_Q, G, E), Ir),
        "s_ef": _embed(Ia, _unit(N_Q, E, F), Ir),
    }
    with np.errstate(**_QUIET):
        ops["n_a"] = ops["a"].conj().T @ ops["a"]
        ops["n_r"] = ops["r"].conj().T @ ops["r"]
    return ops


def drift_hamiltonian(n_c, n_r=2, include_higher_order=False, alpha=dev.alpha_q,
                      chi_ge=dev.chi_ge, chi_gf=dev.chi_gf, chi_qr=dev.chi_qr,
                      K=dev.K_a, chi_q_p=dev.chi_q_p):
    """H_d (diagonal), rad/us. See the module docstring for the frame."""
    diag = np.zeros(dim(n_c, n_r))
    chi_j = (0.0, chi_ge, chi_gf)
    for n in range(n_c):
        for j in range(N_Q):
            for m in range(n_r):
                h = -n * chi_j[j] - m * chi_qr * (j == E) - alpha * (j == F)
                if include_higher_order:
                    h += (-0.5 * K + 0.5 * chi_q_p * j) * n * (n - 1)
                diag[index(n, j, m, n_c, n_r)] = h
    return np.diag(diag).astype(complex)


def control_hamiltonians(n_c, n_r=2):
    """(4, d, d): [a + a', i(a - a'), q + q', i(q - q')]."""
    ops = operators(n_c, n_r)
    a, q = ops["a"], ops["q"]
    ad, qd = a.conj().T, q.conj().T
    return np.stack([a + ad, 1j * (a - ad), q + qd, 1j * (q - qd)])


# ---------------------------------------------------------------------------
# Rates (1/us) derived from Table I
# ---------------------------------------------------------------------------

def rates():
    """Every rate a jump operator can use, in 1/us, with how it is derived."""
    g_ge, g_ef = 1.0 / dev.T1_GE, 1.0 / dev.T1_EF
    k_a = dev.kappa_a
    # Pure dephasing: 1/T2 = 1/(2 T1_relevant) + Gamma_phi.
    #   g-e coherence: decays at g_ge/2 from T1 -> Gamma_phi_ge = 1/T2*_ge - g_ge/2
    #   g-f coherence: |f> leaves at g_ef     -> Gamma_phi_gf = 1/T2*_gf - g_ef/2
    #   storage:       |1> leaves at kappa_a  -> Gamma_phi_a  = 1/T2a    - kappa_a/2
    phi_ge = 1.0 / dev.T2_RAMSEY - 0.5 * g_ge
    phi_gf = 1.0 / dev.T2_GF_RAMSEY - 0.5 * g_ef
    phi_a = 1.0 / dev.T2_CAVITY - 0.5 * k_a
    # Heating: detailed balance at the measured equilibrium population, up = down * P/(1-P).
    up_q = g_ge * dev.P_E_THERMAL / (1.0 - dev.P_E_THERMAL)
    up_a = k_a * dev.P_CAVITY_THERMAL / (1.0 - dev.P_CAVITY_THERMAL)
    return {
        "kappa_a": k_a, "kappa_r": dev.kappa_r, "gamma_ge": g_ge, "gamma_ef": g_ef,
        "phi_ge": phi_ge, "phi_gf": phi_gf, "phi_a": phi_a,
        "up_q": up_q, "up_a": up_a,
    }


def collapse_ops(n_c, n_r=2, decay=True, dephasing=False, heating=False, scale=1.0):
    """List of (name, L_k) jump operators, rates included.

    decay      -- sqrt(kappa_a) a, sqrt(kappa_r) r, sqrt(gamma_ge)|g><e|, sqrt(gamma_ef)|e><f|
                  (identical to AQEC.master_equation.collapse_operators(n_c, 'split') at n_r=2)
    dephasing  -- sqrt(2 phi_ge)|e><e|, sqrt(2 phi_gf)|f><f|, sqrt(2 phi_a) n_a.
                  D[sqrt(G)|j><j|] damps the (g, j) coherence at G/2, so G = 2 phi gives
                  phi exactly; the e-f coherence then dephases at phi_ge + phi_gf.
                  D[sqrt(2 phi_a) n_a] damps |n><m| at phi_a (n - m)^2.
    heating    -- sqrt(up_q)|e><g|, sqrt(up_a) a'
    scale      -- multiplies every RATE (scale=0 gives the closed system, for tests)
    """
    ops = operators(n_c, n_r)
    R = rates()
    out = []
    if decay:
        out += [("a", np.sqrt(R["kappa_a"]) * ops["a"]),
                ("r", np.sqrt(R["kappa_r"]) * ops["r"]),
                ("s_ge", np.sqrt(R["gamma_ge"]) * ops["s_ge"]),
                ("s_ef", np.sqrt(R["gamma_ef"]) * ops["s_ef"])]
    if dephasing:
        out += [("phi_e", np.sqrt(2 * R["phi_ge"]) * ops["Pe"]),
                ("phi_f", np.sqrt(2 * R["phi_gf"]) * ops["Pf"]),
                ("phi_a", np.sqrt(2 * R["phi_a"]) * ops["n_a"])]
    if heating:
        out += [("up_q", np.sqrt(R["up_q"]) * ops["s_ge"].conj().T),
                ("up_a", np.sqrt(R["up_a"]) * ops["a"].conj().T)]
    out = [(name, np.sqrt(scale) * c) for name, c in out]
    return [(name, c) for name, c in out if np.any(c)]


# ---------------------------------------------------------------------------
# Monomial form of the jump operators
# ---------------------------------------------------------------------------

def monomial(c):
    """(src, w) with c = sum_i w_i |i><src_i|, i.e. (c rho c')_{ij} = w_i w_j* rho[src_i, src_j].

    Every jump operator here (ladder operators, transition operators, diagonal
    dephasing) has at most one nonzero per row and per column, so the sandwich
    c rho c' is a gather + elementwise product, O(d^2) instead of two O(d^3) matmuls.
    Rows without an entry get w_i = 0 (src_i = i, arbitrary). Raises otherwise.
    """
    c = np.asarray(c)
    d = c.shape[0]
    nz = c != 0
    if np.any(nz.sum(axis=1) > 1) or np.any(nz.sum(axis=0) > 1):
        raise ValueError("jump operator is not monomial (>1 nonzero in a row or column)")
    src = np.arange(d)
    w = np.zeros(d, dtype=complex)
    rows, cols = np.nonzero(nz)
    src[rows] = cols
    w[rows] = c[rows, cols]
    return src, w


def _flat_sandwich(srcs, Ws):
    """All K sandwiches sum_k W_k * R[src_k][:, src_k] as one flat gather + segment sum.

    Keeps only the nonzero entries of each W_k. Returns (src_idx, w, dst_unique, starts):
        v = R_flat[..., src_idx] * w        (entries sorted by destination)
        out_flat[..., dst_unique] = add.reduceat(v, starts)
    with R_flat = R.reshape(..., d*d).
    """
    d = Ws.shape[-1]
    src_all, dst_all, w_all = [], [], []
    for src, W in zip(srcs, Ws):
        i, j = np.nonzero(W)
        dst_all.append(i * d + j)
        src_all.append(src[i] * d + src[j])
        w_all.append(W[i, j])
    if not src_all:
        z = np.zeros(0, dtype=np.int64)
        return z, np.zeros(0, dtype=complex), z, z
    dst = np.concatenate(dst_all)
    order = np.argsort(dst, kind="stable")
    dst, src, w = dst[order], np.concatenate(src_all)[order], np.concatenate(w_all)[order]
    dst_unique, starts = np.unique(dst, return_index=True)
    return src, w, dst_unique, starts


class Model:
    """Everything the propagators need for one truncation, as plain arrays.

    H_d    (d, d)     drift, diagonal
    Hc     (4, d, d)  control Hamiltonians
    c_ops  list of (name, (d, d)) jump operators
    Gdiag  (d,)       diagonal of sum_k L_k' L_k (diagonal for monomial L_k)
    jump_src, jump_W    (K, d) int / (K, d, d) complex: per jump, the gather index and the
           outer-product weight w w^*, so  sum_k L_k rho L_k' = sum_k W_k * rho[src_k][:, src_k]
    jump_src_adj, jump_W_adj   the same for L_k' (for the adjoint map, sum_k L_k' X L_k)
    shift  real number subtracted from H_d's diagonal. The commutator -i[H, rho] is
           unchanged by H -> H - shift*1; centring the spectrum keeps the Taylor
           terms small (it changes nothing mathematically).
    """

    def __init__(self, n_c, n_r=2, include_higher_order=False, decay=True,
                 dephasing=False, heating=False, rate_scale=1.0):
        self.n_c, self.n_r = n_c, n_r
        self.d = dim(n_c, n_r)
        self.H_d = drift_hamiltonian(n_c, n_r, include_higher_order)
        self.Hc = control_hamiltonians(n_c, n_r)
        self.c_ops = collapse_ops(n_c, n_r, decay, dephasing, heating, rate_scale)
        hd = np.real(np.diag(self.H_d))
        self.shift = 0.5 * (hd.max() + hd.min())
        self.H_d_shifted = self.H_d - self.shift * np.eye(self.d)
        with np.errstate(**_QUIET):
            Gm = sum((c.conj().T @ c for _, c in self.c_ops), np.zeros_like(self.H_d))
        if not np.allclose(Gm, np.diag(np.diag(Gm))):
            raise ValueError("sum L'L is not diagonal")
        self.Gdiag = np.real(np.diag(Gm)).copy()
        K = len(self.c_ops)
        self.jump_src = np.zeros((K, self.d), dtype=np.int64)
        self.jump_W = np.zeros((K, self.d, self.d), dtype=complex)
        self.jump_src_adj = np.zeros((K, self.d), dtype=np.int64)
        self.jump_W_adj = np.zeros((K, self.d, self.d), dtype=complex)
        for k, (_, c) in enumerate(self.c_ops):
            src, w = monomial(c)
            self.jump_src[k] = src
            self.jump_W[k] = np.outer(w, w.conj())
            # c' X c is the same sandwich with c' in place of c, and c' is monomial too
            src, w = monomial(c.conj().T)
            self.jump_src_adj[k] = src
            self.jump_W_adj[k] = np.outer(w, w.conj())
        self.jump_flat = _flat_sandwich(self.jump_src, self.jump_W)
        self.jump_flat_adj = _flat_sandwich(self.jump_src_adj, self.jump_W_adj)
        # Norms for the Taylor truncation bound (spectral norms of the drive operators
        # per unit |eps|: ||eps a' + eps* a|| <= 2|eps| ||a||).
        ops = operators(n_c, n_r)
        self.norm_a = np.linalg.norm(ops["a"], 2)
        self.norm_q = np.linalg.norm(ops["q"], 2)
        self.half_width_d = 0.5 * (hd.max() - hd.min())
        self.diss_norm = 2.0 * sum(np.linalg.norm(c, 2) ** 2 for _, c in self.c_ops)

    def generator_norm_bound(self, eps_c_max, eps_t_max):
        """Upper bound on ||L|| (induced by the Frobenius norm) for any step whose drive
        moduli are <= eps_c_max, eps_t_max:
            ||-i[H - s1, .]|| <= 2 ||H - s1|| <= 2 (half-width(H_d) + 2 eps_c ||a|| + 2 eps_t ||q||)
            ||dissipator||    <= 2 sum_k ||L_k||^2
        """
        h = self.half_width_d + 2 * eps_c_max * self.norm_a + 2 * eps_t_max * self.norm_q
        return 2.0 * h + self.diss_norm


def drive_moduli(u):
    """Max |eps_C| and |eps_T| over the pulse, (N, 4) -> (float, float)."""
    u = np.asarray(u, dtype=float)
    return (float(np.max(np.hypot(u[:, 0], u[:, 1]), initial=0.0)),
            float(np.max(np.hypot(u[:, 2], u[:, 3]), initial=0.0)))
