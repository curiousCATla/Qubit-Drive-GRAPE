"""
Four-wave-mixing (FWM) pump tones as GRAPE controls on top of DGRAPE (fwm_grape.ipynb).

Same chip, drift, frame, jump operators and Taylor step map as DGRAPE/model.py and
DGRAPE/lindblad_np.py. Added: K pump tones k = (comb c, path n), by default the six with n in
{1, 3, 5} (`paths` of fwm_terms; FULL_PATHS = 1..9 gives 18), with complex displacements
beta_k(t) = beta_k^I - i beta_k^Q, piecewise constant on the 2 ns grid, and fixed carriers Delta_k
(the bare tone frequencies in the DGRAPE frame, ~ +-134 MHz). Real tone arrays ("b12") have 2K
columns (beta^I_k, beta^Q_k); the name is kept from the six-tone version.

    xi_c(t)  = sum_{n} beta_{c,n} exp(i Delta_{c,n} t)                        comb displacement
    H_FWM    = sum_c ( g_c xi_c A_c + h.c. ),  A_1 = a' q'^2,  A_2 = q^2 r'     (Shirol App. C)
    H_S      = -( |xi_1|^2 + |xi_2|^2 ) S,      S = 2 alpha q'q + chi_ge n_a + chi_qr n_r
    H_mix    = mu M + h.c.,  mu = xi_1 xi_2,    M = -g_ar a' r'        (opt-in, see below)

PUMP MIXING. A comb-1 and a comb-2 pump of the same path add up to exactly omega_a + omega_r,
so the quartic junction term with one a', one r' and the two pump factors is resonant. Direct
counting of the cosine expansion gives the coefficient E_J phi_a phi_r phi_q^2 = sqrt(chi_qa chi_qr),
with the same overall sign as the comb terms (hence the minus once the comb signs are absorbed into
beta). `mixing` scales it: 0 = off (training default), 1 = sqrt(chi_ge chi_qr), 2 = the larger
reading if beta is sqrt(2) smaller than the true pump displacement (the Eq. C9 vs direct-count
ambiguity of fwm_grape.ipynb §2.3).

SUB-STEPS. The carriers turn by ~1.8 rad per 2 ns step, so the generator is NOT constant over a
step. Each step is cut into S sub-steps; on each one the commutator-free fourth-order Magnus scheme
(CF4, Blanes & Moan) applies two exponentials, each with a generator built from H at the two
Gauss-Legendre nodes:

    rho <- exp(dts/2 L[2a1 H(tau1) + 2a2 H(tau2)]) exp(dts/2 L[2a2 H(tau1) + 2a1 H(tau2)]) rho

(right factor first). "midpoint" is one exponential at the sub-step midpoint (second order).
Within one exponential the generator is constant, so lindblad_np's Taylor action applies as is.

THE GRADIENT (exact for this polynomial). Every exponential e is a Taylor polynomial of a
constant generator L_e, so lindblad_np's W form carries over with ONE W per exponential:
for any real parameter theta entering H_e, dF/dtheta = Re Tr[(dH_e/dtheta) W_e]. H_e is linear in
zeta_ec = sum_nu w_enu xi_c(tau_nu), in sigma_e = sum_nu w_enu sum_c |xi_c(tau_nu)|^2 and in
mu_e = sum_nu w_enu xi_1 xi_2(tau_nu), so the chain rule only needs five traces per exponential:
Tr[A_c W], Tr[A_c' W] (c = 1, 2), Tr[S W], Tr[M W], Tr[M' W] (and Tr[Hc_j W] for the linear drives,
which enter every exponential with weight 1 because the node weights sum to 1). The occupation
costs (forbidden states, transmon) ride extra costate entries exactly as in lindblad_np.
`directional_derivative` (augmented generator, the same polynomial) is the independent reference.

Pulses trained with these tones go to pulses/dgrape/, never pulses/.
"""

import math
from types import SimpleNamespace

import numpy as np
from joblib import Parallel, delayed

from AQEC import device as dev
from core.ramp import make_constraint_chain
from DGRAPE import lindblad_np as lnp
from DGRAPE.code import process_targets
from DGRAPE.model import _QUIET, F, G, Model, drive_moduli, index, operators

def tone_list(paths):
    """[(comb, path n)]: comb 1 on every path, then comb 2 on every path."""
    return [(1, n) for n in paths] + [(2, n) for n in paths]


# The default tone set: Shirol's six tones, repairs into the odd code words n = 1, 3, 5. Every
# function that takes `paths` (or reads fw.tones) defaults to it.
TONES = tone_list(dev.CORRECTED_FOCK)
N_TONES = len(TONES)
# Every repair n-1 -> n up to 8 -> 9 (fwm_grape.ipynb §7): 9 teeth per comb, 18 tones. 9 -> 10 is
# left out because Fock 10 is DGRAPE's forbidden region (forbid_fock_min).
FULL_PATHS = tuple(range(1, 10))
G1 = np.sqrt(dev.alpha_q * dev.chi_ge)          # comb-1 coupling per unit beta (Eq. C9), rad/us
G2 = np.sqrt(dev.alpha_q * dev.chi_qr)          # comb-2 coupling per unit beta
G_MIX = np.sqrt(dev.chi_ge * dev.chi_qr)        # a'r' coupling per unit beta_1 beta_2 (mixing=1)


def beta_table2(paths):
    """Shirol Table II as beta, extended to any paths (fwm_tuning §2, §10.1): comb 1 at 57 kHz on
    every path, so beta_{1,n} = 0.0033/sqrt(n); comb 2 at beta = 0.011 (192 kHz here; Shirol
    measured 160 kHz). Order as tone_list(paths)."""
    return np.array([0.0033 / np.sqrt(n) for n in paths] + [0.011] * len(paths))


BETA_TABLE2 = beta_table2(dev.CORRECTED_FOCK)

CF4_A = ((3 - 2 * np.sqrt(3)) / 12, (3 + 2 * np.sqrt(3)) / 12)     # a1, a2
GAUSS = (0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6)                 # nodes within a sub-step

# Stage 2 / 3 recipe for the tones (fwm_grape.ipynb §3, decisions of 2026-10-11)
BETA_CAP = 0.1               # per tone |beta_k| and per comb |xi_c|
BETA_BAND = (-2.0, 2.0)      # MHz on each envelope: half the 4.14 MHz tone spacing
BETA_BAND_FULL = (-1.0, 1.0)  # the same rule for FULL_PATHS, whose teeth are chi_gf = 2.07 MHz apart


def default_band(paths):
    """BETA_BAND for the six default tones, BETA_BAND_FULL for any other tone set."""
    return BETA_BAND if tuple(paths) == tuple(dev.CORRECTED_FOCK) else BETA_BAND_FULL
U_BOX = 25.0                 # the u pre-image box; x_beta = S_BETA * beta shares it
S_BETA = U_BOX / BETA_CAP
# Weights: the u recipe times S_BETA^2 (same balance in pre-image units). Stage 2 trains with
# forbid OFF: u is frozen at d2, whose forbidden occupation is fixed, and the tones only add one
# photon to states with n <= 4. Its costate would cost ~40 % per evaluation for no change; C_forbid
# is still reported at scoring.
BETA_PENALTIES = {"deriv": 1e-5 * S_BETA ** 2, "amp": 8e-5 * S_BETA ** 2,
                  "comb": 8e-5 * S_BETA ** 2, "forbid": 0.0, "transmon": 0.0}
S_TRAIN = 2      # CF4 sub-steps in training: within 4e-5 of S=8 at the cap (fwm_grape.ipynb §4.4)


# ---------------------------------------------------------------------------
# Operators and carriers
# ---------------------------------------------------------------------------

def fwm_terms(model, mixing=0.0, g_scale=1.0, paths=dev.CORRECTED_FOCK):
    """FWM operators (couplings folded in), Stark diagonal and carriers for a DGRAPE Model.

    g_scale multiplies both comb couplings (1/sqrt(2) tests the coupling convention). `paths` are
    the photon numbers n reached by each repair n-1 -> n; the tones are tone_list(paths), carried
    as fw.tones (fw.n_tones of them), and every beta array has fw.n_tones columns in that order."""
    n_c, n_r = model.n_c, model.n_r
    paths = tuple(int(n) for n in paths)
    if min(paths) < 1 or max(paths) + 1 > n_c - 1:
        raise ValueError(f"paths {paths} need 1 <= n and n + 1 <= n_c - 1 = {n_c - 1}")
    tones = tone_list(paths)
    o = operators(n_c, n_r)
    a, q, r = o["a"], o["q"], o["r"]
    ad, qd, rd = a.conj().T, q.conj().T, r.conj().T
    with np.errstate(**_QUIET):
        A1 = g_scale * G1 * (ad @ qd @ qd)                   # g1 a' q'^2
        A2 = g_scale * G2 * (q @ q @ rd)                     # g2 q^2 r'
        M = -mixing * G_MIX * (ad @ rd)                      # -g_ar a' r'
        S_diag = np.real(np.diag(2 * dev.alpha_q * qd @ q + dev.chi_ge * o["n_a"]
                                 + dev.chi_qr * o["n_r"]))
    Ed = np.real(np.diag(model.H_d))
    Delta = []
    for c, n in tones:
        if c == 1:   # |n-1,g,0> -> |n,f,0>
            Delta.append(-(Ed[index(n, F, 0, n_c, n_r)] - Ed[index(n - 1, G, 0, n_c, n_r)]))
        else:        # |n,f,0> -> |n,g,1>
            Delta.append(-(Ed[index(n, G, 1, n_c, n_r)] - Ed[index(n, F, 0, n_c, n_r)]))
    return SimpleNamespace(
        A=(A1, A2), Ad=(A1.conj().T, A2.conj().T), AT=(A1.T, A2.T), AdT=(A1.conj(), A2.conj()),
        M=M, Md=M.conj().T, MT=M.T, MdT=M.conj(), mixing=float(mixing), g_scale=float(g_scale),
        S_diag=S_diag, Delta=np.array(Delta), comb=np.array([c for c, _ in tones]),
        tones=tones, paths=paths, n_tones=len(tones),
        normA=(np.linalg.norm(A1, 2), np.linalg.norm(A2, 2)), normM=np.linalg.norm(M, 2),
        normS=np.abs(S_diag).max())


def comb_mask(c, tones=TONES):
    """(n_tones,) 1.0 on the tones of comb c, else 0.0."""
    return np.array([1.0 if cc == c else 0.0 for cc, _ in tones])


def tone_rate(k, beta, g_scale=1.0, tones=TONES):
    """Swap rate on tone k's own path (rad/us): g1 beta sqrt(2n) (comb 1), g2 beta sqrt(2) (comb 2)."""
    c, n = tones[k]
    return g_scale * (G1 * beta * np.sqrt(2 * n) if c == 1 else G2 * beta * np.sqrt(2))


def xi(fw, beta_k, t):
    """Comb displacements (xi_1(t), xi_2(t)) for one step's beta_k (n_tones,)."""
    terms = beta_k * np.exp(1j * fw.Delta * t)
    return terms[fw.comb == 1].sum(), terms[fw.comb == 2].sum()


def h_eff(model, fw, Hk, beta_k, t, stark=True):
    """Hk (shifted drift + linear drives - (i/2) sum L'L) plus FWM, Stark, mixing at time t."""
    x1, x2 = xi(fw, beta_k, t)
    V = x1 * fw.A[0] + x2 * fw.A[1]
    H = Hk + V + V.conj().T
    if stark:
        H = H - np.diag((abs(x1) ** 2 + abs(x2) ** 2) * fw.S_diag)
    if fw.mixing:
        H = H + x1 * x2 * fw.M + np.conj(x1 * x2) * fw.Md
    return H


def readout_traced(T, n_c, n_r=2):
    """T_b (x) 1_r: the readout-0 block of each target copied onto every readout level."""
    T00 = T.reshape(T.shape[0], n_c * 3, n_r, n_c * 3, n_r)[:, :, 0, :, 0]
    return np.einsum("bpq,mn->bpmqn", T00, np.eye(n_r)).reshape(T.shape)


def beta_from_real(b12):
    """(N, 2K) real columns (beta^I_k, beta^Q_k) -> (N, K) complex beta = beta^I - i beta^Q."""
    b12 = np.asarray(b12, dtype=float)
    return b12[:, 0::2] - 1j * b12[:, 1::2]


def real_from_beta(beta):
    beta = np.asarray(beta, dtype=complex)
    out = np.empty((beta.shape[0], 2 * beta.shape[1]))
    out[:, 0::2], out[:, 1::2] = beta.real, -beta.imag
    return out


# ---------------------------------------------------------------------------
# Exponential schedule of one step
# ---------------------------------------------------------------------------

def _schedule(S, scheme):
    """(offsets, weights, n_exp): node offsets within a step in units of dt, (S*n_exp, n_nodes),
    and node weights (S*n_exp, n_nodes) of each exponential, in the order they act."""
    if scheme == "cf4":
        offs, wts = [], []
        for j in range(S):
            nodes = [(j + GAUSS[0]) / S, (j + GAUSS[1]) / S]
            offs += [nodes, nodes]
            wts += [[2 * CF4_A[1], 2 * CF4_A[0]], [2 * CF4_A[0], 2 * CF4_A[1]]]
        return np.array(offs), np.array(wts), 2
    if scheme == "midpoint":
        return np.array([[(j + 0.5) / S] for j in range(S)]), np.ones((S, 1)), 1
    raise ValueError(f"scheme must be 'cf4' or 'midpoint', not {scheme!r}")


def node_times(N, dt, S, scheme="cf4"):
    """All distinct node times of the pulse, (N, n_nodes_per_step) - for the comb penalty."""
    offs, _, n_exp = _schedule(S, scheme)
    per_step = offs[::n_exp].ravel()                  # each node once
    return (np.arange(N)[:, None] + per_step[None, :]) * dt


class _Step:
    """Precomputed coefficients of every exponential of one step k."""

    def __init__(self, fw, beta_k, k, dt, offs, wts, stark):
        tau = (k + offs) * dt                                     # (n_e, n_nodes)
        self.ph = np.exp(1j * fw.Delta[:, None, None] * tau[None])  # (K, n_e, n_nodes)
        terms = beta_k[:, None, None] * self.ph
        self.xin = np.stack([terms[fw.comb == 1].sum(0), terms[fw.comb == 2].sum(0)])  # (2, n_e, nn)
        self.w = wts
        self.zeta = (self.xin * wts[None]).sum(-1)                 # (2, n_e)
        self.sigma = ((np.abs(self.xin) ** 2).sum(0) * wts).sum(-1) if stark else np.zeros(len(wts))
        self.mu = (self.xin[0] * self.xin[1] * wts).sum(-1)        # (n_e,)
        self.stark = stark

    def H(self, fw, Hk, e):
        z1, z2 = self.zeta[:, e]
        H = Hk + z1 * fw.A[0] + np.conj(z1) * fw.Ad[0] + z2 * fw.A[1] + np.conj(z2) * fw.Ad[1]
        if self.stark:
            H = H - np.diag(self.sigma[e] * fw.S_diag)
        if fw.mixing:
            H = H + self.mu[e] * fw.M + np.conj(self.mu[e]) * fw.Md
        return H

    def dH(self, fw, e, dbeta):
        """Exact derivative of H_e along dbeta (K,) complex (H_e is quadratic in beta)."""
        dterms = dbeta[:, None, None] * self.ph[:, e:e + 1]
        dxi = np.stack([dterms[fw.comb == 1].sum(0), dterms[fw.comb == 2].sum(0)])[:, 0]  # (2, nn)
        xin, w = self.xin[:, e], self.w[e]
        dz = (dxi * w).sum(-1)
        out = dz[0] * fw.A[0] + np.conj(dz[0]) * fw.Ad[0] + dz[1] * fw.A[1] + np.conj(dz[1]) * fw.Ad[1]
        if self.stark:
            ds = (2 * np.real(np.conj(xin) * dxi).sum(0) * w).sum()
            out = out - np.diag(ds * fw.S_diag)
        if fw.mixing:
            dm = ((dxi[0] * xin[1] + xin[0] * dxi[1]) * w).sum()
            out = out + dm * fw.M + np.conj(dm) * fw.Md
        return out

    def grad_beta(self, fw, e, tr):
        """(K, 2) [dF/dbeta^I, dF/dbeta^Q] of exponential e from its traces
        tr = (tA1, tA2, tAd1, tAd2, tS, tM, tMd), each Tr[X W_e]."""
        tA, tAd = np.array(tr[0:2]), np.array(tr[2:4])
        tS, tM, tMd = tr[4], tr[5], tr[6]
        c = fw.comb - 1                                              # 0/1 per tone
        ph = self.ph[:, e]                                           # (K, nn)
        xin = self.xin[:, e]                                         # (2, nn)
        w = self.w[e]
        out = np.zeros((fw.n_tones, 2))
        for col, f in ((0, 1.0), (1, -1j)):
            d = f * ph                                               # d xi_c / d theta, (K, nn)
            val = (w * (d * tA[c][:, None] + np.conj(d) * tAd[c][:, None])).sum(-1).real
            if self.stark:
                dsig = (w * 2 * np.real(np.conj(xin[c]) * d)).sum(-1)
                val = val - dsig * np.real(tS)
            if fw.mixing:
                dmu = (w * d * xin[1 - c]).sum(-1)
                val = val + np.real(dmu * tM + np.conj(dmu) * tMd)
            out[:, col] = val
        return out


def norm_bound(model, fw, u, beta):
    """Upper bound on ||L|| for every exponential of the pulse (lindblad_np's bound + FWM)."""
    beta = np.asarray(beta, dtype=complex)
    xmax = np.array([np.abs(beta[:, fw.comb == c]).sum(axis=1).max(initial=0.0) for c in (1, 2)])
    ec, et = drive_moduli(u)
    return (model.generator_norm_bound(ec, et) + 4 * (xmax[0] * fw.normA[0] + xmax[1] * fw.normA[1])
            + 2 * (xmax ** 2).sum() * fw.normS + 4 * xmax[0] * xmax[1] * fw.normM)


def spec_for(model, fw, u, beta, dt, S, scheme="cf4"):
    n_exp = 2 if scheme == "cf4" else 1
    return lnp.taylor_spec(norm_bound(model, fw, u, beta) * dt / S / n_exp)


def _hk(model, u_k):
    return model.H_d_shifted + np.tensordot(u_k, model.Hc, axes=1) - 0.5j * np.diag(model.Gdiag)


# ---------------------------------------------------------------------------
# Fast generator actions (same maps as lnp.apply_L / apply_Ldag)
# ---------------------------------------------------------------------------

def _gen(model, H):
    """(fwd, bwd) actions of L and L' for H_eff, with H' precomputed once."""
    Hd = np.ascontiguousarray(H.conj().T)

    def fwd(R):
        out = H @ R
        out -= R @ Hd
        out *= -1j
        out += lnp._jumps(model, R)
        return out

    def bwd(X):
        out = Hd @ X
        out -= X @ H
        out *= 1j
        out += lnp._jumps_adj(model, X)
        return out

    return fwd, bwd


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def propagate(model, fw, u, beta, dt, S, R0, scheme="cf4", stark=True, store=False, spec=None):
    """Propagate the stack R0 through the pulse (u (N, 4), beta (N, K) complex).

    Returns (rho(N), (s, m)), or with store=True (rho(N), [rho(0), ..., rho(N-1)])."""
    u, beta = np.asarray(u, dtype=float), np.asarray(beta, dtype=complex)
    offs, wts, n_exp = _schedule(S, scheme)
    h = dt / S / n_exp                                   # length of one exponential
    s, m = spec_for(model, fw, u, beta, dt, S, scheme) if spec is None else spec
    R = np.asarray(R0, dtype=complex)
    hist = []
    for k in range(u.shape[0]):
        if store:
            hist.append(R)
        Hk = _hk(model, u[k])
        st = _Step(fw, beta[k], k, dt, offs, wts, stark)
        for e in range(len(wts)):
            He = st.H(fw, Hk, e)
            R = lnp.taylor_action(_gen(model, He)[0], R, h / s, s, m)
    return (R, hist) if store else (R, (s, m))


def fidelity(model, fw, u, beta, E, T, w, dt, S, scheme="cf4", stark=True, spec=None):
    R, _ = propagate(model, fw, u, beta, dt, S, E, scheme, stark, spec=spec)
    return float(np.sum(w * np.einsum("bij,bij->b", T.conj(), R).real) / 4.0)


def fidelity_and_grad(model, fw, u, beta, E, T, w, dt, S, scheme="cf4", stark=True, spec=None,
                      occ_masks=None, occ_weights=None, occ_idx=lnp.FORBID_IDX, want_u=False):
    """Exact (F, dF/dbeta [, dF/du]) of the sub-stepped Taylor propagator, W form.

    dF/dbeta is (N, 2K) real, columns (dF/dbeta^I_k, dF/dbeta^Q_k) per tone (beta = I - iQ).
    With occ_masks (K, d) (diagonal projectors, e.g. lnp.forbidden_mask), also returns the
    occupation sums C (K,) = sum_{k=1}^{N} sum_{b in occ_idx} Tr[P rho_b(k)] and the gradient of
    sum_k occ_weights[k] C_k, as a dict {"F", "gF", "gF_u", "C", "gC", "gC_u"} (u entries None
    unless want_u)."""
    u, beta = np.asarray(u, dtype=float), np.asarray(beta, dtype=complex)
    N = u.shape[0]
    offs, wts, n_exp = _schedule(S, scheme)
    n_e = len(wts)
    h = dt / S / n_exp
    s, m = spec_for(model, fw, u, beta, dt, S, scheme) if spec is None else spec
    tau = h / s
    c = np.array([tau ** q / math.factorial(q) for q in range(m + 1)])
    Cm = np.zeros((m, m))
    for q in range(m):
        Cm[q, : m - q] = c[q + 1: m + 1]

    R_N, hist = propagate(model, fw, u, beta, dt, S, E, scheme, stark, store=True, spec=(s, m))
    t = np.einsum("bij,bij->b", T.conj(), R_N)
    Fval = float(np.sum(w * t.real) / 4.0)

    d = model.d
    nb = T.shape[0]
    Lam = (w[:, None, None] / 4.0) * T
    occ = occ_masks is not None and len(occ_idx) > 0
    if occ:
        idx = list(occ_idx)
        masks = np.atleast_2d(np.asarray(occ_masks, dtype=float))
        wo = np.ones(len(masks)) if occ_weights is None else np.asarray(occ_weights, float)
        P = np.diag(wo @ masks).astype(complex)
        Cval = lnp._occupations(hist[1:] + [R_N], masks, idx)
        Lam = np.concatenate([Lam, np.broadcast_to(P, (len(idx), d, d))])
        gC = np.zeros((N, 2 * fw.n_tones))
        gC_u = np.zeros((N, 4)) if want_u else None
    gF = np.zeros((N, 2 * fw.n_tones))
    gF_u = np.zeros((N, 4)) if want_u else None
    HcT = np.transpose(model.Hc, (0, 2, 1)).reshape(model.Hc.shape[0], -1)
    S_d = fw.S_diag

    def traces(W):
        Wd = np.diagonal(W)
        out = [np.sum(fw.AT[0] * W), np.sum(fw.AT[1] * W), np.sum(fw.AdT[0] * W),
               np.sum(fw.AdT[1] * W), np.sum(S_d * Wd), 0.0, 0.0]
        if fw.mixing:
            out[5], out[6] = np.sum(fw.MT * W), np.sum(fw.MdT * W)
        return out

    with np.errstate(**_QUIET):
        for k in range(N - 1, -1, -1):
            Hk = _hk(model, u[k])
            st = _Step(fw, beta[k], k, dt, offs, wts, stark)
            gens = [_gen(model, st.H(fw, Hk, e)) for e in range(n_e)]
            # Recompute this step forward from rho(k), keeping the Taylor terms L^q x (q < m)
            # of every (exponential, Taylor substep): they are the a_q of the W form.
            terms = []
            X = hist[k]
            for e in range(n_e):
                for _ in range(s):
                    a = lnp._taylor_terms(gens[e][0], X, 1.0, m)
                    terms.append(a[:m])
                    X = np.tensordot(c, a, axes=1)
            for e in range(n_e - 1, -1, -1):
                bwd = gens[e][1]
                W = np.zeros((d, d), dtype=complex)
                W_C = np.zeros((d, d), dtype=complex) if occ else None
                for r in range(s - 1, -1, -1):
                    a = terms[e * s + r]
                    b = lnp._taylor_terms(bwd, Lam, 1.0, m)
                    Lam = np.tensordot(c, b, axes=1)
                    Bt = np.tensordot(Cm, b[:m], axes=1)
                    W += lnp._w_term(a, Bt[:, :nb], d)
                    if occ:
                        W_C += lnp._w_term(a[:, idx], Bt[:, nb:], d)
                gF[k] += st.grad_beta(fw, e, traces(W)).ravel()
                if want_u:
                    gF_u[k] += (HcT @ W.ravel()).real
                if occ:
                    gC[k] += st.grad_beta(fw, e, traces(W_C)).ravel()
                    if want_u:
                        gC_u[k] += (HcT @ W_C.ravel()).real
            if occ and k >= 1:
                Lam[nb:] += P
    if occ:
        return {"F": Fval, "gF": gF, "gF_u": gF_u, "C": Cval, "gC": gC, "gC_u": gC_u}
    return (Fval, gF, gF_u) if want_u else (Fval, gF)


def directional_derivative(model, fw, u, beta, E, T, w, dt, S, dbeta, du=None, scheme="cf4",
                           stark=True, spec=None):
    """Exact dF along (dbeta (N, K) complex, du (N, 4)) by the augmented generator
    x' = L x, y' = L y - i[dH, x], run through the SAME Taylor polynomial (reference for tests)."""
    u, beta = np.asarray(u, dtype=float), np.asarray(beta, dtype=complex)
    dbeta = np.asarray(dbeta, dtype=complex)
    du = np.zeros_like(u) if du is None else np.asarray(du, dtype=float)
    offs, wts, n_exp = _schedule(S, scheme)
    h = dt / S / n_exp
    s, m = spec_for(model, fw, u, beta, dt, S, scheme) if spec is None else spec
    X = np.asarray(E, dtype=complex)
    Y = np.zeros_like(X)
    with np.errstate(**_QUIET):
        for k in range(u.shape[0]):
            Hk = _hk(model, u[k])
            dHu = np.tensordot(du[k], model.Hc, axes=1)
            st = _Step(fw, beta[k], k, dt, offs, wts, stark)
            for e in range(len(wts)):
                He = st.H(fw, Hk, e)
                dH = st.dH(fw, e, dbeta[k]) + dHu

                def aug(Z, He=He, dH=dH):
                    out = lnp.apply_L(model, He, Z)
                    out[1] += -1j * (dH @ Z[0] - Z[0] @ dH)
                    return out

                Z = lnp.taylor_action(aug, np.stack([X, Y]), h / s, s, m)
                X, Y = Z[0], Z[1]
    return float(np.sum(w * np.einsum("bij,bij->b", T.conj(), Y).real) / 4.0)


# ---------------------------------------------------------------------------
# Penalties on beta (physical units), each (value, gradient (N, 2K))
# ---------------------------------------------------------------------------

def modulus_penalty(b12, cap=BETA_CAP):
    """sum_{t,k} (|beta_k(t)| - cap)_+^2."""
    bI, bQ = b12[:, 0::2], b12[:, 1::2]
    mod = np.hypot(bI, bQ)
    ex = np.maximum(mod - cap, 0.0)
    g = np.zeros_like(b12)
    safe = np.where(mod > 0, mod, 1.0)
    g[:, 0::2], g[:, 1::2] = 2 * ex * bI / safe, 2 * ex * bQ / safe
    return float(np.sum(ex ** 2)), g


def comb_penalty(fw, b12, dt, S, cap=BETA_CAP, scheme="cf4"):
    """(1/n_nodes) sum_t sum_c sum_nodes (|xi_c(tau)| - cap)_+^2 at the propagator's node times
    (averaged per step, so it weighs like the per-sample modulus penalty)."""
    beta = beta_from_real(b12)
    tt = node_times(beta.shape[0], dt, S, scheme)                       # (N, nn)
    ph = np.exp(1j * fw.Delta[None, :, None] * tt[:, None, :])           # (N, 6, nn)
    val, g = 0.0, np.zeros_like(b12)
    nn = tt.shape[1]
    for cc in (1, 2):
        sel = fw.comb == cc
        xi_c = (beta[:, sel, None] * ph[:, sel]).sum(1)                  # (N, nn)
        mod = np.abs(xi_c)
        ex = np.maximum(mod - cap, 0.0)
        val += float(np.sum(ex ** 2)) / nn
        coef = 2 * ex / np.where(mod > 0, mod, 1.0) / nn                 # (N, nn)
        for k in np.flatnonzero(sel):
            dI = np.real(np.conj(xi_c) * ph[:, k]) * coef
            dQ = np.real(np.conj(xi_c) * (-1j) * ph[:, k]) * coef
            g[:, 2 * k] += dI.sum(1)
            g[:, 2 * k + 1] += dQ.sum(1)
    return val, g


def derivative_penalty(b12):
    """sum_{t,k} |beta_k(t+dt) - beta_k(t)|^2 (core.grape_core.derivative_penalty on 2K columns)."""
    diff = b12[1:] - b12[:-1]
    g = np.zeros_like(b12)
    g[1:] += 2 * diff
    g[:-1] -= 2 * diff
    return float(np.sum(diff ** 2)), g


# ---------------------------------------------------------------------------
# Constraint chain on the 2K tone columns
# ---------------------------------------------------------------------------

class BetaChain:
    """x (N, 2K) pre-image -> b12 (N, 2K) physical: ramp(bandlimit(x)) / S_BETA, applied as
    ceil(K/2) core.ramp chains of (N, 4) (two tones each, both bands = `band`; an odd K pads one
    zero column pair, dropped again on output). Symmetric bands make the I - iQ vs I + iQ
    convention irrelevant."""

    def __init__(self, N, dt, band=BETA_BAND, ramp_ns=48.0, scale=S_BETA, n_tones=N_TONES):
        self.N, self.scale, self.n_cols = N, scale, 2 * n_tones
        self.n_blocks = -(-n_tones // 2)
        self.fwd, self.adj = make_constraint_chain(N, dt, band, band, ramp_ns)

    def _blocks(self, f, x):
        x = np.asarray(x, dtype=float).reshape(self.N, self.n_cols)
        xp = np.zeros((self.N, 4 * self.n_blocks))
        xp[:, :self.n_cols] = x
        out = np.hstack([f(xp[:, 4 * i:4 * i + 4]) for i in range(self.n_blocks)])
        return out[:, :self.n_cols] / self.scale

    def to_physical(self, x):
        return self._blocks(self.fwd, x)

    def to_preimage_grad(self, g):
        return self._blocks(self.adj, g)


def constant_preimage(N, beta6, window=None, dt=0.002):
    """Pre-image (N, 2K) of constant tones beta6 (K,) complex, optionally only on the time window
    (t_start, t_end) us (a box; the band projection and the ramp smooth it). Returns x."""
    b = np.tile(np.asarray(beta6, dtype=complex), (N, 1))
    if window is not None:
        t = (np.arange(N) + 0.5) * dt
        b[(t < window[0]) | (t > window[1])] = 0.0
    return real_from_beta(b) * S_BETA


# ---------------------------------------------------------------------------
# Stage 2 objective: tones only, linear drives frozen
# ---------------------------------------------------------------------------

def _eval_piece(n_c, n_r, hot, E, T, w, u, beta, dt, S, mask_kind, forbid_fock_min, occ_weights,
                occ_idx, want_grad, mixing, paths=dev.CORRECTED_FOCK):
    """One (truncation, process-input subset) evaluation; module level so loky can pickle it.
    Models are rebuilt per worker and cached (cheap compared with one evaluation)."""
    model, fw = _cached_model(n_c, n_r, hot, mixing, paths)
    masks = None
    if mask_kind and occ_idx:
        mk = []
        for kind in mask_kind:
            mk.append(lnp.forbidden_mask(model, forbid_fock_min) if kind == "forbid"
                      else lnp.transmon_mask(model))
        masks = np.array(mk)
    if not want_grad:
        R_N, hist = propagate(model, fw, u, beta, dt, S, E, store=True)
        Fv = float(np.sum(w * np.einsum("bij,bij->b", T.conj(), R_N).real) / 4.0)
        C = lnp._occupations(hist[1:] + [R_N], masks, list(occ_idx)) if masks is not None else None
        return Fv, None, C, None
    if masks is None:
        Fv, g = fidelity_and_grad(model, fw, u, beta, E, T, w, dt, S)
        return Fv, g, None, None
    out = fidelity_and_grad(model, fw, u, beta, E, T, w, dt, S, occ_masks=masks,
                            occ_weights=occ_weights, occ_idx=occ_idx)
    return out["F"], out["gF"], out["C"], out["gC"]


_MODEL_CACHE = {}


def _cached_model(n_c, n_r, hot, mixing, paths=dev.CORRECTED_FOCK):
    paths = tuple(int(n) for n in paths)
    key = (n_c, n_r, hot, mixing, paths)
    if key not in _MODEL_CACHE:
        model = Model(n_c, n_r, include_higher_order=hot)
        _MODEL_CACHE[key] = (model, fwm_terms(model, mixing, paths=paths))
    return _MODEL_CACHE[key]


class Stage2Objective:
    """cost(x), grad(x) on the tone pre-image x (N, 2K); u (N, 4) frozen. K = 2 len(paths) tones;
    use band=BETA_BAND_FULL with paths=FULL_PATHS (teeth 2.07 MHz apart).

        cost = 1 - F_pro(u, beta)                       readout-traced target, one truncation
               + amp   * sum (|beta_k| - cap)_+^2
               + comb  * (1/n_nodes) sum (|xi_c(tau)| - cap)_+^2
               + deriv * sum |beta(t+dt) - beta(t)|^2
               + forbid * C_forbid (+ transmon * C_T)
        b12 = BetaChain(x), beta = b12[:, 0::2] - i b12[:, 1::2]

    The three process inputs run as separate joblib jobs (n_jobs=3; they never interact).
    """

    def __init__(self, u, dt, n_c=16, n_r=2, S=S_TRAIN, gate="X", penalties=None, cap=BETA_CAP,
                 band=BETA_BAND, ramp_ns=48.0, forbid_fock_min=10, include_higher_order=True,
                 mixing=0.0, n_jobs=3, paths=dev.CORRECTED_FOCK):
        self.u = np.asarray(u, dtype=float)
        self.paths = tuple(int(n) for n in paths)
        self.N, self.dt, self.n_c, self.n_r, self.S = self.u.shape[0], dt, n_c, n_r, S
        self.cap, self.band, self.ramp_ns = cap, band, ramp_ns
        self.pen = dict(BETA_PENALTIES)
        if penalties:
            self.pen.update(penalties)
        self.hot, self.mixing, self.forbid_fock_min = include_higher_order, mixing, forbid_fock_min
        self.model, self.fw = _cached_model(n_c, n_r, include_higher_order, mixing, self.paths)
        self.n_cols = 2 * self.fw.n_tones
        E, T, w = process_targets(gate, n_c, n_r)
        self.E, self.T, self.w = E, readout_traced(T, n_c, n_r), w
        self.chain = BetaChain(self.N, dt, band, ramp_ns, n_tones=self.fw.n_tones)
        self.mask_kind = tuple(k for k in ("forbid", "transmon")
                               if self.pen[k] > 0 and (k != "forbid" or n_c > forbid_fock_min))
        self.occ_weights = np.array([self.pen[k] for k in self.mask_kind])
        self.n_jobs = n_jobs
        self._pool = None
        self._cache_x, self._cache = None, None
        self.n_evals = 0
        self.best = {"F": -np.inf, "x": None, "b12": None}

    def _map(self, args):
        if self.n_jobs == 1:
            return [_eval_piece(*a) for a in args]
        if self._pool is None:
            self._pool = Parallel(n_jobs=self.n_jobs, backend="loky")
            self._pool.__enter__()
        return self._pool(delayed(_eval_piece)(*a) for a in args)

    def close(self):
        if self._pool is not None:
            self._pool.__exit__(None, None, None)
            self._pool = None

    def evaluate_physical(self, b12, want_grad=True):
        """(F, dF/db12, C (n_masks,), d(sum w C)/db12) of the physical tones b12 (N, 2K)."""
        beta = beta_from_real(b12)
        tasks = []
        for b in range(self.E.shape[0]):
            oi = (0,) if b in lnp.FORBID_IDX else ()
            tasks.append((self.n_c, self.n_r, self.hot, self.E[b:b + 1], self.T[b:b + 1],
                          self.w[b:b + 1], self.u, beta, self.dt, self.S, self.mask_kind,
                          self.forbid_fock_min, self.occ_weights, oi, want_grad, self.mixing,
                          self.paths))
        out = self._map(tasks)
        Fv = sum(o[0] for o in out)
        gF = sum(o[1] for o in out) if want_grad else None
        Cs = [o[2] for o in out if o[2] is not None]
        C = sum(Cs) if Cs else np.zeros(len(self.mask_kind))
        gCs = [o[3] for o in out if o[3] is not None]
        gC = sum(gCs) if gCs else None
        return Fv, gF, C, gC

    def evaluate(self, x):
        x = np.asarray(x, dtype=float).reshape(self.N, self.n_cols)
        if self._cache_x is not None and np.array_equal(x, self._cache_x):
            return self._cache
        b12 = self.chain.to_physical(x)
        Fv, gF, C, gC = self.evaluate_physical(b12)
        if Fv > self.best["F"]:
            self.best = {"F": Fv, "x": x.copy(), "b12": b12.copy()}
        cost, g = 1.0 - Fv, -gF
        terms = {"F_pro": Fv, "infid": 1.0 - Fv}
        for name, fn in (("amp", lambda: modulus_penalty(b12, self.cap)),
                         ("comb", lambda: comb_penalty(self.fw, b12, self.dt, self.S, self.cap)),
                         ("deriv", lambda: derivative_penalty(b12))):
            if self.pen[name] > 0:
                p, gp = fn()
                cost += self.pen[name] * p
                g = g + self.pen[name] * gp
                terms[name] = self.pen[name] * p
        for i, name in enumerate(self.mask_kind):
            cost += self.pen[name] * float(C[i])
            terms[name] = self.pen[name] * float(C[i])
            terms["C_" + name] = float(C[i])
        if gC is not None:
            g = g + gC
        gx = self.chain.to_preimage_grad(g)
        self.n_evals += 1
        self._cache_x, self._cache = x.copy(), (float(cost), gx.ravel(), terms)
        return self._cache

    def cost(self, x):
        return self.evaluate(x)[0]

    def grad(self, x):
        return self.evaluate(x)[1]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def model_with_channels(n_c, n_r=2, keep=None, include_higher_order=True, dephasing=False,
                        heating=False):
    """A DGRAPE Model whose jump operators are only those named in `keep` (names from
    DGRAPE.model.collapse_ops: 'a', 'r', 's_ge', 's_ef', ...); keep=None keeps them all. It filters
    collapse_ops for the duration of one constructor call, then restores it."""
    import DGRAPE.model as dmodel
    if keep is None:
        return Model(n_c, n_r, include_higher_order, True, dephasing, heating)
    original = dmodel.collapse_ops
    dmodel.collapse_ops = lambda *a, **k: [(nm, c) for nm, c in original(*a, **k) if nm in keep]
    try:
        return Model(n_c, n_r, include_higher_order, True, dephasing, heating)
    finally:
        dmodel.collapse_ops = original


def pulse_stats(b12, fw, dt, S=4):
    """max |beta_k|, max_t |xi_c(t)| over the node times, per comb."""
    beta = beta_from_real(b12)
    tt = node_times(beta.shape[0], dt, S)
    ph = np.exp(1j * fw.Delta[None, :, None] * tt[:, None, :])
    xis = [np.abs((beta[:, fw.comb == c, None] * ph[:, fw.comb == c]).sum(1)).max() for c in (1, 2)]
    return {"max_beta": float(np.abs(beta).max()), "max_beta_per_tone": np.abs(beta).max(0).tolist(),
            "max_xi_1": float(xis[0]), "max_xi_2": float(xis[1])}


def score(u, beta, dt, n_c=16, n_r=2, S=4, gate="X", stark=True, mixing=0.0, g_scale=1.0,
          dephasing=False, heating=False, include_higher_order=True, keep=None, occupations=False,
          forbid_fock_min=10, paths=dev.CORRECTED_FOCK):
    """Readout-traced F_pro (no gradient) of a pulse under one scoring condition, plus
    <n_r>(T_g) and P_f(T_g) averaged over |0_L>, |1_L>. `keep` restricts the jump operators to
    the named ones (DGRAPE.model.collapse_ops names) for the channel split. With occupations=True
    also C_forbid (storage Fock >= forbid_fock_min) and C_transmon, the DGRAPE time sums over the
    two code inputs, and the mean transmon excitation <P_e + P_f> over the gate. beta (N, K) must
    match `paths` (K = 2 len(paths))."""
    model = model_with_channels(n_c, n_r, keep, include_higher_order, dephasing, heating)
    fw = fwm_terms(model, mixing, g_scale, paths)
    E, T, w = process_targets(gate, n_c, n_r)
    out = propagate(model, fw, u, beta, dt, S, E, stark=stark, store=occupations)
    R = out[0]
    Fv = float(np.sum(w * np.einsum("bij,bij->b", readout_traced(T, n_c, n_r).conj(), R).real) / 4.0)
    o = operators(n_c, n_r)
    n_r_end = float(np.mean([np.real(np.trace(o["n_r"] @ R[b])) for b in (0, 1)]))
    p_f_end = float(np.mean([np.real(np.trace(o["Pf"] @ R[b])) for b in (0, 1)]))
    res = {"F_pro": Fv, "n_r_end": n_r_end, "P_f_end": p_f_end}
    if occupations:
        states = out[1][1:] + [R]
        masks = np.array([lnp.forbidden_mask(model, forbid_fock_min), lnp.transmon_mask(model)])
        C = lnp._occupations(states, masks, [0, 1])
        res.update({"C_forbid": float(C[0]), "C_transmon": float(C[1]),
                    "mean_transmon_excitation": float(C[1] / (2 * len(states)))})
    return res
