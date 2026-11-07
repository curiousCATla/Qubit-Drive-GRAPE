"""
FWM tails with Gertler-style combs (FWM_tails_ge.ipynb). An X gate (EsT or Ord, both trained closed,
DGRAPE/est_gate.py) runs as trained, then a TAIL of two combs acts on its mixed output with the cavity
drive off, scored against the ideal X at the end of the tail. Same chip (AQEC/device.py), code, jump
operators and Taylor step map as DGRAPE/model.py, DGRAPE/lindblad_np.py and DGRAPE/fwm.py; none of
those modules is changed.

THE SCHEME (Gertler et al. 2021, Eq. 4), on the odd code (|1>+|5>)/sqrt2, |3> whose losses land on
the even states n = 0, 2, 4:

    H_d = sum_{n even} [ lambda_n |n,e,0><n,g,0| + Omega_n |n+1,g,1><n,e,0| ] + h.c.

  comb T  -- the TRANSMON comb: a linear transmon drive eps_T(t) q' + h.c. (DGRAPE's transmon
             quadratures u[:, 2:4]). Its teeth sit at the photon-number-split g -> e lines,
             about -n chi_ge in the DGRAPE frame (0, -2.24, -4.48 MHz).
  comb M  -- the MIXING comb: a four-wave-mixing pump at omega_a + omega_r - omega_q,
             g_m xi_M(t) a' r' q + h.c., which swaps |n,e,0> -> |n+1,g,1>: the storage photon and
             a readout photon appear together and the transmon returns to g. Teeth at about
             +n chi_ge. g_m = sqrt(chi_ge chi_qr) = 2 pi 1.125 MHz per unit xi (the direct count of
             the cosine expansion, as fwm.G_MIX); the matrix element is sqrt(n + 1). The pump also
             Stark-shifts the modes: - |xi_M|^2 S with S = 2 alpha q'q + chi_ge n_a + chi_qr n_r
             (fwm.py's S).
  Then the readout decays (kappa_r) and the repair cannot be undone. The fidelity traces the
  readout out, so the swap already counts as a repair.

THE TRANSMON IS A QUBIT. Nothing in the scheme visits |f>, so the tail runs on transmon {g, e}
(levels=2): d = 2 n_c n_r (64 at n_c 16, against 96), no alpha in the drift, no chi_ef, no T1_ef. The pump
Stark term is kept by hand (its 2 alpha q'q part comes from |f> in a real transmon). The gate before the
tail still runs on three levels, as trained; its output is projected onto {g, e} (the dropped P_f is
reported as `trace_cut`). levels=3 runs the same tail on the full three-level model (held-out check:
the transmon drive can then leak to |f>, and a' r' q also acts on |f> -> |e>, off resonance by alpha).

THE FRAME is DGRAPE's (rotating at the linear mode frequencies). Both combs sit within a few MHz of
zero in it, the controls are piecewise constant on the 2 ns grid, so one exponential per step is exact:
fwm.propagate with scheme="midpoint", S=1, one tone with zero carrier. Gradients come from
fwm.fidelity_and_grad(..., want_u=True), the exact W-form gradient in both u and beta.

CONTROLS. The real array b4 (N, 4) = (eps_T^I, eps_T^Q, xi_M^I, xi_M^Q), eps = I - iQ (as u and
beta in DGRAPE). The pre-image x (N, 4) maps to b4 = ramp(bandlimit(x)) / scale, one core.ramp chain,
with scale = U_BOX / cap per comb so both share the |x| <= 25 box. Caps: |eps_T| <= CAP_T = 2 pi 1 MHz
(Rabi 2 MHz; dropped-|f> leakage ~ (2 eps / alpha)^2 ~ 2e-4), |xi_M| <= cap_M (0.1 or 0.2, a
parameter), by a modulus penalty as in fwm_tail.

CHIPS (FWM_tails_ge_chips.ipynb). Every function that uses a device constant takes chip= (a
DGRAPE.ge_chip.Chip, a (chi_ge_MHz, chi_qr_MHz) pair, or None = Shirol's chip, bit for bit as track 7).
The chip sets the tail's drift (apply_chip on the full model before restricting), g_m, the Stark diagonal,
the tooth lines, and the default comb-T cap and band. gate_chip= says where the GATE ran: None = Shirol's
chip (the trained EsT/Ord pulses as they are: the "transplant" proxy of the chip scan), or a chip (a gate
trained on that chip).

Pulses trained with these tails go to pulses/dgrape/, never pulses/.
"""

import math
import os
from types import SimpleNamespace

import numpy as np
from joblib import Parallel, delayed

from AQEC import device as dev
from core.ramp import make_constraint_chain
from DGRAPE import fwm
from DGRAPE import fwm_tail as ft
from DGRAPE import ge_chip as gc
from DGRAPE import lindblad_np as lnp
from DGRAPE.model import _QUIET, E, G, Model, _flat_sandwich, index, monomial, operators

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PULSE_DIR = os.path.join(REPO_ROOT, "pulses", "dgrape")

# Recipe (FWM_tails_ge.ipynb, decisions of 2026-11-03)
DT = 0.002                       # us
TAIL_US_GRID = (1.0, 2.0, 3.0)   # trained tail lengths
CAP_M_GRID = (0.1, 0.2)          # trained mixing-comb caps
CAP_M = 0.1                      # default |xi_M| cap
CAP_T = 2 * np.pi * 1.0          # |eps_T| cap, rad/us
BAND = (-12.0, 12.0)             # MHz on both combs: teeth within +-4.5 MHz, plus Stark, plus margin
RAMP_NS = 48.0
U_BOX = 25.0
N_C = 16                         # tail truncation
GATE_N_C = 24                    # the closed-trained gates need n_c 24 to run (cut to N_C after)
READOUT = "traced"
G_M = np.sqrt(dev.chi_ge * dev.chi_qr)      # a' r' q coupling per unit xi_M, rad/us
ERROR_FOCK = (0, 2, 4)                      # error states of the odd code (one loss)
GERTLER_LAMBDA_OVER_OMEGA = 27.0 / 88.0     # Gertler's measured lambda : Omega
PENALTIES = {"amp": 8e-5, "deriv": 1e-5}    # in pre-image units (fwm_tail.TAIL_PENALTIES / S^2)
GATES = {"EsT": os.path.join(PULSE_DIR, "u_X_est_err_closed_s24.npy"),
         "Ord": os.path.join(PULSE_DIR, "u_X_ord_closed.npy")}
# The best Shirol-comb tail per gate (FWM_tails.ipynb §6, held out at n_c 28): reference lines only.
SHIROL_BEST = {"EsT": {"tail_us": 0.8, "gain": 0.0032}, "Ord": {"tail_us": 1.0, "gain": -0.0058}}


def n_steps(tail_us, dt=DT):
    return int(round(tail_us / dt))


def gate_pulse(gate):
    """(N, 4) gate pulse: a name in GATES, a .npy path, or an array. d2 is not part of this study."""
    if gate is None:
        raise ValueError("this study compares EsT and Ord only: name the gate (GATES) or pass its pulse")
    if isinstance(gate, str):
        return np.load(GATES.get(gate, gate))
    return np.asarray(gate, dtype=float)


# ---------------------------------------------------------------------------
# The {g, e} space
# ---------------------------------------------------------------------------

def ge_keep(n_c, n_r=2):
    """Indices of the full (g, e, f) space that have the transmon in g or e, in order."""
    return np.array([index(n, j, m, n_c, n_r) for n in range(n_c) for j in (G, E) for m in range(n_r)])


def ge_index(n, j, m, n_c, n_r=2):
    """|n, j, m> in the {g, e} space: (n*2 + j)*n_r + m."""
    if not (0 <= n < n_c and j in (G, E) and 0 <= m < n_r):
        raise IndexError(f"|{n},{j},{m}> outside the {{g,e}} space n_c={n_c}, n_r={n_r}")
    return (n * 2 + j) * n_r + m


def project_ge(R, n_c, n_r=2):
    """(..., d3, d3) -> (..., d2, d2): the block with the transmon in {g, e} (not renormalised)."""
    k = ge_keep(n_c, n_r)
    R = np.asarray(R)
    return R[..., k, :][..., :, k].copy()


def restrict_model(full):
    """A DGRAPE Model on transmon {g, e}, built from the full one by keeping the g, e rows and columns
    of H_d, Hc and every jump operator (|e><f| and the |f><f| dephasing vanish and are dropped). It
    carries every field lindblad_np and fwm read; levels = 2 marks it."""
    k = ge_keep(full.n_c, full.n_r)
    ix = np.ix_(k, k)
    m = object.__new__(Model)
    m.n_c, m.n_r, m.levels = full.n_c, full.n_r, 2
    m.d = len(k)
    m.H_d = full.H_d[ix].copy()
    m.Hc = full.Hc[:, k][:, :, k].copy()
    m.c_ops = [(nm, c[ix].copy()) for nm, c in full.c_ops if np.any(c[ix])]
    hd = np.real(np.diag(m.H_d))
    m.shift = 0.5 * (hd.max() + hd.min())
    m.H_d_shifted = m.H_d - m.shift * np.eye(m.d)
    with np.errstate(**_QUIET):
        Gm = sum((c.conj().T @ c for _, c in m.c_ops), np.zeros_like(m.H_d))
    if not np.allclose(Gm, np.diag(np.diag(Gm))):
        raise ValueError("sum L'L is not diagonal")
    m.Gdiag = np.real(np.diag(Gm)).copy()
    K = len(m.c_ops)
    m.jump_src = np.zeros((K, m.d), dtype=np.int64)
    m.jump_W = np.zeros((K, m.d, m.d), dtype=complex)
    m.jump_src_adj = np.zeros((K, m.d), dtype=np.int64)
    m.jump_W_adj = np.zeros((K, m.d, m.d), dtype=complex)
    for i, (_, c) in enumerate(m.c_ops):
        src, w = monomial(c)
        m.jump_src[i], m.jump_W[i] = src, np.outer(w, w.conj())
        src, w = monomial(c.conj().T)
        m.jump_src_adj[i], m.jump_W_adj[i] = src, np.outer(w, w.conj())
    m.jump_flat = _flat_sandwich(m.jump_src, m.jump_W)
    m.jump_flat_adj = _flat_sandwich(m.jump_src_adj, m.jump_W_adj)
    ops = operators(full.n_c, full.n_r)
    m.norm_a = np.linalg.norm(ops["a"][ix], 2)
    m.norm_q = np.linalg.norm(ops["q"][ix], 2)
    m.half_width_d = 0.5 * (hd.max() - hd.min())
    m.diss_norm = 2.0 * sum(np.linalg.norm(c, 2) ** 2 for _, c in m.c_ops)
    return m


def ge_operators(n_c, n_r=2, levels=2):
    """operators(n_c, n_r) restricted to {g, e} (levels=2) or as is (levels=3)."""
    ops = operators(n_c, n_r)
    if levels == 3:
        return ops
    k = ge_keep(n_c, n_r)
    return {name: op[np.ix_(k, k)] for name, op in ops.items()}


# ---------------------------------------------------------------------------
# The tail model
# ---------------------------------------------------------------------------

def _terms(A, S_diag, g_scale):
    """An fwm.fwm_terms-shaped namespace: ONE tone (comb 1) with operator A and zero carrier, comb 2
    empty, no pump mixing, Stark diagonal S_diag."""
    Z = np.zeros_like(A)
    return SimpleNamespace(
        A=(A, Z), Ad=(A.conj().T, Z), AT=(A.T, Z), AdT=(A.conj(), Z),
        M=Z, Md=Z, MT=Z, MdT=Z, mixing=0.0, g_scale=float(g_scale),
        S_diag=np.asarray(S_diag, dtype=float), Delta=np.zeros(1), comb=np.array([1]),
        tones=[(1, 0)], paths=(0,), n_tones=1,
        normA=(float(np.linalg.norm(A, 2)), 0.0), normM=0.0, normS=float(np.abs(S_diag).max()))


def tail_model(n_c=N_C, n_r=2, levels=2, keep=None, include_higher_order=True, dephasing=False,
               heating=False, g_scale=1.0, chip=None):
    """SimpleNamespace(model, fw, levels, n_c, n_r): the DGRAPE Model (jump operators restricted to
    `keep` as in fwm.model_with_channels), on {g, e} for levels=2, and the mixing-comb term
    g_scale * G_M a' r' q with the pump Stark diagonal."""
    if levels not in (2, 3):
        raise ValueError("levels must be 2 or 3")
    ch = gc.resolve(chip)
    full = gc.apply_chip(fwm.model_with_channels(n_c, n_r, keep, include_higher_order, dephasing, heating),
                         ch, include_higher_order)
    o = operators(n_c, n_r)
    with np.errstate(**_QUIET):
        A = g_scale * ch.g_m * (o["a"].conj().T @ o["r"].conj().T @ o["q"])
        S = np.real(np.diag(2 * ch.alpha * o["q"].conj().T @ o["q"] + ch.chi_ge * o["n_a"]
                            + ch.chi_qr * o["n_r"]))
    if levels == 2:
        k = ge_keep(n_c, n_r)
        model, A, S = restrict_model(full), A[np.ix_(k, k)], S[k]
    else:
        model = full
        model.levels = 3
    return SimpleNamespace(model=model, fw=_terms(A, S, g_scale), levels=levels, n_c=n_c, n_r=n_r,
                           include_higher_order=include_higher_order, chip=ch)


def level_index(tm, n, j, m):
    return ge_index(n, j, m, tm.n_c, tm.n_r) if tm.levels == 2 else index(n, j, m, tm.n_c, tm.n_r)


def tooth_frequencies(sigma=0.0, n_c=N_C, include_higher_order=True, fock=ERROR_FOCK, chip=None):
    """{"fock", "T", "M"} carriers (rad/us) of the teeth in the DGRAPE frame, as fwm.fwm_terms writes
    them (a tooth is beta exp(i Delta t)): comb T on |n,g,0> -> |n,e,0>, comb M on |n,e,0> ->
    |n+1,g,1>, n in `fock`. With sigma = <|xi_M|^2> the lines include the pump's Stark shift
    (energies E - sigma S)."""
    tm = tail_model(n_c, 2, 2, keep=set(), include_higher_order=include_higher_order, chip=chip)
    Ed = np.real(np.diag(tm.model.H_d)) - sigma * tm.fw.S_diag
    e = lambda n, j, m: Ed[level_index(tm, n, j, m)]  # noqa: E731
    return {"fock": tuple(fock),
            "T": np.array([-(e(n, E, 0) - e(n, G, 0)) for n in fock]),
            "M": np.array([-(e(n + 1, G, 1) - e(n, E, 0)) for n in fock])}


def swap_time(xi, n, chip=None):
    """us for a full swap |n,e,0> -> |n+1,g,1> at constant |xi| (closed, no Stark): pi / (2 Omega)."""
    return np.pi / (2 * gc.resolve(chip).g_m * abs(xi) * np.sqrt(n + 1))


def stark_shift_swap(xi, chip=None):
    """rad/us: how far the pump moves the swap line, (2 alpha - chi_ge - chi_qr) |xi|^2."""
    ch = gc.resolve(chip)
    return (2 * ch.alpha - ch.chi_ge - ch.chi_qr) * abs(xi) ** 2


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def eps_xi(b4):
    """(N, 4) real -> (eps_T (N,), xi_M (N,)) complex, eps = I - iQ."""
    b4 = np.asarray(b4, dtype=float).reshape(-1, 4)
    return b4[:, 0] - 1j * b4[:, 1], b4[:, 2] - 1j * b4[:, 3]


def b4_from(eps, xi):
    eps, xi = np.asarray(eps, dtype=complex), np.asarray(xi, dtype=complex)
    return np.stack([eps.real, -eps.imag, xi.real, -xi.imag], axis=1)


def _u_beta(b4):
    b4 = np.asarray(b4, dtype=float).reshape(-1, 4)
    u = np.zeros_like(b4)
    u[:, 2:4] = b4[:, 0:2]
    return u, (b4[:, 2] - 1j * b4[:, 3])[:, None]


def comb_stats(b4):
    eps, xi = eps_xi(b4)
    if eps.size == 0:
        return {"max_eps_T": 0.0, "max_xi_M": 0.0, "max_eps_T_MHz": 0.0}
    mT = float(np.abs(eps).max())
    return {"max_eps_T": mT, "max_xi_M": float(np.abs(xi).max()), "max_eps_T_MHz": mT / (2 * np.pi)}


class Chain:
    """x (N, 4) pre-image -> b4 (N, 4) physical = ramp(bandlimit(x)) / scale, scale = U_BOX / cap per
    comb (columns 0-1 comb T, 2-3 comb M). Linear, so its adjoint maps the gradient back."""

    def __init__(self, N, dt=DT, band=BAND, ramp_ns=RAMP_NS, cap_T=CAP_T, cap_M=CAP_M):
        self.N = N
        self.scale = np.array([U_BOX / cap_T] * 2 + [U_BOX / cap_M] * 2)
        self.fwd, self.adj = make_constraint_chain(N, dt, tuple(band), tuple(band), ramp_ns)

    def to_physical(self, x):
        return self.fwd(np.asarray(x, dtype=float).reshape(self.N, 4)) / self.scale

    def to_preimage_grad(self, g):
        return self.adj(np.asarray(g, dtype=float).reshape(self.N, 4) / self.scale)

    def preimage(self, b4):
        """x whose chain output is b4 when b4 is in band and zero at the ramps (otherwise the chain
        band-limits and ramps it)."""
        return np.asarray(b4, dtype=float) * self.scale


# ---------------------------------------------------------------------------
# Input, targets, propagation
# ---------------------------------------------------------------------------

_GATE_CHIP_CACHE = {}


def gate_output_on_chip(gate, gate_chip, n_c=GATE_N_C, n_r=2, keep=None, include_higher_order=True,
                        dephasing=False, heating=False):
    """(3, d, d) three-level process outputs of the gate run on gate_chip (fwm_tail.gate_output with the
    chip's drift: n_r = 1, then the readout embedded in |0>). Cached in memory."""
    import hashlib
    ch = gc.resolve(gate_chip)
    u = np.ascontiguousarray(gate_pulse(gate), dtype=float)
    key = (hashlib.sha1(u.tobytes()).hexdigest(), u.shape, ch, n_c, n_r,
           None if keep is None else frozenset(keep), include_higher_order, dephasing, heating)
    if key not in _GATE_CHIP_CACHE:
        m = gc.apply_chip(fwm.model_with_channels(n_c, 1, keep, include_higher_order, dephasing, heating),
                          ch, include_higher_order)
        E_in, _, _ = ft.process_targets("X", n_c, 1)
        _GATE_CHIP_CACHE[key] = ft.embed_readout(lnp.propagate(m, u, E_in, DT), n_r)
    return _GATE_CHIP_CACHE[key].copy()


def tail_input(gate, n_c=N_C, n_r=2, gate_n_c=GATE_N_C, levels=2, keep=None, include_higher_order=True,
               dephasing=False, heating=False, gate_chip=None):
    """(R0 (3, d, d), trace_cut): the gate's process outputs (fwm_tail.tail_input: three levels,
    gate at gate_n_c, cut to n_c, readout embedded in |0>), projected onto {g, e} for levels=2.
    trace_cut = 1 - mean trace over the two code inputs (population above Fock n_c and in |f>).
    gate_chip None: the gate on Shirol's chip (as trained in track 7); a chip: the gate run on that chip."""
    if gate_chip is None:
        R = ft.tail_input(gate_pulse(gate), n_c, n_r, gate_n_c, keep, include_higher_order, dephasing, heating)
    else:
        g_nc = gate_n_c or n_c
        R = gate_output_on_chip(gate, gate_chip, g_nc, n_r, keep, include_higher_order, dephasing, heating)
        if g_nc != n_c:
            R = ft.cut_storage(R, n_c, n_r)
    if levels == 2:
        R = project_ge(R, n_c, n_r)
    cut = float(1.0 - np.mean([np.trace(R[b]).real for b in (0, 1)]))
    return R, cut


def tail_targets(n_c=N_C, n_r=2, gate="X", readout=READOUT, levels=2):
    T, w = ft.tail_targets(n_c, n_r, gate, readout)
    return (project_ge(T, n_c, n_r) if levels == 2 else T), w


def propagate(tm, b4, R0, dt=DT, store=False, stark=True, spec=None):
    u, beta = _u_beta(b4)
    return fwm.propagate(tm.model, tm.fw, u, beta, dt, 1, R0, scheme="midpoint", stark=stark,
                         store=store, spec=spec)


def fidelity(tm, b4, R0, T, w, dt=DT, stark=True):
    R, _ = propagate(tm, b4, R0, dt, stark=stark)
    return ft.fidelity_from_outputs(R, T, w)


def fidelity_and_grad(tm, b4, R0, T, w, dt=DT, stark=True, spec=None):
    """(F, dF/db4 (N, 4)): exact for the propagator (fwm's W form, u and beta)."""
    u, beta = _u_beta(b4)
    Fv, gB, gU = fwm.fidelity_and_grad(tm.model, tm.fw, u, beta, R0, T, w, dt, 1, scheme="midpoint",
                                       stark=stark, spec=spec, want_u=True)
    g = np.empty((u.shape[0], 4))
    g[:, 0:2], g[:, 2:4] = gU[:, 2:4], gB
    return Fv, g


def directional_derivative(tm, b4, R0, T, w, db4, dt=DT, stark=True):
    """Exact dF along db4 by fwm's augmented generator (reference for tests)."""
    u, beta = _u_beta(b4)
    du, dbeta = _u_beta(db4)
    return fwm.directional_derivative(tm.model, tm.fw, u, beta, R0, T, w, dt, 1, dbeta, du=du,
                                      scheme="midpoint", stark=stark)


# ---------------------------------------------------------------------------
# Starts
# ---------------------------------------------------------------------------

def teeth(amp_T, amp_M, N, dt=DT, sigma=0.0, window_T=None, window_M=None, include_higher_order=True,
          chip=None, offset_M=0.0):
    """b4 (N, 4) of constant teeth on ERROR_FOCK: eps_T = sum_n amp_T[n] exp(i Delta_T,n t),
    xi_M = sum_n amp_M[n] exp(i Delta_M,n t) (sampled at step midpoints), each comb optionally only
    inside a time window (t0, t1) us. sigma = <|xi_M|^2> puts the Stark shift into the carriers;
    offset_M (rad/us) moves every comb-M carrier by the same amount."""
    fr = tooth_frequencies(sigma, include_higher_order=include_higher_order, chip=chip)
    t = (np.arange(N) + 0.5) * dt
    eps = (np.asarray(amp_T, dtype=complex)[None] * np.exp(1j * fr["T"][None] * t[:, None])).sum(1)
    xi = (np.asarray(amp_M, dtype=complex)[None]
          * np.exp(1j * (fr["M"][None] + offset_M) * t[:, None])).sum(1)
    for arr, win in ((eps, window_T), (xi, window_M)):
        if win is not None:
            arr[(t < win[0]) | (t > win[1])] = 0.0
    return b4_from(eps, xi)


def gertler_amplitudes(cap_M=CAP_M, chip=None):
    """(lambda_n, xi_n) of Gertler's operating point moved to this chip: equal swap rates Omega on
    every path (xi_n = Omega / (g_m sqrt(n+1))), the comb-M teeth adding up to cap_M, and
    lambda = (27/88) Omega on every comb-T tooth."""
    g_m = gc.resolve(chip).g_m
    inv = np.array([1 / np.sqrt(n + 1) for n in ERROR_FOCK])
    Omega = cap_M * g_m / inv.sum()
    return np.full(len(ERROR_FOCK), GERTLER_LAMBDA_OVER_OMEGA * Omega), Omega / g_m * inv


def start_library(tail_us, cap_M=CAP_M, cap_T=None, t_sel=None, dt=DT, chip=None):
    """{name: b4 (N, 4)} training / scan starts:
      zero        -- nothing (training adds noise)
      gertler     -- Gertler's continuous teeth (gertler_amplitudes), Stark-corrected carriers
      sequential  -- a selective pi pulse on every error state in [0, t_sel] (comb T alone, lambda =
                     pi / (2 t_sel) per tooth), then the swap in [t_sel, T] (comb M alone, the full-swap
                     amplitude per path, scaled so the teeth add up to <= cap_M).
    chip: cap_T defaults to the chip's, t_sel to 0.8 us x (chi_ge Shirol / chi_ge)."""
    ch = gc.resolve(chip)
    if cap_T is None:
        cap_T = ch.cap_T
    if t_sel is None:
        t_sel = 0.8 if ch is gc.SHIROL else 0.8 * gc.CHI_GE_SHIROL_MHZ / ch.chi_ge_MHz
    N = n_steps(tail_us, dt)
    lam, xin = gertler_amplitudes(cap_M, ch)
    lib = {"zero": np.zeros((N, 4)),
           "gertler": teeth(lam, xin, N, dt, sigma=float(np.sum(xin ** 2)), chip=ch)}
    t_sel = min(t_sel, 0.5 * tail_us)
    lam_s = np.full(len(ERROR_FOCK), min(np.pi / (2 * t_sel), cap_T / len(ERROR_FOCK)))
    t_rem = tail_us - t_sel
    xs = np.array([np.pi / (2 * ch.g_m * np.sqrt(n + 1) * t_rem) for n in ERROR_FOCK])
    xs *= min(1.0, cap_M / xs.sum())
    lib["sequential"] = (teeth(lam_s, np.zeros(len(ERROR_FOCK)), N, dt, window_T=(0.0, t_sel), chip=ch)
                         + teeth(np.zeros(len(ERROR_FOCK)), xs, N, dt, sigma=float(np.sum(xs ** 2)),
                                 window_M=(t_sel, tail_us), chip=ch))
    return lib


# ---------------------------------------------------------------------------
# Training objective
# ---------------------------------------------------------------------------

_TAIL_CACHE = {}


def _cached_tail(n_c, n_r, levels, hot, chip=None):
    ch = gc.resolve(chip)
    key = (n_c, n_r, levels, hot, ch)
    if key not in _TAIL_CACHE:
        _TAIL_CACHE[key] = tail_model(n_c, n_r, levels, include_higher_order=hot, chip=ch)
    return _TAIL_CACHE[key]


def _eval_piece(n_c, n_r, levels, hot, R0, T, w, b4, dt, want_grad, chip=None):
    """One process input; module level so loky can pickle it (models cached per worker)."""
    with np.errstate(**_QUIET):
        tm = _cached_tail(n_c, n_r, levels, hot, chip)
        if want_grad:
            return fidelity_and_grad(tm, b4, R0, T, w, dt)
        return fidelity(tm, b4, R0, T, w, dt), None


class TailGEObjective:
    """cost(x), grad(x) on the tail pre-image x (N, 4); the gate fixed (its output is the input).

        cost = 1 - F_pro(gate + tail vs ideal X)          one truncation, readout traced or empty
               + amp   * sum_t sum_c (|c(t)| - cap_c)_+^2  * scale_c^2
               + deriv * sum_t |b4(t + dt) - b4(t)|^2        * scale_c^2   (per comb)
        b4 = Chain(x)

    The penalty weights are in pre-image units (scale_c = U_BOX / cap_c), the balance of
    fwm_tail.TailObjective. The three process inputs run as separate joblib jobs."""

    def __init__(self, gate, tail_us, cap_M=CAP_M, cap_T=None, n_c=N_C, n_r=2, gate_n_c=GATE_N_C,
                 levels=2, target="X", readout=READOUT, band=None, ramp_ns=RAMP_NS, penalties=None,
                 include_higher_order=True, n_jobs=3, inputs=None, dt=DT, chip=None, gate_chip=None):
        self.chip = gc.resolve(chip)
        self.gate_chip = None if gate_chip is None else gc.resolve(gate_chip)
        cap_T = self.chip.cap_T if cap_T is None else cap_T
        band = self.chip.band if band is None else band
        self.N, self.dt, self.tail_us = n_steps(tail_us, dt), dt, tail_us
        self.n_c, self.n_r, self.gate_n_c, self.levels = n_c, n_r, gate_n_c, levels
        self.target, self.readout, self.hot = target, readout, include_higher_order
        self.cap = np.array([cap_T, cap_M])
        self.band, self.ramp_ns = tuple(band), ramp_ns
        self.pen = dict(PENALTIES)
        if penalties:
            self.pen.update(penalties)
        if inputs is None:
            self.R0, self.trace_cut = tail_input(gate, n_c, n_r, gate_n_c, levels,
                                                 include_higher_order=include_higher_order,
                                                 gate_chip=self.gate_chip)
        else:
            self.R0 = np.asarray(inputs, dtype=complex)
            self.trace_cut = float(1.0 - np.mean([np.trace(self.R0[b]).real for b in (0, 1)]))
        self.T, self.w = tail_targets(n_c, n_r, target, readout, levels)
        self.chain = Chain(self.N, dt, self.band, ramp_ns, cap_T, cap_M)
        self.n_jobs = n_jobs
        self._pool = None
        self._cache_x, self._cache = None, None
        self.n_evals = 0
        self.best = {"F": -np.inf, "x": None, "b4": None}

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

    def evaluate_physical(self, b4, want_grad=True):
        tasks = [(self.n_c, self.n_r, self.levels, self.hot, self.R0[b:b + 1], self.T[b:b + 1],
                  self.w[b:b + 1], b4, self.dt, want_grad, self.chip) for b in range(self.R0.shape[0])]
        out = self._map(tasks)
        return sum(o[0] for o in out), (sum(o[1] for o in out) if want_grad else None)

    def penalty_terms(self, b4):
        """{name: (value, gradient (N, 4))} of the amplitude and derivative penalties."""
        out = {}
        val_a, g_a = 0.0, np.zeros_like(b4)
        val_d, g_d = 0.0, np.zeros_like(b4)
        for c, sl in enumerate((slice(0, 2), slice(2, 4))):
            s2 = (U_BOX / self.cap[c]) ** 2
            p, gp = fwm.modulus_penalty(b4[:, sl], self.cap[c])
            val_a += s2 * p
            g_a[:, sl] += s2 * gp
            p, gp = fwm.derivative_penalty(b4[:, sl])
            val_d += s2 * p
            g_d[:, sl] += s2 * gp
        out["amp"] = (self.pen["amp"] * val_a, self.pen["amp"] * g_a)
        out["deriv"] = (self.pen["deriv"] * val_d, self.pen["deriv"] * g_d)
        return out

    def evaluate(self, x):
        x = np.asarray(x, dtype=float).reshape(self.N, 4)
        if self._cache_x is not None and np.array_equal(x, self._cache_x):
            return self._cache
        b4 = self.chain.to_physical(x)
        Fv, gF = self.evaluate_physical(b4)
        if Fv > self.best["F"]:
            self.best = {"F": Fv, "x": x.copy(), "b4": b4.copy()}
        cost, g = 1.0 - Fv, -gF
        terms = {"F_pro": Fv, "infid": 1.0 - Fv, **comb_stats(b4)}
        for name, (p, gp) in self.penalty_terms(b4).items():
            if self.pen[name] > 0:
                cost += p
                g = g + gp
                terms[name] = p
        gx = self.chain.to_preimage_grad(g)
        self.n_evals += 1
        self._cache_x, self._cache = x.copy(), (float(cost), gx.ravel(), terms)
        return self._cache

    def cost(self, x):
        return self.evaluate(x)[0]

    def grad(self, x):
        return self.evaluate(x)[1]


# ---------------------------------------------------------------------------
# Scoring (no gradient)
# ---------------------------------------------------------------------------

def frame_tracked_fidelity(R, T, w, n_c, n_r=2, levels=2, n_grid=721):
    """(F, theta): F_pro after the best FREE cavity-frame rotation exp(i theta n_a) of the outputs (a
    frame update, as an experiment tracks the deterministic Stark and Kerr rotation of the storage;
    FWM_tails.ipynb §8's Kerr decoding is the same idea). A grid over [-pi, pi), refined by a bounded
    1-D search. Training never uses it: decision of 2026-11-03, report it next to F_pro."""
    from scipy.optimize import minimize_scalar
    n = np.real(np.diag(ge_operators(n_c, n_r, levels)["n_a"]))
    A = np.einsum("bij,bij,b->ij", T.conj(), R, w) / 4.0        # F(theta) = Re sum_ij A_ij e^{i theta (n_i - n_j)}
    dn = n[:, None] - n[None, :]
    f = lambda th: float(np.real(np.sum(A * np.exp(1j * th * dn))))  # noqa: E731
    grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    vals = [f(t) for t in grid]
    t0 = grid[int(np.argmax(vals))]
    step = 2 * np.pi / n_grid
    r = minimize_scalar(lambda t: -f(t), bounds=(t0 - step, t0 + step), method="bounded",
                        options={"xatol": 1e-10})
    return (-r.fun, float(r.x)) if -r.fun >= max(vals) else (max(vals), float(t0))


def decoded_fidelity(R, T, w, n_c, n_r=2, levels=2, n_grid=96):
    """(F, theta, phi): F_pro after the best free storage phase exp(i [theta n + phi n(n-1)/2]) of the
    outputs: a frame update plus a Kerr-phase decoding, both deterministic (Gertler 2021, Methods,
    absorbed the Kerr phase into the decoding). Reported only; never trained on."""
    from scipy.optimize import minimize
    n = np.real(np.diag(ge_operators(n_c, n_r, levels)["n_a"]))
    q = 0.5 * n * (n - 1)
    A = np.einsum("bij,bij,b->ij", T.conj(), R, w) / 4.0
    i, j = np.nonzero(np.abs(A) > 1e-14)
    a, dn, dq = A[i, j], n[i] - n[j], q[i] - q[j]
    f = lambda p: float(np.real(np.sum(a * np.exp(1j * (p[0] * dn + p[1] * dq)))))  # noqa: E731
    g = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    TH, PH = np.meshgrid(g, g, indexing="ij")
    vals = np.real(np.exp(1j * (TH[..., None] * dn + PH[..., None] * dq)) @ a)
    k = np.unravel_index(int(np.argmax(vals)), vals.shape)
    p0 = np.array([g[k[0]], g[k[1]]])
    r = minimize(lambda p: -f(p), p0, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-13})
    if -r.fun >= vals[k]:
        return float(-r.fun), float(r.x[0]), float(r.x[1])
    return float(vals[k]), float(p0[0]), float(p0[1])


def score(b4, gate, n_c=N_C, n_r=2, gate_n_c=GATE_N_C, levels=2, target="X", readout=READOUT,
          include_higher_order=True, keep=None, dephasing=False, heating=False, stark=True, g_scale=1.0,
          pump_scale=1.0, dt=DT, chip=None, gate_chip=None, decode=False):
    """F_pro of gate + tail under one condition. The channel options (keep, dephasing, heating) apply to
    the gate AND the tail; stark, g_scale and pump_scale (both combs times this factor) to the tail.
    gate_n_c = None runs the gate at n_c (no cut). b4 with 0 rows scores the gate alone (projected for
    levels=2). Returns F_pro (by readout), F_traced, F_empty, trace_cut, and at the end, averaged over
    |0_L>, |1_L>: <n_r>, P_e, P_f (levels 3), P(n >= 10), plus the comb maxima."""
    b4 = pump_scale * np.asarray(b4, dtype=float).reshape(-1, 4)
    R0, cut = tail_input(gate, n_c, n_r, gate_n_c if gate_n_c else n_c, levels, keep, include_higher_order,
                         dephasing, heating, gate_chip)
    if b4.shape[0] == 0:
        R = R0
    else:
        tm = tail_model(n_c, n_r, levels, keep, include_higher_order, dephasing, heating, g_scale, chip)
        R, _ = propagate(tm, b4, R0, dt, stark=stark)
    T_tr, w = tail_targets(n_c, n_r, target, "traced", levels)
    T_em, _ = tail_targets(n_c, n_r, target, "empty", levels)
    F_tr, F_em = ft.fidelity_from_outputs(R, T_tr, w), ft.fidelity_from_outputs(R, T_em, w)
    F_trk, theta = frame_tracked_fidelity(R, T_tr if readout == "traced" else T_em, w, n_c, n_r, levels)
    o = ge_operators(n_c, n_r, levels)
    mean = lambda op: float(np.mean([np.real(np.trace(op @ R[b])) for b in (0, 1)]))  # noqa: E731
    n_diag = np.real(np.diag(o["n_a"]))
    res = {"F_pro": F_tr if readout == "traced" else F_em, "F_traced": F_tr, "F_empty": F_em,
           "F_tracked": F_trk, "theta_track": theta,
           "trace_cut": cut, "n_r_end": mean(o["n_r"]), "P_e_end": mean(o["Pe"]),
           "P_f_end": mean(o["Pf"]) if levels == 3 else 0.0,
           "P_n10_end": mean(np.diag((n_diag >= 10).astype(float))),
           "tail_us": b4.shape[0] * dt, **comb_stats(b4)}
    if decode:
        res["F_decoded"], res["theta_dec"], res["phi_dec"] = decoded_fidelity(
            R, T_tr if readout == "traced" else T_em, w, n_c, n_r, levels)
    return res


def baselines(gate, tail_us_list=TAIL_US_GRID, cap_M=CAP_M, n_c=N_C, n_r=2, gate_n_c=GATE_N_C,
              readout=READOUT, starts=("gertler", "sequential"), chip=None, gate_chip=None):
    """Rows {gate, case, tail_us, cap_M, F_pro, gain}: the gate alone, gate + idle, gate + each start
    of start_library (raw, no chain) for every length; gain against the gate alone."""
    kw = dict(readout=readout, chip=chip, gate_chip=gate_chip)
    F0 = score(np.zeros((0, 4)), gate, n_c, n_r, gate_n_c, **kw)["F_pro"]
    rows = [{"gate": gate, "case": "alone", "tail_us": 0.0, "cap_M": 0.0, "F_pro": F0, "gain": 0.0}]
    for T in tail_us_list:
        lib = start_library(T, cap_M, chip=chip)
        cases = [("idle", lib["zero"])] + [(s, lib[s]) for s in starts]
        for name, b4 in cases:
            F_ = score(b4, gate, n_c, n_r, gate_n_c, **kw)["F_pro"]
            rows.append({"gate": gate, "case": name, "tail_us": T, "cap_M": 0.0 if name == "idle" else cap_M,
                         "F_pro": F_, "gain": F_ - F0})
    return rows


def prescan_rows(gate, tail_us, cap_M, starts=("gertler", "sequential"), n_c=N_C, gate_n_c=GATE_N_C):
    """Forward-only scan rows for one (gate, length, cap): each start, raw (no chain), and through the
    training chain (band + ramp), against the gate alone."""
    F0 = score(np.zeros((0, 4)), gate, n_c, 2, gate_n_c)["F_pro"]
    lib = start_library(tail_us, cap_M)
    ch = Chain(n_steps(tail_us), cap_M=cap_M)
    rows = []
    for s in starts:
        for kind, b4 in (("raw", lib[s]), ("chain", ch.to_physical(ch.preimage(lib[s])))):
            r = score(b4, gate, n_c, 2, gate_n_c)
            rows.append({"gate": gate, "start": s, "kind": kind, "tail_us": tail_us, "cap_M": cap_M,
                         "F_alone": F0, "F_pro": r["F_pro"], "gain": r["F_pro"] - F0,
                         "P_e_end": r["P_e_end"], "n_r_end": r["n_r_end"],
                         "max_xi_M": r["max_xi_M"], "max_eps_T_MHz": r["max_eps_T_MHz"]})
    return rows


def prescan(gates=("EsT", "Ord"), tail_us_list=TAIL_US_GRID, caps=(0.1, 0.15, 0.2, 0.3), n_jobs=4):
    """All prescan_rows, in parallel over (gate, length, cap)."""
    jobs = [(g, T, c) for g in gates for T in tail_us_list for c in caps]
    out = Parallel(n_jobs=n_jobs, backend="loky")(delayed(prescan_rows)(g, T, c) for g, T, c in jobs)
    return [r for rows in out for r in rows]


def numbers_table(caps=(0.1, 0.14, 0.2, 0.3)):
    """Rows {cap_M, swap_us_n (n = 0, 2, 4), stark_MHz, rate_kHz_n}: the time scales of the mixing comb."""
    rows = []
    for c in caps:
        row = {"cap_M": c, "stark_MHz": stark_shift_swap(c) / (2 * np.pi)}
        for n in ERROR_FOCK:
            row[f"swap_us_{n}"] = swap_time(c, n)
            row[f"rate_kHz_{n}"] = 1e3 * G_M * c * math.sqrt(n + 1) / (2 * np.pi)
        rows.append(row)
    return rows
