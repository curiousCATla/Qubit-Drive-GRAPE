"""
FWM tails (FWM_tails.ipynb): the d2 X gate runs as trained, then a TAIL of four-wave-mixing (FWM)
pumps acts on its mixed output state, with the linear drives off. The fidelity is scored against the
ideal X at the end of the tail. Same chip, drift, jump operators and Taylor step map as
DGRAPE/model.py, DGRAPE/lindblad_np.py and DGRAPE/fwm.py; none of those modules is changed.

ONE WAVEFORM PER COMB. DGRAPE/fwm.py writes each comb as a sum of teeth with fixed carriers,
xi_c(t) = sum_n beta_{c,n}(t) exp(i Delta_{c,n} t), in the DGRAPE frame (carriers ~ +-140 MHz). Here
each comb is ONE complex waveform eta_c(t), piecewise constant on the 2 ns grid, in a frame that
rotates |f> at the reference tooth Delta_ref = Delta_{1,n_ref} (comb 1, repair n_ref - 1 -> n_ref):

    xi_1(t) = eta_1(t) exp(+i Delta_ref t),     xi_2(t) = eta_2(t) exp(-i Delta_ref t),

t counted from the start of the tail. A constant eta_1 is one tooth at n_ref. Tooth n of comb c is
frequency content of eta_c at the offset `tooth_offsets` gives, about (n - n_ref) chi_gf for comb 1
and the mirror image for comb 2. |eta_c(t)| = |xi_c(t)|, so the comb cap of fwm.py carries over.

THE FRAME, AND WHY IT IS EXACT. With the linear drives off, rotate |f> only: rho' = V' rho V with
V = exp(i Delta_ref t P_f). Then
    H' = H_d + Delta_ref P_f + (eta_1 g1 A_1 + eta_2 g2 A_2 + h.c.) - (|eta_1|^2 + |eta_2|^2) S
         [ + eta_1 eta_2 M + h.c. ]
  - A_1 = a' q'^2 raises g -> f and gains exp(-i Delta_ref t) from V, which cancels xi_1's carrier;
    A_2 = q^2 r' lowers f -> g and gains the opposite phase, which cancels xi_2's;
  - H_d, S and M = a' r' are diagonal or do not touch |f>, so they commute with P_f;
  - the jump operators a, r, |g><e| (and the opt-in dephasing and heating ones) commute with P_f;
    |e><f| only gains a global phase, which the dissipator D[c] ignores;
  - V = 1 at the start of the tail, and every target has the transmon in |g>, where V acts as 1,
    so F_pro is the same in both frames.
With eta constant over a step the generator is constant over the step, so ONE exponential per step
is the exact solution: fwm.propagate with scheme="midpoint", S=1 and the carriers set to 0. No
sub-steps. This needs the transmon drive OFF (its e -> f element would turn at Delta_ref). The
cavity drive commutes with P_f and would keep the frame exact.

WHAT IS REUSED. tail_model() returns a DGRAPE Model whose drift carries the + Delta_ref P_f shift
and an fwm.fwm_terms namespace with one "tone" per comb (paths=(n_ref,)) and zero carriers. From
there fwm.propagate, fwm.fidelity_and_grad (exact W-form gradient, Stark, mixing, occupation
costate), fwm.directional_derivative, fwm.modulus_penalty, fwm.derivative_penalty and fwm.BetaChain
(two tones = exactly one (N, 4) block of core.ramp.make_constraint_chain) apply unchanged. The
real control array is (N, 4) = (eta_1^I, eta_1^Q, eta_2^I, eta_2^Q) with eta = eta^I - i eta^Q.

INPUT. d2_output(): the three process inputs E_00, E_11, E_01 propagated through
pulses/dgrape/u_X_open_d2.npy at n_r = 1 (exact: nothing drives the readout during d2), then
embedded with the readout in |0>.

TARGET. The ideal X on the code, transmon in |g>; readout traced out ("traced", the default) or
required empty ("empty"). No free rotation: the deterministic Kerr phase of the storage and the
tail's own Stark rotation are errors the tail has to cancel.

Pulses trained with these tails go to pulses/dgrape/, never pulses/.
"""

import math
import os
from types import SimpleNamespace

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import expm
from scipy.optimize import minimize

from AQEC import device as dev
from DGRAPE import fwm
from DGRAPE import lindblad_np as lnp
from DGRAPE.code import IDEAL_LOGICAL_U, PROCESS_PAIRS, embedded_basis, process_targets
from DGRAPE.model import _QUIET, F, G, Model, drift_hamiltonian, index, operators

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U_D2_PATH = os.path.join(REPO_ROOT, "pulses", "dgrape", "u_X_open_d2.npy")

# Recipe (FWM_tails.ipynb, decisions of 2026-10-19)
DT = 0.002                   # us, the DGRAPE control grid
N_REF = 3                    # reference tooth: comb-1 repair 2 -> 3 (the code's mean photon number)
TAIL_US = 3.0                # default tail length (a parameter everywhere)
ETA_CAP = 0.1                # |eta_c(t)| <= 0.1 per comb (= fwm.py's comb cap on |xi_c|)
ETA_BAND = (-12.0, 12.0)     # MHz on each eta_c: teeth n = 1, 3, 5 at about -4.1, 0, +4.1 MHz,
#                              plus the comb-1 Stark shift at the cap (5.4 MHz), plus margin
RAMP_NS = 48.0
S_ETA = fwm.S_BETA           # x = 250 eta, so the pre-image shares the |x| <= 25 box of u
U_BOX = fwm.U_BOX
REPAIR_PATHS = dev.CORRECTED_FOCK          # teeth that repair a loss from the odd code words
READOUT = "traced"
TAIL_PENALTIES = {"amp": fwm.BETA_PENALTIES["amp"], "deriv": fwm.BETA_PENALTIES["deriv"],
                  "forbid": 0.0, "transmon": 0.0}


def n_steps(tail_us, dt=DT):
    return int(round(tail_us / dt))


def d2_pulse():
    return np.load(U_D2_PATH)


# ---------------------------------------------------------------------------
# The frame and the tail model
# ---------------------------------------------------------------------------

def frame_frequency(n_c, n_r=2, n_ref=N_REF, include_higher_order=True):
    """Delta_ref (rad/us): the bare comb-1 carrier of the repair n_ref - 1 -> n_ref in the DGRAPE frame,
    -(E_{n_ref,f,0} - E_{n_ref-1,g,0}), the same formula as fwm.fwm_terms."""
    Ed = np.real(np.diag(drift_hamiltonian(n_c, n_r, include_higher_order)))
    return -(Ed[index(n_ref, F, 0, n_c, n_r)] - Ed[index(n_ref - 1, G, 0, n_c, n_r)])


def tail_model(n_c, n_r=2, n_ref=N_REF, include_higher_order=True, keep=None, dephasing=False,
               heating=False, mixing=0.0, g_scale=1.0):
    """SimpleNamespace(model, fw, delta_ref, n_c, n_r, n_ref) for the tail, in the frame that rotates
    |f> at Delta_ref (module docstring). model is a DGRAPE Model (jump operators restricted to `keep`
    as in fwm.model_with_channels) with H_d -> H_d + Delta_ref P_f; fw is fwm.fwm_terms with one tone
    per comb, (1, n_ref) and (2, n_ref), and both carriers set to 0, so beta (N, 2) IS eta."""
    model = fwm.model_with_channels(n_c, n_r, keep, include_higher_order, dephasing, heating)
    delta_ref = frame_frequency(n_c, n_r, n_ref, include_higher_order)
    model.H_d = model.H_d + delta_ref * operators(n_c, n_r)["Pf"]
    # The only Model fields derived from H_d (DGRAPE/model.py): the centring shift and the half width
    # used by the Taylor norm bound.
    hd = np.real(np.diag(model.H_d))
    model.shift = 0.5 * (hd.max() + hd.min())
    model.H_d_shifted = model.H_d - model.shift * np.eye(model.d)
    model.half_width_d = 0.5 * (hd.max() - hd.min())
    fw = fwm.fwm_terms(model, mixing, g_scale, paths=(n_ref,))
    fw.Delta = np.zeros(fw.n_tones)
    return SimpleNamespace(model=model, fw=fw, delta_ref=float(delta_ref), n_c=n_c, n_r=n_r,
                           n_ref=n_ref, include_higher_order=include_higher_order)


def tooth_offsets(n_c=16, n_ref=N_REF, paths=REPAIR_PATHS, include_higher_order=True):
    """Where each tooth sits inside eta (rad/us): {"paths", "comb1", "comb2"}. comb1[k] is
    Delta_{1,n} - Delta_ref and comb2[k] is Delta_{2,n} + Delta_ref, n = paths[k], with the bare
    DGRAPE-frame carriers of fwm.fwm_terms."""
    fw0 = fwm.fwm_terms(Model(n_c, 2, include_higher_order=include_higher_order), paths=paths)
    ref = frame_frequency(n_c, 2, n_ref, include_higher_order)
    K = len(fw0.paths)
    return {"paths": fw0.paths, "comb1": fw0.Delta[:K] - ref, "comb2": fw0.Delta[K:] + ref}


# ---------------------------------------------------------------------------
# Input (the d2 output) and targets
# ---------------------------------------------------------------------------

def embed_readout(R, n_r):
    """(..., 3 n_c, 3 n_c) states without readout -> (..., 3 n_c n_r, 3 n_c n_r) with the readout in
    |0> (readout innermost: index (n*3 + j)*n_r + m, DGRAPE/model.py)."""
    R = np.asarray(R, dtype=complex)
    D = R.shape[-1]
    out = np.zeros(R.shape[:-2] + (D * n_r, D * n_r), dtype=complex)
    out[..., ::n_r, ::n_r] = R
    return out


_D2_CACHE = {}


def d2_output(n_c=16, n_r=2, keep=None, include_higher_order=True, dephasing=False, heating=False):
    """(3, d, d): Phi_d2(E_b) for the process inputs E_00, E_11, E_01 (DGRAPE.code.PROCESS_PAIRS),
    propagated at n_r = 1 with the jump operators named in `keep` (None = the decay set, plus the
    opt-in channels), then embedded with the readout in |0>. Cached in memory."""
    key = (n_c, n_r, None if keep is None else frozenset(keep), include_higher_order, dephasing, heating)
    if key not in _D2_CACHE:
        m = fwm.model_with_channels(n_c, 1, keep, include_higher_order, dephasing, heating)
        E_in, _, _ = process_targets("X", n_c, 1)      # the inputs do not depend on the gate
        _D2_CACHE[key] = embed_readout(lnp.propagate(m, d2_pulse(), E_in, DT), n_r)
    return _D2_CACHE[key].copy()


def tail_targets(n_c, n_r=2, gate="X", readout=READOUT):
    """(T, w): the ideal-gate targets of DGRAPE.code.process_targets, readout traced out
    (fwm.readout_traced) or required empty."""
    _, T, w = process_targets(gate, n_c, n_r)
    if readout == "traced":
        return fwm.readout_traced(T, n_c, n_r), w
    if readout == "empty":
        return T, w
    raise ValueError(f"readout must be 'traced' or 'empty', not {readout!r}")


def fidelity_from_outputs(R, T, w):
    return float(np.sum(w * np.einsum("bij,bij->b", T.conj(), R).real) / 4.0)


# ---------------------------------------------------------------------------
# Propagation and the exact gradient (thin wrappers on fwm.py)
# ---------------------------------------------------------------------------

def _u0(eta):
    return np.zeros((np.shape(eta)[0], 4))


def propagate(tm, eta, R0, dt=DT, store=False, stark=True, spec=None):
    """Propagate the stack R0 through the tail eta (N, 2) complex: one exact exponential per step.
    Returns (rho(N), (s, m)) or, with store=True, (rho(N), [rho(0), ..., rho(N-1)])."""
    eta = np.asarray(eta, dtype=complex)
    return fwm.propagate(tm.model, tm.fw, _u0(eta), eta, dt, 1, R0, scheme="midpoint", stark=stark,
                         store=store, spec=spec)


def fidelity(tm, eta, R0, T, w, dt=DT, stark=True):
    R, _ = propagate(tm, eta, R0, dt, stark=stark)
    return fidelity_from_outputs(R, T, w)


def fidelity_and_grad(tm, eta, R0, T, w, dt=DT, stark=True, spec=None, **occ):
    """(F, dF/deta (N, 4) real, columns (eta_1^I, eta_1^Q, eta_2^I, eta_2^Q)), exact for the
    propagator; with occ_masks etc. (fwm.fidelity_and_grad) the occupation dict instead."""
    eta = np.asarray(eta, dtype=complex)
    return fwm.fidelity_and_grad(tm.model, tm.fw, _u0(eta), eta, R0, T, w, dt, 1, scheme="midpoint",
                                 stark=stark, spec=spec, **occ)


def directional_derivative(tm, eta, R0, T, w, deta, dt=DT, stark=True):
    """Exact dF along deta (N, 2) complex by the augmented generator (reference for tests)."""
    eta = np.asarray(eta, dtype=complex)
    return fwm.directional_derivative(tm.model, tm.fw, _u0(eta), eta, R0, T, w, dt, 1, deta,
                                      scheme="midpoint", stark=stark)


def eta_stats(eta):
    eta = np.asarray(eta, dtype=complex)
    if eta.size == 0:
        return {"max_eta_1": 0.0, "max_eta_2": 0.0}
    return {"max_eta_1": float(np.abs(eta[:, 0]).max()), "max_eta_2": float(np.abs(eta[:, 1]).max())}


# ---------------------------------------------------------------------------
# Waveforms from teeth, pre-images, starts
# ---------------------------------------------------------------------------

def eta_from_teeth(beta_teeth, N, dt=DT, n_c=16, n_ref=N_REF, paths=REPAIR_PATHS, window=None,
                   include_higher_order=True):
    """eta (N, 2) of teeth with complex amplitudes beta_teeth, (2K,) constant or (N, 2K), in the order of
    fwm.tone_list(paths) (comb-1 teeth, then comb-2 teeth): eta_c(t) = sum_n beta_{c,n} exp(i delta_{c,n} t),
    sampled at the step midpoints. With window=(t0, t1) us the teeth are on only inside it."""
    off = tooth_offsets(n_c, n_ref, paths, include_higher_order)
    K = len(off["paths"])
    b = np.broadcast_to(np.asarray(beta_teeth, dtype=complex), (N, 2 * K))
    t = (np.arange(N) + 0.5) * dt
    eta = np.stack([(b[:, :K] * np.exp(1j * off["comb1"][None] * t[:, None])).sum(1),
                    (b[:, K:] * np.exp(1j * off["comb2"][None] * t[:, None])).sum(1)], axis=1)
    if window is not None:
        eta[(t < window[0]) | (t > window[1])] = 0.0
    return eta


def eta_preimage(eta):
    """Pre-image x (N, 4) whose chain output is eta when eta is already in band and zero at the ramps
    (in general the chain band-limits and ramps it)."""
    return fwm.real_from_beta(np.asarray(eta, dtype=complex)) * S_ETA


def eta_chain(N, dt=DT, band=ETA_BAND, ramp_ns=RAMP_NS):
    """x (N, 4) -> real eta columns (N, 4): ramp(bandlimit(x)) / S_ETA, one core.ramp chain block."""
    return fwm.BetaChain(N, dt, band, ramp_ns, S_ETA, n_tones=2)


def _cap_combs(eta, cap=ETA_CAP):
    """Scale each comb down so that max_t |eta_c| <= cap."""
    eta = np.array(eta, dtype=complex)
    for c in range(2):
        mx = np.abs(eta[:, c]).max(initial=0.0)
        if mx > cap:
            eta[:, c] *= cap / mx
    return eta


def swap_betas(length_us, factor=1.0, paths=REPAIR_PATHS):
    """Per-tooth |beta| (fwm.tone_list(paths) order) for a full swap (rate * length = pi/2) in a window
    of length_us, times factor (as DGRAPE/fwm_stage2_scan.py)."""
    tones = fwm.tone_list(paths)
    return factor * np.array([np.pi / 2 / (fwm.tone_rate(k, 1.0, tones=tones) * length_us)
                              for k in range(len(tones))])


def start_library(tail_us=TAIL_US, dt=DT, n_c=16, n_ref=N_REF, paths=REPAIR_PATHS, cap=ETA_CAP,
                  period_us=0.6):
    """{name: eta (N, 2)} candidate training starts, each comb scaled to max |eta_c| <= cap:
      zero         -- no tones (training adds noise to it)
      tableII      -- Shirol's Table II teeth (fwm.beta_table2), constant over the tail (PReSPA as run)
      tableII_x3   -- the same, three times stronger
      bursts_f0.5  -- repeated repair: in every period, comb 1 swaps in the first half and comb 2
                      empties the transmon in the second, each at half the full-swap amplitude."""
    N = n_steps(tail_us, dt)
    K = len(tuple(paths))
    t2 = fwm.beta_table2(paths).astype(complex)
    lib = {"zero": np.zeros((N, 2), dtype=complex),
           "tableII": eta_from_teeth(t2, N, dt, n_c, n_ref, paths),
           "tableII_x3": eta_from_teeth(3 * t2, N, dt, n_c, n_ref, paths)}
    half = 0.5 * period_us
    b1 = swap_betas(half, 0.5, paths) * np.r_[np.ones(K), np.zeros(K)]
    b2 = swap_betas(half, 0.5, paths) * np.r_[np.zeros(K), np.ones(K)]
    eta = np.zeros((N, 2), dtype=complex)
    for t0 in np.arange(0.0, N * dt - 1e-9, period_us):
        eta += eta_from_teeth(b1, N, dt, n_c, n_ref, paths, window=(t0, t0 + half))
        eta += eta_from_teeth(b2, N, dt, n_c, n_ref, paths, window=(t0 + half, t0 + period_us))
    lib["bursts_f0.5"] = eta
    return {k: _cap_combs(v, cap) for k, v in lib.items()}


# ---------------------------------------------------------------------------
# Training objective
# ---------------------------------------------------------------------------

_TAIL_CACHE = {}


def _cached_tail(n_c, n_r, n_ref, hot, mixing):
    key = (n_c, n_r, n_ref, hot, mixing)
    if key not in _TAIL_CACHE:
        _TAIL_CACHE[key] = tail_model(n_c, n_r, n_ref, hot, mixing=mixing)
    return _TAIL_CACHE[key]


def _eval_piece(n_c, n_r, n_ref, hot, mixing, R0, T, w, eta, dt, mask_kind, forbid_fock_min,
                occ_weights, occ_idx, want_grad):
    """One process-input subset; module level so loky can pickle it (models cached per worker)."""
    with np.errstate(**_QUIET):
        return _eval_piece_quiet(n_c, n_r, n_ref, hot, mixing, R0, T, w, eta, dt, mask_kind,
                                 forbid_fock_min, occ_weights, occ_idx, want_grad)


def _eval_piece_quiet(n_c, n_r, n_ref, hot, mixing, R0, T, w, eta, dt, mask_kind, forbid_fock_min,
                      occ_weights, occ_idx, want_grad):
    tm = _cached_tail(n_c, n_r, n_ref, hot, mixing)
    masks = None
    if mask_kind and occ_idx:
        masks = np.array([lnp.forbidden_mask(tm.model, forbid_fock_min) if k == "forbid"
                          else lnp.transmon_mask(tm.model) for k in mask_kind])
    if not want_grad:
        R_N, hist = propagate(tm, eta, R0, dt, store=True)
        Fv = fidelity_from_outputs(R_N, T, w)
        C = lnp._occupations(hist[1:] + [R_N], masks, list(occ_idx)) if masks is not None else None
        return Fv, None, C, None
    if masks is None:
        Fv, g = fidelity_and_grad(tm, eta, R0, T, w, dt)
        return Fv, g, None, None
    out = fidelity_and_grad(tm, eta, R0, T, w, dt, occ_masks=masks, occ_weights=occ_weights,
                            occ_idx=occ_idx)
    return out["F"], out["gF"], out["C"], out["gC"]


class TailObjective:
    """cost(x), grad(x) on the tail pre-image x (N, 4); the d2 gate fixed (its output is the input).

        cost = 1 - F_pro(gate + tail vs ideal X)        one truncation, readout traced or empty
               + amp   * sum_t sum_c (|eta_c(t)| - cap)_+^2
               + deriv * sum_t |eta(t + dt) - eta(t)|^2
               [ + forbid * C_forbid + transmon * C_T ]  (off by default; time sums over the tail)
        eta columns = eta_chain(x) = ramp(bandlimit(x)) / S_ETA

    The three process inputs run as separate joblib jobs (n_jobs = 3; they never interact). Mirrors
    fwm.Stage2Objective. `inputs` replaces the d2 output (3, d, d), e.g. for a positive control.
    """

    def __init__(self, tail_us=TAIL_US, dt=DT, n_c=16, n_r=2, n_ref=N_REF, gate="X", readout=READOUT,
                 penalties=None, cap=ETA_CAP, band=ETA_BAND, ramp_ns=RAMP_NS, forbid_fock_min=10,
                 include_higher_order=True, mixing=0.0, n_jobs=3, inputs=None):
        self.N, self.dt, self.tail_us = n_steps(tail_us, dt), dt, tail_us
        self.n_c, self.n_r, self.n_ref, self.gate, self.readout = n_c, n_r, n_ref, gate, readout
        self.cap, self.band, self.ramp_ns = cap, tuple(band), ramp_ns
        self.pen = dict(TAIL_PENALTIES)
        if penalties:
            self.pen.update(penalties)
        self.hot, self.mixing, self.forbid_fock_min = include_higher_order, mixing, forbid_fock_min
        self.tm = _cached_tail(n_c, n_r, n_ref, include_higher_order, mixing)
        self.R0 = (d2_output(n_c, n_r, include_higher_order=include_higher_order) if inputs is None
                   else np.asarray(inputs, dtype=complex))
        self.T, self.w = tail_targets(n_c, n_r, gate, readout)
        self.chain = eta_chain(self.N, dt, self.band, ramp_ns)
        self.mask_kind = tuple(k for k in ("forbid", "transmon")
                               if self.pen[k] > 0 and (k != "forbid" or n_c > forbid_fock_min))
        self.occ_weights = np.array([self.pen[k] for k in self.mask_kind])
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
        """(F, dF/db4, C (n_masks,), d(sum w C)/db4) of the physical real columns b4 (N, 4)."""
        eta = fwm.beta_from_real(b4)
        tasks = []
        for b in range(self.R0.shape[0]):
            oi = (0,) if b in lnp.FORBID_IDX else ()
            tasks.append((self.n_c, self.n_r, self.n_ref, self.hot, self.mixing, self.R0[b:b + 1],
                          self.T[b:b + 1], self.w[b:b + 1], eta, self.dt, self.mask_kind,
                          self.forbid_fock_min, self.occ_weights, oi, want_grad))
        out = self._map(tasks)
        Fv = sum(o[0] for o in out)
        gF = sum(o[1] for o in out) if want_grad else None
        Cs = [o[2] for o in out if o[2] is not None]
        C = sum(Cs) if Cs else np.zeros(len(self.mask_kind))
        gCs = [o[3] for o in out if o[3] is not None]
        return Fv, gF, C, (sum(gCs) if gCs else None)

    def evaluate(self, x):
        x = np.asarray(x, dtype=float).reshape(self.N, 4)
        if self._cache_x is not None and np.array_equal(x, self._cache_x):
            return self._cache
        b4 = self.chain.to_physical(x)
        Fv, gF, C, gC = self.evaluate_physical(b4)
        if Fv > self.best["F"]:
            self.best = {"F": Fv, "x": x.copy(), "b4": b4.copy()}
        cost, g = 1.0 - Fv, -gF
        terms = {"F_pro": Fv, "infid": 1.0 - Fv, **eta_stats(fwm.beta_from_real(b4))}
        for name, fn in (("amp", lambda: fwm.modulus_penalty(b4, self.cap)),
                         ("deriv", lambda: fwm.derivative_penalty(b4))):
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
# Scoring (no gradient): baselines and held-out conditions
# ---------------------------------------------------------------------------

def score(eta, dt=DT, n_c=16, n_r=2, n_ref=N_REF, gate="X", readout=READOUT, include_higher_order=True,
          tail_higher_order=None, keep=None, dephasing=False, heating=False, stark=True, mixing=0.0,
          g_scale=1.0, eta_scale=1.0, occupations=False, forbid_fock_min=10):
    """F_pro of gate + tail under one condition. The channel options (keep, dephasing, heating) apply to
    the gate AND the tail; include_higher_order to the gate, and to the tail unless tail_higher_order
    is given; stark, mixing, g_scale, eta_scale to the tail only. An eta with 0 rows scores d2 alone.
    Returns F_pro (by `readout`), F_traced, F_empty, and <n_r>, P_e, P_f at the end averaged over
    |0_L>, |1_L> (diagonal entries, so frame independent); with occupations, C_forbid and C_transmon
    as time sums over the tail (code inputs only)."""
    eta = eta_scale * np.asarray(eta, dtype=complex).reshape(-1, 2)
    R0 = d2_output(n_c, n_r, keep, include_higher_order, dephasing, heating)
    if eta.shape[0] == 0:
        R, states = R0, []
    else:
        th = include_higher_order if tail_higher_order is None else tail_higher_order
        tm = tail_model(n_c, n_r, n_ref, th, keep, dephasing, heating, mixing, g_scale)
        out = propagate(tm, eta, R0, dt, store=occupations, stark=stark)
        R = out[0]
        states = (out[1][1:] + [R]) if occupations else []
    T_tr, w = tail_targets(n_c, n_r, gate, "traced")
    T_em, _ = tail_targets(n_c, n_r, gate, "empty")
    F_tr, F_em = fidelity_from_outputs(R, T_tr, w), fidelity_from_outputs(R, T_em, w)
    o = operators(n_c, n_r)
    mean = lambda op: float(np.mean([np.real(np.trace(op @ R[b])) for b in (0, 1)]))  # noqa: E731
    res = {"F_pro": F_tr if readout == "traced" else F_em, "F_traced": F_tr, "F_empty": F_em,
           "n_r_end": mean(o["n_r"]), "P_e_end": mean(o["Pe"]), "P_f_end": mean(o["Pf"]),
           "tail_us": eta.shape[0] * dt, **eta_stats(eta)}
    if occupations and states:
        tm_masks = np.array([lnp.forbidden_mask(tm.model, forbid_fock_min), lnp.transmon_mask(tm.model)])
        C = lnp._occupations(states, tm_masks, [0, 1])
        res.update({"C_forbid": float(C[0]), "C_transmon": float(C[1])})
    return res


def baselines(tail_us_list=(0.5, 1.0, 2.0, 3.0), n_c=16, n_r=2, readout=READOUT, dt=DT,
              n_ref=N_REF, teeth=True):
    """Rows {case, tail_us, F_pro, change_vs_d2}: d2 alone; d2 + idle tail; d2 + Shirol's Table II
    teeth held constant over the tail (PReSPA as run)."""
    F_d2 = score(np.zeros((0, 2)), dt, n_c, n_r, n_ref, readout=readout)["F_pro"]
    rows = [{"case": "d2 alone", "tail_us": 0.0, "F_pro": F_d2, "change_vs_d2": 0.0}]
    for T in tail_us_list:
        N = n_steps(T, dt)
        cases = [("d2 + idle", np.zeros((N, 2)))]
        if teeth:
            cases.append(("d2 + Table II teeth", eta_from_teeth(fwm.beta_table2(REPAIR_PATHS), N, dt,
                                                                n_c, n_ref)))
        for name, eta in cases:
            F_ = score(eta, dt, n_c, n_r, n_ref, readout=readout)["F_pro"]
            rows.append({"case": name, "tail_us": T, "F_pro": F_, "change_vs_d2": F_ - F_d2})
    return rows


# ---------------------------------------------------------------------------
# Ceilings: how much could ANY tail gain?
# ---------------------------------------------------------------------------

def ideal_repair_kraus(n_c, n_r=1, starts=None, phases=None):
    """(K_add, K_stay) of an instantaneous, ideal FWM repair: |n,g,m> -> exp(i phi_n) |n+1,g,m> for n in
    `starts` (default every even n < n_c - 1: the error states of the odd code), every other state
    left alone. K_add' K_add + K_stay' K_stay = 1 (a channel)."""
    starts = list(range(0, n_c - 1, 2)) if starts is None else list(starts)
    phases = np.zeros(len(starts)) if phases is None else np.asarray(phases, dtype=float)
    d = 3 * n_c * n_r
    K_add = np.zeros((d, d), dtype=complex)
    K_stay = np.eye(d, dtype=complex)
    for k, n in enumerate(starts):
        for m in range(n_r):
            i0, i1 = index(n, G, m, n_c, n_r), index(n + 1, G, m, n_c, n_r)
            K_add[i1, i0] = np.exp(1j * phases[k])
            K_stay[i0, i0] = 0.0
    return K_add, K_stay


def ideal_repair_fidelity(R_out, T, w, n_c, n_r=1, starts=None, n_starts=8, seed=0):
    """F_pro after the ideal repair, best phase per path (BFGS from n_starts random phases), for the
    process-output stack R_out (3, d, d). Returns {"F", "phases", "F_phase0"}."""
    starts = list(range(0, n_c - 1, 2)) if starts is None else list(starts)

    def F_of(ph):
        Ka, Ks = ideal_repair_kraus(n_c, n_r, starts, ph)
        with np.errstate(**_QUIET):
            Rr = Ka @ R_out @ Ka.conj().T + Ks @ R_out @ Ks.conj().T
        return fidelity_from_outputs(Rr, T, w)

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        r = minimize(lambda p: -F_of(p), rng.uniform(0, 2 * np.pi, len(starts)), method="BFGS")
        if best is None or r.fun < best.fun:
            best = r
    return {"F": float(-best.fun), "phases": np.mod(best.x, 2 * np.pi), "F_phase0": F_of(np.zeros(len(starts)))}


def loss_time_profile(phases, n_c=16, include_higher_order=True, keep=None, starts=None, gate="X"):
    """Which photon losses during d2 can an ideal repair at T_g undo? First-order loss branches
    K_t = sqrt(kappa_a dt) U_nj(T_g, t) a U_nj(t, 0) on the code (U_nj: no-jump evolution with every
    channel in `keep` damping it), n_r = 1. Returns arrays t (us), p (jump probability per step),
    f_repaired and f_unrepaired (conditional process fidelity of that branch against `gate`, with and
    without the ideal repair of `phases`)."""
    m = fwm.model_with_channels(n_c, 1, keep, include_higher_order)
    u = d2_pulse()
    N = u.shape[0]
    Heff = m.H_d - 0.5j * np.diag(m.Gdiag)
    steps = [expm(-1j * (Heff + np.tensordot(u[k], m.Hc, axes=1)) * DT) for k in range(N)]
    B = embedded_basis(n_c, 1)
    P = B.conj().T
    X = IDEAL_LOGICAL_U[gate]
    a = operators(n_c, 1)["a"]
    D = B.shape[0]
    with np.errstate(**_QUIET):
        fwd = [B]
        for k in range(N):
            fwd.append(steps[k] @ fwd[-1])
        bwd = [np.eye(D, dtype=complex)]
        for k in range(N - 1, -1, -1):
            bwd.append(bwd[-1] @ steps[k])
        bwd = bwd[::-1]                               # bwd[k] = U_nj(T_g, t_k)
        Ka, Ks = ideal_repair_kraus(n_c, 1, starts, phases)
        p, f_rep, f_none = np.zeros(N), np.zeros(N), np.zeros(N)
        for k in range(N):
            Kt = np.sqrt(dev.kappa_a * DT) * bwd[k] @ a @ fwd[k]
            p[k] = np.real(np.trace(Kt.conj().T @ Kt)) / 2
            f_rep[k] = sum(abs(np.trace(X.conj().T @ P @ R @ Kt)) ** 2 for R in (Ka, Ks)) / 4 / p[k]
            f_none[k] = abs(np.trace(X.conj().T @ P @ Kt)) ** 2 / 4 / p[k]
    return {"t": np.arange(N) * DT, "p": p, "f_repaired": f_rep, "f_unrepaired": f_none}


def kitten_repair_limit():
    """Conditional process fidelity of a loss right after a perfect gate, followed by the best ideal
    photon-adding repair: (|1>+|5>)/sqrt2 -> (|0>+sqrt5|4>)/sqrt6 -> (|1>+sqrt5|5>)/sqrt6, overlap
    (1 + sqrt5)/sqrt12 with |0_L>, while |1_L> = |3> is restored exactly: (1 + (1+sqrt5)/sqrt12)^2 / 4."""
    return (1 + (1 + math.sqrt(5)) / math.sqrt(12)) ** 2 / 4


def kraus_from_outputs(R_out, tol=1e-13):
    """Kraus operators (D, 2) of a channel on the code from its process outputs R_out = Phi(E_b) for
    DGRAPE.code.PROCESS_PAIRS ((0,0), (1,1), (0,1)); Phi(E_10) = Phi(E_01)'. Via the Choi matrix
    J = sum_ij |i><j| (x) Phi(|i><j|)."""
    R_out = np.asarray(R_out, dtype=complex)
    D = R_out.shape[-1]
    blocks = {pair: R_out[k] for k, pair in enumerate(PROCESS_PAIRS)}
    blocks[(1, 0)] = blocks[(0, 1)].conj().T
    J = np.zeros((2 * D, 2 * D), dtype=complex)
    for (i, j), M in blocks.items():
        J[i * D:(i + 1) * D, j * D:(j + 1) * D] = M
    lam, V = np.linalg.eigh(0.5 * (J + J.conj().T))
    return [np.sqrt(l) * v.reshape(2, D).T for l, v in zip(lam, V.T) if l > tol * max(lam.max(), 1.0)]


def optimal_recovery_fidelity(R_out, B, gate="X", iters=20000, tol=1e-12):
    """The best F_pro any recovery channel (output on the code) could reach after the channel whose
    process outputs are R_out (3, D, D), against `gate`; B (D, 2) is the embedded code basis used for
    'no recovery'. Starts from the Petz (transpose) recovery and runs the Reimpell-Werner fixed-point
    iteration J -> (1 (x) L^-1/2) J C J (1 (x) L^-1/2), L = Tr_out J C J, which increases F
    monotonically. Returns {"F_none", "F_petz", "F_opt", "iters", "tp_violation"}."""
    X = IDEAL_LOGICAL_U[gate]
    A = kraus_from_outputs(R_out)
    with np.errstate(**_QUIET):
        F_none = sum(abs(np.trace(X.conj().T @ B.conj().T @ Ai)) ** 2 for Ai in A) / 4
        S = sum(Ai @ Ai.conj().T for Ai in A) / 2                      # Phi(1/2)
        lam, V = np.linalg.eigh(S)
        keep = lam > 1e-12 * lam.max()
        Vs = V[:, keep]
        S_mh = (Vs / np.sqrt(lam[keep])) @ Vs.conj().T
        petz = [X @ Ai.conj().T @ S_mh / np.sqrt(2) for Ai in A]       # Kraus (2, D)
        d = Vs.shape[1]
        As = [Vs.conj().T @ Ai for Ai in A]                            # restrict to the output support
        bs = np.stack([((Ai @ X.conj().T).T).reshape(-1) for Ai in As])
        C = bs.conj().T @ bs                                           # F = Tr[C J] / 4
        rs = np.stack([(Rj @ Vs).reshape(-1) for Rj in petz])
        J = rs.T @ rs.conj()

        def ptrace_out(M):
            return np.einsum("acad->cd", M.reshape(2, d, 2, d))

        hist = [float(np.real(np.trace(C @ J))) / 4]
        for it in range(iters):
            M = J @ C @ J
            L = ptrace_out(M)
            w_, U_ = np.linalg.eigh(0.5 * (L + L.conj().T))
            ok = w_ > 1e-14 * w_.max()
            Li = (U_[:, ok] / np.sqrt(w_[ok])) @ U_[:, ok].conj().T
            K = np.kron(np.eye(2), Li)
            J = K @ M @ K
            J = 0.5 * (J + J.conj().T)
            hist.append(float(np.real(np.trace(C @ J))) / 4)
            if it > 50 and abs(hist[-1] - hist[-51]) < tol:
                break
        viol = float(np.abs(ptrace_out(J) - np.eye(d)).max())
    return {"F_none": float(F_none), "F_petz": hist[0], "F_opt": hist[-1], "iters": len(hist) - 1,
            "tp_violation": viol}

