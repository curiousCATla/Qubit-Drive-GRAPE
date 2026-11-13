"""
Repeated X gates with FWM tails (notebooks/FWM_tails.ipynb §9). Forward only.

Two protocols, both in the open system with the decay set (or any `keep`), on the three process inputs
E_00, E_11, E_01 at n_c (24: the closed-trained gates need it) and n_r readout levels:

    after_n_gates   N gates back to back, then ONE tail: rows for every N with and without the tail
    pairs           N (gate + tail) pairs: one row after every pair
    every_k         (B3) a tail after every k-th gate, N gates in all: after every gate a row as is ("bare") and a
                    row that ends with a tail ("tail": the scheduled one when k divides N, else a closing tail put
                    on a copy, the chain itself unchanged). k = 1 is `pairs`; k >= N_max is `after_n_gates`.

The target after N X gates is X^N (X for odd N, I for even N); an idle "gate" (a zero pulse) targets I.
`teeth_memory` is the reference for `pairs`: no gates, Shirol's Table II teeth held on for N periods, each tooth
moved by its mean pump Stark shift (stark_tuned=True, fwm_tail.teeth_stark_shifts; False = the bare carriers used
until 2026-11-09).

FRAME HAND-OFF. A gate runs in the DGRAPE frame; a tail runs in the frame that also rotates |f> at
Delta_ref (DGRAPE/fwm_tail.py docstring): rho' = V' rho V, V = exp(i Delta_ref t P_f), t counted from the
start of the tail, so V = 1 when a tail starts. At the end of a tail of length T the state goes back to
the DGRAPE frame, rho = V(T) rho' V(T)', before the next gate. Every tail's carriers restart at its own
t = 0 (one phase reference per tail). Scores do not need the hand-off (every target has the transmon in
|g>, where V acts as 1), the next gate does.

Gates are propagated with the readout present (n_r levels), so a readout photon a tail leaves behind
decays during the next gate. Tails are the ones trained after ONE gate, reused unchanged at every N.

KERR-CORRECTED DECODING (`kerr_decode`, for the memory references only). While the storage waits it picks up
the self-Kerr phase K/2 n(n-1) t (0, 3Kt, 10Kt on |1>, |3>, |5>). Gates and trained tails cancel their own,
plain waits do not. As in Gertler et al. 2021 (Methods, "Cavity Kerr effect on decoding"), the decoding undoes
it for free: C(t) = exp(-i K/2 n(n-1) t) equals, on the code's Fock levels |1>, |3>, |5>, a cavity frame shift
exp(-i 2.5 K n t) (|1> and |5> then rotate together) times a logical Z rotation by 2 K t, both virtual. It is exact only if nothing but
diagonal evolution acts between the wait and the decoding, so it is applied to the idle and teeth references
(rows decode="kerr"), never between gates.
"""

import numpy as np

from DGRAPE import fwm
from DGRAPE import fwm_tail as ft
from DGRAPE import lindblad_np as lnp
from DGRAPE.code import process_targets
from AQEC import device as dev
from DGRAPE.model import operators

DT = ft.DT


def target_of(n_gates, gate="X"):
    """The ideal logical gate after n_gates applications of `gate` (X or I)."""
    if gate == "I":
        return "I"
    if gate != "X":
        raise ValueError(f"only X and I are repeated here, not {gate!r}")
    return "X" if n_gates % 2 else "I"


def inputs(n_c, n_r=2):
    """(3, d, d) process inputs E_00, E_11, E_01, transmon g, readout empty."""
    return process_targets("X", n_c, n_r)[0]


def gate_model(n_c, n_r=2, keep=None, include_higher_order=True, dephasing=False, heating=False):
    return fwm.model_with_channels(n_c, n_r, keep, include_higher_order, dephasing, heating)


def gate_step(model, pulse, R):
    """R through one gate pulse (N, 4) in the DGRAPE frame."""
    return lnp.propagate(model, pulse, R, DT)


def frame_back(tm, tail_us, R):
    """rho = V rho' V' with V = exp(i Delta_ref tail_us P_f): the tail frame at the end of a tail of length
    tail_us back to the DGRAPE frame."""
    pf = np.real(np.diag(operators(tm.n_c, tm.n_r)["Pf"]))
    v = np.exp(1j * tm.delta_ref * tail_us * pf)
    return v[:, None] * R * v.conj()[None, :]


def tail_step(tm, eta, R, stark=True):
    """R through the tail eta (N, 2) complex, returned in the DGRAPE frame."""
    eta = np.asarray(eta, dtype=complex)
    if eta.shape[0] == 0:
        return R
    R1, _ = ft.propagate(tm, eta, R, DT, stark=stark)
    return frame_back(tm, eta.shape[0] * DT, R1)


def kerr_decode(R, n_c, n_r, tau_us, include_higher_order=True):
    """C R C' for every state in the stack R, C = exp(-i K/2 n(n-1) tau_us) on the storage: the decoding that
    undoes tau_us of idle Kerr phase (module docstring). The identity without the higher-order drift terms,
    where the drift has no Kerr."""
    if not include_higher_order or tau_us == 0:
        return R
    n = np.real(np.diag(operators(n_c, n_r)["n_a"]))
    c = np.exp(-0.5j * dev.K_a * n * (n - 1) * tau_us)
    return c[:, None] * R * c.conj()[None, :]


def score_state(R, n_c, n_r, target):
    """F_pro (readout traced), F_empty, and <n_r>, P_e, P_f averaged over |0_L>, |1_L> (as fwm_tail.score)."""
    T_tr, w = ft.tail_targets(n_c, n_r, target, "traced")
    T_em, _ = ft.tail_targets(n_c, n_r, target, "empty")
    o = operators(n_c, n_r)
    mean = lambda op: float(np.mean([np.real(np.trace(op @ R[b])) for b in (0, 1)]))  # noqa: E731
    return {"F_pro": ft.fidelity_from_outputs(R, T_tr, w), "F_empty": ft.fidelity_from_outputs(R, T_em, w),
            "n_r_end": mean(o["n_r"]), "P_e_end": mean(o["Pe"]), "P_f_end": mean(o["Pf"]),
            "trace": float(np.mean([np.real(np.trace(R[b])) for b in (0, 1)]))}


def _models(n_c, n_r, keep, include_higher_order, dephasing, heating):
    gm = gate_model(n_c, n_r, keep, include_higher_order, dephasing, heating)
    tm = ft.tail_model(n_c, n_r, keep=keep, include_higher_order=include_higher_order, dephasing=dephasing,
                       heating=heating)
    return gm, tm


def after_n_gates(pulse, eta, N_max, n_c=24, n_r=2, gate="X", keep=None, include_higher_order=True,
                  dephasing=False, heating=False, with_kerr_decode=False):
    """N = 1..N_max gates back to back, then one tail. Rows {N, tail ("none" | "tail"), decode, time_us,
    target, F_pro, ...}; time_us counts the gates and, on "tail" rows, the tail.

    with_kerr_decode (idle only, gate "I"): every row is also scored after the Kerr-corrected decoding
    (decode="kerr"), for N waits without the tail and N - 1 with it (the idle tail was trained to cancel the
    wait before it and its own length)."""
    if with_kerr_decode and gate != "I":
        raise ValueError("with_kerr_decode is for the idle reference (gate 'I') only")
    eta = np.asarray(eta, dtype=complex).reshape(-1, 2)
    gm, tm = _models(n_c, n_r, keep, include_higher_order, dephasing, heating)
    T_g, T_t = pulse.shape[0] * DT, eta.shape[0] * DT
    R, rows = inputs(n_c, n_r), []

    def add(R_end, N, tail, time_us, tgt, tau_us):
        rows.append({"N": N, "tail": tail, "decode": "none", "time_us": time_us, "target": tgt,
                     **score_state(R_end, n_c, n_r, tgt)})
        if with_kerr_decode:
            Rk = kerr_decode(R_end, n_c, n_r, tau_us, include_higher_order)
            rows.append({"N": N, "tail": tail, "decode": "kerr", "time_us": time_us, "target": tgt,
                         **score_state(Rk, n_c, n_r, tgt)})

    for N in range(1, N_max + 1):
        R = gate_step(gm, pulse, R)
        tgt = target_of(N, gate)
        add(R, N, "none", N * T_g, tgt, N * T_g)
        if eta.shape[0]:
            add(tail_step(tm, eta, R), N, "tail", N * T_g + T_t, tgt, (N - 1) * T_g)
    return rows


def pairs(pulse, eta, N_max, n_c=24, n_r=2, gate="X", keep=None, include_higher_order=True, dephasing=False,
          heating=False):
    """N = 1..N_max (gate + tail) pairs. Rows {N, time_us, target, F_pro, ...}, one after every pair.
    A zero eta is an idle wait of the same length."""
    eta = np.asarray(eta, dtype=complex).reshape(-1, 2)
    gm, tm = _models(n_c, n_r, keep, include_higher_order, dephasing, heating)
    T_pair = (pulse.shape[0] + eta.shape[0]) * DT
    R, rows = inputs(n_c, n_r), []
    for N in range(1, N_max + 1):
        R = tail_step(tm, eta, gate_step(gm, pulse, R))
        tgt = target_of(N, gate)
        rows.append({"N": N, "time_us": N * T_pair, "target": tgt, **score_state(R, n_c, n_r, tgt)})
    return rows


def top_population(R, n_c, n_r, n_top=4):
    """Storage population at Fock >= n_c - n_top, averaged over |0_L>, |1_L>: a truncation check."""
    n = np.real(np.diag(operators(n_c, n_r)["n_a"]))
    top = n >= n_c - n_top - 0.5
    return float(np.mean([np.real(np.diag(R[b]))[top].sum() for b in (0, 1)]))


def every_k(pulse, eta, k, N_max, n_c=24, n_r=2, gate="X", keep=None, include_higher_order=True, dephasing=False,
            heating=False, closing_tail=True):
    """B3: N_max gates with the tail eta after every k-th one. Rows {k, N, end, closing, n_tails, time_us, target,
    F_pro, ..., P_top}, after every gate:

        end="bare"  the state as is (n_tails = tails so far)
        end="tail"  the state ending with a tail: the scheduled one when k divides N (closing False), else one
                    closing tail on a copy (closing True; not when closing_tail=False). The chain goes on from the
                    state without it.

    time_us counts the gates and the tails the scored state has had."""
    if k < 1:
        raise ValueError(f"k must be >= 1, not {k}")
    eta = np.asarray(eta, dtype=complex).reshape(-1, 2)
    gm, tm = _models(n_c, n_r, keep, include_higher_order, dephasing, heating)
    T_g, T_t = pulse.shape[0] * DT, eta.shape[0] * DT
    R, rows, n_tails = inputs(n_c, n_r), [], 0

    def add(R_end, N, end, closing, nt, tgt):
        rows.append({"k": k, "N": N, "end": end, "closing": closing, "n_tails": nt, "time_us": N * T_g + nt * T_t,
                     "target": tgt, **score_state(R_end, n_c, n_r, tgt), "P_top": top_population(R_end, n_c, n_r)})

    for N in range(1, N_max + 1):
        R = gate_step(gm, pulse, R)
        tgt = target_of(N, gate)
        add(R, N, "bare", False, n_tails, tgt)
        if N % k == 0:
            R = tail_step(tm, eta, R)
            n_tails += 1
            add(R, N, "tail", False, n_tails, tgt)
        elif closing_tail:
            add(tail_step(tm, eta, R), N, "tail", True, n_tails + 1, tgt)
    return rows


def teeth_memory(N_max, period_us, n_c=24, n_r=2, keep=None, include_higher_order=True, dephasing=False,
                 heating=False, with_kerr_decode=False, stark_tuned=True):
    """Idle + AQEC: no gates, Shirol's Table II teeth (raw strengths, as notebook Table 2; with stark_tuned each
    tooth moved by its mean pump Stark shift, +-204 kHz, fwm_tail.teeth_stark_shifts) held on for N periods of
    period_us, one continuous waveform from t = 0 in the tail frame. Rows {N, decode, time_us, target I,
    F_pro, ...} after every period (no frame hand-off needed: nothing but the teeth acts, and the score is
    frame free). with_kerr_decode: also score each after the Kerr-corrected decoding for the whole time
    (decode="kerr"; the Kerr formula only, any Stark phase of the teeth is left in)."""
    tm = ft.tail_model(n_c, n_r, keep=keep, include_higher_order=include_higher_order, dephasing=dephasing,
                       heating=heating)
    K = ft.n_steps(period_us)
    eta = ft.eta_from_teeth(fwm.beta_table2(ft.REPAIR_PATHS), K * N_max, DT, n_c,
                            include_higher_order=include_higher_order, stark_tuned=stark_tuned)
    R, rows = inputs(n_c, n_r), []
    for N in range(1, N_max + 1):
        R, _ = ft.propagate(tm, eta[(N - 1) * K:N * K], R, DT)
        rows.append({"N": N, "decode": "none", "time_us": N * K * DT, "target": "I",
                     **score_state(R, n_c, n_r, "I")})
        if with_kerr_decode:
            Rk = kerr_decode(R, n_c, n_r, N * K * DT, include_higher_order)
            rows.append({"N": N, "decode": "kerr", "time_us": N * K * DT, "target": "I",
                         **score_state(Rk, n_c, n_r, "I")})
    return rows
