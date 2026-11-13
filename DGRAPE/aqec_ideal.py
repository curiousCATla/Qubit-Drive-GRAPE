"""
An idealised version of the Cornell two-step active QEC (Roy et al., arXiv:2603.15356, App. C 2) on Shirol's chip and
the odd kitten code, after the EsT X gate of notebooks/FWM_tails.ipynb. Forward only. Notebook:
notebooks/AQEC_compare.ipynb.

THEIR STEP (on their chip, even code): a 1 us GRAPE pulse U with
    U |code word, g> = |code word, g>          U |error word, g> = |matching code word, e>,
then a 1.2 us transmon readout and a pi pulse when the readout says e: in effect an unconditional transmon reset. U
also pre-twists the state so that the storage Kerr phase (both branches) and the chi phase (e branch) of the readout
window cancel.

OUR IDEAL STEP (aqec_step), on the process stack R (3, d, d) after the gate(s):
    0. leftover transmon excitation: mode "forgive" traces the transmon out and puts it in g (the step ends in a reset
       anyway); mode "strict" projects onto g (leftover excitation counts as failure)
    1. tau1 = 1 us lossy idle (zero drive), then the drift phase is undone, D = exp(+i H_d tau1): "the pulse ends with
       the ideal unitary", so the no-jump branch carries no Kerr phase. A jump in this window still leaves a phase.
    2. the ideal map, Kraus K_g = (1 - |E0><E0| - |E1><E1|) (x) |g><g|,  K_e = (|L0><E0| + |L1><E1|) (x) |e><g|
       (code and error words from AQEC.odd_kitten). It undoes one photon loss exactly, including the sqrt(5)
       distortion an FWM photon-adding repair cannot remove (fwm_tail.kitten_repair_limit = 0.935).
    3. pre-twist D = exp(+i H_d tau2): H_d is diagonal, so this one operator cancels the Kerr phase on the g branch and
       Kerr + chi_ge on the e branch over the window that follows
    4. tau2 = 1.2 us lossy idle (the readout window). A loss or a transmon decay here leaves a residual phase.
    5. ideal reset: transmon traced out and put in g.

Model: DGRAPE's (fwm.model_with_channels), decay set unless stated, higher-order terms on. The readout mode is never
populated by a gate or by this step, so n_r = 1 is exact (the default here).

Scores: fwm_repeat.score_state, F_pro against X^N with the transmon in g, readout traced.
"""

import numpy as np

from AQEC.odd_kitten import error_basis, logical_basis
from DGRAPE import fwm
from DGRAPE import fwm_repeat as fr
from DGRAPE import lindblad_np as lnp
from DGRAPE.model import N_Q, G, E

DT = fr.DT
TAU1 = 1.0          # us, their GRAPE pulse
TAU2 = 1.2          # us, their readout + conditional pi pulse
N_C = 24            # the closed-trained EsT gate needs n_c 24 (notebooks/FWM_tails.ipynb §3)
EST_PULSE = "pulses/dgrape/u_X_est_err_closed_s24.npy"

# name: (step-0 mode, tau1, tau2)
VARIANTS = {
    "ideal AQEC": ("forgive", TAU1, TAU2),
    "ideal AQEC, strict": ("strict", TAU1, TAU2),
    "no readout window": ("forgive", TAU1, 0.0),
    "instant": ("forgive", 0.0, 0.0),
}

_QUIET = dict(divide="ignore", over="ignore", invalid="ignore")


def model(n_c=N_C, n_r=1, channels="decay"):
    """DGRAPE model for the gate and the idles: channels "decay" (photon loss, readout and transmon decay) or "all"
    (plus dephasing and heating, DGRAPE's opt-in set)."""
    if channels not in ("decay", "all"):
        raise ValueError(f"channels must be 'decay' or 'all', not {channels!r}")
    on = channels == "all"
    return fwm.model_with_channels(n_c, n_r, None, True, dephasing=on, heating=on)


def _lift(op_a, op_q, n_r):
    return np.kron(np.kron(op_a, op_q), np.eye(n_r))


def kraus(n_c=N_C, n_r=1):
    """(K_g, K_e) of the ideal map on storage (x) transmon (x) readout. K_g'K_g + K_e'K_e = the projector on the
    transmon-g subspace."""
    B, Be = logical_basis(n_c), error_basis(n_c)
    PE = Be @ Be.conj().T
    Pq = np.zeros((N_Q, N_Q)); Pq[G, G] = 1
    Sq = np.zeros((N_Q, N_Q)); Sq[E, G] = 1
    Kg = _lift(np.eye(n_c) - PE, Pq, n_r)
    Ke = _lift(B @ Be.conj().T, Sq, n_r)
    return Kg.astype(complex), Ke.astype(complex)


def _split(R, n_c, n_r):
    b = R.shape[0]
    return R.reshape(b, n_c, N_Q, n_r, n_c, N_Q, n_r)


def forgive(R, n_c, n_r):
    """Trace the transmon out of every state in the stack and put it in g (an ideal reset)."""
    S = _split(R, n_c, n_r)
    red = np.einsum("bnjmpjq->bnmpq", S)                      # (b, n, m, n', m')
    out = np.zeros_like(S)
    out[:, :, G, :, :, G, :] = red
    return out.reshape(R.shape)


def strict(R, n_c, n_r):
    """Project every state in the stack onto transmon g (no renormalisation: the rest is lost)."""
    S = _split(R, n_c, n_r)
    out = np.zeros_like(S)
    out[:, :, G, :, :, G, :] = S[:, :, G, :, :, G, :]
    return out.reshape(R.shape)


def phase(m, tau):
    """Diagonal of D = exp(+i H_d tau): applied as D R D', it undoes tau of drift phase."""
    return np.exp(1j * np.real(np.diag(m.H_d)) * tau)


def rotate(R, v):
    return v[None, :, None] * R * v.conj()[None, None, :]


def idle(m, R, tau):
    """tau of lossy evolution with no drive (exact: one Taylor-split step of the static Lindbladian)."""
    if tau == 0:
        return R
    u = np.zeros((1, 4))
    return lnp.propagate(m, u, R, tau)


def apply_map(R, K):
    Kg, Ke = K
    with np.errstate(**_QUIET):
        return (np.einsum("ij,bjk,lk->bil", Kg, R, Kg.conj()) + np.einsum("ij,bjk,lk->bil", Ke, R, Ke.conj()))


def aqec_step(m, R, K, mode="forgive", tau1=TAU1, tau2=TAU2, stages=False):
    """One ideal AQEC step on the stack R (module docstring). With stages=True also returns the intermediate stacks
    {"step 0", "step 1", "map", "end"}."""
    n_c, n_r = m.n_c, m.n_r
    R0 = forgive(R, n_c, n_r) if mode == "forgive" else strict(R, n_c, n_r)
    R1 = rotate(idle(m, R0, tau1), phase(m, tau1))
    R2 = apply_map(R1, K)
    R3 = idle(m, rotate(R2, phase(m, tau2)), tau2)
    R4 = forgive(R3, n_c, n_r)
    if stages:
        return R4, {"step 0": R0, "step 1": R1, "map": R2, "end": R4}
    return R4


def memory_idle(m, R, tau):
    """Same-time memory reference: tau of lossy idle, then the storage Kerr phase undone for free (Gertler decoding,
    fwm_repeat.kerr_decode). No reset."""
    return fr.kerr_decode(idle(m, R, tau), m.n_c, m.n_r, tau)


def _score(R, m, N):
    return fr.score_state(R, m.n_c, m.n_r, fr.target_of(N))


def populations(R, m):
    """Mean over |0_L>, |1_L> of the weight on the code words, on the error words, and in transmon e / f."""
    n_c, n_r = m.n_c, m.n_r
    B, Be = logical_basis(n_c), error_basis(n_c)
    Pc = _lift(B @ B.conj().T, np.eye(N_Q), n_r)
    Pe_w = _lift(Be @ Be.conj().T, np.eye(N_Q), n_r)
    S = _split(R, n_c, n_r)
    mean = lambda X: float(np.mean([np.real(np.trace(X[b])) for b in (0, 1)]))  # noqa: E731
    with np.errstate(**_QUIET):
        return {"P_code": mean(Pc @ R), "P_error": mean(Pe_w @ R),
                "P_e": float(np.mean([np.real(np.einsum("nmnm->", S[b, :, E, :, :, E, :])) for b in (0, 1)])),
                "P_f": float(np.mean([np.real(np.einsum("nmnm->", S[b, :, 2, :, :, 2, :])) for b in (0, 1)]))}


def single_stages(pulse, n_c=N_C, n_r=1, channels="decay"):
    """N = 1, stage by stage: rows {stage, time_us, F_pro, P_code, P_error, P_e, P_f}. The stage after the map is also
    scored with an instant reset (the transmon flip is the record that a correction happened)."""
    m = model(n_c, n_r, channels)
    K = kraus(n_c, n_r)
    T_g = pulse.shape[0] * DT
    Rg = fr.gate_step(m, pulse, fr.inputs(n_c, n_r))
    _, st = aqec_step(m, Rg, K, "forgive", stages=True)
    rows = []
    for stage, R, t, reset in (("after the gate", Rg, T_g, False),
                               ("step 0: leftover transmon excitation reset", st["step 0"], T_g, False),
                               ("step 1: 1 us lossy idle, drift phase undone", st["step 1"], T_g + TAU1, False),
                               ("step 2a: ideal map (scored with an instant reset)", st["map"], T_g + TAU1, True),
                               ("step 2b: 1.2 us readout window + reset", st["end"], T_g + TAU1 + TAU2, False)):
        Rs = forgive(R, n_c, n_r) if reset else R
        rows.append({"channels": channels, "stage": stage, "time_us": t, "F_pro": _score(Rs, m, 1)["F_pro"],
                     **populations(R, m)})
    return rows


def single_summary(pulse, n_c=N_C, n_r=1, channels="decay"):
    """N = 1: one row per protocol (gate alone, gate + same-time idle, every VARIANT)."""
    m = model(n_c, n_r, channels)
    K = kraus(n_c, n_r)
    T_g = pulse.shape[0] * DT
    Rg = fr.gate_step(m, pulse, fr.inputs(n_c, n_r))
    rows = [{"channels": channels, "protocol": "gate alone", "time_us": T_g, **_score(Rg, m, 1)},
            {"channels": channels, "protocol": "gate + 2.2 us idle (Kerr decoded)", "time_us": T_g + TAU1 + TAU2,
             **_score(memory_idle(m, Rg, TAU1 + TAU2), m, 1)}]
    for name, (mode, t1, t2) in VARIANTS.items():
        rows.append({"channels": channels, "protocol": "gate + " + name, "time_us": T_g + t1 + t2,
                     **_score(aqec_step(m, Rg, K, mode, t1, t2), m, 1)})
    return rows


def after_n_gates(pulse, N_max, n_c=N_C, n_r=1, channels="decay", log=None):
    """B1: N gates back to back, then ONE AQEC step (every VARIANT) or a same-time idle. Rows {N, protocol, time_us,
    F_pro, ...}."""
    m = model(n_c, n_r, channels)
    K = kraus(n_c, n_r)
    T_g = pulse.shape[0] * DT
    R, rows = fr.inputs(n_c, n_r), []
    for N in range(1, N_max + 1):
        R = fr.gate_step(m, pulse, R)
        rows.append({"N": N, "protocol": "gates only", "time_us": N * T_g, **_score(R, m, N)})
        rows.append({"N": N, "protocol": "gates + 2.2 us idle (Kerr decoded)", "time_us": N * T_g + TAU1 + TAU2,
                     **_score(memory_idle(m, R, TAU1 + TAU2), m, N)})
        for name, (mode, t1, t2) in VARIANTS.items():
            rows.append({"N": N, "protocol": "gates + " + name, "time_us": N * T_g + t1 + t2,
                         **_score(aqec_step(m, R, K, mode, t1, t2), m, N)})
        if log:
            log(f"B1 N={N}")
    return rows


def pairs(pulse, N_max, n_c=N_C, n_r=1, channels="decay", log=None):
    """B2: N (gate + AQEC) pairs, one chain per VARIANT. Rows {N, protocol, time_us, F_pro, ...}."""
    m = model(n_c, n_r, channels)
    K = kraus(n_c, n_r)
    T_g = pulse.shape[0] * DT
    names = list(VARIANTS)
    R = np.concatenate([fr.inputs(n_c, n_r)] * len(names))      # one block of 3 per chain, gates run together
    rows = []
    for N in range(1, N_max + 1):
        R = fr.gate_step(m, pulse, R)
        for i, name in enumerate(names):
            mode, t1, t2 = VARIANTS[name]
            blk = slice(3 * i, 3 * i + 3)
            R[blk] = aqec_step(m, R[blk], K, mode, t1, t2)
            rows.append({"N": N, "protocol": "(gate + " + name + ") pairs", "time_us": N * (T_g + t1 + t2),
                         **_score(R[blk], m, N)})
        if log:
            log(f"B2 N={N}")
    return rows
