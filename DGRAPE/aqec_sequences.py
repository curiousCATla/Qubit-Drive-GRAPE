"""
Repeated EsT gates of several kinds (X, H, alternating X/H) with one correction: the ideal Cornell AQEC step
(DGRAPE/aqec_ideal.py), the open-trained AQEC step (DGRAPE/aqec_pulse.open_step, notebooks/AQEC_compare.ipynb §6) or the
FWM tail (DGRAPE/fwm_repeat.tail_step). Forward only. Notebook: notebooks/AQEC_compare.ipynb §6.1.

    b1   N gates back to back, then ONE correction (on a copy): rows "none" (gates only) and one per correction
    b2   N (gate + correction) pairs, scored after every pair

A sequence is a string of gate letters that repeats: "X" = X, X, X, ...; "H" = H, H, ...; "XH" = X, H, X, H, ...
The target after N gates is the product of the ideal logical gates, U_N ... U_1 (DGRAPE.code.IDEAL_LOGICAL_U), scored
with fwm_repeat.score_state (process fidelity on the code, transmon in g, readout traced).

Nothing here is retrained: the AQEC step and the FWM tail were trained after the X gate and are reused unchanged.

Channels: "decay" (the decay set, as every earlier B1/B2 table; reproduces them bit for bit) or "shirol" (Shirol's full
set, DGRAPE.model.shirol_collapse_ops) in the gate, the step and the tail alike. The AQEC chains run at n_r = 1 (exact:
neither the gates nor the step ever fill the readout); the FWM chains at n_r = 2.
"""

import os

import numpy as np

from DGRAPE import aqec_ideal as aq
from DGRAPE import aqec_pulse as ap
from DGRAPE import fwm_repeat as fr
from DGRAPE import fwm_tail as ft
from DGRAPE.code import IDEAL_LOGICAL_U

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = fr.DT                                                          # the gates' 2 ns grid
GATES = {"X": aq.EST_PULSE,                                         # EsT X, seed 24 of C_err, 1.1 us
         "H": "pulses/dgrape/u_H_est_err_closed_T2_s17.npy"}        # EsT H, seed 17 of C_err, 2 us
OPEN_STEP = "aqec_open_T15_w"                                       # the best open-trained step, 1.5 us (+ 1.2 us)
FWM_TAIL = "X_tail_estg_T08_shirol_d01_s8"                          # the seed-8 FWM tail, 0.8 us
CORRECTIONS = ("ideal AQEC", "open step", "FWM tail")
TAIL_NOISE = {"decay": "dgrape", "shirol": "shirol"}                # aq.model channels -> fwm_tail noise


def _load(path):
    return np.load(os.path.join(REPO_ROOT, path))


def gate_pulses():
    return {g: _load(p) for g, p in GATES.items()}


def gate_at(seq, k):
    """The letter of the k-th gate (k = 1, 2, ...) of the repeating sequence seq."""
    return seq[(k - 1) % len(seq)]


def target_after(seq, N):
    """U_N ... U_1, the ideal logical gate after N gates of seq."""
    U = np.eye(2, dtype=complex)
    for k in range(1, N + 1):
        U = IDEAL_LOGICAL_U[gate_at(seq, k)] @ U
    return U


class Corrections:
    """The correction steps on one model (n_c, n_r, channels): name -> (function R -> R, duration in us)."""

    def __init__(self, n_c, n_r, channels, names):
        self.m = aq.model(n_c, n_r, channels)
        self.steps = {}
        if "ideal AQEC" in names:
            K = aq.kraus(n_c, n_r)
            self.steps["ideal AQEC"] = (lambda R: aq.aqec_step(self.m, R, K, "forgive"), aq.TAU1 + aq.TAU2)
        if "open step" in names:
            u = _load(f"pulses/dgrape/u_{OPEN_STEP}.npy")
            self.steps["open step"] = (lambda R: ap.open_step(self.m, R, u), u.shape[0] * ap.DT + aq.TAU2)
        if "FWM tail" in names:
            eta = _load(f"pulses/dgrape/eta_{FWM_TAIL}.npy")
            tm = ft.tail_model(n_c, n_r, noise=TAIL_NOISE[channels])
            self.steps["FWM tail"] = (lambda R: fr.tail_step(tm, eta, R), eta.shape[0] * DT)
        unknown = set(names) - set(self.steps)
        if unknown:
            raise ValueError(f"unknown corrections {sorted(unknown)}; choose from {CORRECTIONS}")


def _row(seq, protocol, correction, N, time_us, R, n_c, n_r, channels):
    return {"sequence": seq, "protocol": protocol, "correction": correction, "N": N, "time_us": time_us,
            "gate_N": gate_at(seq, N), "n_c": n_c, "n_r": n_r, "channels": channels,
            **fr.score_state(R, n_c, n_r, target_after(seq, N))}


def b1(seq, corrections, N_max, n_c=24, n_r=1, channels="shirol", log=None):
    """N = 1..N_max gates of seq back to back; after each N the state as is (correction "none") and, on a copy,
    followed by each correction."""
    cs = Corrections(n_c, n_r, channels, corrections)
    pulses = gate_pulses()
    R, t, rows = fr.inputs(n_c, n_r), 0.0, []
    for N in range(1, N_max + 1):
        u = pulses[gate_at(seq, N)]
        R = fr.gate_step(cs.m, u, R)
        t += u.shape[0] * DT
        rows.append(_row(seq, "B1", "none", N, t, R, n_c, n_r, channels))
        for name in corrections:
            fn, T = cs.steps[name]
            rows.append(_row(seq, "B1", name, N, t + T, fn(R), n_c, n_r, channels))
        if log:
            log(f"B1 {seq} N={N}")
    return rows


def b2(seq, correction, N_max, n_c=24, n_r=1, channels="shirol", log=None):
    """N = 1..N_max (gate + correction) pairs of seq, scored after every pair."""
    cs = Corrections(n_c, n_r, channels, (correction,))
    fn, T = cs.steps[correction]
    pulses = gate_pulses()
    R, t, rows = fr.inputs(n_c, n_r), 0.0, []
    for N in range(1, N_max + 1):
        u = pulses[gate_at(seq, N)]
        R = fn(fr.gate_step(cs.m, u, R))
        t += u.shape[0] * DT + T
        rows.append(_row(seq, "B2", correction, N, t, R, n_c, n_r, channels))
        if log:
            log(f"B2 {seq} {correction} N={N}")
    return rows
