"""
Forward chip map for the Gertler-comb tail (FWM_tails_ge_chips.ipynb §3). For every chip (chi_ge, chi_qr;
DGRAPE/ge_chip.derived) and tail length, a SEQUENTIAL shot (a selective pi pulse on the error states, then
the swap) with seven shape parameters is tuned by Nelder-Mead, forward only, after the EsT gate as trained
on Shirol's chip (the "transplant" proxy: its error mix carries EsT's repairability). No GRAPE.

    p = (f, a_T, xi_0, xi_2, xi_4, s_M, d_M)
      comb T on [0, f T]: one square tooth per error state, lambda = a_T pi / (2 f T), the three teeth
                          together under the chip's cap_T
      comb M on [f T, T]: teeth xi_n (n = 0, 2, 4) together under cap_M, carriers at the chip's swap lines
                          with s_M x the mean pump Stark shift and a common offset d_M (MHz)

The map optimises F_decoded (the best free storage phase theta n + phi n(n-1)/2, a frame update plus a
Kerr decoding): an OPTIMISTIC screen, so a chip that fails here is out. It also records F_pro (no free
rotation: the bar), F_tracked (theta n only), the photon-loss-only gains and the idle tail, all against the
gate alone on the same footing.

    python DGRAPE/fwm_tail_ge_chipmap.py --time                     # one evaluation, timed
    python DGRAPE/fwm_tail_ge_chipmap.py --n-jobs 12                # the full grid -> tables/fwm_tail_ge_chipmap.csv

Each job writes ~/fwm_runs/tailge_chip/map/<tag>_T<ns>.json first, so an interrupted map resumes.
Set OPENBLAS_NUM_THREADS=1 at launch.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from DGRAPE import fwm_tail as ft            # noqa: E402
from DGRAPE import fwm_tail_ge as ge         # noqa: E402
from DGRAPE import ge_chip as gc             # noqa: E402

CHI_GE_GRID = (1.12, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
CHI_QR_GRID = (1.13, 2.0, 3.0, 4.0, 5.0)
EXTRA_CHIPS = ((1.31, 2.8),)                 # Gertler's two shifts, the rest derived
TAILS_US = (0.4, 0.6, 0.8, 1.0, 1.4)
CAP_M = 0.2
GATE = "EsT"
OUT_DIR = os.path.join(os.path.expanduser("~"), "fwm_runs", "tailge_chip", "map")
TABLE = os.path.join(REPO_ROOT, "tables", "fwm_tail_ge_chipmap.csv")


T_FOCK = (0, 2, 4, 6)                         # comb-T teeth: the error states plus a phantom at 6
NEWMAN = np.pi * np.arange(len(T_FOCK)) ** 2 / 4   # tooth phases with a low peak (0.90 of the sum)
N_C_MAP = 12                                  # = n_c 16 to 1e-14 for these inputs, 2.4x faster
M_PHASES = np.pi * np.arange(3) ** 2 / 3      # comb-M tooth phases (Newman), lower peak than in phase


def comb_T(chip, a_T, t_sel, N, dt=ge.DT):
    """eps_T (N,) complex: a square selective pulse on [0, t_sel], one tooth per T_FOCK line with Newman
    phases, each of area a_T pi / 2, scaled down if its peak exceeds the chip's cap_T.

    Why this shape (closed check, FWM_tails_ge_chips.ipynb §2): each code state n = 1, 3, 5 sits between
    two teeth chi_ge away, so it sees a drive 2 lambda cos(chi_ge t), whose area vanishes when
    t_sel = 1 / chi_ge (MHz -> us). There the code states stay in g to ~1e-4 and the error states flip
    (0.96-1.0); the phantom tooth at n = 6 (|6,g> is empty) gives n = 5 its second neighbour."""
    fr = ge.tooth_frequencies(0.0, n_c=N_C_MAP, fock=T_FOCK, chip=chip)
    t = (np.arange(N) + 0.5) * dt
    lam = abs(a_T) * np.pi / (2 * t_sel)
    eps = (lam * np.exp(1j * (fr["T"][None] * t[:, None] + NEWMAN[None]))).sum(1)
    eps[t > t_sel] = 0.0
    peak = np.abs(eps).max() if N else 0.0
    if peak > chip.cap_T:
        eps *= chip.cap_T / peak
    return eps


def shot(p, chip, tail_us, cap_M=CAP_M, dt=ge.DT):
    """b4 (N, 4) of the sequential shot for parameters p (module docstring)."""
    N = ge.n_steps(tail_us, dt)
    f = float(np.clip(p[0], 0.05, 0.95))
    t_sel = f * tail_us
    k = len(ge.ERROR_FOCK)
    xs = np.abs(np.asarray(p[2:5], dtype=float)) * np.exp(1j * M_PHASES)
    M = ge.teeth(np.zeros(k), xs, N, dt, sigma=0.0, window_M=(t_sel, tail_us), chip=chip,
                 offset_M=2 * np.pi * p[6])
    _, xi = ge.eps_xi(M)
    peak = np.abs(xi).max() if N else 0.0
    if peak > cap_M:                       # the cap is on |xi_M(t)|, as the trained tails' modulus penalty
        xi *= cap_M / peak
    # Stark chirp: the pump moves every swap line |n,e,0> -> |n+1,g,1> by the SAME
    # (chi_ge + chi_qr - 2 alpha) |xi(t)|^2 (n drops out), so one phase that follows |xi(t)|^2 keeps all
    # paths resonant while the teeth beat; s_M = p[5] scales it (1 = the model's Stark shift).
    rate = (chip.chi_ge + chip.chi_qr - 2 * chip.alpha) * np.abs(xi) ** 2
    xi = xi * np.exp(1j * p[5] * (np.cumsum(rate) - 0.5 * rate) * dt)
    return ge.b4_from(comb_T(chip, p[1], t_sel, N, dt), xi)


def start_points(chip, tail_us, cap_M=CAP_M):
    """Two starts: the selective pulse at 1 / chi_ge (its zero-crosstalk length) when it fits in 60 % of
    the tail, else 60 %; and 40 % of the tail."""
    t_sel0 = min(1.0 / chip.chi_ge_MHz, 0.6 * tail_us)
    out = []
    for f in (t_sel0 / tail_us, 0.4):
        t_rem = tail_us * (1 - f)
        xs = np.array([np.pi / (2 * chip.g_m * np.sqrt(n + 1) * t_rem) for n in ge.ERROR_FOCK])
        out.append(np.r_[f, 1.0, xs, 1.0, 0.0])        # shot() scales the comb down to the cap
    return out


def _fids(R, T, w, n_c=N_C_MAP):
    return {"F_pro": ft.fidelity_from_outputs(R, T, w),
            "F_tracked": ge.frame_tracked_fidelity(R, T, w, n_c)[0],
            "F_decoded": ge.decoded_fidelity(R, T, w, n_c)[0]}


def run_job(pair, tail_us, gate=GATE, cap_M=CAP_M, max_evals=160, out_dir=OUT_DIR):
    """One (chip, length): tune the shot, then score it; returns the row (also written as json)."""
    from scipy.optimize import minimize
    chip = gc.derived(*pair)
    path = os.path.join(out_dir, f"{chip.tag}_T{int(round(1000 * tail_us)):04d}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    t0 = time.time()
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        R0, cut = ge.tail_input(gate, n_c=N_C_MAP)
        T_, w = ge.tail_targets(n_c=N_C_MAP)
        tm = ge.tail_model(N_C_MAP, chip=chip)
        alone = _fids(R0, T_, w)
        idle = _fids(ge.propagate(tm, np.zeros((ge.n_steps(tail_us), 4)), R0)[0], T_, w)

        def neg(p):
            R, _ = ge.propagate(tm, shot(p, chip, tail_us, cap_M), R0)
            return -ge.decoded_fidelity(R, T_, w, N_C_MAP)[0]

        best = None
        for p0 in start_points(chip, tail_us, cap_M):
            simplex = [p0] + [p0 + np.eye(7)[i] * s for i, s in
                              enumerate((0.1, 0.2, 0.03, 0.03, 0.03, 0.3, 0.2))]
            r = minimize(neg, p0, method="Nelder-Mead",
                         options={"maxfev": max_evals // 2, "initial_simplex": np.array(simplex),
                                  "xatol": 1e-4, "fatol": 1e-6})
            if best is None or r.fun < best.fun:
                best = r
        b4 = shot(best.x, chip, tail_us, cap_M)
        R, _ = ge.propagate(tm, b4, R0)
        tail = _fids(R, T_, w)
        lo_kw = dict(n_c=N_C_MAP, chip=chip, keep={"a"}, decode=True)
        lo_alone = ge.score(np.zeros((0, 4)), gate, **lo_kw)
        lo_tail = ge.score(b4, gate, **lo_kw)
        o = ge.ge_operators(N_C_MAP, 2, 2)
        mean = lambda op: float(np.mean([np.real(np.trace(op @ R[b])) for b in (0, 1)]))  # noqa: E731
    t_sel, t_swap = gc.repair_time_estimate(chip)
    row = {**chip.table_row(), "tag": chip.tag, "gate": gate, "tail_us": tail_us, "cap_M": cap_M,
           "est_t_sel_us": t_sel, "est_t_swap2_us": t_swap,
           **{f"alone_{k}": v for k, v in alone.items()}, **{f"idle_{k}": v for k, v in idle.items()},
           **{f"tail_{k}": v for k, v in tail.items()},
           **{f"gain_{k}": tail[k] - alone[k] for k in alone},
           **{f"idle_gain_{k}": idle[k] - alone[k] for k in alone},
           "loss_only_gain_decoded": lo_tail["F_decoded"] - lo_alone["F_decoded"],
           "loss_only_gain_pro": lo_tail["F_pro"] - lo_alone["F_pro"],
           "n_r_end": mean(o["n_r"]), "P_e_end": mean(o["Pe"]), "trace_cut": cut,
           "p_best": [float(v) for v in best.x], "n_evals": int(best.nfev), "seconds": time.time() - t0}
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(row, fh, indent=1)
    return row


def grid():
    pairs = [(a, b) for a in CHI_GE_GRID for b in CHI_QR_GRID] + list(EXTRA_CHIPS)
    return [(pr, T) for pr in pairs for T in TAILS_US]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-jobs", type=int, default=12)
    p.add_argument("--max-evals", type=int, default=160)
    p.add_argument("--time", action="store_true", help="time one evaluation on chip (3, 5) at 0.8 us and exit")
    p.add_argument("--out", default=TABLE)
    a = p.parse_args()
    if a.time:
        chip = gc.derived(3.0, 5.0)
        R0, _ = ge.tail_input(GATE, n_c=N_C_MAP)
        T_, w = ge.tail_targets(n_c=N_C_MAP)
        tm = ge.tail_model(N_C_MAP, chip=chip)
        b4 = shot(start_points(chip, 0.8)[0], chip, 0.8)
        t0 = time.time()
        R, _ = ge.propagate(tm, b4, R0)
        t1 = time.time()
        f = _fids(R, T_, w)
        print(f"propagate {t1 - t0:.2f} s, score {time.time() - t1:.2f} s; start shot: {f}")
        print(f"alone: {_fids(R0, T_, w)}")
        return
    import pandas as pd
    from joblib import Parallel, delayed
    jobs = grid()
    print(f"{len(jobs)} jobs on {a.n_jobs} workers", flush=True)
    t0 = time.time()
    rows = Parallel(n_jobs=a.n_jobs, backend="loky", verbose=5)(
        delayed(run_job)(pr, T, max_evals=a.max_evals) for pr, T in jobs)
    df = pd.DataFrame(rows).drop(columns=["p_best"])
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(df)} rows) in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
