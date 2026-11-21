#!/usr/bin/env python3
"""
The Fock {0,1} reference of notebooks/AQEC_compare.ipynb §6 (2026-11-21): an X gate on the cavity encoding
|0_L> = |0>, |1_L> = |1> (transmon g) on Shirol's chip, trained in the CLOSED system with est_gate's "ord" cost
(DGRAPE/train_est_gate.py --code fock01 --variant ord), then scored in the OPEN system like the EsT gates. It is the
break-even bar: a kitten-code chain (with or without a correction) is worth having only if it beats this one.

TRAINING (on the PC, outputs in ~/fwm_runs/fock01): 1.1 us and 1.5 us, seeds 42, 43, 44, tags
X_fock01_ord_closed_s<seed> and X_fock01_ord_closed_T15_s<seed>.

SELECTION RULE (fixed 2026-11-21, before any training):
    a run passes if  closed F1 >= 0.999 at n_c 24,  |F1(28) - F1(24)| < 1e-4,  and < 1 % of the pre-image is pinned
    at the box (25).
    If any 1.1 us run passes, the pick is the 1.1 us run with the highest single-gate open F_pro (Shirol's full set,
    n_c 24). Otherwise the same rule picks among the 1.5 us runs. If nothing passes, stop: no longer gate is trained
    without the user.

    python3 DGRAPE/fock01_reference.py select --runs ~/fwm_runs/fock01          -> tables/fock01_gate_runs.csv
    python3 DGRAPE/fock01_reference.py chain --pulse pulses/dgrape/u_<pick>.npy --n-jobs 4
                                                                                -> tables/aqec_fock01_ref.csv

CHAIN: N = 1..N_max Fock gates back to back, scored after each N against X^N (process fidelity on span{|0,g>, |1,g>},
readout traced), for the channel sets "decay", "all" (DGRAPE's dephasing + heating) and "shirol" (Shirol's full set) at
n_c 24, plus a shirol check at n_c 28; n_r = 1, exact (nothing drives the readout). Correction "Fock {0,1} idle" rows
score a zero-drive Fock memory of the same total time against I. Columns follow tables/aqec_seq_B1.csv (sequence X,
protocol "gates") plus the pulse path. Forward only; nothing here trains.
"""

import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

np.seterr(divide="ignore", over="ignore", invalid="ignore")

CODE = "fock01"
N_C = 24
N_C_CHECK = 28
N_R = 1
F1_BAR = 0.999
TRUNC_TOL = 1e-4
PINNED_MAX = 0.01
DURATIONS = (1.1, 1.5)                     # preferred first
CHANNELS = ("decay", "all", "shirol")
GATE_CORR = "Fock {0,1}"
IDLE_CORR = "Fock {0,1} idle"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def one_gate(pulse, channels="shirol", n_c=N_C):
    """Single-gate score dict (fwm_repeat.score_state) of the Fock pulse, open system."""
    from DGRAPE import aqec_ideal as aq
    from DGRAPE import fwm_repeat as fr
    m = aq.model(n_c, N_R, channels)
    R = fr.gate_step(m, np.asarray(pulse, dtype=float), fr.inputs(n_c, N_R, CODE))
    return fr.score_state(R, n_c, N_R, "X", CODE)


def chain(pulse, channels, n_c=N_C, n_max=20, pulse_path="", idle=True, log=None):
    """Rows for N = 1..n_max Fock gates back to back (target X^N) and, if idle, the zero-drive memory of the same
    total time (target I)."""
    from DGRAPE import aqec_ideal as aq
    from DGRAPE import fwm_repeat as fr
    from DGRAPE.code import IDEAL_LOGICAL_U
    u = np.asarray(pulse, dtype=float)
    t_gate = u.shape[0] * fr.DT
    m = aq.model(n_c, N_R, channels)
    R = fr.inputs(n_c, N_R, CODE)
    R_idle = R.copy()
    X = IDEAL_LOGICAL_U["X"]
    base = dict(sequence="X", protocol="gates", gate_N="X", n_c=n_c, n_r=N_R, channels=channels, pulse=pulse_path)
    rows = []
    for N in range(1, n_max + 1):
        R = fr.gate_step(m, u, R)
        target = np.linalg.matrix_power(X, N)
        rows.append({**base, "correction": GATE_CORR, "N": N, "time_us": N * t_gate,
                     **fr.score_state(R, n_c, N_R, target, CODE)})
        if idle:
            R_idle = aq.idle(m, R_idle, t_gate)
            rows.append({**base, "correction": IDLE_CORR, "N": N, "time_us": N * t_gate,
                         **fr.score_state(R_idle, n_c, N_R, "I", CODE)})
        if log is not None:
            log(f"{channels} n_c {n_c} N {N}: F_pro {rows[-1 - int(idle)]['F_pro']:.6f}")
    return rows


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def run_jsons(runs_dir):
    """Final <tag>.json of every Fock run (stage files and partial writes left out)."""
    out = []
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(runs_dir), "*fock01*.json"))):
        name = os.path.basename(f)
        if "_stage" in name or name.endswith(".tmp"):
            continue
        out.append(f)
    return out


def judge(info):
    """(passed, reason) under the rule of the module docstring."""
    scan = {int(k): float(v) for k, v in info["trunc_scan"].items()}
    reasons = []
    if scan[N_C] < F1_BAR:
        reasons.append(f"F1(24) {scan[N_C]:.6f} < {F1_BAR}")
    if abs(scan[N_C_CHECK] - scan[N_C]) >= TRUNC_TOL:
        reasons.append(f"|F1(28) - F1(24)| {abs(scan[N_C_CHECK] - scan[N_C]):.1e} >= {TRUNC_TOL:.0e}")
    if info["frac_preimage_pinned"] >= PINNED_MAX:
        reasons.append(f"pinned {info['frac_preimage_pinned']:.2%} >= {PINNED_MAX:.0%}")
    return not reasons, "; ".join(reasons) or "pass"


def select(runs_dir):
    """DataFrame of every run with the rule applied and the pick marked (column pick)."""
    rows = []
    for f in run_jsons(runs_dir):
        info = json.load(open(f))
        if info.get("code") != CODE:
            continue
        tag = info["tag"]
        ok, why = judge(info)
        scan = {int(k): float(v) for k, v in info["trunc_scan"].items()}
        row = {"tag": tag, "duration_us": round(float(info["duration_us"]), 3), "seed": info["seed"],
               "F1_nc20": scan.get(20, np.nan), "F1_nc24": scan[N_C], "F1_nc28": scan[N_C_CHECK],
               "frac_pinned": info["frac_preimage_pinned"], "max_abs_preimage": info["max_abs_preimage"],
               "max_eps_cav": info["max_drive_modulus"]["cav"], "max_eps_tra": info["max_drive_modulus"]["tra"],
               "nit": sum(s["nit"] for s in info["stages"]), "wall_h": info["wall_s"] / 3600,
               "passes": ok, "reason": why, "F_pro_decay": np.nan, "F_pro_shirol": np.nan}
        if ok:
            u = np.load(os.path.join(os.path.dirname(f), f"u_{tag}.npy"))
            row["F_pro_decay"] = one_gate(u, "decay")["F_pro"]
            row["F_pro_shirol"] = one_gate(u, "shirol")["F_pro"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no Fock {{0,1}} run json in {runs_dir}")
    df = df.sort_values(["duration_us", "seed"]).reset_index(drop=True)
    df["pick"] = False
    for dur in DURATIONS:
        ok = df[(df.duration_us == dur) & df.passes]
        if len(ok):
            df.loc[ok.F_pro_shirol.idxmax(), "pick"] = True
            break
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _chain_job(pulse_path, channels, n_c, n_max, out_dir):
    np.seterr(divide="ignore", over="ignore", invalid="ignore")     # loky workers: Accelerate's benign warnings
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    logf = open(os.path.join(out_dir, "logs", f"fock01_{channels}_nc{n_c}.log"), "a")
    t0 = time.time()

    def log(s):
        logf.write(f"[{time.time() - t0:8.0f} s] {s}\n")
        logf.flush()

    u = np.load(os.path.join(REPO_ROOT, pulse_path) if not os.path.isabs(pulse_path) else pulse_path)
    rows = chain(u, channels, n_c, n_max, pulse_path, log=log)
    return pd.DataFrame(rows).assign(seconds=time.time() - t0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="apply the selection rule to the trained runs")
    s.add_argument("--runs", default="~/fwm_runs/fock01")
    s.add_argument("--out", default=os.path.join(REPO_ROOT, "tables", "fock01_gate_runs.csv"))
    c = sub.add_parser("chain", help="score the picked gate, N = 1..N_max, every channel set")
    c.add_argument("--pulse", required=True, help="u_<tag>.npy, path relative to the repo root or absolute")
    c.add_argument("--n-max", type=int, default=20)
    c.add_argument("--n-jobs", type=int, default=4)
    c.add_argument("--out", default=os.path.join(REPO_ROOT, "tables", "aqec_fock01_ref.csv"))
    c.add_argument("--log-dir", default=os.path.expanduser("~/fwm_runs/fock01"))
    c.add_argument("--dry-run", action="store_true", help="N = 1 only, printed, nothing written")
    a = p.parse_args()

    if a.cmd == "select":
        df = select(a.runs)
        pd.set_option("display.width", 200)
        print(df.drop(columns=["max_abs_preimage"]).to_string(index=False))
        if not df.pick.any():
            print("NO RUN PASSES THE RULE: stop and report; do not train longer gates.")
            df.to_csv(a.out, index=False)
            sys.exit(2)
        pick = df[df.pick].iloc[0]
        df.to_csv(a.out, index=False)
        print(f"pick: {pick.tag} ({pick.duration_us} us, seed {pick.seed}), F_pro decay {pick.F_pro_decay:.6f}, "
              f"shirol {pick.F_pro_shirol:.6f} -> {a.out}")
        return

    from joblib import Parallel, delayed
    n_max = 1 if a.dry_run else a.n_max
    jobs = [(ch, N_C) for ch in CHANNELS] + [("shirol", N_C_CHECK)]
    t0 = time.time()
    parts = Parallel(n_jobs=a.n_jobs, backend="loky")(
        delayed(_chain_job)(a.pulse, ch, n_c, n_max, os.path.expanduser(a.log_dir)) for ch, n_c in jobs)
    df = pd.concat(parts, ignore_index=True)
    g = df[df.correction == GATE_CORR]
    for (ch, n_c), d in g.groupby(["channels", "n_c"]):
        print(f"{ch:6s} n_c {n_c}: N=1 {d.F_pro.iloc[0]:.6f}  N={d.N.iloc[-1]} {d.F_pro.iloc[-1]:.6f}")
    sh = g[g.channels == "shirol"].pivot(index="N", columns="n_c", values="F_pro")
    print(f"shirol n_c {N_C_CHECK} vs {N_C}: max |dF| = {np.abs(sh[N_C_CHECK] - sh[N_C]).max():.1e}  "
          f"({time.time() - t0:.0f} s)")
    if a.dry_run:
        return
    df.to_csv(a.out, index=False)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
