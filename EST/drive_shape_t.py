"""
T gate, drive shape under SINGLE-STAGE training: where does the transmon drive
put its energy, and does the C6 smoothness term even it out? Eigh pipeline,
seed 1, 2000 iterations, one phase at stage-1 weights (1, 0.7, 7, 1).

    python EST/drive_shape_t.py commands    # print the training commands
    python EST/drive_shape_t.py compare     # score control + C6 runs, apply the rule

This is the single-stage successor of EST/stage2_t.py, which stays as the
frozen two-stage record (tables/est_stage2_T_seed1*). Scoring is imported from
there, not copied.

Runs:

    control  stage 1 of est_best2000      (1, 0.7, 7, 1)             REUSED
    w<w6>    est_sm_sp2000_seed1_w<w6>    (1, 0.7, 7, 1 ; w6)        new

The control is reused, not retrained. Stage 1 of a two-stage run IS a
`--stages 1` run: same seed, weights, maxiter and cold start. The four
two-stage T runs' stage-1 pre-images are bit-identical, so the run is
deterministic, and `commands` also prints a fresh `--stages 1` control that
`compare` checks against it (REUSE_ATOL).

Selection rule, fixed before any C6 run finished. The LARGEST w6 that
  (1) drives and keeps the gate: progress > 0.5 and c1 <= C1_RATIO * control c1
      (the control-relative form of EST/smoke_new_terms.py; single stage never
      closes the gate, so an absolute F1 bar would reject everything),
  (2) moves drive out of the end: E_ramp_end below the control's, and
  (3) keeps transparency: F_ET >= control F_ET - ET_DROP_TOL.

Outputs
    tables/est_drive_T_seed1_sp.csv          one row per run
    tables/est_drive_T_seed1_sp_traces.npz   per-time traces, keys "<run>/<name>"
The figures are drawn in est_optimization.ipynb section 3.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from EST import kitten_code
from EST.stage2_t import score_run
from EST.train_est_eigh import LOG_DIR, PULSE_DIR

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_DIR = os.path.join(REPO_ROOT, "tables")
CSV_PATH = os.path.join(TABLE_DIR, "est_drive_T_seed1_sp.csv")
TRACES_PATH = os.path.join(TABLE_DIR, "est_drive_T_seed1_sp_traces.npz")

GATE, SEED, N_C = "T", 1, 20
N_STAGES, MAXITER = 1, 2000
TAG = f"sp{MAXITER}_seed{SEED}"
W6 = (5e-6, 2e-5, 1e-4)
F1_MIN = 0.9999          # "closed" on T, as in the two-stage study
C1_RATIO = 3.0
ET_DROP_TOL = 0.05
REUSE_ATOL = 1e-6

CONTROL_STEM = "est_best2000"          # its _stage1 files are the control
CHECK_STEM = f"est_{TAG}"              # fresh --stages 1 retrain of the control


def _sm_stem(w):
    return f"est_sm_{TAG}_w{w:g}"


def _paths(stem, sfx=""):
    return (os.path.join(PULSE_DIR, f"u_{GATE}_{stem}{sfx}.npy"),
            os.path.join(PULSE_DIR, f"x_{GATE}_{stem}{sfx}.npy"),
            os.path.join(LOG_DIR, f"est_{GATE}_{stem}.json"))


def _info(jpath):
    if not os.path.exists(jpath):
        return {}
    with open(jpath) as fh:
        return json.load(fh)


def _require_phases(jpath):
    """Refuse a C6 run trained with the wrong number of phases (see compare_new_terms)."""
    info = _info(jpath)
    n = info.get("n_stages", len(info.get("stages", [])))
    if info and n != N_STAGES:
        raise SystemExit(f"{jpath} was trained with {n} stage(s); this table is the "
                         f"{N_STAGES}-phase campaign. Retrain it with --stages {N_STAGES}.")
    return info


def commands():
    run = ("OMP_NUM_THREADS=2 python3 EST/train_est_eigh.py --gate {g} --variant {v}"
           " --stages 1{extra} --seed {s} --maxiter {it} --tag {tag}"
           " > logs/drive_{g}_{tag2}.log 2>&1 &")
    for w in W6:
        print(run.format(g=GATE, v="est_sm", extra=f" --w-smooth {w:g}", s=SEED,
                         it=MAXITER, tag=f"{TAG}_w{w:g}", tag2=f"sm_w{w:g}"))
    print("# reuse check: must reproduce the control's stage-1 pre-image")
    print(run.format(g=GATE, v="est", extra="", s=SEED, it=MAXITER, tag=TAG, tag2="ctrl"))
    print("wait")


def reuse_check():
    """max |x| difference between the fresh single-stage control and the reused one."""
    _, x_new, _ = _paths(CHECK_STEM)
    _, x_old, _ = _paths(CONTROL_STEM, "_stage1")
    if not os.path.exists(x_new):
        return None
    return float(np.abs(np.load(x_new) - np.load(x_old)).max())


def _tier(row):
    return ("closed" if row["F1"] >= F1_MIN else
            "open" if row["progress"] > 0.5 else "parked")


def _print_row(label, r):
    print(f"{label:<9s} F1={r['F1']:.6f} ({r['tier']}) F_ET={r['F_ET']:.4f} "
          f"c3={r['c3']:.4f} L(T)={r['L_T']:.4f} eta6={r['eta_avg_mean']:.4f} "
          f"M(T)={r['M_T']:.2e} E_end={r['E_ramp_end']:.3f} "
          f"t_peak={r['t_peak_ns']:.0f}ns c6={r['c6']:.0f}")


def compare():
    dx = reuse_check()
    if dx is None:
        print(f"reuse check: {CHECK_STEM} not trained; control is reused unchecked")
    else:
        print(f"reuse check: max|x_fresh - x_reused| = {dx:.2e} (atol {REUSE_ATOL:g})")
        if dx > REUSE_ATOL:
            raise SystemExit("the single-stage retrain does NOT reproduce the reused "
                             "control; do not use the stage-1 file as the control")

    rows, traces = [], {}
    u0, x0, j0 = _paths(CONTROL_STEM, "_stage1")
    ctrl, tr = score_run(np.load(u0), np.load(x0))
    info0 = _info(j0)
    ctrl.update(run="control", w6=0.0, source=f"stage 1 of {CONTROL_STEM} (reused)",
                nit=info0["stages"][0]["nit"], message=info0["stages"][0]["message"],
                reuse_dx=dx)
    ctrl["tier"] = _tier(ctrl)
    rows.append(ctrl)
    traces.update({f"control/{k}": np.asarray(v) for k, v in tr.items()})
    _print_row("control", ctrl)

    c1_ctrl = 1.0 - ctrl["F1"]
    best = None
    for w in W6:
        upath, xpath, jpath = _paths(_sm_stem(w))
        if not (os.path.exists(upath) and os.path.exists(xpath)):
            print(f"missing {upath}; skipped")
            continue
        info = _require_phases(jpath)
        r, tr = score_run(np.load(upath), np.load(xpath))
        label = f"w{w:g}"
        r.update(run=label, w6=w, source="new", nit=info["stages"][0]["nit"],
                 message=info["stages"][0]["message"],
                 gate_ok=bool(r["progress"] > 0.5 and (1.0 - r["F1"]) <= C1_RATIO * c1_ctrl),
                 end_ok=bool(r["E_ramp_end"] < ctrl["E_ramp_end"]),
                 et_ok=bool(r["F_ET"] >= ctrl["F_ET"] - ET_DROP_TOL))
        r["tier"] = _tier(r)
        rows.append(r)
        traces.update({f"{label}/{k}": np.asarray(v) for k, v in tr.items()})
        _print_row(label, r)
        if r["gate_ok"] and r["end_ok"] and r["et_ok"]:
            best = w                              # W6 ascending: keep the largest

    df = pd.DataFrame(rows)
    df["selected"] = df.w6.eq(best) if best is not None else False
    front = ["run", "w6", "source", "tier", "selected", "gate_ok", "end_ok", "et_ok",
             "nit", "message"]
    df = df[[c for c in front if c in df] + [c for c in df.columns if c not in front]]
    os.makedirs(TABLE_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    np.savez_compressed(TRACES_PATH, **traces)
    print(f"\ntable  -> {CSV_PATH}\ntraces -> {TRACES_PATH}")
    print(f"selected w6: {best:g}" if best is not None else "selected w6: NONE qualifies")
    if not (df.F1 >= F1_MIN).any():
        print(f"note: no run reaches F1 >= {F1_MIN}; single-stage training holds "
              "w3 = 7 to the end. Parked F1 on T is "
              f"{kitten_code.parked_fidelity(GATE):.4f}.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["commands", "compare"])
    a = p.parse_args()
    {"commands": commands, "compare": compare}[a.action]()


if __name__ == "__main__":
    main()
