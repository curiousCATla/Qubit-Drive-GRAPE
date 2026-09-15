"""
T gate, end-of-pulse drive: does changing STAGE 2 of the `est` schedule move the
transmon drive away from t = T? Eigh pipeline, seed 1, cold two-stage starts.

    python EST/stage2_t.py commands     # print the training commands
    python EST/stage2_t.py scan         # score the w6 scan, apply the rule below
    python EST/stage2_t.py compare      # score control + configs 1-3

The problem (notebook §6, Figure 3c). On the delivered T pulses, stage 2 (w3 = 0)
finishes the gate inside the closing 48 ns ramp window: on u_T_est_best2000 about
36% of the transmon drive energy sits in the last 50 ns, with a 25 rad/us spike at
t = 584 ns, and fidelity to the target goes 0.91 -> 1.00 over that window.

Runs (EST/train_est_eigh.py STAGE2_VARIANTS; stage 1 is `est` stage 1 in all):

    control   est_best2000          stage 2 (1, 0.1, 0, 1)             notebook §9.2
    c3s2      est_c3s2_best2000     stage 2 (1, 0.1, 7, 1)
    d2        est_d2_best2000       stage 2 (1, 0.1, 0, 1 ; w6 = W6)
    c3s2_d2   est_c3s2_d2_best2000  stage 2 (1, 0.1, 7, 1 ; w6 = W6)

Scan for W6. est_d2 at w6 in SCAN_W6, 500 it/stage, seed 1. Its control is §9's
screen pulse u_T_est_seed1: the same seed, budget and pipeline, so it costs
nothing. C6 = sum ||u_{k+1} - u_k||^2 is ~500 on the delivered T pulse and the
whole stage-2 cost is ~5.5e-3, so the grid puts w6*C6 at ~0.5x, 2x and 10x of it.

Selection rule, fixed before any scan run finished. The LARGEST w6 that
  (1) reaches the gate: F1 >= F1_MIN,
  (2) moves drive out of the end: E_ramp_end (transmon drive energy fraction in
      the closing ramp window) below the control's, and
  (3) keeps transparency: F_ET >= control F_ET - ET_DROP_TOL
      (the lesson recorded in EST/smoke_new_terms.py's docstring).
If none qualifies, `scan` says so and configs 2-3 are not trained.

Outputs
    tables/est_stage2_T_seed1_scan.csv      one row per scan run (+ control)
    tables/est_stage2_T_seed1.csv           one row per (run, stage)
    tables/est_stage2_T_seed1_traces.npz    per-time traces, keys "<run>/<stage>/<name>"
The figures are drawn in est_experiments.ipynb §12.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from EST import compare_new_terms, diagnostics, kitten_code, subspace_evolution
from EST.device import DT, N_T, RAMP_NS, make_hamiltonian_est
from EST.train_est import check_constraints
from EST.train_est_eigh import LOG_DIR, PULSE_DIR

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_DIR = os.path.join(REPO_ROOT, "tables")
SCAN_CSV = os.path.join(TABLE_DIR, "est_stage2_T_seed1_scan.csv")
CSV_PATH = os.path.join(TABLE_DIR, "est_stage2_T_seed1.csv")
TRACES_PATH = os.path.join(TABLE_DIR, "est_stage2_T_seed1_traces.npz")

GATE, SEED, N_C = "T", 1, 20
SCAN_W6 = (5e-6, 2e-5, 1e-4)
SCAN_MAXITER, FULL_MAXITER = 500, 2000
F1_MIN = 0.9999
ET_DROP_TOL = 0.05
STAGE1_ATOL = 1e-6      # configs 1-3 must share the control's stage-1 pre-image

SCAN_CONTROL = "est_seed1"
FULL_RUNS = [  # (label, name stem)
    ("control", "est_best2000"),
    ("c3s2", "est_c3s2_best2000"),
    ("d2", "est_d2_best2000"),
    ("c3s2_d2", "est_c3s2_d2_best2000"),
]


def _scan_stem(w):
    return f"est_d2_seed{SEED}_smoke{SCAN_MAXITER}_w{w:g}"


def _paths(stem, stage=2):
    sfx = "_stage1" if stage == 1 else ""
    return (os.path.join(PULSE_DIR, f"u_{GATE}_{stem}{sfx}.npy"),
            os.path.join(PULSE_DIR, f"x_{GATE}_{stem}{sfx}.npy"),
            os.path.join(LOG_DIR, f"est_{GATE}_{stem}.json"))


def drive_metrics(u, dt=DT, ramp_ns=RAMP_NS):
    """Where in time the transmon drive |eps_T|(t) puts its energy."""
    N = len(u)
    n_ramp = int(round(ramp_ns * 1e-3 / dt))
    t_ns = np.arange(N) * dt * 1e3
    mag = np.hypot(u[:, 2], u[:, 3])
    e = mag ** 2
    cum = np.cumsum(e) / e.sum()
    d = np.sum(np.diff(u, axis=0) ** 2, axis=1)       # C6 integrand, (N-1,)
    frac = lambda lo, hi: float(e[(t_ns >= lo) & (t_ns < hi)].sum() / e.sum())
    T_ns = N * dt * 1e3
    row = {
        "E_ramp_start": float(e[:n_ramp].sum() / e.sum()),
        "E_0_100": frac(0, 100), "E_100_300": frac(100, 300),
        "E_300_rampend": frac(300, T_ns - ramp_ns),
        "E_ramp_end": float(e[-n_ramp:].sum() / e.sum()),
        "t50_ns": float(t_ns[np.searchsorted(cum, 0.5)]),
        "t90_ns": float(t_ns[np.searchsorted(cum, 0.9)]),
        "peak_eps_T": float(mag.max()), "t_peak_ns": float(t_ns[mag.argmax()]),
        "c6_ramp_end_share": float(d[-n_ramp:].sum() / d.sum()),
    }
    return row, {"eps_T": mag, "E_cum": cum}


def dynamics(u, gate=GATE, n_c=N_C, dt=DT, ramp_ns=RAMP_NS):
    """
    State speed v(t) -- C3's quantity, mean over the six code cardinals -- and the
    fidelity to the gate target F(t), both on the eigh propagator.
    """
    H0, Hc = make_hamiltonian_est(N_T, n_c)
    code = diagnostics.propagate_states(u, H0, Hc, kitten_code.cardinals(N_T, n_c), dt)
    ov = np.abs(np.einsum('tim,tim->tm', code[:-1].conj(), code[1:]))
    v = (2.0 * np.arccos(np.clip(ov, 0.0, 1.0)) / dt).mean(axis=1)
    tgt = kitten_code.gate_target(gate, N_T, n_c)
    F = (np.abs(np.einsum('im,tim->tm', tgt.conj(), code)) ** 2).mean(axis=1)
    n_ramp = int(round(ramp_ns * 1e-3 / dt))
    t_ns = np.arange(len(u)) * dt * 1e3
    mid = (t_ns >= 100) & (t_ns < t_ns[-n_ramp])
    row = {"v_mean": float(v.mean()),
           "v_ramp_end_over_mid": float(v[-n_ramp:].mean() / v[mid].mean()),
           "F_before_ramp_end": float(F[len(u) - n_ramp])}
    return row, {"v": v, "F": F}


def score_run(u, x):
    row, traces = compare_new_terms.score(u, x, gate=GATE)
    # On T, Delta_QEC and eta(|0_L>) are machine zero for ANY pulse (a diagonal,
    # transmon-only gate; notebook §5, Fig. 5c), so the transparency signal is L,
    # the six-cardinal eta and the map mismatch M(t) -- carried here explicitly.
    ea = np.asarray(traces["eta_avg"])
    tail = slice(int(compare_new_terms.TAIL_FRAC * len(ea)), None)
    row.update(eta_avg_mean=float(ea.mean()), eta_avg_tail=float(ea[tail].mean()),
               eta_avg_T=float(ea[-1]))
    M = np.asarray(subspace_evolution.analyze_subspace(u, gate=GATE, n_c=N_C)["mismatch"])
    row.update(M_mean=float(M.mean()), M_T=float(M[-1]))
    traces["M"] = M
    for fn in (drive_metrics, dynamics):
        r, tr = fn(u)
        row.update(r)
        traces.update(tr)
    c = check_constraints(u, DT)
    row.update(oob_transmon=c["out_of_band_fraction"]["transmon"],
               endpoint_rel_max=max(c["endpoint_rel_to_mid"]),
               cap_ok=c["amplitude_cap_satisfied"])
    return row, traces


def _info(jpath):
    if not os.path.exists(jpath):
        return {}
    with open(jpath) as fh:
        return json.load(fh)


def _print_row(label, stage, r):
    print(f"{label:<9s} {stage:<7s} F1={r['F1']:.6f} F_ET={r['F_ET']:.4f} "
          f"c3={r['c3']:.4f} L(T)={r['L_T']:.4f} eta6={r['eta_avg_mean']:.4f} "
          f"M(T)={r['M_T']:.2e} E_end={r['E_ramp_end']:.3f} "
          f"t_peak={r['t_peak_ns']:.0f}ns c6={r['c6']:.0f}")


def commands():
    w6 = selected_w6()
    run = ("OMP_NUM_THREADS=2 python3 EST/train_est_eigh.py --gate {g} --variant {v}"
           "{extra} --seed {s} --maxiter {it} --tag {tag} > logs/stage2_{g}_{v}_{tag}.log 2>&1 &")
    print("# config 1 and the w6 scan (independent of each other)")
    print(run.format(g=GATE, v="est_c3s2", extra="", s=SEED, it=FULL_MAXITER, tag="best2000"))
    for w in SCAN_W6:
        print(run.format(g=GATE, v="est_d2", extra=f" --w-smooth {w:g}", s=SEED,
                         it=SCAN_MAXITER, tag=f"seed{SEED}_smoke{SCAN_MAXITER}_w{w:g}"))
    print("# configs 2 and 3, after `scan`")
    w = f"{w6:g}" if w6 is not None else "<W6 from scan>"
    for v in ("est_d2", "est_c3s2_d2"):
        print(run.format(g=GATE, v=v, extra=f" --w-smooth {w}", s=SEED,
                         it=FULL_MAXITER, tag="best2000"))
    print("wait")


def selected_w6():
    if not os.path.exists(SCAN_CSV):
        return None
    df = pd.read_csv(SCAN_CSV)
    sel = df[df.selected.astype(bool)]
    return float(sel.w6.iloc[0]) if len(sel) else None


def scan():
    u0, x0, _ = _paths(SCAN_CONTROL)
    ctrl, _ = score_run(np.load(u0), np.load(x0))
    rows = [{"run": "control", "w6": 0.0, **ctrl}]
    _print_row("control", "final", ctrl)
    best = None
    for w in SCAN_W6:
        upath, xpath, jpath = _paths(_scan_stem(w))
        if not (os.path.exists(upath) and os.path.exists(xpath)):
            print(f"missing {upath}; skipped")
            continue
        r, _ = score_run(np.load(upath), np.load(xpath))
        info = _info(jpath)
        r["nit"] = "+".join(str(st["nit"]) for st in info.get("stages", []))
        r.update(run="est_d2", w6=w,
                 gate_ok=r["F1"] >= F1_MIN,
                 end_ok=r["E_ramp_end"] < ctrl["E_ramp_end"],
                 et_ok=r["F_ET"] >= ctrl["F_ET"] - ET_DROP_TOL)
        rows.append(r)
        _print_row(f"w6={w:g}", "final", r)
        if r["gate_ok"] and r["end_ok"] and r["et_ok"]:
            best = w                                 # SCAN_W6 ascending: keep the largest
    df = pd.DataFrame(rows)
    df["selected"] = df.w6.eq(best) if best is not None else False
    front = ["run", "w6", "selected", "gate_ok", "end_ok", "et_ok"]
    df = df[[c for c in front if c in df] + [c for c in df.columns if c not in front]]
    os.makedirs(TABLE_DIR, exist_ok=True)
    df.to_csv(SCAN_CSV, index=False)
    print(f"\ntable -> {SCAN_CSV}")
    print(f"selected w6: {best:g}" if best is not None
          else "selected w6: NONE qualifies -- do not train configs 2-3")


def compare():
    _, xs1_ctrl, _ = _paths(FULL_RUNS[0][1], stage=1)
    x_ref = np.load(xs1_ctrl) if os.path.exists(xs1_ctrl) else None
    rows, traces, missing = [], {}, []
    for label, stem in FULL_RUNS:
        info = _info(_paths(stem)[2])
        for stage in (1, 2):
            upath, xpath, _ = _paths(stem, stage)
            if not (os.path.exists(upath) and os.path.exists(xpath)):
                missing.append(upath)
                continue
            x = np.load(xpath)
            row, tr = score_run(np.load(upath), x)
            stage_name = "stage1" if stage == 1 else "final"
            st2 = info["stages"][1] if len(info.get("stages", [])) > 1 else {}
            row.update(run=label, stage=stage_name,
                       w3_stage2=st2.get("weights", [np.nan] * 3)[2],
                       w6_stage2=info.get("w_smooth", 0.0),
                       nit_stage=(info["stages"][stage - 1]["nit"]
                                  if info.get("stages") else np.nan))
            if stage == 1 and x_ref is not None:
                dx = float(np.abs(x - x_ref).max())
                row["stage1_dx_vs_control"] = dx
                if dx > STAGE1_ATOL:
                    print(f"WARNING {label}: stage-1 pre-image differs from the "
                          f"control's by {dx:.2e} > {STAGE1_ATOL:g}; the stage-2 "
                          "comparison is not from a common starting point")
            rows.append(row)
            for k, v in tr.items():
                traces[f"{label}/{stage_name}/{k}"] = np.asarray(v)
            _print_row(label, stage_name, row)
    for m in missing:
        print(f"missing {m}")
    if not rows:
        return
    df = pd.DataFrame(rows)
    front = ["run", "stage", "w3_stage2", "w6_stage2", "nit_stage"]
    df = df[front + [c for c in df.columns if c not in front]]
    os.makedirs(TABLE_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    np.savez_compressed(TRACES_PATH, **traces)
    print(f"\ntable  -> {CSV_PATH}\ntraces -> {TRACES_PATH}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["commands", "scan", "compare"])
    a = p.parse_args()
    {"commands": commands, "scan": scan, "compare": compare}[a.action]()


if __name__ == "__main__":
    main()
