"""
Ablation of the extra EsT cost terms on X, seed 6, eigh pipeline, 2000 it/stage,
against section 9.2's `best2000` pulse of the same seed and budget.

    python EST/compare_new_terms.py

Runs compared (all two-stage; each scored after stage 1 and at the end):

    baseline     u_X_est_best2000              C1..C4 (the section 9.2 recipe)
    est_err      u_X_est_err_seed6             + terminal error-space infidelity
    est_dn       u_X_est_dn_seed6              + Eq. A7 photon-number mismatch C5
    est_err_dn   u_X_est_err_dn_seed6          + both
    est_all      u_X_est_all_seed6             + both + smoothness C6

Every pulse is scored the same way: the extended grape_eigh report at n_c = 20
(c1..c6, cerr) and EST/diagnostics.py's eigh propagator for Eqs. 6-8. "tail"
columns average over the last 10% of the gate, t >= 0.9 T, where the endpoint
leak of limitation 3 lives.

Outputs (the figures are drawn in est_optimization.ipynb section 1)
    tables/est_newterms_X_seed6.csv            one row per (run, stage)
    tables/est_newterms_X_seed6_traces.npz     per-time traces, keys "<run>/<stage>/<name>"
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core.grape_core import make_ops
from EST import diagnostics, grape_eigh, kitten_code
from EST.device import DT, N_T, make_hamiltonian_est
from EST.train_est_eigh import LOG_DIR, PULSE_DIR

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_DIR = os.path.join(REPO_ROOT, "tables")
CSV_PATH = os.path.join(TABLE_DIR, "est_newterms_X_seed6.csv")
TRACES_PATH = os.path.join(TABLE_DIR, "est_newterms_X_seed6_traces.npz")

GATE, N_C = "X", 20
EVAL_TRUNCS = (16, 18, 20, 22, 24, 28)
TAIL_FRAC = 0.9     # "tail" columns: t >= 0.9 T

RUNS = [  # (label, name stem)
    ("baseline", "est_best2000"),
    ("est_err", "est_err_seed6"),
    ("est_dn", "est_dn_seed6"),
    ("est_err_dn", "est_err_dn_seed6"),
    ("est_all", "est_all_seed6"),
]


def _paths(stem, stage):
    sfx = "_stage1" if stage == 1 else ""
    return (os.path.join(PULSE_DIR, f"u_{GATE}_{stem}{sfx}.npy"),
            os.path.join(PULSE_DIR, f"x_{GATE}_{stem}{sfx}.npy"),
            os.path.join(LOG_DIR, f"est_{GATE}_{stem}.json"))


def photon_mismatch_trace(u, n_c=N_C):
    """r(t) = Delta_nbar_L / (n0 + n1), the C5 integrand, on the eigh propagator."""
    H0, Hc = make_hamiltonian_est(N_T, n_c)
    A, _ = make_ops(N_T, n_c)
    words = diagnostics.propagate_states(u, H0, Hc,
                                         kitten_code.logical_basis(N_T, n_c), DT)
    _, _, r = grape_eigh.photon_number_mismatch_cost(np.asarray(words),
                                                     A.conj().T @ A)
    return r


def score(u, x, gate=GATE):
    """Scalar row + per-time traces for one pulse. `gate` lets EST/stage2_t.py reuse it on T."""
    _, report, _ = grape_eigh.build_gate_objective(
        gate, len(x), trunc_list=[N_C], extra_terms=True)
    t = report(np.asarray(x, dtype=np.float64).ravel())[N_C]
    r = diagnostics.analyze(u, gate=gate, n_c=N_C)
    rn = photon_mismatch_trace(u)
    scan = diagnostics.truncation_scan(u, gate, trunc_list=EVAL_TRUNCS)
    held = [scan[n] for n in EVAL_TRUNCS if n != N_C]
    T = len(r["leakage"])
    tail = slice(int(TAIL_FRAC * T), None)
    parked = kitten_code.parked_fidelity(gate)
    row = {
        "F1": r["F1"], "F1_train_minus_rescore": (1 - t["c1"]) - r["F1"],
        "progress": (r["F1"] - parked) / (1 - parked),
        "heldout_spread": max(held) - min(held),
        "F_ET": 1 - t["c2"], "c3": t["c3"],
        "F_err_T": 1 - t["cerr"], "c5": t["c5"], "c6": t["c6"],
        "L_mean": float(r["leakage"].mean()), "L_tail": float(r["leakage"][tail].mean()),
        "L_T": float(r["leakage"][-1]),
        "dqec_mean": float(r["delta_qec"].mean()),
        "dqec_tail": float(r["delta_qec"][tail].mean()),
        "dnbar_abs_mean": float(np.abs(rn).mean()),
        "dnbar_abs_tail": float(np.abs(rn[tail]).mean()),
        "eta0_mean": float(r["eta_0L"].mean()), "eta0_tail": float(r["eta_0L"][tail].mean()),
        "max_active_fock": int(r["max_active_fock"]),
        "max_abs_preimage": float(np.abs(x).max()),
    }
    traces = {"t": r["t_us"], "L": r["leakage"], "dqec": r["delta_qec"],
              "eta0": r["eta_0L"], "eta_avg": r["eta_avg"], "dnbar": rn}
    return row, traces


def main():
    rows, traces, missing = [], {}, []
    for label, stem in RUNS:
        for stage in (1, 2):
            upath, xpath, jpath = _paths(stem, stage)
            if not (os.path.exists(upath) and os.path.exists(xpath)):
                missing.append(upath)
                continue
            row, tr = score(np.load(upath), np.load(xpath))
            info = json.load(open(jpath)) if os.path.exists(jpath) else {}
            stage_name = "stage1" if stage == 1 else "final"
            # The JSON's constraint audit is on the FINAL pulse only.
            oob = (max(info["constraints"]["out_of_band_fraction"].values())
                   if stage == 2 and "constraints" in info else np.nan)
            row.update(run=label, stage=stage_name, oob_max=oob,
                       w_err=info.get("w_err", 0.0), w_dn=info.get("w_dn", 0.0),
                       w_smooth=info.get("w_smooth", 0.0))
            rows.append(row)
            for k, v in tr.items():
                traces[f"{label}/{stage_name}/{k}"] = np.asarray(v)
            print(f"scored {label:<11s} {stage_name:<7s} F1={row['F1']:.6f} "
                  f"F_ET={row['F_ET']:.4f} F_err(T)={row['F_err_T']:.4f} "
                  f"L(T)={row['L_T']:.4f}")
    for m in missing:
        print(f"missing {m}")
    if not rows:
        return
    df = pd.DataFrame(rows)
    front = ["run", "stage", "w_err", "w_dn", "w_smooth"]
    df = df[front + [c for c in df.columns if c not in front]]
    os.makedirs(TABLE_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    np.savez_compressed(TRACES_PATH, **traces)
    print(f"\ntable  -> {CSV_PATH}\ntraces -> {TRACES_PATH}")


if __name__ == "__main__":
    main()
