"""
Ablation of the extra EsT cost terms on X, seed 0, eigh pipeline, SINGLE PHASE
at 2000 iterations.

    python EST/compare_new_terms.py

Every run is `train_est_eigh.py --stages 1`: stage 1's weights
(w1,w2,w3,w4) = (1, 0.7, 7, 1) held for the whole run, plus that variant's extra
terms. There is no stage 2, so the objective reported below is the objective
that was optimized, and each run contributes ONE row rather than a
stage-1/final pair. Stage 2 is what closes the gate in the two-stage schedule,
so read F1 and `progress` on every row before reading anything else.

Runs compared (all cold starts from the same seed-0 pre-image):

    baseline     u_X_est_sp2000_seed0          C1..C4 only
    est_err      u_X_est_err_sp2000_seed0      + terminal error-space infidelity
    est_dn       u_X_est_dn_sp2000_seed0       + Eq. A7 photon-number mismatch C5
    est_err_dn   u_X_est_err_dn_sp2000_seed0   + both
    est_all      u_X_est_all_sp2000_seed0      + both + smoothness C6

The baseline is TRAINED here. The two-stage version of this table reused
`u_X_est_best2000` (est_experiments section 9.2), which is seed 6 and two-stage,
so it is not a control for anything below.

Every pulse is scored the same way: the extended grape_eigh report at n_c = 20
(c1..c6, cerr) and EST/diagnostics.py's eigh propagator for Eqs. 6-8. "tail"
columns average over the last 10% of the gate, t >= 0.9 T, where the endpoint
leak of limitation 3 lives.

Train with:

    for v in est est_err est_dn est_err_dn est_all; do
      OMP_NUM_THREADS=2 python3 EST/train_est_eigh.py --gate X --variant $v \
          --stages 1 --seed 0 --maxiter 2000 --w-err 0.3 --w-dn 3 \
          --tag sp2000_seed0 &
    done; wait

Outputs (the figures are drawn in est_optimization.ipynb section 1)
    tables/est_newterms_X_seed0_sp.csv         one row per run
    tables/est_newterms_X_seed0_sp_traces.npz  per-time traces, keys "<run>/<name>"

The two-stage seed-6 tables this replaces stay frozen on disk as
tables/est_newterms_X_seed6.csv and tables/est_newterms_X_seed6_traces.npz.

EST/stage2_t.py imports `score` and `TAIL_FRAC` from this module for the T-gate
stage-2 study, which is still two-stage. Both must keep their signature.
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
CSV_PATH = os.path.join(TABLE_DIR, "est_newterms_X_seed0_sp.csv")
TRACES_PATH = os.path.join(TABLE_DIR, "est_newterms_X_seed0_sp_traces.npz")

GATE, N_C = "X", 20
SEED, N_STAGES, MAXITER = 0, 1, 2000
TAG = f"sp{MAXITER}_seed{SEED}"
# Three tiers, not two. `progress > 0.5` only says a pulse escaped the parked
# trap at F1 = 1/3; F1_MIN says it is a gate. Single-phase training is expected
# to land BETWEEN them -- the two-stage X finals read 0.9991-0.9995, while the
# stage-1 pulses of those same runs (which are single-phase runs by
# construction) read 0.94-0.98. Report the tier; do not tune to hide it.
F1_MIN = 0.999
EVAL_TRUNCS = (16, 18, 20, 22, 24, 28)
TAIL_FRAC = 0.9     # "tail" columns: t >= 0.9 T

RUNS = [  # (label, name stem); the baseline is a trained run, not a reused pulse
    ("baseline", f"est_{TAG}"),
    ("est_err", f"est_err_{TAG}"),
    ("est_dn", f"est_dn_{TAG}"),
    ("est_err_dn", f"est_err_dn_{TAG}"),
    ("est_all", f"est_all_{TAG}"),
]


def _require_phases(jpath):
    """Refuse to score a pulse trained with a different number of phases.

    pulses/est/ holds 600+ files and the two-stage `*_seed6` / `best2000` /
    `smoke500` artifacts all look plausible next to these. Scoring one into a
    single-phase table would be silent and wrong. `n_stages` is absent from
    JSONs written before the flag existed, so fall back to counting stages.
    """
    if not os.path.exists(jpath):
        return {}
    with open(jpath) as fh:
        info = json.load(fh)
    n = info.get("n_stages", len(info.get("stages", [])))
    if n != N_STAGES:
        raise SystemExit(
            f"{jpath} was trained with {n} stage(s); this table is the "
            f"{N_STAGES}-phase campaign. Retrain it with --stages {N_STAGES}.")
    return info


def _paths(stem):
    """(u, x, json) for one run. Single phase, so there is only the final pulse."""
    return (os.path.join(PULSE_DIR, f"u_{GATE}_{stem}.npy"),
            os.path.join(PULSE_DIR, f"x_{GATE}_{stem}.npy"),
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
        # Six-cardinal mean of Eq. 8, the quantity est_optimization.ipynb section 1
        # reports; eta_0L is the single-state reading the paper's Fig. 1f plots.
        "eta_avg_mean": float(r["eta_avg"].mean()),
        "eta_avg_tail": float(r["eta_avg"][tail].mean()),
        "max_active_fock": int(r["max_active_fock"]),
        "max_abs_preimage": float(np.abs(x).max()),
    }
    traces = {"t": r["t_us"], "L": r["leakage"], "dqec": r["delta_qec"],
              "eta0": r["eta_0L"], "eta_avg": r["eta_avg"], "dnbar": rn}
    return row, traces


def main():
    rows, traces, missing = [], {}, []
    for label, stem in RUNS:
        upath, xpath, jpath = _paths(stem)
        if not (os.path.exists(upath) and os.path.exists(xpath)):
            missing.append(upath)
            continue
        info = _require_phases(jpath)
        row, tr = score(np.load(upath), np.load(xpath))
        # Single phase, so the JSON's constraint audit IS this pulse's; under
        # two stages this column was only ever populated on the final row.
        oob = (max(info["constraints"]["out_of_band_fraction"].values())
               if "constraints" in info else np.nan)
        row.update(run=label, oob_max=oob,
                   n_stages=info.get("n_stages", N_STAGES),
                   nit=info["stages"][0]["nit"] if info.get("stages") else np.nan,
                   gate_closed=bool(row["F1"] >= F1_MIN),
                   w_err=info.get("w_err", 0.0), w_dn=info.get("w_dn", 0.0),
                   w_smooth=info.get("w_smooth", 0.0))
        rows.append(row)
        for k, v in tr.items():
            traces[f"{label}/{k}"] = np.asarray(v)
        tier = ("closed" if row["gate_closed"] else
                "OPEN, not a gate" if row["progress"] > 0.5 else "PARKED")
        print(f"scored {label:<11s} F1={row['F1']:.6f} ({tier})  "
              f"F_ET={row['F_ET']:.4f} F_err(T)={row['F_err_T']:.4f} "
              f"L(T)={row['L_T']:.4f}")
    for m in missing:
        print(f"missing {m}")
    if not rows:
        return
    df = pd.DataFrame(rows)
    parked = df.loc[df.progress <= 0.5, "run"].tolist()
    if parked:
        print(f"\nWARNING: {len(parked)}/{len(df)} runs are PARKED at F1 = 1/3: "
              f"{parked}. A parked pulse reads F_ET ~ 1 and L(T) ~ 0 for free "
              "-- it must not be compared with the rest.")
    if not df.gate_closed.all():
        names = ", ".join(f"{r.run} F1={r.F1:.4f}"
                          for r in df[~df.gate_closed].itertuples())
        print(f"\nWARNING: {(~df.gate_closed).sum()}/{len(df)} runs have "
              f"F1 < {F1_MIN} and are NOT gates: {names}.\n"
              "  Single-phase training holds w2=0.7 and w3=7 to the end; stage 2 "
              "is what finishes the gate.\n"
              "  The comparison below is still internally valid -- all runs pay "
              "the same price from the same pre-image -- but these pulses must "
              "be reported as an ablation, not as delivered gates.")
    front = ["run", "w_err", "w_dn", "w_smooth", "n_stages", "nit", "gate_closed"]
    df = df[front + [c for c in df.columns if c not in front]]
    os.makedirs(TABLE_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    np.savez_compressed(TRACES_PATH, **traces)
    print(f"\ntable  -> {CSV_PATH}\ntraces -> {TRACES_PATH}")


if __name__ == "__main__":
    main()
