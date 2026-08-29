"""
Compare the two EsT training pipelines on the X gate.

    python EST/compare_propagators.py

    baseline : logs/est_X_est.json      + pulses/est/u_X_est.npy
               EST/grape_jax.py  -- JAX autodiff over an expm propagator
    new      : logs/est_X_est_eigh.json + pulses/est/u_X_est_eigh.npy
               EST/grape_eigh.py -- numpy eigh with a hand-derived adjoint

Both runs use the same schedule, the same --seed 0 cold start (so the same
initial pulse) and the same --maxiter, so quality is compared at an equal
iteration budget and efficiency at an equal iteration count.

Two things this script is careful about.

CROSS-SCORING. Each saved pulse is scored through BOTH pipelines, not just its
own. A pulse must not look better merely because it is graded by the propagator
it was trained against; the eigh-vs-expm columns for a single pulse are a
numerics check, and they should agree to ~1e-10.

THE PER-CALL COST DOES NOT COME FROM THESE LOGS. `seconds / nfev` is reported
because it is the honest end-to-end number, but it folds in L-BFGS-B's
line-search bookkeeping, JIT compiles and whatever else the machine was doing
for three hours. The clean per-call figure comes from
EST/bench_propagators.py -> tables/est_propagator_benchmark.csv, which is read
in here if it exists.

Outputs tables/est_propagator_comparison.csv.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from EST import diagnostics as dg
from EST.device import DT, N_T

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PULSE_DIR = os.path.join(REPO_ROOT, "pulses", "est")
LOG_DIR = os.path.join(REPO_ROOT, "logs")
TABLE_DIR = os.path.join(REPO_ROOT, "tables")

GATE = "X"
N_C = 20

# (label, pulse basename, json basename, which pipeline trained it)
RUNS = [
    ("expm_jax", "u_X_est.npy", "est_X_est.json", "EST/grape_jax.py"),
    ("eigh_numpy", "u_X_est_eigh.npy", "est_X_est_eigh.json", "EST/grape_eigh.py"),
]


def _terms(stage, n_c=N_C):
    """stage['terms'] is keyed by int n_c in memory and by str after a JSON
    round-trip; accept either."""
    t = stage["terms"]
    return t.get(n_c, t.get(str(n_c)))


def score_both_pipelines(u, gate=GATE, n_c=N_C, weights=(1.0, 0.7, 7.0, 1.0)):
    """
    Evaluate C1..C4 for one pulse through each pipeline's own cost code.

    Both take the PHYSICAL pulse, so the constraint chain is bypassed here (the
    saved u is already constrained) and only the propagation/cost math differs.
    """
    from core.grape_core import make_ops
    from EST.device import make_hamiltonian_est
    from EST import kitten_code, grape_eigh
    from EST.grape_jax import make_cost_terms
    import jax.numpy as jnp

    H0, Hc_list = make_hamiltonian_est(N_T, n_c)
    A, _ = make_ops(N_T, n_c)
    P0c = kitten_code.cardinals(N_T, n_c)
    P0e = kitten_code.error_cardinals(N_T, n_c)
    Ptg = kitten_code.gate_target(gate, N_T, n_c)

    _, _, t_eigh = grape_eigh.cost_and_grad(
        u, np.asarray(H0, dtype=complex), np.stack(Hc_list),
        np.asarray(A, dtype=complex), P0c, P0e, Ptg, DT, weights,
        want_grad=False)

    jt = make_cost_terms(
        jnp.asarray(H0, dtype=jnp.complex128),
        jnp.stack([jnp.asarray(h, dtype=jnp.complex128) for h in Hc_list]),
        jnp.asarray(A, dtype=jnp.complex128),
        jnp.asarray(P0c, dtype=jnp.complex128),
        jnp.asarray(P0e, dtype=jnp.complex128),
        jnp.asarray(Ptg, dtype=jnp.complex128), DT, constrain=None)
    t_jax = {k: float(v) for k, v in jt(jnp.asarray(u)).items()}
    return t_eigh, t_jax


def main():
    rows, pulses, infos = [], {}, {}

    for label, upath, jpath, module in RUNS:
        up = os.path.join(PULSE_DIR, upath)
        jp = os.path.join(LOG_DIR, jpath)
        if not (os.path.exists(up) and os.path.exists(jp)):
            print(f"skipping {label}: missing {up if not os.path.exists(up) else jp}")
            continue
        u = np.load(up)
        with open(jp) as fh:
            info = json.load(fh)
        pulses[label] = u
        infos[label] = info

        total_s = sum(s["seconds"] for s in info["stages"])
        met = dg.analyze(u, gate=GATE, n_c=N_C)
        scan = dg.truncation_scan(u, GATE)
        t_eigh, t_jax = score_both_pipelines(u)

        row = {
            "pipeline": label, "module": module,
            "maxiter": info["maxiter"], "seed": info["seed"],
            "N": info["N"], "trunc": "+".join(map(str, info["trunc_list"])),
        }
        for s in info["stages"]:
            i = s["stage"]
            t = _terms(s)
            row.update({
                f"s{i}_c1": t["c1"], f"s{i}_c2": t["c2"],
                f"s{i}_c3": t["c3"], f"s{i}_c4": t["c4"],
                f"s{i}_F1": 1 - t["c1"], f"s{i}_F_ET": 1 - t["c2"],
                f"s{i}_nit": s["nit"], f"s{i}_nfev": s["nfev"],
                f"s{i}_seconds": s["seconds"],
                f"s{i}_s_per_nfev": s["seconds"] / s["nfev"],
                f"s{i}_converged": "ITERATIONS" not in s["message"],
            })
        row.update({
            "total_seconds": total_s, "total_hours": total_s / 3600.0,
            # Eqs. 6-8, scored identically for both pulses by EST/diagnostics.py
            "F1_rescore_eigh": met["F1"],
            "delta_qec_mean": float(met["delta_qec"].mean()),
            "leakage_mean": float(met["leakage"].mean()),
            "eta_mean": float(met["eta"].mean()),
            "max_active_fock": met["max_active_fock"],
            "trunc_scan_spread": max(scan.values()) - min(scan.values()),
            # cross-scoring: same pulse, both cost implementations
            "xscore_c1_eigh": t_eigh["c1"], "xscore_c1_jax": t_jax["c1"],
            "xscore_c2_eigh": t_eigh["c2"], "xscore_c2_jax": t_jax["c2"],
            "xscore_max_absdiff": max(abs(t_eigh[k] - t_jax[k])
                                      for k in ("c1", "c2", "c3", "c4")),
            "perf_caveat": "s_per_nfev is end-to-end, not a clean per-call cost; "
                           "see tables/est_propagator_benchmark.csv",
        })
        rows.append(row)

    if not rows:
        raise SystemExit("no runs found to compare")

    df = pd.DataFrame(rows)

    # Clean per-call cost, if the benchmark has been run.
    bench_path = os.path.join(TABLE_DIR, "est_propagator_benchmark.csv")
    if os.path.exists(bench_path):
        b = pd.read_csv(bench_path)
        b = b[b.get("note", "ok") == "ok"]
        if len(b):
            best = b.groupby("pipeline")["min_s"].min()
            df["bench_s_per_call"] = df["pipeline"].map(best)

    os.makedirs(TABLE_DIR, exist_ok=True)
    out = os.path.join(TABLE_DIR, "est_propagator_comparison.csv")
    df.to_csv(out, index=False)

    # ---- printed summary ----
    pd.set_option("display.width", 200)
    print("\n=== quality, final stage (equal iteration budget) ===")
    cols = ["pipeline", "s2_c1", "s2_F1", "s2_c2", "s2_F_ET", "s2_c3", "s2_c4"]
    print(df[cols].to_string(index=False, float_format="%.6f"))

    print("\n=== transparency metrics (Eqs. 6-8), scored identically ===")
    cols = ["pipeline", "F1_rescore_eigh", "delta_qec_mean", "leakage_mean",
            "eta_mean", "max_active_fock", "trunc_scan_spread"]
    print(df[cols].to_string(index=False, float_format="%.6g"))

    print("\n=== efficiency ===")
    cols = ["pipeline", "s1_seconds", "s2_seconds", "total_hours",
            "s1_nfev", "s2_nfev", "s1_s_per_nfev", "s2_s_per_nfev"]
    if "bench_s_per_call" in df:
        cols.append("bench_s_per_call")
    print(df[cols].to_string(index=False, float_format="%.4f"))

    print("\n=== cross-scoring: same pulse, both cost implementations ===")
    print(df[["pipeline", "xscore_c1_eigh", "xscore_c1_jax",
              "xscore_max_absdiff"]].to_string(index=False, float_format="%.10g"))

    if len(pulses) == 2:
        a, b = pulses["expm_jax"], pulses["eigh_numpy"]
        print(f"\nwaveform distance  ||u_eigh - u_expm|| / ||u_expm|| = "
              f"{np.linalg.norm(b - a) / np.linalg.norm(a):.4f}")
        for k in ("total_hours",):
            va = df.loc[df.pipeline == "expm_jax", k].iloc[0]
            vb = df.loc[df.pipeline == "eigh_numpy", k].iloc[0]
            print(f"end-to-end speedup ({k}): {va / vb:.2f}x")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
