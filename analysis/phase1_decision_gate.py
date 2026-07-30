#!/usr/bin/env python3
"""
phase1_decision_gate.py — post–Phase 1 decision-gate numbers.

For each logical gate I/X/Y/Z/H/T, report the Phase-0 measuring stick
on the current main pulses (and any pulses/phase1/u_<gate>.npy if present):

  - F_pedersen  : Pedersen average gate fidelity (THE reported number)
  - L1          : average logical-subspace leakage
  - F_coh       : coherent process objective (optimizer metric only;
                  not the decision threshold)

Decision rule (Phase 1 deliverable 4):
  retrain warm-started into pulses/phase1/ ONLY if a relative-phase defect
  remains. The bar used here is F_pedersen < 0.99 at n_c=24 (same bar as
  analysis/retrain_phase_coherent_gates.F_AVG_GATE_TARGET and
  validate_logical_gates tier3 warnings).

Nothing here feeds any training objective.

Usage:
    python3 analysis/phase1_decision_gate.py
"""

import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.grape_core import make_hamiltonian, coherent_fidelity_multi_state
from core.cat_code import get_coherent_state_pairs
from core.propagator import (
    full_propagator,
    logical_basis,
    logical_block,
    pedersen_gate_fidelity,
    leakage_L1,
)
from validation.validate_logical_gates import IDEAL_LOGICAL_U, N_T, DT, ALPHA

PULSE_DIR = os.path.join(REPO_ROOT, "pulses")
PHASE1_DIR = os.path.join(PULSE_DIR, "phase1")
TABLE_DIR = os.path.join(REPO_ROOT, "tables")

GATES = ["I", "X", "Y", "Z", "H", "T"]
# Training truncations + the headline evaluation point.
TRUNC_LIST = [22, 24, 26]
HEADLINE_N_C = 24
DECISION_THRESHOLD = 0.99
EXPECTED_SHAPE = (550, 4)


def _load(path):
    u = np.load(path)
    assert u.shape == EXPECTED_SHAPE, (
        f"{path} has shape {u.shape}, expected {EXPECTED_SHAPE}"
    )
    return u


def discover_pulse_sets():
    """
    (set_name, path_template_fn) for every pulse set we should score.
    Main is always present; phase1 only if the directory has files.
    """
    sets = []
    sets.append((
        "main",
        lambda gate: os.path.join(PULSE_DIR, f"u_{gate}_main.npy"),
    ))
    if os.path.isdir(PHASE1_DIR):
        phase1_paths = {
            g: os.path.join(PHASE1_DIR, f"u_{g}.npy")
            for g in GATES
            if os.path.isfile(os.path.join(PHASE1_DIR, f"u_{g}.npy"))
        }
        if phase1_paths:
            sets.append((
                "phase1",
                lambda gate, _p=phase1_paths: _p.get(gate),
            ))
    return sets


def score_pulse(gate, u, n_c, H0, Hc, B):
    U_full = full_propagator(u, H0, Hc, DT)
    U_log = logical_block(U_full, B)
    U_ideal = IDEAL_LOGICAL_U[gate]
    F_ped = float(pedersen_gate_fidelity(U_ideal, U_log))
    L1 = float(leakage_L1(U_log))

    # Optimizer objective on the canonical Phase-1 targets (process fidelity).
    psi_i, psi_f = get_coherent_state_pairs(gate, n_c=n_c, n_t=N_T, alpha=ALPHA)
    F_coh, _ = coherent_fidelity_multi_state(
        u, H0, Hc, psi_i, psi_f, DT, want_grad=False
    )
    F_coh = float(F_coh)

    return {
        "F_pedersen": F_ped,
        "L1": L1,
        "F_coh": F_coh,
        "unitarity_err_2x2": float(
            np.linalg.norm(U_log.conj().T @ U_log - np.eye(2))
        ),
    }


def build_table():
    rows = []
    pulse_sets = discover_pulse_sets()

    for n_c in TRUNC_LIST:
        H0, Hc = make_hamiltonian(N_T, n_c)
        B = logical_basis(N_T, n_c, alpha=ALPHA)

        for set_name, path_fn in pulse_sets:
            for gate in GATES:
                path = path_fn(gate)
                if path is None or not os.path.isfile(path):
                    continue
                u = _load(path)
                metrics = score_pulse(gate, u, n_c, H0, Hc, B)
                retrain = metrics["F_pedersen"] < DECISION_THRESHOLD
                rows.append({
                    "pulse_set": set_name,
                    "gate": gate,
                    "n_c": n_c,
                    "path": os.path.relpath(path, REPO_ROOT),
                    "F_pedersen": metrics["F_pedersen"],
                    "L1": metrics["L1"],
                    "F_coh": metrics["F_coh"],
                    "unitarity_err_2x2": metrics["unitarity_err_2x2"],
                    "decision_threshold": DECISION_THRESHOLD,
                    "retrain_required": retrain,
                    "decision": "RETRAIN" if retrain else "HOLD",
                })
    return pd.DataFrame(rows)


def print_decision_table(df):
    print("=" * 88)
    print("PHASE 1 — decision-gate numbers (Pedersen + L1)")
    print(f"n_t={N_T}, dt={DT} us, alpha={ALPHA:.6f}, N=550, "
          f"n_c in {TRUNC_LIST}")
    print(f"Decision bar: F_pedersen >= {DECISION_THRESHOLD} at n_c={HEADLINE_N_C} "
          f"-> HOLD (no retrain); else RETRAIN into pulses/phase1/")
    print("F_coh is the optimizer process objective only — not the decision metric.")
    print("=" * 88)

    for set_name in df["pulse_set"].unique():
        sub = (
            df[(df["pulse_set"] == set_name) & (df["n_c"] == HEADLINE_N_C)]
            .set_index("gate")
            .reindex([g for g in GATES if g in set(df.loc[df["pulse_set"] == set_name, "gate"])])
        )

        print(f"\n{'-' * 88}")
        print(f"Headline n_c={HEADLINE_N_C}  |  pulse set: {set_name}")
        print(f"{'-' * 88}")
        cols = ["F_pedersen", "L1", "F_coh", "unitarity_err_2x2", "decision"]
        print(sub[cols].to_string(
            float_format=lambda x: f"{x:.6e}" if isinstance(x, float) and abs(x) < 1e-2
            else (f"{x:.9f}" if isinstance(x, float) else str(x))
        ))

        n_hold = int((sub["decision"] == "HOLD").sum())
        n_retrain = int((sub["decision"] == "RETRAIN").sum())
        print(f"\n  HOLD={n_hold}   RETRAIN={n_retrain}   "
              f"(threshold F_pedersen >= {DECISION_THRESHOLD})")

    # Compact pivot: F_pedersen and L1 vs n_c for main
    main = df[df["pulse_set"] == "main"]
    if not main.empty:
        print(f"\n{'-' * 88}")
        print("F_pedersen vs n_c  (main pulses)")
        print(f"{'-' * 88}")
        pivot_f = main.pivot(index="gate", columns="n_c", values="F_pedersen").reindex(GATES)
        print(pivot_f.to_string(float_format=lambda x: f"{x:.9f}"))

        print(f"\n{'-' * 88}")
        print("L1 vs n_c  (main pulses)")
        print(f"{'-' * 88}")
        pivot_l = main.pivot(index="gate", columns="n_c", values="L1").reindex(GATES)
        print(pivot_l.to_string(float_format=lambda x: f"{x:.6e}"))

    # Overall retrain decision
    headline = df[(df["pulse_set"] == "main") & (df["n_c"] == HEADLINE_N_C)]
    any_retrain = bool(headline["retrain_required"].any()) if not headline.empty else False
    print(f"\n{'=' * 88}")
    if any_retrain:
        bad = headline.loc[headline["retrain_required"], "gate"].tolist()
        print(f"PHASE 1 DECISION: RETRAIN required for {bad}")
        print(f"  Write new pulses to pulses/phase1/u_<gate>.npy (do not overwrite main).")
    else:
        print("PHASE 1 DECISION: HOLD — no relative-phase defect on main pulses.")
        print("  Deliverables 1–3 stand; skip retrain; existing u_*_main.npy are canonical.")
    print(f"{'=' * 88}")


def main():
    df = build_table()
    os.makedirs(TABLE_DIR, exist_ok=True)
    out_csv = os.path.join(TABLE_DIR, "phase1_decision_gate.csv")
    df.to_csv(out_csv, index=False)

    print_decision_table(df)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
