#!/usr/bin/env python3
"""
refine_enc_max_trunc.py

Refine pulses/u_enc_refined_t3v2.npy (the encoding pulse U_enc) using the new
max-truncation objective (optimizer_max_trunc.refine_pulse_max_trunc), which
drives L-BFGS-B on the bare fidelity at the max training truncation only and
periodically (every `refresh_every` iterations) pulls the lower-truncation
fidelities toward it via a consistency penalty.

Mirrors the config/workflow of refine_and_compare.py's "U_enc" entry, just
swapping refine_pulse -> refine_pulse_max_trunc and using the v2 pulse as the
new warm start.

Usage:
    python refine_enc_max_trunc.py
"""

import os
import numpy as np
import pandas as pd
from optimizer_max_trunc import refine_pulse_max_trunc
from cat_code import get_encode_state_pairs, validate_pulse_truncations
from grape_core import make_hamiltonian, fidelity_multi_state


# ============================================================
# CONFIGURATION
# ============================================================

N_T = 3
GET_STATE_PAIRS = get_encode_state_pairs

INPUT_PULSE_PATH = "pulses/u_enc_refined_t3v2.npy"
OUTPUT_PULSE_PATH = "pulses/u_enc_refined_t3v2_maxtrunc.npy"

TRAINING_TRUNC_LIST = [22, 24, 26]   # max(trunc_list)=26 is the primary target
VALIDATION_TRUNC_RANGE = list(range(18, 31, 2))

PENALTIES = {
    'deriv': 0.0001,
    'boundary': 0.00002,
    'amp': 0.00008,
    'amp_max': 40.0,
    'disc': 0.5,          # weight on the (F_max - F_k)^2 consistency term
}
PENALTY_SCALE = 1
EXTRA_MAXITER = 1500
REFRESH_EVERY = 10        # recompute the consistency term every 10 iterations

CAV_BAND = (-27.0, 27.0)
TRA_BAND = (-33.0, 33.0)


# ============================================================
# HELPERS
# ============================================================

def evaluate_fidelity_on_truncations(u, get_state_pairs, trunc_list, n_t=3):
    results = {}
    for nc in trunc_list:
        pairs = get_state_pairs(n_c=nc, n_t=n_t)
        psi_i = [p[0] for p in pairs]
        psi_f = [p[1] for p in pairs]
        H0, Hc = make_hamiltonian(n_t=n_t, n_c=nc)
        F, _ = fidelity_multi_state(u, H0, Hc, psi_i, psi_f, dt=0.002, want_grad=False)
        results[nc] = F
    return results


def print_comparison_table(before_dict, after_dict, title="Fidelity Comparison"):
    df = pd.DataFrame({
        'n_c': list(before_dict.keys()),
        'Before': list(before_dict.values()),
        'After': list(after_dict.values())
    })
    df['Improvement'] = df['After'] - df['Before']
    df['Improvement %'] = (df['Improvement'] / df['Before'] * 100).round(2)

    print(f"\n{title}")
    print("=" * 70)
    print(df.to_string(index=False, float_format="%.6f"))
    print("=" * 70)

    avg_before = np.mean(list(before_dict.values()))
    avg_after = np.mean(list(after_dict.values()))
    print(f"Average Before : {avg_before:.6f}")
    print(f"Average After  : {avg_after:.6f}")
    print(f"Average Gain   : {avg_after - avg_before:.6f} "
          f"({(avg_after - avg_before)/avg_before*100:.2f}%)")


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("REFINE U_enc WITH MAX-TRUNCATION OBJECTIVE")
    print("=" * 70)

    if not os.path.exists(INPUT_PULSE_PATH):
        raise FileNotFoundError(f"Could not find pulse: {INPUT_PULSE_PATH}")

    u_original = np.load(INPUT_PULSE_PATH)
    print(f"\nLoaded pulse: {INPUT_PULSE_PATH}")
    print(f"Shape: {u_original.shape}")

    print("\n--- Evaluating ORIGINAL (v2) pulse ---")
    before_results = evaluate_fidelity_on_truncations(
        u_original, GET_STATE_PAIRS, TRAINING_TRUNC_LIST, n_t=N_T
    )

    u_refined, info = refine_pulse_max_trunc(
        get_state_pairs=GET_STATE_PAIRS,
        initial_pulse=u_original,
        trunc_list=TRAINING_TRUNC_LIST,
        n_t=N_T,
        extra_maxiter=EXTRA_MAXITER,
        penalties=PENALTIES,
        penalty_scale=PENALTY_SCALE,
        refresh_every=REFRESH_EVERY,
        save_path=OUTPUT_PULSE_PATH,
        cav_band=CAV_BAND,
        tra_band=TRA_BAND,
        verbose=True
    )

    print("\n--- Evaluating REFINED (max-trunc) pulse ---")
    after_results = evaluate_fidelity_on_truncations(
        u_refined, GET_STATE_PAIRS, TRAINING_TRUNC_LIST, n_t=N_T
    )

    print_comparison_table(
        before_results,
        after_results,
        title="Training Truncation Fidelity: U_enc (v2 vs max-trunc refinement)"
    )

    print("\n--- Wide-range Validation on REFINED pulse ---")
    validate_pulse_truncations(
        u=u_refined,
        get_targets_func=GET_STATE_PAIRS,
        trunc_range=VALIDATION_TRUNC_RANGE,
        n_t=N_T,
        title="Refined Pulse - Wide Validation"
    )

    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE - U_enc refined with max-trunc objective")
    print(f"Refined pulse saved as: {OUTPUT_PULSE_PATH}")
    print(f"info: {info}")
    print("=" * 70)


if __name__ == "__main__":
    main()
