#!/usr/bin/env python3
"""
refine_and_compare.py2

Convenience script to refine an existing GRAPE pulse and compare
performance before vs after refinement.

Features:
- Loads an existing optimized pulse
- Runs refinement using refine_pulse()
- Evaluates fidelity on training truncations + wider validation set
- Prints a clean comparison table (Before vs After)
- Saves the refined pulse

Usage:
    python refine_and_compare.py
"""

import os
import numpy as np
import pandas as pd
from optimizer import refine_pulse
from cat_code import (
    get_logical_X_state_pairs,
    get_logical_Y_state_pairs,
    get_logical_Z_state_pairs,
    get_logical_H_state_pairs,
    get_logical_T_state_pairs,
    get_identity_state_pairs,
    get_encode_state_pairs,
    get_decode_state_pairs,
    validate_pulse_truncations
)


# ============================================================
# CONFIGURATION - Edit this section
# ============================================================

GATE = "H"   # Options: "X", "Y", "Z", "H", "T", "I", "U_enc", "U_dec"

N_T = 3  # number of transmon levels used throughout refinement/evaluation

# Mapping of gate name to (factory, default input pulse, default output pulse)
GATE_CONFIG = {
    "U_enc": {
        "factory": get_encode_state_pairs,
        "input":  "pulses/u_enc_refined_t3.npy",
        "output": "pulses/u_enc_refined_t3v2.npy"
    },
    "U_dec": {
        "factory": get_decode_state_pairs,
        "input":  "pulses/u_dec_refined_t3.npy",
        "output": "pulses/u_dec_refined_t3v2.npy"
    },
    "X": {
        "factory": get_logical_X_state_pairs,
        "input":  "pulses/u_X_refined_t3.npy",
        "output": "pulses/u_X_refined_t3v2.npy"
    },
    "Y": {
        "factory": get_logical_Y_state_pairs,
        "input":  "pulses/u_Y_refined_t3.npy",
        "output": "pulses/u_Y_refined_t3v2.npy"
    },
    "Z": {
        "factory": get_logical_Z_state_pairs,
        "input":  "pulses/u_Z_refined_t3.npy",
        "output": "pulses/u_Z_refined_t3v2.npy"
    },
    "H": {
        "factory": get_logical_H_state_pairs,
        "input":  "pulses/u_H_logical_v2.npy",
        "output": "pulses/u_H_refined_t3v2.npy"
    },
    "T": {
        "factory": get_logical_T_state_pairs,
        "input":  "pulses/u_T_refined_t3.npy",      # Use refined version as starting point
        "output": "pulses/u_T_refined_t3v2.npy"
    },
    "I": {
        "factory": get_identity_state_pairs,
        "input":  "pulses/u_I_logical_v2.npy",
        "output": "pulses/u_I_refined_t3v2.npy"
    },
}

if GATE not in GATE_CONFIG:
    raise ValueError(f"Unknown GATE '{GATE}'. Choose from: {list(GATE_CONFIG.keys())}")

config = GATE_CONFIG[GATE]
GET_STATE_PAIRS   = config["factory"]
INPUT_PULSE_PATH  = config["input"]
OUTPUT_PULSE_NAME = config["output"]

# Refinement settings
# Base penalty (lambda) values used during refinement. Edit these directly to
# tune regularization strength; PENALTY_SCALE (below) applies on top of these.
PENALTIES = {
    'deriv': 0.0001,     # smoothness penalty
    'boundary': 0.00002,  # start/end-at-zero penalty
    'amp': 0.00008,       # amplitude-excess penalty
    'amp_max': 40.0       # amplitude cap (rad/us) used by the amp penalty
}
PENALTY_SCALE = 1
EXTRA_MAXITER = 1500
TRAINING_TRUNC_LIST = [22, 24, 26]

WIDEN_TRAINING = False
VALIDATION_TRUNC_RANGE = list(range(18, 31, 2))


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def evaluate_fidelity_on_truncations(u, get_state_pairs, trunc_list, n_t=3):
    """Evaluate average fidelity on a list of truncations."""
    results = {}
    for nc in trunc_list:
        pairs = get_state_pairs(n_c=nc, n_t=n_t)
        psi_i = [p[0] for p in pairs]
        psi_f = [p[1] for p in pairs]
        # We need H0 and Hc for this truncation
        from grape_core import make_hamiltonian, fidelity_multi_state
        H0, Hc = make_hamiltonian(n_t=n_t, n_c=nc)
        F, _ = fidelity_multi_state(u, H0, Hc, psi_i, psi_f, dt=0.002, want_grad=False)
        results[nc] = F
    return results


def print_comparison_table(before_dict, after_dict, title="Fidelity Comparison"):
    """Print a nice Before vs After comparison table."""
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
    print("REFINE AND COMPARE WORKFLOW")
    print("=" * 70)

    # --- Load pulse ---
    pulse_path = INPUT_PULSE_PATH
    if not os.path.exists(pulse_path):
        alt_path = os.path.join("pulses", INPUT_PULSE_PATH)
        if os.path.exists(alt_path):
            pulse_path = alt_path
        else:
            raise FileNotFoundError(f"Could not find pulse: {INPUT_PULSE_PATH}")

    u_original = np.load(pulse_path)
    print(f"\nLoaded pulse: {pulse_path}")
    print(f"Shape: {u_original.shape}")

    # --- Evaluate BEFORE refinement ---
    print("\n--- Evaluating ORIGINAL pulse ---")
    before_results = evaluate_fidelity_on_truncations(
        u_original, GET_STATE_PAIRS, TRAINING_TRUNC_LIST, n_t=N_T
    )

    # --- Run refinement ---
    u_refined, info = refine_pulse(
        get_state_pairs=GET_STATE_PAIRS,
        initial_pulse=u_original,
        trunc_list=TRAINING_TRUNC_LIST,
        n_t=N_T,
        extra_maxiter=EXTRA_MAXITER,
        penalties=PENALTIES,
        penalty_scale=PENALTY_SCALE,
        widen_training=WIDEN_TRAINING,
        save_path=OUTPUT_PULSE_NAME,
        cav_band=(-27.0, 27.0),
        tra_band=(-33.0, 33.0),
        verbose=True
    )

    # --- Evaluate AFTER refinement ---
    print("\n--- Evaluating REFINED pulse ---")
    after_results = evaluate_fidelity_on_truncations(
        u_refined, GET_STATE_PAIRS, TRAINING_TRUNC_LIST, n_t=N_T
    )

    # --- Print comparison table ---
    print_comparison_table(
        before_results,
        after_results,
        title=f"Training Truncation Fidelity: {GATE} Gate (Before vs After Refinement)"
    )

    # --- Wide-range validation (optional but useful) ---
    print("\n--- Wide-range Validation on REFINED pulse ---")
    validate_pulse_truncations(
        u=u_refined,
        get_targets_func=GET_STATE_PAIRS,
        trunc_range=VALIDATION_TRUNC_RANGE,
        n_t=N_T,
        title="Refined Pulse - Wide Validation"
    )

    print("\n" + "=" * 70)
    print(f"WORKFLOW COMPLETE - {GATE} Gate Refined")
    print(f"Refined pulse saved as: {OUTPUT_PULSE_NAME}")
    print("=" * 70)


if __name__ == "__main__":
    main()
