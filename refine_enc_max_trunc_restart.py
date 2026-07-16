#!/usr/bin/env python3
"""
refine_enc_max_trunc_restart.py

Restart the max-truncation refinement of U_enc from the pulse produced by
refine_enc_max_trunc.py (pulses/u_enc_refined_t3v2_maxtrunc.npy). That run
stopped after only 21/1500 iterations on L-BFGS-B's own relative-reduction
criterion (REL_REDUCTION_OF_F <= FACTR*EPSMCH) -- i.e. it looked converged to
the current quasi-Newton Hessian approximation, not because it ran out of
budget. Restarting with a fresh L-BFGS-B call (fresh Hessian approximation)
often makes further progress from a point like that.

Same config as refine_enc_max_trunc.py, just warm-starting from its output
and saving to a new file so nothing gets overwritten.

Usage:
    python refine_enc_max_trunc_restart.py
"""

import os
import numpy as np
from optimizer_max_trunc import refine_pulse_max_trunc
from cat_code import get_encode_state_pairs, validate_pulse_truncations
from refine_enc_max_trunc import (
    evaluate_fidelity_on_truncations,
    print_comparison_table,
    N_T,
    TRAINING_TRUNC_LIST,
    VALIDATION_TRUNC_RANGE,
    PENALTIES,
    PENALTY_SCALE,
    EXTRA_MAXITER,
    REFRESH_EVERY,
    CAV_BAND,
    TRA_BAND,
)

GET_STATE_PAIRS = get_encode_state_pairs
INPUT_PULSE_PATH = "pulses/u_enc_refined_t3v2_maxtrunc.npy"
OUTPUT_PULSE_PATH = "pulses/u_enc_refined_t3v2_maxtrunc_restart.npy"


def main():
    print("\n" + "=" * 70)
    print("RESTART: U_enc MAX-TRUNCATION REFINEMENT")
    print("=" * 70)

    if not os.path.exists(INPUT_PULSE_PATH):
        raise FileNotFoundError(f"Could not find pulse: {INPUT_PULSE_PATH}")

    u_original = np.load(INPUT_PULSE_PATH)
    print(f"\nLoaded pulse: {INPUT_PULSE_PATH}")
    print(f"Shape: {u_original.shape}")

    print("\n--- Evaluating pulse BEFORE restart ---")
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

    print("\n--- Evaluating pulse AFTER restart ---")
    after_results = evaluate_fidelity_on_truncations(
        u_refined, GET_STATE_PAIRS, TRAINING_TRUNC_LIST, n_t=N_T
    )

    print_comparison_table(
        before_results,
        after_results,
        title="Training Truncation Fidelity: U_enc (before vs after restart)"
    )

    print("\n--- Wide-range Validation on RESTARTED pulse ---")
    validate_pulse_truncations(
        u=u_refined,
        get_targets_func=GET_STATE_PAIRS,
        trunc_range=VALIDATION_TRUNC_RANGE,
        n_t=N_T,
        title="Restarted Pulse - Wide Validation"
    )

    print("\n" + "=" * 70)
    print("RESTART COMPLETE")
    print(f"Refined pulse saved as: {OUTPUT_PULSE_PATH}")
    print(f"info: {info}")
    print("=" * 70)


if __name__ == "__main__":
    main()
