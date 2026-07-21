#!/usr/bin/env python3
"""
refine_dt_and_compare.py

Refine pulses/u_opt_eq23eq24_coldstart.npy (the |g,0> -> |g,6> cavity state
prep, N=250, dt=0.002 us) onto a 10x finer time grid via refine_pulse_dt,
then compare fidelity, amplitude, and penalty behavior before vs after.

Usage:
    python refine_dt_and_compare.py
"""

import numpy as np
import pandas as pd

from optimizer import refine_pulse_dt
from compare_pulses import get_g6_state_pairs
from cat_code import validate_pulse_truncations
from grape_core import derivative_penalty, boundary_penalty, amplitude_penalty

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PULSE_PATH = "pulses/u_opt_eq23eq24_coldstart.npy"
OUTPUT_PULSE_PATH = "pulses/u_opt10x"

GET_STATE_PAIRS = get_g6_state_pairs
N_T = 3
DT = 0.002          # original step size (us)
S = 10              # dt shrink factor -> new_dt = DT / S

TRAINING_TRUNC_LIST = [20, 24, 28]
VALIDATION_TRUNC_RANGE = list(range(18, 31, 2))

# Same penalty defaults refine_pulse/refine_pulse_dt use -- kept unchanged
# (not rescaled for the finer dt) per the user's request.
PENALTIES = {
    'deriv': 0.00001,
    'boundary': 0.00002,
    'amp': 0.00008,
    'amp_max': 40.0
}
EXTRA_MAXITER = 2000


# ============================================================
# HELPERS
# ============================================================

def evaluate_fidelity_on_truncations(u, dt, trunc_list, n_t=N_T):
    """Evaluate average fidelity over a list of truncations at a given dt."""
    from grape_core import make_hamiltonian, fidelity_multi_state
    results = {}
    for nc in trunc_list:
        pairs = GET_STATE_PAIRS(n_c=nc, n_t=n_t)
        psi_i = [p[0] for p in pairs]
        psi_f = [p[1] for p in pairs]
        H0, Hc = make_hamiltonian(n_t=n_t, n_c=nc)
        F, _ = fidelity_multi_state(u, H0, Hc, psi_i, psi_f, dt=dt, want_grad=False)
        results[nc] = F
    return results


def print_comparison_table(before_dict, after_dict, title="Fidelity Comparison"):
    df = pd.DataFrame({
        'n_c': list(before_dict.keys()),
        'Before (dt)': list(before_dict.values()),
        'After (dt/10)': list(after_dict.values())
    })
    df['Improvement'] = df['After (dt/10)'] - df['Before (dt)']

    print(f"\n{title}")
    print("=" * 70)
    print(df.to_string(index=False, float_format="%.6f"))
    print("=" * 70)

    avg_before = np.mean(list(before_dict.values()))
    avg_after = np.mean(list(after_dict.values()))
    print(f"Average Before : {avg_before:.6f}")
    print(f"Average After  : {avg_after:.6f}")
    print(f"Average Gain   : {avg_after - avg_before:.6f}")


def print_penalty_comparison(u_before, u_after, s):
    """Raw (unweighted) penalty values before vs after, to see the ~s and
    ~1/s scaling directly (see refine_pulse_dt's docstring)."""
    d_before, _ = derivative_penalty(u_before)
    d_after, _ = derivative_penalty(u_after)
    b_before, _ = boundary_penalty(u_before)
    b_after, _ = boundary_penalty(u_after)
    a_before, _ = amplitude_penalty(u_before, amp_max=PENALTIES['amp_max'])
    a_after, _ = amplitude_penalty(u_after, amp_max=PENALTIES['amp_max'])

    print("\nRaw penalty values (unweighted by lambda)")
    print("=" * 70)
    print(f"{'Penalty':<12}{'Before':>15}{'After':>15}{'After/Before':>18}")
    print(f"{'derivative':<12}{d_before:>15.6f}{d_after:>15.6f}{(d_after/d_before if d_before else float('nan')):>18.4f}")
    print(f"{'boundary':<12}{b_before:>15.6f}{b_after:>15.6f}{(b_after/b_before if b_before else float('nan')):>18.4f}")
    print(f"{'amplitude':<12}{a_before:>15.6f}{a_after:>15.6f}{(a_after/a_before if a_before else float('nan')):>18.4f}")
    print("=" * 70)
    print(f"Expectation: derivative ~ 1/{s} of before, amplitude ~ {s}x before, boundary ~ unchanged.")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("REFINE-DT AND COMPARE: U_opt (|g,0> -> |g,6>)")
    print("=" * 70)

    u_original = np.load(INPUT_PULSE_PATH)
    print(f"\nLoaded pulse: {INPUT_PULSE_PATH}")
    print(f"Shape: {u_original.shape}, dt={DT}, duration={u_original.shape[0]*DT:.4f} us")

    print("\n--- Evaluating ORIGINAL pulse ---")
    before_results = evaluate_fidelity_on_truncations(u_original, DT, VALIDATION_TRUNC_RANGE)

    u_refined, info = refine_pulse_dt(
        get_state_pairs=GET_STATE_PAIRS,
        initial_pulse=u_original,
        s=S,
        dt=DT,
        trunc_list=TRAINING_TRUNC_LIST,
        n_t=N_T,
        extra_maxiter=EXTRA_MAXITER,
        penalties=PENALTIES,
        save_path=OUTPUT_PULSE_PATH,
        verbose=True
    )
    new_dt = info['dt']

    print("\n--- Evaluating REFINED pulse ---")
    after_results = evaluate_fidelity_on_truncations(u_refined, new_dt, VALIDATION_TRUNC_RANGE)

    print_comparison_table(
        before_results, after_results,
        title="Validation-range Fidelity: U_opt (Before vs After dt-refinement)"
    )

    print_penalty_comparison(u_original, u_refined, S)

    print("\nAmplitude/shape check")
    print("=" * 70)
    print(f"Peak |u| before: {np.max(np.abs(u_original)):.4f}  after: {np.max(np.abs(u_refined)):.4f}")
    print(f"u[0]   before: {u_original[0]}  after: {u_refined[0]}")
    print(f"u[-1]  before: {u_original[-1]}  after: {u_refined[-1]}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print(f"Refined pulse saved as: {OUTPUT_PULSE_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
