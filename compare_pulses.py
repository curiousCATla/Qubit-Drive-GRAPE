#!/usr/bin/env python3
"""
compare_pulses.py

Lightweight, standalone comparison table across the project's saved GRAPE
pulses: fidelity at several cavity truncations (n_c) plus basic pulse-shape
metrics (peak amplitude, RMS amplitude, duration, smoothness).

Usage:
    python compare_pulses.py
"""

import os
import numpy as np
import pandas as pd

from grape_core import make_hamiltonian, fidelity_multi_state, basis_state, two_pi
from cat_code import (
    get_logical_X_state_pairs,
    get_logical_Y_state_pairs,
    get_logical_Z_state_pairs,
    get_logical_H_state_pairs,
    get_logical_T_state_pairs,
    get_identity_state_pairs,
    get_encode_state_pairs,
    get_decode_state_pairs,
)

# ============================================================
# CONFIGURATION
# ============================================================
PULSE_DIR = "pulses"
N_T = 3
DT = 0.002
N_C_LIST = [20, 22, 24, 26, 28]


def get_g6_state_pairs(n_c, n_t=3):
    """Factory for the |g,0> -> |g,6> state transfer (pulse_analysis.py's u_opt.npy)."""
    psi_i = basis_state(n_t, n_c, 0, 0)
    psi_f = basis_state(n_t, n_c, 0, 6)
    return [(psi_i, psi_f)]


# filename -> (label, get_state_pairs factory)
PULSE_MAP = {
    "u_X_refined_t3v2.npy":   ("X",           get_logical_X_state_pairs),
    "u_Y_refined_t3v2.npy":   ("Y",           get_logical_Y_state_pairs),
    "u_Z_refined_t3v2.npy":   ("Z",           get_logical_Z_state_pairs),
    "u_H_refined_t3v2.npy":   ("H",           get_logical_H_state_pairs),
    "u_T_refined_t3v2.npy":   ("T",           get_logical_T_state_pairs),
    "u_I_refined_t3v2.npy":   ("I",           get_identity_state_pairs),
    "u_enc_refined_t3v2.npy": ("U_enc",       get_encode_state_pairs),
    "u_dec_refined_t3v2.npy": ("U_dec",       get_decode_state_pairs),
    "u_opt.npy":              ("g0->g6 prep", get_g6_state_pairs),
}


# ============================================================
# METRICS
# ============================================================
def evaluate_fidelity(u, factory, n_c_list, n_t=N_T, dt=DT):
    """Evaluate average fidelity of pulse u at each n_c in n_c_list."""
    results = {}
    for nc in n_c_list:
        pairs = factory(n_c=nc, n_t=n_t)
        psi_i_list = [p[0] for p in pairs]
        psi_f_list = [p[1] for p in pairs]
        H0, Hc = make_hamiltonian(n_t, nc)
        F, _ = fidelity_multi_state(u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=False)
        results[nc] = F
    return results


def pulse_shape_metrics(u, dt=DT):
    """Cheap shape metrics computed directly from the control array (no simulation)."""
    u_MHz = u / two_pi
    return {
        'N_steps': u.shape[0],
        'Duration_ns': u.shape[0] * dt * 1000,
        'Peak_amp_MHz': np.max(np.abs(u_MHz)),
        'RMS_amp_MHz': np.sqrt(np.mean(u_MHz ** 2)),
        'Smoothness_MHz_per_ns': np.mean(np.abs(np.diff(u_MHz, axis=0))) / (dt * 1000),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("PULSE COMPARISON TABLE")
    print("=" * 70)

    rows = []
    for filename, (label, factory) in PULSE_MAP.items():
        path = os.path.join(PULSE_DIR, filename)
        if not os.path.exists(path):
            print(f"[SKIP] {filename}: not found in {PULSE_DIR}/")
            continue

        u = np.load(path)
        print(f"Evaluating {label:14s} ({filename}) shape={u.shape}")

        fid_results = evaluate_fidelity(u, factory, N_C_LIST)
        shape_metrics = pulse_shape_metrics(u)

        row = {'Label': label, 'Pulse_file': filename}
        for nc, F in fid_results.items():
            row[f'F(n_c={nc})'] = F
        fids = list(fid_results.values())
        row['F_mean'] = np.mean(fids)
        row['F_min'] = np.min(fids)
        row['F_std'] = np.std(fids)
        row.update(shape_metrics)
        rows.append(row)

    if not rows:
        print("\nNo pulses found — nothing to compare.")
        return

    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print(df.to_string(index=False, float_format="%.6f"))
    print("=" * 70)

    os.makedirs("tables", exist_ok=True)
    out_path = os.path.join("tables", "pulse_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved table to {out_path}")


if __name__ == "__main__":
    main()
