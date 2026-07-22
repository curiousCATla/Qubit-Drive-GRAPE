#!/usr/bin/env python3
"""
truncation_convergence.py

Validates that the training truncation n_c is "large enough" in the sense
used by Heeres et al. 2017, Supplementary Note 2 (Eq. 23-24): the bare
fidelity F_N should stop changing appreciably as N is increased beyond the
training truncation. This is the paper's own truncation-validity criterion
-- convergence of fidelity across nearby truncations, NOT a bound on
photon-number leakage (which the paper doesn't penalize at all; see
Supplementary Note 2: "the choice of N determines the maximum photon
number population which can be populated during the pulse... faster
pulses can be achieved with higher N").

Reuses cat_code.validate_pulse_truncations (already implements this exact
check) across every saved gate pulse.

Usage:
    python validation/truncation_convergence.py           # checks pulses/u_*_mt.npy
    python validation/truncation_convergence.py --leak    # checks pulses/u_*_mt_leak.npy instead
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.cat_code import (
    get_logical_X_state_pairs,
    get_logical_Y_state_pairs,
    get_logical_Z_state_pairs,
    get_logical_H_state_pairs,
    get_logical_T_state_pairs,
    get_identity_state_pairs,
    get_encode_state_pairs,
    get_decode_state_pairs,
    validate_pulse_truncations,
)
from core.compare_pulses import get_g6_state_pairs

PULSE_DIR = "pulses"
TABLE_DIR = "tables"
FIG_DIR = "figures"
N_T = 3
DT = 0.002
TRAINING_NC = 26
TRUNC_RANGE = list(range(18, 37, 2))

# name -> (factory, base _mt.npy filename)
GATES = {
    "U_opt": (get_g6_state_pairs,        "u_opt_mt.npy"),
    "U_enc": (get_encode_state_pairs,    "u_enc_mt.npy"),
    "U_dec": (get_decode_state_pairs,    "u_dec_mt.npy"),
    "U_X":   (get_logical_X_state_pairs, "u_X_mt.npy"),
    "U_Y":   (get_logical_Y_state_pairs, "u_Y_mt.npy"),
    "U_Z":   (get_logical_Z_state_pairs, "u_Z_mt.npy"),
    "U_H":   (get_logical_H_state_pairs, "u_H_mt.npy"),
    "U_T":   (get_logical_T_state_pairs, "u_T_mt.npy"),
    "U_I":   (get_identity_state_pairs,  "u_I_mt.npy"),
}


def main():
    use_leak = "--leak" in sys.argv[1:]
    suffix = "_mt_leak.npy" if use_leak else "_mt.npy"
    tag = "_leak" if use_leak else ""

    all_results = {}
    for name, (factory, mt_filename) in GATES.items():
        filename = mt_filename.replace("_mt.npy", suffix)
        path = os.path.join(PULSE_DIR, filename)
        if not os.path.exists(path):
            print(f"[SKIP] {name}: {path} not found")
            continue
        u = np.load(path)
        results = validate_pulse_truncations(
            u=u, get_targets_func=factory, trunc_range=TRUNC_RANGE, n_t=N_T, dt=DT,
            title=f"{name} ({filename})"
        )
        all_results[name] = results

    if not all_results:
        print("No pulses found.")
        return

    # --- Table ---
    os.makedirs(TABLE_DIR, exist_ok=True)
    df = pd.DataFrame(all_results).T
    df.index.name = 'gate'
    out_csv = os.path.join(TABLE_DIR, f"truncation_convergence{tag}.csv")
    df.to_csv(out_csv)
    print(f"\nSaved table: {out_csv}")

    # --- Convergence summary: max |F_nc - F_max_nc| for nc >= training truncation ---
    print("\n" + "=" * 70)
    print("CONVERGENCE SUMMARY (drift in F for n_c >= training n_c=26)")
    print("=" * 70)
    for name, results in all_results.items():
        ncs_above = [nc for nc in TRUNC_RANGE if nc >= TRAINING_NC]
        Fs_above = [results[nc] for nc in ncs_above]
        drift = max(Fs_above) - min(Fs_above)
        flag = "  <-- NOT converged" if drift > 0.01 else ""
        print(f"{name:6s} | F(26)={results[TRAINING_NC]:.4f}  "
              f"range over n_c>=26: [{min(Fs_above):.4f}, {max(Fs_above):.4f}]  "
              f"drift={drift:.4f}{flag}")
    print("=" * 70)

    # --- Plot ---
    n_gates = len(all_results)
    ncols = 3
    nrows = int(np.ceil(n_gates / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    for i, (name, results) in enumerate(all_results.items()):
        ax = axes[i // ncols][i % ncols]
        ncs = sorted(results.keys())
        Fs = [results[nc] for nc in ncs]
        ax.plot(ncs, Fs, 'o-', color='tab:blue')
        ax.axvline(TRAINING_NC, color='gray', linestyle='--', linewidth=1, label=f'n_c={TRAINING_NC} (training)')
        ax.set_title(name)
        ax.set_xlabel('n_c')
        ax.set_ylabel('Bare fidelity F')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(n_gates, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    plt.suptitle("Truncation convergence check (Heeres et al. 2017 Eq. 23-24 criterion)"
                 + (" -- leak-refined pulses" if use_leak else ""))
    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out_fig = os.path.join(FIG_DIR, f"truncation_convergence{tag}.png")
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"Saved figure: {out_fig}")


if __name__ == "__main__":
    main()
