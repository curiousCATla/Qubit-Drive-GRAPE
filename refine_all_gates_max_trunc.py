#!/usr/bin/env python3
"""
refine_all_gates_max_trunc.py

Apply the max-truncation refinement recipe (refine + one restart, same as
refine_enc_max_trunc.py / refine_enc_max_trunc_restart.py did for U_enc) to
every other saved pulse: the logical gates X/Y/Z/H/T/I, U_dec, and the
g0->g6 cavity-prep pulse (u_opt.npy).

For each pulse:
  1. Load the existing v2 pulse (warm start).
  2. optimize_multi_state_pulse_max_trunc (stage 1: fresh L-BFGS-B run).
  3. optimize_multi_state_pulse_max_trunc again, warm-started from stage 1's
     output (stage 2: "restart", fresh Hessian approximation -- this is what
     recovered the training-truncation dip for U_enc).
  4. Save the stage-2 result under a concise "_mt.npy" filename.

Called directly via optimize_multi_state_pulse_max_trunc (not the
refine_pulse_max_trunc wrapper) because u_opt.npy has a different pulse
length (N=250) than the rest (N=550), and the wrapper hard-codes N=550.

Usage:
    python refine_all_gates_max_trunc.py
"""

import os
import numpy as np
from optimizer_max_trunc import optimize_multi_state_pulse_max_trunc
from cat_code import (
    get_logical_X_state_pairs,
    get_logical_Y_state_pairs,
    get_logical_Z_state_pairs,
    get_logical_H_state_pairs,
    get_logical_T_state_pairs,
    get_identity_state_pairs,
    get_decode_state_pairs,
)
from compare_pulses import get_g6_state_pairs
from grape_core import make_hamiltonian, fidelity_multi_state

PULSE_DIR = "pulses"
N_T = 3
DT = 0.002
TRAINING_TRUNC_LIST = [22, 24, 26]

PENALTIES = {
    'deriv': 0.0001,
    'boundary': 0.00002,
    'amp': 0.00008,
    'amp_max': 40.0,
    'disc': 0.5,
}
EXTRA_MAXITER = 1500
REFRESH_EVERY = 10
CAV_BAND = (-27.0, 27.0)
TRA_BAND = (-33.0, 33.0)
HARD_AMP_LIMIT = 40.0

# name -> (factory, input filename, concise output filename)
GATES = {
    "X":     (get_logical_X_state_pairs, "u_X_refined_t3v2.npy",   "u_X_mt.npy"),
    "Y":     (get_logical_Y_state_pairs, "u_Y_refined_t3v2.npy",   "u_Y_mt.npy"),
    "Z":     (get_logical_Z_state_pairs, "u_Z_refined_t3v2.npy",   "u_Z_mt.npy"),
    "H":     (get_logical_H_state_pairs, "u_H_refined_t3v2.npy",   "u_H_mt.npy"),
    "T":     (get_logical_T_state_pairs, "u_T_refined_t3v2.npy",   "u_T_mt.npy"),
    "I":     (get_identity_state_pairs,  "u_I_refined_t3v2.npy",   "u_I_mt.npy"),
    "U_dec": (get_decode_state_pairs,    "u_dec_refined_t3v2.npy", "u_dec_mt.npy"),
    "u_opt": (get_g6_state_pairs,        "u_opt.npy",              "u_opt_mt.npy"),
}


def eval_fidelity(u, factory, trunc_list, n_t=N_T, dt=DT):
    results = {}
    for nc in trunc_list:
        pairs = factory(n_c=nc, n_t=n_t)
        psi_i = [p[0] for p in pairs]
        psi_f = [p[1] for p in pairs]
        H0, Hc = make_hamiltonian(n_t=n_t, n_c=nc)
        F, _ = fidelity_multi_state(u, H0, Hc, psi_i, psi_f, dt, want_grad=False)
        results[nc] = F
    return results


def run_one_gate(name, factory, input_filename, output_filename):
    input_path = os.path.join(PULSE_DIR, input_filename)
    output_path = os.path.join(PULSE_DIR, output_filename)

    print("\n" + "#" * 70)
    print(f"# {name}: {input_filename} -> {output_filename}")
    print("#" * 70)

    u0 = np.load(input_path)
    N = u0.shape[0]
    print(f"Loaded {input_path}  shape={u0.shape}  (N={N})")

    before = eval_fidelity(u0, factory, TRAINING_TRUNC_LIST)

    print(f"\n--- {name}: stage 1 (fresh max-trunc refinement) ---")
    u1, info1 = optimize_multi_state_pulse_max_trunc(
        get_state_pairs=factory,
        trunc_list=TRAINING_TRUNC_LIST,
        n_t=N_T,
        N=N,
        dt=DT,
        penalties=PENALTIES.copy(),
        refresh_every=REFRESH_EVERY,
        warm_start=u0,
        maxiter=EXTRA_MAXITER,
        cav_band=CAV_BAND,
        tra_band=TRA_BAND,
        hard_amp_limit=HARD_AMP_LIMIT,
        verbose=True,
    )

    print(f"\n--- {name}: stage 2 (restart, fresh Hessian) ---")
    u2, info2 = optimize_multi_state_pulse_max_trunc(
        get_state_pairs=factory,
        trunc_list=TRAINING_TRUNC_LIST,
        n_t=N_T,
        N=N,
        dt=DT,
        penalties=PENALTIES.copy(),
        refresh_every=REFRESH_EVERY,
        warm_start=u1,
        maxiter=EXTRA_MAXITER,
        cav_band=CAV_BAND,
        tra_band=TRA_BAND,
        hard_amp_limit=HARD_AMP_LIMIT,
        verbose=True,
    )

    after = eval_fidelity(u2, factory, TRAINING_TRUNC_LIST)

    np.save(output_path, u2)
    print(f"\nSaved: {output_path}")

    print(f"\n{name} summary (bare fidelity on training truncations):")
    for nc in TRAINING_TRUNC_LIST:
        marker = " (max, target)" if nc == max(TRAINING_TRUNC_LIST) else ""
        print(f"  n_c={nc:2d}: before={before[nc]:.6f}  after={after[nc]:.6f}  "
              f"delta={after[nc]-before[nc]:+.6f}{marker}")

    return {'name': name, 'before': before, 'after': after,
            'stage1_iters': info1['iterations'], 'stage2_iters': info2['iterations']}


def main():
    print("\n" + "=" * 70)
    print("MAX-TRUNCATION REFINEMENT: ALL REMAINING PULSES")
    print("=" * 70)

    summaries = []
    for name, (factory, input_filename, output_filename) in GATES.items():
        input_path = os.path.join(PULSE_DIR, input_filename)
        if not os.path.exists(input_path):
            print(f"\n[SKIP] {name}: {input_path} not found")
            continue
        summaries.append(run_one_gate(name, factory, input_filename, output_filename))

    print("\n" + "=" * 70)
    print("FINAL SUMMARY (bare fidelity, training truncations, before -> after)")
    print("=" * 70)
    for s in summaries:
        nc_max = max(TRAINING_TRUNC_LIST)
        print(f"{s['name']:6s} | n_c={nc_max}: {s['before'][nc_max]:.6f} -> {s['after'][nc_max]:.6f} "
              f"({s['after'][nc_max]-s['before'][nc_max]:+.6f})  "
              f"[stage1={s['stage1_iters']} iters, stage2={s['stage2_iters']} iters]")
    print("=" * 70)


if __name__ == "__main__":
    main()
