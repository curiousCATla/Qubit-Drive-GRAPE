#!/usr/bin/env python3
"""
qutip_validate.py

Independent cross-check of a saved GRAPE pulse using QuTiP's own ODE-based
Schrodinger-equation solver (sesolve), instead of grape_core's hand-rolled
propagator (fidelity_grad, which exponentiates each step via eigh).

Why this is worth doing: grape_core.fidelity_grad is the function the
optimizer maximizes AND the function that reports "final fidelity". If it had
a bug (wrong operator ordering, a missing factor of 2, wrong dt convention),
the optimizer would happily converge and the fidelity report would look
great -- nothing inside grape_core.py can catch that. QuTiP is a separately
implemented, widely-validated propagator, so agreement between the two is
real evidence the physics is right, not just that the code is internally
consistent.

QuTiP helpers (Hamiltonian build, sesolve propagation, multi-state fidelity)
live in ``qutip_grape_optimizer.py``; this file only runs the pulse table
against grape_core's propagator.

Start here: pulses/u_opt_eq23eq24_coldstart.npy, the |g,0> -> |g,6> cavity
Fock-state preparation pulse (N=250 steps, 4 channels: cavity I/Q, transmon I/Q).

Usage:
    python qutip_validate.py
"""
import os
import numpy as np

from grape_core import fidelity_multi_state, make_hamiltonian
from compare_pulses import get_g6_state_pairs
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
from qutip_grape_optimizer import (
    DT_DEFAULT,
    qutip_multi_state_fidelity,
)

PULSE_DIR = "pulses"
TABLE_DIR = "tables"
DT = DT_DEFAULT         # microseconds, matches grape_core.dt
N_T = 3                 # transmon levels every pulse was trained with
TRAINED_TRUNC = (22, 24, 26)
TRUNC_LIST = [22, 24, 26, 28, 30, 32, 34, 36, 38, 40]   # 22/24/26 = trained on; rest = generalization check

# filename -> (label, get_state_pairs factory) -- same factories the
# optimizer and compare_pulses.py use, so the *targets* are shared, but the
# *propagator* (qutip.sesolve in qutip_grape_optimizer) is independent of
# grape_core's eigh-based fidelity_grad.
PULSE_MAP = {
    "u_opt_eq23eq24_coldstart.npy": ("g0->g6 prep", get_g6_state_pairs),
    "u_X_eq23eq24_coldstart.npy":   ("X",           get_logical_X_state_pairs),
    "u_Y_eq23eq24_coldstart.npy":   ("Y",           get_logical_Y_state_pairs),
    "u_Z_eq23eq24_coldstart.npy":   ("Z",           get_logical_Z_state_pairs),
    "u_H_eq23eq24_coldstart.npy":   ("H",           get_logical_H_state_pairs),
    "u_T_eq23eq24_coldstart.npy":   ("T",           get_logical_T_state_pairs),
    "u_I_eq23eq24_coldstart.npy":   ("I",           get_identity_state_pairs),
    "u_enc_eq23eq24_coldstart.npy": ("U_enc",       get_encode_state_pairs),
    "u_dec_eq23eq24_coldstart.npy": ("U_dec",       get_decode_state_pairs),
}


def validate_pulse(filename, label, factory, rows):
    path = os.path.join(PULSE_DIR, filename)
    u = np.load(path)
    print(f"\n{'='*60}")
    print(f"{label}  ({filename}, shape={u.shape})")
    print(f"{'='*60}")
    print(f"{'n_c':>4}  {'grape_core F':>14}  {'qutip F':>10}  {'|diff|':>10}")

    max_diff = 0.0
    for n_c in TRUNC_LIST:
        pairs = factory(n_c=n_c, n_t=N_T)
        psi_i_list = [p[0] for p in pairs]
        psi_f_list = [p[1] for p in pairs]

        # --- grape_core's own propagator ---
        H0_np, Hc_np = make_hamiltonian(N_T, n_c)
        F_gc, _ = fidelity_multi_state(u, H0_np, Hc_np, psi_i_list, psi_f_list, DT, want_grad=False)

        # --- QuTiP's independent ODE propagator (helpers in qutip_grape_optimizer) ---
        F_qt = qutip_multi_state_fidelity(u, factory, N_T, n_c, dt=DT)

        diff = abs(F_gc - F_qt)
        max_diff = max(max_diff, diff)
        trained = n_c in TRAINED_TRUNC
        marker = " (trained)" if trained else " (held out)"
        print(f"{n_c:>4}  {F_gc:>14.6f}  {F_qt:>10.6f}  {diff:>10.2e}{marker}")
        rows.append({
            "label": label, "n_c": n_c, "F_gc": float(F_gc), "F_qt": float(F_qt),
            "diff": float(diff), "trained": trained,
        })

    return max_diff


def main():
    print("QuTiP cross-check of every saved *_mt.npy pulse against grape_core's own propagator.")
    summary = []
    rows = []
    for filename, (label, factory) in PULSE_MAP.items():
        path = os.path.join(PULSE_DIR, filename)
        if not os.path.exists(path):
            print(f"\n[SKIP] {label}: {path} not found")
            continue
        max_diff = validate_pulse(filename, label, factory, rows)
        summary.append((label, max_diff))

    print(f"\n{'='*60}")
    print("SUMMARY (max |grape_core F - qutip F| across truncations)")
    print(f"{'='*60}")
    for label, max_diff in summary:
        flag = "  <-- CHECK THIS" if max_diff > 1e-4 else ""
        print(f"  {label:14s} {max_diff:.2e}{flag}")

    os.makedirs(TABLE_DIR, exist_ok=True)
    csv_path = os.path.join(TABLE_DIR, "qutip_validation.csv")
    with open(csv_path, "w") as f:
        f.write("label,n_c,F_gc,F_qt,diff,trained\n")
        for r in rows:
            f.write(f"{r['label']},{r['n_c']},{r['F_gc']!r},{r['F_qt']!r},{r['diff']!r},{r['trained']}\n")
    print(f"\nWrote per-truncation data: {csv_path}")


if __name__ == "__main__":
    main()
