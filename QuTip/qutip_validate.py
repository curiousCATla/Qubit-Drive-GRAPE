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

Start here: pulses/u_opt_main.npy, the |g,0> -> |g,6> cavity
Fock-state preparation pulse (N=250 steps, 4 channels: cavity I/Q, transmon I/Q).

Usage:
    python QuTip/qutip_validate.py
"""
import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.grape_core import fidelity_multi_state, make_hamiltonian
from core.compare_pulses import get_g6_state_pairs
from core.cat_code import (
    get_logical_X_state_pairs,
    get_logical_Y_state_pairs,
    get_logical_Z_state_pairs,
    get_logical_H_state_pairs,
    get_logical_T_state_pairs,
    get_identity_state_pairs,
    get_encode_state_pairs,
    get_decode_state_pairs,
)
from QuTip.qutip_grape_optimizer import (
    DT_DEFAULT,
    qutip_multi_state_fidelity,
    qutip_gate_logical_unitary,
    qutip_enc_logical_unitary,
    qutip_dec_logical_unitary,
)
# average_gate_fidelity/IDEAL_LOGICAL_U are a linear-algebra formula and a
# static target lookup, not propagator code -- reusing them doesn't
# compromise the "independent propagator" cross-check below (this file
# already imports state-pair factories and fidelity_multi_state itself from
# core/ for the *targets* side of the existing per-truncation comparison).
from validation.validate_logical_gates import (
    average_gate_fidelity,
    IDEAL_LOGICAL_U,
    tier3_gate_algebra,
    tier4_enc_relative_phase,
    tier4_dec_relative_phase,
)

PULSE_DIR = os.path.join(REPO_ROOT, "pulses")
TABLE_DIR = os.path.join(REPO_ROOT, "tables")
DT = DT_DEFAULT         # microseconds, matches grape_core.dt
N_T = 3                 # transmon levels every pulse was trained with
TRAINED_TRUNC = (22, 24, 26)
TRUNC_LIST = [22, 24, 26, 28, 30, 32, 34, 36, 38, 40]   # 22/24/26 = trained on; rest = generalization check

# filename -> (label, get_state_pairs factory) -- same factories the
# optimizer and compare_pulses.py use, so the *targets* are shared, but the
# *propagator* (qutip.sesolve in qutip_grape_optimizer) is independent of
# grape_core's eigh-based fidelity_grad.
PULSE_MAP = {
    "u_opt_main.npy": ("g0->g6 prep", get_g6_state_pairs),
    "u_X_main.npy":   ("X",           get_logical_X_state_pairs),
    "u_Y_main.npy":   ("Y",           get_logical_Y_state_pairs),
    "u_Z_main.npy":   ("Z",           get_logical_Z_state_pairs),
    "u_H_main.npy":   ("H",           get_logical_H_state_pairs),
    "u_T_main.npy":   ("T",           get_logical_T_state_pairs),
    "u_I_main.npy":   ("I",           get_identity_state_pairs),
    "u_enc_main.npy": ("U_enc",       get_encode_state_pairs),
    "u_dec_main.npy": ("U_dec",       get_decode_state_pairs),
}

# filename -> gate code used to dispatch the relative-phase cross-check
# below. u_opt_main.npy (a single-state Fock-state prep, not a two-branch
# logical map) has no gate code and is skipped by that check.
PHASE_CHECK_GATE = {
    "u_X_main.npy": "X", "u_Y_main.npy": "Y", "u_Z_main.npy": "Z",
    "u_H_main.npy": "H", "u_T_main.npy": "T", "u_I_main.npy": "I",
    "u_enc_main.npy": "enc", "u_dec_main.npy": "dec",
}
PHASE_CHECK_N_C = 24   # matches tier3_gate_algebra/tier4_enc_relative_phase's own default
PHASE_DIFF_FLAG_THRESHOLD = 1e-3   # looser than max_diff's 1e-4: F_avg_gate is a squared/combined quantity


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


def qutip_phase_cross_check(gate, u, n_c=PHASE_CHECK_N_C, n_t=N_T, dt=DT):
    """
    Independent QuTiP-sesolve confirmation of F_avg_gate -- the relative-
    branch-phase-sensitive check tier3_gate_algebra/tier4_enc_relative_phase/
    tier4_dec_relative_phase compute using grape_core's own eigendecomposition
    propagator. Building the *same* 2x2 U_log via a completely independent
    ODE solver and comparing F_avg_gate from each rules out the possibility
    that grape_core's propagator itself has a phase-convention bug that
    happens to move F_avg_gate for reasons unrelated to the pulse -- unlike
    validate_pulse/qutip_multi_state_fidelity above, which only ever compare
    the phase-blind per-branch fidelity and would agree with each other even
    if the underlying pulse had the wrong relative phase.

    Returns (F_avg_gc, F_avg_qt) -- None, None if `gate` isn't one of the
    eight two-branch gates this applies to (i.e. not in PHASE_CHECK_GATE).
    """
    if gate in IDEAL_LOGICAL_U:
        F_avg_gc = tier3_gate_algebra(gate, u, n_c=n_c, n_t=n_t, dt=dt)["F_avg_gate"]
        U_log_qt = qutip_gate_logical_unitary(u, n_t, n_c, dt=dt)
        F_avg_qt = average_gate_fidelity(IDEAL_LOGICAL_U[gate], U_log_qt)
    elif gate == "enc":
        F_avg_gc = tier4_enc_relative_phase(u, n_c=n_c, n_t=n_t, dt=dt)["F_avg_gate"]
        U_log_qt = qutip_enc_logical_unitary(u, n_t, n_c, dt=dt)
        F_avg_qt = average_gate_fidelity(np.eye(2, dtype=complex), U_log_qt)
    elif gate == "dec":
        F_avg_gc = tier4_dec_relative_phase(u, n_c=n_c, n_t=n_t, dt=dt)["F_avg_gate"]
        U_log_qt = qutip_dec_logical_unitary(u, n_t, n_c, dt=dt)
        F_avg_qt = average_gate_fidelity(np.eye(2, dtype=complex), U_log_qt)
    else:
        return None, None
    return float(F_avg_gc), float(F_avg_qt)


def main():
    print("QuTiP cross-check of every saved *_mt.npy pulse against grape_core's own propagator.")
    summary = []
    rows = []
    phase_rows = []
    for filename, (label, factory) in PULSE_MAP.items():
        path = os.path.join(PULSE_DIR, filename)
        if not os.path.exists(path):
            print(f"\n[SKIP] {label}: {path} not found")
            continue
        max_diff = validate_pulse(filename, label, factory, rows)
        summary.append((label, max_diff))

        gate = PHASE_CHECK_GATE.get(filename)
        if gate is not None:
            u = np.load(path)
            F_avg_gc, F_avg_qt = qutip_phase_cross_check(gate, u)
            phase_diff = abs(F_avg_gc - F_avg_qt)
            phase_rows.append({
                "label": label, "F_avg_gc": F_avg_gc, "F_avg_qt": F_avg_qt, "diff": phase_diff,
            })
            print(f"  [{label}] F_avg_gate: grape_core={F_avg_gc:.6f}  qutip={F_avg_qt:.6f}  "
                  f"|diff|={phase_diff:.2e}")

    print(f"\n{'='*60}")
    print("SUMMARY (max |grape_core F - qutip F| across truncations)")
    print(f"{'='*60}")
    for label, max_diff in summary:
        flag = "  <-- CHECK THIS" if max_diff > 1e-4 else ""
        print(f"  {label:14s} {max_diff:.2e}{flag}")

    print(f"\n{'='*60}")
    print("SUMMARY (relative-branch-phase check: |F_avg_gate(grape_core) - F_avg_gate(qutip)|)")
    print(f"{'='*60}")
    for r in phase_rows:
        flag = "  <-- CHECK THIS" if r["diff"] > PHASE_DIFF_FLAG_THRESHOLD else ""
        print(f"  {r['label']:14s} gc={r['F_avg_gc']:.6f}  qt={r['F_avg_qt']:.6f}  "
              f"diff={r['diff']:.2e}{flag}")

    os.makedirs(TABLE_DIR, exist_ok=True)
    csv_path = os.path.join(TABLE_DIR, "qutip_validation.csv")
    with open(csv_path, "w") as f:
        f.write("label,n_c,F_gc,F_qt,diff,trained\n")
        for r in rows:
            f.write(f"{r['label']},{r['n_c']},{r['F_gc']!r},{r['F_qt']!r},{r['diff']!r},{r['trained']}\n")
    print(f"\nWrote per-truncation data: {csv_path}")

    phase_csv_path = os.path.join(TABLE_DIR, "qutip_phase_validation.csv")
    with open(phase_csv_path, "w") as f:
        f.write("label,F_avg_gc,F_avg_qt,diff\n")
        for r in phase_rows:
            f.write(f"{r['label']},{r['F_avg_gc']!r},{r['F_avg_qt']!r},{r['diff']!r}\n")
    print(f"Wrote relative-phase cross-check: {phase_csv_path}")


if __name__ == "__main__":
    main()
