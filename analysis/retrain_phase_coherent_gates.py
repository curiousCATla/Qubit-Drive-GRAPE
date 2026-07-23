#!/usr/bin/env python3
"""
retrain_phase_coherent_gates.py

Re-optimize U_T and U_H -- the only two gates in the logical gate set whose
correctness depends on the *relative* phase between their |+Z_L>/|-Z_L>
training pairs -- using core.grape_core.coherent_fidelity_multi_state
instead of the production recipe's default fidelity_multi_state.

Background: fidelity_multi_state trains on the simple average of
|<f_m|U|i_m>|^2 per state pair, which only requires each output to match
its own target up to an ARBITRARY, independent global phase -- it cannot
see or constrain the phase *between* different pairs of the same gate.
validation/validate_logical_gates.py's tier3_gate_algebra caught the
consequence directly in the production pulses:
  - T: measured relative phase 0.344 rad vs the required pi/4 = 0.785 rad.
  - H: H^2 fidelity to Identity = 0.001 (should be ~1) -- H is not actually
    self-inverse.
coherent_fidelity_multi_state sums the raw overlaps before squaring
(F = |sum_m v_m|^2 / M^2), which is only maximal when every branch matches
its target with the SAME global phase -- exactly the property needed here.

Both pulses are warm-started from the existing production pulses (already
near-perfect per-branch magnitude/leakage -- only the relative phase needs
correcting), so this is a short refinement, not a from-scratch optimization.

Usage:
    python -m analysis.retrain_phase_coherent_gates
"""
import os
import shutil
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.grape_core import coherent_fidelity_multi_state
from core.optimizer import optimize_multi_state_pulse
from core.cat_code import get_logical_T_state_pairs, get_logical_H_state_pairs
from validation.validate_logical_gates import (
    GATE_PULSE_MAP, tier1_fidelity_robustness, tier3_gate_algebra,
)

N_T = 3
N_STEPS = 550
DT = 0.002

# Same production recipe as experiments.ipynb Section 1 (OPTIMIZATION_RECIPE).
TRUNC_LIST = [22, 24, 26]
PENALTIES = {"deriv": 1e-5, "boundary": 2e-5, "amp": 8e-5, "amp_max": 40.0, "disc": 0.5}
CAV_BAND = (-27.0, 27.0)
TRA_BAND = (-33.0, 33.0)
HARD_AMP_LIMIT = 40.0
N_JOBS = 3

REFINE_MAXITER = 600

GATES = {
    "T": get_logical_T_state_pairs,
    "H": get_logical_H_state_pairs,
}


def retrain_gate(gate, factory):
    pulse_path = GATE_PULSE_MAP[gate]
    backup_path = pulse_path + ".prephase.bak"
    u_old = np.load(pulse_path)
    shutil.copy(pulse_path, backup_path)
    print(f"[{gate}] backed up {pulse_path} -> {backup_path}")

    print(f"\n{'='*70}\nRetraining U_{gate} with coherent_fidelity_multi_state\n{'='*70}")
    u_new, info = optimize_multi_state_pulse(
        get_state_pairs=factory,
        trunc_list=TRUNC_LIST,
        n_t=N_T,
        N=N_STEPS,
        dt=DT,
        penalties=PENALTIES,
        warm_start=u_old,
        maxiter=REFINE_MAXITER,
        cav_band=CAV_BAND,
        tra_band=TRA_BAND,
        hard_amp_limit=HARD_AMP_LIMIT,
        n_jobs=N_JOBS,
        fidelity_fn=coherent_fidelity_multi_state,
        verbose=True,
    )

    print(f"\n[{gate}] Validating retrained pulse...")
    alg = tier3_gate_algebra(gate, u_new)
    df_fid, stats = tier1_fidelity_robustness(gate, factory, u_new)

    ok = True
    if gate == "T":
        phase_m = np.angle(alg["U_log"][1, 1] / alg["U_log"][0, 0])
        phase_err = abs(phase_m - np.pi / 4)
        print(f"[{gate}] phase_err = {phase_err:.4e} (require < 0.05)")
        ok &= phase_err < 0.05
    if gate == "H":
        # Recompute H^2 fidelity to Identity directly (tier3 prints it but
        # doesn't return it).
        from core.grape_core import make_hamiltonian
        from core.cat_code import get_logical_cat_states, embed_in_joint_space
        from validation.validate_logical_gates import propagate_pulse
        H0, Hc = make_hamiltonian(N_T, 24)
        psi_p_cav, psi_m_cav = get_logical_cat_states(alpha=np.sqrt(3.0), n_c=24)
        psi_pg = embed_in_joint_space(psi_p_cav, n_t=N_T, n_c=24, t_level=0)
        psi_mg = embed_in_joint_space(psi_m_cav, n_t=N_T, n_c=24, t_level=0)
        psi_p_out = propagate_pulse(u_new, H0, Hc, psi_pg, DT)
        psi_m_out = propagate_pulse(u_new, H0, Hc, psi_mg, DT)
        psi_p2 = propagate_pulse(u_new, H0, Hc, psi_p_out, DT)
        psi_m2 = propagate_pulse(u_new, H0, Hc, psi_m_out, DT)
        fid_p = np.abs(np.vdot(psi_pg, psi_p2)) ** 2
        fid_m = np.abs(np.vdot(psi_mg, psi_m2)) ** 2
        h2_fid = (fid_p + fid_m) / 2
        print(f"[{gate}] H^2 fidelity to I = {h2_fid:.6f} (require > 0.99)")
        ok &= h2_fid > 0.99

    ok &= stats["mean"] >= 0.985 and stats["min"] >= 0.985

    if ok:
        np.save(pulse_path, u_new)
        print(f"[{gate}] PASSED validation -- saved retrained pulse to {pulse_path}")
    else:
        print(f"[{gate}] FAILED validation -- NOT overwriting {pulse_path}. "
              f"Original pulse remains in place; backup at {backup_path}.")
    return ok


if __name__ == "__main__":
    results = {}
    for gate, factory in GATES.items():
        results[gate] = retrain_gate(gate, factory)
    print("\n" + "=" * 70)
    print("Retrain summary:", results)
    print("=" * 70)
