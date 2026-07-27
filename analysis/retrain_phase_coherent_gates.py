#!/usr/bin/env python3
"""
retrain_phase_coherent_gates.py

Re-optimize the logical gates whose correctness depends on the *relative*
phase between their |+Z_L>/|-Z_L> training pairs, using
core.grape_core.coherent_fidelity_multi_state instead of the production
recipe's default fidelity_multi_state.

Background: fidelity_multi_state trains on the simple average of
|<f_m|U|i_m>|^2 per state pair, which only requires each output to match
its own target up to an ARBITRARY, independent global phase -- it cannot
see or constrain the phase *between* different pairs of the same gate.
validation/validate_logical_gates.py's tier3_gate_algebra (X/Y/Z/H/T/I) and
tier4_enc_relative_phase/tier4_dec_relative_phase (enc/dec) now report this
directly via F_avg_gate (average gate fidelity of the extracted 2x2 U_log
vs. the ideal target -- IDEAL_LOGICAL_U[gate] for the six logical gates,
the identity for enc/dec -- sensitive to relative phase). It caught the
consequence in every production pulse trained with the phase-blind
objective:
  - T (originally): measured relative phase 0.344 rad vs pi/4 = 0.785 rad.
  - H (originally): H^2 fidelity to Identity = 0.001 (should be ~1).
  - X/Y/Z/I (originally): F_avg_gate = 0.95 / 0.38 / 0.35 / 0.80
    respectively, despite per-branch F_mean >= 0.996 for all four -- the
    self-composition checks (U^2=I) that used to be the only "algebra"
    check for these four are themselves blind to this defect, since
    applying a phase-scrambled swap/diagonal twice cancels the per-branch
    phases regardless of what they are.
  - enc/dec (still unfixed as of this writing): F_avg_gate = 0.62 / 0.33
    respectively -- tier4_enc_gate_dec_pipeline is likewise blind to this,
    since it only ever propagates |g,0>/|e,0> individually through the
    whole enc-gate-dec chain, never a superposition.
coherent_fidelity_multi_state sums the raw overlaps before squaring
(F = |sum_m v_m|^2 / M^2), which is only maximal when every branch matches
its target with the SAME global phase -- exactly the property needed here.

All pulses are warm-started from the existing production pulses (already
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
from core.cat_code import (
    get_logical_T_state_pairs, get_logical_H_state_pairs,
    get_logical_X_state_pairs, get_logical_Y_state_pairs,
    get_logical_Z_state_pairs, get_identity_state_pairs,
    get_encode_state_pairs, get_decode_state_pairs,
)
from validation.validate_logical_gates import (
    GATE_PULSE_MAP, tier1_fidelity_robustness, tier3_gate_algebra,
    tier4_enc_relative_phase, tier4_dec_relative_phase,
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
    "X": get_logical_X_state_pairs,
    "Y": get_logical_Y_state_pairs,
    "Z": get_logical_Z_state_pairs,
    "I": get_identity_state_pairs,
    "enc": get_encode_state_pairs,
    "dec": get_decode_state_pairs,
}

# F_avg_gate (average gate fidelity of the extracted 2x2 U_log vs. the ideal
# target) is the generic, gate-agnostic pass criterion: it's the metric that
# actually depends on relative branch phase for every gate in GATES, unlike
# U^2=I (blind to it for X/Y/Z/I) or tier4_enc_gate_dec_pipeline (blind to
# it for enc/dec) or the T/H-specific checks below (kept as extra
# corroborating evidence).
F_AVG_GATE_TARGET = 0.99


def _f_avg_gate(gate, u):
    """Dispatch to the right relative-phase check: enc/dec don't have an
    IDEAL_LOGICAL_U entry (tier3_gate_algebra propagates the cat states,
    which is the wrong input space for enc and meaningless for a check
    that should stand on its own for dec), so they get their own dedicated
    single-pulse checks instead."""
    if gate == "enc":
        return tier4_enc_relative_phase(u)["F_avg_gate"]
    if gate == "dec":
        return tier4_dec_relative_phase(u)["F_avg_gate"]
    return tier3_gate_algebra(gate, u)["F_avg_gate"]


def retrain_gate(gate, factory):
    pulse_path = GATE_PULSE_MAP[gate]
    u_old = np.load(pulse_path)

    # Idempotent: if the pulse already on disk passes the phase-sensitive
    # check, don't waste compute (or overwrite the backup) re-optimizing it.
    current_f_avg = _f_avg_gate(gate, u_old)
    if current_f_avg >= F_AVG_GATE_TARGET:
        print(f"[{gate}] already passes F_avg_gate = {current_f_avg:.6f} "
              f">= {F_AVG_GATE_TARGET} -- skipping retrain.")
        return True

    backup_path = pulse_path + ".prephase.bak"
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
    alg = tier3_gate_algebra(gate, u_new) if gate not in ("enc", "dec") else None
    df_fid, stats = tier1_fidelity_robustness(gate, factory, u_new)

    ok = True

    f_avg_gate = alg["F_avg_gate"] if alg is not None else _f_avg_gate(gate, u_new)
    print(f"[{gate}] F_avg_gate = {f_avg_gate:.6f} (require >= {F_AVG_GATE_TARGET})")
    ok &= f_avg_gate >= F_AVG_GATE_TARGET

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
