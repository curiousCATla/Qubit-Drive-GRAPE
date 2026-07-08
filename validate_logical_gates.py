#!/usr/bin/env python3
"""
validate_logical_gates.py

Comprehensive validation suite for GRAPE-optimized logical gates on the
4-component even cat code (α=√3) in a transmon-cavity system (n_t=3).

This script implements the validation plan for the project:
- Tier 1: Fidelity robustness across truncations (extend existing validate_pulse_truncations)
- Tier 2: Explicit logical action + code-space preservation / leakage analysis
- Tier 3: Gate algebra & self-consistency (X²≈I, H²≈I, conjugation relations, phase for T)
- Tier 4: Full encode–logical_gate–decode pipeline fidelity (the key "usable logical qubit" test)
- Tier 5: Effective logical unitary extraction + closeness to ideal single-qubit gate

Run this after refining pulses with refine_and_compare.py (or optimize_multi_state_pulse).
It produces clean pandas tables and actionable diagnostics.

Usage:
    python validate_logical_gates.py

Expected pulse files (in ./pulses/ relative to run dir or absolute):
    u_enc_refined_t3.npy, u_dec_refined_t3.npy (or v1)
    u_X_refined_t3.npy, u_Y_refined_t3.npy, u_Z_refined_t3.npy,
    u_H_refined_t3.npy, u_T_refined_t3.npy, u_I_logical_v1.npy (or refined)

Author: Project mentor (Grok) + Ian Dong
Date: 2026-07-06
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.linalg import norm

# ============================================================
# PROJECT IMPORTS (adjust path if running from elsewhere)
# ============================================================
# Try common locations so it works whether you run from artifacts/, attachments/, or project root
possible_paths = [
    os.path.dirname(__file__),
    "/home/workdir/attachments",
    "/home/workdir/artifacts",
    ".",
    ".."
]
for p in possible_paths:
    if os.path.exists(os.path.join(p, "cat_code.py")):
        sys.path.insert(0, p)
        break

from cat_code import (
    get_logical_cat_states,
    embed_in_joint_space,
    get_encode_state_pairs,
    get_decode_state_pairs,
    get_encode_targets,          # needed for Tier 4 ideal state construction
    get_logical_X_state_pairs,
    get_logical_Y_state_pairs,
    get_logical_Z_state_pairs,
    get_logical_H_state_pairs,
    get_logical_T_state_pairs,
    get_identity_state_pairs,
    validate_pulse_truncations,
)
from grape_core import make_hamiltonian, step_data

# ============================================================
# CONFIGURATION
# ============================================================
PULSE_DIR = "pulses"          # change if your pulses live elsewhere
DT = 0.002                    # µs, same as optimization
N_T = 3                       # transmon levels (must match refinement)
ALPHA = np.sqrt(3.0)

# Expected refined pulse filenames (from refine_and_compare.py convention)
GATE_PULSE_MAP = {
    "enc": os.path.join(PULSE_DIR, "u_enc_refined_t3.npy"),
    "dec": os.path.join(PULSE_DIR, "u_dec_refined_t3.npy"),
    "X":   os.path.join(PULSE_DIR, "u_X_refined_t3.npy"),
    "Y":   os.path.join(PULSE_DIR, "u_Y_refined_t3.npy"),
    "Z":   os.path.join(PULSE_DIR, "u_Z_refined_t3.npy"),
    "H":   os.path.join(PULSE_DIR, "u_H_refined_t3.npy"),
    "T":   os.path.join(PULSE_DIR, "u_T_refined_t3.npy"),
    "I":   os.path.join(PULSE_DIR, "u_I_refined_t3.npy"),   # or u_I_logical_v1.npy
}

# Wide validation range (used in Tier 1)
VALIDATION_TRUNC_RANGE = list(range(18, 33, 2))

# ============================================================
# HELPER: Robust propagate (works with n_t=3)
# ============================================================
def propagate_pulse(u, H0, Hc, psi0, dt=DT):
    """Propagate initial state through the full control pulse u."""
    psi = psi0.copy().astype(complex)
    for uk in u:
        Uk, _, _ = step_data(H0, Hc, uk, dt)
        psi = Uk @ psi
    return psi

# ============================================================
# TIER 1: FIDELITY ROBUSTNESS (wrapper + summary table)
# ============================================================
def tier1_fidelity_robustness(gate_name, get_state_pairs, u, title=None):
    """Run wide truncation validation and return nice DataFrame + stats."""
    if title is None:
        title = f"{gate_name} - Wide Truncation Validation (n_t={N_T})"
    print(f"\n{'='*70}")
    print(title)
    print('='*70)
    results = validate_pulse_truncations(
        u=u,
        get_targets_func=get_state_pairs,
        trunc_range=VALIDATION_TRUNC_RANGE,
        n_t=N_T,
        dt=DT,
        title=title
    )
    # Build DataFrame for pretty printing / saving
    df = pd.DataFrame({
        'n_c': list(results.keys()),
        'Fidelity': list(results.values())
    })
    df['Deviation_from_max'] = df['Fidelity'].max() - df['Fidelity']
    mean_f = df['Fidelity'].mean()
    min_f = df['Fidelity'].min()
    std_f = df['Fidelity'].std()
    print(f"\nSummary: mean={mean_f:.6f} | min={min_f:.6f} | std={std_f:.6f} | range={min_f:.6f}–{df['Fidelity'].max():.6f}")
    return df, {'mean': mean_f, 'min': min_f, 'std': std_f}

# ============================================================
# TIER 2: LOGICAL ACTION + LEAKAGE / CODE-SPACE PRESERVATION
# ============================================================
def tier2_logical_action_and_leakage(gate_name, get_state_pairs, u, n_c=24, n_t=N_T, dt=DT):
    """
    For the two logical basis states:
      - Recompute state fidelity (sanity)
      - Measure transmon |g> population after gate (should be ~1)
      - Measure population on odd photon numbers (should be ~0 for even cat code)
      - Measure population outside the ideal logical support (mod-4 subspaces)
    Returns detailed per-basis-state report + aggregate leakage metrics.
    """
    print(f"\n{'='*70}")
    print(f"TIER 2: Logical Action + Leakage Analysis — {gate_name} (n_c={n_c}, n_t={n_t})")
    print('='*70)

    H0, Hc = make_hamiltonian(n_t, n_c)
    pairs = get_state_pairs(n_c=n_c, n_t=n_t)
    assert len(pairs) == 2, "Expected exactly two state pairs for logical gates"

    records = []
    for idx, (psi_i, psi_f) in enumerate(pairs):
        label = "+Z_L" if idx == 0 else "-Z_L"
        psi_out = propagate_pulse(u, H0, Hc, psi_i, dt)

        # 1. State fidelity to ideal target
        fid = np.abs(np.vdot(psi_f, psi_out))**2

        # 2. Transmon population in |g> (indices 0:n_c)
        p_g = np.sum(np.abs(psi_out[:n_c])**2)
        leakage_transmon = 1.0 - p_g

        # 3. Odd photon number population (in transmon g subspace)
        cavity_amps = psi_out[:n_c]
        odd_pop = np.sum(np.abs(cavity_amps[1::2])**2)

        # 4. Population outside correct mod-4 logical support
        #    +Z_L support: n % 4 == 0 ;  -Z_L support: n % 4 == 2
        n = np.arange(n_c)
        if idx == 0:  # +Z_L
            wrong_mod4_pop = np.sum(np.abs(cavity_amps[n % 4 != 0])**2)
        else:         # -Z_L
            wrong_mod4_pop = np.sum(np.abs(cavity_amps[n % 4 != 2])**2)

        total_leakage = leakage_transmon + odd_pop + wrong_mod4_pop   # approximate (some overlap)

        records.append({
            'basis_state': label,
            'state_fidelity': fid,
            'p_transmon_g': p_g,
            'odd_photon_pop': odd_pop,
            'wrong_mod4_pop': wrong_mod4_pop,
            'approx_total_leakage': total_leakage
        })

        print(f"  {label:6s} | F={fid:.6f} | p_g={p_g:.6f} | odd_pop={odd_pop:.2e} | wrong_mod4={wrong_mod4_pop:.2e}")

    df = pd.DataFrame(records)
    avg_fid = df['state_fidelity'].mean()
    max_leak = df['approx_total_leakage'].max()
    print(f"\n  Avg logical fidelity: {avg_fid:.6f}")
    print(f"  Max approx leakage (any basis): {max_leak:.2e}")
    if max_leak > 1e-3:
        print("  ⚠️  WARNING: Significant leakage detected — consider more refinement or wider training.")
    else:
        print("  ✅  Leakage well suppressed (<0.1%).")
    return df

# ============================================================
# TIER 3: GATE ALGEBRA & SELF-CONSISTENCY
# ============================================================
def tier3_gate_algebra(gate_name, u, n_c=24, n_t=N_T, dt=DT):
    """
    Check fundamental relations:
      - For X,Y,Z:  U_gate² ≈ Identity on logical subspace (fidelity of composed action)
      - For H:      H² ≈ I   and   H X H ≈ Z   (conjugation, via effective logical unitaries)
      - For T:      Check relative phase arg(U_mm) ≈ +π/4 on |-Z_L>
      - For all:    Extract effective 2x2 logical unitary and check unitarity (U U† ≈ I)
    """
    print(f"\n{'='*70}")
    print(f"TIER 3: Gate Algebra & Unitarity Check — {gate_name}")
    print('='*70)

    H0, Hc = make_hamiltonian(n_t, n_c)

    # Prepare logical basis states
    psi_p_cav, psi_m_cav = get_logical_cat_states(alpha=ALPHA, n_c=n_c)
    psi_pg = embed_in_joint_space(psi_p_cav, n_t=n_t, n_c=n_c, t_level=0)
    psi_mg = embed_in_joint_space(psi_m_cav, n_t=n_t, n_c=n_c, t_level=0)

    # Apply gate once
    psi_p_out = propagate_pulse(u, H0, Hc, psi_pg, dt)
    psi_m_out = propagate_pulse(u, H0, Hc, psi_mg, dt)

    # Effective logical unitary matrix elements (in {|+Z_L>, |-Z_L>} basis)
    U_pp = np.vdot(psi_pg, psi_p_out)
    U_pm = np.vdot(psi_mg, psi_p_out)
    U_mp = np.vdot(psi_pg, psi_m_out)
    U_mm = np.vdot(psi_mg, psi_m_out)
    U_log = np.array([[U_pp, U_pm],
                      [U_mp, U_mm]])

    # Check unitarity of extracted logical map
    U_dag_U = U_log.conj().T @ U_log
    unitarity_err = np.linalg.norm(U_dag_U - np.eye(2))
    det = np.linalg.det(U_log)
    print(f"  Extracted logical U (|+Z>, |-Z> basis):")
    print(np.round(U_log, decimals=6))
    print(f"  Unitarity error ||U†U - I|| = {unitarity_err:.2e}   (ideal 0)")
    print(f"  det(U) = {det:.6f}   (ideal |det|=1)")

    # Gate-specific algebra checks
    if gate_name in ["X", "Y", "Z", "I"]:
        # Apply gate twice
        psi_p2 = propagate_pulse(u, H0, Hc, psi_p_out, dt)
        psi_m2 = propagate_pulse(u, H0, Hc, psi_m_out, dt)
        fid_p = np.abs(np.vdot(psi_pg, psi_p2))**2
        fid_m = np.abs(np.vdot(psi_mg, psi_m2))**2
        avg_fid_sq = (fid_p + fid_m) / 2
        print(f"  {gate_name}² fidelity to Identity: {avg_fid_sq:.6f}   (should be ~1.000)")
        if avg_fid_sq < 0.99:
            print("  ⚠️  Significant deviation from U² = I — possible leakage or optimization issue.")

    if gate_name == "H":
        # H² should be I
        psi_p2 = propagate_pulse(u, H0, Hc, psi_p_out, dt)
        psi_m2 = propagate_pulse(u, H0, Hc, psi_m_out, dt)
        fid_p = np.abs(np.vdot(psi_pg, psi_p2))**2
        fid_m = np.abs(np.vdot(psi_mg, psi_m2))**2
        print(f"  H² fidelity to I: {(fid_p + fid_m)/2:.6f}")

        # Also check conjugation H X H ~ Z would require loading X pulse too — skipped for now
        # (can be added if all pulses present)

    if gate_name == "T":
        # T should apply +1 to |+Z_L> and e^{i π/4} to |-Z_L>
        # Check relative phase
        phase_m = np.angle(U_mm / U_pp) if abs(U_pp) > 1e-8 else np.angle(U_mm)
        expected = np.pi / 4
        phase_err = abs(phase_m - expected)
        print(f"  T relative phase on |-Z_L>: {phase_m:.6f} rad  (expected +π/4 = {expected:.6f})")
        print(f"  Phase error: {phase_err:.2e} rad")
        if phase_err > 0.05:
            print("  ⚠️  Phase deviates significantly from π/4 — check optimization or α value.")

    return {
        'U_log': U_log,
        'unitarity_err': unitarity_err,
        'det': det
    }

# ============================================================
# TIER 4: ENCODE — LOGICAL_GATE — DECODE PIPELINE
# ============================================================
def tier4_enc_gate_dec_pipeline(gate_name, u_gate, u_enc=None, u_dec=None, n_c_list=None, dt=DT):
    """
    The most important practical validation:
    U_dec ∘ U_gate ∘ U_enc  should implement the logical gate action
    directly on the transmon computational subspace {|g,0>, |e,0>}.
    
    For X: |g0> → |e0>, |e0> → |g0>
    For Z: |g0> → |g0>, |e0> → -|e0>
    For H: |g0> → (|g0> + |e0>)/√2 , etc.
    """
    if n_c_list is None:
        n_c_list = [20, 24, 28]

    if u_enc is None or u_dec is None:
        enc_path = GATE_PULSE_MAP.get("enc")
        dec_path = GATE_PULSE_MAP.get("dec")
        if os.path.exists(enc_path) and os.path.exists(dec_path):
            u_enc = np.load(enc_path)
            u_dec = np.load(dec_path)
        else:
            print(f"\n[SKIP] Tier 4 pipeline for {gate_name}: u_enc or u_dec not found.")
            return None

    print(f"\n{'='*70}")
    print(f"TIER 4: Encode–{gate_name}–Decode Pipeline Fidelity (n_t={N_T})")
    print('='*70)

    results = {}
    for nc in n_c_list:
        H0, Hc = make_hamiltonian(N_T, nc)

        # Build computational initial states for THIS nc
        init_g = np.zeros(N_T * nc, dtype=complex); init_g[0] = 1.0
        init_e = np.zeros(N_T * nc, dtype=complex); init_e[nc] = 1.0

        # Logical cat states (target of enc)
        psi_plus_cav, psi_minus_cav = get_logical_cat_states(alpha=ALPHA, n_c=nc)
        psi_plus_g  = embed_in_joint_space(psi_plus_cav,  n_t=N_T, n_c=nc, t_level=0)
        psi_minus_g = embed_in_joint_space(psi_minus_cav, n_t=N_T, n_c=nc, t_level=0)

        # Apply logical gate to cats  →  build ideal final states for pipeline
        if gate_name == "X":
            # X: +Z → -Z , -Z → +Z   ⇒  pipeline maps |g0>→|e0>, |e0>→|g0>
            target_after_gate_plus  = psi_minus_g.copy()
            target_after_gate_minus = psi_plus_g.copy()
            ideal_final_g = init_e.copy()
            ideal_final_e = init_g.copy()
        elif gate_name == "Z":
            # Z: +Z → +Z , -Z → -|-Z>   ⇒  |g0>→|g0>, |e0>→ -|e0>
            target_after_gate_plus  = psi_plus_g.copy()
            target_after_gate_minus = -psi_minus_g.copy()
            ideal_final_g = init_g.copy()
            ideal_final_e = -init_e.copy()
        elif gate_name == "H":
            # H on logical  ⇒  H on comp basis
            psi_H_plus  = (psi_plus_g + psi_minus_g) / np.sqrt(2)
            psi_H_minus = (psi_plus_g - psi_minus_g) / np.sqrt(2)
            target_after_gate_plus  = psi_H_plus
            target_after_gate_minus = psi_H_minus
            ideal_final_g = (init_g + init_e) / np.sqrt(2)
            ideal_final_e = (init_g - init_e) / np.sqrt(2)
        elif gate_name == "T":
            phase = np.exp(1j * np.pi / 4)
            target_after_gate_plus  = psi_plus_g.copy()
            target_after_gate_minus = phase * psi_minus_g.copy()
            ideal_final_g = init_g.copy()
            ideal_final_e = phase * init_e.copy()
        elif gate_name == "Y":
            # Y: +Z → -i|-Z>, -Z → i|+Z>  ⇒  |g0> → -i|e0>, |e0> → i|g0>
            target_after_gate_plus  = -1j * psi_minus_g
            target_after_gate_minus = 1j * psi_plus_g
            ideal_final_g = -1j * init_e.copy()
            ideal_final_e = 1j * init_g.copy()
        else:
            # Identity
            target_after_gate_plus  = psi_plus_g.copy()
            target_after_gate_minus = psi_minus_g.copy()
            ideal_final_g = init_g.copy()
            ideal_final_e = init_e.copy()

        # Now run the pipeline on |g,0> and |e,0>
        init_g = np.zeros(N_T * nc, dtype=complex); init_g[0] = 1.0
        init_e = np.zeros(N_T * nc, dtype=complex); init_e[nc] = 1.0

        # Encode
        after_enc_g = propagate_pulse(u_enc, H0, Hc, init_g, dt)
        after_enc_e = propagate_pulse(u_enc, H0, Hc, init_e, dt)

        # Logical gate
        after_gate_g = propagate_pulse(u_gate, H0, Hc, after_enc_g, dt)
        after_gate_e = propagate_pulse(u_gate, H0, Hc, after_enc_e, dt)

        # Decode
        final_g = propagate_pulse(u_dec, H0, Hc, after_gate_g, dt)
        final_e = propagate_pulse(u_dec, H0, Hc, after_gate_e, dt)

        # Fidelities to ideal
        fid_g = np.abs(np.vdot(ideal_final_g, final_g))**2
        fid_e = np.abs(np.vdot(ideal_final_e, final_e))**2
        avg_fid = (fid_g + fid_e) / 2.0

        results[nc] = {'fid_g': fid_g, 'fid_e': fid_e, 'avg': avg_fid}
        print(f"  n_c={nc:2d}: |g0>→ideal F={fid_g:.6f}   |e0>→ideal F={fid_e:.6f}   avg={avg_fid:.6f}")

    overall = np.mean([r['avg'] for r in results.values()])
    print(f"\n  Overall pipeline avg fidelity: {overall:.6f}")
    if overall > 0.98:
        print("  ✅  Excellent — logical gate correctly transferred to computational subspace.")
    elif overall > 0.95:
        print("  ⚠️  Good but room for improvement (refine further or check U_enc/U_dec).")
    else:
        print("  ❌  Low pipeline fidelity — investigate leakage or gate optimization.")
    return results

# ============================================================
# MAIN VALIDATION RUNNER
# ============================================================
def main():
    print("\n" + "="*70)
    print("LOGICAL GATE VALIDATION SUITE — 4-Component Even Cat Code (α=√3, n_t=3)")
    print("Reproducing & extending Heeres et al. (2017) GRAPE results")
    print("="*70)

    os.makedirs(PULSE_DIR, exist_ok=True)

    # Gate factory mapping
    GATE_FACTORIES = {
        "X": get_logical_X_state_pairs,
        "Y": get_logical_Y_state_pairs,
        "Z": get_logical_Z_state_pairs,
        "H": get_logical_H_state_pairs,
        "T": get_logical_T_state_pairs,
        "I": get_identity_state_pairs,
    }

    summary_rows = []

    for gate_name, factory in GATE_FACTORIES.items():
        pulse_path = GATE_PULSE_MAP.get(gate_name)
        if not pulse_path or not os.path.exists(pulse_path):
            # Try fallback names
            fallbacks = {
                "I": os.path.join(PULSE_DIR, "u_I_logical_v1.npy"),
            }
            alt = fallbacks.get(gate_name)
            if alt and os.path.exists(alt):
                pulse_path = alt
            else:
                print(f"\n[SKIP] {gate_name}: pulse not found at {pulse_path}")
                continue

        print(f"\n{'#'*70}")
        print(f"VALIDATING GATE: {gate_name}")
        print(f"Loaded: {pulse_path}")
        u = np.load(pulse_path)
        print(f"Shape: {u.shape}")

        # Tier 1
        df_fid, stats = tier1_fidelity_robustness(gate_name, factory, u)

        # Tier 2
        df_leak = tier2_logical_action_and_leakage(gate_name, factory, u)

        # Tier 3
        alg = tier3_gate_algebra(gate_name, u)

        # Tier 4 (only if enc/dec exist)
        pipe = tier4_enc_gate_dec_pipeline(gate_name, u)

        # Collect summary
        summary_rows.append({
            'Gate': gate_name,
            'F_mean': stats['mean'],
            'F_min': stats['min'],
            'F_std': stats['std'],
            'Unitarity_err': alg['unitarity_err'],
            'Pipeline_avg': pipe['avg'] if pipe else np.nan,
            'Pulse_file': os.path.basename(pulse_path)
        })

    # Final summary table
    if summary_rows:
        print("\n" + "="*70)
        print("OVERALL VALIDATION SUMMARY")
        print("="*70)
        df_sum = pd.DataFrame(summary_rows)
        print(df_sum.to_string(index=False, float_format="%.6f"))
        print("\nInterpretation guide:")
        print("  • F_mean / F_min : higher is better; aim >0.99 / >0.98")
        print("  • Unitarity_err  : <1e-3 excellent (extracted logical map is unitary)")
        print("  • Pipeline_avg   : measures how well logical gate works in full enc/gate/dec stack")
        print("                     (most important number for practical use of the logical qubit)")

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("Next steps if any metric is below target:")
    print("  1. Run refine_and_compare.py for under-performing gates (increase extra_maxiter, tune penalty_scale)")
    print("  2. Widen TRAINING trunc_list in refinement (e.g. [20,22,24,26,28])")
    print("  3. Check pulse smoothness with pulse_viz.py — overly sharp pulses can cause leakage")
    print("  4. Re-run this validation script after each refinement iteration")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
