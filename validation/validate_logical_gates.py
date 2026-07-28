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

Run this after cold-start (or refined) pulses are saved under ./pulses/.
It produces clean pandas tables and actionable diagnostics.

Usage:
    python validation/validate_logical_gates.py

Expected pulse files (in ./pulses/):
    u_enc_main.npy, u_dec_main.npy
    u_X_main.npy, u_Y_main.npy, u_Z_main.npy,
    u_H_main.npy, u_T_main.npy, u_I_main.npy

Author: Project mentor (Grok) + Ian Dong
Date: 2026-07-06
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.linalg import norm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.cat_code import (
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
from core.grape_core import make_hamiltonian, step_data

# ============================================================
# CONFIGURATION
# ============================================================
PULSE_DIR = "pulses"          # change if your pulses live elsewhere
DT = 0.002                    # µs, same as optimization
N_T = 3                       # transmon levels (must match refinement)
ALPHA = np.sqrt(3.0)

# Eq. 23 + Eq. 24 cold-start pulses (same set as compare_pulses.py)
GATE_PULSE_MAP = {
    "enc": os.path.join(PULSE_DIR, "u_enc_main.npy"),
    "dec": os.path.join(PULSE_DIR, "u_dec_main.npy"),
    "X":   os.path.join(PULSE_DIR, "u_X_main.npy"),
    "Y":   os.path.join(PULSE_DIR, "u_Y_main.npy"),
    "Z":   os.path.join(PULSE_DIR, "u_Z_main.npy"),
    "H":   os.path.join(PULSE_DIR, "u_H_main.npy"),
    "T":   os.path.join(PULSE_DIR, "u_T_main.npy"),
    "I":   os.path.join(PULSE_DIR, "u_I_main.npy"),
}

# Wide validation range (used in Tier 1)
VALIDATION_TRUNC_RANGE = list(range(18, 33, 2))

# Ideal 2x2 logical unitary for each gate, in the {|+Z_L>, |-Z_L>} basis
# (rows/cols ordered +,-). Must match the conventions baked into
# core/cat_code.py's get_logical_*_state_pairs -- these are NOT generic
# textbook Pauli/H/T matrices, they are the exact targets those factories
# train against (e.g. logical Y here is -i on the |+Z_L> branch, per the
# phase convention documented in get_logical_Y_state_pairs).
IDEAL_LOGICAL_U = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, 1j], [-1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
}


def average_gate_fidelity(U_ideal, U_actual):
    """
    Standard average gate fidelity (Nielsen 2002) between a d-dim ideal
    unitary and an actual (possibly sub-unitary, if there's leakage) map:
        F_avg = (|Tr(U_ideal^dagger U_actual)|^2 + d) / (d(d+1))
    Reduces to 1 iff U_actual = U_ideal exactly. Leakage (U_actual having
    less than unit operator norm because population left the 2D logical
    subspace) also reduces F_avg, so this single number folds in both
    coherent rotation error and leakage -- unlike the per-branch state
    fidelities in Tier 2, which only see the diagonal.
    """
    d = U_ideal.shape[0]
    overlap = np.trace(U_ideal.conj().T @ U_actual)
    return (np.abs(overlap) ** 2 + d) / (d * (d + 1))

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
    print("Target: aim for mean ≥ 0.985 and min ≥ 0.985 (literature target from Heeres et al.)")
    return df, {'mean': mean_f, 'min': min_f, 'std': std_f}

# ============================================================
# TIER 2: LOGICAL ACTION + LEAKAGE / CODE-SPACE PRESERVATION
# ============================================================
def tier2_logical_action_and_leakage(gate_name, get_state_pairs, u, n_c=24, n_t=N_T, dt=DT):
    """
    For the two logical basis states:
      - Recompute state fidelity (sanity)
      - Measure population in the target's own transmon level(s) (should be ~1)
      - Measure population on odd photon numbers (should be ~0 for even cat code)
      - Measure population outside the ideal logical support (mod-4 subspaces)
    Returns detailed per-basis-state report + aggregate leakage metrics.

    The "correct" transmon level(s) and mod-4 class(es) are derived from
    each pair's own ideal target psi_f, not hardcoded -- a fixed "transmon
    stays |g>, cavity stays in the class matching the pair's index"
    assumption is only true for enc/Z/T/I. It's false for dec (whose
    second branch's ideal output is transmon |e>), X/Y (which swap the
    mod-4 class between branches by design), and H (whose ideal outputs
    are an equal superposition of both mod-4 classes) -- for those gates
    the old hardcoded check reported >=50% "leakage" that was actually
    just correctly-placed population the check didn't recognize.
    """
    print(f"\n{'='*70}")
    print(f"TIER 2: Logical Action + Leakage Analysis — {gate_name} (n_c={n_c}, n_t={n_t})")
    print('='*70)

    H0, Hc = make_hamiltonian(n_t, n_c)
    pairs = get_state_pairs(n_c=n_c, n_t=n_t)
    assert len(pairs) == 2, "Expected exactly two state pairs for logical gates"

    n = np.arange(n_c)
    records = []
    for idx, (psi_i, psi_f) in enumerate(pairs):
        label = "+Z_L" if idx == 0 else "-Z_L"
        psi_out = propagate_pulse(u, H0, Hc, psi_i, dt)

        # 1. State fidelity to ideal target
        fid = np.abs(np.vdot(psi_f, psi_out))**2

        # 2. Population in the target's own transmon level(s) (generalizes
        #    the old fixed "|g> = indices 0:n_c" assumption)
        target_blocks = np.abs(psi_f.reshape(n_t, n_c))**2   # (n_t, n_c) target |amp|^2
        level_pop = target_blocks.sum(axis=1)
        target_levels = np.where(level_pop > 1e-8)[0]

        out_blocks = psi_out.reshape(n_t, n_c)
        p_target_transmon = np.sum(np.abs(out_blocks[target_levels])**2)
        leakage_transmon = 1.0 - p_target_transmon

        # 3 & 4. Odd-photon population and population outside the target's
        #    own mod-4 class(es), evaluated within each of the target's
        #    transmon level(s) (a single class for most gates; both classes
        #    for H's superposition targets).
        odd_pop = 0.0
        wrong_mod4_pop = 0.0
        for lvl in target_levels:
            cavity_amps = out_blocks[lvl]
            odd_pop += np.sum(np.abs(cavity_amps[1::2])**2)
            allowed_classes = np.unique(n[target_blocks[lvl] > 1e-8] % 4)
            wrong_mask = ~np.isin(n % 4, allowed_classes)
            wrong_mod4_pop += np.sum(np.abs(cavity_amps[wrong_mask])**2)

        total_leakage = leakage_transmon + odd_pop + wrong_mod4_pop   # approximate (some overlap)

        records.append({
            'basis_state': label,
            'state_fidelity': fid,
            'p_target_transmon': p_target_transmon,
            'odd_photon_pop': odd_pop,
            'wrong_mod4_pop': wrong_mod4_pop,
            'approx_total_leakage': total_leakage
        })

        print(f"  {label:6s} | F={fid:.6f} | p_target={p_target_transmon:.6f} | odd_pop={odd_pop:.2e} | wrong_mod4={wrong_mod4_pop:.2e}")

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

    # Effective logical unitary matrix elements (in {|+Z_L>, |-Z_L>} basis) (logical subspace)
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

    # Average gate fidelity of the extracted subspace map vs. the ideal
    # single-qubit gate (Tier 5: closeness to ideal, promised by the module
    # docstring but not previously computed -- see average_gate_fidelity()).
    F_avg_gate = np.nan
    U_ideal = IDEAL_LOGICAL_U.get(gate_name)
    if U_ideal is not None:
        F_avg_gate = average_gate_fidelity(U_ideal, U_log)
        print(f"  Average gate fidelity vs. ideal {gate_name}: {F_avg_gate:.6f}")
        if F_avg_gate < 0.99:
            print("  ⚠️  Average gate fidelity below 0.99 — rotation error and/or leakage.")

    phase_err = np.nan
    involution_fid = np.nan

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
        involution_fid = (fid_p + fid_m) / 2
        print(f"  H² fidelity to I: {involution_fid:.6f}")
        if involution_fid < 0.99:
            print("  ⚠️  H is not self-inverse (H² should be ≈ I) — check optimization.")

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
        'det': det,
        'phase_err': phase_err,        # NaN except for T: |measured relative phase - pi/4|
        'involution_fid': involution_fid,  # NaN except for H: H^2 fidelity to Identity
        'F_avg_gate': F_avg_gate,      # average gate fidelity vs. IDEAL_LOGICAL_U[gate_name]
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
            # H on logical  ⇒  H on comp basis.
            # Sign convention verified numerically against the actual
            # enc->H->dec composition (encode/decode's own +Z_L/-Z_L
            # convention is independently trained, so it need not match
            # cat_code.get_logical_H_state_pairs' convention at face
            # value): |g,0> lands on (g+e)/sqrt2 and |e,0> on (g-e)/sqrt2.
            # NOTE: this flipped from the previous verification (g-e / g+e)
            # after U_enc/U_dec were retrained with coherent_fidelity_multi_state
            # (see analysis/retrain_phase_coherent_gates.py) -- that fixed
            # their *relative* branch phase but didn't pin the resulting
            # absolute phase to match the old (also-arbitrary) one, so this
            # hardcoded sign needs re-verifying any time U_enc/U_dec change.
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
    if overall >= 0.985:
        print("  ✅  Excellent — logical gate correctly transferred to computational subspace (≥0.985).")
    elif overall >= 0.97:
        print("  ⚠️  Acceptable (≥0.97) but room for improvement — refine further or check U_enc/U_dec.")
    else:
        print("  ❌  Below target (<0.97) — investigate leakage or re-optimize gate.")
    return results


def _computational_basis_states(n_c, n_t):
    init_g = np.zeros(n_t * n_c, dtype=complex); init_g[0] = 1.0
    init_e = np.zeros(n_t * n_c, dtype=complex); init_e[n_c] = 1.0
    return init_g, init_e


def _logical_cat_basis_states(n_c, n_t):
    psi_p_cav, psi_m_cav = get_logical_cat_states(alpha=ALPHA, n_c=n_c)
    psi_pg = embed_in_joint_space(psi_p_cav, n_t=n_t, n_c=n_c, t_level=0)
    psi_mg = embed_in_joint_space(psi_m_cav, n_t=n_t, n_c=n_c, t_level=0)
    return psi_pg, psi_mg


def tier4_enc_relative_phase(u_enc, n_c=24, n_t=N_T, dt=DT):
    """
    Single-pulse half of tier4_enc_dec_relative_phase, for U_enc alone --
    depends only on u_enc (not u_dec), so it's usable standalone (e.g. by
    analysis/retrain_phase_coherent_gates.py, which retrains one pulse at a
    time). See tier4_enc_dec_relative_phase's docstring for the rationale.
    Extracts the effective 2x2 map {|g,0>,|e,0>} -> {|+Z_L>,|-Z_L>} and
    compares it to the identity (get_encode_state_pairs defines no extra
    relative phase between its two branches).
    """
    H0, Hc = make_hamiltonian(n_t, n_c)
    init_g, init_e = _computational_basis_states(n_c, n_t)
    psi_pg, psi_mg = _logical_cat_basis_states(n_c, n_t)

    out_g = propagate_pulse(u_enc, H0, Hc, init_g, dt)
    out_e = propagate_pulse(u_enc, H0, Hc, init_e, dt)
    U_enc_log = np.array([
        [np.vdot(psi_pg, out_g), np.vdot(psi_pg, out_e)],
        [np.vdot(psi_mg, out_g), np.vdot(psi_mg, out_e)],
    ])
    return {'U_log': U_enc_log, 'F_avg_gate': average_gate_fidelity(np.eye(2, dtype=complex), U_enc_log)}


def tier4_dec_relative_phase(u_dec, n_c=24, n_t=N_T, dt=DT):
    """Single-pulse half of tier4_enc_dec_relative_phase, for U_dec alone -- see tier4_enc_relative_phase."""
    H0, Hc = make_hamiltonian(n_t, n_c)
    init_g, init_e = _computational_basis_states(n_c, n_t)
    psi_pg, psi_mg = _logical_cat_basis_states(n_c, n_t)

    out_p = propagate_pulse(u_dec, H0, Hc, psi_pg, dt)
    out_m = propagate_pulse(u_dec, H0, Hc, psi_mg, dt)
    U_dec_log = np.array([
        [np.vdot(init_g, out_p), np.vdot(init_g, out_m)],
        [np.vdot(init_e, out_p), np.vdot(init_e, out_m)],
    ])
    return {'U_log': U_dec_log, 'F_avg_gate': average_gate_fidelity(np.eye(2, dtype=complex), U_dec_log)}


def tier4_enc_dec_relative_phase(u_enc, u_dec, n_c=24, n_t=N_T, dt=DT):
    """
    Checks whether U_enc/U_dec coherently preserve the *relative* phase
    between their two computational-branch training pairs -- the same
    defect Tier 3's F_avg_gate catches for the single-qubit logical gates
    (see core.grape_core.coherent_fidelity_multi_state's docstring).

    tier4_enc_gate_dec_pipeline can't see this: it only ever propagates
    |g,0> and |e,0> individually through the whole enc-gate-dec chain, and
    a single trajectory can only ever pick up an *overall* phase from each
    stage's independently-uncontrolled branch phase -- exactly the
    quantity state fidelity is blind to (the same structural reason
    U^2=I can't see it for X/Y/Z/I either).

    Thin wrapper combining tier4_enc_relative_phase and
    tier4_dec_relative_phase (each depends only on its own pulse) for the
    combined report used by the notebook/standalone validation summary.
    """
    print(f"\n{'='*70}")
    print(f"TIER 4b: Encode/Decode Relative-Phase Coherence Check (n_c={n_c}, n_t={n_t})")
    print('='*70)

    results = {
        'enc': tier4_enc_relative_phase(u_enc, n_c=n_c, n_t=n_t, dt=dt),
        'dec': tier4_dec_relative_phase(u_dec, n_c=n_c, n_t=n_t, dt=dt),
    }

    for label in ('enc', 'dec'):
        r = results[label]
        print(f"\n  U_{label} extracted logical map (vs. identity):")
        print(np.round(r['U_log'], decimals=6))
        print(f"  F_avg_gate: {r['F_avg_gate']:.6f}")
        if r['F_avg_gate'] < 0.99:
            print(f"  ⚠️  U_{label} does not coherently preserve relative phase across its two branches.")

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
        if pipe:
            pipeline_avg = np.mean([v['avg'] for v in pipe.values()])
        else:
            pipeline_avg = np.nan

        summary_rows.append({
            'Gate': gate_name,
            'F_mean': stats['mean'],
            'F_min': stats['min'],
            'F_std': stats['std'],
            'Unitarity_err': alg['unitarity_err'],
            'F_avg_gate': alg['F_avg_gate'],
            'Pipeline_avg': pipeline_avg,
            'Pulse_file': os.path.basename(pulse_path)
        })

    # Tier 4b: U_enc/U_dec relative-phase coherence (independent of any
    # single gate, so checked once rather than inside the gate loop above)
    enc_path = GATE_PULSE_MAP.get("enc")
    dec_path = GATE_PULSE_MAP.get("dec")
    if enc_path and dec_path and os.path.exists(enc_path) and os.path.exists(dec_path):
        enc_dec_results = tier4_enc_dec_relative_phase(np.load(enc_path), np.load(dec_path))
        for label in ("enc", "dec"):
            summary_rows.append({
                'Gate': label,
                'F_mean': np.nan, 'F_min': np.nan, 'F_std': np.nan,
                'Unitarity_err': np.nan,
                'F_avg_gate': enc_dec_results[label]['F_avg_gate'],
                'Pipeline_avg': np.nan,
                'Pulse_file': os.path.basename(GATE_PULSE_MAP[label]),
            })
    else:
        print("\n[SKIP] Tier 4b enc/dec relative-phase check: u_enc or u_dec not found.")

    # Final summary table
    if summary_rows:
        print("\n" + "="*70)
        print("OVERALL VALIDATION SUMMARY")
        print("="*70)
        df_sum = pd.DataFrame(summary_rows)
        print(df_sum.to_string(index=False, float_format="%.6f"))
        print("\nInterpretation guide:")
        print("  • F_mean / F_min : higher is better; target ≥ 0.985 (matches Heeres et al. 98.5% literature target)")
        print("  • Unitarity_err  : <1e-3 excellent (extracted logical map is unitary)")
        print("  • F_avg_gate     : average gate fidelity of extracted 2x2 U_log vs. ideal gate; target ≥ 0.99")
        print("  • Pipeline_avg   : most important practical metric — target ≥ 0.985")
        print("                     (measures how well the logical gate works inside the full encode/gate/decode stack)")

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
