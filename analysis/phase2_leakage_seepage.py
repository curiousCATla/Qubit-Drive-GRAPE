#!/usr/bin/env python3
"""
phase2_leakage_seepage.py — Leakage / seepage characterization (Phase 2).

Answers: if we start outside the code space, after the OCT pulse, where
could it land?  Uses the Phase-0 full propagator U_full.  Does NOT invent
an ideal unitary on the full D-dimensional space.  Does NOT change pulses.

Outputs
-------
  tables/phase2_population_detail.csv
  tables/phase2_summary.csv
  tables/phase2_truncation_crosscheck.csv
  tables/phase2_error_subspace.csv
  validation/phase2_leakage_report.md

Usage
-----
  python3 analysis/phase2_leakage_seepage.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.grape_core import make_hamiltonian, basis_state
from core.propagator import (
    full_propagator,
    logical_basis,
    logical_block,
    leakage_L1,
    pedersen_gate_fidelity,
    error_basis,
    knill_laflamme_residual,
)
from validation.outside_inputs import (
    outside_input_list,
    assert_outside_probes,
    code_input_list_logical,
    code_input_list_computational,
)
from validation.resolve_populations import (
    resolve_after_pulse,
    resolve_state,
    PARTITION_KEYS,
    aggregate_leak_destination,
    P_computational,
    LEAK_DESTINATION_KEYS,
)
from validation.validate_logical_gates import (
    IDEAL_LOGICAL_U,
    N_T,
    DT,
    ALPHA,
)

PULSE_DIR = os.path.join(REPO_ROOT, "pulses")
TABLE_DIR = os.path.join(REPO_ROOT, "tables")
REPORT_PATH = os.path.join(REPO_ROOT, "validation", "phase2_leakage_report.md")

# Priority order from the Phase-2 brief, then the rest.
GATES = ["enc", "dec", "X", "Y", "H", "Z", "T", "I"]
LOGICAL_GATES = ("I", "X", "Y", "Z", "H", "T")
HEADLINE_N_C = 24
TRUNC_CROSSCHECK = (22, 24, 26)
EXPECTED_SHAPE = (550, 4)
TRUE_OUTSIDE_TOL = 1e-12

# Heuristic flags (drive report language; not CI hard-fails).
L2_NOTE_THRESHOLD = 3e-3
DANGEROUS_EDGE_FRACTION = 0.10   # of leaked mass at trunc edge
DANGEROUS_EXCITED_FRACTION = 0.50  # of leaked mass in t≥1


def pulse_path(gate):
    return os.path.join(PULSE_DIR, f"u_{gate}_main.npy")


def load_pulse(gate):
    path = pulse_path(gate)
    u = np.load(path)
    assert u.shape == EXPECTED_SHAPE, (
        f"{path} shape {u.shape}, expected {EXPECTED_SHAPE}"
    )
    return u


def L1_for_gate(gate, U_full, B, n_t, n_c):
    """
    Average leakage out of the gate's natural 2×2 subspace.

    Logical gates: leakage_L1(B† U B) on the cat code space.
    enc: M maps {|g,0>,|e,0>} → cat coords; L1 = mean miss of C.
    dec: M maps cats → {|g,0>,|e,0>} coords; L1 = mean miss of computational.
    """
    if gate in LOGICAL_GATES:
        U_log = logical_block(U_full, B)
        return float(leakage_L1(U_log)), U_log

    if gate == "enc":
        # Columns of M: cat-basis coordinates of U|g,0>, U|e,0>
        kets_in = [basis_state(n_t, n_c, 0, 0), basis_state(n_t, n_c, 1, 0)]
        cols = []
        for ket in kets_in:
            out = U_full @ ket
            cols.append(B.conj().T @ out)
        M = np.column_stack(cols)
        return float(leakage_L1(M)), M

    if gate == "dec":
        # Columns of M: computational coords of U|+Z_L>, U|-Z_L>
        # computational basis B_comp = [|g,0>, |e,0>]
        B_comp = np.column_stack([
            basis_state(n_t, n_c, 0, 0),
            basis_state(n_t, n_c, 1, 0),
        ])
        cols = []
        for j in range(2):
            out = U_full @ B[:, j]
            cols.append(B_comp.conj().T @ out)
        M = np.column_stack(cols)
        return float(leakage_L1(M)), M

    raise ValueError(gate)


ERROR_LEAK_KEYS = tuple(f"err_{k}" for k in LEAK_DESTINATION_KEYS)


def error_space_metrics(gate, U_full, B, E, n_t, n_c, U_ideal=None):
    """
    The same L1 / L2 pair, measured on the single-photon-loss error subspace.

    E = span{a|+Z_L>, a|-Z_L>} (`core.propagator.error_basis`) is the subspace a
    cavity photon loss actually lands the code in. This function asks the two
    questions the code's correctability rests on:

      L1_E       = leakage_L1(E† U E), the population the pulse removes from the
                   error subspace. Zero means the loss's logical content is
                   still intact after the gate.
      L2_E_to_C  = mean_j ||B† U |e_j>||², the population that comes BACK into
                   the even code space. This is the dangerous direction: a state
                   that re-enters C is one the parity check will report as
                   error-free, turning a flagged, correctable loss into a silent
                   logical error.
      F_ET       = pedersen_gate_fidelity(U_ideal, E† U E), i.e. did the gate
                   perform its intended logical rotation on the error subspace
                   too? That is the error-transparency condition. Nothing in
                   this repo's training objective ever referenced E, so this is
                   a measurement of an unconstrained quantity, not a regression.

    F_ET needs no phase alignment: Pedersen's |Tr M|² and Tr(M M†) are both
    invariant under M -> e^{iθ}M, so a global phase on the block cannot move it.
    Do not add an alignment step.

    Read enc's row with care. enc's input subspace is {|g,0>, |e,0>}, not the
    cat code, so a post-loss state is not an input enc ever legitimately sees;
    its numbers are a robustness probe, not an error-transparency statement.
    dec is the opposite case — its input IS the code space, so E is exactly
    "a photon was lost before decode fired", and its L2_E_to_C is the number
    that matters.

    Returns a dict; F_ET is NaN when U_ideal is None (enc/dec).
    """
    M_E = logical_block(U_full, E)

    p_init = np.linalg.norm(B.conj().T @ E, axis=0) ** 2
    assert np.max(p_init) < TRUE_OUTSIDE_TOL, (
        f"{gate}: error subspace is not outside the code space "
        f"(P_logical_initial = {p_init}); see core.propagator.error_basis"
    )

    seep = []
    totals = {k: 0.0 for k in LEAK_DESTINATION_KEYS}
    for j in range(2):
        bins = resolve_after_pulse(U_full, E[:, j], B, n_t, n_c)
        seep.append(bins["P_logical"])
        for k in LEAK_DESTINATION_KEYS:
            totals[k] += bins[k]

    dom = max(totals, key=totals.get) if sum(totals.values()) > 0 else "none"
    out = {
        "L1_E": float(leakage_L1(M_E)),
        "L2_E_to_C": float(np.mean(seep)),
        "F_ET": (float(pedersen_gate_fidelity(U_ideal, M_E))
                 if U_ideal is not None else float("nan")),
        "err_dominant_destination": dom,
        "M_E_offdiag": float(max(abs(M_E[0, 1]), abs(M_E[1, 0]))),
    }
    out.update({f"err_{k}": totals[k] for k in LEAK_DESTINATION_KEYS})
    return out, seep


def drift_baseline(n_steps, n_t=N_T, n_c=HEADLINE_N_C, dt=DT, alpha=ALPHA):
    """
    The idle reference: the same metrics with the drive turned off.

    This is what "a single photon loss is correctable in idle" means
    quantitatively. `make_hamiltonian`'s H0 is exactly diagonal in the joint
    Fock/transmon basis, so drift cannot move amplitude between parity sectors
    at all: L2_E_to_C is 0 to machine precision and a loss stays flagged
    forever. The residual L1_E is pure Kerr n² dephasing *within* the error
    word — the drift block is diagonal (M_E_offdiag = 0), so it is a
    deterministic, known rotation, i.e. a frame update rather than lost
    information. Contrast the trained gates, where L1_E is large AND the block
    is off-diagonal.

    Runs through `full_propagator` with u = 0 rather than a hand-written
    exp(-i H0 T), so the baseline and the gate rows share one code path.
    """
    H0, Hc = make_hamiltonian(n_t, n_c)
    U = full_propagator(np.zeros((n_steps, 4)), H0, Hc, dt)
    B = logical_basis(n_t, n_c, alpha=alpha)
    E, _nrm = error_basis(n_t, n_c, alpha=alpha, B=B)

    out, _seep = error_space_metrics(
        "drift", U, B, E, n_t, n_c, U_ideal=np.eye(2, dtype=complex)
    )
    out["gate"] = "drift(u=0)"
    out["n_c"] = n_c
    out["L1"] = float(leakage_L1(logical_block(U, B)))
    out["T_us"] = n_steps * dt
    return out


def code_inputs_for_gate(gate, n_t, n_c, B):
    """Inputs whose non-target population is 'leaked' for destination analysis."""
    if gate == "enc":
        return code_input_list_computational(n_t, n_c)
    # dec and logical gates: start in the cat code
    return code_input_list_logical(n_t, n_c, alpha=ALPHA, B=B)


def _dec_leak_bins(phi, bins, n_t, n_c):
    """
    For U_dec, success is landing in span{|g,0>,|e,0>}.  Subtract those two
    Fock populations from the ground/excited even bins so the destination
    of *failed* decode is not the computational basis itself.
    """
    blocks = np.asarray(phi, dtype=complex).reshape(n_t, n_c)
    pop = np.abs(blocks) ** 2
    p_g0 = float(pop[0, 0])
    p_e0 = float(pop[1, 0]) if n_t > 1 else 0.0
    n_idx = np.arange(n_c)
    even_mask = (n_idx % 2) == 0
    # All even-g except |g,0>; all even-e except |e,0>
    P_g_even_other = float(pop[0, even_mask].sum()) - p_g0
    P_e_even_other = (
        float(pop[1:, :][:, even_mask].sum()) - p_e0 if n_t > 1 else 0.0
    )
    return {
        "P_g_even_nonlogical": max(P_g_even_other, 0.0),
        "P_g_odd": bins["P_g_odd"],
        "P_e_even": max(P_e_even_other, 0.0),
        "P_e_odd": bins["P_e_odd"],
    }


def target_population(gate, phi, B, n_t, n_c):
    """Population remaining in the gate's intended subspace after U."""
    if gate == "dec":
        return P_computational(phi, n_t, n_c)
    # enc and logical gates: intended code is the cat subspace
    return float(np.linalg.norm(B.conj().T @ phi) ** 2)


def classify_flag(L1, L2, leak_totals, max_P_logical_outside):
    """
    Heuristic decision flag for the summary table / report.

    Destination class is judged on the *fraction* of leaked mass, but
    absolute L1 must be non-negligible before a destination is called
    dangerous (tiny residuals of a nearly code-preserving pulse are not
    a Phase-3 trigger).
    """
    leaked = sum(leak_totals.values())
    P_e = leak_totals.get("P_e_even", 0.0) + leak_totals.get("P_e_odd", 0.0)

    seepage = L2 >= L2_NOTE_THRESHOLD or max_P_logical_outside >= 5e-2

    # Absolute L1 floors for severity
    L1_WATCH = 5e-4
    L1_DANGER = 1e-3

    if leaked > 1e-12 and L1 >= L1_WATCH:
        frac_e = P_e / leaked
        if frac_e >= DANGEROUS_EXCITED_FRACTION and L1 >= L1_DANGER:
            base = "dangerous"
        elif frac_e >= 0.25 or (
            frac_e >= DANGEROUS_EXCITED_FRACTION and L1 >= L1_WATCH
        ):
            base = "watch"
        else:
            dom = max(leak_totals, key=leak_totals.get)
            base = "document" if dom in ("P_g_even_nonlogical", "P_g_odd") else "watch"
    else:
        # Tiny absolute leakage: document destination without alarm
        if leaked > 1e-12:
            dom = max(leak_totals, key=leak_totals.get)
            base = "document" if dom in ("P_g_even_nonlogical", "P_g_odd", "P_e_even", "P_e_odd") else "document"
        else:
            base = "document"

    if seepage and base == "document":
        return "document+seepage_note"
    if seepage and base == "watch":
        return "watch+seepage_note"
    if seepage and base == "dangerous":
        return "dangerous+seepage_note"
    return base


def analyze_gate(gate, n_c, n_t=N_T, dt=DT, alpha=ALPHA):
    """Full Phase-2 analysis for one gate at one truncation. Returns (detail_rows, summary_row)."""
    u = load_pulse(gate)
    H0, Hc = make_hamiltonian(n_t, n_c)
    U = full_propagator(u, H0, Hc, dt)
    B = logical_basis(n_t, n_c, alpha=alpha)

    outside = outside_input_list(n_t=n_t, n_c=n_c, alpha=alpha, B=B)
    assert_outside_probes(outside, B, tol=TRUE_OUTSIDE_TOL)

    L1, _M = L1_for_gate(gate, U, B, n_t, n_c)

    # Single-photon-loss error subspace. Kept separate from the 9-probe outside
    # set below: those probes sample the complement of C generically, E is the
    # one subspace a real loss actually produces, and mixing them would blur the
    # existing L2 number that Section 11.2 of experiments.ipynb reports.
    E, _e_nrm = error_basis(n_t, n_c, alpha=alpha, B=B)
    err_metrics, err_seep = error_space_metrics(
        gate, U, B, E, n_t, n_c, U_ideal=IDEAL_LOGICAL_U.get(gate)
    )

    detail_rows = []
    seepage_pops = []
    max_P_logical_outside = 0.0

    # For U_enc, |e,0> is an intentional training input that is *supposed* to
    # land in the cat code — counting it in L2 would report design action as
    # seepage. Exclude only that ket from the L2 average (still listed in detail).
    enc_exclude_from_L2 = {"|e,0>"} if gate == "enc" else set()

    for name, group, ket in outside:
        bins = resolve_after_pulse(U, ket, B, n_t, n_c)
        is_true_outside = bins["P_logical_initial"] < TRUE_OUTSIDE_TOL
        counts_for_L2 = is_true_outside and name not in enc_exclude_from_L2
        if counts_for_L2:
            seepage_pops.append(bins["P_logical"])
            max_P_logical_outside = max(max_P_logical_outside, bins["P_logical"])
        detail_rows.append({
            "gate": gate,
            "n_c": n_c,
            "input_name": name,
            "group": group,
            "is_code_input": False,
            "counts_for_L2": counts_for_L2,
            "P_logical_initial": bins["P_logical_initial"],
            "P_logical": bins["P_logical"],
            "P_g_even_nonlogical": bins["P_g_even_nonlogical"],
            "P_g_odd": bins["P_g_odd"],
            "P_e_even": bins["P_e_even"],
            "P_e_odd": bins["P_e_odd"],
            "P_trunc_edge": bins["P_trunc_edge"],
            "partition_sum": sum(bins[k] for k in PARTITION_KEYS),
        })

    L2 = float(np.mean(seepage_pops)) if seepage_pops else float("nan")

    # Error-subspace probes, listed with counts_for_L2=False so they are visible
    # in the detail table without changing the 9-probe L2 above.
    for j, label in enumerate(("a|+Z_L>", "a|-Z_L>")):
        bins = resolve_after_pulse(U, E[:, j], B, n_t, n_c)
        detail_rows.append({
            "gate": gate,
            "n_c": n_c,
            "input_name": label,
            "group": "E",
            "is_code_input": False,
            "counts_for_L2": False,
            "P_logical_initial": bins["P_logical_initial"],
            "P_logical": bins["P_logical"],
            "P_g_even_nonlogical": bins["P_g_even_nonlogical"],
            "P_g_odd": bins["P_g_odd"],
            "P_e_even": bins["P_e_even"],
            "P_e_odd": bins["P_e_odd"],
            "P_trunc_edge": bins["P_trunc_edge"],
            "partition_sum": sum(bins[k] for k in PARTITION_KEYS),
        })

    # Code inputs: where does leaked amplitude land?
    code_bins_list = []
    edge_leaked = 0.0
    leaked_mass_total = 0.0
    for name, group, ket in code_inputs_for_gate(gate, n_t, n_c, B):
        phi = U @ ket
        bins = resolve_state(phi, B, n_t, n_c)
        # For dec, target subspace is computational, not cat logical.
        if gate == "dec":
            P_target = P_computational(phi, n_t, n_c)
            leaked = max(1.0 - P_target, 0.0)
            # Destination bins for dec: exclude |g,0> and |e,0| from the
            # partition so successful decode is not counted as a "leak land".
            leak_bins = _dec_leak_bins(phi, bins, n_t, n_c)
        else:
            P_target = bins["P_logical"]
            leaked = max(1.0 - P_target, 0.0)
            leak_bins = {k: bins[k] for k in LEAK_DESTINATION_KEYS}
        code_bins_list.append(leak_bins)
        edge_leaked += bins["P_trunc_edge"]
        leaked_mass_total += leaked

        detail_rows.append({
            "gate": gate,
            "n_c": n_c,
            "input_name": name,
            "group": group,
            "is_code_input": True,
            "P_logical_initial": float(np.linalg.norm(B.conj().T @ ket) ** 2),
            "P_logical": bins["P_logical"],
            "P_g_even_nonlogical": bins["P_g_even_nonlogical"],
            "P_g_odd": bins["P_g_odd"],
            "P_e_even": bins["P_e_even"],
            "P_e_odd": bins["P_e_odd"],
            "P_trunc_edge": bins["P_trunc_edge"],
            "partition_sum": sum(bins[k] for k in PARTITION_KEYS),
            "P_target": P_target,
            "leaked_mass": leaked,
        })

    dom, leak_totals = aggregate_leak_destination(code_bins_list)

    # Trunc-edge fraction of leaked mass (for danger flag)
    if leaked_mass_total > 1e-12:
        edge_frac = edge_leaked / max(leaked_mass_total, edge_leaked)
        # edge_leaked can overlap other bins; use as soft diagnostic
        if edge_frac >= DANGEROUS_EDGE_FRACTION and leaked_mass_total > 1e-4:
            danger_edge = True
        else:
            danger_edge = False
    else:
        danger_edge = False

    flag = classify_flag(L1, L2, leak_totals, max_P_logical_outside)
    if danger_edge and flag not in ("dangerous",):
        flag = "dangerous" if leaked_mass_total > 5e-3 else "watch"

    # Plain-language destination label
    DEST_LABEL = {
        "P_g_even_nonlogical": "g, even photon (outside code / radial)",
        "P_g_odd": "g, odd photon (wrong parity)",
        "P_e_even": "transmon excited, even photon",
        "P_e_odd": "transmon excited, odd photon",
        "none": "none (no measurable leak)",
    }

    summary = {
        "gate": gate,
        "n_c": n_c,
        "L1": L1,
        "L2": L2,
        "dominant_leak_destination": dom,
        "dominant_leak_label": DEST_LABEL.get(dom, dom),
        "leak_P_g_even_nonlogical": leak_totals.get("P_g_even_nonlogical", 0.0),
        "leak_P_g_odd": leak_totals.get("P_g_odd", 0.0),
        "leak_P_e_even": leak_totals.get("P_e_even", 0.0),
        "leak_P_e_odd": leak_totals.get("P_e_odd", 0.0),
        "max_P_logical_outside": max_P_logical_outside,
        "flag": flag,
        **err_metrics,
    }
    return detail_rows, summary


def run_headline(n_c=HEADLINE_N_C):
    all_detail = []
    summaries = []
    print("=" * 88)
    print(f"PHASE 2 — leakage / seepage characterization  (n_c={n_c}, n_t={N_T}, dt={DT})")
    print("=" * 88)
    for gate in GATES:
        print(f"  analyzing U_{gate} ...", flush=True)
        detail, summary = analyze_gate(gate, n_c=n_c)
        all_detail.extend(detail)
        summaries.append(summary)
        print(
            f"    L1={summary['L1']:.6e}  L2={summary['L2']:.6e}  "
            f"dom={summary['dominant_leak_destination']}  flag={summary['flag']}"
        )
        print(
            f"    error subspace: L1_E={summary['L1_E']:.6e}  "
            f"L2_E->C={summary['L2_E_to_C']:.6e}  F_ET={summary['F_ET']:.6f}  "
            f"dom={summary['err_dominant_destination']}"
        )

    # Knill-Laflamme for E={I,a}: how exactly correctable is a loss at rest?
    B_ref = logical_basis(N_T, n_c, alpha=ALPHA)
    kl = knill_laflamme_residual(B_ref, N_T, n_c)
    print("\n" + "-" * 88)
    print("Knill-Laflamme for E = {I, a} (is a single loss exactly correctable?)")
    print("-" * 88)
    print(f"  <+Z_L|a^dag a|+Z_L> = {kl['n_bar_plus']:.6f}    "
          f"<-Z_L|a^dag a|-Z_L> = {kl['n_bar_minus']:.6f}")
    print(f"  off-diagonal        = {kl['n_offdiag']:.3e}      "
          f"||B^dag a B||       = {kl['a_block_norm']:.3e}   (both must be ~0)")
    print(f"  relative diagonal mismatch = {kl['rel_mismatch']:.4f}  -> the "
          "alpha=sqrt(3) cat code is an\n  APPROXIMATE single-loss code, not an "
          "exact one. EST/kitten_code.py's binomial code\n  is the exact version "
          "(a|0_L>=sqrt2|3>, a|1_L>=sqrt2|1>, equal weight).")

    drift = drift_baseline(EXPECTED_SHAPE[0], n_t=N_T, n_c=n_c)
    print("\n" + "-" * 88)
    print(f"Idle baseline: drive off, same duration (T = {drift['T_us']:.3f} us)")
    print("-" * 88)
    print(f"  L1_E = {drift['L1_E']:.6e}   L2_E->C = {drift['L2_E_to_C']:.3e}   "
          f"|M_E offdiag| = {drift['M_E_offdiag']:.3e}")
    print("  H0 is exactly diagonal, so drift cannot cross parity sectors: the "
          "loss stays flagged\n  exactly. The residual L1_E is Kerr n^2 dephasing "
          "inside the error word -- a known,\n  deterministic rotation (diagonal "
          "block), i.e. a frame update, not lost information.")

    return pd.DataFrame(all_detail), pd.DataFrame(summaries), kl, drift


def _coarse_dest_class(dest_key):
    """Coarse landing class for truncation stability (excited vs g-parity)."""
    if dest_key in ("P_e_even", "P_e_odd"):
        return "transmon_excited"
    if dest_key == "P_g_odd":
        return "g_odd_parity"
    if dest_key == "P_g_even_nonlogical":
        return "g_even_radial"
    return dest_key


def run_truncation_crosscheck(top_gates, truncs=TRUNC_CROSSCHECK):
    rows = []
    print("\n" + "=" * 88)
    print(f"Truncation cross-check on top-L1 gates: {top_gates}")
    print("=" * 88)
    for gate in top_gates:
        dests = {}
        coarse = {}
        for n_c in truncs:
            print(f"  U_{gate} @ n_c={n_c} ...", flush=True)
            _detail, summary = analyze_gate(gate, n_c=n_c)
            dests[n_c] = summary["dominant_leak_destination"]
            coarse[n_c] = _coarse_dest_class(dests[n_c])
            rows.append({
                "gate": gate,
                "n_c": n_c,
                "L1": summary["L1"],
                "L2": summary["L2"],
                "L1_E": summary["L1_E"],
                "L2_E_to_C": summary["L2_E_to_C"],
                "F_ET": summary["F_ET"],
                "dominant_leak_destination": summary["dominant_leak_destination"],
                "coarse_destination_class": coarse[n_c],
                "flag": summary["flag"],
            })
        stable_fine = len(set(dests.values())) == 1
        stable_coarse = len(set(coarse.values())) == 1
        print(f"    destinations: {dests}  fine_stable={stable_fine}  "
              f"coarse={coarse}  coarse_stable={stable_coarse}")

        # Truncation convergence of the error-subspace numbers. The criterion is
        # a PLATEAU at and above the production truncation, not a flat line over
        # the whole scan — same reading as the Heeres Eqs. 23–24 check in
        # core/cat_code.validate_pulse_truncations. That distinction is load
        # bearing here: the error words sit one Fock rung off the code words and
        # carry n̄ ≈ 3 before the pulse even starts, so a pulse that already
        # reaches high Fock can still be clipped at n_c=22 while being converged
        # at 24. U_enc is exactly that case (0.9435 / 0.9407 / 0.9407); asserting
        # on the full spread would fail on a truncation below production.
        gate_rows = [r for r in rows if r["gate"] == gate]
        tail = [r["L1_E"] for r in gate_rows if r["n_c"] >= HEADLINE_N_C]
        spread_L1E = max(r["L1_E"] for r in gate_rows) - min(r["L1_E"] for r in gate_rows)
        spread_tail = (max(tail) - min(tail)) if len(tail) > 1 else 0.0
        print(f"    L1_E across truncations: "
              f"{ {r['n_c']: round(r['L1_E'], 6) for r in gate_rows} }  "
              f"full spread={spread_L1E:.2e}  plateau spread (n_c>={HEADLINE_N_C})"
              f"={spread_tail:.2e}")
        assert spread_tail < 1e-3, (
            f"U_{gate}: L1_E varies by {spread_tail:.3e} across n_c >= "
            f"{HEADLINE_N_C}; the error subspace has not plateaued and the "
            "error-subspace metrics are pressed against the truncation wall"
        )

        for r in rows:
            if r["gate"] == gate:
                r["destination_stable"] = stable_fine
                r["coarse_destination_stable"] = stable_coarse
                r["L1_E_spread"] = spread_L1E
                r["L1_E_plateau_spread"] = spread_tail
    return pd.DataFrame(rows)


def _error_subspace_section(head, kl, drift):
    """Report block for the single-photon-loss error subspace."""
    lines = []
    lines.append("## Single-photon-loss error subspace")
    lines.append("")
    lines.append("The even cat code has one physically distinguished outside subspace:")
    lines.append("")
    lines.append("    E = span{ a|+Z_L>, a|-Z_L> }   (`core.propagator.error_basis`)")
    lines.append("")
    lines.append("exactly orthonormal and exactly orthogonal to the code space (the cats")
    lines.append("live on Fock n ≡ 0, 2 mod 4, their loss images on n ≡ 3, 1). Group **B**")
    lines.append("above samples the odd manifold generically; E *is* the state a real loss")
    lines.append("produces.")
    lines.append("")
    lines.append("**Is a loss exactly correctable to begin with?** Knill–Laflamme for")
    lines.append("E = {I, a} needs `<i_L|a†a|j_L> = c·δ_ij`. The cross term `||B†aB||` and the")
    lines.append(f"off-diagonal are exactly zero ({kl['a_block_norm']:.1e}, {kl['n_offdiag']:.1e}), "
                 "but the diagonal is not:")
    lines.append("")
    lines.append(f"| n̄(+Z_L) | n̄(−Z_L) | relative mismatch |")
    lines.append("|----------|----------|-------------------|")
    lines.append(f"| {kl['n_bar_plus']:.4f} | {kl['n_bar_minus']:.4f} | "
                 f"{kl['rel_mismatch']:.3f} |")
    lines.append("")
    lines.append("So the α=√3 four-component cat is an **approximate** single-loss code, not")
    lines.append("an exact one. The exact version is the binomial kitten code in")
    lines.append("`EST/kitten_code.py`, where `a|0_L> = √2|3>` and `a|1_L> = √2|1>` carry equal")
    lines.append("weight — the coincidence its `error_cardinals` assertion pins.")
    lines.append("")
    lines.append("### Metrics")
    lines.append("")
    lines.append("- **L1_E** (E → out): `leakage_L1(E† U E)`. Population the pulse removes from")
    lines.append("  the error subspace, i.e. logical content of the loss that does not survive.")
    lines.append("- **L2_E→C** (E → code): mean `||B† U|e_j>||²`. The dangerous direction —")
    lines.append("  amplitude back in the *even* code space reads as error-free to a parity")
    lines.append("  check, converting a flagged, correctable loss into a silent logical error.")
    lines.append("- **F_ET**: `pedersen_gate_fidelity(U_ideal, E† U E)` — did the gate perform")
    lines.append("  its intended rotation on E as well? This is error transparency. No training")
    lines.append("  objective in this repo ever referenced E, so it is unconstrained, not")
    lines.append("  regressed.")
    lines.append("")
    lines.append(f"### Idle baseline (drive off, T = {drift['T_us']:.3f} μs)")
    lines.append("")
    lines.append(f"`L1_E = {drift['L1_E']:.4e}`, `L2_E→C = {drift['L2_E_to_C']:.2e}`, "
                 f"`|M_E offdiag| = {drift['M_E_offdiag']:.2e}`.")
    lines.append("")
    lines.append("H0 is exactly diagonal, so drift cannot move amplitude across parity")
    lines.append("sectors: seepage back into the code space is zero to machine precision and")
    lines.append("the loss stays flagged. The residual L1_E is Kerr n² dephasing *within* the")
    lines.append("error word, and the block is diagonal — a deterministic, known rotation, so")
    lines.append("a frame update rather than lost information. **This is what \"correctable in")
    lines.append("idle\" means quantitatively**, and it is the reference the gate rows below")
    lines.append("are measured against.")
    lines.append("")
    lines.append(f"### Under the trained pulses (n_c={HEADLINE_N_C})")
    lines.append("")
    lines.append("| Gate | L1_E (E→out) | L2_E→C (E→code) | F_ET | Dominant destination |")
    lines.append("|------|--------------|-----------------|------|----------------------|")
    for _, r in head.sort_values("L1_E", ascending=False).iterrows():
        f_et = "—" if np.isnan(r["F_ET"]) else f"{r['F_ET']:.4f}"
        lines.append(
            f"| {r['gate']} | {r['L1_E']:.4e} | {r['L2_E_to_C']:.4e} | {f_et} | "
            f"`{r['err_dominant_destination']}` |"
        )
    lines.append("")
    lines.append("Read enc's row as a robustness probe only: enc's input subspace is")
    lines.append("{|g,0>, |e,0>}, so a post-loss state is not an input it legitimately sees.")
    lines.append("dec is the opposite — its input *is* the code space, so E is exactly \"a")
    lines.append("photon was lost before decode fired\".")
    lines.append("")

    worst_seep = head.sort_values("L2_E_to_C", ascending=False).iloc[0]
    logical = head[head["gate"].isin(LOGICAL_GATES)]
    lines.append("### Reading")
    lines.append("")
    ratio = (logical["L1_E"] / logical["L1"])
    lines.append(
        f"The trained gates are **not error-transparent**: on the six logical gates L1_E "
        f"runs {logical['L1_E'].min():.2e}–{logical['L1_E'].max():.2e} and F_ET falls as "
        f"low as {logical['F_ET'].min():.3f}, while the *code-space* L1 for the same "
        f"gates stays at {logical['L1'].max():.2e} or below — the pulses protect C between "
        f"{ratio.min():.0f}× and {ratio.max():.0f}× better than they protect E. Expected, "
        "since E was never in any objective, but it is the gap the `EST/` track's C2 cost "
        "is built to close on a different code."
    )
    lines.append("")
    lines.append(
        f"Seepage back into the code space stays small for the logical gates "
        f"(L2_E→C ≤ {logical['L2_E_to_C'].max():.2e}): a loss is still *detected*, it "
        f"just stops being *correctable*. The exception is "
        f"**U_{worst_seep['gate']}** at L2_E→C = {worst_seep['L2_E_to_C']:.2e} — that "
        "fraction of a post-loss state re-enters the even manifold, where the parity "
        "check reports no error at all."
    )
    lines.append("")
    return lines


def write_report(df_sum, df_cross, path=REPORT_PATH, kl=None, drift=None):
    # Rank by L1 at headline n_c
    head = df_sum[df_sum["n_c"] == HEADLINE_N_C].sort_values("L1", ascending=False)
    top3 = head.head(3)

    lines = []
    lines.append("# Phase 2 — Leakage / Seepage Report")
    lines.append("")
    lines.append("Measurement only. No pulses were changed. No ideal full-space unitary was invented.")
    lines.append(f"Parameters: n_t={N_T}, dt={DT} μs, α=√3, main pulses 550×4, headline n_c={HEADLINE_N_C}.")
    lines.append("")
    lines.append("## Probe set")
    lines.append("")
    lines.append("Fixed in `validation/outside_inputs.py`, identical for every gate:")
    lines.append("")
    lines.append("- **A** Transmon-excited: `|e,0>`, `|e,1>`, `|e,2>`")
    lines.append("- **B** Opposite parity: `|g,1>`, `|g,3>`, `|g,5>`")
    lines.append("- **C** Same-parity orthogonal-to-code: residuals of `|g,4>`, `|g,6>`, `|g,8>`")
    lines.append("  after projecting out the logical cats (Gram–Schmidt; true outside probes).")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("- **L1** (code → out): average leakage on the gate’s natural 2×2 block")
    lines.append("  (`leakage_L1`). For I/X/Y/Z/H/T this is the cat-subspace block; for enc/dec")
    lines.append("  it is the encode/decode map of Phase-0 tier4.")
    lines.append("- **L2** (out → code): mean cat-subspace population after the pulse, averaged")
    lines.append("  over true-outside probes (P_logical_initial < 1e-12). Localization diagnostic;")
    lines.append("  do not add to L1 as a double-counted error budget.")
    lines.append("")
    lines.append("## Summary at n_c=24")
    lines.append("")
    lines.append("| Gate | L1 (code→out) | L2 (out→code) | Dominant leak destination | Flag |")
    lines.append("|------|---------------|---------------|---------------------------|------|")
    for _, r in head.iterrows():
        lines.append(
            f"| {r['gate']} | {r['L1']:.6e} | {r['L2']:.6e} | "
            f"{r['dominant_leak_label']} | {r['flag']} |"
        )
    lines.append("")
    lines.append("## Three highest-L1 gates — plain language")
    lines.append("")
    for rank, (_, r) in enumerate(top3.iterrows(), 1):
        lines.append(f"### {rank}. U_{r['gate']}  (L1 = {r['L1']:.4e})")
        lines.append("")
        lines.append(
            f"When amplitude leaves the intended subspace, it lands predominantly in "
            f"**{r['dominant_leak_label']}** "
            f"(`{r['dominant_leak_destination']}`). "
            f"Seepage L2 = {r['L2']:.4e} "
            f"(max single-probe P_logical from outside = {r['max_P_logical_outside']:.4e}). "
            f"Flag: **{r['flag']}**."
        )
        lines.append("")

    if kl is not None and drift is not None:
        lines.extend(_error_subspace_section(head, kl, drift))

    lines.append("## Truncation cross-check")
    lines.append("")
    if df_cross is not None and len(df_cross):
        lines.append("Top-L1 gates re-scored at n_c ∈ {22, 24, 26}. "
                     "Quantitative bin weights may drift; destination **class** should not.")
        lines.append("")
        for gate in df_cross["gate"].unique():
            sub = df_cross[df_cross["gate"] == gate]
            dests = {int(row.n_c): row.dominant_leak_destination for _, row in sub.iterrows()}
            coarse = {int(row.n_c): row.coarse_destination_class for _, row in sub.iterrows()}
            stable_fine = bool(sub["destination_stable"].iloc[0])
            stable_coarse = bool(sub["coarse_destination_stable"].iloc[0])
            lines.append(
                f"- **U_{gate}**: fine bins {dests} "
                f"({'STABLE' if stable_fine else 'shifts within class'}); "
                f"coarse class {coarse} — "
                f"{'STABLE' if stable_coarse else 'UNSTABLE'}"
            )
        lines.append("")
    else:
        lines.append("(no cross-check data)")
        lines.append("")

    lines.append("## Decision")
    lines.append("")
    flags = set(head["flag"].tolist())
    if any("dangerous" in f for f in flags):
        lines.append(
            "At least one gate is flagged **dangerous** (leaked amplitude prefers "
            "excited-transmon or truncation-edge subspaces). Consider a Phase-3 "
            "targeted constraint; do **not** retrain blindly."
        )
    elif any("watch" in f for f in flags):
        lines.append(
            "Some gates are flagged **watch** (non-negligible excited-transmon "
            "component of leakage and/or elevated seepage). Documented here; no pulse "
            "change in Phase 2. Revisit only if a protocol is sensitive to that landing."
        )
    else:
        lines.append(
            "Leaked amplitude from the highest-L1 gates lands predominantly in "
            "benign / parity-related cavity subspaces (ground-transmon even residual "
            "or odd photon). **Document and move on** — no Phase-3 constraint required "
            "from this characterization alone."
        )
    if any("seepage" in f for f in flags):
        lines.append("")
        lines.append(
            "Seepage L2 is non-negligible on at least one gate: outside population can "
            "enter the code space under the pulse. For protocols that assume the code "
            "space is only entered via encode, this is more harmful than ordinary "
            "leakage — note for system-level design."
        )
    lines.append("")
    lines.append("No pulses were modified in Phase 2.")
    lines.append("")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote report: {path}")


def print_pasteable_summary(df_sum):
    head = df_sum[df_sum["n_c"] == HEADLINE_N_C].sort_values("L1", ascending=False)
    print("\n" + "=" * 88)
    print("PASTE-BACK SUMMARY (n_c=24)")
    print("=" * 88)
    cols = ["gate", "L1", "L2", "dominant_leak_destination", "dominant_leak_label", "flag"]
    print(head[cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6e}" if isinstance(x, float) else str(x),
    ))
    print()


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)

    df_detail, df_sum, kl, drift = run_headline(HEADLINE_N_C)

    # Top-3 L1 for truncation cross-check
    top3 = (
        df_sum.sort_values("L1", ascending=False)["gate"].head(3).tolist()
    )
    df_cross = run_truncation_crosscheck(top3)

    # Merge headline-only summary rows that may get duplicated if crosscheck
    # re-ran n_c=24 — keep one summary file with all n_c from crosscheck + any
    # gates only at 24.
    df_sum_full = pd.concat([df_sum, df_cross], ignore_index=True)
    # Drop exact duplicates on (gate, n_c) preferring the crosscheck columns
    df_sum_full = df_sum_full.drop_duplicates(subset=["gate", "n_c"], keep="last")

    detail_path = os.path.join(TABLE_DIR, "phase2_population_detail.csv")
    summary_path = os.path.join(TABLE_DIR, "phase2_summary.csv")
    cross_path = os.path.join(TABLE_DIR, "phase2_truncation_crosscheck.csv")

    # Detail currently only headline; re-run detail for crosscheck gates at other n_c
    # for completeness (optional). Keep headline detail + note crosscheck is summary-level.
    df_detail.to_csv(detail_path, index=False)
    # Summary: headline for all gates at 24, plus crosscheck rows
    # Rebuild a clean summary: all gates @24 from df_sum, plus crosscheck extras
    sum_24 = df_sum.copy()
    extra = df_cross[df_cross["n_c"] != HEADLINE_N_C]
    # For crosscheck gates at 24, update destination_stable into sum if useful
    sum_out = pd.concat([sum_24, extra], ignore_index=True)
    sum_out.to_csv(summary_path, index=False)
    df_cross.to_csv(cross_path, index=False)

    # Error-subspace summary, with the idle baseline as its own row so the gate
    # numbers are never read without the reference they are measured against.
    err_cols = ["gate", "n_c", "L1", "L1_E", "L2_E_to_C", "F_ET",
                "M_E_offdiag", "err_dominant_destination", *ERROR_LEAK_KEYS]
    df_err = pd.concat(
        [sum_24[err_cols], pd.DataFrame([{k: drift.get(k, np.nan) for k in err_cols}])],
        ignore_index=True,
    )
    err_path = os.path.join(TABLE_DIR, "phase2_error_subspace.csv")
    df_err.to_csv(err_path, index=False)

    write_report(sum_24, df_cross, kl=kl, drift=drift)
    print_pasteable_summary(sum_24)

    print(f"\nSaved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {cross_path}")
    print(f"Saved: {err_path}")


if __name__ == "__main__":
    main()
