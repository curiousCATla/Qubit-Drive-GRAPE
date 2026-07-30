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
    return pd.DataFrame(all_detail), pd.DataFrame(summaries)


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
                "dominant_leak_destination": summary["dominant_leak_destination"],
                "coarse_destination_class": coarse[n_c],
                "flag": summary["flag"],
            })
        stable_fine = len(set(dests.values())) == 1
        stable_coarse = len(set(coarse.values())) == 1
        print(f"    destinations: {dests}  fine_stable={stable_fine}  "
              f"coarse={coarse}  coarse_stable={stable_coarse}")
        for r in rows:
            if r["gate"] == gate:
                r["destination_stable"] = stable_fine
                r["coarse_destination_stable"] = stable_coarse
    return pd.DataFrame(rows)


def write_report(df_sum, df_cross, path=REPORT_PATH):
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

    df_detail, df_sum = run_headline(HEADLINE_N_C)

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

    write_report(sum_24, df_cross)
    print_pasteable_summary(sum_24)

    print(f"\nSaved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {cross_path}")


if __name__ == "__main__":
    main()
