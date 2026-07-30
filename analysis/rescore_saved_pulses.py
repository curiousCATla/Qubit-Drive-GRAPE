#!/usr/bin/env python3
"""
rescore_saved_pulses.py — Phase 0: re-score the saved logical-gate pulses with
the honest propagator-based measuring stick.

What this changes relative to previously reported numbers:

  1. Average gate fidelity is now the Pedersen (2007) form, with leakage
     entering through Tr(M M^dagger) instead of an assumed `+ d`. For d=2 the
     old form over-reported by exactly L1/3 (asserted per row below), i.e. by
     7e-5 to 1.0e-3 for these pulses. That is the whole correction -- it is a
     real bias but a small one, and it is NOT the explanation for any
     large/low fidelity figure elsewhere in the project.
  2. The 2x2 logical map is extracted by an explicit projection B^dag U B of the
     full D x D propagator, under one documented index convention
     (column-per-input), rather than from four hand-assembled overlaps.
  3. The truncation sweep includes truncations the pulses were NOT trained on.
     n_c in (22, 24, 26) is exactly the training set (trained as a mean with
     lambda_disc penalizing their spread), so agreement across those three is
     close to tautological -- the `trained` column marks them, and the
     convergence flag is evaluated on the held-out truncations only.

Nothing here feeds back into any training objective. L1 is reported as a
diagnostic, not used as a penalty.

Encode/decode are scored separately, in `rescore_enc_dec`. They are not logical
gates: encode maps the transmon computational subspace {|g,0>, |e,0>} to the cat
subspace and decode maps it back, so their effective 2x2 map is an OFF-DIAGONAL
block of U_full with a different basis on each side (`cross_subspace_block`),
not `logical_block`'s B^dag U B. Applying logical_block to them type-checks and
returns ~0.07 instead of ~0.997 -- see validation/test_phase1b.py. The Pedersen
formula itself is unchanged; only the projectors differ.

Usage:
    python3 analysis/rescore_saved_pulses.py
"""

import os
import sys

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
    cross_subspace_block,
    encode_bases,
    decode_bases,
    pedersen_gate_fidelity,
    leakage_L1,
    seepage_report,
)
# IDEAL_LOGICAL_U / N_T / DT / ALPHA are imported rather than redeclared so this
# script cannot drift from the conventions the pulses were trained and validated
# under -- in particular n_t: at n_t=2 make_hamiltonian silently drops the
# transmon anharmonicity, which would quietly score these pulses under different
# physics. GATE_PULSE_MAP is deliberately NOT imported: its PULSE_DIR is
# cwd-relative and does not resolve from here.
from validation.validate_logical_gates import IDEAL_LOGICAL_U, N_T, DT, ALPHA

PULSE_DIR = os.path.join(REPO_ROOT, "pulses")
TABLE_DIR = os.path.join(REPO_ROOT, "tables")

GATES = ["I", "X", "Y", "Z", "H", "T"]

# Encode/decode, with the basis builder that gives each one its (input, output)
# subspace pair. E_ideal comes back as eye(2) from both builders, valid only
# under the matched basis-column / training-pair order asserted in
# validation/test_phase1b.py (test b).
ENC_DEC_BASES = {"enc": encode_bases, "dec": decode_bases}

# Truncations for the enc/dec sweep. Narrower than TRUNC_LIST on purpose: this
# is a drift check across the trained set, not a held-out convergence test.
ENC_DEC_TRUNC_LIST = [22, 24, 26]

# Spread of F_avg across ENC_DEC_TRUNC_LIST above which enc/dec drift is flagged.
ENC_DEC_DRIFT_TOL = 1e-3

# Same wide range as validate_logical_gates.VALIDATION_TRUNC_RANGE.
TRUNC_LIST = list(range(18, 33, 2))
# The truncations the pulses were actually trained on (analysis/retrain_phase_coherent_gates.py).
TRAINED_TRUNC = (22, 24, 26)

# Spread of F_pedersen across HELD-OUT truncations above which we call the
# pulse's truncation convergence into question.
CONVERGENCE_TOL = 1e-3

EXPECTED_PULSE_SHAPE = (550, 4)


def _old_plus_d_fidelity(U_ideal, U_actual):
    """
    The pre-Phase-0 formula, kept here (rather than imported) so that the
    comparison column stays fixed even as validate_logical_gates evolves.
    """
    d = U_ideal.shape[0]
    overlap = np.trace(U_ideal.conj().T @ U_actual)
    return (np.abs(overlap) ** 2 + d) / (d * (d + 1))


def load_pulses(gates=GATES):
    pulses = {}
    for gate in gates:
        path = os.path.join(PULSE_DIR, f"u_{gate}_main.npy")
        u = np.load(path)
        # dt is hardcoded to DT below, so reject anything that isn't a main
        # pulse: the dt-refined pulses/*10x_light.npy are 5500x4 at dt=2e-4 and
        # would be scored at the wrong timestep.
        assert u.shape == EXPECTED_PULSE_SHAPE, (
            f"{path} has shape {u.shape}, expected {EXPECTED_PULSE_SHAPE}; this "
            f"script scores 550-step main pulses at dt={DT}"
        )
        pulses[gate] = u
    return pulses


def rescore(pulses):
    rows = []
    for n_c in TRUNC_LIST:
        # H0/Hc/B depend only on n_c, so build them once per truncation and
        # reuse across all six gates.
        H0, Hc = make_hamiltonian(N_T, n_c)
        B = logical_basis(N_T, n_c, alpha=ALPHA)
        basis_err = np.linalg.norm(B.conj().T @ B - np.eye(2))

        for gate in GATES:
            U_full = full_propagator(pulses[gate], H0, Hc, DT)
            U_log = logical_block(U_full, B)
            U_ideal = IDEAL_LOGICAL_U[gate]

            F_ped = pedersen_gate_fidelity(U_ideal, U_log)
            F_old = _old_plus_d_fidelity(U_ideal, U_log)
            L1 = leakage_L1(U_log)

            # Exact d=2 identity: the entire correction is L1/3. Free self-check
            # that F_pedersen, F_old and L1 are mutually consistent.
            assert abs((F_old - F_ped) - L1 / 3.0) < 1e-12, (
                f"{gate} n_c={n_c}: F_old - F_pedersen = {F_old - F_ped:.3e} "
                f"!= L1/3 = {L1 / 3.0:.3e}"
            )

            rows.append({
                "gate": gate,
                "n_c": n_c,
                "trained": n_c in TRAINED_TRUNC,
                "F_pedersen": F_ped,
                "F_old_plus_d": F_old,
                "F_delta": F_old - F_ped,
                "L1": L1,
                # The 2x2 unitarity defect is the physically meaningful one and
                # is directly diffable against tables/validation_master_summary.csv.
                # The D x D defect of U_full is ~1e-13 regardless of whether the
                # pulse is any good, so it is asserted in validation/test_phase0.py
                # rather than reported as a metric here.
                "unitarity_err_2x2": np.linalg.norm(U_log.conj().T @ U_log - np.eye(2)),
                # Expected to be ~1e-16 on every row. That is the point: it
                # documents that the n_c dependence below comes from truncating
                # the propagator, not from truncating the reference basis.
                "basis_orthonormality_err": basis_err,
            })
    return pd.DataFrame(rows)


def rescore_enc_dec(pulses):
    """
    Score encode/decode with the CROSS-subspace projectors.

    Same Pedersen formula as `rescore`, same L1 -- the only difference is that
    the block is B_out^dag U B_in with B_out != B_in. Here Tr(M M^dag) is the
    fraction of the input-subspace basis states that arrived in the OUTPUT
    subspace: for encode, how much of {|g,0>, |e,0>} landed in the cat code.

    Bases are rebuilt inside the n_c loop because the cat states depend on the
    truncation; reusing a single B across truncations would score every pulse
    against the n_c of whichever basis happened to be built first.
    """
    rows = []
    for n_c in ENC_DEC_TRUNC_LIST:
        H0, Hc = make_hamiltonian(N_T, n_c)
        for gate, bases_fn in ENC_DEC_BASES.items():
            B_in, B_out, E_ideal = bases_fn(N_T, n_c, alpha=ALPHA)
            U_full = full_propagator(pulses[gate], H0, Hc, DT)
            M = cross_subspace_block(U_full, B_in, B_out)

            F_avg = pedersen_gate_fidelity(E_ideal, M)
            L1 = leakage_L1(M)
            assert abs((F_avg - (np.abs(np.trace(E_ideal.conj().T @ M)) ** 2 + 2)
                        / 6.0) + L1 / 3.0) < 1e-12, (
                f"{gate} n_c={n_c}: Pedersen/L1 inconsistency"
            )

            rows.append({
                "gate": gate,
                "n_c": n_c,
                "F_avg": F_avg,
                "L1": L1,
                # Sub-unitarity of the 2x2 block == the leakage deficit. Not the
                # D x D unitarity of U_full (that is ~1e-13 for any pulse, good
                # or bad, and is asserted in validation/test_phase0.py instead).
                "unitarity_err": np.linalg.norm(M.conj().T @ M - np.eye(2)),
            })
    return pd.DataFrame(rows)


def print_enc_dec_summary(df):
    print("\n" + "=" * 78)
    print("PHASE 1B — encode/decode under the CROSS-subspace projectors")
    print("=" * 78)
    print("enc: {|g,0>,|e,0>} -> {|+Z_L>,|-Z_L>}   dec: the reverse.")
    print("Off-diagonal 2x2 block of U_full (input basis != output basis).")
    print("logical_block is the WRONG projector here and reports ~0.07; see")
    print("validation/test_phase1b.py test (d).")
    print("-" * 78)
    print(f"{'gate':<6}{'n_c':>5}{'F_avg':>12}{'L1':>13}{'unitarity_err':>16}")
    for _, r in df.iterrows():
        print(f"{r['gate']:<6}{int(r['n_c']):>5}{r['F_avg']:>12.6f}"
              f"{r['L1']:>13.3e}{r['unitarity_err']:>16.3e}")

    print("-" * 78)
    print(f"Drift of F_avg across n_c in {ENC_DEC_TRUNC_LIST} "
          f"(tol {ENC_DEC_DRIFT_TOL:.0e})")
    for gate in ENC_DEC_BASES:
        s = df[df.gate == gate]["F_avg"]
        spread = s.max() - s.min()
        flag = spread > ENC_DEC_DRIFT_TOL
        print(f"  {gate:<5}spread = {spread:.3e}"
              + ("   <-- exceeds tolerance" if flag else ""))
    print("Note: all three truncations here were TRAINED on, so a small spread is")
    print("expected and is not evidence of truncation convergence on its own.")


def print_summary(df):
    print("=" * 78)
    print("PHASE 0 — propagator-based re-scoring of saved logical-gate pulses")
    print(f"n_t={N_T}, dt={DT} us, alpha={ALPHA:.6f}, N=550 steps "
          f"({550 * DT:.3f} us), n_c in {TRUNC_LIST}")
    print("=" * 78)
    print("Headline: the old '+d' average gate fidelity over-reported by exactly")
    print("L1/3 -- at most ~1e-3 for these pulses. A real bias, a small one.")
    print("Leakage L1 is reported as a diagnostic only; nothing here was retrained.")

    print("\n" + "-" * 78)
    print(f"Per-gate detail at n_c=24 (dt={DT})")
    print("-" * 78)
    at24 = df[df["n_c"] == 24].set_index("gate")
    print(at24[["F_pedersen", "F_old_plus_d", "F_delta", "L1", "unitarity_err_2x2"]]
          .to_string(float_format=lambda x: f"{x:.6e}" if abs(x) < 1e-2 else f"{x:.9f}"))

    print("\n" + "-" * 78)
    print("F_pedersen vs truncation  (T = trained truncation)")
    print("-" * 78)
    pivot = df.pivot(index="gate", columns="n_c", values="F_pedersen").loc[GATES]
    header = "gate  " + "".join(
        f"{('T' if n_c in TRAINED_TRUNC else ' ')}{n_c:>9}" for n_c in TRUNC_LIST)
    print(header)
    for gate, row in pivot.iterrows():
        print(f"{gate:<6}" + "".join(f"{row[n_c]:>10.6f}" for n_c in TRUNC_LIST))

    print("\n" + "-" * 78)
    print("Truncation convergence  (spread = max - min of F_pedersen)")
    print("-" * 78)
    trained = df[df["trained"]]
    heldout = df[~df["trained"]]
    print(f"{'gate':<6}{'trained spread':>16}{'held-out spread':>18}{'flag':>8}")
    flagged = []
    for gate in GATES:
        s_tr = trained[trained.gate == gate]["F_pedersen"]
        s_ho = heldout[heldout.gate == gate]["F_pedersen"]
        spread_tr = s_tr.max() - s_tr.min()
        spread_ho = s_ho.max() - s_ho.min()
        flag = spread_ho > CONVERGENCE_TOL
        if flag:
            flagged.append((gate, spread_ho))
        print(f"{gate:<6}{spread_tr:>16.3e}{spread_ho:>18.3e}{'  <-- ' if flag else '':>8}")
    print(f"\nSpread across the trained truncations {TRAINED_TRUNC} is near-tautological:")
    print("  those three were optimized as a mean with their discrepancy penalized.")
    print(f"Convergence is therefore flagged on the held-out spread only (tol {CONVERGENCE_TOL:.0e}).")
    if flagged:
        for gate, spread in flagged:
            print(f"  ⚠️  {gate}: held-out spread {spread:.3e} > {CONVERGENCE_TOL:.0e} — "
                  "truncation NOT converged across the full range")
        print("  (Inspect the row above: for these pulses the deviation sits at the")
        print("   LOW-n_c end, where the cat state itself is under-resolved, and the")
        print("   high-n_c tail is flat — read it as 'n_c>=22 required', not as a")
        print("   pulse that exploits the truncation wall.)")
    else:
        print("  No gate exceeds the tolerance on held-out truncations.")


def print_seepage(pulses, n_c=24):
    """
    What the pulses do to states OUTSIDE the logical subspace -- the question a
    two-state propagation structurally cannot answer, and the reason Phase 0
    builds a full propagator at all.

    Printed, not tabulated: these are a localization diagnostic, not an
    independent error metric. For a unitary propagator, aggregate seepage and
    leakage are rigidly linked (d*L1 = (D-d)*L2), so adding these to L1 would
    double-count.
    """
    print("\n" + "-" * 78)
    print(f"Seepage: where non-logical inputs land (n_c={n_c}, D={N_T * n_c})")
    print("-" * 78)
    H0, Hc = make_hamiltonian(N_T, n_c)
    B = logical_basis(N_T, n_c, alpha=ALPHA)
    # Probes must be orthogonal to BOTH cat states. |e,0> is (different transmon
    # level); odd cavity Fock states are (the even cat code has no odd support).
    # |g,2> is NOT a valid probe despite looking like one -- it is 81.4% inside
    # the code space, since |-Z_L> peaks at n=2. See seepage_report's docstring.
    inputs = [
        ("|e,0>", basis_state(N_T, n_c, 1, 0)),
        ("|g,1>", basis_state(N_T, n_c, 0, 1)),
        ("|g,3>", basis_state(N_T, n_c, 0, 3)),
    ]
    print(f"{'gate':<6}{'input':<8}{'pop |g>':>10}{'pop |e>':>10}{'pop |f>':>10}"
          f"{'logical t=0':>13}{'logical t=T':>13}")
    for gate in GATES:
        U_full = full_propagator(pulses[gate], H0, Hc, DT)
        for rep in seepage_report(U_full, N_T, n_c, B, inputs):
            assert rep["pop_in_logical_initial"] < 1e-12, (
                f"{rep['name']} is not orthogonal to the code space "
                f"(initial logical population {rep['pop_in_logical_initial']:.3f}) "
                "-- it cannot be used as a seepage probe"
            )
            pops = rep["pop_by_transmon_level"]
            print(f"{gate:<6}{rep['name']:<8}" + "".join(f"{p:>10.6f}" for p in pops)
                  + f"{rep['pop_in_logical_initial']:>13.6f}"
                  + f"{rep['pop_in_logical']:>13.6f}")


def main():
    pulses = load_pulses()
    df = rescore(pulses)

    os.makedirs(TABLE_DIR, exist_ok=True)
    out_csv = os.path.join(TABLE_DIR, "phase0_corrected_fidelities.csv")
    df.to_csv(out_csv, index=False)

    print_summary(df)
    print_seepage(pulses)
    print(f"\nSaved: {out_csv}")

    # enc/dec go to their own CSV rather than into phase0_corrected_fidelities:
    # the columns differ (no F_old_plus_d / trained / basis_orthonormality_err)
    # and, more importantly, so do the projectors, so the two tables must not be
    # concatenated and read as one fidelity column.
    enc_dec = rescore_enc_dec(load_pulses(list(ENC_DEC_BASES)))
    enc_dec_csv = os.path.join(TABLE_DIR, "phase1b_enc_dec_fidelities.csv")
    enc_dec.to_csv(enc_dec_csv, index=False)
    print_enc_dec_summary(enc_dec)
    print(f"\nSaved: {enc_dec_csv}")


if __name__ == "__main__":
    main()
