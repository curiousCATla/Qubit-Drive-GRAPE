#!/usr/bin/env python3
"""
train_pedersen_warm.py -- retrain ONE operation with the Pedersen-fidelity
objective, warm-started from that operation's already-trained coherent pulse
instead of from a cold random start.

Motivation (experiments.ipynb Section 12.2): the cold-start Pedersen run for Y
converged to F_pedersen = 1/3. Its logical block came out purely DIAGONAL,
M = diag(0.829+0.559i, 0.972+0.231i), while the ideal U_Y is purely
off-diagonal -- so Tr(U_Y+ M) = 0 exactly, by structure, every term in the
trace being an off-diagonal entry. The gradient of the gate half of
F_pedersen is proportional to Tr(U+ M) itself, so at that point the optimizer
gets ZERO gate signal and only the leakage half is still moving; leakage was
already near its maximum, so the run sat there. 1/3 is exactly the score of a
leakage-free pulse implementing the wrong gate:

    F_pedersen = (d/(d+1)) * (|Tr(U+ M)|/d)^2 + (1/(d+1)) * (Tr(MM+)/d)
               = (2/3) * gate^2 + (1/3) * (1 - L1)      for d = 2

This script tests the cheapest available fix: start the SAME objective from a
pulse that already implements the gate, so Tr(U+ M) is never near zero. For Y
the coherent pulse starts at gate = 0.998, not 0. Because L-BFGS-B's line
search only accepts steps that decrease the cost, a run starting at
F_pedersen = 0.9959 cannot reach the F_pedersen = 0.333 plateau at all -- the
trap is strictly uphill in cost from here, so it is unreachable rather than
merely unlikely.

Everything else is byte-identical to analysis/train_pedersen_gates.py: same
OPTIMIZATION_RECIPE, same PRODUCTION_MAXITER, same objective, same scoring
sweep. The ONLY change is `warm_start`. All machinery is imported from that
script rather than reimplemented, so the two cannot drift apart.

Caveat this run must be read against: warm-starting is the pattern README.md's
"Truncation convergence and wall exploitation" section warns about. It is
mitigated here -- trunc_list stays [22, 24, 26] (three truncations, not one),
the warm start was itself trained on those same three, and scoring still sweeps
the full SCORE_TRUNC_LIST -- but read the HELD-OUT column, not the trained
mean, before believing the result.

Writes NEW filenames only. pulses/u_<gate>_pedersen.npy (the cold-start
result) is never overwritten: it is the evidence for the failure mode.

Usage:
    python3 analysis/train_pedersen_warm.py                  # Y, warm from u_Y_coh_fresh.npy
    python3 analysis/train_pedersen_warm.py --gate X
    python3 analysis/train_pedersen_warm.py --gate Y --init pulses/u_Y_main.npy
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.grape_core import make_hamiltonian
from core.optimizer import optimize_multi_state_pulse
from core.propagator import (
    logical_basis,
    encode_bases,
    decode_bases,
    full_propagator,
    logical_block,
    cross_subspace_block,
)

from analysis.train_pedersen_gates import (
    ALPHA,
    DT,
    LOGICAL_GATES,
    LOG_DIR,
    N_STEPS,
    N_T,
    OPTIMIZATION_RECIPE,
    PRODUCTION_MAXITER,
    PULSE_DIR,
    RESULTS_DIR,
    TABLE_DIR,
    TRAINED_TRUNC,
    GATE_FACTORIES,
    get_ideal_2x2,
    make_pedersen_fidelity_fn,
    make_pedersen_probe_factory,
    score_pulse,
    self_test,
    summarize_score,
)

# The truncation the notebook's Section 12.1 benchmarks at, and the largest
# training truncation -- used only for the diagnostic print below, never for
# training or scoring.
DIAG_N_C = max(OPTIMIZATION_RECIPE["trunc_list"])


def gate_and_leak_terms(u, gate, n_c=DIAG_N_C):
    """The two halves of F_pedersen, separately, at one truncation.

    Returns (gate, leak) with gate = |Tr(U_ideal+ M)| / d in [0, 1] and
    leak = Tr(MM+) / d = 1 - L1 in [0, 1], so that
    F_pedersen = (2/3) gate^2 + (1/3) leak for d = 2. `gate` is the quantity
    whose vanishing defines the plateau this script is trying to avoid, which
    is why it is reported on its own rather than folded into F_pedersen.
    """
    H0, Hc = make_hamiltonian(N_T, n_c)
    U_full = full_propagator(u, H0, Hc, DT)
    if gate in LOGICAL_GATES:
        B = logical_basis(N_T, n_c, alpha=ALPHA)
        M = logical_block(U_full, B)
        U_ideal = get_ideal_2x2(gate)
    elif gate == "enc":
        B_in, B_out, U_ideal = encode_bases(N_T, n_c, alpha=ALPHA)
        M = cross_subspace_block(U_full, B_in, B_out)
    elif gate == "dec":
        B_in, B_out, U_ideal = decode_bases(N_T, n_c, alpha=ALPHA)
        M = cross_subspace_block(U_full, B_in, B_out)
    else:
        raise ValueError(gate)
    d = U_ideal.shape[0]
    return (abs(np.trace(U_ideal.conj().T @ M)) / d,
            np.trace(M.conj().T @ M).real / d)


def truncation_table(gate, tag, variants=("main", "coh_fresh", "pedersen")):
    """Per-truncation F_pedersen and L1 for every saved variant of one gate.

    The summary rows in results/pedersen_vs_main_comparison.json keep only
    trained/held-out means, which cannot distinguish "this pulse degrades away
    from its training truncations" (wall exploitation, the failure mode
    README.md warns about) from "every variant dips at the smallest n_c"
    (a Hilbert space too small to hold the alpha=sqrt(3) cat at all). Reading
    the held-out MINIMUM without this table invites exactly that confusion.
    """
    frames = []
    for v in list(variants) + [tag]:
        path = os.path.join(PULSE_DIR, f"u_{gate}_{v}.npy")
        if not os.path.exists(path):
            continue
        df = score_pulse(np.load(path), gate)
        df["method"] = v
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gate", default="Y", choices=list(GATE_FACTORIES),
                        help="Operation to retrain (default: Y).")
    parser.add_argument("--init", default=None,
                        help="Warm-start pulse (default: pulses/u_<gate>_coh_fresh.npy).")
    parser.add_argument("--tag", default="pedersen_warm",
                        help="Suffix for the new pulse/log/results filenames.")
    parser.add_argument("--skip-self-test", action="store_true",
                        help="Skip the finite-difference gradient check.")
    parser.add_argument("--score-only", action="store_true",
                        help="Do not train: re-score the already-saved "
                             "u_<gate>_<tag>.npy and rebuild the output tables.")
    args = parser.parse_args()

    gate = args.gate
    init_path = args.init or os.path.join(PULSE_DIR, f"u_{gate}_coh_fresh.npy")
    save_path = os.path.join(PULSE_DIR, f"u_{gate}_{args.tag}.npy")
    log_path = os.path.join(LOG_DIR, f"train_{gate}_{args.tag}.log")

    if not os.path.exists(init_path):
        raise SystemExit(f"warm-start pulse not found: {init_path}")
    if os.path.abspath(save_path) == os.path.abspath(init_path):
        raise SystemExit("refusing to overwrite the warm-start pulse")
    cold_path = os.path.join(PULSE_DIR, f"u_{gate}_pedersen.npy")
    if os.path.abspath(save_path) == os.path.abspath(cold_path):
        raise SystemExit("refusing to overwrite the cold-start Pedersen pulse "
                         "(it is the evidence for the failure mode)")

    if args.score_only and not os.path.exists(save_path):
        raise SystemExit(f"--score-only needs an existing pulse: {save_path}")
    if not args.score_only and not args.skip_self_test:
        print(f"Finite-difference gradient self-test for {gate}...")
        self_test(gate=gate)
        print("Self-test passed.\n")

    u_init = np.load(init_path)
    g0, l0 = gate_and_leak_terms(u_init, gate)
    print(f"{'='*70}\n{gate} / pedersen, WARM start from {os.path.basename(init_path)}\n{'='*70}")
    print(f"  warm start at n_c={DIAG_N_C}:  gate = {g0:.4f}   1-L1 = {l0:.6f}   "
          f"F_pedersen = {2/3*g0**2 + 1/3*l0:.6f}")
    print( "  (the plateau this run is avoiding sits at gate = 0, F_pedersen = 1/3)\n")

    recipe = dict(OPTIMIZATION_RECIPE, warm_start=init_path)

    if args.score_only:
        prev = json.load(open(os.path.join(RESULTS_DIR, f"pedersen_warm_{gate}.json")))
        u_opt = np.load(save_path)
        wall_clock_s = prev["wall_clock_s"]
        info = {k: prev[k] for k in ("iterations", "success", "final_fidelity")}
        print(f"--score-only: re-scoring {os.path.basename(save_path)}, trained "
              f"earlier in {wall_clock_s:.1f} s. No optimization run.\n")
    else:
        get_state_pairs = make_pedersen_probe_factory(gate)
        fidelity_fn = make_pedersen_fidelity_fn(get_ideal_2x2(gate))

        t0 = time.perf_counter()
        with open(log_path, "w") as logf:
            class _Tee:
                def write(self, s):
                    sys.__stdout__.write(s)
                    logf.write(s)
                def flush(self):
                    sys.__stdout__.flush()
                    logf.flush()
            old_stdout = sys.stdout
            sys.stdout = _Tee()
            try:
                u_opt, info = optimize_multi_state_pulse(
                    get_state_pairs=get_state_pairs,
                    n_t=N_T, N=N_STEPS, dt=DT,
                    maxiter=PRODUCTION_MAXITER,
                    verbose=True,
                    fidelity_fn=fidelity_fn,
                    save_path=save_path,
                    **recipe,
                )
            finally:
                sys.stdout = old_stdout
        wall_clock_s = time.perf_counter() - t0

    g1, l1 = gate_and_leak_terms(u_opt, gate)
    df_score = score_pulse(u_opt, gate)
    summary = summarize_score(df_score)

    print(f"\ntrained in {wall_clock_s:.1f} s, {info['iterations']} iterations, "
          f"success={info['success']}")
    print(f"  gate term {g0:.4f} -> {g1:.4f}   1-L1 {l0:.6f} -> {l1:.6f}")
    print(f"\nper-truncation score for u_{gate}_{args.tag}.npy:")
    print(df_score.to_string(index=False, float_format="%.6f"))

    # Compare against the three variants already in the Section 12.2 table.
    ref_csv = os.path.join(TABLE_DIR, "pedersen_vs_main_comparison.csv")
    cols = ["method", "F_pedersen_trained_mean", "F_pedersen_heldout_mean",
            "F_pedersen_heldout_min", "L1_trained_mean", "L1_heldout_mean"]
    rows = []
    if os.path.exists(ref_csv):
        ref = pd.read_csv(ref_csv)
        rows.append(ref[ref["gate"] == gate][cols])
    rows.append(pd.DataFrame([dict(method=args.tag, **summary)])[cols])
    df_cmp = pd.concat(rows, ignore_index=True)
    print(f"\n{gate}: all variants")
    print(df_cmp.to_string(index=False, float_format="%.6f"))

    # Per-truncation curves for every variant. Needed to tell a wall-exploiting
    # pulse (degrades AWAY from the trained truncations, especially upward)
    # from the small-n_c dip that every variant shares.
    df_trunc = truncation_table(gate, args.tag)
    print(f"\n{gate}: F_pedersen vs truncation, all variants "
          f"(trained: {', '.join(str(t) for t in sorted(TRAINED_TRUNC))})")
    print(df_trunc.pivot(index="n_c", columns="method", values="F_pedersen")
                  .to_string(float_format="%.6f"))
    print(f"\n{gate}: L1 vs truncation, all variants")
    print(df_trunc.pivot(index="n_c", columns="method", values="L1")
                  .to_string(float_format="%.6f"))

    # New filenames only -- results/pedersen_train_rows.json and the Section
    # 12.2 comparison CSV are left exactly as they are.
    out_json = os.path.join(RESULTS_DIR, f"pedersen_warm_{gate}.json")
    with open(out_json, "w") as f:
        json.dump(dict(
            gate=gate, method=args.tag, init_pulse=init_path, pulse_path=save_path,
            recipe=recipe, maxiter=PRODUCTION_MAXITER,
            wall_clock_s=wall_clock_s, iterations=info["iterations"],
            success=info["success"], final_fidelity=info["final_fidelity"],
            gate_term_before=g0, gate_term_after=g1,
            leak_term_before=l0, leak_term_after=l1,
            summary=summary, score_rows=df_score.to_dict("records"),
        ), f, indent=1)
    out_csv = os.path.join(TABLE_DIR, f"pedersen_warm_{gate}.csv")
    df_cmp.to_csv(out_csv, index=False)
    trunc_csv = os.path.join(TABLE_DIR, f"pedersen_warm_{gate}_truncation.csv")
    df_trunc.to_csv(trunc_csv, index=False)
    print(f"\nSaved: {save_path}\n       {out_json}\n       {out_csv}"
          f"\n       {trunc_csv}\n       {log_path}")


if __name__ == "__main__":
    main()
