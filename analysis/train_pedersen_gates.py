#!/usr/bin/env python3
"""
train_pedersen_gates.py -- train all 8 of Section 5's operations (encode,
decode, X, Y, Z, H, T, I) with a leakage-aware, full-propagator Pedersen-
fidelity objective (numpy adjoint, no JAX) in place of the production
2-state coherent objective, and compare against a freshly-trained coherent
baseline under the identical recipe/budget.

Background: experiments.ipynb Section 12 only benchmarked the COST of a
single Pedersen-fidelity gradient call (numpy adjoint: ~1.01x the current
training objective's own cost) on the already-trained u_X_main.npy pulse --
it never actually trained anything. This script does the real thing: for
each operation, run core.optimizer.optimize_multi_state_pulse() twice under
the exact experiments.ipynb OPTIMIZATION_RECIPE (trunc_list=[22,24,26],
penalties, bands, hard_amp_limit, n_jobs=3, cold start, maxiter=1500 =
PRODUCTION_MAXITER) -- once with coherent_fidelity_multi_state (a FRESH
baseline, timed the same way, since no historical timing exists anywhere for
the on-disk u_<gate>_main.npy pulses -- results/gate_campaign_info.json has
Iterations/Wall_clock_s = NaN for every gate), and once with a new
Pedersen-fidelity objective built from _fidelity_core's raw per-pair output.

Encode/decode are NOT code-space-to-code-space like the six logical gates --
they map between the transmon computational subspace and the cat code
subspace -- so their Pedersen probe pairs and scoring use
core.propagator.encode_bases/decode_bases/cross_subspace_block (the same
machinery analysis/rescore_saved_pulses.py's rescore_enc_dec uses) instead of
logical_basis/logical_block. See make_pedersen_probe_factory/score_pulse.

New pulses are saved under NEW filenames (u_<gate>_coh_fresh.npy,
u_<gate>_pedersen.npy) -- pulses/u_<gate>_main.npy is never read for warm
start or written to. Already-trained (gate, method) pairs found in
results/pedersen_train_rows.json are skipped on re-run (idempotent), so
re-running after adding new gates does not retrain ones already done.

Both objectives' pulses, plus the existing production u_<gate>_main.npy, are
then scored across a truncation sweep with the honest Pedersen gate fidelity
and L1 leakage (core.propagator, called directly -- validate_logical_gates.py
and rescore_saved_pulses.py are both hardcoded to the u_<gate>_main.npy
filename pattern and have no override).

Timing caveat: this machine's process contention can swing measured
per-call/per-run wall-clock by a large factor (a concurrent EsT benchmark on
this repo found a 4.3x-to-30-54x swing) -- treat wall_clock_s here as
indicative, not precise, especially when comparing two runs from different
points in a long sequential batch.

Usage:
    python3 analysis/train_pedersen_gates.py                 # self-test + full run
    python3 analysis/train_pedersen_gates.py --self-test-only # just the FD check
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

from core.grape_core import (
    make_hamiltonian,
    coherent_fidelity_multi_state,
    _fidelity_core,
)
from core.cat_code import (
    get_encode_state_pairs,
    get_decode_state_pairs,
    get_logical_X_state_pairs,
    get_logical_Y_state_pairs,
    get_logical_Z_state_pairs,
    get_logical_H_state_pairs,
    get_logical_T_state_pairs,
    get_identity_state_pairs,
)
from core.propagator import (
    logical_basis,
    encode_bases,
    decode_bases,
    full_propagator,
    logical_block,
    cross_subspace_block,
    pedersen_gate_fidelity,
    leakage_L1,
)
from core.optimizer import optimize_multi_state_pulse
# IDEAL_LOGICAL_U / N_T / DT / ALPHA imported (not redeclared) so this script
# cannot silently drift from the physics/convention the pulses are trained
# and scored under -- same rationale as rescore_saved_pulses.py.
from validation.validate_logical_gates import IDEAL_LOGICAL_U, N_T, DT, ALPHA

PULSE_DIR = os.path.join(REPO_ROOT, "pulses")
LOG_DIR = os.path.join(REPO_ROOT, "logs")
TABLE_DIR = os.path.join(REPO_ROOT, "tables")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
for _d in (PULSE_DIR, LOG_DIR, TABLE_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# Matches Section 5's GATE_FACTORIES_FULL order exactly: "encode, decode, X,
# Y, Z, H, T, I" (its own docstring).
GATE_FACTORIES = {
    "enc": get_encode_state_pairs,
    "dec": get_decode_state_pairs,
    "X": get_logical_X_state_pairs,
    "Y": get_logical_Y_state_pairs,
    "Z": get_logical_Z_state_pairs,
    "H": get_logical_H_state_pairs,
    "T": get_logical_T_state_pairs,
    "I": get_identity_state_pairs,
}
GATES = list(GATE_FACTORIES.keys())

# The six single-qubit logical gates train/score code-space-to-code-space
# (one basis, logical_basis, for both input and output). enc/dec map between
# two DIFFERENT subspaces and need cross_subspace_block/encode_bases/
# decode_bases instead -- see make_pedersen_probe_factory and score_pulse.
LOGICAL_GATES = ["I", "X", "Y", "Z", "H", "T"]
CROSS_GATES = ["enc", "dec"]

N_STEPS = 550

# Exactly experiments.ipynb's OPTIMIZATION_RECIPE (cell a17d42af) +
# PRODUCTION_MAXITER (the value Section 5's own code uses when
# RUN_OPTIMIZATION=True, not a rough estimate).
OPTIMIZATION_RECIPE = dict(
    trunc_list=[22, 24, 26],
    penalties={"deriv": 1e-5, "boundary": 2e-5, "amp": 8e-5, "amp_max": 40.0, "disc": 0.5},
    warm_start=None,
    cav_band=(-27.0, 27.0),
    tra_band=(-33.0, 33.0),
    hard_amp_limit=40.0,
    n_jobs=3,
)
PRODUCTION_MAXITER = 1500

# Same wide range as validation/validate_logical_gates.py's
# VALIDATION_TRUNC_RANGE / analysis/rescore_saved_pulses.py's TRUNC_LIST.
SCORE_TRUNC_LIST = list(range(18, 33, 2))
TRAINED_TRUNC = (22, 24, 26)


# ============================================================
# Pedersen fidelity objective -- per-operation probe states, gate-specific
# reduction. Generalizes experiments.ipynb cell 4cb95b5c's
# pedersen_fidelity_and_grad_numpy (X-only, want_grad=True only, module-scope
# globals) to any operation and to optimize_multi_state_pulse's real
# fidelity_fn contract:
# (u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad) -> (F, grad_or_None).
# ============================================================

# Fixed pairing convention shared between make_pedersen_probe_factory and
# make_pedersen_fidelity_fn: (r, c) for c in range(2) for r in range(2), i.e.
# (0,0), (1,0), (0,1), (1,1). Both must agree on this order or the reduction
# below silently scores the wrong 2x2 map.
_PAIRS = [(r, c) for c in range(2) for r in range(2)]


def make_pedersen_probe_factory(gate):
    """
    Returns get_state_pairs(n_c, n_t) -> 4 (psi_i, psi_f) pairs, built from
    B_in/B_out's columns, matching _PAIRS order.

    For the six logical gates, B_in == B_out == logical_basis(...) (one
    basis, code-space-to-code-space -- logical_basis does not depend on
    which gate is being trained, only on n_c/n_t/alpha). For enc/dec, B_in
    and B_out are the two DIFFERENT subspaces encode_bases/decode_bases
    return (transmon computational subspace <-> cat code subspace) -- the
    same bases analysis/rescore_saved_pulses.py's rescore_enc_dec scores
    with via cross_subspace_block.
    """
    if gate in LOGICAL_GATES:
        def get_state_pairs(n_c=24, n_t=3, alpha=ALPHA):
            B = logical_basis(n_t, n_c, alpha=alpha)
            return [(B[:, c], B[:, r]) for (r, c) in _PAIRS]
    elif gate == "enc":
        def get_state_pairs(n_c=24, n_t=3, alpha=ALPHA):
            B_in, B_out, _ = encode_bases(n_t, n_c, alpha=alpha)
            return [(B_in[:, c], B_out[:, r]) for (r, c) in _PAIRS]
    elif gate == "dec":
        def get_state_pairs(n_c=24, n_t=3, alpha=ALPHA):
            B_in, B_out, _ = decode_bases(n_t, n_c, alpha=alpha)
            return [(B_in[:, c], B_out[:, r]) for (r, c) in _PAIRS]
    else:
        raise ValueError(gate)
    return get_state_pairs


def get_ideal_2x2(gate):
    """The 2x2 target M is scored/trained against: IDEAL_LOGICAL_U[gate] for
    the six logical gates, or eye(2) for enc/dec (encode_bases/decode_bases'
    E_ideal -- valid only because get_encode_state_pairs/get_decode_state_pairs'
    pair order matches B_in/B_out's column order, asserted in
    validation/test_phase1b.py test (b))."""
    if gate in LOGICAL_GATES:
        return IDEAL_LOGICAL_U[gate]
    elif gate in CROSS_GATES:
        return np.eye(2, dtype=complex)
    else:
        raise ValueError(gate)


def make_pedersen_fidelity_fn(U_ideal):
    """
    Returns a fidelity_fn(u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad)
    closure computing the honest Pedersen (2007) gate fidelity and its
    analytic gradient via core.grape_core._fidelity_core's raw per-pair
    output -- the same construction Section 12 prototyped for X, generalized
    to any 2x2 ideal target and given a real want_grad=False branch (needed:
    optimize_multi_state_pulse calls fidelity_fn(..., want_grad=False) for
    post-optimization diagnostics).

    d=2 throughout -- IDEAL_LOGICAL_U entries are all 2x2.
    """
    d = 2
    coeffs = U_ideal.conj().T.reshape(-1, order='F')  # matches _PAIRS order

    def fidelity_fn(u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=True):
        if want_grad:
            F, grad, v, dv_all = _fidelity_core(
                u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=True, return_raw=True
            )
            S = np.sum(coeffs * v)
            dS = np.tensordot(dv_all, coeffs, axes=([2], [0]))
            leak = np.sum(F)
            dleak = np.sum(grad, axis=2)
            F_ped = (np.abs(S) ** 2 + leak) / (d * (d + 1))
            dF_ped = (2.0 * np.real(np.conj(S) * dS) + dleak) / (d * (d + 1))
            return F_ped, dF_ped
        else:
            F, _, v, _ = _fidelity_core(
                u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=False, return_raw=True
            )
            S = np.sum(coeffs * v)
            leak = np.sum(F)
            F_ped = (np.abs(S) ** 2 + leak) / (d * (d + 1))
            return F_ped, None

    return fidelity_fn


def self_test(gate="X", n_c=22, eps=1e-6, idx=(100, 1), seed=0):
    """Finite-difference check of make_pedersen_fidelity_fn's analytic
    gradient, run before any real training to catch a reduction bug cheaply
    (mirrors Section 12's own eps=1e-6 check, generalized off a random pulse
    rather than the already-converged u_X_main.npy). Works for any of the 8
    operations, including enc/dec's asymmetric-basis probes."""
    rng = np.random.default_rng(seed)
    u = rng.normal(scale=2.0, size=(N_STEPS, 4))
    H0, Hc = make_hamiltonian(N_T, n_c)
    get_state_pairs = make_pedersen_probe_factory(gate)
    psi_i_list, psi_f_list = zip(*get_state_pairs(n_c=n_c, n_t=N_T))
    psi_i_list, psi_f_list = list(psi_i_list), list(psi_f_list)
    fidelity_fn = make_pedersen_fidelity_fn(get_ideal_2x2(gate))

    F0, grad = fidelity_fn(u, H0, Hc, psi_i_list, psi_f_list, DT, want_grad=True)

    u_p = u.copy(); u_p[idx] += eps
    u_m = u.copy(); u_m[idx] -= eps
    Fp, _ = fidelity_fn(u_p, H0, Hc, psi_i_list, psi_f_list, DT, want_grad=False)
    Fm, _ = fidelity_fn(u_m, H0, Hc, psi_i_list, psi_f_list, DT, want_grad=False)
    fd = (Fp - Fm) / (2 * eps)
    analytic = grad[idx]

    ok = abs(analytic - fd) < 1e-6
    print(f"[self-test] gate={gate} n_c={n_c} F={F0:.6f} "
          f"analytic grad[{idx}]={analytic:.6e} finite-diff={fd:.6e} "
          f"{'OK' if ok else 'MISMATCH'}")
    assert ok, "Pedersen fidelity gradient failed finite-difference check"
    return True


# ============================================================
# Training
# ============================================================

def train_one(gate, method, save_path, log_path):
    """method in {'coherent', 'pedersen'}. Both branches call
    optimize_multi_state_pulse with IDENTICAL settings from
    OPTIMIZATION_RECIPE/PRODUCTION_MAXITER -- only get_state_pairs and
    fidelity_fn differ."""
    if method == "coherent":
        get_state_pairs = GATE_FACTORIES[gate]
        fidelity_fn = coherent_fidelity_multi_state
    elif method == "pedersen":
        get_state_pairs = make_pedersen_probe_factory(gate)
        fidelity_fn = make_pedersen_fidelity_fn(get_ideal_2x2(gate))
    else:
        raise ValueError(method)

    print(f"\n{'#'*70}\n# {gate} / {method} -> {save_path}\n{'#'*70}")

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
                **OPTIMIZATION_RECIPE,
            )
        finally:
            sys.stdout = old_stdout
    wall_clock_s = time.perf_counter() - t0

    return dict(
        gate=gate, method=method, pulse_path=save_path,
        wall_clock_s=wall_clock_s,
        iterations=info["iterations"],
        final_fidelity=info["final_fidelity"],
        best_bare_F_during_opt=info["best_bare_F_during_opt"],
        success=info["success"],
    )


# ============================================================
# Scoring -- direct core.propagator calls, not the hardcoded
# validate_logical_gates.py / rescore_saved_pulses.py scripts, since both
# only ever look up pulses/u_<gate>_main.npy.
# ============================================================

def score_pulse(u, gate):
    """F_pedersen and leakage_L1 across SCORE_TRUNC_LIST. Returns a DataFrame
    with one row per truncation. Logical gates use logical_block (one basis,
    B_in==B_out); enc/dec use cross_subspace_block with the two DIFFERENT
    bases encode_bases/decode_bases return -- same construction as
    analysis/rescore_saved_pulses.py's rescore_enc_dec, but swept over the
    same wide SCORE_TRUNC_LIST used for the logical gates (a held-out
    convergence check) rather than that script's narrower enc/dec-specific
    [22,24,26] drift-only range."""
    rows = []
    for n_c in SCORE_TRUNC_LIST:
        H0, Hc = make_hamiltonian(N_T, n_c)
        U_full = full_propagator(u, H0, Hc, DT)
        if gate in LOGICAL_GATES:
            B = logical_basis(N_T, n_c, alpha=ALPHA)
            U_log = logical_block(U_full, B)
            U_ideal = IDEAL_LOGICAL_U[gate]
        elif gate == "enc":
            B_in, B_out, U_ideal = encode_bases(N_T, n_c, alpha=ALPHA)
            U_log = cross_subspace_block(U_full, B_in, B_out)
        elif gate == "dec":
            B_in, B_out, U_ideal = decode_bases(N_T, n_c, alpha=ALPHA)
            U_log = cross_subspace_block(U_full, B_in, B_out)
        else:
            raise ValueError(gate)
        rows.append(dict(
            n_c=n_c,
            trained=n_c in TRAINED_TRUNC,
            F_pedersen=pedersen_gate_fidelity(U_ideal, U_log),
            L1=leakage_L1(U_log),
        ))
    return pd.DataFrame(rows)


def summarize_score(df_score):
    trained = df_score[df_score["trained"]]
    heldout = df_score[~df_score["trained"]]
    return dict(
        F_pedersen_trained_mean=trained["F_pedersen"].mean(),
        F_pedersen_heldout_mean=heldout["F_pedersen"].mean(),
        F_pedersen_heldout_min=heldout["F_pedersen"].min(),
        L1_trained_mean=trained["L1"].mean(),
        L1_heldout_mean=heldout["L1"].mean(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test-only", action="store_true",
                         help="Run only the finite-difference gradient check and exit.")
    args = parser.parse_args()

    print("Running Pedersen-fidelity gradient self-test for each operation...")
    for gate in GATES:
        self_test(gate=gate)
    print("Self-test passed for all operations.\n")

    if args.self_test_only:
        return

    # Idempotent: load any training rows a previous run already produced (the
    # X/H/T round) and skip re-training those (gate, method) pairs -- only
    # train what's missing.
    train_rows_path = os.path.join(RESULTS_DIR, "pedersen_train_rows.json")
    if os.path.exists(train_rows_path):
        with open(train_rows_path) as f:
            train_rows = json.load(f)
    else:
        train_rows = []
    done_keys = {(r["gate"], r["method"]) for r in train_rows}

    for gate in GATES:
        for method in ["coherent", "pedersen"]:
            if (gate, method) in done_keys:
                print(f"[skip] {gate}/{method}: already trained (found in {train_rows_path})")
                continue
            suffix = "coh_fresh" if method == "coherent" else "pedersen"
            save_path = os.path.join(PULSE_DIR, f"u_{gate}_{suffix}.npy")
            log_path = os.path.join(LOG_DIR, f"train_{gate}_{suffix}.log")
            row = train_one(gate, method, save_path, log_path)
            train_rows.append(row)
            # Persist incrementally so a partial run isn't a total loss.
            with open(train_rows_path, "w") as f:
                json.dump(train_rows, f, indent=2, default=str)

    print("\nAll training runs complete. Scoring...")

    score_rows = []
    for gate in GATES:
        pulse_paths = {
            "main": os.path.join(PULSE_DIR, f"u_{gate}_main.npy"),
            "coh_fresh": os.path.join(PULSE_DIR, f"u_{gate}_coh_fresh.npy"),
            "pedersen": os.path.join(PULSE_DIR, f"u_{gate}_pedersen.npy"),
        }
        for label, path in pulse_paths.items():
            if not os.path.exists(path):
                print(f"[skip] {gate}/{label}: {path} not found")
                continue
            u = np.load(path)
            df_score = score_pulse(u, gate)
            summary = summarize_score(df_score)
            summary.update(gate=gate, method=label, pulse_path=path)
            score_rows.append(summary)

    df_train = pd.DataFrame(train_rows)
    df_score_summary = pd.DataFrame(score_rows)

    # method: main has no training row (never trained here); merge on gate +
    # method label, matching "coh_fresh"/"pedersen" between the two frames
    # and leaving "main" rows with NaN training columns.
    train_method_map = {"coherent": "coh_fresh", "pedersen": "pedersen"}
    df_train_for_merge = df_train.copy()
    df_train_for_merge["method"] = df_train_for_merge["method"].map(train_method_map)

    df_combined = df_score_summary.merge(
        df_train_for_merge[["gate", "method", "wall_clock_s", "iterations", "final_fidelity"]],
        on=["gate", "method"], how="left",
    )

    csv_path = os.path.join(TABLE_DIR, "pedersen_vs_main_comparison.csv")
    df_combined.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(df_combined.to_string(index=False, float_format="%.6f"))

    json_path = os.path.join(RESULTS_DIR, "pedersen_vs_main_comparison.json")
    with open(json_path, "w") as f:
        json.dump(
            dict(
                recipe=OPTIMIZATION_RECIPE, maxiter=PRODUCTION_MAXITER,
                train_rows=train_rows, score_rows=score_rows,
            ),
            f, indent=2, default=str,
        )
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
