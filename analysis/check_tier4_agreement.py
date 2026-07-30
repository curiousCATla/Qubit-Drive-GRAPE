#!/usr/bin/env python3
"""
check_tier4_agreement.py — Phase 1c gate: prove the two enc/dec measuring sticks
already agree, BEFORE consolidating them.

`validation.validate_logical_gates.tier4_enc_relative_phase` /
`tier4_dec_relative_phase` and the Phase-1b block route compute the same 2x2
cross-subspace map by different means:

  old : propagate |g,0>/|e,0> (enc) as state vectors, then assemble four
        <cat_k|out_j> overlaps by hand.
  new : build the full D x D propagator once, then project B_out^dag U B_in.

Both compare to eye(2) and -- since Phase 0 replaced the bare `+d` in
`average_gate_fidelity` with Tr(M M^dag) -- both use the same Pedersen formula.
So this must pass. It is run first precisely so that the refactor is a
consolidation of two verified-identical routes rather than a silent swap of one
number for another.

If it FAILS, do not average the two and do not adjust the block route. The block
route is authoritative (it also yields leakage for free). Diagnose the wiring:
  (a) a stale/unfixed average_gate_fidelity still carrying `+ d`;
  (b) the legacy basis construction disagreeing with logical_basis /
      transmon_basis on alpha, normalization or index order;
  (c) an n_c or dt mismatch between the two paths.

IMPORTANT -- why the legacy route is FROZEN here rather than called from
validate_logical_gates: after Phase 1c, `tier4_enc_relative_phase` *is* the
block route. Comparing it against the block route compares the block route to
itself and passes unconditionally -- it printed exactly 0.00e+00 when first run
that way. The pre-refactor algorithm is therefore reproduced verbatim below as
`_legacy_*`, so this remains a real two-route check that can still fail if
either side drifts. `_legacy_average_gate_fidelity` is likewise a frozen copy of
the Phase-0-corrected Pedersen form, inlined so the comparison stays fixed even
if validate_logical_gates' own copy is later edited.

The live tier4 functions are checked too, against the same frozen reference, so
the delegation itself is covered rather than assumed.

Usage:
    python3 analysis/check_tier4_agreement.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.cat_code import get_logical_cat_states, embed_in_joint_space
from core.grape_core import make_hamiltonian, step_data
from core.propagator import (
    full_propagator,
    cross_subspace_block,
    encode_bases,
    decode_bases,
    pedersen_gate_fidelity,
    leakage_L1,
)
from validation.validate_logical_gates import (
    tier4_enc_relative_phase,
    tier4_dec_relative_phase,
    N_T,
    DT,
    ALPHA,
)

PULSE_DIR = os.path.join(REPO_ROOT, "pulses")
TRUNC_LIST = [22, 24, 26]
TOL = 1e-9
EXPECTED_PULSE_SHAPE = (550, 4)

# enc: in=transmon, out=cat. dec: in=cat, out=transmon. Do not swap.
CASES = {
    "enc": (tier4_enc_relative_phase, encode_bases),
    "dec": (tier4_dec_relative_phase, decode_bases),
}


# ------------------------------------------------------------------
# FROZEN pre-Phase-1c implementation. Do not "simplify" this to call the
# current helpers -- its whole value is being an independent second route.
# ------------------------------------------------------------------

def _legacy_average_gate_fidelity(U_ideal, U_actual):
    """Frozen copy of the Phase-0-corrected Pedersen form."""
    d = U_ideal.shape[0]
    overlap = np.trace(U_ideal.conj().T @ U_actual)
    leak_term = np.trace(U_actual.conj().T @ U_actual).real
    return (np.abs(overlap) ** 2 + leak_term) / (d * (d + 1))


def _legacy_propagate(u, H0, Hc, psi0, dt):
    psi = psi0.copy().astype(complex)
    for uk in u:
        Uk, _, _ = step_data(H0, Hc, uk, dt)
        psi = Uk @ psi
    return psi


def _legacy_computational_basis_states(n_c, n_t):
    init_g = np.zeros(n_t * n_c, dtype=complex); init_g[0] = 1.0
    init_e = np.zeros(n_t * n_c, dtype=complex); init_e[n_c] = 1.0
    return init_g, init_e


def _legacy_cat_basis_states(n_c, n_t):
    psi_p_cav, psi_m_cav = get_logical_cat_states(alpha=ALPHA, n_c=n_c)
    psi_pg = embed_in_joint_space(psi_p_cav, n_t=n_t, n_c=n_c, t_level=0)
    psi_mg = embed_in_joint_space(psi_m_cav, n_t=n_t, n_c=n_c, t_level=0)
    return psi_pg, psi_mg


def legacy_tier4_enc(u_enc, n_c, n_t, dt):
    """Pre-Phase-1c tier4_enc_relative_phase: rows = cat, cols = transmon."""
    H0, Hc = make_hamiltonian(n_t, n_c)
    init_g, init_e = _legacy_computational_basis_states(n_c, n_t)
    psi_pg, psi_mg = _legacy_cat_basis_states(n_c, n_t)
    out_g = _legacy_propagate(u_enc, H0, Hc, init_g, dt)
    out_e = _legacy_propagate(u_enc, H0, Hc, init_e, dt)
    U_log = np.array([
        [np.vdot(psi_pg, out_g), np.vdot(psi_pg, out_e)],
        [np.vdot(psi_mg, out_g), np.vdot(psi_mg, out_e)],
    ])
    return {'U_log': U_log,
            'F_avg_gate': _legacy_average_gate_fidelity(np.eye(2, dtype=complex), U_log)}


def legacy_tier4_dec(u_dec, n_c, n_t, dt):
    """Pre-Phase-1c tier4_dec_relative_phase: rows = transmon, cols = cat."""
    H0, Hc = make_hamiltonian(n_t, n_c)
    init_g, init_e = _legacy_computational_basis_states(n_c, n_t)
    psi_pg, psi_mg = _legacy_cat_basis_states(n_c, n_t)
    out_p = _legacy_propagate(u_dec, H0, Hc, psi_pg, dt)
    out_m = _legacy_propagate(u_dec, H0, Hc, psi_mg, dt)
    U_log = np.array([
        [np.vdot(init_g, out_p), np.vdot(init_g, out_m)],
        [np.vdot(init_e, out_p), np.vdot(init_e, out_m)],
    ])
    return {'U_log': U_log,
            'F_avg_gate': _legacy_average_gate_fidelity(np.eye(2, dtype=complex), U_log)}


LEGACY = {"enc": legacy_tier4_enc, "dec": legacy_tier4_dec}


def _load(gate):
    u = np.load(os.path.join(PULSE_DIR, f"u_{gate}_main.npy"))
    assert u.shape == EXPECTED_PULSE_SHAPE, (
        f"u_{gate}_main.npy has shape {u.shape}, expected {EXPECTED_PULSE_SHAPE}; "
        f"this script scores 550-step main pulses at dt={DT}"
    )
    return u


def main():
    print("=" * 78)
    print("PHASE 1C GATE — tier4 state-propagation route vs. Phase-1b block route")
    print(f"n_t={N_T}, dt={DT} us, alpha={ALPHA:.6f}, n_c in {TRUNC_LIST}, tol={TOL:.0e}")
    print("=" * 78)
    print("F_legacy = frozen pre-Phase-1c state-propagation route (this file).")
    print("F_block  = canonical cross-subspace route (core/propagator.py).")
    print("F_live   = validate_logical_gates.tier4_*_relative_phase as it stands now.")
    print("-" * 78)
    print(f"{'gate':<6}{'n_c':>5}{'F_legacy':>12}{'F_block':>12}{'|dF|':>10}"
          f"{'||dU_log||':>12}{'live-vs-block':>15}{'L1':>11}")

    failures = []
    for gate, (tier4_fn, bases_fn) in CASES.items():
        u = _load(gate)
        for n_c in TRUNC_LIST:
            old = LEGACY[gate](u, n_c=n_c, n_t=N_T, dt=DT)

            # Bases are rebuilt per n_c: the cat states depend on the truncation.
            H0, Hc = make_hamiltonian(N_T, n_c)
            B_in, B_out, E_ideal = bases_fn(N_T, n_c, alpha=ALPHA)
            M = cross_subspace_block(full_propagator(u, H0, Hc, DT), B_in, B_out)
            new = pedersen_gate_fidelity(E_ideal, M)

            # The live function must equal the block route exactly -- it delegates
            # to it. Any nonzero here means the delegation grew arithmetic of its
            # own, which is precisely what Phase 1c removed.
            live = tier4_fn(u, n_c=n_c, n_t=N_T, dt=DT)
            d_live = max(abs(live["F_avg_gate"] - new),
                         np.linalg.norm(live["U_log"] - M))

            d_f = abs(old["F_avg_gate"] - new)
            d_u = np.linalg.norm(old["U_log"] - M)
            ok = d_f < TOL and d_u < TOL and d_live == 0.0
            if not ok:
                failures.append((gate, n_c, d_f, d_u, d_live))

            print(f"{gate:<6}{n_c:>5}{old['F_avg_gate']:>12.6f}{new:>12.6f}"
                  f"{d_f:>10.2e}{d_u:>12.2e}{d_live:>15.2e}{leakage_L1(M):>11.3e}"
                  + ("" if ok else "   <-- MISMATCH"))

    print("-" * 78)
    if failures:
        print("❌ FAIL — the routes disagree. Do NOT proceed with the refactor.")
        for gate, n_c, d_f, d_u, d_live in failures:
            print(f"   {gate} n_c={n_c}: |dF|={d_f:.3e}  ||dU_log||={d_u:.3e}  "
                  f"live-vs-block={d_live:.3e}")
        print("   Diagnose (a) stale '+d' average_gate_fidelity, (b) basis-helper")
        print("   mismatch vs logical_basis/transmon_basis, (c) n_c/dt mismatch.")
        print("   The block route is authoritative; fix the old path to match.")
        raise SystemExit(1)

    print("✅ PASS — legacy and block routes give the same 2x2 block and the same")
    print("   fidelity on both pulses at every n_c, and the live tier4 functions")
    print("   reproduce the block route exactly (0.00e+00 -- they delegate to it).")
    print("   The block route additionally yields L1, which the legacy route discarded.")


if __name__ == "__main__":
    main()
