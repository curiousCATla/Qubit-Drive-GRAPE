#!/usr/bin/env python3
"""
resolve_populations.py — Phase 2 population bins for leakage / seepage.

Resolves a joint-space state |φ> (D = n_t * n_c) into physically meaningful
bins without inventing an ideal full-space unitary.

Partition bins (sum to ≈1):
  P_logical            population in the 2D cat code space C = span(B)
  P_g_even_nonlogical  t=0, n even (≡0 or 2 mod 4 union all even), minus
                       the logical population (cats are pure |g> ⊗ even)
  P_g_odd              t=0, n odd
  P_e_even             t≥1, n even
  P_e_odd              t≥1, n odd

Extra diagnostic (overlaps the partition; not part of the sum):
  P_trunc_edge         any t, cavity Fock n ≥ n_c - 2

Index convention matches make_ops: index(|t,c>) = t*n_c + c.
"""

from __future__ import annotations

import numpy as np


PARTITION_KEYS = (
    "P_logical",
    "P_g_even_nonlogical",
    "P_g_odd",
    "P_e_even",
    "P_e_odd",
)

# Bins used when locating the dominant destination of leaked amplitude
# (everything except the target / logical bin).
LEAK_DESTINATION_KEYS = (
    "P_g_even_nonlogical",
    "P_g_odd",
    "P_e_even",
    "P_e_odd",
)


def resolve_state(phi, B, n_t, n_c):
    """
    Resolve |φ> into population bins.

    Parameters
    ----------
    phi : (D,) complex
    B   : (D, 2) logical basis columns
    n_t, n_c : level counts

    Returns
    -------
    dict of float populations
    """
    phi = np.asarray(phi, dtype=complex).ravel()
    D = n_t * n_c
    assert phi.shape == (D,), f"phi shape {phi.shape}, expected ({D},)"
    blocks = phi.reshape(n_t, n_c)
    pop = np.abs(blocks) ** 2  # (n_t, n_c)

    n_idx = np.arange(n_c)
    even_mask = (n_idx % 2) == 0
    odd_mask = ~even_mask

    # Logical subspace population
    P_logical = float(np.linalg.norm(B.conj().T @ phi) ** 2)

    # Ground-transmon even / odd
    P_g_even = float(pop[0, even_mask].sum())
    P_g_odd = float(pop[0, odd_mask].sum())

    # Excited-transmon (e, f, ...) even / odd
    if n_t > 1:
        P_e_even = float(pop[1:, :][:, even_mask].sum())
        P_e_odd = float(pop[1:, :][:, odd_mask].sum())
    else:
        P_e_even = 0.0
        P_e_odd = 0.0

    # Cats live entirely in t=0 even Fock support, so the logical population
    # is a subset of P_g_even.  The non-logical even-g remainder is:
    P_g_even_nonlogical = max(P_g_even - P_logical, 0.0)

    # Truncation-edge diagnostic (any transmon, last two Fock levels)
    edge_start = max(n_c - 2, 0)
    P_trunc_edge = float(pop[:, edge_start:].sum())

    return {
        "P_logical": P_logical,
        "P_g_even_nonlogical": P_g_even_nonlogical,
        "P_g_odd": P_g_odd,
        "P_e_even": P_e_even,
        "P_e_odd": P_e_odd,
        "P_trunc_edge": P_trunc_edge,
        # helpers (not in the published partition, but useful)
        "P_g_even": P_g_even,
    }


def partition_sum(bins):
    """Sum of the five partition bins."""
    return sum(bins[k] for k in PARTITION_KEYS)


def resolve_after_pulse(U_full, ket, B, n_t, n_c):
    """Propagate ket through U_full and resolve the output state."""
    phi = U_full @ ket
    bins = resolve_state(phi, B, n_t, n_c)
    bins["P_logical_initial"] = float(np.linalg.norm(B.conj().T @ ket) ** 2)
    bins["norm_out"] = float(np.linalg.norm(phi) ** 2)
    return bins


def dominant_leak_destination(bins, exclude_logical=True):
    """
    Argmax among non-logical partition bins (destination of leaked amplitude).

    Returns (bin_key, population) or ("none", 0.0) if no mass outside logical.
    """
    keys = LEAK_DESTINATION_KEYS if exclude_logical else PARTITION_KEYS
    best_key = "none"
    best_val = 0.0
    for k in keys:
        v = bins.get(k, 0.0)
        if v > best_val:
            best_val = v
            best_key = k
    return best_key, best_val


def aggregate_leak_destination(per_input_bins, target_key="P_logical"):
    """
    Mass-weighted dominant destination of leaked amplitude across code inputs.

    For each input, leaked mass = 1 - bins[target_key] (clipped at 0).
    Destination weights accumulate bins[dest] (already absolute populations).

    Returns (dominant_key, weight_dict).
    """
    totals = {k: 0.0 for k in LEAK_DESTINATION_KEYS}
    for bins in per_input_bins:
        # Absolute populations in each non-target bin are the leaked pieces.
        for k in LEAK_DESTINATION_KEYS:
            totals[k] += bins.get(k, 0.0)
    if sum(totals.values()) <= 0:
        return "none", totals
    dom = max(totals, key=totals.get)
    return dom, totals


def P_computational(phi, n_t, n_c):
    """Population in span{|g,0>, |e,0>} — natural target subspace for U_dec."""
    blocks = np.asarray(phi, dtype=complex).reshape(n_t, n_c)
    p = abs(blocks[0, 0]) ** 2
    if n_t > 1:
        p += abs(blocks[1, 0]) ** 2
    return float(p)
