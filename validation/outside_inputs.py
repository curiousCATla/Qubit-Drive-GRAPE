#!/usr/bin/env python3
"""
outside_inputs.py — fixed outside-code-space probe set for Phase 2.

The code space is the 2D span of the logical cats (transmon ground):

    C = span{|+Z_L, g>, |-Z_L, g>}

Any state orthogonal to both columns of logical_basis is a legitimate
outside-space probe.  Every gate is tested against the SAME list.

Groups
------
A. Transmon-excited: |e,0>, |e,1>, |e,2>
B. Opposite parity (cavity n ≡ 1 or 3 mod 4, transmon g): |g,1>, |g,3>, |g,5>
C. Same-parity, orthogonal-to-code (preferred construction):
   Take the natural radial Fock probes |g,4>, |g,6>, |g,8> (n ≡ 0 mod 4,
   live in the even manifold with the cats).  Project out the logical
   subspace (I - B B†), renormalize, then Gram-Schmidt-orthonormalize the
   survivors.  Keep up to 3 vectors with residual weight above a floor.
   These check for unwanted radial / photon-number mixing inside the even
   manifold without the |g,2> trap (which is 81% inside the code space).

Fallback (only if fewer than 2 Gram survivors, e.g. tiny n_c): raw |g,4>,
|g,6>, |g,8> labelled group "C_raw" with non-zero P_logical_initial —
report residual after projection; not used at production n_c >= 22.

Usage
-----
    from validation.outside_inputs import outside_input_list, assert_outside_probes
"""

from __future__ import annotations

import numpy as np

from core.grape_core import basis_state


# Residual weight floor after projecting out the logical subspace.
# Below this, a radial Fock is almost entirely in the code space and is
# not a useful outside probe.
_C_RESIDUAL_FLOOR = 1e-6

# Maximum number of group-C Gram vectors to keep.
_C_MAX_KEEP = 3


def _project_out_logical(ket, B):
    """(I - B B†) |ket>, without materializing a D×D matrix."""
    coeffs = B.conj().T @ ket  # (2,)
    return ket - B @ coeffs


def _group_c_gram(n_t, n_c, B, alpha):
    """
    Same-parity orthogonal-to-code probes via residual Gram-Schmidt.

    Procedure (documented for reproducibility):
      1. Start from |g,4>, |g,6>, |g,8> (skip any n >= n_c).
      2. Project each out of C: r = (I - BB†)|g,n>.
      3. Drop vectors with ||r||² < _C_RESIDUAL_FLOOR.
      4. Sort survivors by residual weight (descending).
      5. Classical Gram-Schmidt among survivors (and already-accepted
         vectors), renormalize; keep up to _C_MAX_KEEP unit vectors.
      6. Assert each kept vector has ||B† ψ||² < 1e-12.
    """
    # Lazy import: logical_basis lives in propagator, which imports cat_code.
    # outside_inputs only needs B passed in for the main path; rebuild if needed.
    radial_ns = [4, 6, 8]
    candidates = []
    for n in radial_ns:
        if n >= n_c:
            continue
        ket = basis_state(n_t, n_c, 0, n)
        residual = _project_out_logical(ket, B)
        weight = float(np.vdot(residual, residual).real)
        if weight < _C_RESIDUAL_FLOOR:
            continue
        candidates.append((n, weight, residual / np.sqrt(weight)))

    # Prefer higher residual weight first (most "outside" radial direction).
    candidates.sort(key=lambda t: -t[1])

    accepted = []  # list of (name, ket)
    for n, weight, unit_res in candidates:
        # Gram-Schmidt against already-accepted group-C vectors
        v = unit_res.copy()
        for _, prev in accepted:
            v = v - np.vdot(prev, v) * prev
        # Also kill any numerical re-entry into the logical subspace
        v = _project_out_logical(v, B)
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            continue
        v = v / norm
        name = f"|g,{n}>_orth"
        accepted.append((name, v))
        if len(accepted) >= _C_MAX_KEEP:
            break

    return accepted


def _group_c_raw_fallback(n_t, n_c):
    """Raw |g,4/6/8> with non-zero cat overlap — last resort only."""
    out = []
    for n in (4, 6, 8):
        if n >= n_c:
            continue
        out.append((f"|g,{n}>", basis_state(n_t, n_c, 0, n)))
    return out


def outside_input_list(n_t=3, n_c=24, alpha=np.sqrt(3.0), B=None):
    """
    Fixed outside-space probe set for Phase 2.

    Parameters
    ----------
    n_t, n_c, alpha : Hilbert-space / cat parameters.
    B : optional (D, 2) logical basis.  Built via logical_basis if omitted.

    Returns
    -------
    list of (name, group, ket)
        group in {"A", "B", "C", "C_raw"}.
        Every gate must use this same list at a given (n_t, n_c, alpha).
    """
    if B is None:
        from core.propagator import logical_basis
        B = logical_basis(n_t, n_c, alpha=alpha)

    probes = []

    # --- Group A: transmon excited -----------------------------------------
    if n_t >= 2:
        for n in (0, 1, 2):
            if n >= n_c:
                continue
            probes.append((f"|e,{n}>", "A", basis_state(n_t, n_c, 1, n)))

    # --- Group B: opposite parity, ground transmon -------------------------
    for n in (1, 3, 5):
        if n >= n_c:
            continue
        probes.append((f"|g,{n}>", "B", basis_state(n_t, n_c, 0, n)))

    # --- Group C: same-parity orthogonal complement ------------------------
    gram = _group_c_gram(n_t, n_c, B, alpha)
    if len(gram) >= 2:
        for name, ket in gram:
            probes.append((name, "C", ket))
    else:
        # Fallback: raw radial Fock (document non-zero cat overlap).
        for name, ket in _group_c_raw_fallback(n_t, n_c):
            probes.append((name, "C_raw", ket))

    return probes


def assert_outside_probes(inputs, B, tol=1e-12, require_true_outside_groups=("A", "B", "C")):
    """
    Sanity checks for the probe set.

    - Every ket is unit-norm.
    - Groups listed in require_true_outside_groups have ||B†ψ||² < tol.
    - C_raw (if present) is allowed a larger logical overlap; only unit-norm
      is enforced for those.
    """
    for name, group, ket in inputs:
        nrm = np.linalg.norm(ket)
        assert abs(nrm - 1.0) < 1e-10, f"{name} not unit-norm: ||ψ||={nrm}"
        p_log = float(np.linalg.norm(B.conj().T @ ket) ** 2)
        if group in require_true_outside_groups:
            assert p_log < tol, (
                f"{name} (group {group}) is not outside the code space: "
                f"P_logical={p_log:.3e} (tol={tol})"
            )
    return True


def code_input_list_logical(n_t=3, n_c=24, alpha=np.sqrt(3.0), B=None):
    """
    The two logical cat basis states (code-space inputs) for leakage-destination
    analysis on I/X/Y/Z/H/T (and as "already encoded" probes).
    """
    if B is None:
        from core.propagator import logical_basis
        B = logical_basis(n_t, n_c, alpha=alpha)
    return [
        ("|+Z_L,g>", "code", B[:, 0].copy()),
        ("|-Z_L,g>", "code", B[:, 1].copy()),
    ]


def code_input_list_computational(n_t=3, n_c=24):
    """Computational basis {|g,0>, |e,0>} — natural inputs for U_enc."""
    return [
        ("|g,0>", "code_comp", basis_state(n_t, n_c, 0, 0)),
        ("|e,0>", "code_comp", basis_state(n_t, n_c, 1, 0)),
    ]
