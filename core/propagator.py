#!/usr/bin/env python3
"""
propagator.py — full D×D propagator and honest logical-subspace metrics.

Everything else in this repo propagates *state vectors* (`psi = Uk @ psi`, e.g.
`validation.validate_logical_gates.propagate_pulse` and
`core.grape_core._fidelity_core`). That is all the training objective ever needs:
the objective only asks where the two logical basis states end up.

Measuring a *gate* honestly needs more than that. This module builds the full
D×D propagator U for a saved pulse, so that:

  - the effective 2×2 logical map is extracted by an explicit projection
    B† U B, with one documented index convention (see `logical_block`), rather
    than by hand-assembling four overlaps;
  - encode/decode, whose input and output subspaces DIFFER, are extracted by the
    same projection with two different bases (see `cross_subspace_block`), so
    they are scored by the same Pedersen formula as the logical gates rather
    than by a separate ad-hoc metric;
  - leakage enters the average gate fidelity correctly, via Tr(M M†) --- the
    Pedersen (2007) form. See `pedersen_gate_fidelity`;
  - we can ask what the pulse does to states *outside* the logical subspace,
    which a two-state propagation simply cannot see. See `seepage_report`.

This module is MEASUREMENT ONLY. Nothing here is part of any training
objective, and `leakage_L1` is a reported diagnostic, not a penalty.

Import direction: this module depends on `core.grape_core` and `core.cat_code`
and deliberately does NOT import from `validation/`. Ideal target unitaries are
always passed in as arguments (never looked up here), so that
`validation/validate_logical_gates.py` --- which owns `IDEAL_LOGICAL_U` --- can
import this module without creating a cycle.
"""

import numpy as np

from core.grape_core import step_data, basis_state, make_ops
from core.cat_code import get_logical_cat_states, embed_in_joint_space


# The six Pauli eigenstates, as coefficient vectors in the {|+Z_L>, |-Z_L>}
# logical basis. These six points on the Bloch sphere (the octahedron) form a
# spherical 2-design in d=2, which is why the mean state-transfer fidelity over
# them reproduces the Haar average exactly --- see `pedersen_gate_fidelity`.
PAULI_EIGENSTATE_COEFFS = {
    "+Z": np.array([1.0, 0.0], dtype=complex),
    "-Z": np.array([0.0, 1.0], dtype=complex),
    "+X": np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0),
    "-X": np.array([1.0, -1.0], dtype=complex) / np.sqrt(2.0),
    "+Y": np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2.0),
    "-Y": np.array([1.0, -1.0j], dtype=complex) / np.sqrt(2.0),
}


def full_propagator(u, H0, Hc, dt):
    """
    Build the full D×D unitary propagator for the entire control sequence u.

    Parameters
    ----------
    u : (N, 4) array
        Control amplitudes per timestep, ordered
        (cavity I, cavity Q, transmon I, transmon Q) --- matching the column
        order of `Hc` from `core.grape_core.make_hamiltonian`.
    H0, Hc : drift Hamiltonian and list of 4 control Hamiltonians.
    dt : float
        Timestep. REQUIRED, no default: the dt-refined `pulses/*10x_light.npy`
        waveforms are 5500×4 at dt=2e-4, and silently reusing the 2e-3 of the
        550-step pulses would produce a plausible-looking wrong answer.

    Returns
    -------
    U : (D, D) complex ndarray, D = H0.shape[0].

    Convention: LEFT-multiply, U <- Uk @ U, so later timesteps end up on the
    left and U = U_{N-1} ... U_1 U_0. This is forced by the rest of the repo:
    both `propagate_pulse` and `_fidelity_core` iterate the rows of u in order
    doing `psi = Uk @ psi`, so U @ psi_0 must reproduce their final state.
    Reversing the order still yields a perfectly unitary matrix representing the
    wrong gate, which is why `validation/test_phase0.py` checks this product
    against `propagate_pulse` directly rather than merely checking unitarity.

    Cost is not a concern: 550 steps × eigh(72×72) is well under a second.
    """
    D = H0.shape[0]
    U = np.eye(D, dtype=complex)
    for u_k in u:
        Uk, _, _ = step_data(H0, Hc, u_k, dt)
        U = Uk @ U
    return U


def logical_basis(n_t, n_c, alpha=np.sqrt(3.0)):
    """
    Orthonormal basis of the 2D logical subspace, embedded in the joint space.

    Returns
    -------
    B : (D, 2) complex ndarray, D = n_t * n_c.
        Column 0 is |+Z_L> ⊗ |g>, column 1 is |-Z_L> ⊗ |g> (transmon ground,
        t_level=0), matching the states every `get_logical_*_state_pairs`
        factory in `core.cat_code` trains against.

    `n_t` is REQUIRED and has no default on purpose. Every saved pulse was
    trained at n_t=3, and `make_hamiltonian` silently drops the transmon
    anharmonicity term (alpha/2)·b†b†bb when n_t < 3 --- evaluating an n_t=3
    pulse at n_t=2 is different physics, not a cheaper approximation. (Note
    that `embed_in_joint_space` itself defaults to n_t=2, so n_t is always
    passed through explicitly below.)

    B is orthonormal to machine precision at every truncation, exactly --- not
    approximately: |+Z_L> lives on Fock n ≡ 0 (mod 4) and |-Z_L> on n ≡ 2
    (mod 4), so their supports are disjoint and <+Z_L|-Z_L> = 0 identically,
    while `get_logical_cat_states` normalizes each column independently. The
    assertion below is therefore a cheap guard against a future change to the
    cat-state construction, not a tolerance we are relying on.
    """
    psi_p_cav, psi_m_cav = get_logical_cat_states(alpha=alpha, n_c=n_c)
    psi_pg = embed_in_joint_space(psi_p_cav, n_t=n_t, n_c=n_c, t_level=0)
    psi_mg = embed_in_joint_space(psi_m_cav, n_t=n_t, n_c=n_c, t_level=0)
    B = np.stack([psi_pg, psi_mg], axis=1)

    orth_err = np.linalg.norm(B.conj().T @ B - np.eye(2))
    assert orth_err < 1e-12, (
        f"logical basis is not orthonormal at n_c={n_c} "
        f"(||B^dag B - I|| = {orth_err:.3e}); the projection B^dag U B and the "
        "state-fidelity interpretation of pedersen_gate_fidelity both assume it is"
    )
    return B


def error_basis(n_t, n_c, alpha=np.sqrt(3.0), B=None):
    """
    Orthonormal basis of the single-photon-loss error subspace, and its norms.

    The even cat code has one physically distinguished subspace outside itself:
    the image of the code space under one cavity photon loss,

        E = span{ a|+Z_L>, a|-Z_L> }.

    A loss is supposed to be *detectable* (parity flips) and *correctable* (the
    logical content survives inside E). This function makes E a first-class
    probe subspace so both claims can be measured on a saved pulse, instead of
    being sampled indirectly by the odd-Fock probes in group B of
    `validation.outside_inputs.outside_input_list` --- those live in the odd
    manifold, but they are not the states a real loss actually produces.

    Returns
    -------
    E : (D, 2) complex ndarray, D = n_t * n_c.
        Column j is a|j_L> renormalized, column order matched to
        `logical_basis` (column 0 from |+Z_L>, column 1 from |-Z_L>).
    nrm : (2,) float ndarray
        The pre-normalization norms, ||a|j_L>||. Returned rather than discarded
        because they carry the physics below and `knill_laflamme_residual` needs
        their squares; recomputing them invites the two callers to disagree.

    `n_t` is REQUIRED for the same reason as in `logical_basis`.

    Both structural facts below are EXACT, not tolerances we are leaning on ---
    the assertions are guards against a future change to the cat-state
    construction, in the same spirit as `logical_basis`:

      - E is orthonormal. |+Z_L> lives on Fock n = 0 (mod 4) and |-Z_L> on
        n = 2 (mod 4), so a|+Z_L> lives on n = 3 (mod 4) and a|-Z_L> on
        n = 1 (mod 4): disjoint supports, <e_0|e_1> = 0 identically.
      - E is orthogonal to the code space, B† E = 0 identically, because the
        cats are purely even and their loss images purely odd.

    UNEQUAL NORMS --- read before reusing this for anything else. At
    alpha=sqrt(3), ||a|+Z_L>|| = 1.8067 and ||a|-Z_L>|| = 1.6602, a ratio of
    1.088. So `a` restricted to the code space is NOT proportional to an
    isometry: it distorts superpositions. The concrete way to get this wrong is
    to assume a|+X_L> renormalized is the error-space |+X> --- it is not, it is
    tilted toward column 0. That is why this function returns a *basis* only and
    there is deliberately no `error_cardinals` analogue here: for this code the
    six error-space cardinal points are ambiguous and a caller must say which
    construction it means. `EST/kitten_code.py:error_cardinals` is the mirror
    image --- for the binomial kitten code the two norms are equal (both
    sqrt(2)), so the construction is unambiguous there, and it carries a live
    assertion pinning exactly that coincidence.

    The same inequality is what makes this code only APPROXIMATELY correctable
    for a single loss; see `knill_laflamme_residual`.
    """
    if B is None:
        B = logical_basis(n_t, n_c, alpha=alpha)
    A, _ = make_ops(n_t, n_c)

    raw = A @ B
    nrm = np.linalg.norm(raw, axis=0)
    assert np.all(nrm > 1e-9), (
        f"a|j_L> has vanishing norm at n_c={n_c} (norms {nrm}); the error "
        "subspace is not well defined"
    )
    E = raw / nrm

    orth_err = np.linalg.norm(E.conj().T @ E - np.eye(2))
    assert orth_err < 1e-12, (
        f"error basis is not orthonormal at n_c={n_c} "
        f"(||E^dag E - I|| = {orth_err:.3e}); the projection E^dag U E and "
        "leakage_L1 both assume it is"
    )
    cross = np.linalg.norm(B.conj().T @ E)
    assert cross < 1e-12, (
        f"error subspace is not orthogonal to the code space at n_c={n_c} "
        f"(||B^dag E|| = {cross:.3e}); a loss would then be partly invisible to "
        "the parity check, and L2_(E->C) below would not measure seepage"
    )
    return E, nrm


def knill_laflamme_residual(B, n_t, n_c):
    """
    How exactly correctable is a single photon loss? Knill-Laflamme for E={I,a}.

    A code corrects the error set {I, a} exactly iff

        <i_L| a       |j_L> = 0                (I-vs-a cross term)
        <i_L| a^dag a |j_L> = c * delta_ij     (a-vs-a term)

    for a single constant c independent of i, j. The first condition and the
    off-diagonal of the second are satisfied *identically* by the even cat code
    (parity, and disjoint mod-4 supports). The DIAGONAL of the second is not:
    the two cats carry different mean photon numbers, so there is no single c.

    Returns
    -------
    dict with
        n_bar_plus, n_bar_minus : <j_L| a^dag a |j_L>, i.e. ||a|j_L>||^2
        n_offdiag               : |<+Z_L| a^dag a |-Z_L>|      (should be ~0)
        a_block_norm            : ||B^dag a B||                (should be ~0)
        rel_mismatch            : (n_bar_plus - n_bar_minus) / mean, the
                                  fractional violation of the c*delta_ij
                                  condition --- 0 for an exactly correctable
                                  code, 0.169 for the alpha=sqrt(3) cat code.

    Interpretation: the alpha=sqrt(3) four-component cat is an APPROXIMATE
    single-loss code, not an exact one. `EST/kitten_code.py`'s binomial kitten
    code is the exact version --- a|0_L> = sqrt(2)|3> and a|1_L> = sqrt(2)|1>
    have equal weight, so rel_mismatch is 0 there by construction.

    Measurement only, and deliberately independent of `EST/`:
    `EST.diagnostics.delta_qec` is the time-resolved analogue on the EsT track
    (it folds these same residuals, plus their values mid-pulse, into one
    number). Importing it here would invert the track separation set up in
    CLAUDE.md, so the two are cross-referenced in prose rather than shared.
    """
    A, _ = make_ops(n_t, n_c)
    n_op = A.conj().T @ A
    n_block = B.conj().T @ n_op @ B

    n_bar_plus = float(n_block[0, 0].real)
    n_bar_minus = float(n_block[1, 1].real)
    mean = 0.5 * (n_bar_plus + n_bar_minus)
    return {
        "n_bar_plus": n_bar_plus,
        "n_bar_minus": n_bar_minus,
        "n_offdiag": float(abs(n_block[0, 1])),
        "a_block_norm": float(np.linalg.norm(B.conj().T @ A @ B)),
        "rel_mismatch": float((n_bar_plus - n_bar_minus) / mean),
    }


def logical_block(U_full, B):
    """
    Project the full propagator onto the logical subspace: the effective 2×2 map.

    Returns B† U B, i.e. element [i, j] = <b_i| U |b_j>: **column per input**.
    Column j is where input basis state j goes, expressed in the logical basis.

    This is the convention that `IDEAL_LOGICAL_U` in
    `validation/validate_logical_gates.py` is written in. Check it against the
    trained targets: `core.cat_code.get_logical_Y_state_pairs` maps
    |+Z_L> -> -i|-Z_L> and |-Z_L> -> +i|+Z_L>, so the columns are (0, -i) and
    (+i, 0), giving [[0, i], [-i, 0]] --- exactly IDEAL_LOGICAL_U["Y"].

    The result is generally sub-unitary: any population the pulse pushed out of
    the logical subspace is simply absent from the block. That deficit is what
    `leakage_L1` reads off and what `pedersen_gate_fidelity` correctly charges
    for.
    """
    return B.conj().T @ U_full @ B


def cross_subspace_block(U_full, B_in, B_out):
    """
    Effective 2×2 map between two DIFFERENT subspaces: M[k, j] = <out_k| U |in_j>.

    `logical_block` is the special case B_in is B_out. Encode and decode are not
    that case: encode maps the transmon subspace {|g,0>, |e,0>} to the cat
    subspace {|+Z_L>, |-Z_L>}, and decode maps it back. The block is an
    OFF-DIAGONAL 2×2 block of U_full, with the input basis on the right and the
    output basis on the left.

    Same column-per-input convention as `logical_block`: column j is where input
    basis state j went, expressed in the OUTPUT basis.

    Why this is not a 4×4 block over {|g,0>, |e,0>, |+Z_L>, |-Z_L>}: that set is
    not orthonormal --- <g,0|+Z_L> ≈ 0.469 at alpha=sqrt(3), since vacuum overlaps
    the even cat --- so no 4×4 "ideal" is even well defined. Two inputs, each
    resolved against both output basis states, already determine the 2×2 map
    completely, relative phase included.

    B_in and B_out must each have orthonormal columns (checked by
    `logical_basis` / `transmon_basis`); they need NOT be orthogonal to each
    other, and here they are not.
    """
    return B_out.conj().T @ U_full @ B_in


def transmon_basis(n_t, n_c):
    """
    Orthonormal basis of the transmon computational subspace, cavity in vacuum.

    Returns
    -------
    B : (D, 2) complex ndarray, D = n_t * n_c.
        Column 0 is |g,0>, column 1 is |e,0>, matching the input order of
        `core.cat_code.get_encode_state_pairs` and the target order of
        `get_decode_state_pairs`.

    Exactly orthonormal by construction: two distinct computational basis
    vectors of the joint space (index t*n_c + c, see `make_ops`). `n_t` is
    required for the same reason as in `logical_basis` --- the index stride is
    n_c, and n_t fixes the dimension the saved pulse was trained at.
    """
    B = np.stack([basis_state(n_t, n_c, 0, 0), basis_state(n_t, n_c, 1, 0)], axis=1)
    orth_err = np.linalg.norm(B.conj().T @ B - np.eye(2))
    assert orth_err < 1e-12, (
        f"transmon basis is not orthonormal (||B^dag B - I|| = {orth_err:.3e})"
    )
    return B


def encode_bases(n_t, n_c, alpha=np.sqrt(3.0)):
    """
    (B_in, B_out, E_ideal) for scoring an encode pulse: transmon -> cat.

    B_in  columns: |g,0>, |e,0>        (transmon computational subspace)
    B_out columns: |+Z_L>, |-Z_L>      (cat code subspace)
    E_ideal = I, i.e. |g,0> -> |+Z_L> and |e,0> -> |-Z_L>.

    E_ideal is the identity ONLY because the column order above matches the pair
    order of `core.cat_code.get_encode_state_pairs`, which returns
    [(|g,0>, |g,C+>), (|e,0>, |g,C->)]. That coincidence is load-bearing and is
    asserted in `validation/test_phase1b.py` (test b) rather than trusted here:
    swapping either order silently turns this into a logical-X-composed-with-
    encode ideal, which would report ~0 for a perfect pulse.

    Bases must be rebuilt per truncation --- the cat states depend on n_c.
    """
    B_in = transmon_basis(n_t, n_c)
    B_out = logical_basis(n_t, n_c, alpha=alpha)
    E_ideal = np.eye(2, dtype=complex)
    return B_in, B_out, E_ideal


def decode_bases(n_t, n_c, alpha=np.sqrt(3.0)):
    """
    (B_in, B_out, E_ideal) for scoring a decode pulse: cat -> transmon.

    Exactly `encode_bases` with the two bases swapped:
    B_in  columns: |+Z_L>, |-Z_L>
    B_out columns: |g,0>, |e,0>
    E_ideal = I, i.e. |+Z_L> -> |g,0> and |-Z_L> -> |e,0>, matching the pair
    order of `core.cat_code.get_decode_state_pairs`. Same caveat as
    `encode_bases`: the identity ideal is a statement about column order.
    """
    B_in = logical_basis(n_t, n_c, alpha=alpha)
    B_out = transmon_basis(n_t, n_c)
    E_ideal = np.eye(2, dtype=complex)
    return B_in, B_out, E_ideal


def pedersen_gate_fidelity(U_ideal_2x2, U_log_2x2):
    """
    Average gate fidelity of a leaky (sub-unitary) 2×2 map, Pedersen (2007):

        F = (|Tr(M)|² + Tr(M M†)) / (d(d+1)),   M = U_ideal† U_log,  d = 2

    Note Tr(M M†) = Tr(U_log† U_log) since U_ideal is unitary, which is how the
    formula is evaluated below.

    Why the second term matters: the form with a bare `+ d` in place of
    Tr(M M†) is only valid when U_log is unitary, i.e. when there is no
    leakage. Using it on a leaky block silently over-reports. For d=2 the
    discrepancy is exactly `leakage_L1(U_log) / 3`.

    Interpretation: this equals the average, over Haar-random pure logical input
    states, of the state-transfer fidelity |<psi_target|U|psi_in>|². Because
    `PAULI_EIGENSTATE_COEFFS` is a spherical 2-design and that fidelity is
    quadratic in the input state, the plain mean over those six states hits the
    same number exactly (verified in `validation/test_phase0.py`).

    Applies unchanged to a `cross_subspace_block` (encode/decode): nothing in the
    derivation requires the input and output bases to be the same subspace, only
    that each is orthonormal so that Tr(M M†) is a population. The average is
    then over Haar-random inputs in the INPUT subspace.
    """
    d = U_ideal_2x2.shape[0]
    overlap = np.trace(U_ideal_2x2.conj().T @ U_log_2x2)
    leak_term = np.trace(U_log_2x2.conj().T @ U_log_2x2).real
    return (np.abs(overlap) ** 2 + leak_term) / (d * (d + 1))


def leakage_L1(U_log_2x2):
    """
    Average leakage out of the logical subspace:

        L1 = 1 - Tr(U_log† U_log) / d,   d = 2

    i.e. the mean population the pulse removes from the logical subspace,
    averaged over the logical basis states. Zero for a perfectly code-preserving
    pulse; target-independent (it does not reference any ideal gate), so it
    separates "left the code space" from "wrong rotation within the code space".

    Reported as a diagnostic only. Phase 0 deliberately does not add this to any
    training objective.

    On a `cross_subspace_block` this reads as "population that failed to arrive
    in the OUTPUT subspace", averaged over the input basis states --- for encode,
    the fraction of {|g,0>, |e,0>} that did not land in the cat code.
    """
    d = U_log_2x2.shape[0]
    return 1.0 - np.trace(U_log_2x2.conj().T @ U_log_2x2).real / d


def seepage_report(U_full, n_t, n_c, B, inputs):
    """
    Where does the pulse send states that start OUTSIDE the logical subspace?

    This is the question a two-state propagation structurally cannot answer, and
    the reason a full propagator is worth building.

    Parameters
    ----------
    U_full : (D, D) propagator from `full_propagator`.
    n_t, n_c : level counts, with index(|t,c>) = t*n_c + c (see `make_ops`).
    B : (D, 2) logical basis from `logical_basis`.
    inputs : list of (name, ket) for outside-logical states, e.g.
        [("|e,0>", basis_state(n_t, n_c, 1, 0)),
         ("|g,1>", basis_state(n_t, n_c, 0, 1))]

        Choose these carefully: "outside the logical subspace" means orthogonal
        to BOTH cat states, which is a stronger condition than "not one of the
        two logical basis states". |e,0> qualifies (different transmon level) and
        any odd cavity Fock state qualifies (the even cat code has no odd
        support). |g,2> does NOT: |-Z_L> is supported on n = 2, 6, 10, ... with
        its largest coefficient at n=2, so |<g,2|-Z_L>|^2 = 0.814 at alpha=sqrt(3).
        Feeding it in and reading pop_in_logical ~ 0.81 as seepage would be
        badly wrong -- that population started there. `pop_in_logical_initial`
        below exists so this is impossible to miss.

    Returns
    -------
    list of dicts, one per input:
        name                   : the label passed in
        pop_by_transmon_level  : list of n_t populations, summed over cavity Fock
        pop_in_logical         : population in the logical subspace after the pulse
        pop_in_logical_initial : population already in the logical subspace before
                                 the pulse -- must be ~0 for the input to be a
                                 meaningful seepage probe (see above)

    NOT an independent error metric --- do not add it to L1. For a unitary
    U_full the aggregate seepage L2 and leakage L1 are rigidly linked,
    d·L1 = (D-d)·L2, because whatever leaves the code space is exactly what
    enters it. These per-state numbers are a "where did it go" diagnostic:
    they localize the leak (which transmon level, whether it returns), which the
    single aggregate number cannot.

    Basic version only. Resolving the cavity population by mod-4 / parity
    subspace is Phase 2.
    """
    reports = []
    for name, ket in inputs:
        out = U_full @ ket
        pop_by_transmon_level = (np.abs(out.reshape(n_t, n_c)) ** 2).sum(axis=1)
        # ||B^dag out||^2 == out^dag (B B^dag) out, without materializing a D×D projector
        pop_in_logical = np.linalg.norm(B.conj().T @ out) ** 2
        reports.append({
            "name": name,
            "pop_by_transmon_level": pop_by_transmon_level.tolist(),
            "pop_in_logical": float(pop_in_logical),
            "pop_in_logical_initial": float(np.linalg.norm(B.conj().T @ ket) ** 2),
        })
    return reports
