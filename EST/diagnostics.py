"""
Error-transparency diagnostics: the Fig. 1d-f replication target.

    python EST/diagnostics.py                 # EsT vs Ord, all metrics + figure
    python EST/diagnostics.py --gate H        # same, for another trained gate
    python EST/diagnostics.py --suffix _it600 # an archived/alternate pulse pair

Three time-resolved quantities from lossless unitary simulation (paper Eqs. 6-8):

    Delta_QEC(t)   instantaneous Knill-Laflamme violation of the evolved code
    L_Ej(t)        instantaneous leakage of the evolved error state out of the
                   instantaneous error space
    eta_Ej,psi(t)  trajectory mismatch, as the distance between the code and
                   projected-error Bloch vectors

Propagation here uses core.grape_core.step_data's EIGENDECOMPOSITION propagator,
deliberately not the JAX expm path used for training. Re-scoring a trained pulse
through a different algorithm is an independent check on the training loop, in
the same spirit as the repo's existing QuTiP cross-check -- a pulse that scores
well under its own propagator and badly under another has a numerics problem, not
a physics result.

TRANSCRIBED DEFINITIONS -- absolute numbers ARE comparable to the paper
-----------------------------------------------------------------------
All three are transcriptions of the published equations, not reconstructions
from the transparency conditions:

  * `delta_qec`   -> Eq. 6, via the closed form derived in App. A (Eqs. A8-A11).
                    Unnormalized, as printed. Where the main text says "2-norm"
                    and App. A says Frobenius, App. A wins; Eq. 6 points there.
  * `leakage_Ej`  -> Eq. 7 verbatim. The six-cardinal average IS the paper's
                    maximally-mixed initial state.
  * `eta_mismatch`-> Eq. 8 verbatim: a Bloch-vector distance in [0, 2], on the
                    error state PROJECTED into the instantaneous error space.

So both the absolute values and the EsT:Ord ratios carry across to the paper.
This replaced an earlier set of reconstructions whose normalizations were this
module's own choice and which, per the paper comparison, did not.

ONE interpretive choice remains: the paper fixes sigma_Ej and P_Ej only up to a
proportionality constant, which is resolved here by the polar isometry of
a P_C(t). Sensitivity is 0.4% on the time-averaged eta; see
`_instantaneous_error_basis` for the argument and the pinning test.

NOT a paper equation: `c2_integrand` is the C2 training cost's integrand, kept so
the numpy and JAX code paths can be cross-checked. It is NOT Eq. 8 and must not
be reported as a transparency metric.

Reminder from the paper's App. A: exact error transparency is unreachable with
linear drives, because [K/2 a^dag^2 a^2 + chi'/2 a^dag^2 a^2 q^dag q, a] is
uncorrectable. These curves should approach zero and stop -- a floor is the
physics, not a convergence failure.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.grape_core import make_ops, step_data
from EST import kitten_code
from EST.device import DT, N_T, make_hamiltonian_est

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PULSE_DIR = os.path.join(REPO_ROOT, "pulses", "est")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "est")
TABLE_DIR = os.path.join(REPO_ROOT, "tables")


# ---------------------------------------------------------------------------
# Propagation (numpy / eigh -- independent of the JAX training path)
# ---------------------------------------------------------------------------

def propagate_states(u, H0, Hc, Psi0, dt):
    """
    Full state-vector trajectory, (N+1, n, M).

    The repo's four existing `simulate_trajectory` variants
    (visualization/pulse_viz.py:138, analysis/pulse_analysis.py:96, and two in
    analysis/compare_dt_light_vs_main*.py) all reduce to populations inside the
    loop and discard the amplitudes, so none of them can supply the overlaps
    these metrics need. This keeps the state vectors, and amortizes one
    eigendecomposition per step across all M columns the way
    grape_core._fidelity_core does.
    """
    Psi0 = np.asarray(Psi0, dtype=complex)
    traj = np.empty((len(u) + 1,) + Psi0.shape, dtype=complex)
    traj[0] = Psi0
    psi = Psi0
    for k, u_k in enumerate(u):
        Uk, _, _ = step_data(H0, Hc, u_k, dt)
        psi = Uk @ psi
        traj[k + 1] = psi
    return traj


# ---------------------------------------------------------------------------
# Eqs. 6-8, transcribed
# ---------------------------------------------------------------------------

# Logical Pauli matrices in the instantaneous code basis {|0_L(t)>, |1_L(t)>}.
_PAULI = (
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),      # X_C
    np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),       # Y_C
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),     # Z_C
)


def _require_code_words(code_words):
    """
    Guard the shape every error-space construction here depends on.

    `a` restricted to the code space has RANK 2, so handing the (T, n, 6)
    cardinal stack to a QR/SVD silently returns six orthonormal columns, four of
    which are an arbitrary completion OUTSIDE the error space. Projecting onto
    those absorbs real leakage: L_Ej(u_X_est) read 0.148 that way against 0.258
    on the correct basis. Always pass the two evolved code WORDS -- the cardinals
    span the same space but carry a shape that hides the rank.
    """
    code_words = np.asarray(code_words)
    if code_words.ndim != 3 or code_words.shape[2] != 2:
        raise ValueError(
            "expected the two evolved code words, shape (T, n, 2); got "
            f"{code_words.shape}. `a` on the code space is rank 2, so a wider "
            "stack yields spurious basis vectors outside the error space.")
    return code_words


def _instantaneous_error_basis(code_words, A):
    """
    Orthonormal basis of the instantaneous error space (Eq. 2),

        E(t) = span{a|0_L(t)>, a|1_L(t)>} ,

    as the polar isometry W(t) of a P_C(t), shape (T, n, 2). Its columns are the
    basis `leakage_Ej` projects onto, and W sigma_C W^dag is the paper's
    sigma_Ej(t) ∝ Ej sigma_C(t) Ej^dag that `eta_mismatch` needs.

    POLAR RATHER THAN QR, DELIBERATELY -- this is the one interpretive choice
    left in Eqs. 6-8. The paper fixes sigma_Ej and P_Ej only up to a
    proportionality constant. That is exact for this code at t = 0, where
    a|0_L> and a|1_L> both have norm sqrt(2) so a P_C is sqrt(2) times an
    isometry (the coincidence EST/kitten_code.error_cardinals asserts), and only
    approximate later: the singular values of a P_C(t) drift apart by up to 19%
    mid-gate. The polar factor is the canonical, column-order-independent nearest
    isometry, so it reduces to the literal reading wherever the coincidence
    holds. Measured on u_X_est, the time-averaged eta differs from the literal
    `a sigma_C a^dag / nbar` reading by 0.4% (0.16879 vs 0.16817), while a QR
    basis -- which depends on column order -- differs pointwise by up to 0.1.
    `DiagnosticsTest.test_eta_basis_convention_is_pinned` fails if this drifts.
    """
    _require_code_words(code_words)
    img = np.einsum('ij,tjm->tim', A, code_words)         # a P_C(t), (T, n, 2)
    U, _, Vh = np.linalg.svd(img, full_matrices=False)
    return U @ Vh


def delta_qec(code_words, A):
    """
    Eq. 6, via the closed form the paper derives in App. A (Eqs. A8-A11):

        M_ik(t)      = P_C(t) Ei^dag Ek P_C(t)                          (A8)
        M_ik         = c P_C + x X_C + y Y_C + z Z_C                    (A9)
        [c,x,y,z]_ik = 1/2 Tr{[P,X,Y,Z]_C(t) M_ik(t)}
        Delta_QEC    = sum_ik || M_ik - 1/2 Tr[M_ik] P_C ||
                     = sum_ik 2 (|x_ik|^2 + |y_ik|^2 + |z_ik|^2)       (A11)

    over the error set {I, a}, all four ordered pairs. The (I, I) pair
    contributes exactly zero: M_II = P_C is pure identity by orthonormality of
    the evolved code words, which unitary evolution preserves.

    NOT NORMALIZED, because the paper normalizes by nothing. An earlier version
    divided by the mean photon number to make the result dimensionless; that
    division is what made both the absolute values and the EsT:Ord ratios
    incomparable to the paper.

    One ambiguity, resolved toward App. A: main-text Eq. 6 calls ||.||_2 "the
    2-norm when applied to matrices" (i.e. spectral), while App. A says Frobenius
    and defines ||A||_2 = Tr(A^dag A) -- with no square root, which is what
    yields the factor-2 closed form above. Eq. 6 sends the reader to App. A for
    details, so App. A wins.

    Parameters
    ----------
    code_words : (T, n, 2) the two evolved CODE WORDS |0_L(t)>, |1_L(t)>
    A          : (n, n) cavity annihilation operator

    Returns (T,) real, >= 0.
    """
    code_words = _require_code_words(code_words)
    errors = (np.eye(A.shape[0], dtype=complex), A)

    out = np.zeros(code_words.shape[0])
    for Ei in errors:
        for Ek in errors:
            # M_ik in the instantaneous logical basis, (T, 2, 2).
            M = np.einsum('tim,ij,tjn->tmn',
                          np.conj(code_words), Ei.conj().T @ Ek, code_words)
            for P in _PAULI:
                coeff = 0.5 * np.einsum('ij,tji->t', P, M)
                out += 2.0 * np.abs(coeff) ** 2
    return out


def leakage_Ej(code_words, err_traj, A):
    """
    Eq. 7. Instantaneous leakage of the evolved error state out of the
    instantaneous error space:

        L_Ej(t) = 1 - Tr[P_Ej(t) rho(t)] = 1 - || P_E(t)|psi_Ej(t)> ||^2 .

    The paper initializes rho as the MAXIMALLY MIXED state on E_j(0). Averaging
    the six columns returned here is exactly that, since the six error cardinals
    average to I_E / 2; the per-column values are kept so a single trajectory can
    still be inspected.

    Takes the two evolved code WORDS, not the cardinal stack -- see
    `_require_code_words` for why the wider array silently understates this.

    Parameters
    ----------
    code_words : (T, n, 2) evolved code words, defining E(t) via Eq. 2
    err_traj   : (T, n, M) evolved error-space states
    A          : (n, n) cavity annihilation operator

    Returns (T, M) in [0, 1].
    """
    W = _instantaneous_error_basis(code_words, A)             # (T, n, 2)
    coeff = np.einsum('tik,tim->tkm', np.conj(W), err_traj)   # (T, 2, M)
    return 1.0 - np.sum(np.abs(coeff) ** 2, axis=1)


def eta_mismatch(code_words, code_traj, err_traj, A):
    """
    Eq. 8. Trajectory mismatch, as the Euclidean distance between the Bloch
    vector of the code state and that of the PROJECTED error state, each read in
    its own instantaneous subspace:

        eta_Ej,psi(t) = || r_C(t) - r_Ej(t) ||_2 ,   in [0, 2]

        r_C  = Tr[sigma_C(t) rho_psi(t)]
        r_Ej = Tr[sigma_Ej(t) rho~_psiEj(t)] ,  sigma_Ej(t) ∝ Ej sigma_C(t) Ej^dag

    with rho~ the evolved error state projected onto E_j(t) and renormalized, and
    sigma_Ej built from the polar isometry (see `_instantaneous_error_basis`).

    Two consequences of transcribing this faithfully, both load-bearing when
    comparing against numbers predating the transcription:

      * It is LEAKAGE-INSENSITIVE by construction. The projection divides out
        whatever left the error space -- which is precisely what L_Ej measures.
        So `L <= eta` is NO LONGER true by construction, as it was for the [0,1]
        state infidelity this replaced.
      * It quotients out phases a state-overlap comparison keeps. The T gate,
        whose error state picks up a different Fock-dependent phase than its code
        state, reads 0.055 under the old definition and ~1e-14 here.

    `c2_integrand` carries the old formula. That is the C2 training cost's
    integrand and a different quantity; do not substitute one for the other.

    Parameters
    ----------
    code_words : (T, n, 2) evolved code words, defining C(t) and E(t)
    code_traj  : (T, n, M) evolved code-space states
    err_traj   : (T, n, M) evolved error-space states, column-matched to code_traj
    A          : (n, n) cavity annihilation operator

    Returns (T, M) in [0, 2].
    """
    W = _instantaneous_error_basis(code_words, A)

    # Coefficients in the instantaneous code and error bases, (T, 2, M).
    cC = np.einsum('tik,tim->tkm', np.conj(code_words), code_traj)
    cE = np.einsum('tik,tim->tkm', np.conj(W), err_traj)
    cE = cE / np.maximum(np.linalg.norm(cE, axis=1, keepdims=True), 1e-30)

    def bloch(c):
        return np.stack([np.real(np.einsum('tkm,kl,tlm->tm', np.conj(c), P, c))
                         for P in _PAULI], axis=1)                  # (T, 3, M)

    return np.linalg.norm(bloch(cC) - bloch(cE), axis=1)


def c2_integrand(code_traj, err_traj, A):
    """
    The time-resolved integrand of the C2 TRAINING COST (grape_jax.et_cost), not
    a paper equation:

        1 - |<psi_Ej(t)| a |psi_C(t)>|^2 / || a|psi_C(t)> ||^2

    so mean(c2_integrand) == et_cost == 1 - F_ET. That identity is what lets the
    numpy analysis path and the JAX training path be cross-checked against each
    other, which is the only reason this function exists. It carries C2's
    N_norm^2 deviation from the paper's printed Eq. (C2) (see EST/grape_jax.py)
    deliberately -- the point is to reproduce what training actually optimized.

    This WAS `eta_mismatch`, before Eqs. 6-8 were transcribed. It is not Eq. 8:
    it compares states rather than Bloch vectors, does not project the error
    state onto E(t), and is bounded by 1 rather than 2.

    Returns (T, M) in [0, 1].
    """
    img = np.einsum('ij,tjm->tim', A, code_traj)
    nrm_sq = np.sum(np.abs(img) ** 2, axis=1)                # (T, M)
    ov = np.einsum('tim,tim->tm', np.conj(err_traj), img)
    return 1.0 - np.abs(ov) ** 2 / np.maximum(nrm_sq, 1e-30)


# ---------------------------------------------------------------------------
# Scoring and convergence
# ---------------------------------------------------------------------------

def rescore(u, gate, n_t=N_T, n_c=20, dt=DT):
    """
    Independent re-score of C1 through the eigh propagator. Should agree with
    the JAX training value to ~1e-8; a larger gap means the training loop and
    the physics disagree.
    """
    H0, Hc = make_hamiltonian_est(n_t, n_c)
    C0 = kitten_code.cardinals(n_t, n_c)
    T = kitten_code.gate_target(gate, n_t, n_c)
    final = propagate_states(u, H0, Hc, C0, dt)[-1]
    per_col = np.abs(np.sum(np.conj(T) * final, axis=0)) ** 2
    return float(per_col.mean()), per_col


def truncation_scan(u, gate, n_t=N_T, trunc_list=(16, 18, 20, 22, 24, 28), dt=DT):
    """
    Verification step 5: re-score at held-out truncations. A pulse exploiting the
    Hilbert-space wall shows a fidelity that drifts with n_c instead of
    plateauing -- the failure mode core/cat_code.py:272 guards against, and the
    reason training at a single truncation needs this check.
    """
    return {n_c: rescore(u, gate, n_t, n_c, dt)[0] for n_c in trunc_list}


def max_active_fock(u, n_t=N_T, n_c=20, dt=DT, thresh=0.01):
    """
    Highest Fock level reaching transient population > `thresh`, over all six
    code cardinal points. This is the paper's fair-comparison criterion for
    EsT vs Ord -- matched on MAX ACTIVE FOCK LEVEL, not on gate duration.
    """
    H0, Hc = make_hamiltonian_est(n_t, n_c)
    traj = propagate_states(u, H0, Hc, kitten_code.cardinals(n_t, n_c), dt)
    pop = (np.abs(traj) ** 2).reshape(len(traj), n_t, n_c, -1)
    per_n = pop.sum(axis=1).max(axis=(0, 2))          # (n_c,) peak over t and cardinals
    active = np.nonzero(per_n > thresh)[0]
    return int(active.max()) if active.size else 0, per_n


def analyze(u, gate="X", n_t=N_T, n_c=20, dt=DT):
    """Everything for one pulse, as a dict of arrays."""
    H0, Hc = make_hamiltonian_est(n_t, n_c)
    A, _ = make_ops(n_t, n_c)

    B = kitten_code.logical_basis(n_t, n_c)            # the two code WORDS
    code_words = propagate_states(u, H0, Hc, B, dt)    # (T, n, 2), for Delta_QEC

    C0 = kitten_code.cardinals(n_t, n_c)
    E0 = kitten_code.error_cardinals(n_t, n_c)
    both = propagate_states(u, H0, Hc, np.concatenate([C0, E0], axis=1), dt)
    code, err = both[:, :, :6], both[:, :, 6:]

    F1, per_col = rescore(u, gate, n_t, n_c, dt)
    nmax, per_n = max_active_fock(u, n_t, n_c, dt)
    eta = eta_mismatch(code_words, code, err, A)          # (T, 6), Eq. 8
    return {
        "t_us": np.arange(len(u) + 1) * dt,
        "delta_qec": delta_qec(code_words, A),
        "leakage": leakage_Ej(code_words, err, A).mean(axis=1),
        # Eq. 8 is state-dependent and the paper's Fig. 1f plots |0_L> alone.
        # CARDINAL_ORDER[0] is "+Z", i.e. |0_L>. The six-cardinal mean is kept
        # alongside it as the gate-wide summary.
        "eta_0L": eta[:, 0],
        "eta_avg": eta.mean(axis=1),
        # Not a transparency metric -- the C2 cross-check only. See c2_integrand.
        "c2_integrand": c2_integrand(code, err, A).mean(axis=1),
        "F1": F1, "F1_per_cardinal": per_col,
        "max_active_fock": nmax, "fock_peak": per_n,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_fig1(results, path, scale="log"):
    """
    Three-panel Fig. 1d-f: EsT vs Ord for each metric.

    `scale="log"` reproduces the paper's own axes and is the default; it is the
    only view that resolves the near-zero EsT floor App. A predicts. `"linear"`
    is the complementary view: for gates whose metrics sit at O(0.1) it shows the
    EsT-vs-Ord GAP at its true relative size, which a decade grid compresses.
    Each panel autoscales independently, so the narrow Delta_QEC channel stays
    legible next to the wide eta channel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Panel (f) carries TWO traces per variant: the paper's Fig. 1f plots the
    # single initial state |0_L>, while the six-cardinal mean is the gate-wide
    # summary Table 9 ranks on. Both are drawn so the panel is comparable to the
    # paper without hiding the average.
    panels = [("d", r"$\Delta_{\rm QEC}(t)$", [("delta_qec", None)]),
              ("e", r"$L_{E_j}(t)$", [("leakage", None)]),
              ("f", r"$\eta_{E_j,\psi}(t)$   (Eq. 8, $\in[0,2]$)",
               [("eta_0L", r"$|0_L\rangle$"), ("eta_avg", "6-cardinal mean")])]
    styles = {"est": dict(color="#2a6fb5", lw=1.8, label="EsT"),
              "ord": dict(color="#c4453c", lw=1.8, ls="--", label="Ord"),
              "paper": dict(color="#2e8b57", lw=1.8, ls="-.", label="EsT (paper form)")}
    log = scale == "log"

    FLOOR = 1e-12
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (tag, ylabel, keys) in zip(axes, panels):
        drawn = 0
        for variant, r in results.items():
            base = styles.get(variant, dict(label=variant))
            for i, (key, sub) in enumerate(keys):
                st = dict(base)
                if i:                      # the faint companion trace
                    st.update(alpha=0.35, lw=1.0)
                if sub is not None:
                    st["label"] = f"{base.get('label', variant)}, {sub}"
                # A series entirely below the floor is at NUMERICAL ZERO, and on a
                # log axis clamping it would draw a flat line at 1e-12 that reads
                # as a measured value. T hits this on Delta_QEC and eta(|0_L>).
                # Say so instead of drawing it.
                if log and np.max(r[key]) < FLOOR:
                    continue
                y = np.maximum(r[key], FLOOR) if log else r[key]
                ax.plot(r["t_us"], y, **st)
                drawn += 1
        if log:
            ax.set_yscale("log")
            if not drawn:
                ax.text(0.5, 0.5, "all variants at\nnumerical zero\n"
                                  rf"($< 10^{{{int(np.log10(FLOOR))}}}$)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="0.35")
                ax.set_yticks([])
                # Nothing was plotted, so matplotlib would fall back to its
                # default 0-1 x-range and this panel would silently disagree
                # with its neighbours about the gate duration.
                t = next(iter(results.values()))["t_us"]
                ax.set_xlim(t[0], t[-1])
        else:
            ax.set_ylim(bottom=0.0)
        ax.set_xlabel(r"$t\ (\mu s)$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({tag})", loc="left", fontweight="bold")
        ax.grid(alpha=0.25, which="both" if log else "major", lw=0.5)
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(gate="X", scale="both", suffix=""):
    """
    Score the est/ord pair for one gate and write the figure + table.

    `suffix` selects a non-default pulse pair, e.g. "_it600" for the archived
    600-iteration run or "_warm" for the warm restart. It is appended to the
    pulse name AND to every output name, so an alternate pair can never clobber
    the delivered one's artifacts.
    """
    import pandas as pd

    results, rows = {}, []
    for variant in ("est", "ord"):
        p = os.path.join(PULSE_DIR, f"u_{gate}_{variant}{suffix}.npy")
        if not os.path.exists(p):
            print(f"[skip] {p} not found -- run EST/train_est.py "
                  f"--gate {gate} --variant {variant}")
            continue
        u = np.load(p)
        r = analyze(u, gate=gate)
        results[variant] = r

        conv = truncation_scan(u, gate)
        spread = max(conv.values()) - min(conv.values())
        print(f"\n=== {gate} / {variant} ===")
        print(f"  C1 fidelity (eigh re-score, n_c=20) : {r['F1']:.6f}")
        print(f"  per-cardinal                        : "
              + " ".join(f"{v:.4f}" for v in r['F1_per_cardinal']))
        print(f"  max active Fock level (>1% pop)     : {r['max_active_fock']}")
        print(f"  truncation scan                     : "
              + "  ".join(f"n_c={k}:{v:.5f}" for k, v in conv.items()))
        print(f"  truncation spread                   : {spread:.2e}"
              + ("   OK" if spread < 1e-3 else "   <-- NOT converged, retrain --trunc 16 20 24"))
        print(f"  Eqs. 6-8, time-averaged             : "
              f"dQEC={r['delta_qec'].mean():.4e}  L={r['leakage'].mean():.4e}  "
              f"eta(|0_L>)={r['eta_0L'].mean():.4e}  eta(avg)={r['eta_avg'].mean():.4e}")

        rows.append({"gate": gate, "variant": variant, "F1": r["F1"],
                     "max_active_fock": r["max_active_fock"],
                     "trunc_spread": spread,
                     "delta_qec_mean": r["delta_qec"].mean(),
                     "leakage_mean": r["leakage"].mean(),
                     "eta_0L_mean": r["eta_0L"].mean(),
                     "eta_avg_mean": r["eta_avg"].mean(),
                     "c2_integrand_mean": r["c2_integrand"].mean()})

    if len(results) == 2:
        print("\n--- EsT vs Ord (time-averaged, lower is better) ---")
        for key in ("delta_qec", "leakage", "eta_0L", "eta_avg"):
            e, o = results["est"][key].mean(), results["ord"][key].mean()
            # Below this both variants are at numerical zero (T is the case that
            # hits it): the quotient is then a ratio of rounding error, so say so
            # rather than print a number a reader would rank on.
            if max(e, o) < 1e-12:
                print(f"  {key:10s}  EsT={e:.4e}  Ord={o:.4e}  "
                      "ratio=n/a (both at numerical zero)")
            else:
                print(f"  {key:10s}  EsT={e:.4e}  Ord={o:.4e}  "
                      f"ratio={o/max(e,1e-30):.2f}x")
        if results["est"]["max_active_fock"] != results["ord"]["max_active_fock"]:
            print("  NOTE: max active Fock levels differ "
                  f"({results['est']['max_active_fock']} vs "
                  f"{results['ord']['max_active_fock']}); the paper's fair "
                  "comparison matches these, so retune before quoting a ratio.")

    if results:
        # X keeps the unsuffixed names it has always written -- EST/README.md
        # references figures/est/fig1def_est_vs_ord.png by that path. Other gates
        # are suffixed so they cannot clobber it, and a --suffix run always
        # carries the gate name too (hence est_fig1_metrics_X_it600.csv).
        sfx = f"_{gate}{suffix}" if suffix else ("" if gate == "X" else f"_{gate}")
        print()
        for s in (("log", "linear") if scale == "both" else (scale,)):
            # The log figure keeps the unsuffixed name EST/README.md references.
            tail = "" if s == "log" else f"_{s}"
            path = plot_fig1(
                results,
                os.path.join(FIG_DIR, f"fig1def_est_vs_ord{sfx}{tail}.png"),
                scale=s)
            print(f"figure ({s:6s}) -> {path}")
        os.makedirs(TABLE_DIR, exist_ok=True)
        csv = os.path.join(TABLE_DIR, f"est_fig1_metrics{sfx}.csv")
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"table  -> {csv}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", default="X",
                    choices=sorted(kitten_code.IDEAL_LOGICAL_U),
                    help="which trained gate to score (default X)")
    ap.add_argument("--scale", default="both", choices=("log", "linear", "both"),
                    help="y-axis of the Fig. 1d-f panels (default both)")
    ap.add_argument("--suffix", default="",
                    help="score u_<gate>_<variant><SUFFIX>.npy instead of the "
                         "delivered pair, e.g. _it600; outputs are suffixed too")
    args = ap.parse_args()
    main(args.gate, args.scale, args.suffix)
