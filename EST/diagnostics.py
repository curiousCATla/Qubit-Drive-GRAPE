"""
Error-transparency diagnostics: the Fig. 1d-f replication target.

    python EST/diagnostics.py                 # EsT vs Ord, all metrics + figure
    python EST/diagnostics.py --gate H        # same, for another trained gate

Three time-resolved quantities from lossless unitary simulation (paper Eqs. 6-8):

    Delta_QEC(t)   instantaneous Knill-Laflamme violation of the evolved code
    L_Ej(t)        instantaneous leakage of the evolved error state out of the
                   instantaneous error space
    eta_Ej,psi(t)  trajectory mismatch between the evolved error state and the
                   photon-loss image of the evolved code state

Propagation here uses core.grape_core.step_data's EIGENDECOMPOSITION propagator,
deliberately not the JAX expm path used for training. Re-scoring a trained pulse
through a different algorithm is an independent check on the training loop, in
the same spirit as the repo's existing QuTiP cross-check -- a pulse that scores
well under its own propagator and badly under another has a numerics problem, not
a physics result.

RECONSTRUCTED DEFINITIONS -- READ BEFORE QUOTING ABSOLUTE NUMBERS
-----------------------------------------------------------------
Eqs. 6-8 are reconstructed here from the error-transparency conditions (Eqs. 3-5)
rather than transcribed, so the normalizations are this module's choice:

  * `delta_qec` is the RMS residual of the Knill-Laflamme conditions for the
    error set {I, a} on the instantaneous code space, normalized by mean photon
    number to make it dimensionless.
  * `leakage_Ej` and `eta_mismatch` are bounded in [0, 1] by construction, so
    they have no free normalization.

The EsT-vs-Ord SEPARATION -- which is the replication target -- is insensitive to
these choices, since both variants are scored identically. The absolute axis
values are not. Check Eqs. 6-7 against the paper before publishing numbers.

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
# Eqs. 6-8
# ---------------------------------------------------------------------------

def delta_qec(code_traj, A):
    """
    Eq. 6, reconstructed. Knill-Laflamme violation for the error set {I, a} on
    the instantaneous code space spanned by the two evolved code words.

    The independent KL residuals for {I, a} are, writing n = a^dag a:

        off-diagonal   <psi_0|a|psi_1>,  <psi_0|n|psi_1>
        diagonal       <psi_0|a|psi_0> - <psi_1|a|psi_1>
                       <psi_0|n|psi_0> - <psi_1|n|psi_1>

    (The E=I conditions are satisfied identically by orthonormality of the
    evolved words, since the evolution is unitary.) Returns the RMS of these
    four residuals, normalized by the mean photon number so the result is
    dimensionless and comparable across pulses of different photon budgets.

    Parameters
    ----------
    code_traj : (T, n, 2) the two evolved CODE WORDS |0_L(t)>, |1_L(t)>
    A         : (n, n) cavity annihilation operator
    """
    n_op = A.conj().T @ A
    p0, p1 = code_traj[:, :, 0], code_traj[:, :, 1]

    def exp(op, x, y):
        return np.einsum('ti,ti->t', np.conj(x), np.einsum('ij,tj->ti', op, y))

    a01, n01 = exp(A, p0, p1), exp(n_op, p0, p1)
    da = exp(A, p0, p0) - exp(A, p1, p1)
    dn = exp(n_op, p0, p0) - exp(n_op, p1, p1)

    nbar = 0.5 * np.real(exp(n_op, p0, p0) + exp(n_op, p1, p1))
    resid = (np.abs(a01) ** 2 + np.abs(n01) ** 2
             + 0.25 * np.abs(da) ** 2 + 0.25 * np.abs(dn) ** 2)
    return np.sqrt(resid / 4.0) / np.maximum(nbar, 1e-12)


def _instantaneous_error_basis(code_traj, A):
    """
    Orthonormal basis of the instantaneous error space span{a|psi_0(t)>,
    a|psi_1(t)>}, as (T, n, 2), via per-step QR.

    This is the space Eq. 4 requires the evolved error space to coincide with.
    """
    img = np.einsum('ij,tjm->tim', A, code_traj)          # (T, n, 2)
    Q, _ = np.linalg.qr(img)                              # batched QR
    return Q


def leakage_Ej(code_traj, err_traj, A):
    """
    Eq. 7, reconstructed. Instantaneous leakage of each evolved error cardinal
    point out of the instantaneous error space:

        L_Ej(t) = 1 - || P_E(t) |psi_Ej(t)> ||^2

    Returns (T, M) in [0, 1]. Zero means the evolved error state still lies
    entirely inside the space reachable by losing a photon from the code -- the
    condition that makes the error detectable and correctable mid-gate.
    """
    Q = _instantaneous_error_basis(code_traj, A)          # (T, n, 2)
    coeff = np.einsum('tik,tim->tkm', np.conj(Q), err_traj)   # (T, 2, M)
    return 1.0 - np.sum(np.abs(coeff) ** 2, axis=1)


def eta_mismatch(code_traj, err_traj, A):
    """
    Eq. 8, reconstructed. Trajectory mismatch between the evolved error state
    and the normalized photon-loss image of the evolved code state:

        eta(t) = 1 - |<psi_Ej(t)| a|psi_C(t)>/|| a|psi_C(t)> || |^2

    Returns (T, M) in [0, 1]. This is exactly the time-resolved integrand of the
    C2 training cost (EST/grape_jax.et_cost), so for the EsT pulse it is the
    quantity that was directly optimized, and for Ord it is a pure diagnostic.
    """
    img = np.einsum('ij,tjm->tim', A, code_traj)
    nrm = np.sqrt(np.sum(np.abs(img) ** 2, axis=1))       # (T, M)
    ov = np.einsum('tim,tim->tm', np.conj(err_traj), img)
    return 1.0 - np.abs(ov) ** 2 / np.maximum(nrm ** 2, 1e-30)


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
    return {
        "t_us": np.arange(len(u) + 1) * dt,
        "delta_qec": delta_qec(code_words, A),
        "leakage": leakage_Ej(code, err, A).mean(axis=1),
        "eta": eta_mismatch(code, err, A).mean(axis=1),
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

    panels = [("delta_qec", r"$\Delta_{\rm QEC}(t)$", "d"),
              ("leakage", r"$L_{E_j}(t)$", "e"),
              ("eta", r"$\eta_{E_j,\psi}(t)$", "f")]
    styles = {"est": dict(color="#2a6fb5", lw=1.8, label="EsT"),
              "ord": dict(color="#c4453c", lw=1.8, ls="--", label="Ord")}
    log = scale == "log"

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, ylabel, tag) in zip(axes, panels):
        for variant, r in results.items():
            # The 1e-12 floor exists only so a zero cannot drop off a log axis.
            y = np.maximum(r[key], 1e-12) if log else r[key]
            ax.plot(r["t_us"], y, **styles.get(variant, dict(label=variant)))
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0.0)
        ax.set_xlabel(r"$t\ (\mu s)$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({tag})", loc="left", fontweight="bold")
        ax.grid(alpha=0.25, which="both" if log else "major", lw=0.5)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(gate="X", scale="both"):
    import pandas as pd

    results, rows = {}, []
    for variant in ("est", "ord"):
        p = os.path.join(PULSE_DIR, f"u_{gate}_{variant}.npy")
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
        print(f"  time-averaged Delta_QEC / L / eta    : "
              f"{r['delta_qec'].mean():.4e}  {r['leakage'].mean():.4e}  {r['eta'].mean():.4e}")

        rows.append({"gate": gate, "variant": variant, "F1": r["F1"],
                     "max_active_fock": r["max_active_fock"],
                     "trunc_spread": spread,
                     "delta_qec_mean": r["delta_qec"].mean(),
                     "leakage_mean": r["leakage"].mean(),
                     "eta_mean": r["eta"].mean()})

    if len(results) == 2:
        print("\n--- EsT vs Ord (time-averaged, lower is better) ---")
        for key in ("delta_qec", "leakage", "eta"):
            e, o = results["est"][key].mean(), results["ord"][key].mean()
            print(f"  {key:10s}  EsT={e:.4e}  Ord={o:.4e}  ratio={o/max(e,1e-30):.2f}x")
        if results["est"]["max_active_fock"] != results["ord"]["max_active_fock"]:
            print("  NOTE: max active Fock levels differ "
                  f"({results['est']['max_active_fock']} vs "
                  f"{results['ord']['max_active_fock']}); the paper's fair "
                  "comparison matches these, so retune before quoting a ratio.")

    if results:
        # X keeps the unsuffixed names it has always written -- EST/README.md
        # references figures/est/fig1def_est_vs_ord.png by that path. Other gates
        # are suffixed so they cannot clobber it.
        sfx = "" if gate == "X" else f"_{gate}"
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
    args = ap.parse_args()
    main(args.gate, args.scale)
