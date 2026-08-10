"""
Fig. 1a: the code space and the error space evolving side by side under U(t).

    python EST/subspace_evolution.py                  # H gate, EsT vs Ord
    python EST/subspace_evolution.py --gate X         # another trained gate

Fig. 1a of Roy, Wetherbee & Fatemi (arXiv:2603.15356) is the paper's central
claim in picture form: an error-transparent gate drives the error space along
the SAME trajectory it drives the code space, at every instant of the gate
rather than only at t = T. This module draws that as the mean cavity photon
number of each subspace, per Bloch cardinal, EsT against Ord.

WHAT IS PLOTTED
---------------
Three curves per cardinal j, from one lossless unitary simulation:

    <n>_C(t)     mean photon number of the evolved CODE cardinal   |psi_Cj(t)>
    <n>_E(t)     mean photon number of the evolved ERROR cardinal  |psi_Ej(t)>
    <n>_loss(t)  mean photon number of the photon-loss image of the code state,
                 a|psi_Cj(t)> / || a|psi_Cj(t)> ||

The third is the prediction the second has to match. Error transparency (Eqs.
3-5) says the evolved error state stays equal to the loss image of the evolved
code state, so <n>_E and <n>_loss coincide exactly for a transparent gate and
peel apart for one that is not. <n>_C is the reference the whole thing is
driven around, and its excursion is what sets the max-active-Fock fair-
comparison criterion `diagnostics.max_active_fock` reports.

At t = 0 the values are fixed by the code, not by the pulse: every code cardinal
has <n> = 2 exactly (both code words carry <n> = 2 and <0_L|n|1_L> = 0), while
the error cardinals split 3 / 1 / 2 / 2 / 2 / 2 over +Z / -Z / +X / -X / +Y /
-Y, since |0_E> = |3> and |1_E> = |1>. Both facts are pinned in
test_grape_jax.SubspaceEvolutionTest.

WHAT <n> CANNOT SHOW -- read before drawing a conclusion from agreement
-----------------------------------------------------------------------
The drift Hamiltonian is a function of the cavity and transmon number operators
alone, so [H0, n] = 0 and <n> is EXACTLY conserved under drift. Every feature in
these curves is drive-induced. That also means <n> is blind to App. A's
obstruction: free Kerr evolution drives eta (Eq. 8) away from zero while leaving
all three photon-number curves flat, because it rearranges phases within a fixed
photon distribution. Two states can share <n> to machine precision and still be
orthogonal.

So <n>_E == <n>_loss is NECESSARY for transparency and not sufficient. The
sufficient statement is the state-overlap one, which is `diagnostics.eta_mismatch`
(Eq. 8), and the map-level one, which is `map_mismatch` below. This module plots
all three so the photon-number picture is never read on its own.

The two subspaces are directly comparable only because of a property of this
specific code: a|0_L> = sqrt(2)|3> and a|1_L> = sqrt(2)|1> carry EQUAL weight, so
`a` restricted to the code space is sqrt(2) times an isometry onto the error
space and does not distort superpositions. That coincidence is asserted live in
kitten_code.error_cardinals -- read its docstring before reusing any of this for
a different code.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.grape_core import make_ops
from EST import kitten_code
from EST.device import DT, N_T, make_hamiltonian_est
from EST.diagnostics import FIG_DIR, PULSE_DIR, TABLE_DIR, propagate_states

# Row colours match EST/diagnostics.py:253 so the two figures read as one set.
VARIANT_STYLE = {"est": dict(color="#2a6fb5", label="EsT"),
                 "ord": dict(color="#c4453c", label="Ord")}

# Within one panel the curves are the two SUBSPACES, not two cardinals: the code
# space is always black, the error space takes the variant colour above, and the
# photon-loss prediction is grey.
CODE_COLOR = "#1f1f1f"
LOSS_COLOR = "#7a7f85"

# Three cardinals fix a 2x2 map up to a global phase, and a panel each keeps the
# figure to one screen. Pass --cardinals to widen to all six.
DEFAULT_CARDINALS = ["+Z", "+X", "+Y"]


# ---------------------------------------------------------------------------
# Trajectories, photon numbers, and the two 2x2 blocks
# ---------------------------------------------------------------------------

def subspace_trajectories(u, H0, Hc, B, E, dt):
    """
    Joint-space trajectories of the code and error BASIS columns, each (T, n, 2).

    Propagating the two bases rather than the twelve cardinal states is not an
    optimization detail, it is the whole reason this is cheap: evolution is
    linear, so every cardinal trajectory is a fixed coefficient combination of
    these four columns. The four are propagated in a single `propagate_states`
    call, which amortizes one eigendecomposition per time step across them.
    """
    traj = propagate_states(u, H0, Hc, np.concatenate([B, E], axis=1), dt)
    return traj[:, :, :2], traj[:, :, 2:]


def cardinal_states(traj, coeffs=None):
    """(T, n, M) cardinal trajectories from a (T, n, 2) basis trajectory."""
    if coeffs is None:
        coeffs = kitten_code._coeff_matrix()
    return np.einsum('tik,km->tim', traj, np.asarray(coeffs))


def photon_numbers(states, n_op):
    """
    <psi(t)|n|psi(t)> for each column, (T, M).

    No renormalization: unitary evolution preserves the norm of every column, so
    these arrive already normalized. `loss_image_photon_numbers` is the one that
    has to divide, because `a` is not unitary.
    """
    return np.einsum('tim,tim->tm', np.conj(states),
                     np.einsum('ij,tjm->tim', n_op, states)).real


def loss_image_photon_numbers(code_states, A, n_op):
    """
    <n> of the normalized photon-loss image a|psi_C(t)> / || a|psi_C(t)> ||.

    This is what the evolved error state must equal, instant by instant, for the
    gate to be error transparent -- the same object `diagnostics.eta_mismatch`
    takes the overlap against, read here through a physical observable instead.
    """
    img = np.einsum('ij,tjm->tim', A, code_states)
    nrm = np.sum(np.abs(img) ** 2, axis=1)
    return photon_numbers(img, n_op) / np.maximum(nrm, 1e-30)


def subspace_blocks(traj_code, traj_err, B, E):
    """U_L(t) = B^dag U(t) B and U_E(t) = E^dag U(t) E, each (T, 2, 2)."""
    return (np.einsum('ik,tim->tkm', np.conj(B), traj_code),
            np.einsum('ik,tim->tkm', np.conj(E), traj_err))


def subspace_weight(U2, coeffs=None):
    """
    Surviving norm-squared of each cardinal inside its own fixed span, (T, M).

    1 - this is population outside the SPAN, which is not all leakage in the
    usual sense: |0_L> = (|0> + |4>)/sqrt(2) has its two Fock components dephase
    against each other under the Kerr terms and rotate toward the orthogonal
    combination (|0> - |4>)/sqrt(2), which costs weight here with nothing having
    escaped the {0, 2, 4} sector. The error span {|3>, |1>} carries no internal
    superposition and so has no such channel -- under drift alone it is exactly
    invariant.
    """
    if coeffs is None:
        coeffs = kitten_code._coeff_matrix()
    c = np.einsum('tkj,jm->tkm', np.asarray(U2), np.asarray(coeffs))
    return np.sum(np.abs(c) ** 2, axis=1)


def map_mismatch(UL, UE):
    """
    Phase-insensitive disagreement between the two blocks, (T,) in [0, 1]:

        1 - |Tr(U_L^dag U_E)|^2 / (Tr(U_L^dag U_L) Tr(U_E^dag U_E))

    Zero exactly when U_E(t) = e^{i phi} U_L(t), i.e. when the gate acts
    identically on both subspaces. Unlike the photon-number curves this is
    sensitive to the phase structure the drift stirs up, and unlike
    diagnostics.eta_mismatch it compares the MAPS rather than one propagated
    state pair, so it is insensitive to which cardinal is probed and is
    normalized by both blocks' norms, meaning leakage common to the two
    subspaces does not register as disagreement.

    Note this is a different normalization from Eqs. 6-8; it is a companion to
    the figure, not one of the paper's printed metrics, so its EsT:Ord ratio is
    meaningful and its absolute value is not.
    """
    ov = np.abs(np.einsum('tij,tij->t', np.conj(UL), UE)) ** 2
    nl = np.einsum('tij,tij->t', np.conj(UL), UL).real
    ne = np.einsum('tij,tij->t', np.conj(UE), UE).real
    return 1.0 - ov / np.maximum(nl * ne, 1e-30)


def analyze_subspace(u, gate="H", n_t=N_T, n_c=20, dt=DT):
    """Everything for one pulse, as a dict of arrays. Mirrors diagnostics.analyze."""
    H0, Hc = make_hamiltonian_est(n_t, n_c)
    A, _ = make_ops(n_t, n_c)
    n_op = A.conj().T @ A
    B = kitten_code.logical_basis(n_t, n_c)
    E = kitten_code.error_basis(n_t, n_c)

    traj_code, traj_err = subspace_trajectories(u, H0, Hc, B, E, dt)
    code, err = cardinal_states(traj_code), cardinal_states(traj_err)
    UL, UE = subspace_blocks(traj_code, traj_err, B, E)

    nbar_err = photon_numbers(err, n_op)
    nbar_loss = loss_image_photon_numbers(code, A, n_op)
    return {"t_us": np.arange(len(u) + 1) * dt,
            "UL": UL, "UE": UE,
            "nbar_code": photon_numbers(code, n_op),
            "nbar_err": nbar_err,
            "nbar_loss": nbar_loss,
            "nbar_track_err": np.abs(nbar_err - nbar_loss),
            "r2_code": subspace_weight(UL), "r2_err": subspace_weight(UE),
            "mismatch": map_mismatch(UL, UE)}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_fig1a(results, path, gate="H", cardinals=None):
    """
    Fig. 1a: rows = pulse variant, one <n>(t) panel per traced cardinal, plus a
    surviving-norm / map-mismatch panel that backs them up.

    Each panel carries BOTH subspaces for one cardinal -- code in black, error in
    the variant colour, and the photon-loss prediction the error curve must match
    in grey dashes. That pairing is the comparison the figure exists to make;
    a code panel beside a separate error panel would not support it.

    `results` maps variant name -> `analyze_subspace` output, the same contract
    as diagnostics.plot_fig1, and a single-variant dict is fine (one row).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cardinals = list(cardinals or DEFAULT_CARDINALS)
    cols = [j for j, k in enumerate(kitten_code.CARDINAL_ORDER) if k in cardinals]
    variants = [v for v in ("est", "ord") if v in results] or list(results)

    hi = max(max(r[k][:, cols].max() for k in ("nbar_code", "nbar_err", "nbar_loss"))
             for r in results.values())
    ylim = (-0.15, hi * 1.12)

    nrow, ncard = len(variants), len(cols)
    fig = plt.figure(figsize=(2.95 * ncard + 4.2, 3.35 * nrow))
    gs = fig.add_gridspec(nrow, ncard + 1,
                          width_ratios=[1.0] * ncard + [1.25],
                          wspace=0.22, hspace=0.22,
                          left=0.065, right=0.985, top=0.885, bottom=0.165)

    for i, variant in enumerate(variants):
        r = results[variant]
        style = VARIANT_STYLE.get(variant, dict(color="#444444", label=variant))
        last = i == nrow - 1

        for j, m in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            ax.plot(r["t_us"], r["nbar_loss"][:, m], color=LOSS_COLOR, lw=1.2,
                    ls="--", zorder=1)
            ax.plot(r["t_us"], r["nbar_code"][:, m], color=CODE_COLOR, lw=1.5,
                    zorder=2)
            ax.plot(r["t_us"], r["nbar_err"][:, m], color=style["color"], lw=1.5,
                    alpha=0.9, zorder=3)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_title(f"{kitten_code.CARDINAL_ORDER[m]}$_L$", fontsize=10.5)
            if j == 0:
                ax.set_ylabel(r"$\langle n \rangle$")
                ax.text(-0.30, 0.5, style["label"], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=13,
                        fontweight="bold", color=style["color"])
            else:
                ax.set_yticklabels([])
            if last:
                ax.set_xlabel(r"$t\ (\mu s)$")
            else:
                ax.set_xticklabels([])

        ax = fig.add_subplot(gs[i, ncard])
        ax.plot(r["t_us"], r["mismatch"], color=style["color"], lw=1.6,
                label=r"$\mathcal{M}(t)$")
        # Averaged over ALL six cardinals, not just the traced ones, so these
        # match the CSV and the stdout summary rather than the panels beside them.
        ax.plot(r["t_us"], r["r2_code"].mean(axis=1), color=CODE_COLOR,
                lw=1.0, alpha=0.75, label=r"$r^2$ code")
        ax.plot(r["t_us"], r["r2_err"].mean(axis=1), color=CODE_COLOR,
                lw=1.0, alpha=0.75, ls="--", label=r"$r^2$ error")
        ax.set_ylim(-0.02, 1.32)
        ax.set_ylabel("fraction")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_title(r"$\overline{\mathcal{M}}$ = "
                     f"{r['mismatch'].mean():.3f}    "
                     r"$\overline{|\Delta \langle n \rangle|}$ = "
                     f"{r['nbar_track_err'].mean():.3f}",
                     fontsize=9.5, loc="right", color=style["color"],
                     fontweight="bold")
        ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3,
                  handlelength=1.6, columnspacing=1.1)
        if last:
            ax.set_xlabel(r"$t\ (\mu s)$")
        else:
            ax.set_xticklabels([])

    handles = [Line2D([], [], color=CODE_COLOR, lw=2.0,
                      label=r"code  $\langle n\rangle_C$"),
               Line2D([], [], color=VARIANT_STYLE["est"]["color"], lw=2.0,
                      label=r"error  $\langle n\rangle_E$  (EsT)"),
               Line2D([], [], color=VARIANT_STYLE["ord"]["color"], lw=2.0,
                      label=r"error  $\langle n\rangle_E$  (Ord)"),
               Line2D([], [], color=LOSS_COLOR, lw=2.0, ls="--",
                      label=r"loss image  $a|\psi_C\rangle$  (target for $\langle n\rangle_E$)")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"{gate} gate: mean photon number of the code and error "
                 "subspaces under $U(t)$", fontsize=11.5)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close(fig)
    return path


def main(gate="H", variant="both", cardinals=None, n_c=20):
    import pandas as pd

    wanted = ("est", "ord") if variant == "both" else (variant,)
    results, rows = {}, []
    for v in wanted:
        p = os.path.join(PULSE_DIR, f"u_{gate}_{v}.npy")
        if not os.path.exists(p):
            print(f"[skip] {p} not found -- run EST/train_est.py "
                  f"--gate {gate} --variant {v}")
            continue
        r = analyze_subspace(np.load(p), gate=gate, n_c=n_c)
        results[v] = r

        print(f"\n=== {gate} / {v} ===")
        print(f"  <n> code   mean / peak      : "
              f"{r['nbar_code'].mean():.3f} / {r['nbar_code'].max():.3f}")
        print(f"  <n> error  mean / peak      : "
              f"{r['nbar_err'].mean():.3f} / {r['nbar_err'].max():.3f}")
        print(f"  |<n>_E - <n>_loss|  mean/max: "
              f"{r['nbar_track_err'].mean():.4f} / {r['nbar_track_err'].max():.4f}")
        print(f"  map mismatch  mean / at t=T : "
              f"{r['mismatch'].mean():.4e} / {r['mismatch'][-1]:.4e}")
        print(f"  min surviving norm code/err : "
              f"{r['r2_code'].min():.4f} / {r['r2_err'].min():.4f}")
        rows.append({"gate": gate, "variant": v,
                     "nbar_code_mean": r["nbar_code"].mean(),
                     "nbar_code_peak": r["nbar_code"].max(),
                     "nbar_err_mean": r["nbar_err"].mean(),
                     "nbar_err_peak": r["nbar_err"].max(),
                     "nbar_track_err_mean": r["nbar_track_err"].mean(),
                     "nbar_track_err_max": r["nbar_track_err"].max(),
                     "mismatch_mean": r["mismatch"].mean(),
                     "mismatch_final": r["mismatch"][-1],
                     "r2_code_min": r["r2_code"].min(),
                     "r2_err_min": r["r2_err"].min()})

    if len(results) == 2:
        print("\n--- EsT vs Ord (lower is better) ---")
        for key, label in (("nbar_track_err", "|<n>_E - <n>_loss|"),
                           ("mismatch", "map mismatch    ")):
            e = results["est"][key].mean()
            o = results["ord"][key].mean()
            print(f"  {label}  EsT={e:.4e}  Ord={o:.4e}  "
                  f"ratio={o / max(e, 1e-30):.2f}x")

    if results:
        sfx = "" if gate == "X" else f"_{gate}"
        path = plot_fig1a(results,
                          os.path.join(FIG_DIR, f"fig1a_subspace{sfx}.png"),
                          gate=gate, cardinals=cardinals)
        print(f"\nfigure -> {path}")
        os.makedirs(TABLE_DIR, exist_ok=True)
        csv = os.path.join(TABLE_DIR, f"est_fig1a_subspace{sfx}.csv")
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"table  -> {csv}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", default="H",
                    choices=sorted(kitten_code.IDEAL_LOGICAL_U),
                    help="which trained gate to draw (default H)")
    ap.add_argument("--variant", default="both", choices=("est", "ord", "both"))
    ap.add_argument("--cardinals", default=",".join(DEFAULT_CARDINALS),
                    help="comma-separated subset of "
                         + ",".join(kitten_code.CARDINAL_ORDER)
                         + f" (default {','.join(DEFAULT_CARDINALS)})")
    ap.add_argument("--nc", type=int, default=20, help="cavity truncation")
    a = ap.parse_args()
    main(a.gate, a.variant, [c.strip() for c in a.cardinals.split(",") if c.strip()],
         a.nc)
