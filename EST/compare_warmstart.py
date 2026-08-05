"""
Warm-restart comparison: the cold-trained EsT pulse against the two stages of a
warm-restarted rerun, with Ord as the reference.

    python EST/compare_warmstart.py

Scores every pulse twice, through both code paths, because the two disagreeing
would mean a numerics problem rather than a physics result:

  * EST/diagnostics.analyze -- eigh propagator, Eqs. 6-8 (Delta_QEC, L, eta)
  * EST/grape_jax           -- expm propagator, the four training cost terms

mean(eta) and c2 are the same quantity computed through those two independent
paths (diagnostics.eta_mismatch is the time-resolved integrand of grape_jax.
et_cost), so their agreement is the cross-check, not a redundancy.

Reads whatever of the four pulses exist and skips the rest, so it is usable
mid-campaign. Writes tables/est_warmstart_comparison.csv and
figures/est/warmstart_est_vs_stages.png.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from EST import diagnostics as dg
from EST.device import DT
from EST.grape_jax import build_gate_objective
from EST.train_est import deramp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PULSE_DIR = os.path.join(REPO_ROOT, "pulses", "est")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "est")
TABLE_DIR = os.path.join(REPO_ROOT, "tables")

# (key, filename, description). Order sets the row order of the output table.
PULSES = [
    ("orig",   "u_X_est.npy",             "cold start, 600 it/stage"),
    ("stage1", "u_X_est_warm_stage1.npy", "warm restart, after stage 1"),
    ("stage2", "u_X_est_warm_stage2.npy", "warm restart, after stage 2"),
    ("ord",    "u_X_ord.npy",             "Ord control"),
]

STYLE = {
    "orig":   dict(color="#888888", lw=1.4, ls="-",  label="EsT cold (600 it)"),
    "stage1": dict(color="#e08a1e", lw=1.4, ls="-.", label="warm stage 1"),
    "stage2": dict(color="#2a6fb5", lw=1.8, ls="-",  label="warm stage 2"),
    "ord":    dict(color="#c4453c", lw=1.2, ls="--", label="Ord"),
}


def jax_terms(u, gate="X", trunc=20):
    """
    The four training cost terms for a SAVED (physical) pulse.

    build_gate_objective's report takes the raw pre-image, not the pulse, so the
    constraint chain has to be inverted first. `deramp` does that exactly and
    asserts the roundtrip, so this re-applies the same chain the pulse was
    trained under rather than an approximation of it.
    """
    _, report, _ = build_gate_objective(gate, len(u), dt=DT, trunc_list=[trunc])
    x, _ = deramp(u, DT)
    return report(x.ravel())[trunc]


def main():
    import pandas as pd

    rows, results = {}, {}
    for key, fname, desc in PULSES:
        path = os.path.join(PULSE_DIR, fname)
        if not os.path.exists(path):
            print(f"[skip] {fname} not found")
            continue

        u = np.load(path)
        r = dg.analyze(u, gate="X")
        t = jax_terms(u)
        conv = dg.truncation_scan(u, "X")
        spread = max(conv.values()) - min(conv.values())
        results[key] = r

        magC = np.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2)
        magT = np.sqrt(u[:, 2] ** 2 + u[:, 3] ** 2)
        rows[key] = {
            "pulse": key, "description": desc,
            "F1_eigh": r["F1"], "infid": 1.0 - r["F1"], "c1_jax": t["c1"],
            "F_ET": 1.0 - t["c2"], "eta_mean_eq8": r["eta"].mean(),
            "c3_velvar": t["c3"], "c4_amp": t["c4"],
            "delta_qec_mean": r["delta_qec"].mean(),
            "leakage_mean_eq7": r["leakage"].mean(),
            "max_active_fock": r["max_active_fock"], "trunc_spread": spread,
            "max_eps_cav": magC.max(), "max_eps_tra": magT.max(),
        }

        print(f"{key:8s} F1={r['F1']:.6f}  F_ET={1-t['c2']:.5f}  c3={t['c3']:.5f}  "
              f"dQEC={r['delta_qec'].mean():.4e}  L={r['leakage'].mean():.4e}  "
              f"eta={r['eta'].mean():.5f}  nmax={r['max_active_fock']}  "
              f"spread={spread:.2e}")
        # cross-check: same quantity, two propagators
        gap = abs(r["eta"].mean() - t["c2"])
        print(f"         |mean(eta) - c2| = {gap:.2e}"
              + ("" if gap < 1e-9 else "   <-- code paths disagree"))

    if not rows:
        print("nothing to compare")
        return

    os.makedirs(TABLE_DIR, exist_ok=True)
    csv = os.path.join(TABLE_DIR, "est_warmstart_comparison.csv")
    pd.DataFrame([rows[k] for k, _, _ in PULSES if k in rows]).to_csv(csv, index=False)
    print(f"\ntable  -> {csv}")

    if "orig" in rows:
        u0 = np.load(os.path.join(PULSE_DIR, "u_X_est.npy"))
        for key, fname, _ in PULSES[1:3]:
            p = os.path.join(PULSE_DIR, fname)
            if os.path.exists(p):
                d = np.linalg.norm(np.load(p) - u0) / np.linalg.norm(u0)
                print(f"  ||{key} - orig|| / ||orig|| = {d:.4f}")

    print(f"figure -> {plot(results)}")


def plot(results):
    """Fig. 1d-f overlay, same panels as diagnostics.plot_fig1 with the stages added."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("delta_qec", r"$\Delta_{\rm QEC}(t)$", "d"),
              ("leakage",   r"$L_{E_j}(t)$",          "e"),
              ("eta",       r"$\eta_{E_j,\psi}(t)$",  "f")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, ylabel, tag) in zip(axes, panels):
        for name, r in results.items():
            ax.plot(r["t_us"], np.maximum(r[key], 1e-12), **STYLE[name])
        ax.set_yscale("log")
        ax.set_xlabel(r"$t\ (\mu s)$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({tag})", loc="left", fontweight="bold")
        ax.grid(alpha=0.25, which="both", lw=0.5)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "warmstart_est_vs_stages.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
