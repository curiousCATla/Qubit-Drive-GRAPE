#!/usr/bin/env python3
"""
compare_dt_light_vs_main.py

Compare Stage-1 coarse pulses (*_main.npy, dt=0.002 us) against Stage-2
light dt-refined pulses (*10x_light.npy, dt=0.0002 us, s=10):

  - fidelity vs cavity truncation n_c
  - photon-number trajectory <n>(t) and P(n,t)
  - peak amplitude / trajectory metrics
  - wall-clock summary from light-refine logs

Outputs:
  tables/dt_refine_fidelity_comparison.csv
  tables/dt_refine_trajectory_metrics.csv
  tables/dt_refine_cost_summary.csv
  figures/dt_refine_fidelity_vs_nc.pdf/.png
  figures/dt_refine_photon_trajectory_opt.pdf/.png
  figures/dt_refine_photon_trajectory_enc.pdf/.png
  figures/dt_refine_fock_heatmap_opt.pdf/.png
  figures/dt_refine_fock_heatmap_enc.pdf/.png
"""

from __future__ import annotations

import os
import sys
import re
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.grape_core import (
    make_hamiltonian,
    fidelity_multi_state,
    basis_state,
    step_data,
)
from core.cat_code import get_encode_state_pairs

FIG_DIR = os.path.join(REPO_ROOT, "figures")
TAB_DIR = os.path.join(REPO_ROOT, "tables")
LOG_DIR = os.path.join(REPO_ROOT, "logs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

N_T = 3
DT_MAIN = 0.002          # us
DT_LIGHT = 0.0002        # us
TRUNC_LIST = list(range(18, 31, 2))
TRAJ_N_C = 24


def get_g6_state_pairs(n_c, n_t=N_T):
    return [(basis_state(n_t, n_c, 0, 0), basis_state(n_t, n_c, 0, 6))]


PULSE_SPECS = {
    "opt": {
        "label": r"$U_{\mathrm{opt}}$ ($|g,0\rangle\!\to\!|g,6\rangle$)",
        "main_path": os.path.join(REPO_ROOT, "pulses", "u_opt_main.npy"),
        "light_path": os.path.join(REPO_ROOT, "pulses", "u_opt10x_light.npy"),
        "get_pairs": get_g6_state_pairs,
        "psi0_fn": lambda n_c: basis_state(N_T, n_c, 0, 0),
        "log_path": os.path.join(LOG_DIR, "light_refine_10x_run.log"),
        "n_plot": 18,
    },
    "enc": {
        "label": r"$U_{\mathrm{enc}}$ (cat-code encode)",
        "main_path": os.path.join(REPO_ROOT, "pulses", "u_enc_main.npy"),
        "light_path": os.path.join(REPO_ROOT, "pulses", "u_enc10x_light.npy"),
        "get_pairs": get_encode_state_pairs,
        # Encode |g,0> branch for trajectory visualization
        "psi0_fn": lambda n_c: basis_state(N_T, n_c, 0, 0),
        "log_path": os.path.join(LOG_DIR, "light_refine_enc10x_run.log"),
        "n_plot": 18,
    },
}


def evaluate_fidelity(u, dt, get_pairs, trunc_list, n_t=N_T):
    results = {}
    for nc in trunc_list:
        pairs = get_pairs(n_c=nc, n_t=n_t)
        psi_i = [p[0] for p in pairs]
        psi_f = [p[1] for p in pairs]
        H0, Hc = make_hamiltonian(n_t=n_t, n_c=nc)
        F, _ = fidelity_multi_state(u, H0, Hc, psi_i, psi_f, dt=dt, want_grad=False)
        results[nc] = float(F)
        print(f"    n_c={nc:2d}: F={F:.6f}")
    return results


def simulate_trajectory(u, H0, Hc, psi_i, dt, n_c, n_t):
    """Propagate and record P(n,t), <n>(t), transmon excited population."""
    N_steps = u.shape[0]
    psi = psi_i.copy().astype(complex)
    times = np.arange(N_steps + 1) * dt
    n_mean = np.zeros(N_steps + 1)
    P = np.zeros((n_c, N_steps + 1))
    transmon_ex = np.zeros(N_steps + 1)

    def record(k, state):
        for nc in range(n_c):
            p = 0.0
            for nt in range(n_t):
                p += np.abs(state[nt * n_c + nc]) ** 2
            P[nc, k] = p
        n_mean[k] = np.sum(np.arange(n_c) * P[:, k])
        prob_g = np.sum(np.abs(state[0:n_c]) ** 2)
        transmon_ex[k] = 1.0 - prob_g

    record(0, psi)
    for k in range(N_steps):
        Uk, _, _ = step_data(H0, Hc, u[k], dt)
        psi = Uk @ psi
        record(k + 1, psi)
    return times, n_mean, P, transmon_ex


def parse_wall_clock(log_path):
    """Extract wall-clock seconds and final fidelity from a light-refine log."""
    if not os.path.exists(log_path):
        return None
    text = open(log_path).read()
    wall = None
    m = re.search(r"Wall clock:\s*([\d.]+)\s*s", text)
    if m:
        wall = float(m.group(1))
    fid = None
    m = re.search(r"Bare fidelity at training n_c=\d+:\s*([\d.]+)", text)
    if m:
        fid = float(m.group(1))
    nit = None
    m = re.search(r"'iterations':\s*(\d+)", text)
    if m:
        nit = int(m.group(1))
    N = None
    m = re.search(r"Refined:\s*N=(\d+),\s*dt=([\d.]+)", text)
    if m:
        N = int(m.group(1))
        dt = float(m.group(2))
    else:
        dt = None
    return {"wall_s": wall, "final_F": fid, "iterations": nit, "N": N, "dt": dt}


def plot_fidelity(fidelity_data):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=False)
    for ax, key in zip(axes, ("opt", "enc")):
        data = fidelity_data[key]
        nc = data["n_c"]
        ax.plot(nc, data["F_main"], "o-", color="#1f77b4", lw=1.8, ms=5,
                label=rf"Stage 1 main ($dt=2\,\mathrm{{ns}}$)")
        ax.plot(nc, data["F_light"], "s-", color="#d62728", lw=1.8, ms=5,
                label=rf"Stage 2 light ($dt=0.2\,\mathrm{{ns}}$)")
        ax.axvline(24, color="gray", ls="--", lw=1.0, alpha=0.7, label=r"train $n_c=24$")
        ax.set_xlabel(r"Cavity truncation $n_c$")
        ax.set_ylabel("Fidelity $F$")
        ax.set_title(PULSE_SPECS[key]["label"])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        ymin = min(min(data["F_main"]), min(data["F_light"]))
        ax.set_ylim(max(0.96, ymin - 0.01), 1.001)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"dt_refine_fidelity_vs_nc.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def plot_photon_trajectory(key, traj_main, traj_light):
    t_m, n_m, _, _ = traj_main
    t_l, n_l, _, _ = traj_light
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.plot(t_m * 1000, n_m, "-", color="#1f77b4", lw=2.0,
            label=r"Stage 1 main ($dt=2\,\mathrm{ns}$)")
    ax.plot(t_l * 1000, n_l, "-", color="#d62728", lw=1.6, alpha=0.9,
            label=r"Stage 2 light ($dt=0.2\,\mathrm{ns}$)")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"Mean photon number $\langle n\rangle$")
    ax.set_title(PULSE_SPECS[key]["label"] + r" — photon trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"dt_refine_photon_trajectory_{key}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def plot_fock_heatmap(key, traj_main, traj_light, n_plot=18):
    t_m, n_m, P_m, _ = traj_main
    t_l, n_l, P_l, _ = traj_light
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    vmax = max(P_m[:n_plot].max(), P_l[:n_plot].max())
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-12))

    for ax, (t, n_mean, P, title) in zip(
        axes,
        [
            (t_m, n_m, P_m, r"Stage 1 main ($dt=2\,\mathrm{ns}$)"),
            (t_l, n_l, P_l, r"Stage 2 light ($dt=0.2\,\mathrm{ns}$)"),
        ],
    ):
        t_ns = t * 1000
        im = ax.pcolormesh(
            t_ns, np.arange(n_plot), P[:n_plot, :],
            cmap="Blues", shading="nearest", norm=norm,
        )
        ax.plot(t_ns, n_mean, color="red", lw=1.8, alpha=0.85, label=r"$\langle n\rangle(t)$")
        ax.set_xlabel("Time (ns)")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Photon number $n$")
    fig.suptitle(PULSE_SPECS[key]["label"] + r" — $P(n,t)$", y=1.02)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.02, fraction=0.03)
    cbar.set_label(r"$P(n,t)$")
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"dt_refine_fock_heatmap_{key}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def main():
    fidelity_rows = []
    fidelity_data = {}
    traj_metrics = []
    cost_rows = []
    trajectories = {}

    for key, spec in PULSE_SPECS.items():
        print(f"\n{'='*60}\nPulse: {key}\n{'='*60}")
        u_main = np.load(spec["main_path"])
        u_light = np.load(spec["light_path"])
        print(f"  main  shape={u_main.shape}, peak={np.max(np.abs(u_main)):.4f}")
        print(f"  light shape={u_light.shape}, peak={np.max(np.abs(u_light)):.4f}")

        print("  Fidelity sweep — main:")
        F_main = evaluate_fidelity(u_main, DT_MAIN, spec["get_pairs"], TRUNC_LIST)
        print("  Fidelity sweep — light:")
        F_light = evaluate_fidelity(u_light, DT_LIGHT, spec["get_pairs"], TRUNC_LIST)

        ncs = sorted(F_main.keys())
        fidelity_data[key] = {
            "n_c": ncs,
            "F_main": [F_main[n] for n in ncs],
            "F_light": [F_light[n] for n in ncs],
        }
        for nc in ncs:
            fidelity_rows.append({
                "pulse": key,
                "n_c": nc,
                "F_main": F_main[nc],
                "F_light": F_light[nc],
                "delta_F": F_light[nc] - F_main[nc],
            })

        # Trajectories at training truncation
        print(f"  Trajectory simulation at n_c={TRAJ_N_C}...")
        H0, Hc = make_hamiltonian(N_T, TRAJ_N_C)
        psi0 = spec["psi0_fn"](TRAJ_N_C)
        traj_main = simulate_trajectory(u_main, H0, Hc, psi0, DT_MAIN, TRAJ_N_C, N_T)
        traj_light = simulate_trajectory(u_light, H0, Hc, psi0, DT_LIGHT, TRAJ_N_C, N_T)
        trajectories[key] = (traj_main, traj_light)

        for tag, u, traj, dt in (
            ("main", u_main, traj_main, DT_MAIN),
            ("light", u_light, traj_light, DT_LIGHT),
        ):
            times, n_mean, P, tex = traj
            traj_metrics.append({
                "pulse": key,
                "stage": tag,
                "N": u.shape[0],
                "dt_us": dt,
                "duration_us": u.shape[0] * dt,
                "peak_amp_rad_per_us": float(np.max(np.abs(u))),
                "mean_abs_amp": float(np.mean(np.abs(u))),
                "max_n_mean": float(np.max(n_mean)),
                "final_n_mean": float(n_mean[-1]),
                "max_transmon_ex": float(np.max(tex)),
                "final_transmon_ex": float(tex[-1]),
                "max_fock_occupied": int(np.max(np.where(P.max(axis=1) > 1e-4)[0])),
            })

        log_info = parse_wall_clock(spec["log_path"])
        if log_info:
            cost_rows.append({
                "pulse": key,
                "N_refined": log_info["N"],
                "dt_us": log_info["dt"],
                "s": 10,
                "n_c_train": 24,
                "iterations": log_info["iterations"],
                "wall_clock_s": log_info["wall_s"],
                "wall_clock_min": log_info["wall_s"] / 60.0 if log_info["wall_s"] else None,
                "final_F_train": log_info["final_F"],
            })
            print(f"  Log wall clock: {log_info['wall_s']:.1f} s "
                  f"({log_info['wall_s']/60:.2f} min), F={log_info['final_F']}")

    # Write CSVs
    fid_csv = os.path.join(TAB_DIR, "dt_refine_fidelity_comparison.csv")
    with open(fid_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pulse", "n_c", "F_main", "F_light", "delta_F"])
        w.writeheader()
        w.writerows(fidelity_rows)
    print(f"\nSaved {fid_csv}")

    traj_csv = os.path.join(TAB_DIR, "dt_refine_trajectory_metrics.csv")
    with open(traj_csv, "w", newline="") as f:
        fields = list(traj_metrics[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(traj_metrics)
    print(f"Saved {traj_csv}")

    cost_csv = os.path.join(TAB_DIR, "dt_refine_cost_summary.csv")
    with open(cost_csv, "w", newline="") as f:
        fields = list(cost_rows[0].keys()) if cost_rows else [
            "pulse", "N_refined", "dt_us", "s", "n_c_train",
            "iterations", "wall_clock_s", "wall_clock_min", "final_F_train",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(cost_rows)
    print(f"Saved {cost_csv}")

    # Figures
    print("\nPlotting figures...")
    plot_fidelity(fidelity_data)
    for key in ("opt", "enc"):
        tm, tl = trajectories[key]
        plot_photon_trajectory(key, tm, tl)
        plot_fock_heatmap(key, tm, tl, n_plot=PULSE_SPECS[key]["n_plot"])

    print("\nDone.")


if __name__ == "__main__":
    main()
