#!/usr/bin/env python3
"""
compare_dt_light_vs_main_gates.py

Compare Stage-1 coarse pulses (u_dec_main.npy, u_H_main.npy, u_Y_main.npy,
dt=0.002 us) against their Stage-2 light dt-refined counterparts
(u_dec10x_light.npy, u_H10x_light.npy, u_Y10x_light.npy, dt=0.0002 us, s=10).

Same protocol/analysis structure as compare_dt_light_vs_main.py (opt/enc),
applied to the decode gate and the logical H, Y gates.

Outputs:
  tables/dt_refine_fidelity_comparison_gates.csv
  tables/dt_refine_trajectory_metrics_gates.csv
  tables/dt_refine_cost_summary_gates.csv
  figures/dt_refine_fidelity_vs_nc_gates.pdf/.png
  figures/dt_refine_photon_trajectory_gates.pdf/.png
  figures/dt_refine_waveform_dec.pdf/.png
  figures/dt_refine_waveform_H.pdf/.png
  figures/dt_refine_waveform_Y.pdf/.png
"""

from __future__ import annotations

import os
import sys
import re
import csv

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.grape_core import make_hamiltonian, fidelity_multi_state, step_data
from core.cat_code import (
    get_decode_state_pairs,
    get_logical_H_state_pairs,
    get_logical_Y_state_pairs,
    get_logical_cat_states,
    embed_in_joint_space,
)

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
TWO_PI = 2 * np.pi


def psi0_cat_plus(n_c):
    """|g> ⊗ |+Z_L>, the shared input state for U_dec / logical H / logical Y."""
    psi_plus_cav, _ = get_logical_cat_states(alpha=np.sqrt(3.0), n_c=n_c)
    return embed_in_joint_space(psi_plus_cav, n_t=N_T, n_c=n_c, t_level=0)


PULSE_SPECS = {
    "dec": {
        "label": r"$U_{\mathrm{dec}}$ (cat-code decode)",
        "short": r"$U_{\mathrm{dec}}$",
        "main_path": os.path.join(REPO_ROOT, "pulses", "u_dec_main.npy"),
        "light_path": os.path.join(REPO_ROOT, "pulses", "u_dec10x_light.npy"),
        "get_pairs": get_decode_state_pairs,
        "psi0_fn": psi0_cat_plus,
        "log_path": os.path.join(LOG_DIR, "light_refine_dec10x_run.log"),
        "n_plot": 18,
    },
    "H": {
        "label": r"$U_{H}$ (logical Hadamard)",
        "short": r"$U_{H}$",
        "main_path": os.path.join(REPO_ROOT, "pulses", "u_H_main.npy"),
        "light_path": os.path.join(REPO_ROOT, "pulses", "u_H10x_light.npy"),
        "get_pairs": get_logical_H_state_pairs,
        "psi0_fn": psi0_cat_plus,
        "log_path": os.path.join(LOG_DIR, "light_refine_H10x_run.log"),
        "n_plot": 18,
    },
    "Y": {
        "label": r"$U_{Y}$ (logical Y)",
        "short": r"$U_{Y}$",
        "main_path": os.path.join(REPO_ROOT, "pulses", "u_Y_main.npy"),
        "light_path": os.path.join(REPO_ROOT, "pulses", "u_Y10x_light.npy"),
        "get_pairs": get_logical_Y_state_pairs,
        "psi0_fn": psi0_cat_plus,
        "log_path": os.path.join(LOG_DIR, "light_refine_Y10x_run.log"),
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
    dt = None
    m = re.search(r"Refined:\s*N=(\d+),\s*dt=([\d.]+)", text)
    if m:
        N = int(m.group(1))
        dt = float(m.group(2))
    return {"wall_s": wall, "final_F": fid, "iterations": nit, "N": N, "dt": dt}


def _save(fig, stem):
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def plot_fidelity(fidelity_data):
    keys = list(PULSE_SPECS.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.6), sharey=False)
    for ax, key in zip(axes, keys):
        data = fidelity_data[key]
        nc = data["n_c"]
        ax.plot(nc, data["F_main"], "o-", color="#1f77b4", lw=1.8, ms=5,
                label=r"Stage 1 main ($dt=2\,\mathrm{ns}$)")
        ax.plot(nc, data["F_light"], "s-", color="#d62728", lw=1.8, ms=5,
                label=r"Stage 2 light ($dt=0.2\,\mathrm{ns}$)")
        ax.axvline(24, color="gray", ls="--", lw=1.0, alpha=0.7, label=r"train $n_c=24$")
        ax.set_xlabel(r"Cavity truncation $n_c$")
        ax.set_ylabel("Fidelity $F$")
        ax.set_title(PULSE_SPECS[key]["label"])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        ymin = min(min(data["F_main"]), min(data["F_light"]))
        ax.set_ylim(max(0.96, ymin - 0.01), 1.001)
    fig.tight_layout()
    _save(fig, "dt_refine_fidelity_vs_nc_gates")


def plot_photon_trajectory_combined(trajectories):
    keys = list(PULSE_SPECS.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.5), sharey=False)
    for ax, key in zip(axes, keys):
        traj_main, traj_light = trajectories[key]
        t_m, n_m, _, _ = traj_main
        t_l, n_l, _, _ = traj_light
        ax.plot(t_m * 1000, n_m, "-", color="#1f77b4", lw=2.0,
                label=r"Stage 1 ($dt=2\,\mathrm{ns}$)")
        ax.plot(t_l * 1000, n_l, "-", color="#d62728", lw=1.6, alpha=0.9,
                label=r"Stage 2 ($dt=0.2\,\mathrm{ns}$)")
        ax.set_xlabel("Time (ns)")
        ax.set_title(PULSE_SPECS[key]["label"], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel(r"Mean photon number $\langle n\rangle$")
    fig.tight_layout()
    _save(fig, "dt_refine_photon_trajectory_gates")


def plot_waveform_comparison(key, u_main, u_light, dt_main=DT_MAIN, dt_light=DT_LIGHT):
    t_m = np.arange(u_main.shape[0]) * dt_main * 1000  # ns
    t_l = np.arange(u_light.shape[0]) * dt_light * 1000
    u_m = u_main / TWO_PI
    u_l = u_light / TWO_PI

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.2), sharex=True)

    ax = axes[0]
    ax.plot(t_m, u_m[:, 0], color="#1f77b4", lw=1.2, label=r"Stage 1 $C_I$")
    ax.plot(t_m, u_m[:, 1], color="#1f77b4", lw=1.0, ls="--", alpha=0.85,
            label=r"Stage 1 $C_Q$")
    ax.plot(t_l, u_l[:, 0], color="#d62728", lw=0.9, alpha=0.85,
            label=r"Stage 2 $C_I$")
    ax.plot(t_l, u_l[:, 1], color="#d62728", lw=0.8, ls="--", alpha=0.75,
            label=r"Stage 2 $C_Q$")
    ax.set_ylabel("Cavity drive (MHz)")
    ax.set_title(PULSE_SPECS[key]["label"] + r" — control waveforms")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=7.5, loc="upper right")

    ax = axes[1]
    ax.plot(t_m, u_m[:, 2], color="#1f77b4", lw=1.2, label=r"Stage 1 $T_I$")
    ax.plot(t_m, u_m[:, 3], color="#1f77b4", lw=1.0, ls="--", alpha=0.85,
            label=r"Stage 1 $T_Q$")
    ax.plot(t_l, u_l[:, 2], color="#d62728", lw=0.9, alpha=0.85,
            label=r"Stage 2 $T_I$")
    ax.plot(t_l, u_l[:, 3], color="#d62728", lw=0.8, ls="--", alpha=0.75,
            label=r"Stage 2 $T_Q$")
    ax.set_ylabel("Transmon drive (MHz)")
    ax.set_xlabel("Time (ns)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=7.5, loc="upper right")

    fig.tight_layout()
    _save(fig, f"dt_refine_waveform_{key}")


def main():
    fidelity_rows = []
    fidelity_data = {}
    traj_metrics = []
    cost_rows = []
    trajectories = {}
    pulses = {}

    for key, spec in PULSE_SPECS.items():
        print(f"\n{'='*60}\nPulse: {key}\n{'='*60}")
        u_main = np.load(spec["main_path"])
        u_light = np.load(spec["light_path"])
        pulses[key] = (u_main, u_light)
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
            occupied = np.where(P.max(axis=1) > 1e-4)[0]
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
                "max_fock_occupied": int(np.max(occupied)) if len(occupied) else 0,
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

    fid_csv = os.path.join(TAB_DIR, "dt_refine_fidelity_comparison_gates.csv")
    with open(fid_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pulse", "n_c", "F_main", "F_light", "delta_F"])
        w.writeheader()
        w.writerows(fidelity_rows)
    print(f"\nSaved {fid_csv}")

    traj_csv = os.path.join(TAB_DIR, "dt_refine_trajectory_metrics_gates.csv")
    with open(traj_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(traj_metrics[0].keys()))
        w.writeheader()
        w.writerows(traj_metrics)
    print(f"Saved {traj_csv}")

    cost_csv = os.path.join(TAB_DIR, "dt_refine_cost_summary_gates.csv")
    with open(cost_csv, "w", newline="") as f:
        fields = list(cost_rows[0].keys()) if cost_rows else [
            "pulse", "N_refined", "dt_us", "s", "n_c_train",
            "iterations", "wall_clock_s", "wall_clock_min", "final_F_train",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(cost_rows)
    print(f"Saved {cost_csv}")

    print("\nPlotting figures...")
    plot_fidelity(fidelity_data)
    plot_photon_trajectory_combined(trajectories)
    for key in PULSE_SPECS:
        u_main, u_light = pulses[key]
        plot_waveform_comparison(key, u_main, u_light)

    print("\nDone.")


if __name__ == "__main__":
    main()
