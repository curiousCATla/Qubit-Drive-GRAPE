#!/usr/bin/env python3
"""
plot_qutip_validation.py

Renders tables/qutip_validation.csv (written by QuTip/qutip_validate.py) as a
small-multiples figure: one panel per gate, |grape_core F - qutip F| vs.
cavity truncation n_c, log scale. Trained truncations (22/24/26) and
held-out truncations (28/30/32) are distinguished by both color and marker
shape, so the distinction survives grayscale printing.

Usage:
    python plot_qutip_validation.py
"""
import csv
from collections import defaultdict

import matplotlib.pyplot as plt

CSV_PATH = "tables/qutip_validation.csv"
OUT_PNG = "figures/qutip_validation.png"
OUT_PDF = "figures/qutip_validation.pdf"

COLOR_TRAINED = "#2a78d6"    # categorical slot 1 (blue)
COLOR_HELDOUT = "#eb6834"    # categorical slot 6 (orange)
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED = "#898781"
INK = "#0b0b0b"

# Panel order: production prep pulse first, then the six logical gates,
# then encode/decode -- matches the order gates are introduced in the README.
PANEL_ORDER = ["g0->g6 prep", "X", "Y", "Z", "H", "T", "I", "U_enc", "U_dec"]

TOL_LINE = 1e-6   # representative ODE solver tolerance, drawn for scale


def load_rows():
    data = defaultdict(list)
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            data[row["label"]].append({
                "n_c": int(row["n_c"]),
                "diff": max(float(row["diff"]), 1e-12),  # floor for log scale
                "trained": row["trained"] == "True",
            })
    for label in data:
        data[label].sort(key=lambda r: r["n_c"])
    return data


def main():
    data = load_rows()
    labels = [l for l in PANEL_ORDER if l in data]

    fig, axes = plt.subplots(3, 3, figsize=(9.5, 7.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("#fcfcfb")

    for ax, label in zip(axes.flat, labels):
        ax.set_facecolor("#fcfcfb")
        rows = data[label]
        n_c = [r["n_c"] for r in rows]
        diff = [r["diff"] for r in rows]
        trained_pts = [(r["n_c"], r["diff"]) for r in rows if r["trained"]]
        heldout_pts = [(r["n_c"], r["diff"]) for r in rows if not r["trained"]]

        ax.plot(n_c, diff, color=AXIS, linewidth=1, zorder=1)
        if trained_pts:
            tx, ty = zip(*trained_pts)
            ax.scatter(tx, ty, marker="o", s=42, color=COLOR_TRAINED,
                       edgecolor="white", linewidth=0.6, zorder=3, label="trained (22/24/26)")
        if heldout_pts:
            hx, hy = zip(*heldout_pts)
            ax.scatter(hx, hy, marker="^", s=46, facecolor="none",
                       edgecolor=COLOR_HELDOUT, linewidth=1.4, zorder=3,
                       label="held out (28-40)")

        ax.axhline(TOL_LINE, color=MUTED, linewidth=0.8, linestyle="--", zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 1e-4)
        ax.set_title(label, fontsize=10, color=INK, loc="left")
        ax.grid(True, which="major", color=GRIDLINE, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)

    for ax in axes.flat[len(labels):]:
        ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("cavity truncation $n_c$", fontsize=9, color="#52514e")
    for ax in axes[:, 0]:
        ax.set_ylabel("$|F_{grape\\_core} - F_{qutip}|$", fontsize=9, color="#52514e")

    handles, plot_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, plot_labels, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("grape_core vs. QuTiP sesolve: independent-propagator agreement per gate",
                 fontsize=11, color=INK, y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    fig.savefig(OUT_PNG, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(OUT_PDF, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {OUT_PNG}\nWrote {OUT_PDF}")


if __name__ == "__main__":
    main()
