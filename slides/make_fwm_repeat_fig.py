#!/usr/bin/env python3
"""
Build the image for the "repeated X gates, then one AQEC tail" slide.

    python3 slides/make_fwm_repeat_fig.py

Writes only slides/img/ours_fwm_repeat_B1.png. It redraws panel (a) of notebooks/FWM_tails.ipynb Figure 8 (protocol B1:
N X gates back to back, then one 0.8 us tail; open system, decay set, n_c 24, n_r 2) from the cached tables, with
fewer curves and slide names:

  kept     EsT and Ord, with and without the tail; idle + its tail (Kerr-corrected decoding)
  dropped  idle without a tail, and idle with Shirol's Table II teeth
  renamed  "+ one 0.8 us tail" -> "+ AQEC tail"; "EsT-good" -> "EsT"

The selection is the notebook's (cell `REP1 = ...`, function `rep`). Colours follow this deck (EsT blue, Ord red),
not the notebook. The N = 20 values are asserted, so a changed table cannot silently change the slide.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "img", "ours_fwm_repeat_B1.png")

ESTBLUE = (31 / 255, 105 / 255, 175 / 255)     # preamble.tex estblue
ORDRED = (190 / 255, 55 / 255, 45 / 255)       # preamble.tex ordred
IDLE = "0.45"

# (legend, table, label, tail, colour, line width, F_pro at N = 20)
CURVES = [
    ("EsT + AQEC tail", "B1", "EsT-good", "tail", ESTBLUE, 2.0, 0.5932),
    ("EsT, no tail", "B1", "EsT-good", "none", ESTBLUE, 2.0, 0.5200),
    ("Ord + AQEC tail", "B1", "Ord", "tail", ORDRED, 2.0, 0.4969),
    ("Ord, no tail", "B1", "Ord", "none", ORDRED, 2.0, 0.4818),
    ("idle + AQEC tail", "kerr", "idle", "tail", IDLE, 1.6, 0.8487),
]


def rep(df, label, system="open", tail=None, n_r=2):
    q = (df.label == label) & (df.system == system) & (df.n_r == n_r)
    if tail is not None:
        q &= df["tail"] == tail
    return df[q].sort_values("N")


def main():
    rep1 = pd.read_csv(os.path.join(REPO, "tables", "fwm_tail_repeat_B1.csv"))
    repk = pd.read_csv(os.path.join(REPO, "tables", "fwm_tail_repeat_kerr.csv"))
    tables = {"B1": rep1, "kerr": repk[(repk.decode == "kerr") & (repk.protocol == "B1")]}

    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for name, tab, label, tail, col, lw, f20 in CURVES:
        d = rep(tables[tab], label, tail=tail)
        assert list(d.N) == list(range(1, 21)), name
        got = float(d[d.N == 20].F_pro.iloc[0])
        assert abs(got - f20) < 1e-4, f"{name}: F_pro(N=20) = {got:.5f}, expected {f20}"
        with_tail = tail == "tail"
        ax.plot(d.N, d.F_pro, color=col, lw=lw, ls="-" if with_tail else ":",
                marker="o" if with_tail else None, ms=4.5, label=name)
        print(f"  {name:18s} N=1 {d.F_pro.iloc[0]:.4f}  N=10 {d[d.N == 10].F_pro.iloc[0]:.4f}  N=20 {got:.4f}")

    ax.set_xlabel("number of gates N (idle: N waits of 1.1 μs)")
    ax.set_ylabel(r"$F_{\rm pro}$ against $X^N$ (idle: I)")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.4, 1.0)
    ax.grid(color="0.9", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=13.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=300)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
