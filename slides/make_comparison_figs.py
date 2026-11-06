#!/usr/bin/env python3
"""
Build the two images for the "our X gate vs the paper" slides.

    python3 slides/make_comparison_figs.py

Writes only into slides/img/ (never into figures/):

  paper_fig1def.png  Roy et al. Fig. 1d, 1e, 1f, cropped from page 2 of the paper
                     PDF in sources/ and laid out in one row in d-e-f order, so it
                     lines up panel for panel with ours.
  ours_fig2a.png     figures/est/fig1def_est_vs_ord_linear.png (est_experiments.ipynb
                     Figure 2a, made by EST/diagnostics.py) with its title strip
                     trimmed. Regenerate the source with `python3 EST/diagnostics.py`.
  ours_err_pick.png  the same three panels for three X gates: slide 42's EsT (blue) and
                     Ord (red), and the EsT gate with C_err in stage 1 only that the pick
                     rule of est_optimization.ipynb section 7.2 chose: seed 6, final
                     two-stage gate (green; the blue curve of Figure 7c). Scored here with
                     EST/diagnostics.analyze at n_c = 20, the scorer of both slide 42 and
                     section 7.2, and checked against section 7.2's traces.

The crop boxes are page pixels at 600 dpi. They were set by eye on this exact PDF;
if the PDF changes, re-check them by viewing the output.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
PAPER = os.path.join(REPO, "sources",
                     "Error Semitransparent Universal Control of a Bosonic Logical Qubit.pdf")
OURS = os.path.join(REPO, "figures", "est", "fig1def_est_vs_ord_linear.png")
OUT = os.path.join(HERE, "img")

DPI = 600
PAGE = 2
# (left, top, right, bottom) in page pixels at 600 dpi. Each box starts just below its
# panel letter, so no text from a neighbouring panel (panel c's "Qubit", e's x-label,
# the letter f) leaks in; the slide labels the row instead.
BOXES = {
    "d": (2263, 1222, 3333, 1996),
    "e": (3400, 440, 4489, 1185),
    "f": (3350, 1222, 4489, 1996),
}
OURS_TRIM_TOP = 60          # px: the suptitle strip of the matplotlib figure

# The C_err comparison: slide 42's pair plus the section 7.2 pick (seed 6, final gate).
ERR_PULSES = {"est": "u_X_est.npy", "ord": "u_X_ord.npy",
              "pick": "u_X_est_err_err2st_seed6.npy"}
ERR_TRACES = os.path.join(REPO, "tables", "est_err2st_X_seeds_traces.npz")
ERR_TRACE_KEY = "run/seed6/final"
ERR_STYLE = {"pick": dict(color="#1baf7a", lw=1.8, ls="-.",
                          label=r"EsT + $C_{\rm err}$, seed 6 (the pick), final")}


def render_page():
    tmp = tempfile.mkdtemp()
    stem = os.path.join(tmp, "page")
    subprocess.run(["pdftoppm", "-r", str(DPI), "-f", str(PAGE), "-l", str(PAGE),
                    "-png", PAPER, stem], check=True)
    (name,) = [f for f in os.listdir(tmp) if f.endswith(".png")]
    return Image.open(os.path.join(tmp, name)).convert("RGB")


def row(panels, gap=40):
    """Lay panels side by side, bottom-aligned on a white canvas."""
    h = max(p.height for p in panels)
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (w, h), "white")
    x = 0
    for p in panels:
        canvas.paste(p, (x, h - p.height))
        x += p.width + gap
    return canvas


def ours_err_pick(path):
    """Slide 42's EsT and Ord with the section 7.2 picked C_err gate, linear axes."""
    from EST import diagnostics
    results = {k: diagnostics.analyze(np.load(os.path.join(diagnostics.PULSE_DIR, f)), gate="X")
               for k, f in ERR_PULSES.items()}
    # Same pulse, same scorer: the green curves must be Figure 7c's to rounding.
    tr = np.load(ERR_TRACES)
    for ours, theirs in (("delta_qec", "dqec"), ("leakage", "L"), ("eta_0L", "eta0"),
                         ("eta_avg", "eta_avg")):
        err = np.max(np.abs(results["pick"][ours] - tr[f"{ERR_TRACE_KEY}/{theirs}"]))
        assert err < 1e-10, f"{ours} differs from {ERR_TRACES} by {err:.2e}"
    for k, r in results.items():
        print(f"  {k:5s} F1 {r['F1']:.4f}  L(T) {r['leakage'][-1]:.3f}  "
              f"max Delta_QEC {r['delta_qec'].max():.2f}  eta(0_L)(T) {r['eta_0L'][-1]:.3f}  "
              f"max Fock {r['max_active_fock']}")
    return diagnostics.plot_fig1(results, path, scale="linear", title=None,
                                 styles=ERR_STYLE, paper_labels=False, eta_mean=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    page = render_page()
    panels = [page.crop(BOXES[k]) for k in ("d", "e", "f")]
    paper = row(panels)
    paper.save(os.path.join(OUT, "paper_fig1def.png"))

    ours = Image.open(OURS).convert("RGB")
    ours = ours.crop((0, OURS_TRIM_TOP, ours.width, ours.height))
    ours.save(os.path.join(OUT, "ours_fig2a.png"))
    print("wrote", os.path.join(OUT, "paper_fig1def.png"), paper.size)
    print("wrote", os.path.join(OUT, "ours_fig2a.png"), ours.size)
    print("wrote", ours_err_pick(os.path.join(OUT, "ours_err_pick.png")))


if __name__ == "__main__":
    main()
