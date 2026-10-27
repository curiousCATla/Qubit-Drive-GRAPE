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

The crop boxes are page pixels at 600 dpi. They were set by eye on this exact PDF;
if the PDF changes, re-check them by viewing the output.
"""

import os
import subprocess
import tempfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
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


if __name__ == "__main__":
    main()
