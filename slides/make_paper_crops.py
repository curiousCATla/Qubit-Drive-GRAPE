#!/usr/bin/env python3
"""
Crop the paper figures shown in the "Bosonic qubits: the basics" section.

    python3 slides/make_paper_crops.py

Writes only into slides/img/ (never into figures/). Every image is cut from the paper
PDF in sources/ (Roy, Wetherbee & Fatemi, arXiv:2603.15356) and cited on its slide:

  paper_fig1ab.png  Fig. 1a-b, page 2: an error during an ordinary gate vs an ET gate.
  paper_figA1.png   Fig. A1, page 11: schematic of the 3D package.
  paper_figA2.png   Fig. A2, page 13: room-temperature and cryogenic wiring.
  paper_figA5.png   Fig. A5, page 18: Fock occupation during the Ord and EsT X gates.
  paper_fig3a.png   Fig. 3a, page 5: the experimental sequence (encode, N gates, parity, decode).
  paper_fig3b.png   Fig. 3b, page 5: X-gate code and error space fidelities, measured.
  paper_fig3c.png   Fig. 3c, page 5: the same, modelled without parity-selection errors.
  paper_fig3d.png   Fig. 3d, page 5: per-gate infidelities for X, H, T.
  paper_fig4ab.png  Fig. 4a-b, page 6: gates protected by AQEC, experiment.
  paper_figA6.png   Fig. A6, page 18: gates with AQEC, simulation.
  paper_figA7.png   Fig. A7, page 19: H and T code and error space fidelities, experiment.

Each crop stops above the figure caption; the slide retells the caption in its own
words. Table I is retyped on its slide, not cropped.

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
OUT = os.path.join(HERE, "img")

DPI = 600
# name: (page, (left, top, right, bottom) in page pixels at 600 dpi)
CROPS = {
    "paper_fig1ab": (2, (710, 430, 2205, 2000)),
    "paper_figA1": (11, (3290, 1200, 4020, 1975)),
    "paper_figA2": (13, (1030, 420, 4440, 3540)),
    "paper_figA5": (18, (740, 360, 4520, 1850)),
    "paper_fig3a": (5, (900, 400, 2860, 860)),
    "paper_fig3b": (5, (900, 860, 2860, 2290)),
    "paper_fig3c": (5, (2850, 400, 4275, 1310)),
    "paper_fig3d": (5, (2850, 1330, 4275, 2290)),
    "paper_fig4ab": (6, (550, 400, 2375, 2210)),
    "paper_figA6": (18, (725, 2425, 4400, 4310)),
    "paper_figA7": (19, (330, 360, 4650, 2440)),
}


def render_page(page):
    tmp = tempfile.mkdtemp()
    stem = os.path.join(tmp, "page")
    subprocess.run(["pdftoppm", "-r", str(DPI), "-f", str(page), "-l", str(page),
                    "-png", PAPER, stem], check=True)
    (name,) = [f for f in os.listdir(tmp) if f.endswith(".png")]
    return Image.open(os.path.join(tmp, name)).convert("RGB")


def main():
    os.makedirs(OUT, exist_ok=True)
    pages = {}
    for name, (page, box) in CROPS.items():
        if page not in pages:
            pages[page] = render_page(page)
        img = pages[page].crop(box)
        path = os.path.join(OUT, name + ".png")
        img.save(path)
        print("wrote", path, img.size)


if __name__ == "__main__":
    main()
