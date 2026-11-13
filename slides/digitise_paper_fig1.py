#!/usr/bin/env python3
"""
Digitise Roy et al. Fig. 1d, 1e, 1f (X gate, EsT blue, Ord red) from the paper PDF and
print the time average, peak and end value of each curve, for the numbers table of
slide 43 ("What agrees, and what does not").

    python3 slides/digitise_paper_fig1.py                  # 600 dpi
    python3 slides/digitise_paper_fig1.py --dpi 400        # resolution check
    python3 slides/digitise_paper_fig1.py --overlay out.png

Writes nothing into the repo unless --overlay points there. The page is rendered with
pdftoppm and cropped with the panel boxes of make_comparison_figs.py.

How a curve is read:
  - axes: x = 0 and x = 1 are the centres of the left and right spines (the x tick at
    0.5 sits exactly halfway, checked below). y is set by the major ticks on the left
    spine: 2 and 1 on panels d and f, and 1 plus the dotted zero line on panel e.
  - curve: in every pixel column inside the frame, the median row of pixels within
    COLOR_TOL of the line colour and not grey (MIN_CHROMA). The legend swatches are masked (LEGEND, in data units, set by eye).
    Columns where the other curve hides this one are filled by linear interpolation.
  - time average: the mean over the uniform column grid on [0, 1] us.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from make_comparison_figs import BOXES, DPI, PAGE, PAPER  # noqa: E402

COLORS = {"EsT": (76, 114, 176), "Ord": (196, 78, 82)}   # matplotlib's muted blue / red
COLOR_TOL = 60
MIN_CHROMA = 40         # max - min channel: grey anti-aliasing of text and spines is ~0
DARK = 80               # max channel value of an axis-ink pixel
NAMES = {"d": "Delta_QEC", "e": "L", "f": "eta(|0_L>)"}
# Major y ticks on the left spine, top to bottom. On e the 0 tick is under the dotted
# line, so the dotted line itself is used for y = 0.
YTICKS = {"d": (2.0, 1.0), "e": (1.0,), "f": (2.0, 1.0)}
# Legend swatches, (x_min, x_max, y_min) in data units: pixels there are not curve.
LEGEND = {"d": (0.66, 0.84, 1.35), "e": (0.0, 0.16, 0.65), "f": (0.0, 0.16, 1.2)}


def render_page(dpi):
    tmp = tempfile.mkdtemp()
    stem = os.path.join(tmp, "page")
    subprocess.run(["pdftoppm", "-r", str(dpi), "-f", str(PAGE), "-l", str(PAGE),
                    "-png", PAPER, stem], check=True)
    (name,) = [f for f in os.listdir(tmp) if f.endswith(".png")]
    return Image.open(os.path.join(tmp, name)).convert("RGB")


def runs(idx):
    """Group sorted indices into (first, last) runs of consecutive values."""
    out = []
    for i in idx:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [tuple(r) for r in out]


def centre(run):
    return 0.5 * (run[0] + run[1])


def axes(dark, panel):
    """Pixel -> data maps for one panel: returns (x0, x1, y_of_row, inner_rows)."""
    h, w = dark.shape
    cols, rows = dark.sum(0), dark.sum(1)
    left = runs([i for i in range(w // 2) if cols[i] > 0.5 * h])[0]
    right = runs([i for i in range(w // 2, w) if cols[i] > 0.5 * h])[0]
    top = runs([i for i in range(h // 2) if rows[i] > 0.5 * w])[0]
    bottom = runs([i for i in range(h // 2, h) if rows[i] > 0.5 * w])[0]
    x0, x1 = centre(left), centre(right)
    s = int(round(20 * h / 774))            # tick length scales with the dpi

    # The x tick at 0.5 must sit halfway between the spines.
    band = dark[bottom[0] - s:bottom[0] - 2, :].sum(0)
    xt = [centre(r) for r in runs([i for i in range(w) if band[i] >= 0.75 * (s - 2)])]
    mid = [x for x in xt if left[1] + 5 < x < right[0] - 5]
    assert len(mid) == 1 and abs(mid[0] - 0.5 * (x0 + x1)) < 2, (panel, xt)

    band = dark[:, left[1] + 2:left[1] + s].sum(1)
    yt = [centre(r) for r in runs([i for i in range(h) if band[i] >= 0.75 * (s - 2)])
          if top[1] + 5 < centre(r) < bottom[0] - 5]
    ticks = YTICKS[panel]
    assert len(yt) == len(ticks), (panel, yt)
    if len(ticks) == 2:
        per_unit = (yt[1] - yt[0]) / (ticks[0] - ticks[1])
        y_zero = yt[1] + ticks[1] * per_unit
    else:
        inner = dark[:, left[1] + 30:right[0] - 30].sum(1)
        dotted = runs([i for i in range(top[1] + 5, bottom[0] - s)
                       if inner[i] > 0.2 * (right[0] - left[1])])
        assert len(dotted) == 1, (panel, dotted)
        y_zero = centre(dotted[0])
        per_unit = (y_zero - yt[0]) / ticks[0]
    return x0, x1, lambda r: (y_zero - r) / per_unit, (top[1] + 1, bottom[0])


def curve(img, x0, x1, y_of_row, inner, color, legend):
    """Median curve row per pixel column, as (t, value) on the column grid."""
    near = ((np.abs(img - np.array(color)).max(2) <= COLOR_TOL)
            & (img.max(2) - img.min(2) >= MIN_CHROMA))
    near[:inner[0]] = False
    near[inner[1]:] = False
    cols = np.arange(int(np.ceil(x0)), int(np.floor(x1)) + 1)
    t = (cols - x0) / (x1 - x0)
    val = np.full(len(cols), np.nan)
    rows = np.arange(img.shape[0])
    for k, c in enumerate(cols):
        r = rows[near[:, c]]
        if len(r) == 0:
            continue
        y = y_of_row(r.astype(float))
        if legend[0] <= t[k] <= legend[1]:
            y = y[y < legend[2]]
        if len(y):
            val[k] = np.median(y)
    ok = ~np.isnan(val)
    val = np.interp(t, t[ok], val[ok])
    return t, val, ok.mean()


def digitise(dpi):
    page = render_page(dpi)
    scale = dpi / DPI
    out = {}
    for panel, box in BOXES.items():
        crop = page.crop(tuple(int(round(v * scale)) for v in box))
        img = np.asarray(crop).astype(int)
        x0, x1, y_of_row, inner = axes(img.max(2) < DARK, panel)
        for label, color in COLORS.items():
            t, v, seen = curve(img, x0, x1, y_of_row, inner, color, LEGEND[panel])
            out[panel, label] = dict(t=t, v=v, seen=seen, crop=crop,
                                     x0=x0, x1=x1, y_of_row=y_of_row)
    return out


def overlay(res, path):
    """The three crops with the digitised curves drawn over them in black."""
    tiles = []
    for panel in BOXES:
        crop = res[panel, "EsT"]["crop"].copy()
        draw = ImageDraw.Draw(crop)
        for label in COLORS:
            r = res[panel, label]
            y_of_row = r["y_of_row"]
            # invert the linear row -> value map
            a, b = y_of_row(0.0), y_of_row(1.0) - y_of_row(0.0)
            xs = r["x0"] + r["t"] * (r["x1"] - r["x0"])
            ys = (r["v"] - a) / b
            draw.line(list(zip(xs, ys)), fill=(0, 0, 0), width=2)
        tiles.append(crop)
    h = max(c.height for c in tiles)
    canvas = Image.new("RGB", (sum(c.width for c in tiles), h), "white")
    x = 0
    for c in tiles:
        canvas.paste(c, (x, 0))
        x += c.width
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--overlay", help="save the crops with the digitised curves drawn on")
    args = ap.parse_args()
    res = digitise(args.dpi)
    print(f"Roy et al. Fig. 1d-f, X gate, digitised at {args.dpi} dpi")
    print(f"{'panel':<6}{'metric':<12}{'curve':<6}{'mean':>8}{'peak':>8}{'t_peak':>8}"
          f"{'end':>8}{'seen':>7}")
    for (panel, label), r in res.items():
        k = int(np.argmax(r["v"]))
        print(f"{panel:<6}{NAMES[panel]:<12}{label:<6}{r['v'].mean():8.3f}{r['v'][k]:8.3f}"
              f"{r['t'][k]:8.2f}{r['v'][-1]:8.3f}{r['seen']:7.0%}")
    if args.overlay:
        overlay(res, args.overlay)
        print("overlay:", args.overlay)


if __name__ == "__main__":
    main()
