#!/usr/bin/env python3
"""
GRAPE on the chip proxy (FWM_tails_ge_chips.ipynb §4): Gertler-comb tails trained on other chips
(DGRAPE/ge_chip.derived) after the EsT gate as trained on Shirol's chip (the "transplant" input,
--gate-chip shirol), cap |xi_M| 0.2, 300 iterations, n_c 16. Runs of DGRAPE/train_fwm_tail_ge.py, at most
--parallel at a time (longest first), each with 3 joblib jobs and OPENBLAS_NUM_THREADS=1, in --outdir
(default ~/fwm_runs/tailge_chip). A run whose <tag>.json exists is skipped (resumable).

Starts:
  m   the chip map's best sequential shot (DGRAPE/fwm_tail_ge_chipmap.py), its parameters taken from the
      map entry of the nearest length and rebuilt at this length, then through the training chain
  z   zero plus noise 0.05 of each cap (seed 1), as track 7

Tags: X_tailgech_est_<chip tag>_T<ns>_{m,z}, e.g. X_tailgech_est_ch300x500_T0800_m.

    python DGRAPE/fwm_tail_ge_chip_grid.py --list
    python DGRAPE/fwm_tail_ge_chip_grid.py --chips 1.31,2.8 2,3 3,3 3,5 4,5 --parallel 8
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np                                   # noqa: E402

TRAIN = os.path.join(REPO_ROOT, "DGRAPE", "train_fwm_tail_ge.py")
DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "fwm_runs", "tailge_chip")
CHIPS = ((1.31, 2.8), (2.0, 3.0), (3.0, 3.0), (3.0, 5.0), (4.0, 5.0))
TAILS = (1.0, 0.8, 0.5)              # longest first
CAP = 0.2
MAX_ITER = 300


def tag_of(chip, T, s, gate="EsT"):
    return f"X_tailgech_{gate.lower()}_{chip.tag}_T{int(round(1000 * T)):04d}_{s}"


def map_start(chip, T, outdir):
    """Pre-image .npy of the map's best shot rebuilt at length T (nearest map length); None if no entry."""
    from DGRAPE import fwm_tail_ge as ge
    from DGRAPE import fwm_tail_ge_chipmap as cm
    entries = glob.glob(os.path.join(cm.OUT_DIR, f"{chip.tag}_T*.json"))
    if not entries:
        return None
    rows = [json.load(open(f)) for f in entries]
    row = min(rows, key=lambda r: abs(r["tail_us"] - T))
    b4 = cm.shot(np.asarray(row["p_best"]), chip, T, CAP)
    x = ge.Chain(ge.n_steps(T), band=chip.band, cap_T=chip.cap_T, cap_M=CAP).preimage(b4)
    x = np.clip(x, -ge.U_BOX, ge.U_BOX)
    path = os.path.join(outdir, f"start_{tag_of(chip, T, 'm')}.npy")
    np.save(path, x)
    return path


def grid(chips, outdir):
    from DGRAPE import ge_chip as gc
    for T in TAILS:
        for pair in chips:
            chip = gc.derived(*pair)
            base = ["--gate", "EsT", "--tail-us", f"{T:g}", "--cap", f"{CAP:g}", "--max-iter", str(MAX_ITER),
                    "--chi-ge", f"{pair[0]:g}", "--chi-qr", f"{pair[1]:g}", "--gate-chip", "shirol"]
            for s in ("m", "z"):
                tag = tag_of(chip, T, s)
                if s == "m":
                    start = map_start(chip, T, outdir)
                    if start is None:
                        print(f"no map entry for {chip.tag}: skipping {tag}", flush=True)
                        continue
                    extra = ["--start", start]
                else:
                    extra = ["--start", "zero", "--noise", "0.05", "--seed", "1"]
                yield tag, base + extra + ["--tag", tag]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chips", nargs="+", default=[f"{a:g},{b:g}" for a, b in CHIPS],
                   help="chi_ge,chi_qr pairs in MHz")
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    chips = [tuple(float(v) for v in c.split(",")) for c in a.chips]
    os.makedirs(a.outdir, exist_ok=True)
    todo = [(t, args) for t, args in grid(chips, a.outdir) if not os.path.exists(os.path.join(a.outdir, f"{t}.json"))]
    if a.list:
        for t, args in todo:
            print(" ".join([sys.executable, TRAIN] + args + ["--outdir", a.outdir]))
        return
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
               PYTHONIOENCODING="utf-8")
    running = {}
    t0 = time.time()
    print(f"{len(todo)} runs, {a.parallel} at a time", flush=True)
    while todo or running:
        while todo and len(running) < a.parallel:
            tag, args = todo.pop(0)
            log = open(os.path.join(a.outdir, f"{tag}.log"), "w")
            running[tag] = (subprocess.Popen([sys.executable, "-u", TRAIN] + args + ["--outdir", a.outdir],
                                             cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log)
            print(f"[{(time.time() - t0) / 3600:5.2f} h] start {tag}", flush=True)
        time.sleep(30)
        for tag in list(running):
            proc, log = running[tag]
            if proc.poll() is not None:
                log.close()
                del running[tag]
                print(f"[{(time.time() - t0) / 3600:5.2f} h] done  {tag} (exit {proc.returncode})", flush=True)
    print(f"grid finished after {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
