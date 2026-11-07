#!/usr/bin/env python3
"""
The training grid of FWM_tails_ge.ipynb §4 (decisions of 2026-11-03): gates EsT, Ord x tail 1, 2, 3 us x
mixing cap 0.1, 0.2 x starts gertler, zero (zero with noise 0.05 of each cap, seed 1) = 24 runs of
DGRAPE/train_fwm_tail_ge.py, 600 iterations, at most --parallel at a time (longest first), each with
3 joblib jobs and OPENBLAS_NUM_THREADS=1. Logs and outputs in --outdir (default ~/fwm_runs/tailge).
A run whose <tag>.json exists is skipped, so the grid resumes after an interruption (a killed run
restarts from its start; resume it by hand with --init-x <outdir>/x_<tag>_best.npy).

Tags: X_tailge_{est,ord}_T{1,2,3}_c{10,20}_{g,z}.

    python DGRAPE/fwm_tail_ge_grid.py --parallel 8            (~10 h on the 16-core PC)
    python DGRAPE/fwm_tail_ge_grid.py --list                   (print the commands only)
"""

import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(REPO_ROOT, "DGRAPE", "train_fwm_tail_ge.py")
DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "fwm_runs", "tailge")
GATES = ("EsT", "Ord")
TAILS = (3.0, 2.0, 1.0)              # longest first
CAPS = (0.1, 0.2)
STARTS = {"g": ("gertler", []), "z": ("zero", ["--noise", "0.05", "--seed", "1"])}


def tag_of(gate, T, cap, s):
    return f"X_tailge_{gate.lower()}_T{T:g}_c{int(round(100 * cap))}_{s}"


def grid():
    for T in TAILS:
        for gate in GATES:
            for cap in CAPS:
                for s, (start, extra) in STARTS.items():
                    tag = tag_of(gate, T, cap, s)
                    yield tag, ["--gate", gate, "--tail-us", f"{T:g}", "--cap", f"{cap:g}", "--start", start,
                                "--tag", tag, "--max-iter", "600"] + extra


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    todo = [(t, args) for t, args in grid() if not os.path.exists(os.path.join(a.outdir, f"{t}.json"))]
    if a.list:
        for t, args in todo:
            print(" ".join([sys.executable, TRAIN] + args + ["--outdir", a.outdir]))
        return
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
               PYTHONIOENCODING="utf-8")
    running = {}
    t0 = time.time()
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
