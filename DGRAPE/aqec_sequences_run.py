#!/usr/bin/env python3
"""
Run DGRAPE/aqec_sequences.py for notebooks/AQEC_compare.ipynb §6.1: EsT X, EsT H and alternating X/H gates with the
ideal AQEC step, the open-trained step (aqec_open_T15_w) and the FWM tail (seed 8), B1 and B2, N = 1..N_max. From the
repo root:

    OPENBLAS_NUM_THREADS=1 python3 DGRAPE/aqec_sequences_run.py --n-jobs 9 --out-dir ~/fwm_runs/aqec_seq
        -> <out-dir>/parts/<job>.csv (one per job, reused on a rerun), merged to
           <out-dir>/aqec_seq_B1.csv and <out-dir>/aqec_seq_B2.csv (copy them to tables/)

Jobs per sequence (X, H, XH): B1 at n_r 1 (gates only, + ideal AQEC, + open step), B1 at n_r 2 (gates only, + FWM tail),
B2 ideal AQEC, B2 open step (n_r 1), B2 FWM tail (n_r 2); plus truncation checks (B1 ideal AQEC at n_c 28 for H, XH).
Progress per job: <out-dir>/logs/<job>.log.

    --dry-run              N = 1 only, timed, nothing merged (estimate the wall time first)
    --channels decay       the decay set (regression against the existing tables); default shirol
    --sequences X H XH     a subset of sequences
"""

import argparse
import glob
import os
import sys
import time

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from joblib import Parallel, delayed                  # noqa: E402

np.seterr(divide="ignore", over="ignore", invalid="ignore")

N_C = 24
N_C_CHECK = 28


def jobs_for(seqs, with_checks=True):
    """[(name, kind, kwargs)]: kind "b1" or "b2" of aqec_sequences."""
    jobs = []
    for s in seqs:
        jobs += [(f"B1_{s}_aqec", "b1", dict(seq=s, corrections=("ideal AQEC", "open step"), n_c=N_C, n_r=1)),
                 (f"B1_{s}_fwm", "b1", dict(seq=s, corrections=("FWM tail",), n_c=N_C, n_r=2)),
                 (f"B2_{s}_ideal", "b2", dict(seq=s, correction="ideal AQEC", n_c=N_C, n_r=1)),
                 (f"B2_{s}_open", "b2", dict(seq=s, correction="open step", n_c=N_C, n_r=1)),
                 (f"B2_{s}_fwm", "b2", dict(seq=s, correction="FWM tail", n_c=N_C, n_r=2))]
        if with_checks and s != "X":
            jobs.append((f"B1_{s}_aqec_nc{N_C_CHECK}", "b1",
                         dict(seq=s, corrections=("ideal AQEC",), n_c=N_C_CHECK, n_r=1)))
    # longest first (H gates and FWM chains), so the pool ends together
    return sorted(jobs, key=lambda j: (("H" in j[0]) + ("fwm" in j[0]) + ("B2" in j[0])), reverse=True)


def run_job(name, kind, kw, n_max, channels, out_dir):
    from DGRAPE import aqec_sequences as asq
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    logf = open(os.path.join(out_dir, "logs", f"{name}.log"), "a")
    t0 = time.time()

    def log(s):
        logf.write(f"[{time.time() - t0:8.0f} s] {s}\n")
        logf.flush()

    fn = asq.b1 if kind == "b1" else asq.b2
    rows = fn(N_max=n_max, channels=channels, log=log, **kw)
    df = pd.DataFrame(rows).assign(job=name, seconds=time.time() - t0)
    log("done")
    return name, df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-max", type=int, default=20)
    p.add_argument("--n-jobs", type=int, default=9)
    p.add_argument("--channels", choices=["shirol", "decay"], default="shirol")
    p.add_argument("--sequences", nargs="+", default=["X", "H", "XH"])
    p.add_argument("--out-dir", default=os.path.expanduser("~/fwm_runs/aqec_seq"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-checks", action="store_true")
    a = p.parse_args()
    out_dir = os.path.expanduser(a.out_dir)
    parts = os.path.join(out_dir, "parts" if not a.dry_run else "dry_parts")
    os.makedirs(parts, exist_ok=True)
    n_max = 1 if a.dry_run else a.n_max
    jobs = jobs_for(a.sequences, not a.no_checks)
    todo = [j for j in jobs if a.dry_run or not os.path.exists(os.path.join(parts, f"{j[0]}.csv"))]
    print(f"{len(jobs)} jobs, {len(todo)} to run, N_max {n_max}, channels {a.channels}, n_jobs {a.n_jobs}", flush=True)
    for name, df in Parallel(n_jobs=a.n_jobs, backend="loky", return_as="generator_unordered")(
            delayed(run_job)(*j, n_max, a.channels, out_dir) for j in todo):
        df.to_csv(os.path.join(parts, f"{name}.csv"), index=False)
        print(f"{name}: {df.seconds.iloc[0]:.0f} s", flush=True)
    if a.dry_run:
        return
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(parts, "*.csv")))], ignore_index=True)
    for prot in ("B1", "B2"):
        df[df.protocol == prot].to_csv(os.path.join(out_dir, f"aqec_seq_{prot}.csv"), index=False)
    print(f"merged -> {out_dir}/aqec_seq_B1.csv, aqec_seq_B2.csv")


if __name__ == "__main__":
    main()
