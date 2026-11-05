#!/usr/bin/env python3
"""
Repeated X gates with FWM tails (FWM_tails.ipynb §8), DGRAPE/fwm_repeat.py. Forward only.

Each --tails entry LABEL=TAG names a trained tail (eta_<TAG>.npy, <TAG>.json in --indir); its json gives the
gate pulse before it and the target (X, or I for the idle tail). For every entry, in the open system (decay
set) and closed:

    B1  after_n_gates: N = 1..N_max gates back to back, then that one tail (rows with and without it)
    B2  pairs:         N (gate + tail) pairs, and N (gate + idle of the tail's length) pairs
                       (not for a target-I entry: an idle "gate" in pairs is just waiting)

plus the B2 reference teeth_memory (no gates, Shirol's Table II teeth for N periods of one gate + tail),
open and closed, and a readout check: B2 pairs at n_r = 3 for every X entry, open.

    python DGRAPE/fwm_repeat_run.py --tails EsT-good=X_tail_estg_T08_t2 Ord=X_tail_ord_T08_t2 \\
        EsT-mid=X_tail_est_T08_z idle=X_tail_idle_T08_t2 --out-dir tables

--kerr-decode runs only the memory references, with every row also scored after the Kerr-corrected decoding
(fwm_repeat.kerr_decode), open and closed, to fwm_tail_repeat_kerr.csv (the B1/B2 tables are left alone). For
each target-I entry: B1 = that entry (idle waits of the gate's length, with and without its tail) and the teeth
sampled every gate length; B2 = plain idle and the teeth, both sampled every gate + tail period:

    python DGRAPE/fwm_repeat_run.py --tails idle=X_tail_idle_T08_t2 --kerr-decode --n-jobs 4 --out-dir tables
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from joblib import Parallel, delayed                  # noqa: E402

from DGRAPE import fwm_repeat as rp                   # noqa: E402

CLOSED = frozenset()


def load_tail(indir, tag):
    with open(os.path.join(indir, f"{tag}.json")) as f:
        info = json.load(f)
    eta = np.load(os.path.join(indir, f"eta_{tag}.npy"))
    pulse = np.load(os.path.join(REPO_ROOT, info["gate_pulse"]))
    return info, eta, pulse


def job(protocol, label, tag, pulse, eta, target, n_max, n_c, n_r, keep, period_us=None, kerr=False, as_protocol=None):
    t0 = time.time()
    kw = {"n_c": n_c, "n_r": n_r, "keep": None if keep is None else set(keep)}
    gate = "I" if target == "I" else "X"
    if protocol == "B1":
        rows = rp.after_n_gates(pulse, eta, n_max, gate=gate, with_kerr_decode=kerr, **kw)
    elif protocol == "B2":
        rows = rp.pairs(pulse, eta, n_max, gate=gate, **kw)
    else:
        rows = rp.teeth_memory(n_max, period_us, with_kerr_decode=kerr, **kw)
    if as_protocol is None:
        as_protocol = "B2" if protocol == "teeth" else protocol
    meta = {"protocol": as_protocol, "label": label, "tag": tag,
            "system": "closed" if keep == CLOSED else "open", "n_c": n_c, "n_r": n_r,
            "tail_us": 0.0 if eta is None else eta.shape[0] * rp.DT, "seconds": time.time() - t0}
    return [{**meta, **r} for r in rows]


def run_kerr_decode(a):
    jobs = []
    for entry in a.tails:
        label, tag = entry.split("=", 1)
        info, eta, pulse = load_tail(a.indir, tag)
        if info.get("target", "X") != "I":
            raise SystemExit(f"--kerr-decode takes idle (target I) entries only, not {entry}")
        T = round((pulse.shape[0] + eta.shape[0]) * rp.DT, 9)
        T_gate = round(pulse.shape[0] * rp.DT, 9)
        wait = np.zeros((pulse.shape[0] + eta.shape[0], 4))          # plain idle of one gate + tail period
        for keep in (None, CLOSED):
            jobs.append(("B1", label, tag, pulse, eta, "I", a.n_max, a.n_c, 2, keep, None, True))
            jobs.append(("teeth", "idle + Table II teeth", "", None, None, "I", a.n_max, a.n_c, 2, keep, T, True))
            jobs.append(("teeth", "idle + Table II teeth", "", None, None, "I", a.n_max, a.n_c, 2, keep, T_gate, True,
                         "B1"))
            jobs.append(("B1", "idle", "", wait, np.zeros((0, 2)), "I", a.n_max, a.n_c, 2, keep, None, True, "B2"))
    out = Parallel(n_jobs=a.n_jobs, backend="loky", verbose=5)(delayed(job)(*j) for j in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "fwm_tail_repeat_kerr.csv")
    df.to_csv(path, index=False)
    print(f"-> {path}")
    pd.set_option("display.width", 200)
    show = df[df.N.isin([1, 2, 5, 10, 20])].fillna({"tail": "-"})
    print(show.pivot_table(index="N", columns=["protocol", "label", "tail", "system", "decode"],
                           values="F_pro").to_string(float_format="%.4f"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tails", nargs="+", required=True, help="LABEL=TAG")
    p.add_argument("--indir", default=os.path.join(REPO_ROOT, "pulses", "dgrape"))
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "tables"))
    p.add_argument("--n-max", type=int, default=20)
    p.add_argument("--n-c", type=int, default=24)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--kerr-decode", action="store_true",
                   help="only the memory references, also scored after the Kerr-corrected decoding")
    a = p.parse_args()
    if a.kerr_decode:
        return run_kerr_decode(a)

    jobs, periods = [], set()
    for entry in a.tails:
        label, tag = entry.split("=", 1)
        info, eta, pulse = load_tail(a.indir, tag)
        target = info.get("target", "X")
        for keep in (None, CLOSED):
            jobs.append(("B1", label, tag, pulse, eta, target, a.n_max, a.n_c, 2, keep))
            if target != "I":
                jobs.append(("B2", label, tag, pulse, eta, target, a.n_max, a.n_c, 2, keep))
                jobs.append(("B2", f"{label} + idle", tag, pulse, np.zeros_like(eta), target, a.n_max, a.n_c, 2,
                             keep))
        if target != "I":
            jobs.append(("B2", label, tag, pulse, eta, target, a.n_max, a.n_c, 3, None))
            periods.add(round((pulse.shape[0] + eta.shape[0]) * rp.DT, 9))
    for T in sorted(periods):
        for keep in (None, CLOSED):
            jobs.append(("teeth", "idle + Table II teeth", "", None, None, "I", a.n_max, a.n_c, 2, keep, T))
    jobs.sort(key=lambda j: -j[8])                    # the n_r = 3 jobs are the slowest: start them first
    out = Parallel(n_jobs=a.n_jobs, backend="loky", verbose=5)(delayed(job)(*j) for j in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    os.makedirs(a.out_dir, exist_ok=True)
    for prot in ("B1", "B2"):
        path = os.path.join(a.out_dir, f"fwm_tail_repeat_{prot}.csv")
        df[df.protocol == prot].to_csv(path, index=False)
        print(f"-> {path}")
    pd.set_option("display.width", 200)
    show = df[(df.system == "open") & (df.n_r == 2) & df.N.isin([1, 2, 5, 10, 20])]
    show = show[(show.protocol == "B2") | (show["tail"] == "tail")]
    print(show.pivot_table(index="N", columns=["protocol", "label"], values="F_pro").to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
