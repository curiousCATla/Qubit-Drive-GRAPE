#!/usr/bin/env python3
"""
Repeated X gates with FWM tails (notebooks/FWM_tails.ipynb §8), DGRAPE/fwm_repeat.py. Forward only.

Each --tails entry LABEL=TAG names a trained tail (eta_<TAG>.npy, <TAG>.json in --indir); its json gives the
gate pulse before it and the target (X, or I for the idle tail). For every entry, in the open system (decay
set) and closed:

    B1  after_n_gates: N = 1..N_max gates back to back, then that one tail (rows with and without it)
    B2  pairs:         N (gate + tail) pairs, and N (gate + idle of the tail's length) pairs
                       (not for a target-I entry: an idle "gate" in pairs is just waiting)

plus the B2 reference teeth_memory (no gates, Shirol's Table II teeth for N periods of one gate + tail, each tooth
moved by its mean pump Stark shift), open and closed, and a readout check: B2 pairs at n_r = 3 for every X entry, open.

    python DGRAPE/fwm_repeat_run.py --tails EsT-good=X_tail_estg_T08_t2 Ord=X_tail_ord_T08_t2 \\
        EsT-mid=X_tail_est_T08_z idle=X_tail_idle_T08_t2 --out-dir tables

--kerr-decode runs only the memory references, with every row also scored after the Kerr-corrected decoding
(fwm_repeat.kerr_decode), open and closed, to fwm_tail_repeat_kerr.csv (the B1/B2 tables are left alone). For
each target-I entry: B1 = that entry (idle waits of the gate's length, with and without its tail) and the teeth
sampled every gate length; B2 = plain idle and the teeth, both sampled every gate + tail period. The teeth run twice:
Stark-tuned (label TEETH) and with the bare carriers used until 2026-11-09 (label TEETH_BARE, kept for the record):

    python DGRAPE/fwm_repeat_run.py --tails idle=X_tail_idle_T08_t2 --kerr-decode --n-jobs 4 --out-dir tables

--teeth-only (with the main mode or --every-k) runs only the Stark-tuned teeth jobs of that mode and replaces the
TEETH rows of the existing fwm_tail_repeat_B2.csv / fwm_tail_repeat_B3.csv in --out-dir; every other row is kept:

    python DGRAPE/fwm_repeat_run.py --tails EsT-good=X_tail_estg_T08_t2 --teeth-only --out-dir tables
    python DGRAPE/fwm_repeat_run.py --tails EsT-good=X_tail_estg_T08_t2 --every-k 1 --n-max 40 --teeth-only \
        --out-dir tables

--every-k K [K ...] runs B3 only (fwm_repeat.every_k: a tail after every k-th gate, N_max gates), for ONE X entry,
open and closed, plus the same gates back to back (no tail) open and closed, plus open memory references with the
Kerr-corrected decoding sampled every gate length out to the longest k = 1 time (N_max gates + N_max tails): plain
idle and the Table II teeth. Every job writes <out-dir>/B3_parts/<name>.csv when it ends; all of them are merged to
<out-dir>/fwm_tail_repeat_B3.csv (B1/B2 tables left alone). Finished parts are reused on a rerun:

    OPENBLAS_NUM_THREADS=1 python DGRAPE/fwm_repeat_run.py --tails EsT-good=X_tail_estg_T08_t2 \
        --every-k 1 2 3 4 5 6 7 8 --n-max 40 --n-jobs 20 --out-dir ~/fwm_runs/repeat_B3

--matched scores tails trained after m gates (train_fwm_tail.py --n-gates m; m from each json's n_gates, 1 if
absent). For every entry: m gates then that tail (after_n_gates, row N = m, with and without the tail), open and
closed. For each m the entry with the best training F_pro is "kept" and also runs B3 with k = m (every_k, a tail
after every m gates, scored after each tail and as is, no closing tails), open, --n-max gates. Parts in
<out-dir>/matched_parts, merged to <out-dir>/fwm_tail_matched.csv:

    OPENBLAS_NUM_THREADS=1 python DGRAPE/fwm_repeat_run.py --matched --n-max 40 --n-jobs 20 \
        --out-dir ~/fwm_runs/tail_m --tails m1_t2=X_tail_estg_T08_t2 m1_z=X_tail_estg_T08_z \
        m2_t2=X_tail_estg_T08_m2_t2 m2_z=X_tail_estg_T08_m2_z ...
"""

import argparse
import json
import math
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
TEETH = "idle + Table II teeth"                          # Stark-tuned (fwm_tail.teeth_stark_shifts)
TEETH_BARE = "idle + Table II teeth, bare carriers"      # the untuned teeth, as run until 2026-11-09


def replace_label_rows(path, new, label):
    """Rewrite the csv at path with its rows of `label` replaced by the DataFrame `new`; other rows unchanged."""
    old = pd.read_csv(path)
    if not (old.label == label).any():
        raise SystemExit(f"{path} has no rows labelled {label!r}")
    extra = set(new.columns) - set(old.columns)
    if extra:
        raise SystemExit(f"new rows have columns {sorted(extra)} that {path} lacks")
    df = pd.concat([old[old.label != label], new.reindex(columns=old.columns)], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"-> {path}: {int((old.label == label).sum())} {label!r} rows replaced by {len(new)}")


def load_tail(indir, tag):
    with open(os.path.join(indir, f"{tag}.json")) as f:
        info = json.load(f)
    eta = np.load(os.path.join(indir, f"eta_{tag}.npy"))
    pulse = np.load(os.path.join(REPO_ROOT, info["gate_pulse"]))
    return info, eta, pulse


def job(protocol, label, tag, pulse, eta, target, n_max, n_c, n_r, keep, period_us=None, kerr=False, as_protocol=None,
        stark_tuned=True):
    t0 = time.time()
    kw = {"n_c": n_c, "n_r": n_r, "keep": None if keep is None else set(keep)}
    gate = "I" if target == "I" else "X"
    if protocol == "B1":
        rows = rp.after_n_gates(pulse, eta, n_max, gate=gate, with_kerr_decode=kerr, **kw)
    elif protocol == "B2":
        rows = rp.pairs(pulse, eta, n_max, gate=gate, **kw)
    else:
        rows = rp.teeth_memory(n_max, period_us, with_kerr_decode=kerr, stark_tuned=stark_tuned, **kw)
    if as_protocol is None:
        as_protocol = "B2" if protocol == "teeth" else protocol
    meta = {"protocol": as_protocol, "label": label, "tag": tag,
            "system": "closed" if keep == CLOSED else "open", "n_c": n_c, "n_r": n_r,
            "tail_us": 0.0 if eta is None else eta.shape[0] * rp.DT, "seconds": time.time() - t0}
    return [{**meta, **r} for r in rows]


def b3_job(kind, label, tag, pulse, eta, k, n_max, n_c, keep, part, period_us=None, stark_tuned=True):
    """One B3 job; writes its rows to `part` (csv) and returns them."""
    t0 = time.time()
    kw = {"n_c": n_c, "n_r": 2, "keep": None if keep is None else set(keep)}
    if kind == "every_k":
        rows = rp.every_k(pulse, eta, k, n_max, **kw)
    elif kind == "back_to_back":
        rows = [{**r, "end": "bare", "n_tails": 0} for r in rp.after_n_gates(pulse, np.zeros((0, 2)), n_max, **kw)]
    elif kind == "idle":
        rows = [{**r, "end": "bare", "n_tails": 0}
                for r in rp.after_n_gates(pulse, np.zeros((0, 2)), n_max, gate="I", with_kerr_decode=True, **kw)]
    else:
        rows = [{**r, "end": "bare", "n_tails": 0}
                for r in rp.teeth_memory(n_max, period_us, with_kerr_decode=True, stark_tuned=stark_tuned, **kw)]
    meta = {"protocol": "B3" if kind in ("every_k", "back_to_back") else "memory", "label": label, "tag": tag,
            "system": "closed" if keep == CLOSED else "open", "n_c": n_c, "n_r": 2,
            "tail_us": 0.0 if eta is None else eta.shape[0] * rp.DT, "seconds": time.time() - t0}
    rows = [{**meta, "k": k, **r} for r in rows]
    pd.DataFrame(rows).to_csv(part, index=False)
    return rows


def run_every_k(a):
    if len(a.tails) != 1:
        raise SystemExit("--every-k takes exactly one --tails entry")
    label, tag = a.tails[0].split("=", 1)
    info, eta, pulse = load_tail(a.indir, tag)
    if info.get("target", "X") != "X":
        raise SystemExit("--every-k takes an X entry")
    T_g, T_t = pulse.shape[0] * rp.DT, eta.shape[0] * rp.DT
    n_mem = math.ceil(round(a.n_max * (T_g + T_t) / T_g, 9))         # memory sampled every gate length
    parts = os.path.join(a.out_dir, "B3_parts")
    os.makedirs(parts, exist_ok=True)
    if a.teeth_only:                                                # replace the teeth rows of the merged table
        rows = b3_job("teeth", TEETH, "", None, None, None, n_mem, a.n_c, None,
                      os.path.join(parts, "teeth_open.csv"), T_g)
        return replace_label_rows(os.path.join(a.out_dir, "fwm_tail_repeat_B3.csv"), pd.DataFrame(rows), TEETH)
    jobs = []
    for keep, sysname in ((None, "open"), (CLOSED, "closed")):
        for k in a.every_k:
            jobs.append(("every_k", label, tag, pulse, eta, k, a.n_max, a.n_c, keep, f"k{k}_{sysname}"))
        jobs.append(("back_to_back", f"{label}, back to back", tag, pulse, None, None, a.n_max, a.n_c, keep,
                     f"b2b_{sysname}"))
    jobs.append(("idle", "idle", "", np.zeros_like(pulse), None, None, n_mem, a.n_c, None, "idle_open"))
    jobs.append(("teeth", TEETH, "", None, None, None, n_mem, a.n_c, None, "teeth_open", T_g))
    jobs.sort(key=lambda j: j[0] == "back_to_back")                  # the short jobs last
    out, todo = [], []
    for j in jobs:
        part = os.path.join(parts, j[9] + ".csv")
        if os.path.exists(part):
            print(f"reuse {part}")
            out.append(pd.read_csv(part))
        else:
            todo.append(j[:9] + (part,) + j[10:])
    if todo:
        res = Parallel(n_jobs=a.n_jobs, backend="loky", verbose=5)(delayed(b3_job)(*j) for j in todo)
        out += [pd.DataFrame(r) for r in res]
    df = pd.concat(out, ignore_index=True)
    path = os.path.join(a.out_dir, "fwm_tail_repeat_B3.csv")
    df.to_csv(path, index=False)
    print(f"-> {path}")
    pd.set_option("display.width", 200)
    show = df[(df.protocol == "B3") & (df.end == "tail") & df.N.isin([1, 2, 5, 10, 20, 30, 40])]
    print(show.pivot_table(index="N", columns=["system", "k"], values="F_pro").to_string(float_format="%.4f"))


def matched_job(kind, label, tag, m, kept, pulse, eta, n_max, n_c, keep, part):
    t0 = time.time()
    kw = {"n_c": n_c, "n_r": 2, "keep": None if keep is None else set(keep)}
    if kind == "block":
        rows = [{**r, "end": "tail" if r["tail"] == "tail" else "bare"}
                for r in rp.after_n_gates(pulse, eta, m, **kw) if r["N"] == m]
    else:
        rows = rp.every_k(pulse, eta, m, n_max, closing_tail=False, **kw)
    meta = {"protocol": kind, "label": label, "tag": tag, "m": m, "kept": kept,
            "system": "closed" if keep == CLOSED else "open", "n_c": n_c, "n_r": 2,
            "tail_us": eta.shape[0] * rp.DT, "seconds": time.time() - t0}
    rows = [{**meta, **{k: v for k, v in r.items() if k != "tail"}} for r in rows]
    pd.DataFrame(rows).to_csv(part, index=False)
    return rows


def run_matched(a):
    entries = []
    for entry in a.tails:
        label, tag = entry.split("=", 1)
        info, eta, pulse = load_tail(a.indir, tag)
        m = int(info.get("n_gates", 1))
        if info.get("target", "X") != rp.target_of(m):
            raise SystemExit(f"{entry}: target {info.get('target')} is not X^{m}")
        entries.append((label, tag, m, info["F_pro"], pulse, eta))
    best = {}
    for label, tag, m, F, _, _ in entries:
        if m not in best or F > best[m][1]:
            best[m] = (tag, F)
    parts = os.path.join(a.out_dir, "matched_parts")
    os.makedirs(parts, exist_ok=True)
    jobs = []
    for label, tag, m, F, pulse, eta in entries:
        kept = best[m][0] == tag
        if kept:
            jobs.append(("every_k", label, tag, m, kept, pulse, eta, a.n_max, a.n_c, None, f"repeat_{tag}"))
        for keep, sysname in ((None, "open"), (CLOSED, "closed")):
            jobs.append(("block", label, tag, m, kept, pulse, eta, a.n_max, a.n_c, keep, f"block_{tag}_{sysname}"))
    out, todo = [], []
    for j in jobs:
        part = os.path.join(parts, j[-1] + ".csv")
        if os.path.exists(part):
            print(f"reuse {part}")
            out.append(pd.read_csv(part))
        else:
            todo.append(j[:-1] + (part,))
    if todo:
        res = Parallel(n_jobs=a.n_jobs, backend="loky", verbose=5)(delayed(matched_job)(*j) for j in todo)
        out += [pd.DataFrame(r) for r in res]
    df = pd.concat(out, ignore_index=True)
    path = os.path.join(a.out_dir, "fwm_tail_matched.csv")
    df.to_csv(path, index=False)
    print(f"-> {path}")
    pd.set_option("display.width", 200)
    blk = df[df.protocol == "block"]
    print(blk.pivot_table(index=["m", "tag"], columns=["system", "end"], values="F_pro").to_string(float_format="%.5f"))


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
            for lab, tuned in ((TEETH, True), (TEETH_BARE, False)):
                jobs.append(("teeth", lab, "", None, None, "I", a.n_max, a.n_c, 2, keep, T, True, "B2", tuned))
                jobs.append(("teeth", lab, "", None, None, "I", a.n_max, a.n_c, 2, keep, T_gate, True, "B1", tuned))
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


def run_teeth_only(a):
    """The main mode's B2 teeth jobs alone (one per gate + tail period, open and closed), replacing the TEETH rows of
    <out-dir>/fwm_tail_repeat_B2.csv."""
    periods = set()
    for entry in a.tails:
        info, eta, pulse = load_tail(a.indir, entry.split("=", 1)[1])
        if info.get("target", "X") != "I":
            periods.add(round((pulse.shape[0] + eta.shape[0]) * rp.DT, 9))
    jobs = [("teeth", TEETH, "", None, None, "I", a.n_max, a.n_c, 2, keep, T)
            for T in sorted(periods) for keep in (None, CLOSED)]
    out = Parallel(n_jobs=a.n_jobs, backend="loky", verbose=5)(delayed(job)(*j) for j in jobs)
    new = pd.DataFrame([r for rows in out for r in rows])
    if (new["decode"] != "none").any():
        raise SystemExit("B2 teeth rows are plain (undecoded) only")
    new = new.drop(columns="decode")                       # the B2 table predates the decode column; all "none"
    replace_label_rows(os.path.join(a.out_dir, "fwm_tail_repeat_B2.csv"), new, TEETH)


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
    p.add_argument("--every-k", type=int, nargs="+", default=None,
                   help="B3 only: a tail after every k-th gate, for each k given")
    p.add_argument("--teeth-only", action="store_true",
                   help="main mode or --every-k: only the Stark-tuned teeth, replacing the teeth rows of the existing table")
    p.add_argument("--matched", action="store_true",
                   help="tails trained after m gates: m gates + tail, and B3 with k = m for the kept start")
    a = p.parse_args()
    a.out_dir = os.path.expanduser(a.out_dir)
    if a.every_k:
        return run_every_k(a)
    if a.matched:
        return run_matched(a)
    if a.kerr_decode:
        return run_kerr_decode(a)

    if a.teeth_only:
        return run_teeth_only(a)
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
            jobs.append(("teeth", TEETH, "", None, None, "I", a.n_max, a.n_c, 2, keep, T))
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
