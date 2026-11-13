#!/usr/bin/env python3
"""
Score the trained AQEC pulses (DGRAPE/train_aqec_pulse.py) for notebooks/AQEC_compare.ipynb §5 (closed-trained) and §6 (--family open). From the repo root:

    python3 DGRAPE/aqec_pulse_score.py runs                 # closed results of every run -> tables/aqec_pulse_runs.csv
    python3 DGRAPE/aqec_pulse_score.py single               # N = 1 after the EsT gate, decay set and all channels
                                                            #   -> tables/aqec_pulse_single.csv
    python3 DGRAPE/aqec_pulse_score.py channels             # the step alone on the code words, one decay channel at a
                                                            #   time -> tables/aqec_pulse_channels.csv
    python3 DGRAPE/aqec_pulse_score.py B1 --n-max 20        # -> tables/aqec_pulse_B1.csv
    python3 DGRAPE/aqec_pulse_score.py B2 --n-max 20        # -> tables/aqec_pulse_B2.csv

Pulses are read from pulses/dgrape/u_aqec_T*_s*.npy (with their .json); "single", "B1" and "B2" use the best seed per
duration (highest closed F at n_c 24), or --tags.

--family open scores the open-system-trained pulses instead (DGRAPE/train_aqec_open.py, pulses/dgrape/u_aqec_open_*.npy)
-> tables/aqec_open_*.csv; there "single", "channels", "B1" and "B2" use every run unless --tags is given.
"""

import argparse
import glob
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402

np.seterr(divide="ignore", over="ignore", invalid="ignore")

from DGRAPE import aqec_ideal as aq   # noqa: E402
from DGRAPE import aqec_pulse as ap   # noqa: E402

PULSE_DIR = os.path.join(REPO_ROOT, "pulses", "dgrape")


PATTERN = {"closed": "aqec_T*_s*.json", "open": "aqec_open_T*.json"}


def runs_table(family="closed"):
    rows = []
    for js in sorted(glob.glob(os.path.join(PULSE_DIR, PATTERN[family]))):
        info = json.load(open(js))
        r = {"tag": info["tag"], "duration_us": info["duration_us"], "seed": info["seed"], "nit": info["nit"],
             "minutes": info["seconds"] / 60, "message": info["message"], "F_train_nc20": info["F_train"],
             **({"F_start_nc20": info["F_start"], "start": "closed pulse" if info.get("init_x_path") else "cold",
                 "best_it": info["best_it"]} if family == "open" else {}),
             "max_eps_cav": info["max_drive_modulus"]["cav"], "max_eps_tra": info["max_drive_modulus"]["tra"],
             "frac_pinned": info["frac_preimage_pinned"]}
        for n, v in info["trunc_scan"].items():
            for k, val in v.items():
                r[f"{k}_nc{n}"] = val
        rows.append(r)
    return pd.DataFrame(rows)


def best_tags(runs):
    return list(runs.sort_values("F_nc24", ascending=False).groupby("duration_us").head(1)
                .sort_values("duration_us").tag)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["runs", "single", "channels", "B1", "B2"])
    p.add_argument("--tags", nargs="*", default=None)
    p.add_argument("--n-max", type=int, default=20)
    p.add_argument("--n-c", type=int, default=24)
    p.add_argument("--family", choices=["closed", "open"], default="closed")
    a = p.parse_args()
    stem = "aqec_pulse" if a.family == "closed" else "aqec_open"
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.0f} s] {s}", flush=True)  # noqa: E731
    runs = runs_table(a.family)
    if a.mode == "runs":
        runs.to_csv(os.path.join(REPO_ROOT, "tables", f"{stem}_runs.csv"), index=False)
        print(runs.to_string())
        return
    tags = a.tags or (best_tags(runs) if a.family == "closed" else list(runs.tag))
    pulses = {t: np.load(os.path.join(PULSE_DIR, f"u_{t}.npy")) for t in tags}
    gate = np.load(os.path.join(REPO_ROOT, aq.EST_PULSE))
    log(f"pulses {tags}")
    if a.mode == "single":
        rows = []
        for ch in ("decay", "all"):
            rows += ap.single_rows(gate, pulses, a.n_c, ch)
            log(f"single {ch}")
        df = pd.DataFrame(rows)
    elif a.mode == "channels":
        df = pd.DataFrame(ap.channel_rows(pulses, a.n_c))
    else:
        fn = ap.b1_rows if a.mode == "B1" else ap.b2_rows
        df = pd.DataFrame(fn(gate, pulses, a.n_max, a.n_c, log=log))
    df.assign(n_c=a.n_c).to_csv(os.path.join(REPO_ROOT, "tables", f"{stem}_{a.mode}.csv"), index=False)
    log("done")


if __name__ == "__main__":
    main()
