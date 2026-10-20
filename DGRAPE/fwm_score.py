#!/usr/bin/env python3
"""
Held-out scoring of Stage 2 FWM pulses (fwm_grape.ipynb §6.5 and Table 11, §7.5). Forward only.

Each pulse's tone set comes from its <tag>.json ("paths"; (1, 3, 5) when the key is missing, as for
the six-tone X_fwm_s2_* runs). All pulses in one call must share a tone set, which the tones-off
reference "off" then uses. 18-tone results go to their own table:

    python DGRAPE/fwm_score.py --tags X_fwm_s2full_a ... --out tables/fwm_stage2_full_heldout.csv

Every pulse (b_<tag>.npy with the frozen d2 drives), and the tones-off reference "off", is scored
under the training condition and under conditions it was not trained on:

    train        n_c 16, CF4 S 2, decay set, mixing off (the training cost's fidelity)
    S4, S8       sub-step error                 nc18, nc20   truncation
    nr3          a second readout photon        all_ch       dephasing + heating (AQEC-style)
    g/sqrt2      coupling convention            beta x0.9, x1.1   pump miscalibration
    stark_off    reliance on the Stark model    mix1, mix2   pump-mixing a'r' term on
    closed, loss_only, t1_only                  channel split (Table 1 with tones on)

    python DGRAPE/fwm_score.py --tags X_fwm_s2_a X_fwm_s2_b --indir ~/fwm_runs --out tables/fwm_stage2_heldout.csv
"""

import argparse
import json
import os
import sys

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from joblib import Parallel, delayed                  # noqa: E402

from DGRAPE import fwm                                # noqa: E402

DT = 0.002
U_D2 = os.path.join(REPO_ROOT, "pulses", "dgrape", "u_X_open_d2.npy")

CONDITIONS = {
    "train": {},
    "S4": {"S": 4},
    "S8": {"S": 8},
    "nc18": {"n_c": 18},
    "nc20": {"n_c": 20},
    "nr3": {"n_r": 3},
    "all_ch": {"dephasing": True, "heating": True},
    "g/sqrt2": {"g_scale": 1 / np.sqrt(2)},
    "beta_x0.9": {"beta_scale": 0.9},
    "beta_x1.1": {"beta_scale": 1.1},
    "stark_off": {"stark": False},
    "mix1": {"mixing": 1.0},
    "mix2": {"mixing": 2.0},
    "closed": {"keep": set()},
    "loss_only": {"keep": {"a"}},
    "t1_only": {"keep": {"s_ge", "s_ef"}},
}


def _paths(indir, tag):
    path = os.path.join(indir, f"{tag}.json")
    if not os.path.exists(path):
        return tuple(fwm.dev.CORRECTED_FOCK)
    with open(path) as f:
        return tuple(json.load(f).get("paths", fwm.dev.CORRECTED_FOCK))


def run(tag, beta, cond, u, paths=fwm.dev.CORRECTED_FOCK):
    kw = dict(n_c=16, n_r=2, S=fwm.S_TRAIN, paths=paths)
    kw.update({k: v for k, v in CONDITIONS[cond].items() if k != "beta_scale"})
    b = beta * CONDITIONS[cond].get("beta_scale", 1.0)
    res = fwm.score(u, b, DT, occupations=(cond == "train"), **kw)
    return {"tag": tag, "condition": cond, **res}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--indir", default=os.path.join(os.path.expanduser("~"), "fwm_runs"))
    p.add_argument("--out", required=True)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--n-jobs", type=int, default=14)
    a = p.parse_args()
    u = np.load(U_D2)
    sets = {_paths(a.indir, t) for t in a.tags}
    if len(sets) != 1:
        raise SystemExit(f"tags mix tone sets {sorted(sets)}; score each set into its own table")
    paths = sets.pop()
    pulses = {"off": np.zeros((u.shape[0], 2 * len(paths)), dtype=complex)}
    for t in a.tags:
        pulses[t] = np.load(os.path.join(a.indir, f"b_{t}.npy"))
        if pulses[t].shape[1] != 2 * len(paths):
            raise SystemExit(f"b_{t}.npy has {pulses[t].shape[1]} tones, paths {paths} need {2 * len(paths)}")
    jobs = [delayed(run)(t, b, c, u, paths) for t, b in pulses.items() for c in a.conditions]
    rows = Parallel(n_jobs=a.n_jobs, backend="loky", verbose=5)(jobs)
    df = pd.DataFrame(rows)
    off = df[df.tag == "off"].set_index("condition")["F_pro"]
    df["gain_vs_off"] = df.F_pro - df.condition.map(off)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 220)
    print(df.pivot(index="condition", columns="tag", values="F_pro").to_string())


if __name__ == "__main__":
    main()
