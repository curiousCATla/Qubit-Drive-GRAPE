#!/usr/bin/env python3
"""
Channel breakdown for notebooks/tail_experiments.ipynb (forward only).

The EsT-good X gate (u_X_est_err_closed_s24.npy, run at n_c 24, cut to 16) alone and followed by 0.8 us FWM tails
(decay-trained and Shirol-trained), scored with Shirol's full channel set (noise="shirol",
DGRAPE.model.shirol_collapse_ops) restricted to ONE group of jump operators at a time:

    closed           no jumps
    photon loss      a
    other decay      r, s_ge, s_ef        (readout decay, transmon T1)
    transmon heat    up_ge, up_ef         (thermal + drive-induced, as Shirol App. D)
    storage heat     up_a
    transmon deph    phi_e, phi_f
    storage deph     phi_a                (the residual 0.23 ms^-1)
    all              every channel (= the "shirol" condition of fwm_tail_score.py)

A group's cost is F(closed) - F(group only); its effect on the tail is gain(group only) - gain(closed).

    python DGRAPE/tail_experiments_channels.py --tags X_tail_estg_T08_t2 X_tail_estg_T08_shirol_t2 ... \\
        --indir pulses/dgrape --out tables/tail_experiments_channels.csv --n-jobs 8
"""

import argparse
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from joblib import Parallel, delayed                  # noqa: E402

from DGRAPE import fwm_tail as ft                     # noqa: E402

GATE = os.path.join("pulses", "dgrape", "u_X_est_err_closed_s24.npy")
GROUPS = {
    "closed": set(),
    "photon loss": {"a"},
    "other decay": {"r", "s_ge", "s_ef"},
    "transmon heating": {"up_ge", "up_ef"},
    "storage heating": {"up_a"},
    "transmon dephasing": {"phi_e", "phi_f"},
    "storage dephasing": {"phi_a"},
    "all": None,
}


def run(tag, eta, group):
    pulse = np.load(os.path.join(REPO_ROOT, GATE))
    r = ft.score(eta, n_c=16, gate_n_c=24, pulse=pulse, noise="shirol", keep=GROUPS[group])
    return {"tag": tag, "group": group, "F_pro": r["F_pro"], "P_e_end": r["P_e_end"], "P_f_end": r["P_f_end"],
            "n_r_end": r["n_r_end"]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--indir", default=os.path.join("pulses", "dgrape"))
    p.add_argument("--out", required=True)
    p.add_argument("--n-jobs", type=int, default=8)
    a = p.parse_args()
    etas = {"alone": np.zeros((0, 2), dtype=complex)}
    etas.update({t: np.load(os.path.join(a.indir, f"eta_{t}.npy")) for t in a.tags})
    rows = Parallel(n_jobs=a.n_jobs, backend="loky", verbose=5)(
        delayed(run)(t, e, g) for t, e in etas.items() for g in GROUPS)
    df = pd.DataFrame(rows)
    alone = df[df.tag == "alone"].set_index("group")["F_pro"]
    df["gain_vs_alone"] = df.F_pro - df.group.map(alone)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.pivot(index="group", columns="tag", values="gain_vs_alone").to_string(float_format="%+.5f"))


if __name__ == "__main__":
    main()
