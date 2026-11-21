#!/usr/bin/env python3
"""
The 0.8 us tails of notebooks/tail_experiments.ipynb with NO gate in front (§7 there; the same test as
notebooks/FWM_tails.ipynb §6.6). Forward only; DGRAPE/fwm_tail.py is used unchanged.

Tails: seed 8 of the Shirol-set noise-start sweep (X_tail_estg_T08_shirol_d01_s8, derivative weight / 10), the
decay-trained tail of FWM_tails.ipynb (X_tail_estg_T08_t2), and a 0.8 us idle tail. Each acts on the code's process
inputs ("code words") and on the clean one-loss error words a E a'/3 ("one-loss error words"), scored against the
logical identity (fwm_tail.code_word_scores), under:

    closed            no jumps
    decay set         noise "dgrape", every channel (a, r, s_ge, s_ef): what FWM_tails §6.6 calls "open"
    Shirol full set   noise "shirol", every channel
    one Shirol group at a time: photon loss {a}, readout decay {r}, transmon decay {s_ge, s_ef},
        transmon heating {up_ge, up_ef}, storage heating {up_a}, transmon dephasing {phi_e, phi_f},
        storage dephasing {phi_a}   (DGRAPE/tail_experiments_channels.py's groups, "other decay" split in two)

n_c 16, two readout levels, readout traced. The decay-tail rows under closed / decay set equal
tables/fwm_tail_code_lengths.csv (tag X_tail_estg_T08_t2, systems closed / open) bit for bit.

    python3 DGRAPE/tail_experiments_code.py --n-jobs 8          -> tables/tail_experiments_code.csv (~1 min)
"""

import argparse
import os
import sys

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from joblib import Parallel, delayed                  # noqa: E402

from DGRAPE import fwm_tail as ft                     # noqa: E402

PULSE_DIR = os.path.join(REPO_ROOT, "pulses", "dgrape")
OUT = os.path.join(REPO_ROOT, "tables", "tail_experiments_code.csv")
TAIL_US = 0.8
TAILS = {"seed 8": "X_tail_estg_T08_shirol_d01_s8", "decay t2": "X_tail_estg_T08_t2", "idle": None}
SYSTEMS = {                                           # name -> (noise, keep)
    "closed": ("dgrape", frozenset()),
    "decay set": ("dgrape", None),
    "Shirol full set": ("shirol", None),
    "photon loss": ("shirol", frozenset({"a"})),
    "readout decay": ("shirol", frozenset({"r"})),
    "transmon decay": ("shirol", frozenset({"s_ge", "s_ef"})),
    "transmon heating": ("shirol", frozenset({"up_ge", "up_ef"})),
    "storage heating": ("shirol", frozenset({"up_a"})),
    "transmon dephasing": ("shirol", frozenset({"phi_e", "phi_f"})),
    "storage dephasing": ("shirol", frozenset({"phi_a"})),
}
N_C, N_R = 16, 2


def load_eta(tag):
    if tag is None:
        return np.zeros((ft.n_steps(TAIL_US), 2), dtype=complex)
    return np.load(os.path.join(PULSE_DIR, f"eta_{tag}.npy"))


def score_one(tail, tag, system):
    noise, keep = SYSTEMS[system]
    return [{"tail": tail, "tag": tag or "idle", "system": system, **r}
            for r in ft.code_word_scores(load_eta(tag), N_C, N_R, keep=keep, noise=noise)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    jobs = [(t, tag, sy) for t, tag in TAILS.items() for sy in SYSTEMS]
    res = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(score_one)(*j) for j in jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    df["kitten_limit"] = ft.kitten_repair_limit()
    df.to_csv(a.out, index=False)
    print(df.pivot_table(index="system", columns=["tail", "input"], values="F_vs_I")
          .loc[list(SYSTEMS)].to_string(float_format="%.5f"))


if __name__ == "__main__":
    main()
