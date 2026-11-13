#!/usr/bin/env python3
"""
The EsT X tails with NO gate in front, at every trained length (notebooks/FWM_tails.ipynb §6.6). Forward only;
DGRAPE/fwm_tail.py is used unchanged.

Does a tail fix damage without changing the logical information? Each tail X_tail_estg_T{05,08,1,14,2}_{t2,z}
(trained after the EsT X gate, seed 24) and an idle tail of each length act on two ideal inputs
(fwm_tail.code_word_inputs): the code's process inputs ("code words") and the clean one-loss error words a E a'
("one-loss error words", normalised by <n> = 3). Both are scored against the logical identity, with and without a
free logical Z rotation, closed (no jumps), open (decay set) and with one decay channel at a time (photon loss a,
transmon decay s_ge + s_ef, readout decay r). n_c 16, two readout levels, readout traced.

The 50/50 mixture of the two inputs is not computed: process fidelity is linear in the input map and both error
words have the same norm, so F(50/50) = (F_code + F_err) / 2 exactly (validation/test_fwm_tail.py, CodeWords).

    python3 DGRAPE/fwm_tail_code.py --n-jobs 8          -> tables/fwm_tail_code_lengths.csv (~2 min)
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
OUT = os.path.join(REPO_ROOT, "tables", "fwm_tail_code_lengths.csv")
LENGTHS = {0.5: "05", 0.8: "08", 1.0: "1", 1.4: "14", 2.0: "2"}       # us -> tag part (notebook T_TAG)
STARTS = ("t2", "z")
SYSTEMS = {"closed": frozenset(), "open": None,                        # keep: no jumps / the decay set
           "photon loss only": frozenset({"a"}),                       # one channel at a time: where the
           "transmon decay only": frozenset({"s_ge", "s_ef"}),         # open-system cost comes from
           "readout decay only": frozenset({"r"})}
N_C, N_R = 16, 2


def tail_jobs():
    """(tail_us, tail, start, tag, eta) for every EsT X tail and an idle tail of each length."""
    jobs = []
    for T, tt in LENGTHS.items():
        jobs.append((T, "idle", "", "idle", np.zeros((ft.n_steps(T), 2), dtype=complex)))
        for s in STARTS:
            tag = f"X_tail_estg_T{tt}_{s}"
            jobs.append((T, "X tail", s, tag, np.load(os.path.join(PULSE_DIR, f"eta_{tag}.npy"))))
    return jobs


def score_one(T, tail, start, tag, eta, system):
    return [{"tail_us": T, "tail": tail, "start": start, "tag": tag, "system": system, **r}
            for r in ft.code_word_scores(eta, N_C, N_R, keep=SYSTEMS[system])]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    jobs = [(*j, sy) for j in tail_jobs() for sy in SYSTEMS]
    res = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(score_one)(*j) for j in jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    df["kitten_limit"] = ft.kitten_repair_limit()
    df.to_csv(a.out, index=False)
    print(df.to_string(float_format="%.5f"))


if __name__ == "__main__":
    main()
