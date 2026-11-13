#!/usr/bin/env python3
"""
Forward-only pre-scan of Gertler-style tails (notebooks/FWM_tails_ge.ipynb §2): for EsT and Ord, every tail length
and mixing-comb cap, the fixed starts of fwm_tail_ge.start_library (gertler: Gertler's continuous teeth;
sequential: selective pi, then the swap), raw and through the training chain, plus the gate alone and
gate + idle. No training. Writes tables/fwm_tail_ge_prescan.csv.

    OPENBLAS_NUM_THREADS=1 python DGRAPE/fwm_tail_ge_prescan.py --n-jobs 8        (a few minutes)
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

from DGRAPE import fwm_tail_ge as ge                  # noqa: E402


def idle_rows(gate, tail_us):
    F0 = ge.score(np.zeros((0, 4)), gate)["F_pro"]
    r = ge.score(np.zeros((ge.n_steps(tail_us), 4)), gate)
    return [{"gate": gate, "start": "idle", "kind": "raw", "tail_us": tail_us, "cap_M": 0.0, "F_alone": F0,
             "F_pro": r["F_pro"], "gain": r["F_pro"] - F0, "P_e_end": r["P_e_end"], "n_r_end": r["n_r_end"],
             "max_xi_M": 0.0, "max_eps_T_MHz": 0.0}]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gates", nargs="+", default=["EsT", "Ord"])
    p.add_argument("--tail-us", type=float, nargs="+", default=list(ge.TAIL_US_GRID))
    p.add_argument("--caps", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.3])
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "tables", "fwm_tail_ge_prescan.csv"))
    a = p.parse_args()
    jobs = [(idle_rows, (g, T)) for g in a.gates for T in a.tail_us]
    jobs += [(ge.prescan_rows, (g, T, c)) for g in a.gates for T in a.tail_us for c in a.caps]
    out = Parallel(n_jobs=a.n_jobs, backend="loky", verbose=5)(delayed(f)(*args) for f, args in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
