"""
Run the idealised Cornell AQEC (DGRAPE/aqec_ideal.py) after the EsT X gate on Shirol's chip and write the tables of
notebooks/AQEC_compare.ipynb. From the repo root:

    python3 DGRAPE/aqec_ideal_run.py single          # N = 1, decay set and all channels -> tables/aqec_ideal_single.csv
                                                     #   and tables/aqec_ideal_stages.csv
    python3 DGRAPE/aqec_ideal_run.py B1 --n-max 20   # N gates + one AQEC   -> tables/aqec_ideal_B1.csv
    python3 DGRAPE/aqec_ideal_run.py B2 --n-max 20   # N (gate + AQEC) pairs -> tables/aqec_ideal_B2.csv

n_c 24, n_r 1 (exact: the readout is never populated). Minutes each on an idle Mac; the three modes can run side by side.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
np.seterr(divide="ignore", over="ignore", invalid="ignore")

from DGRAPE import aqec_ideal as aq   # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["single", "B1", "B2"])
    p.add_argument("--n-max", type=int, default=20)
    p.add_argument("--n-c", type=int, default=aq.N_C)
    p.add_argument("--pulse", default=aq.EST_PULSE)
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "tables"))
    a = p.parse_args()
    u = np.load(os.path.join(REPO_ROOT, a.pulse))
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.0f} s] {s}", flush=True)  # noqa: E731
    meta = {"gate_pulse": a.pulse, "n_c": a.n_c, "n_r": 1, "tau1_us": aq.TAU1, "tau2_us": aq.TAU2}
    if a.mode == "single":
        rows, stages = [], []
        for ch in ("decay", "all"):
            rows += aq.single_summary(u, a.n_c, channels=ch)
            stages += aq.single_stages(u, a.n_c, channels=ch)
            log(f"single {ch}")
        pd.DataFrame(rows).assign(**meta).to_csv(os.path.join(a.out_dir, "aqec_ideal_single.csv"), index=False)
        pd.DataFrame(stages).assign(**meta).to_csv(os.path.join(a.out_dir, "aqec_ideal_stages.csv"), index=False)
    else:
        fn = aq.after_n_gates if a.mode == "B1" else aq.pairs
        rows = fn(u, a.n_max, a.n_c, log=log)
        pd.DataFrame(rows).assign(**meta, channels="decay").to_csv(
            os.path.join(a.out_dir, f"aqec_ideal_{a.mode}.csv"), index=False)
    log("done")


if __name__ == "__main__":
    main()
