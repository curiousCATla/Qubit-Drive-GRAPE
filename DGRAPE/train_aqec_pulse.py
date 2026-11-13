#!/usr/bin/env python3
"""
Train a real AQEC pulse (Cornell step 1, Eqs. C9-C10, odd kitten code) on Shirol's chip in the closed system
(DGRAPE/aqec_pulse.py, notebooks/AQEC_compare.ipynb §5).

    python3 DGRAPE/train_aqec_pulse.py --duration-us 1.0 --dry-run      # one timed evaluation
    python3 DGRAPE/train_aqec_pulse.py --duration-us 1.0 --seed 42      # tag aqec_T10_s42

Writes to --outdir (default ~/fwm_runs/aqec, outside iCloud): u_<tag>.npy (physical pulse (N, 4), dt 1 ns),
x_<tag>.npy (pre-image; resume with --init-x), <tag>.json (settings, closed truncation scan), and the best-so-far
checkpoint u_/x_<tag>_ckpt.npy every 100 iterations. Copy the results into pulses/dgrape/ when done.
"""

import argparse
import json
import os
import sys
import time

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np                                    # noqa: E402

from DGRAPE import aqec_pulse as ap                   # noqa: E402
from DGRAPE import est_gate as eg                     # noqa: E402

DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "fwm_runs", "aqec")


def _dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def tag_of(duration_us, seed):
    return f"aqec_T{int(round(duration_us * 10)):02d}_s{seed}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration-us", type=float, required=True)
    p.add_argument("--seed", type=int, default=ap.SEED)
    p.add_argument("--maxiter", type=int, default=4000)
    p.add_argument("--n-c", type=int, default=ap.N_C_TRAIN)
    p.add_argument("--init-x", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    N = ap.n_steps(a.duration_us)
    if a.dry_run:
        obj = ap.AqecObjective(a.n_c, N)
        x0 = eg.start_preimage(a.seed, N)
        t0 = time.time()
        cost, g, terms = obj.evaluate(x0)
        print(json.dumps({"N": N, "cost": cost, "grad_norm": float(np.linalg.norm(g)), "eval_s": time.time() - t0,
                          **terms}, indent=1, default=float), flush=True)
        return
    tag = a.tag or tag_of(a.duration_us, a.seed)
    os.makedirs(a.outdir, exist_ok=True)

    def ckpt(u, x, terms):
        np.save(os.path.join(a.outdir, f"u_{tag}_ckpt.npy"), u)
        np.save(os.path.join(a.outdir, f"x_{tag}_ckpt.npy"), x)
        _dump(os.path.join(a.outdir, f"{tag}_ckpt.json"), terms)

    init_x = None if a.init_x is None else np.load(a.init_x)
    u, x, info = ap.train(a.duration_us, a.seed, a.maxiter, a.n_c, init_x=init_x, checkpoint=ckpt)
    info["tag"] = tag
    info["init_x_path"] = a.init_x
    info["trunc_scan"] = {str(k): v for k, v in ap.truncation_scan(u).items()}
    np.save(os.path.join(a.outdir, f"u_{tag}.npy"), u)
    np.save(os.path.join(a.outdir, f"x_{tag}.npy"), x)
    _dump(os.path.join(a.outdir, f"{tag}.json"), info)
    print(f"{tag}: F {info['F_train']:.6f}  trunc {info['trunc_scan']}  max|eps| "
          f"{info['max_drive_modulus']['cav']:.2f}/{info['max_drive_modulus']['tra']:.2f}  "
          f"pinned {info['frac_preimage_pinned']:.2%}", flush=True)


if __name__ == "__main__":
    main()
