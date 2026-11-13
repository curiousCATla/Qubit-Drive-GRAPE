#!/usr/bin/env python3
"""
Train the real AQEC step in the OPEN system (DGRAPE/aqec_open.py, notebooks/AQEC_compare.ipynb §6): EsT gate output ->
pulse -> 1.2 us readout window -> reset, cost 1 - F_pro (decay set) + drive cap + Fock >= 16 guard.

    python3 DGRAPE/train_aqec_open.py --duration-us 2.0 --dry-run          # one timed evaluation
    python3 DGRAPE/train_aqec_open.py --duration-us 2.0 --start closed      # warm: the closed pulse x_aqec_T20_s42
    python3 DGRAPE/train_aqec_open.py --duration-us 2.0 --start cold --seed 42
    python3 DGRAPE/train_aqec_open.py --duration-us 2.0 --init-x path/x_....npy --tag ...   # resume

Tags: aqec_open_T20_w (warm from the closed pulse), aqec_open_T20_s42 (cold). Writes to --outdir (default
~/fwm_runs/aqec_open, outside iCloud): u_/x_<tag>.npy, <tag>.json (settings, history, open + closed truncation scan)
and the best-so-far checkpoint u_/x_<tag>_ckpt.npy every 25 iterations. Copy results into pulses/dgrape/ when done.
"""

import argparse
import json
import os
import sys
import time

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np                                    # noqa: E402

np.seterr(divide="ignore", over="ignore", invalid="ignore")

from DGRAPE import aqec_open as ao                    # noqa: E402
from DGRAPE import aqec_pulse as ap                   # noqa: E402
from DGRAPE import est_gate as eg                     # noqa: E402

DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "fwm_runs", "aqec_open")
PULSE_DIR = os.path.join(REPO_ROOT, "pulses", "dgrape")


def _dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration-us", type=float, required=True)
    p.add_argument("--start", choices=["closed", "cold"], default="closed")
    p.add_argument("--seed", type=int, default=eg.SEED)
    p.add_argument("--init-x", default=None, help="pre-image to start from (overrides --start)")
    p.add_argument("--maxiter", type=int, default=1000)
    p.add_argument("--n-c", type=int, default=ao.N_C_TRAIN)
    p.add_argument("--n-jobs", type=int, default=3)
    p.add_argument("--lambda-forbid", type=float, default=ao.LAMBDA_FORBID)
    p.add_argument("--channels", choices=["decay", "all"], default="decay")
    p.add_argument("--tag", default=None)
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    N = ap.n_steps(a.duration_us)
    T10 = int(round(a.duration_us * 10))
    if a.init_x is not None:
        init_path = a.init_x
    elif a.start == "closed":
        init_path = os.path.join(PULSE_DIR, f"x_{ao.CLOSED_START[a.duration_us]}.npy")
    else:
        init_path = None
    init_x = None if init_path is None else np.load(init_path)
    if a.dry_run:
        obj = ao.OpenAqecObjective(N, a.n_c, channels=a.channels, lambda_forbid=a.lambda_forbid, n_jobs=a.n_jobs)
        x0 = eg.start_preimage(a.seed, N) if init_x is None else init_x
        obj.evaluate(x0)
        obj._cache_x = None
        t0 = time.time()
        cost, g, terms = obj.evaluate(x0)
        obj.close()
        print(json.dumps({"N": N, "start": init_path or f"seed {a.seed}", "cost": cost,
                          "grad_norm": float(np.linalg.norm(g)), "eval_s": time.time() - t0, **terms},
                         indent=1, default=float), flush=True)
        return
    tag = a.tag or (f"aqec_open_T{T10:02d}_w" if init_path is not None else f"aqec_open_T{T10:02d}_s{a.seed}")
    os.makedirs(a.outdir, exist_ok=True)

    def ckpt(u, x, terms):
        np.save(os.path.join(a.outdir, f"u_{tag}_ckpt.npy"), u)
        np.save(os.path.join(a.outdir, f"x_{tag}_ckpt.npy"), x)
        _dump(os.path.join(a.outdir, f"{tag}_ckpt.json"), terms)

    u, x, info = ao.train(a.duration_us, init_x=init_x, seed=a.seed, maxiter=a.maxiter, n_c=a.n_c,
                          channels=a.channels, lambda_forbid=a.lambda_forbid, n_jobs=a.n_jobs, checkpoint=ckpt)
    info.update(tag=tag, init_x_path=init_path)
    np.save(os.path.join(a.outdir, f"u_{tag}.npy"), u)
    np.save(os.path.join(a.outdir, f"x_{tag}.npy"), x)
    _dump(os.path.join(a.outdir, f"{tag}.json"), info)
    info["trunc_scan"] = {str(k): v for k, v in ao.truncation_scan(u).items()}
    _dump(os.path.join(a.outdir, f"{tag}.json"), info)
    print(f"{tag}: F {info['F_start']:.6f} -> {info['F_train']:.6f}  trunc {info['trunc_scan']}  "
          f"pinned {info['frac_preimage_pinned']:.2%}", flush=True)


if __name__ == "__main__":
    main()
