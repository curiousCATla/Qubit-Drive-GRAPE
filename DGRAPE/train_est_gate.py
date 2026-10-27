#!/usr/bin/env python3
"""
Train an EsT (or Ord control) X gate on the Shirol chip in the closed system (DGRAPE/est_gate.py,
FWM_tails.ipynb §3): EST's two-stage weights, d2's chip, code, duration, drive limits and start.

    python3 DGRAPE/train_est_gate.py --variant est --dry-run          # one timed evaluation
    python3 DGRAPE/train_est_gate.py --variant est --maxiter 2000     # tag X_est_closed
    python3 DGRAPE/train_est_gate.py --variant ord --maxiter 2000     # tag X_ord_closed

Writes to --outdir (default ~/fwm_runs, outside iCloud): u_<tag>.npy (physical pulse, (N, 4)),
x_<tag>.npy (pre-image; resume with --init-x), the same per stage (_stage1, _stage2) and <tag>.json
(settings, per-stage terms, tier, closed truncation scan). Copy them into pulses/dgrape/ when done
(never pulses/ or pulses/est/).
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

from DGRAPE import est_gate as eg                     # noqa: E402

DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "fwm_runs")


def _dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def dry_run(variant, seed, n_c):
    x0 = eg.start_preimage(seed)
    t0 = time.time()
    obj = eg.EstObjective(eg.stage_weights(variant)[0], n_c)
    t_build = time.time() - t0
    t0 = time.time()
    cost, g, terms = obj.evaluate(x0)
    out = {"variant": variant, "n_c": n_c, "N": obj.N, "cost": cost, "grad_norm": float(np.linalg.norm(g)),
           "build_s": t_build, "eval_s": time.time() - t0, **terms}
    print(json.dumps(out, indent=1, default=float), flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=sorted(eg.SCHEDULES), default="est")
    p.add_argument("--tag", default=None, help="default X_<variant>_closed")
    p.add_argument("--maxiter", type=int, default=2000, help="per stage")
    p.add_argument("--seed", type=int, default=eg.SEED)
    p.add_argument("--n-c", type=int, default=eg.N_C_TRAIN)
    p.add_argument("--init-x", default=None, help="pre-image .npy to start from instead of d2's start")
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--dry-run", action="store_true", help="one timed cost + gradient evaluation, no training")
    a = p.parse_args()
    if a.dry_run:
        dry_run(a.variant, a.seed, a.n_c)
        return
    tag = a.tag or f"{eg.GATE}_{a.variant}_closed"
    os.makedirs(a.outdir, exist_ok=True)

    def hook(i, u, x, terms):
        np.save(os.path.join(a.outdir, f"u_{tag}_stage{i}.npy"), u)
        np.save(os.path.join(a.outdir, f"x_{tag}_stage{i}.npy"), x)
        _dump(os.path.join(a.outdir, f"{tag}_stage{i}.json"), terms)

    init_x = None if a.init_x is None else np.load(a.init_x)
    u, x, info = eg.train(a.variant, a.maxiter, a.seed, a.n_c, init_x, stage_hook=hook)
    info["tag"] = tag
    info["trunc_scan"] = {str(k): v for k, v in eg.truncation_scan(u).items()}
    np.save(os.path.join(a.outdir, f"u_{tag}.npy"), u)
    np.save(os.path.join(a.outdir, f"x_{tag}.npy"), x)
    _dump(os.path.join(a.outdir, f"{tag}.json"), info)
    print(f"{tag}: F1 {info['F1']:.6f}  F_ET {info['F_ET']:.4f}  tier {info['tier']}  "
          f"max|eps| {info['max_drive_modulus']['cav']:.2f}/{info['max_drive_modulus']['tra']:.2f}  "
          f"pinned {info['frac_preimage_pinned']:.2%}  trunc {info['trunc_scan']}", flush=True)


if __name__ == "__main__":
    main()
