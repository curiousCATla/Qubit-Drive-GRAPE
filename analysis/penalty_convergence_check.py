#!/usr/bin/env python3
"""
analysis/penalty_convergence_check.py

Did the penalty sweep's pulses actually converge, or just run out of iterations?

Every run in `analysis/penalty_sweep.py` at `--maxiter 1500` stops with
`success=False` and `iterations == 1500`, i.e. L-BFGS-B hit the iteration cap
rather than a gradient/tolerance criterion. That matters for two different
reasons, and they have different answers:

  * For the *fidelity ranking* -- if all configs are equally under-trained, the
    ranking may still be meaningful.
  * For the *bandwidth budget* -- selection is "best fidelity subject to
    bandwidth <= budget". If bandwidth is still drifting when the optimizer
    stops, a config near the budget line could cross it with more iterations
    and change the recommendation.

This script answers both empirically: it reloads a cached pulse, continues
optimizing it for `--extra` more iterations under the identical objective, and
reports how far the scored metrics moved. A metric that barely moves is
plateaued; one that keeps drifting is not.

Warm-starting from the cached `u` is exact here: the pulse is already
band-limited and `project_bandlimit` is idempotent, so `P(u) = u`.

Usage
-----
    python analysis/penalty_convergence_check.py                  # default pair
    python analysis/penalty_convergence_check.py --extra 1500
    python analysis/penalty_convergence_check.py --labels "deriv=0_boundary=2e-05_amp=8e-05_disc=0.5"

Writes `tables/penalty_convergence_check.csv`.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis import penalty_sweep as ps
from core.grape_core import coherent_fidelity_multi_state
from core.optimizer import optimize_multi_state_pulse
from main import GATE_FACTORIES

# Incumbent, and the recipe the budgeted selection lands on.
DEFAULT_LABELS = (
    "deriv=1e-05_boundary=2e-05_amp=8e-05_disc=0.5",
    "deriv=1e-06_boundary=2e-05_amp=8e-05_disc=0.5",
)


def continue_config(gate, cfg, seed, maxiter, extra, eval_truncs, n_jobs=3):
    """Reload the cached pulse for `cfg`, optimize `extra` more iterations, rescore."""
    h = ps._config_hash(gate, cfg["penalties"], seed, maxiter,
                        amp_max=cfg.get("amp_max", ps.FIXED["amp_max"]))
    pulse_path, _ = ps._cache_paths(gate, h)
    if not os.path.exists(pulse_path):
        raise FileNotFoundError(
            f"no cached pulse for {cfg['label']} at maxiter={maxiter} "
            f"(expected {pulse_path}); run the sweep first"
        )

    u0 = np.load(pulse_path)
    trained = set(ps.FIXED["trunc_list"])
    s0 = ps.score_pulse(gate, u0, eval_truncs, trained, ps.FIXED["n_t"], ps.FIXED["dt"])
    m0 = ps.pulse_metrics(u0, ps.FIXED["dt"])

    pen = dict(cfg["penalties"])
    pen["amp_max"] = cfg.get("amp_max", ps.FIXED["amp_max"])

    t0 = time.time()
    u1, info = optimize_multi_state_pulse(
        GATE_FACTORIES[gate],
        trunc_list=ps.FIXED["trunc_list"], n_t=ps.FIXED["n_t"],
        N=ps.FIXED["N"], dt=ps.FIXED["dt"], penalties=pen,
        warm_start=u0, save_path=None, n_jobs=n_jobs, maxiter=extra,
        cav_band=ps.FIXED["cav_band"], tra_band=ps.FIXED["tra_band"],
        hard_amp_limit=ps.FIXED["hard_amp_limit"],
        fidelity_fn=coherent_fidelity_multi_state, verbose=False,
    )
    elapsed = time.time() - t0

    s1 = ps.score_pulse(gate, u1, eval_truncs, trained, ps.FIXED["n_t"], ps.FIXED["dt"])
    m1 = ps.pulse_metrics(u1, ps.FIXED["dt"])

    return {
        "label": cfg["label"],
        "extra_iters": extra,
        "reconverged": bool(info["success"]),
        "iters_used": info["iterations"],
        "elapsed_s": elapsed,
        "F_held_before": s0["F_ped_heldout_mean"],
        "F_held_after": s1["F_ped_heldout_mean"],
        "dF_held": s1["F_ped_heldout_mean"] - s0["F_ped_heldout_mean"],
        "F_min_before": s0["F_ped_heldout_min"],
        "F_min_after": s1["F_ped_heldout_min"],
        "spread_before": s0["robustness_spread"],
        "spread_after": s1["robustness_spread"],
        "BW_before": m0["bandwidth_MHz"],
        "BW_after": m1["bandwidth_MHz"],
        "dBW": m1["bandwidth_MHz"] - m0["bandwidth_MHz"],
        "peak_before": m0["peak_amp"],
        "peak_after": m1["peak_amp"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("Usage")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate", default="X", choices=sorted(GATE_FACTORIES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--maxiter", type=int, default=1500,
                   help="maxiter the cached pulses were produced at (selects the cache entry)")
    p.add_argument("--extra", type=int, default=750, help="additional iterations to run")
    p.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    p.add_argument("--n-jobs", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    by_label = {c["label"]: c for c in ps.build_ofat_configs()}
    by_label.update({c["label"]: c for c in ps.build_grid_configs("deriv", "boundary")})

    rows = []
    for label in args.labels:
        if label not in by_label:
            raise SystemExit(f"error: unknown config label {label!r}")
        print(f"continuing {label} for {args.extra} iters ...", flush=True)
        row = continue_config(args.gate, by_label[label], args.seed, args.maxiter,
                              args.extra, ps.DEFAULT_EVAL_TRUNCS, n_jobs=args.n_jobs)
        rows.append(row)
        print(f"  F_held {row['F_held_before']:.6f} -> {row['F_held_after']:.6f} "
              f"({row['dF_held']:+.2e}) | BW {row['BW_before']:.3f} -> "
              f"{row['BW_after']:.3f} ({row['dBW']:+.3f} MHz) | "
              f"reconverged={row['reconverged']}", flush=True)

    df = pd.DataFrame(rows)
    out = args.out or os.path.join(REPO_ROOT, "tables", "penalty_convergence_check.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
