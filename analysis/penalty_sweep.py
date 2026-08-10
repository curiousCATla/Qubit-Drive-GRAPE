#!/usr/bin/env python3
"""
analysis/penalty_sweep.py

Multi-objective (Pareto) sweep over the four GRAPE soft-penalty weights.

The cost function in `core/optimizer.optimize_multi_state_pulse` carries four
regularizer weights -- `deriv`, `boundary`, `amp`, `disc`. Because they are
regularizers, maximizing the *training* objective F_coh drives all four toward
zero and yields rough, high-bandwidth pulses that overfit the training
truncations `trunc_list`. Choosing them is therefore not a scalar optimization:
it is a trade-off between the reported fidelity on *held-out* cavity
truncations and the pulse's hardware cost (bandwidth, peak amplitude) and
robustness.

This harness sweeps the weights, retrains a pulse per configuration, and scores
each on metrics that training never sees:

  * `F_ped_heldout_{mean,min}` -- Pedersen gate fidelity (main.report_pedersen_
    gate_fidelity) averaged over cavity truncations NOT in `trunc_list`. This is
    the reported figure of merit, distinct from the optimizer's F_coh.
  * `robustness_spread`        -- max-min of F_ped across the held-out set.
  * `leakage_L1_mean`          -- mean population leaving the logical subspace.
  * `peak_amp`, `roughness`, `bandwidth_MHz` -- pulse cost.

Only the penalty weights vary. Everything else (warm start, `trunc_list`,
frequency bands, amplitude limits, N, dt) is pinned in `FIXED` below so the
comparison is clean.

Both the trained pulse and the scored metric row are disk-cached under
`results/penalty_sweep_cache/`, keyed by a hash of the full configuration, so an
interrupted sweep resumes for free and re-scoring never retrains.

Usage
-----
    python analysis/penalty_sweep.py --gate X --mode ofat --seeds 42 --maxiter 1500
    python analysis/penalty_sweep.py --gate X --mode grid --grid-x deriv --grid-y disc

Outputs
-------
    tables/penalty_sweep_<gate>_ofat.csv
    tables/penalty_sweep_<gate>_grid_<x>_<y>.csv
    results/penalty_sweep_<gate>_<mode>.json      (manifest: configs + provenance)
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.grape_core import coherent_fidelity_multi_state
from core.optimizer import optimize_multi_state_pulse
from core.propagator import leakage_L1
from main import GATE_FACTORIES, report_pedersen_gate_fidelity


# ============================================================
# Fixed experimental conditions -- everything that is NOT swept
# ============================================================
#
# These mirror main.py's production defaults. They are pinned here rather than
# exposed as sweep axes so that every row in the output CSV differs ONLY in the
# penalty weights. Changing anything here invalidates the cache (it is part of
# the config hash), which is the intended behavior.

FIXED = {
    "trunc_list": [22, 24, 26],
    "n_t": 3,
    "N": 550,
    "dt": 0.002,
    "cav_band": (-27.0, 27.0),
    "tra_band": (-33.0, 33.0),
    "hard_amp_limit": 40.0,
    "amp_max": 40.0,
    "warm_start": None,          # smooth random warm start, seeded per-run
}

# Incumbent recipe = main.py's current defaults. Every OFAT ladder holds the
# other three weights here.
BASELINE = {
    "deriv": 1e-5,
    "boundary": 2e-5,
    "amp": 8e-5,
    "disc": 0.5,
}

PENALTY_NAMES = ("deriv", "boundary", "amp", "disc")

# OFAT ladder: multiples of the incumbent value, plus a 0 negative control.
# The 0 row is the point of the exercise -- it should show HIGHER training
# fidelity but worse bandwidth/robustness, demonstrating the trade-off is real.
OFAT_MULTIPLIERS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)

# --- The amp_max axis --------------------------------------------------
#
# Sweeping `lambda_amp` is a NO-OP under these fixed conditions, and measurably
# so: `amplitude_penalty(u, amp_max=40)` returns exactly 0.0 with an exactly
# zero gradient, because converged pulses here peak near 17 rad/us -- well under
# the 40 rad/us soft threshold. Multiplying an identically-zero term by any
# weight changes nothing (verified: F_coh range 0.0e+00 across lambda_amp
# 0 -> 8e-4).
#
# The live knob is the THRESHOLD, not the weight. This ladder brackets the
# observed peak amplitude, so the low end binds hard and the high end never
# binds -- which is what makes it a real peak-power constraint.
AMP_MAX_VALUES = (12.0, 16.0, 20.0, 28.0, 40.0)

# Axis names that are valid sweep targets. "amp_max" is not a penalty weight,
# so it is handled separately from PENALTY_NAMES throughout.
AXIS_NAMES = ("deriv", "boundary", "amp_max", "disc")

# Focused 2-D grid axes (absolute values, not multipliers).
GRID_VALUES = {
    "deriv": [0.0, 3e-6, 1e-5, 3e-5, 1e-4],
    "boundary": [0.0, 6e-6, 2e-5, 6e-5, 2e-4],
    "amp_max": list(AMP_MAX_VALUES),
    "disc": [0.0, 0.15, 0.5, 1.5, 5.0],
}

# Cavity truncations used for SCORING. Those in FIXED["trunc_list"] were trained
# on; the rest are held out and carry the headline metric.
DEFAULT_EVAL_TRUNCS = list(range(16, 41, 2))

CACHE_DIR = os.path.join(REPO_ROOT, "results", "penalty_sweep_cache")
TABLE_DIR = os.path.join(REPO_ROOT, "tables")
RESULT_DIR = os.path.join(REPO_ROOT, "results")


# ============================================================
# Configuration enumeration
# ============================================================

def _label(penalties, amp_max):
    """
    Compact human-readable label for a recipe. `amp_max` is appended only when
    it differs from the fixed default, so the common case stays readable and
    labels from before the amp_max axis existed are unchanged.
    """
    s = "_".join(f"{k}={penalties[k]:.3g}" for k in PENALTY_NAMES)
    if amp_max != FIXED["amp_max"]:
        s += f"_ampmax={amp_max:.3g}"
    return s


def _make_config(pen, amp_max, swept, multiplier=np.nan):
    return {
        "penalties": pen,
        "amp_max": amp_max,
        "swept": swept,
        "multiplier": multiplier,
        "label": _label(pen, amp_max),
        "is_baseline": (
            all(pen[k] == BASELINE[k] for k in PENALTY_NAMES)
            and amp_max == FIXED["amp_max"]
        ),
    }


def build_ofat_configs():
    """
    One-factor-at-a-time: vary each axis while the others sit at BASELINE. The
    baseline point appears once, deduplicated.

    The `amp` weight is deliberately NOT an axis -- it is provably inert here
    (see AMP_MAX_VALUES). Its slot is taken by `amp_max`, the threshold that
    actually determines whether the amplitude penalty ever activates.
    """
    configs, seen = [], set()

    for name in ("deriv", "boundary", "disc"):
        for mult in OFAT_MULTIPLIERS:
            pen = dict(BASELINE)
            pen[name] = BASELINE[name] * mult
            key = (tuple(pen[k] for k in PENALTY_NAMES), FIXED["amp_max"])
            if key in seen:
                continue
            seen.add(key)
            configs.append(_make_config(pen, FIXED["amp_max"], name, mult))

    for amp_max in AMP_MAX_VALUES:
        pen = dict(BASELINE)
        key = (tuple(pen[k] for k in PENALTY_NAMES), amp_max)
        if key in seen:
            continue
        seen.add(key)
        configs.append(_make_config(pen, amp_max, "amp_max", np.nan))

    return configs


def _axis_apply(pen, amp_max, name, value):
    """Set one axis, returning the updated (penalties, amp_max) pair."""
    if name == "amp_max":
        return pen, value
    pen = dict(pen)
    pen[name] = value
    return pen, amp_max


def build_grid_configs(x_name, y_name):
    """Full factorial over two axes; everything else stays at BASELINE."""
    configs = []
    for xv in GRID_VALUES[x_name]:
        for yv in GRID_VALUES[y_name]:
            pen, amp_max = dict(BASELINE), FIXED["amp_max"]
            pen, amp_max = _axis_apply(pen, amp_max, x_name, xv)
            pen, amp_max = _axis_apply(pen, amp_max, y_name, yv)
            configs.append(_make_config(pen, amp_max, f"{x_name}x{y_name}"))
    return configs


# ============================================================
# Pulse-cost metrics
# ============================================================

def pulse_metrics(u, dt):
    """
    Hardware-cost and smoothness metrics for a pulse, computed on the physical
    (band-limited, as-returned) waveform.

    Drives are complex: eps_C = u[:,0] + i u[:,1], eps_T = u[:,2] + i u[:,3].
    `bandwidth_MHz` is the RMS spectral width of whichever drive is wider --
    that is the one a hardware budget binds on.
    """
    u = np.asarray(u)
    eps_c = u[:, 0] + 1j * u[:, 1]
    eps_t = u[:, 2] + 1j * u[:, 3]

    # dt is in microseconds, so fftfreq returns cycles/us == MHz directly.
    freqs = np.fft.fftfreq(u.shape[0], d=dt)

    def rms_bw(eps):
        power = np.abs(np.fft.fft(eps)) ** 2
        total = power.sum()
        if total <= 0:
            return 0.0
        return float(np.sqrt((freqs ** 2 * power).sum() / total))

    bw_c, bw_t = rms_bw(eps_c), rms_bw(eps_t)

    peak_amp = float(max(np.abs(eps_c).max(), np.abs(eps_t).max()))
    # Per-step RMS first difference, in rad/us: a direct proxy for how hard the
    # waveform is to synthesize.
    roughness = float(np.sqrt(np.mean(np.sum(np.diff(u, axis=0) ** 2, axis=1))))

    return {
        "peak_amp": peak_amp,
        "roughness": roughness,
        "bandwidth_MHz": float(max(bw_c, bw_t)),
        "bandwidth_cav_MHz": bw_c,
        "bandwidth_tra_MHz": bw_t,
    }


def score_pulse(gate, u, eval_truncs, trained_truncs, n_t, dt):
    """
    Score a trained pulse on the reported (Pedersen) fidelity across cavity
    truncations, separating trained from held-out.

    The held-out aggregate is the headline: a pulse that exploits the Hilbert
    space truncation wall scores well on `trained` and poorly here.
    """
    per_trunc, leaks = {}, {}
    for nc in eval_truncs:
        F, U_log = report_pedersen_gate_fidelity(gate, u, n_c=nc, n_t=n_t, dt=dt)
        per_trunc[nc] = float(F)
        leaks[nc] = float(leakage_L1(U_log))

    held = [nc for nc in eval_truncs if nc not in trained_truncs]
    trained = [nc for nc in eval_truncs if nc in trained_truncs]

    F_held = np.array([per_trunc[nc] for nc in held], dtype=float)
    F_train = np.array([per_trunc[nc] for nc in trained], dtype=float) if trained else np.array([np.nan])

    out = {
        "F_ped_heldout_mean": float(F_held.mean()),
        "F_ped_heldout_min": float(F_held.min()),
        "F_ped_heldout_std": float(F_held.std()),
        "robustness_spread": float(F_held.max() - F_held.min()),
        "F_ped_trained_mean": float(np.nanmean(F_train)),
        "leakage_L1_mean": float(np.mean([leaks[nc] for nc in held])),
        "leakage_L1_max": float(np.max([leaks[nc] for nc in held])),
    }
    # Overfit gap: positive means the pulse does better where it was trained.
    out["overfit_gap"] = out["F_ped_trained_mean"] - out["F_ped_heldout_mean"]
    out["_per_trunc"] = {str(k): v for k, v in per_trunc.items()}
    return out


# ============================================================
# Caching
# ============================================================

def _config_hash(gate, penalties, seed, maxiter, amp_max=None, extra=None):
    """
    Stable hash over everything that affects the trained pulse.

    `amp_max` enters the payload only when it differs from the fixed default.
    That keeps hashes computed before amp_max became a sweep axis valid, so an
    existing cache is not invalidated by adding the axis.
    """
    payload = {
        "gate": gate,
        "penalties": {k: float(penalties[k]) for k in PENALTY_NAMES},
        "seed": int(seed),
        "maxiter": int(maxiter),
        "fixed": {
            k: (list(v) if isinstance(v, (list, tuple)) else v)
            for k, v in FIXED.items()
        },
    }
    if amp_max is not None and float(amp_max) != float(FIXED["amp_max"]):
        payload["amp_max"] = float(amp_max)
    if extra:
        payload["extra"] = extra
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _cache_paths(gate, h):
    d = os.path.join(CACHE_DIR, gate)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"u_{h}.npy"), os.path.join(d, f"row_{h}.json")


# ============================================================
# One run
# ============================================================

def run_one(gate, cfg, seed, maxiter, eval_truncs, n_jobs=3, verbose=False,
            force=False):
    """
    Train (or load from cache) one pulse for one penalty recipe + seed, then
    score it. Returns a flat dict = one CSV row.
    """
    penalties = cfg["penalties"]
    amp_max = cfg.get("amp_max", FIXED["amp_max"])
    h = _config_hash(gate, penalties, seed, maxiter, amp_max=amp_max)
    pulse_path, row_path = _cache_paths(gate, h)

    row_extra_key = _config_hash(
        gate, penalties, seed, maxiter, amp_max=amp_max,
        extra={"eval": list(eval_truncs)},
    )

    # Fully cached row (same pulse AND same scoring set)?
    if not force and os.path.exists(row_path):
        with open(row_path) as f:
            cached = json.load(f)
        if cached.get("_score_key") == row_extra_key:
            cached["cached"] = True
            return cached

    # Trained pulse cached?
    if not force and os.path.exists(pulse_path):
        u = np.load(pulse_path)
        info = {"iterations": np.nan, "success": None, "final_fidelity": np.nan}
        train_time = np.nan
        reused_pulse = True
    else:
        pen = dict(penalties)
        pen["amp_max"] = amp_max
        t0 = time.time()
        u, info = optimize_multi_state_pulse(
            GATE_FACTORIES[gate],
            trunc_list=FIXED["trunc_list"],
            n_t=FIXED["n_t"],
            N=FIXED["N"],
            dt=FIXED["dt"],
            penalties=pen,
            warm_start=FIXED["warm_start"],
            warm_start_seed=seed,
            save_path=None,
            n_jobs=n_jobs,
            maxiter=maxiter,
            cav_band=FIXED["cav_band"],
            tra_band=FIXED["tra_band"],
            hard_amp_limit=FIXED["hard_amp_limit"],
            fidelity_fn=coherent_fidelity_multi_state,
            verbose=verbose,
        )
        train_time = time.time() - t0
        np.save(pulse_path, u)
        reused_pulse = False

    scores = score_pulse(
        gate, u, eval_truncs, set(FIXED["trunc_list"]), FIXED["n_t"], FIXED["dt"]
    )
    metrics = pulse_metrics(u, FIXED["dt"])

    row = {
        "gate": gate,
        "label": cfg["label"],
        "swept": cfg["swept"],
        "multiplier": cfg["multiplier"],
        "is_baseline": cfg["is_baseline"],
        "seed": seed,
        "maxiter": maxiter,
        **{f"lambda_{k}": float(penalties[k]) for k in PENALTY_NAMES},
        "amp_max": float(amp_max),
        "F_coh_train": float(info.get("final_fidelity", np.nan)),
        "iterations": info.get("iterations", np.nan),
        "converged": info.get("success", None),
        "train_time_s": train_time,
        **scores,
        **metrics,
        "config_hash": h,
        "_score_key": row_extra_key,
        "reused_pulse": reused_pulse,
        "cached": False,
    }

    with open(row_path, "w") as f:
        json.dump(row, f, indent=2)
    return row


# ============================================================
# Sweep drivers
# ============================================================

def run_sweep(gate, configs, seeds, maxiter, eval_truncs, out_csv, manifest_path,
              n_jobs=3, verbose=False, force=False):
    import pandas as pd

    rows = []
    total = len(configs) * len(seeds)
    done = 0
    t_start = time.time()

    for cfg in configs:
        for seed in seeds:
            done += 1
            print(
                f"[{done}/{total}] gate={gate} seed={seed} {cfg['label']}",
                flush=True,
            )
            row = run_one(
                gate, cfg, seed, maxiter, eval_truncs,
                n_jobs=n_jobs, verbose=verbose, force=force,
            )
            tag = "cache" if row.get("cached") else (
                "rescored" if row.get("reused_pulse") else "trained"
            )
            print(
                f"    [{tag}] F_held={row['F_ped_heldout_mean']:.6f} "
                f"min={row['F_ped_heldout_min']:.6f} "
                f"spread={row['robustness_spread']:.3e} "
                f"BW={row['bandwidth_MHz']:.2f} MHz "
                f"peak={row['peak_amp']:.1f}",
                flush=True,
            )
            rows.append(row)

            # Write incrementally so an interrupted sweep still leaves a usable
            # table behind.
            df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                               for r in rows])
            os.makedirs(TABLE_DIR, exist_ok=True)
            df.to_csv(out_csv, index=False)

    elapsed = time.time() - t_start

    os.makedirs(RESULT_DIR, exist_ok=True)
    manifest = {
        "gate": gate,
        "n_configs": len(configs),
        "seeds": list(seeds),
        "maxiter": maxiter,
        "eval_truncs": list(eval_truncs),
        "trained_truncs": FIXED["trunc_list"],
        "heldout_truncs": [nc for nc in eval_truncs if nc not in FIXED["trunc_list"]],
        "fixed": {k: (list(v) if isinstance(v, tuple) else v) for k, v in FIXED.items()},
        "baseline": BASELINE,
        "csv": os.path.relpath(out_csv, REPO_ROOT),
        "elapsed_s": elapsed,
        "configs": [
            {"label": c["label"], "swept": c["swept"], "penalties": c["penalties"],
             "amp_max": c.get("amp_max", FIXED["amp_max"])}
            for c in configs
        ],
        "per_trunc": {
            f"{r['label']}|seed={r['seed']}": r.get("_per_trunc", {}) for r in rows
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {out_csv}")
    print(f"Wrote {manifest_path}")
    print(f"Total wall time: {elapsed/60:.1f} min")
    return rows


def build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gate", default="X", choices=sorted(GATE_FACTORIES))
    p.add_argument("--mode", default="ofat", choices=["ofat", "grid"])
    # Default grid is deriv x boundary: both axes measurably move the pulse.
    # deriv x disc was the original plan but disc is inert at these training
    # truncations (disc_cost ~ 6e-8 at convergence, against an O(1) fidelity
    # term), so that grid is constant along one axis.
    p.add_argument("--grid-x", default="deriv", choices=AXIS_NAMES)
    p.add_argument("--grid-y", default="boundary", choices=AXIS_NAMES)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--maxiter", type=int, default=1500)
    p.add_argument("--eval-truncs", type=int, nargs="+", default=DEFAULT_EVAL_TRUNCS,
                   help="Cavity truncations to SCORE on. Those also in "
                        "trunc_list are reported as 'trained'; the rest are held out.")
    p.add_argument("--n-jobs", type=int, default=3)
    p.add_argument("--tag", default="", help="Suffix for output filenames, e.g. 'demo'.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only run the first K configs (smoke tests).")
    p.add_argument("--force", action="store_true", help="Ignore cache and retrain.")
    p.add_argument("--verbose", action="store_true", help="Optimizer chatter.")
    return p


def main():
    args = build_arg_parser().parse_args()

    if args.mode == "ofat":
        configs = build_ofat_configs()
        stem = f"penalty_sweep_{args.gate}_ofat"
    else:
        if args.grid_x == args.grid_y:
            raise SystemExit("error: --grid-x and --grid-y must differ")
        configs = build_grid_configs(args.grid_x, args.grid_y)
        stem = f"penalty_sweep_{args.gate}_grid_{args.grid_x}_{args.grid_y}"

    if args.tag:
        stem = f"{stem}_{args.tag}"
    if args.limit:
        configs = configs[: args.limit]

    out_csv = os.path.join(TABLE_DIR, f"{stem}.csv")
    manifest = os.path.join(RESULT_DIR, f"{stem}.json")

    held = [nc for nc in args.eval_truncs if nc not in FIXED["trunc_list"]]
    print(f"mode={args.mode}  configs={len(configs)}  seeds={args.seeds}  "
          f"maxiter={args.maxiter}")
    print(f"trained truncs {FIXED['trunc_list']}  |  held-out {held}")

    run_sweep(
        args.gate, configs, args.seeds, args.maxiter, args.eval_truncs,
        out_csv, manifest, n_jobs=args.n_jobs, verbose=args.verbose,
        force=args.force,
    )


if __name__ == "__main__":
    main()
