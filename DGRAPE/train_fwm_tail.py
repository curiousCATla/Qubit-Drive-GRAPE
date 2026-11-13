#!/usr/bin/env python3
"""
FWM-tail training (notebooks/FWM_tails.ipynb §7): the d2 X gate runs as trained, then a tail of the two FWM combs
(one complex waveform eta_c(t) per comb, DGRAPE/fwm_tail.py) is optimized by L-BFGS-B on its pre-image
x (N, 4), scored against the ideal X at the end of the tail. Nothing else moves.

Settings are DGRAPE/fwm_tail.py's recipe: tail 1 us (--tail-us), n_c = 16, n_r = 2, one exact
exponential per 2 ns step in the |f>-rotating frame, readout traced out (--readout empty to require it
empty), decay set (--channels dgrape_all | shirol for more), higher-order terms on, cap |eta_c| <= 0.1 per comb (modulus penalty), band +-12 MHz
on each eta_c, 48 ns ramp, box |x| <= 25 with x = 250 eta, gtol 1e-12 (as DGRAPE/train.py),
best-vs-final. Penalty weights are fwm_tail.TAIL_PENALTIES (amp 5.0, deriv 0.625); --deriv-scale multiplies the derivative
(smoothness) weight (0.1 in notebooks/tail_experiments.ipynb).

The gate before the tail is d2 unless --gate-pulse FILE names another (N, 4) pulse on the 2 ns grid
(e.g. pulses/dgrape/u_X_est_closed.npy, DGRAPE/est_gate.py); --target names the ideal gate at the end
(default X; I after the idle pulse pulses/dgrape/u_idle_X_duration.npy). --gate-n-c propagates the gate at that
truncation and cuts its output to --n-c for the tail (fwm_tail.tail_input): the closed-trained gates
need n_c 24 to run but end with <= 2e-5 above Fock 16. --n-gates m runs that gate m times back to back
before the tail (one propagation of the pulse tiled m times; the gates share the DGRAPE frame); the default
target is then X^m (X for odd m, I for even m).

On Windows (OpenBLAS) set OPENBLAS_NUM_THREADS=2 at launch: loky otherwise gives each of the three
workers cpu_count / 3 threads, which is slower at n_c 16 (31 s against 12 s per 1 us evaluation).

The start is --start NAME (fwm_tail.start_library: zero, tableII, tableII_x3, bursts_f0.5) or --init-x
FILE (a pre-image (N, 4) .npy), optionally plus seeded noise. Everything is written to --outdir (default
~/fwm_runs, outside iCloud): a checkpoint json + best pre-image after every iteration (resume with
--init-x <outdir>/xe_<tag>_best.npy), and at the end eta_<tag>.npy ((N, 2) complex), xe_<tag>.npy
(pre-image) and <tag>.json. Copy them into pulses/dgrape/ when done (never pulses/).

    python3 DGRAPE/train_fwm_tail.py --start tableII --tag X_tail_a --dry-run     # one evaluation, timed
    python3 DGRAPE/train_fwm_tail.py --start tableII --tag X_tail_a --max-iter 300
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np                                    # noqa: E402
from scipy.optimize import minimize                   # noqa: E402

from DGRAPE import fwm                                # noqa: E402
from DGRAPE import fwm_tail as ft                     # noqa: E402

DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "fwm_runs")


def _dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def start_preimage(start, tail_us, n_c):
    """Pre-image (N, 4) from a start_library name or a .npy file."""
    if start in ft.start_library(tail_us, n_c=n_c):
        return ft.eta_preimage(ft.start_library(tail_us, n_c=n_c)[start])
    return np.load(start)


def default_target(n_gates):
    """X^m: X after an odd number of X gates, I after an even number."""
    return "X" if n_gates % 2 else "I"


def make_objective(tail_us, n_c, n_jobs, readout, band, cap, mixing, gate_pulse=None, gate_n_c=None, target="X",
                   n_gates=1, channels="decay", deriv_scale=1.0):
    if n_gates < 1:
        raise ValueError(f"n_gates must be >= 1, not {n_gates}")
    if n_gates > 1 and gate_pulse is None:
        raise ValueError("--n-gates needs --gate-pulse")
    pulse = None if gate_pulse is None else np.tile(np.load(gate_pulse), (n_gates, 1))
    return ft.TailObjective(tail_us=tail_us, n_c=n_c, gate=target, readout=readout, band=band, cap=cap,
                            mixing=mixing, n_jobs=n_jobs, pulse=pulse, gate_n_c=gate_n_c, noise=channels,
                            penalties=None if deriv_scale == 1.0 else
                            {"deriv": deriv_scale * ft.TAIL_PENALTIES["deriv"]})


def dry_run(start, tail_us=ft.TAIL_US, n_c=16, n_jobs=3, readout=ft.READOUT, band=ft.ETA_BAND,
            cap=ft.ETA_CAP, mixing=0.0, gate_pulse=None, gate_n_c=None, target="X", n_gates=1, channels="decay",
            deriv_scale=1.0):
    """One cost + gradient evaluation of the start, timed. No optimizer."""
    x0 = start_preimage(start, tail_us, n_c)
    t0 = time.time()
    obj = make_objective(tail_us, n_c, n_jobs, readout, band, cap, mixing, gate_pulse, gate_n_c, target, n_gates,
                         channels, deriv_scale)
    t_build = time.time() - t0
    try:
        t0 = time.time()
        cost, g, terms = obj.evaluate(x0)
        t_eval = time.time() - t0
    finally:
        obj.close()
    out = {"start": start, "gate_pulse": gate_pulse or "d2", "gate_n_c": gate_n_c or n_c, "n_gates": n_gates,
           "channels": channels, "deriv_scale": deriv_scale, "penalties": obj.pen,
           "target": target,
           "trace_cut": obj.trace_cut, "tail_us": tail_us, "N": obj.N, "n_c": n_c, "n_jobs": n_jobs, "cost": cost,
           "grad_norm": float(np.linalg.norm(g)), "build_s": t_build, "eval_s": t_eval, **terms}
    print(json.dumps(out, indent=1, default=float), flush=True)
    return out


def train(start, tag, outdir, tail_us=ft.TAIL_US, max_iter=300, seed=None, noise=0.0, n_c=16, n_jobs=3,
          readout=ft.READOUT, band=ft.ETA_BAND, cap=ft.ETA_CAP, mixing=0.0, max_hours=None, verbose=True,
          gate_pulse=None, gate_n_c=None, target="X", n_gates=1, channels="decay", deriv_scale=1.0):
    x0 = start_preimage(start, tail_us, n_c)
    N = ft.n_steps(tail_us)
    if x0.shape != (N, 4):
        raise ValueError(f"start has shape {x0.shape}, expected {(N, 4)}")
    if noise > 0:
        x0 = x0 + noise * ft.S_ETA * np.random.default_rng(seed).normal(size=x0.shape)
    x0 = np.clip(x0, -ft.U_BOX, ft.U_BOX)
    obj = make_objective(tail_us, n_c, n_jobs, readout, band, cap, mixing, gate_pulse, gate_n_c, target, n_gates,
                         channels, deriv_scale)
    history = []
    t0 = time.time()
    try:
        cost0, _, terms0 = obj.evaluate(x0)
        if verbose:
            print(f"start: cost {cost0:.6e}  F_pro {terms0['F_pro']:.6f}", flush=True)

        class Stop(Exception):
            pass

        def callback(xk):
            c, _, terms = obj.evaluate(xk)
            history.append({"it": len(history) + 1, "cost": c, **terms, "t": time.time() - t0,
                            "n_evals": obj.n_evals})
            np.save(os.path.join(outdir, f"xe_{tag}_best.npy"), obj.best["x"])
            _dump(os.path.join(outdir, f"{tag}_checkpoint.json"),
                  {"history": history, "best_F": obj.best["F"], "start": str(start)})
            if verbose:
                extra = "  ".join(f"{k} {terms[k]:.2e}" for k in ("amp", "deriv") if k in terms)
                print(f"it {len(history):4d}  cost {c:.6e}  F_pro {terms['F_pro']:.6f}  "
                      f"max|eta| {terms['max_eta_1']:.4f}/{terms['max_eta_2']:.4f}  {extra}  "
                      f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
            if max_hours and time.time() - t0 > 3600 * max_hours:
                raise Stop

        try:
            res = minimize(obj.cost, x0.ravel(), jac=obj.grad, method="L-BFGS-B",
                           bounds=[(-ft.U_BOX, ft.U_BOX)] * (N * 4),
                           options={"maxiter": max_iter, "gtol": 1e-12, "ftol": 1e-15},
                           callback=callback)
            x_final, message, nit = res.x.reshape(N, 4), str(res.message), int(res.nit)
        except Stop:
            x_final, message, nit = obj._cache_x, f"stopped at the {max_hours} h wall limit", len(history)

        F_final = obj.evaluate(x_final)[2]["F_pro"]
        if obj.best["F"] > F_final:
            x, picked = obj.best["x"].copy(), "best_seen"
        else:
            x, picked = x_final, "final"
        b4 = obj.chain.to_physical(x)
        _, _, terms = obj.evaluate(x)
    finally:
        obj.close()
    eta = fwm.beta_from_real(b4)
    pinned = float(np.mean(np.abs(x) >= ft.U_BOX * (1 - 1e-9)))
    info = {"tag": tag, "start": str(start), "gate_pulse": gate_pulse or "d2", "gate_n_c": gate_n_c or n_c,
            "n_gates": n_gates, "target": target, "channels": channels, "deriv_scale": deriv_scale,
            "trace_cut": obj.trace_cut, "seed": seed, "noise": noise,
            "tail_us": tail_us, "N": N,
            "n_c": n_c, "n_r": obj.n_r, "n_ref": obj.n_ref, "readout": readout, "mixing": mixing,
            "max_iter": max_iter, "message": message, "nit": nit, "n_evals": obj.n_evals,
            "wall_s": time.time() - t0, "returned": picked, "cost0": cost0, "F_pro_start": terms0["F_pro"],
            "F_pro": terms["F_pro"], "terms": terms, "penalties": obj.pen, "cap": cap, "band_MHz": band,
            "ramp_ns": obj.ramp_ns, "s_eta": ft.S_ETA, "delta_ref": obj.tm.delta_ref,
            "max_abs_preimage": float(np.abs(x).max()), "frac_preimage_pinned": pinned,
            **ft.eta_stats(eta), "history": history}
    np.save(os.path.join(outdir, f"eta_{tag}.npy"), eta)
    np.save(os.path.join(outdir, f"xe_{tag}.npy"), x)
    _dump(os.path.join(outdir, f"{tag}.json"), info)
    if verbose:
        print(f"returned {picked}: F_pro {terms['F_pro']:.6f} (start {terms0['F_pro']:.6f}); "
              f"max|eta| {info['max_eta_1']:.4f}/{info['max_eta_2']:.4f}; {pinned:.2%} of x pinned; "
              f"{message}", flush=True)
    return eta, x, info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True,
                   help="start_library name (zero, tableII, tableII_x3, bursts_f0.5) or a pre-image .npy")
    p.add_argument("--tag", required=True)
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--tail-us", type=float, default=ft.TAIL_US)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--max-hours", type=float, default=None, help="wall limit; returns best-vs-final")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--noise", type=float, default=0.0, help="std of added eta noise (eta units)")
    p.add_argument("--channels", choices=("decay", "dgrape_all", "shirol"), default="decay",
                   help="jump operators of the gate and the tail: decay set (default), DGRAPE dephasing + heating, "
                        "or Shirol's full set (fwm_tail.noise_flags)")
    p.add_argument("--deriv-scale", type=float, default=1.0,
                   help="multiply the derivative (smoothness) penalty weight, fwm_tail.TAIL_PENALTIES['deriv'] = 0.625")
    p.add_argument("--n-c", type=int, default=16)
    p.add_argument("--n-jobs", type=int, default=3)
    p.add_argument("--readout", choices=("traced", "empty"), default=ft.READOUT)
    p.add_argument("--band", type=float, nargs=2, default=list(ft.ETA_BAND), help="MHz on each eta_c")
    p.add_argument("--cap", type=float, default=ft.ETA_CAP)
    p.add_argument("--mixing", type=float, default=0.0)
    p.add_argument("--gate-pulse", default=None, help="(N, 4) gate pulse .npy before the tail (default d2)")
    p.add_argument("--gate-n-c", type=int, default=None,
                   help="propagate the gate at this n_c, then cut its output to --n-c (default: --n-c)")
    p.add_argument("--target", default=None,
                   help="ideal logical gate at the end of the tail (default X^m: X for odd --n-gates, I for even; "
                        "pass I after an idle gate pulse)")
    p.add_argument("--n-gates", type=int, default=1, help="run the gate pulse this many times before the tail")
    p.add_argument("--dry-run", action="store_true", help="one timed cost + gradient evaluation, no training")
    a = p.parse_args()
    if a.target is None:
        a.target = default_target(a.n_gates)
    if a.dry_run:
        dry_run(a.start, a.tail_us, a.n_c, a.n_jobs, a.readout, tuple(a.band), a.cap, a.mixing, a.gate_pulse,
                a.gate_n_c, a.target, a.n_gates, a.channels, a.deriv_scale)
        return
    os.makedirs(a.outdir, exist_ok=True)
    train(a.start, a.tag, a.outdir, a.tail_us, a.max_iter, a.seed, a.noise, a.n_c, a.n_jobs, a.readout,
          tuple(a.band), a.cap, a.mixing, a.max_hours, gate_pulse=a.gate_pulse, gate_n_c=a.gate_n_c,
          target=a.target, n_gates=a.n_gates, channels=a.channels, deriv_scale=a.deriv_scale)


if __name__ == "__main__":
    main()
