"""
Two-stage GRAPE driver for the EsT replication, on the pure-numpy
eigh + analytic-adjoint pipeline (EST/grape_eigh.py).

    python EST/train_est_eigh.py --gate X --variant est --maxiter 2000

This is EST/train_est.py's twin. It runs the SAME weight schedule, the same
L-BFGS-B settings, the same box bounds, and writes the same JSON schema -- the
only difference is which module supplies the objective:

    train_est.py       EST.grape_jax.build_gate_objective     (JAX, expm)
    train_est_eigh.py  EST.grape_eigh.build_gate_objective    (numpy, eigh)

That is deliberate and load-bearing. The comparison in
EST/compare_propagators.py is a field-by-field diff of the two runs' JSON, so
anything that differs between the drivers other than the gradient backend would
confound it. The schedule, the deramp helper, the constraint audit and the
output paths are therefore IMPORTED from train_est rather than restated, so they
cannot drift.

`--tag` defaults to "eigh" rather than None, so this driver cannot overwrite the
JAX run's artifacts (u_X_est.npy, x_X_est.npy, logs/est_X_est.json) even if
invoked with no arguments at all.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import minimize

from EST import kitten_code
from EST.device import DT, EPS_MAX, N_T, BAND_MHZ, RAMP_NS, n_steps
from EST.grape_eigh import build_gate_objective
# Imported, not restated: identical semantics to the JAX driver by construction.
from EST.train_est import (LOG_DIR, PULSE_DIR, SCHEDULES, TRAIN_TRUNC,
                           _print_report, check_constraints, deramp)


def train(gate="X", variant="est", n_t=N_T, dt=DT, trunc_list=TRAIN_TRUNC,
          w_amp=1.0, maxiter=600, seed=0, amp0=20.0, hard_bound=60.0,
          verbose=True, init=None, init_x=None, stage_hook=None):
    """
    Run the two-stage schedule and return (u_physical, x_preimage, info).

    Mirrors EST.train_est.train exactly, including the cold-start RNG
    (`np.random.default_rng(seed).standard_normal(N*4)`), so a given --seed
    produces the IDENTICAL initial pulse on both pipelines and the two runs are
    comparable from the same starting point.
    """
    if variant not in SCHEDULES:
        raise KeyError(f"unknown variant {variant!r}; have {sorted(SCHEDULES)}")
    if init is not None and init_x is not None:
        raise ValueError("pass init or init_x, not both")

    N = n_steps(gate, dt)
    warm = None

    def _check_box(x, source):
        """L-BFGS-B silently projects an infeasible start into the box; refuse."""
        if np.abs(x).max() > hard_bound:
            raise ValueError(
                f"warm-start pre-image reaches {np.abs(x).max():.1f} > hard_bound="
                f"{hard_bound}. L-BFGS-B would clip it into the box and the run "
                f"would not start from the {source} you asked for. Raise "
                "--hard-bound (the box constrains the pre-image, not the physical "
                "amplitude, which is carried by the C4 penalty)."
            )

    if init is None and init_x is None:
        rng = np.random.default_rng(seed)
        x = amp0 * rng.standard_normal(N * 4)
    elif init_x is not None:
        x0 = np.load(init_x)
        if x0.shape != (N, 4):
            raise ValueError(f"{init_x} has shape {x0.shape}, expected {(N, 4)}")
        x = np.asarray(x0, dtype=np.float64).ravel()
        warm = {"path": os.path.abspath(init_x), "kind": "preimage",
                "max_abs_preimage": float(np.abs(x).max())}
        _check_box(x, "pre-image")
        if verbose:
            print(f"exact warm start from {init_x}: "
                  f"max|x0| = {np.abs(x).max():.2f} (bound {hard_bound})")
    else:
        u0 = np.load(init)
        if u0.shape != (N, 4):
            raise ValueError(f"{init} has shape {u0.shape}, expected {(N, 4)}")
        x0, rt_err = deramp(u0, dt)
        x = x0.ravel()
        warm = {"path": os.path.abspath(init), "kind": "pulse",
                "roundtrip_err": rt_err,
                "max_abs_preimage": float(np.abs(x).max())}
        _check_box(x, "pulse")
        if verbose:
            print(f"warm start from {init}: roundtrip err {rt_err:.2e}, "
                  f"max|x0| = {np.abs(x).max():.2f} (bound {hard_bound})")

    bounds = [(-hard_bound, hard_bound)] * (N * 4)

    stages = []
    for i, (w1, w2, w3) in enumerate(SCHEDULES[variant], start=1):
        weights = (w1, w2, w3, w_amp)
        objective, report, constrain_np = build_gate_objective(
            gate, N, dt=dt, n_t=n_t, trunc_list=trunc_list, weights=weights)

        if verbose:
            print(f"\n=== {gate} / {variant} / stage {i}  weights={weights} ===")
            print(f"    N={N} dt={dt*1e3:.1f} ns  trunc_list={trunc_list}  "
                  f"[eigh + analytic adjoint]")
            _print_report("    start", report(x))

        t0 = time.time()
        res = minimize(objective, x, method="L-BFGS-B", jac=True, bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8})
        x = res.x
        elapsed = time.time() - t0

        rep = report(x)
        if verbose:
            _print_report("    end  ", rep)
            print(f"    {res.message}")
            print(f"    nit={res.nit} nfev={res.nfev} cost={res.fun:.6e} "
                  f"({elapsed:.1f}s)")

        stages.append({
            "stage": i, "weights": list(weights), "cost": float(res.fun),
            "nit": int(res.nit), "nfev": int(res.nfev),
            "message": str(res.message), "seconds": elapsed,
            "terms": rep,
            "max_abs_preimage": float(np.abs(x).max()),
        })

        if stage_hook is not None:
            stage_hook(i, constrain_np(x), x.reshape(N, 4), rep)

    u = constrain_np(x)
    x_out = np.asarray(x, dtype=np.float64).reshape(N, 4)
    info = {
        "gate": gate, "variant": variant, "n_t": n_t, "dt": dt, "N": N,
        "trunc_list": list(trunc_list), "w_amp": w_amp, "seed": seed,
        "amp0": amp0, "hard_bound": hard_bound, "maxiter": maxiter,
        "init": warm,
        "band_mhz": list(BAND_MHZ), "ramp_ns": RAMP_NS,
        "eps_max_rad_per_us": float(EPS_MAX),
        # The one field train_est.py does not write. Everything else is schema-
        # identical, so a comparison script can diff the two JSONs directly.
        "pipeline": "eigh_adjoint",
        "stages": stages,
        "constraints": check_constraints(u, dt),
    }
    return u, x_out, info


def save(u, x, info, gate, variant):
    """Write the physical pulse, the raw pre-image, and the run metadata."""
    os.makedirs(PULSE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    pulse_path = os.path.join(PULSE_DIR, f"u_{gate}_{variant}.npy")
    x_path = os.path.join(PULSE_DIR, f"x_{gate}_{variant}.npy")
    json_path = os.path.join(LOG_DIR, f"est_{gate}_{variant}.json")
    np.save(pulse_path, u)
    np.save(x_path, x)
    with open(json_path, "w") as fh:
        json.dump(info, fh, indent=2)
    return pulse_path, x_path, json_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate", default="X", choices=sorted(kitten_code.IDEAL_LOGICAL_U))
    p.add_argument("--variant", default="est", choices=sorted(SCHEDULES))
    p.add_argument("--dt", type=float, default=DT, help="us (default 1 ns)")
    p.add_argument("--n-t", type=int, default=N_T)
    p.add_argument("--trunc", type=int, nargs="+", default=TRAIN_TRUNC,
                   help="training truncations (default single; see TRAIN_TRUNC)")
    p.add_argument("--w-amp", type=float, default=1.0)
    p.add_argument("--maxiter", type=int, default=600, help="per stage")
    p.add_argument("--seed", type=int, default=0,
                   help="cold-start RNG seed. The same seed gives the same "
                        "initial pulse as EST/train_est.py, which is what makes "
                        "the two pipelines' runs comparable.")
    p.add_argument("--amp0", type=float, default=20.0,
                   help="std of the random initial RAW controls, rad/us")
    p.add_argument("--hard-bound", type=float, default=60.0,
                   help="box on the RAW pre-image, not the physical amplitude "
                        "(which the C4 penalty carries)")
    init_group = p.add_mutually_exclusive_group()
    init_group.add_argument("--init", default=None,
                            help="warm-start from a saved PHYSICAL pulse "
                                 "(u_*.npy), via deramp. Prefer --init-x.")
    init_group.add_argument("--init-x", default=None,
                            help="warm-start from a saved raw PRE-IMAGE "
                                 "(x_*.npy). Exact; no deramp needed.")
    p.add_argument("--tag", default="eigh",
                   help="suffix for the output names (default 'eigh', giving "
                        "u_X_est_eigh.npy). Defaults to a non-empty value on "
                        "purpose so this driver cannot clobber the JAX "
                        "pipeline's u_<gate>_<variant>.npy artifacts.")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    name = args.variant if not args.tag else f"{args.variant}_{args.tag}"

    def stage_hook(i, u_stage, x_stage, rep):
        if args.no_save:
            return
        os.makedirs(PULSE_DIR, exist_ok=True)
        for prefix, arr in (("u", u_stage), ("x", x_stage)):
            path = os.path.join(PULSE_DIR,
                                f"{prefix}_{args.gate}_{name}_stage{i}.npy")
            np.save(path, arr)
            print(f"    -> {path}")

    u, x, info = train(gate=args.gate, variant=args.variant, n_t=args.n_t,
                       dt=args.dt, trunc_list=args.trunc, w_amp=args.w_amp,
                       maxiter=args.maxiter, seed=args.seed, amp0=args.amp0,
                       hard_bound=args.hard_bound, init=args.init,
                       init_x=args.init_x, stage_hook=stage_hook)

    print("\n--- constraints on the saved pulse ---")
    for k, v in info["constraints"].items():
        print(f"  {k}: {v}")

    if not args.no_save:
        pulse_path, x_path, json_path = save(u, x, info, args.gate, name)
        print(f"\nsaved {pulse_path}\n      {x_path}\n      {json_path}")


if __name__ == "__main__":
    main()
