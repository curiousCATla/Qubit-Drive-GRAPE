"""
Two-stage GRAPE driver for the EsT gate replication.

    python EST/train_est.py --gate X --variant est
    python EST/train_est.py --gate X --variant ord

Weight schedule (App. C):

    stage 1   w = (1, 0.7, 7, w_amp)   -> reaches ET fidelity ~0.83
    stage 2   w = (1, 0.1, 0, w_amp)   -> polishes ordinary fidelity, ET and
                                          velocity metrics move by ~1%

The 'ord' variant is the SAME code path with w2 = w3 = 0. That matters: the whole
replication rests on an EsT-vs-Ord comparison, so the control must differ from
EsT by weights alone and not by a separate implementation.

Optimizer is scipy L-BFGS-B with jac=True, matching core/optimizer.py:236. Box
bounds constrain the raw pre-image rather than the physical pulse (the band-limit
and ramp are applied inside the cost), which is the same caveat documented at
core/optimizer.py:73-76; the physical amplitude cap is carried by the C4 penalty
and verified after the fact by `check_constraints`.

Each run saves both the physical pulse u_<gate>_<variant>.npy and the raw
optimizer pre-image x_<gate>_<variant>.npy, per stage and at the end. Only u is
physically meaningful, but only x can resume a run exactly: u = constrain(x), and
inverting that (`deramp`) amplifies the pulse edges by up to ~151x, which is what
pushes a warm start outside the default box. Use --init-x, not --init, whenever
an x_*.npy exists.

Outputs land in pulses/est/ and NOT in pulses/. Every consumer of pulses/ --
validation/validate_logical_gates.py:77's GATE_PULSE_MAP, analysis/*.py -- hard-
codes the u_<g>_main.npy name and assumes the alpha=sqrt(3) four-component cat
code, so a kitten-code pulse dropped in pulses/ would be silently mis-scored.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import minimize

from core.fourier_cutoff import out_of_band_energy_fraction
from EST import kitten_code
from EST.device import (BAND_MHZ, DT, EPS_MAX, N_T, RAMP_NS, n_steps,
                        ramp_envelope)
from EST.grape_jax import build_gate_objective

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PULSE_DIR = os.path.join(REPO_ROOT, "pulses", "est")
LOG_DIR = os.path.join(REPO_ROOT, "logs")

# Training truncation. DELIBERATELY a single value, unlike core/optimizer.py's
# trunc_list=[20,24,28] default, and unlike device.TRUNC_LIST which stays the
# VALIDATION sweep. Measured cost at N=1000, dt=1 ns: 2.65 s per gradient
# evaluation at one truncation, 9.9 s at three -- a 3.7x tax on every L-BFGS-B
# step. The cat-code pipeline needs multi-truncation training because
# alpha=sqrt(3) cats carry population out to n ~ 15 and a pulse can exploit the
# truncation wall; the kitten code's highest code word is |4> and the drive is
# capped at 4 MHz, so the wall is much further from the dynamics. The protocol
# here is therefore train at one truncation, then VERIFY convergence on held-out
# truncations (EST/diagnostics.py) -- and retrain with --trunc 16 20 24 if that
# check fails, rather than paying 3.7x up front on the assumption that it will.
TRAIN_TRUNC = [20]

# (w1, w2, w3) per stage; the amplitude weight w4 is constant and passed separately.
SCHEDULES = {
    "est": [(1.0, 0.7, 7.0), (1.0, 0.1, 0.0)],
    "ord": [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    # 'le' (C1 + error-space fidelity, no C2) is the paper's third variant. It is
    # not implemented here: it needs an error-space terminal-fidelity term, which
    # is a new cost function rather than a reweighting. Out of scope for the
    # X-gate slice; see the plan.
}


def deramp(u, dt=DT, ramp_ns=RAMP_NS, atol=1e-10):
    """
    Invert the constraint chain: recover a raw optimizer pre-image x with
    constrain(x) == u, for warm-starting from a SAVED (physical) pulse.

    Saved pulses are constrain(x) = mask * (env * P x), where P is the
    band-limit projection. Dividing by the envelope is an exact inverse, not an
    approximation: u/env = mask * P x lies in the range of P, so re-projecting
    is idempotent and re-ramping restores u. Verified here rather than assumed --
    the roundtrip is asserted to `atol`.

    Note the amplification: env dips to ~0.7% of full scale on the first and last
    steps (EST/device.ramp_envelope), so |x| can exceed |u| by ~3x. Check the
    result against `hard_bound` before handing it to L-BFGS-B; a clipped warm
    start is no longer the pulse you meant to start from.
    """
    from EST.grape_jax import make_constrainer
    import jax.numpy as jnp

    u = np.asarray(u, dtype=np.float64)
    env = ramp_envelope(len(u), dt, ramp_ns)
    x = u / env[:, None]

    constrain = make_constrainer(len(u), dt, ramp_ns=ramp_ns)
    err = float(np.abs(np.asarray(constrain(jnp.asarray(x))) - u).max())
    if err > atol:
        raise ValueError(
            f"constraint-chain inversion failed: roundtrip error {err:.3e} > {atol:.1e}. "
            "The saved pulse is not in the range of the current constraint chain "
            "(different band, ramp_ns or dt than it was trained with?)."
        )
    return x, err


def check_constraints(u, dt=DT):
    """Post-hoc verification that the saved pulse obeys App. C's constraints."""
    magC = np.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2)
    magT = np.sqrt(u[:, 2] ** 2 + u[:, 3] ** 2)
    oob = out_of_band_energy_fraction(u, dt, BAND_MHZ, BAND_MHZ)
    mid = np.abs(u[len(u) // 2 - 20:len(u) // 2 + 20]).max()
    return {
        "max_eps_cavity_rad_per_us": float(magC.max()),
        "max_eps_transmon_rad_per_us": float(magT.max()),
        "eps_max_rad_per_us": float(EPS_MAX),
        "amplitude_cap_satisfied": bool(max(magC.max(), magT.max()) <= EPS_MAX * 1.01),
        "out_of_band_fraction": {k: float(v) for k, v in oob.items()},
        "endpoint_rel_to_mid": [float(np.abs(u[0]).max() / mid),
                                float(np.abs(u[-1]).max() / mid)],
    }


def train(gate="X", variant="est", n_t=N_T, dt=DT, trunc_list=TRAIN_TRUNC,
          w_amp=1.0, maxiter=600, seed=0, amp0=20.0, hard_bound=60.0,
          verbose=True, init=None, init_x=None, stage_hook=None):
    """
    Run the two-stage schedule and return (u_physical, x_preimage, info).

    Both are returned because only `u` is physically meaningful but only `x` can
    resume a run exactly: `u` is constrain(x), and inverting that map (`deramp`)
    amplifies the pulse edges by up to ~151x, which is what pushes a warm start
    outside the default box. See EST/README.md, "The box does bind on a warm
    start".

    init   : path to a saved (N,4) PHYSICAL pulse to warm-start from. The
             pre-image is recovered by `deramp`, exactly but with that
             amplification; needs a raised `hard_bound` in general.
    init_x : path to a saved (N,4) raw PRE-IMAGE to warm-start from. This is the
             exact route -- it is the vector L-BFGS-B stopped on, feasible in the
             box it ran under by construction. Preferred over `init` when
             available. Mutually exclusive with it.
    stage_hook : optional f(stage_index, u_physical, x_preimage, report) called
             after each stage, for saving the intermediate pulse and pre-image.
    """
    if variant not in SCHEDULES:
        raise KeyError(f"unknown variant {variant!r}; have {sorted(SCHEDULES)}")
    if init is not None and init_x is not None:
        raise ValueError("pass init or init_x, not both")

    N = n_steps(gate, dt)
    warm = None

    def _check_box(x, source):
        """L-BFGS-B silently projects an infeasible start into the box; refuse instead."""
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
        # No roundtrip assertion here, and none is possible: x IS the pre-image,
        # there is nothing to invert. The box check still applies -- an x saved
        # under --hard-bound 300 would be clipped by a rerun at the default 60.
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
            print(f"    N={N} dt={dt*1e3:.1f} ns  trunc_list={trunc_list}")
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
            # so a run can be audited for whether the box ever bound; it should
            # not, the physical cap is C4's job.
            "max_abs_preimage": float(np.abs(x).max()),
        })

        if stage_hook is not None:
            stage_hook(i, constrain_np(x), x.reshape(N, 4), rep)

    u = constrain_np(x)
    # (N,4) rather than the flat (4N,) L-BFGS-B works in, to mirror the pulse's
    # layout on disk. train() and --init-x are the only places that ravel it.
    x_out = np.asarray(x, dtype=np.float64).reshape(N, 4)
    info = {
        "gate": gate, "variant": variant, "n_t": n_t, "dt": dt, "N": N,
        "trunc_list": list(trunc_list), "w_amp": w_amp, "seed": seed,
        "amp0": amp0, "hard_bound": hard_bound, "maxiter": maxiter,
        "init": warm,
        "band_mhz": list(BAND_MHZ), "ramp_ns": RAMP_NS,
        "eps_max_rad_per_us": float(EPS_MAX),
        "stages": stages,
        "constraints": check_constraints(u, dt),
    }
    return u, x_out, info


def _print_report(label, rep):
    """rep is {n_c: {c1..c4}}; print the mean over truncations plus the spread."""
    keys = ["c1", "c2", "c3", "c4"]
    means = {k: np.mean([v[k] for v in rep.values()]) for k in keys}
    print(f"{label}  " + "  ".join(f"{k}={means[k]:.5f}" for k in keys)
          + f"   [F1={1-means['c1']:.5f}  F_ET={1-means['c2']:.5f}]")


def save(u, x, info, gate, variant):
    """
    Write the physical pulse, the raw pre-image, and the run metadata.

    u and x go to SEPARATE .npy files rather than one .npz because every
    downstream consumer -- EST/diagnostics.py, EST/compare_warmstart.py -- loads
    u_<gate>_<variant>.npy with a plain np.load, and an archive would break them.
    """
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp0", type=float, default=20.0,
                   help="std of the random initial RAW controls, rad/us. The "
                        "50 MHz band-limit keeps only ~10%% of the FFT bins of "
                        "white noise, so the physical pulse is much weaker than "
                        "this: amp0=20 lands at |eps|_max ~ 24 rad/us, just under "
                        "the 25.1 rad/us cap. amp0=8 gives ~10, which starts the "
                        "optimizer in a nearly flat region.")
    p.add_argument("--hard-bound", type=float, default=60.0,
                   help="box on the RAW pre-image, not the physical amplitude "
                        "(which the C4 penalty carries). A warm start needs a "
                        "larger box than a cold one: the deramp amplifies the "
                        "pulse edges by up to ~150x.")
    init_group = p.add_mutually_exclusive_group()
    init_group.add_argument("--init", default=None,
                            help="warm-start from a saved PHYSICAL pulse "
                                 "(u_*.npy). The pre-image is recovered by "
                                 "deramp, which amplifies the pulse edges by up "
                                 "to ~151x, so this usually needs a raised "
                                 "--hard-bound. Prefer --init-x when an x_*.npy "
                                 "exists for the run.")
    init_group.add_argument("--init-x", default=None,
                            help="warm-start from a saved raw PRE-IMAGE "
                                 "(x_*.npy). Exact: it is the vector L-BFGS-B "
                                 "stopped on, feasible at the bound it ran under "
                                 "by construction, so no deramp and no raised "
                                 "--hard-bound are needed.")
    p.add_argument("--tag", default=None,
                   help="suffix for the output names, e.g. --tag warm writes "
                        "u_X_est_warm.npy; keeps a rerun from clobbering the "
                        "pulse it warm-started from")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    name = args.variant if args.tag is None else f"{args.variant}_{args.tag}"

    def stage_hook(i, u_stage, x_stage, rep):
        """
        Save each stage's pulse AND pre-image, so the stage-1 -> stage-2 delta
        stays measurable and either stage can be resumed from exactly.
        """
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
