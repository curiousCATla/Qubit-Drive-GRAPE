#!/usr/bin/env python3
"""
L-BFGS-B driver for open-system GRAPE on the Shirol chip -- the run logic of
core/optimizer.optimize_multi_state_pulse (experiments.ipynb §1) around the Lindblad
objective of DGRAPE/objective.py. Tested by 5-iteration smoke runs
(validation/test_dgrape.py); NOT yet used to train any gate.

Defaults are the experiments.ipynb / main.py production recipe (DGRAPE/recipe.py): bands
+-27 / +-33 MHz, 48 ns ramp, lambda_deriv 1e-5, lambda_amp 8e-5 on |I+iQ| > 25 rad/us,
lambda_disc 0.5, hard_amp_limit 25, smooth seeded cold start. They were tuned on the Heeres
chip; dissipation_grape.ipynb §12 lists what is still to validate here. Gate duration and dt
stay REQUIRED: they depend on this chip. The truncation list defaults to [12, 14, 16] (not the
Heeres [22, 24, 26]), evaluated as --n-jobs 3 parallel jobs, and the DGRAPE-only forbidden-state
penalty (lambda_forbid 1e-3 on storage Fock >= 10) is on; see DGRAPE/recipe.py.

Pulses go to pulses/dgrape/, never pulses/: every analysis/ and validation/ script locates
pulses there by name and assumes the Heeres cat code.

    python3 DGRAPE/train.py --gate X --duration-us 1.1 --dt 0.002 --max-iter 500
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.optimizer import make_smooth_warm_start      # noqa: E402
from core.ramp import constraint_report, deramp         # noqa: E402
from DGRAPE.objective import OpenGrapeObjective         # noqa: E402
from DGRAPE.recipe import RECIPE                         # noqa: E402

PULSE_DIR = os.path.join(REPO_ROOT, "pulses", "dgrape")


def initial_preimage(obj, seed, init_x=None, warm_start=None, warm_start_strict=True,
                     warm_start_amp=RECIPE["warm_start_amp"],
                     warm_start_cutoff_frac=RECIPE["warm_start_cutoff_frac"]):
    """(x0, kind, roundtrip_err), as core.optimizer: init_x is a raw pre-image used as-is;
    warm_start is a PHYSICAL pulse pulled back through the chain with core.ramp.deramp
    (or 'zero'); otherwise a smooth low-pass seeded cold start."""
    N = obj.N
    if init_x is not None and warm_start is not None:
        raise ValueError("pass init_x or warm_start, not both")
    if init_x is not None:
        x0 = np.load(init_x) if isinstance(init_x, str) else np.asarray(init_x, dtype=float)
        if x0.shape != (N, 4):
            raise ValueError(f"init_x has shape {x0.shape}, expected {(N, 4)}")
        return x0, "preimage", None
    if isinstance(warm_start, str) and warm_start in ("zero", "zeros"):
        return np.zeros((N, 4)), "zero", None
    if warm_start is not None:
        u_ws = np.load(warm_start) if isinstance(warm_start, str) else np.asarray(warm_start)
        try:
            x0, err = deramp(u_ws, obj.dt, obj.cav_band, obj.tra_band, obj.ramp_ns)
        except ValueError as exc:
            if warm_start_strict:
                raise
            print(f"WARNING: warm_start_strict=False, proceeding anyway.\n  {exc}")
            x0, err = np.asarray(u_ws, dtype=float), None
        return x0, "pulse", err
    x0 = make_smooth_warm_start(N, amp_max=warm_start_amp, cutoff_frac=warm_start_cutoff_frac,
                                seed=seed)
    return x0, "cold", None


def train(gate, N, dt, n_c=None, maxiter=1500, seed=42, init_x=None, warm_start=None,
          warm_start_strict=True, hard_amp_limit=RECIPE["hard_amp_limit"], backend="np",
          verbose=True, gtol=1e-12, ftol=1e-15, **obj_kw):
    """Returns (u, x, info). u is the physical pulse, x the raw pre-image with
    to_physical(x) == u (asserted), so x resumes the run exactly via init_x.

    gtol/ftol are far below scipy's defaults on purpose: from a random start F_pro is ~1e-4
    and every gradient entry ~1e-6, which the default gtol=1e-5 reads as converged at
    iteration 0.
    """
    obj = OpenGrapeObjective(gate, N, dt, n_c, backend, **obj_kw)
    if obj.penalties["forbid"] > 0 and min(obj.trunc) <= obj.forbid_fock_min:
        obj.close()
        raise ValueError(f"lambda_forbid > 0 but n_c={min(obj.trunc)} <= forbid_fock_min="
                         f"{obj.forbid_fock_min}: P_F is empty there, so the penalty would be "
                         "silently diluted. Train on larger n_c, lower forbid_fock_min, "
                         "or pass lambda_forbid=0.")
    try:
        return _run(obj, gate, N, dt, maxiter, seed, init_x, warm_start, warm_start_strict,
                    hard_amp_limit, verbose, gtol, ftol)
    finally:
        obj.close()


def _run(obj, gate, N, dt, maxiter, seed, init_x, warm_start, warm_start_strict,
         hard_amp_limit, verbose, gtol, ftol):
    x0, kind, roundtrip_err = initial_preimage(obj, seed, init_x, warm_start, warm_start_strict)
    # L-BFGS-B silently PROJECTS an infeasible start into the box; refuse instead.
    if np.abs(x0).max() > hard_amp_limit:
        raise ValueError(f"initial pre-image peak {np.abs(x0).max():.3f} exceeds "
                         f"hard_amp_limit={hard_amp_limit}; L-BFGS-B would silently clip it.")
    history = []
    t0 = time.time()
    cost0 = obj.cost(x0)

    def callback(xk):
        c, _, terms = obj.evaluate(xk)
        history.append({"it": len(history) + 1, "cost": c,
                        **{k: v for k, v in terms.items() if k != "F_pro"},
                        "F_pro": list(terms["F_pro"]), "t": time.time() - t0})
        if verbose:
            extra = f"  forbid {terms['forbid']:.3e}" if "forbid" in terms else ""
            print(f"it {len(history):4d}  cost {c:.6e}  F_pro {terms['F_pro']}{extra}",
                  flush=True)

    res = minimize(obj.cost, x0.ravel(), jac=obj.grad, method="L-BFGS-B",
                   bounds=[(-hard_amp_limit, hard_amp_limit)] * (N * 4),
                   options={"maxiter": maxiter, "gtol": gtol, "ftol": ftol},
                   callback=callback)
    x_final = res.x.reshape(N, 4)
    u_final = obj.to_physical(x_final)

    # Best-vs-final, as core.optimizer: return the best bare-F (u, x) PAIR seen during the
    # run if it beats the final point; never mix the two.
    F_final = float(np.mean(obj.fidelities(u_final, want_grad=False)))
    if obj.best["u"] is not None and obj.best["F"] > 0.5 and obj.best["F"] > F_final:
        u, x, picked = obj.best["u"].copy(), obj.best["x"].copy(), "best_seen"
    else:
        u, x, picked = u_final, x_final, "final"
    assert np.allclose(obj.to_physical(x), u, atol=1e-12), \
        "saved pre-image does not reproduce the saved pulse"

    per_trunc = obj.report(u)
    Fs = [r["F_pro"] for r in per_trunc]
    pinned = float(np.mean(np.abs(x) >= hard_amp_limit * (1 - 1e-9)))
    info = {
        "gate": gate, "N": N, "dt": dt, "seed": seed, "maxiter": maxiter,
        "message": str(res.message), "nit": int(res.nit), "n_evals": obj.n_evals,
        "wall_s": time.time() - t0, "cost0": cost0, "history": history,
        **obj.config(), "hard_amp_limit": hard_amp_limit,
        "warm_start_kind": kind, "roundtrip_err": roundtrip_err, "returned": picked,
        "per_trunc": per_trunc, "F_pro_mean": float(np.mean(Fs)),
        "max_pairwise_dF": float(max(Fs) - min(Fs)),
        # Check BOTH before trusting a pulse (root CLAUDE.md): the physical Eq. 19 limit
        # the penalty enforces, and the numerical box (at the bound = clipped, not converged).
        "max_drive_modulus": {"cav": float(np.hypot(u[:, 0], u[:, 1]).max()),
                              "tra": float(np.hypot(u[:, 2], u[:, 3]).max())},
        "max_abs_preimage": float(np.abs(x).max()),
        "frac_preimage_pinned": pinned,
        "constraints": constraint_report(u, dt, obj.cav_band, obj.tra_band),
    }
    if verbose:
        c, m = info["constraints"], info["max_drive_modulus"]
        print(f"returned {picked}; F_pro per truncation {Fs}; "
              f"C_forbid {[r['C_forbid'] for r in per_trunc]}")
        print(f"peak |I+iQ|: cavity {m['cav']:.2f}, transmon {m['tra']:.2f} rad/us "
              f"(amp_max {obj.penalties['amp_max']}); max|x| {info['max_abs_preimage']:.2f}, "
              f"{pinned:.2%} pinned at {hard_amp_limit}; endpoints "
              f"{c['endpoint_rel_to_peak']:.2%} of peak")
    return u, x, info


def save(u, x, info, tag):
    os.makedirs(PULSE_DIR, exist_ok=True)
    np.save(os.path.join(PULSE_DIR, f"u_{tag}.npy"), u)
    np.save(os.path.join(PULSE_DIR, f"x_{tag}.npy"), x)
    with open(os.path.join(PULSE_DIR, f"{tag}.json"), "w") as f:
        json.dump(info, f, indent=1, default=float)


def parse_band(values):
    """'F_LO F_HI' or 'none' (main.py convention)."""
    if len(values) == 1 and values[0].lower() == "none":
        return None
    if len(values) != 2:
        raise argparse.ArgumentTypeError(f"band needs 'F_LO F_HI' or 'none', got {values}")
    return (float(values[0]), float(values[1]))


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate", required=True, choices=["I", "X", "Y", "Z", "H", "T"])
    p.add_argument("--duration-us", type=float, required=True)
    p.add_argument("--dt", type=float, required=True, help="us")
    p.add_argument("--n-c", type=int, nargs="+", default=RECIPE["trunc_list"],
                   help="training truncations (a guess for this chip, not the Heeres list)")
    pen = RECIPE["penalties"]
    p.add_argument("--lambda-deriv", type=float, default=pen["deriv"])
    p.add_argument("--lambda-amp", type=float, default=pen["amp"])
    p.add_argument("--lambda-disc", type=float, default=pen["disc"])
    p.add_argument("--lambda-forbid", type=float, default=pen["forbid"],
                   help="forbidden-state penalty weight (0 disables)")
    p.add_argument("--forbid-fock-min", type=int, default=RECIPE["forbid_fock_min"],
                   help="P_F projects on storage Fock >= this")
    p.add_argument("--n-jobs", type=int, default=RECIPE["n_jobs"],
                   help="parallel truncation jobs (joblib)")
    p.add_argument("--amp-max", type=float, default=pen["amp_max"], help="rad/us")
    p.add_argument("--amp-norm", choices=["modulus", "quadrature"], default=RECIPE["amp_norm"])
    p.add_argument("--hard-amp-limit", type=float, default=RECIPE["hard_amp_limit"])
    p.add_argument("--cav-band", nargs="+", default=[str(v) for v in RECIPE["cav_band"]],
                   help="MHz: 'F_LO F_HI' or 'none'")
    p.add_argument("--tra-band", nargs="+", default=[str(v) for v in RECIPE["tra_band"]],
                   help="MHz: 'F_LO F_HI' or 'none'")
    p.add_argument("--ramp-ns", type=float, default=RECIPE["ramp_ns"], help="0 disables")
    p.add_argument("--dephasing", action="store_true")
    p.add_argument("--heating", action="store_true")
    p.add_argument("--higher-order", action="store_true")
    p.add_argument("--max-iter", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-x", default=None, help="raw pre-image x_*.npy (exact resume)")
    p.add_argument("--warm-start", default=None, help="physical pulse u_*.npy, or 'zero'")
    p.add_argument("--backend", choices=["np", "jax"], default="np")
    p.add_argument("--tag", default=None)
    return p


def main():
    a = build_arg_parser().parse_args()
    N = int(round(a.duration_us / a.dt))
    u, x, info = train(
        a.gate, N, a.dt, a.n_c, a.max_iter, a.seed, a.init_x, a.warm_start,
        hard_amp_limit=a.hard_amp_limit, backend=a.backend,
        cav_band=parse_band(a.cav_band), tra_band=parse_band(a.tra_band), ramp_ns=a.ramp_ns,
        penalties={"deriv": a.lambda_deriv, "amp": a.lambda_amp, "amp_max": a.amp_max,
                   "disc": a.lambda_disc, "forbid": a.lambda_forbid},
        forbid_fock_min=a.forbid_fock_min, n_jobs=a.n_jobs,
        amp_norm=a.amp_norm, dephasing=a.dephasing, heating=a.heating,
        include_higher_order=a.higher_order)
    tag = a.tag or f"{a.gate}_open"
    save(u, x, info, tag)
    print(json.dumps({k: v for k, v in info.items() if k != "history"}, indent=1, default=float))


if __name__ == "__main__":
    main()
