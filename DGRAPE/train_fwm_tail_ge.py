#!/usr/bin/env python3
"""
Training of Gertler-style FWM tails (notebooks/FWM_tails_ge.ipynb §4): an X gate (EsT or Ord, --gate) runs as
trained, then a tail of the transmon comb eps_T(t) and the mixing comb xi_M(t) (DGRAPE/fwm_tail_ge.py)
is optimized by L-BFGS-B on its pre-image x (N, 4), scored against the ideal X at the end of the tail.

Settings are fwm_tail_ge's recipe: n_c 16 for the tail on transmon {g, e}, the gate run at n_c 24
(three levels) and cut, n_r 2, one exact exponential per 2 ns step, readout traced out (--readout
empty to require it empty), decay set, higher-order terms on, caps |eps_T| <= 2 pi 1 MHz and
|xi_M| <= --cap (0.1 or 0.2) by modulus penalties, band +-12 MHz on both, 48 ns ramp, box |x| <= 25,
gtol 1e-12, best-vs-final.

The start is --start NAME (fwm_tail_ge.start_library: zero, gertler, sequential) or --init-x FILE (a
pre-image (N, 4) .npy), plus --noise (std as a FRACTION of each comb's cap; 0.05 is fwm_tail's 0.005 on
a 0.1 cap) with --seed. Everything is written to --outdir (default ~/fwm_runs/tailge, outside iCloud):
a checkpoint json + best pre-image after every iteration (resume with --init-x
<outdir>/x_<tag>_best.npy), and at the end b4_<tag>.npy ((N, 4) physical eps_T^I, eps_T^Q, xi_M^I,
xi_M^Q), x_<tag>.npy (pre-image) and <tag>.json. Copy them into pulses/dgrape/ when done (never pulses/).

On Windows (OpenBLAS) set OPENBLAS_NUM_THREADS at launch (1 when many runs share the machine).

Another chip (FWM_tails_ge_chips.ipynb): --chi-ge MHZ --chi-qr MHZ runs the tail on DGRAPE.ge_chip.derived(...)
(its comb-T cap, band and the start's t_sel follow the chip), and --gate-chip says where the gate ran:
shirol (default: the gate pulse as trained on Shirol's chip, the "transplant" proxy) or same (the gate
run on the tail's chip; use with a gate trained there).

    python DGRAPE/train_fwm_tail_ge.py --gate EsT --tail-us 2 --cap 0.2 --start gertler --tag X_tailge_est_T2_c20_g --dry-run
    python DGRAPE/train_fwm_tail_ge.py --gate EsT --tail-us 2 --cap 0.2 --start gertler --tag X_tailge_est_T2_c20_g --max-iter 600
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

from DGRAPE import fwm_tail_ge as ge                  # noqa: E402
from DGRAPE import ge_chip as gc                      # noqa: E402

DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "fwm_runs", "tailge")


def _dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def start_preimage(start, tail_us, cap, chip=None):
    """Pre-image (N, 4) from a start_library name or a .npy file."""
    ch = gc.resolve(chip)
    lib = ge.start_library(tail_us, cap, chip=ch)
    if start in lib:
        return ge.Chain(ge.n_steps(tail_us), band=ch.band, cap_T=ch.cap_T, cap_M=cap).preimage(lib[start])
    return np.load(start)


def make_objective(gate, tail_us, cap, n_c, gate_n_c, n_jobs, readout, target="X", chip=None, gate_chip=None):
    return ge.TailGEObjective(gate, tail_us, cap_M=cap, n_c=n_c, gate_n_c=gate_n_c, target=target,
                              readout=readout, n_jobs=n_jobs, chip=chip, gate_chip=gate_chip)


def dry_run(gate, start, tail_us, cap=ge.CAP_M, n_c=ge.N_C, gate_n_c=ge.GATE_N_C, n_jobs=3,
            readout=ge.READOUT, chip=None, gate_chip=None):
    """One cost + gradient evaluation of the start, timed. No optimizer."""
    x0 = start_preimage(start, tail_us, cap, chip)
    t0 = time.time()
    obj = make_objective(gate, tail_us, cap, n_c, gate_n_c, n_jobs, readout, chip=chip, gate_chip=gate_chip)
    t_build = time.time() - t0
    try:
        t0 = time.time()
        cost, g, terms = obj.evaluate(x0)
        t_eval = time.time() - t0
    finally:
        obj.close()
    out = {"gate": gate, "start": start, "tail_us": tail_us, "cap_M": cap, "N": obj.N, "n_c": n_c,
           "gate_n_c": gate_n_c, "trace_cut": obj.trace_cut, "n_jobs": n_jobs, "cost": cost,
           "grad_norm": float(np.linalg.norm(g)), "build_s": t_build, "eval_s": t_eval, **terms}
    print(json.dumps(out, indent=1, default=float), flush=True)
    return out


def train(gate, start, tag, outdir, tail_us, cap=ge.CAP_M, max_iter=600, seed=None, noise=0.0, n_c=ge.N_C,
          gate_n_c=ge.GATE_N_C, n_jobs=3, readout=ge.READOUT, max_hours=None, verbose=True, chip=None,
          gate_chip=None):
    x0 = start_preimage(start, tail_us, cap, chip)
    N = ge.n_steps(tail_us)
    if x0.shape != (N, 4):
        raise ValueError(f"start has shape {x0.shape}, expected {(N, 4)}")
    if noise > 0:
        x0 = x0 + noise * ge.U_BOX * np.random.default_rng(seed).normal(size=x0.shape)
    x0 = np.clip(x0, -ge.U_BOX, ge.U_BOX)
    obj = make_objective(gate, tail_us, cap, n_c, gate_n_c, n_jobs, readout, chip=chip, gate_chip=gate_chip)
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
            np.save(os.path.join(outdir, f"x_{tag}_best.npy"), obj.best["x"])
            _dump(os.path.join(outdir, f"{tag}_checkpoint.json"),
                  {"history": history, "best_F": obj.best["F"], "start": str(start)})
            if verbose:
                extra = "  ".join(f"{k} {terms[k]:.2e}" for k in ("amp", "deriv") if k in terms)
                print(f"it {len(history):4d}  cost {c:.6e}  F_pro {terms['F_pro']:.6f}  "
                      f"max|eps_T| {terms['max_eps_T_MHz']:.3f} MHz  max|xi_M| {terms['max_xi_M']:.4f}  "
                      f"{extra}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
            if max_hours and time.time() - t0 > 3600 * max_hours:
                raise Stop

        try:
            res = minimize(obj.cost, x0.ravel(), jac=obj.grad, method="L-BFGS-B",
                           bounds=[(-ge.U_BOX, ge.U_BOX)] * (N * 4),
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
    pinned = float(np.mean(np.abs(x) >= ge.U_BOX * (1 - 1e-9)))
    info = {"tag": tag, "gate": str(gate), "start": str(start), "target": obj.target, "seed": seed,
            "noise": noise, "tail_us": tail_us, "N": N, "n_c": n_c, "n_r": obj.n_r, "gate_n_c": gate_n_c,
            "levels": obj.levels, "trace_cut": obj.trace_cut, "readout": readout, "cap_M": cap,
            "cap_T": float(obj.cap[0]), "band_MHz": list(obj.band), "ramp_ns": obj.ramp_ns,
            "penalties": obj.pen, "max_iter": max_iter, "message": message, "nit": nit,
            "n_evals": obj.n_evals, "wall_s": time.time() - t0, "returned": picked, "cost0": cost0,
            "F_pro_start": terms0["F_pro"], "F_pro": terms["F_pro"], "terms": terms,
            "max_abs_preimage": float(np.abs(x).max()), "frac_preimage_pinned": pinned,
            **ge.comb_stats(b4), "history": history}
    if chip is not None:
        info["chip"] = obj.chip.table_row()
        info["gate_chip"] = None if obj.gate_chip is None else obj.gate_chip.table_row()
    np.save(os.path.join(outdir, f"b4_{tag}.npy"), b4)
    np.save(os.path.join(outdir, f"x_{tag}.npy"), x)
    _dump(os.path.join(outdir, f"{tag}.json"), info)
    if verbose:
        print(f"returned {picked}: F_pro {terms['F_pro']:.6f} (start {terms0['F_pro']:.6f}); "
              f"max|eps_T| {info['max_eps_T_MHz']:.3f} MHz, max|xi_M| {info['max_xi_M']:.4f}; "
              f"{pinned:.2%} of x pinned; {message}", flush=True)
    return b4, x, info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate", required=True, help="EsT, Ord (fwm_tail_ge.GATES) or an (N, 4) gate .npy")
    p.add_argument("--start", required=True, help="zero, gertler, sequential, or a pre-image .npy")
    p.add_argument("--tag", required=True)
    p.add_argument("--tail-us", type=float, required=True)
    p.add_argument("--cap", type=float, default=ge.CAP_M, help="|xi_M| cap")
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--max-iter", type=int, default=600)
    p.add_argument("--max-hours", type=float, default=None, help="wall limit; returns best-vs-final")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--noise", type=float, default=0.0, help="std of added noise, as a fraction of each cap")
    p.add_argument("--n-c", type=int, default=ge.N_C)
    p.add_argument("--gate-n-c", type=int, default=ge.GATE_N_C)
    p.add_argument("--n-jobs", type=int, default=3)
    p.add_argument("--readout", choices=("traced", "empty"), default=ge.READOUT)
    p.add_argument("--dry-run", action="store_true", help="one timed cost + gradient evaluation, no training")
    p.add_argument("--chi-ge", type=float, default=None, help="MHz; with --chi-qr: the tail on ge_chip.derived")
    p.add_argument("--chi-qr", type=float, default=None, help="MHz")
    p.add_argument("--gate-chip", choices=("shirol", "same"), default="shirol")
    a = p.parse_args()
    if (a.chi_ge is None) != (a.chi_qr is None):
        p.error("--chi-ge and --chi-qr go together")
    chip = None if a.chi_ge is None else gc.derived(a.chi_ge, a.chi_qr)
    gate_chip = chip if (chip is not None and a.gate_chip == "same") else None
    if a.dry_run:
        dry_run(a.gate, a.start, a.tail_us, a.cap, a.n_c, a.gate_n_c, a.n_jobs, a.readout, chip, gate_chip)
        return
    os.makedirs(a.outdir, exist_ok=True)
    train(a.gate, a.start, a.tag, a.outdir, a.tail_us, a.cap, a.max_iter, a.seed, a.noise, a.n_c, a.gate_n_c,
          a.n_jobs, a.readout, a.max_hours, chip=chip, gate_chip=gate_chip)


if __name__ == "__main__":
    main()
