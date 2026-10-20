#!/usr/bin/env python3
"""
Stage 2 training (fwm_grape.ipynb §6): the d2 X gate frozen, only the FWM tone envelopes
optimized, by L-BFGS-B on the tone pre-image x (N, 2K) of DGRAPE/fwm.py's Stage2Objective. By
default the six tones (--paths 1 3 5, x is (N, 12)); --paths 1 2 3 4 5 6 7 8 9 trains all 18 (every
repair n-1 -> n up to 8 -> 9, fwm_grape.ipynb §7), and x is (N, 36). --band defaults to +-2 MHz for
the six tones and +-1 MHz for any other set (fwm.default_band).

Settings are DGRAPE/fwm.py's Stage 2 recipe: n_c = 16, CF4 S = 2, readout-traced target, decay set,
higher-order terms on, cap |beta_k| <= 0.1 and |xi_c| <= 0.1, band on each envelope, 48 ns
ramp, box |x| <= 25 with x = 250 beta, gtol 1e-12 (as DGRAPE/train.py), best-vs-final.

The start is a pre-image file (from DGRAPE/fwm_stage2_scan.py), optionally plus seeded noise.
Everything is written to --outdir (default ~/fwm_runs, outside iCloud, which renames files that are
rewritten while it syncs): a checkpoint json + best pre-image after every iteration (resume with
--init-x <outdir>/xb_<tag>_best.npy), and at the end u_<tag>.npy (the frozen d2 pulse),
b_<tag>.npy ((N, K) complex beta), xb_<tag>.npy (pre-image) and <tag>.json. Copy those into
pulses/dgrape/ when done (never pulses/).

    python DGRAPE/train_fwm.py --init-x ~/fwm_runs/start_seq_long_f1.0.npy --tag X_fwm_s2_a --max-iter 300
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

DT = 0.002
U_D2 = os.path.join(REPO_ROOT, "pulses", "dgrape", "u_X_open_d2.npy")
DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "fwm_runs")


def _dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def train(init_x, tag, outdir, max_iter=300, seed=None, noise=0.0, n_c=16, S=fwm.S_TRAIN,
          n_jobs=3, mixing=0.0, max_hours=None, verbose=True, paths=fwm.dev.CORRECTED_FOCK,
          band=fwm.BETA_BAND):
    u = np.load(U_D2)
    N = u.shape[0]
    n_cols = 2 * len(fwm.tone_list(paths))
    x0 = np.load(init_x) if isinstance(init_x, str) else np.asarray(init_x, dtype=float)
    if x0.shape != (N, n_cols):
        raise ValueError(f"init_x has shape {x0.shape}, expected {(N, n_cols)}")
    if noise > 0:
        x0 = x0 + noise * fwm.S_BETA * np.random.default_rng(seed).normal(size=x0.shape)
    x0 = np.clip(x0, -fwm.U_BOX, fwm.U_BOX)
    obj = fwm.Stage2Objective(u, DT, n_c=n_c, S=S, mixing=mixing, n_jobs=n_jobs, paths=paths,
                              band=band)
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
            np.save(os.path.join(outdir, f"xb_{tag}_best.npy"), obj.best["x"])
            _dump(os.path.join(outdir, f"{tag}_checkpoint.json"),
                  {"history": history, "best_F": obj.best["F"], "init_x": str(init_x)})
            if verbose:
                extra = "  ".join(f"{k} {terms[k]:.2e}" for k in ("amp", "comb", "deriv") if k in terms)
                print(f"it {len(history):4d}  cost {c:.6e}  F_pro {terms['F_pro']:.6f}  {extra}  "
                      f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
            if max_hours and time.time() - t0 > 3600 * max_hours:
                raise Stop

        try:
            res = minimize(obj.cost, x0.ravel(), jac=obj.grad, method="L-BFGS-B",
                           bounds=[(-fwm.U_BOX, fwm.U_BOX)] * (N * n_cols),
                           options={"maxiter": max_iter, "gtol": 1e-12, "ftol": 1e-15},
                           callback=callback)
            x_final, message, nit = res.x.reshape(N, n_cols), str(res.message), int(res.nit)
        except Stop:
            x_final, message, nit = obj._cache_x, f"stopped at the {max_hours} h wall limit", len(history)

        F_final = obj.evaluate(x_final)[2]["F_pro"]
        if obj.best["F"] > F_final:
            x, picked = obj.best["x"].copy(), "best_seen"
        else:
            x, picked = x_final, "final"
        b12 = obj.chain.to_physical(x)
        _, _, terms = obj.evaluate(x)
    finally:
        obj.close()
    beta = fwm.beta_from_real(b12)
    assert np.allclose(fwm.beta_from_real(obj.chain.to_physical(x)), beta, atol=1e-14)
    pinned = float(np.mean(np.abs(x) >= fwm.U_BOX * (1 - 1e-9)))
    info = {"tag": tag, "init_x": str(init_x), "seed": seed, "noise": noise, "n_c": n_c, "S": S,
            "mixing": mixing, "paths": list(obj.paths), "n_tones": obj.fw.n_tones,
            "max_iter": max_iter, "message": message, "nit": nit,
            "n_evals": obj.n_evals, "wall_s": time.time() - t0, "returned": picked,
            "cost0": cost0, "F_pro_start": terms0["F_pro"], "F_pro": terms["F_pro"], "terms": terms,
            "penalties": obj.pen, "cap": obj.cap, "band_MHz": obj.band, "ramp_ns": obj.ramp_ns,
            "s_beta": fwm.S_BETA, "max_abs_preimage": float(np.abs(x).max()),
            "frac_preimage_pinned": pinned, **fwm.pulse_stats(b12, obj.fw, DT, S), "history": history}
    np.save(os.path.join(outdir, f"u_{tag}.npy"), u)
    np.save(os.path.join(outdir, f"b_{tag}.npy"), beta)
    np.save(os.path.join(outdir, f"xb_{tag}.npy"), x)
    _dump(os.path.join(outdir, f"{tag}.json"), info)
    if verbose:
        print(f"returned {picked}: F_pro {terms['F_pro']:.6f} (start {terms0['F_pro']:.6f}); "
              f"max|beta| {info['max_beta']:.4f}, max|xi| {info['max_xi_1']:.4f}/{info['max_xi_2']:.4f}; "
              f"{pinned:.2%} of x pinned; {message}", flush=True)
    return beta, x, info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init-x", required=True, help="tone pre-image (N, 2K) .npy, K = 2 len(paths)")
    p.add_argument("--tag", required=True)
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--max-hours", type=float, default=None, help="wall limit; returns best-vs-final")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--noise", type=float, default=0.0, help="std of added beta noise (beta units)")
    p.add_argument("--n-c", type=int, default=16)
    p.add_argument("--S", type=int, default=fwm.S_TRAIN)
    p.add_argument("--n-jobs", type=int, default=3)
    p.add_argument("--mixing", type=float, default=0.0)
    p.add_argument("--paths", type=int, nargs="+", default=list(fwm.dev.CORRECTED_FOCK),
                   help="photon numbers n of the repairs n-1 -> n (one tone per comb each)")
    p.add_argument("--band", type=float, nargs=2, default=None,
                   help="envelope band in MHz (default +-2 for paths 1 3 5, else +-1)")
    a = p.parse_args()
    paths = tuple(a.paths)
    band = tuple(a.band) if a.band is not None else fwm.default_band(paths)
    os.makedirs(a.outdir, exist_ok=True)
    train(a.init_x, a.tag, a.outdir, a.max_iter, a.seed, a.noise, a.n_c, a.S, a.n_jobs, a.mixing,
          a.max_hours, paths=paths, band=band)


if __name__ == "__main__":
    main()
