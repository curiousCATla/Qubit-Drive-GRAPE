#!/usr/bin/env python3
"""
Efficiency of the three exact gradient paths of DGRAPE (dissipation_grape.ipynb §9):

    np_W     lindblad_np.fidelity_and_grad            hand-derived adjoint, W-matrix form
    np_aug   lindblad_np.fidelity_and_grad_augmented  hand-derived adjoint, augmented generator
    jax      lindblad_jax value_and_grad              reverse-mode autodiff (checkpointed scan)
    np_fwd   lindblad_np.fidelity                     forward only, the unit of cost

Each (method, n_c, N) runs in its own subprocess so its peak RSS (ru_maxrss) is its own.
Time per call is the MIN over repeats: this machine is shared, so wall-clock is noisy and
the minimum is the least-contaminated estimate. JAX's first call (trace + compile) is
reported separately. Same pulse, same Taylor (s, m), process fidelity of the X gate,
decay-only Lindbladian, readout included (n_r = 2).

    python3 DGRAPE/bench_gradients.py                 # full sweep -> tables/dgrape_gradient_benchmark.csv
    python3 DGRAPE/bench_gradients.py --n-c 7 --N 100 # subset
"""

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUT_CSV = os.path.join(REPO_ROOT, "tables", "dgrape_gradient_benchmark.csv")
METHODS = ("np_fwd", "np_W", "np_aug", "jax")
DT = 0.002


def run_one(method, n_c, N, repeats):
    import numpy as np
    from DGRAPE import lindblad_np as lnp
    from DGRAPE.code import process_targets
    from DGRAPE.model import Model

    model = Model(n_c)
    E, T, w = process_targets("X", n_c)
    u = np.random.default_rng(0).normal(0.0, 15.0, (N, 4))
    spec = lnp.spec_for_pulse(model, u, DT)
    first = None
    if method == "np_fwd":
        call = lambda: lnp.fidelity(model, u, E, T, w, DT, spec)                  # noqa: E731
    elif method == "np_W":
        call = lambda: lnp.fidelity_and_grad(model, u, E, T, w, DT, spec)         # noqa: E731
    elif method == "np_aug":
        call = lambda: lnp.fidelity_and_grad_augmented(model, u, E, T, w, DT, spec)  # noqa: E731
    elif method == "jax":
        from DGRAPE import lindblad_jax as ljx
        _, fg = ljx.make_fidelity_fns(model, E, T, w, DT)
        call = lambda: fg(u, *spec)[1].block_until_ready()                        # noqa: E731
        t0 = time.perf_counter()
        call()
        first = time.perf_counter() - t0
    else:
        raise ValueError(method)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 2**20 if sys.platform == "darwin" else rss / 2**10   # bytes on macOS, KiB on Linux
    return {"method": method, "n_c": n_c, "d": model.d, "N": N, "s": spec[0], "m": spec[1],
            "t_min_s": min(times), "t_median_s": sorted(times)[len(times) // 2],
            "repeats": repeats, "jax_first_call_s": first, "peak_rss_mb": rss_mb}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-c", type=int, nargs="+", default=[7, 10, 12, 14])
    p.add_argument("--N", type=int, nargs="+", default=[100, 500])
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default=OUT_CSV)
    p.add_argument("--one", nargs=3, metavar=("METHOD", "N_C", "N"), help=argparse.SUPPRESS)
    a = p.parse_args()

    if a.one:
        print(json.dumps(run_one(a.one[0], int(a.one[1]), int(a.one[2]), a.repeats)))
        return

    rows = []
    for N in a.N:
        for n_c in a.n_c:
            for method in a.methods:
                reps = a.repeats if N <= 100 else max(2, a.repeats - 1)
                out = subprocess.run([sys.executable, os.path.abspath(__file__), "--one",
                                      method, str(n_c), str(N), "--repeats", str(reps)],
                                     cwd=REPO_ROOT, capture_output=True, text=True, check=True)
                row = json.loads(out.stdout.strip().splitlines()[-1])
                rows.append(row)
                print(f"{method:7s} n_c={n_c:3d} N={N:4d}  t_min={row['t_min_s']:8.3f} s  "
                      f"rss={row['peak_rss_mb']:7.0f} MB"
                      + (f"  first={row['jax_first_call_s']:.1f} s"
                         if row["jax_first_call_s"] else ""), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
