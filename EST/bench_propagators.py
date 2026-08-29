"""
Per-gradient-call cost: EST/grape_jax.py (JAX, expm) vs EST/grape_eigh.py
(numpy, eigh + analytic adjoint).

    python EST/bench_propagators.py                 # X gate, production size
    python EST/bench_propagators.py --reps 9

Measures ONE `objective(x)` call -- cost and gradient together, exactly what
L-BFGS-B asks for on every iteration -- at the real training configuration:
gate X, N = 1000, dt = 1 ns, trunc_list = [20], the full band-limit/ramp/mask
constraint chain, stage-1 weights.

Protocol, and why each part is there
------------------------------------
* JIT warmup is excluded. The JAX objective compiles on its first call; timing
  that would measure the compiler. Compile time is reported separately because
  a training run pays it twice (once per stage) and it is a fair, if minor,
  point of comparison.
* Calls are INTERLEAVED A/B/A/B rather than run in two blocks, so a machine
  load that ramps during the benchmark hits both sides equally.
* The reported number is the MIN over reps. max/min is reported alongside as a
  contention indicator: this machine has previously swung the same measurement
  by an order of magnitude between idle and loaded, so a spread above ~1.3 means
  the numbers are not reportable and the benchmark should be re-run idle.
* Per-call cost is NEVER derived from a training log. `5029.6 s / 2138 nfev`
  looks like a clean per-call number and is not -- it folds in L-BFGS-B's
  line-search bookkeeping, two compiles, and hours of unrecorded machine load.

Both objectives are also checked against each other for agreement before being
timed, so a fast-but-wrong implementation cannot post a good number.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from EST.device import DT, N_T, n_steps
from EST.train_est import TRAIN_TRUNC

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_DIR = os.path.join(REPO_ROOT, "tables")

STAGE1_WEIGHTS = (1.0, 0.7, 7.0, 1.0)


def _build(gate, N, dt, n_t, trunc, weights):
    """Both objectives, built the way train drivers build them."""
    from EST.grape_jax import build_gate_objective as jax_build
    from EST.grape_eigh import build_gate_objective as eigh_build
    jax_obj, _, _ = jax_build(gate, N, dt=dt, n_t=n_t, trunc_list=trunc,
                              weights=weights)
    eigh_obj, _, _ = eigh_build(gate, N, dt=dt, n_t=n_t, trunc_list=trunc,
                                weights=weights)
    return {"expm_jax": jax_obj, "eigh_numpy": eigh_obj}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate", default="X")
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp0", type=float, default=20.0)
    p.add_argument("--trunc", type=int, nargs="+", default=TRAIN_TRUNC)
    p.add_argument("--csv", default=os.path.join(TABLE_DIR,
                                                 "est_propagator_benchmark.csv"))
    args = p.parse_args()

    dt, n_t = DT, N_T
    N = n_steps(args.gate, dt)
    x = args.amp0 * np.random.default_rng(args.seed).standard_normal(N * 4)

    print(f"gate={args.gate}  N={N}  dt={dt*1e3:.0f} ns  trunc={args.trunc}  "
          f"weights={STAGE1_WEIGHTS}")
    objs = _build(args.gate, N, dt, n_t, args.trunc, STAGE1_WEIGHTS)

    # --- warmup, and the correctness gate before any timing is reported ---
    warm = {}
    for name, obj in objs.items():
        t0 = time.perf_counter()
        c, g = obj(x)
        warm[name] = (time.perf_counter() - t0, float(c), np.asarray(g))
        obj(x)   # second warmup call; JAX caches after the first
    ref = warm["expm_jax"]
    cur = warm["eigh_numpy"]
    dcost = abs(cur[1] - ref[1])
    dgrad = np.max(np.abs(cur[2] - ref[2])) / max(np.max(np.abs(ref[2])), 1e-300)
    print(f"\nagreement before timing:  |dcost| = {dcost:.3e}   "
          f"rel|dgrad| = {dgrad:.3e}")
    if dgrad > 1e-6:
        raise SystemExit("the two objectives disagree; timing them is meaningless")
    for name, (t, _, _) in warm.items():
        print(f"  first-call (build+compile) {name:<11} {t:8.2f} s")

    # --- interleaved timing ---
    names = list(objs)
    samples = {n: [] for n in names}
    for _ in range(args.reps):
        for name in names:
            t0 = time.perf_counter()
            objs[name](x)
            samples[name].append(time.perf_counter() - t0)

    print(f"\nper objective(x) call, min of {args.reps} interleaved reps:")
    rows, contended = [], False
    for name in names:
        s = np.array(samples[name])
        spread = s.max() / s.min()
        contended |= spread > 1.3
        rows.append({"pipeline": name, "gate": args.gate, "N": N,
                     "trunc": "+".join(map(str, args.trunc)),
                     "min_s": s.min(), "median_s": float(np.median(s)),
                     "max_s": s.max(), "spread": spread, "reps": args.reps,
                     "first_call_s": warm[name][0]})
        print(f"  {name:<11} min {s.min():7.4f} s   median {np.median(s):7.4f} s"
              f"   max {s.max():7.4f} s   spread {spread:5.2f}x")

    base = rows[0]["min_s"]
    print()
    for r in rows:
        r["speedup_vs_expm_jax"] = base / r["min_s"]
        print(f"  {r['pipeline']:<11} {base / r['min_s']:5.2f}x vs expm_jax")

    if contended:
        print("\n  WARNING: spread > 1.3x on at least one pipeline. The machine "
              "was not idle;\n  these numbers are not reportable. Re-run with "
              "nothing else running.")
    for r in rows:
        r["note"] = "contended" if contended else "ok"

    os.makedirs(TABLE_DIR, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    # Append rather than overwrite, so contended runs stay on the record and a
    # favourable one cannot be cherry-picked by re-running.
    header = not os.path.exists(args.csv)
    df.to_csv(args.csv, mode="a", header=header, index=False)
    print(f"\nappended {len(df)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
