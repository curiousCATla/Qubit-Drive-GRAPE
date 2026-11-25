#!/usr/bin/env python3
"""
Benchmark the repo's GRAPE against QuTiP's GRAPE (the qutip-qtrl package).

What is compared
----------------
* Repo: core/grape_core.py (batched eigendecomposition propagators, exact
  adjoint gradient) driven by scipy L-BFGS-B, the same settings that
  core/optimizer.py uses (ftol 1e-12, gtol 1e-8, maxcor 10).
* QuTiP: qutip_qtrl.pulseoptim.create_pulse_optimizer with unitary dynamics
  (dyn_type="UNIT"). Its default propagator ("DIAG") also uses an
  eigendecomposition and an exact gradient, so the two codes do the SAME
  maths. The benchmark therefore compares implementations, not algorithms.
  QuTiP 5 moved GRAPE out of qutip.control into qutip-qtrl
  (pip install qutip-qtrl).

Both codes get the same Hamiltonian, built once by core.grape_core.make_hamiltonian
(transmon x cavity, n_t = 3, units rad/us) and handed to QuTiP as Qobj, the
same time step (dt = 2 ns, piecewise-constant controls) and the same targets.

Fidelity conventions
--------------------
QuTiP's phase-insensitive ("PSU") fidelity is f = |Tr(W^dag U V)| / k, with V the
k input states and W the k targets as columns. The repo's objective is
F = |sum_m <f_m|U|i_m>|^2 / k^2 (coherent_fidelity_multi_state; for k = 1 this
is fidelity_grad's |<f|U|i>|^2). So F = f^2, and dF/du = 2 f df/du. Every
number in the outputs is written as F (repo convention).

The four benchmarks (see QuTip/benchmark_results/README.md for results)
-----------------------------------------------------------------------
B1 agree   Same pulse -> same fidelity? (both codes solve the same problem)
B2 grad    Gradient accuracy against central finite differences
B3 timing  Seconds per fidelity+gradient evaluation vs size n_c and length N
B4 opt     Full L-BFGS-B optimisation from the same start: time to solution

Usage (from the repository root)
--------------------------------
    python QuTip/benchmark_qtrl.py run --part all [--quick] [--out DIR] [--threads 1]
    python QuTip/benchmark_qtrl.py report [--out DIR] [--dest QuTip/benchmark_results]

`run` writes one JSON file per case into DIR/raw (default ~/fwm_runs/qutip_bench)
and skips cases that already exist, so a killed run resumes where it stopped
(a half-finished case restarts). `report` has two stages: export_tables turns
DIR/raw into the CSVs in --dest (skipped when DIR/raw does not exist, e.g. on a
machine that did not run the benchmark), then write_report builds README.md and
both figures from the CSVs in --dest alone. Timings use single-threaded BLAS by
default (--threads), set before numpy is imported.
"""
import argparse
import json
import os
import sys


def _early_threads():
    """Set BLAS thread counts before numpy loads (must run first)."""
    n = "1"
    if "--threads" in sys.argv:
        n = sys.argv[sys.argv.index("--threads") + 1]
    for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = n
    return int(n)


THREADS = _early_threads()

import io                     # noqa: E402
import logging                # noqa: E402
import platform               # noqa: E402
import time                   # noqa: E402
import contextlib             # noqa: E402
import warnings               # noqa: E402

import numpy as np            # noqa: E402
from scipy.optimize import minimize   # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.grape_core import (  # noqa: E402
    make_hamiltonian, basis_state, fidelity_grad, coherent_fidelity_multi_state,
    fidelity_multi_state, dt as DT,
)
from core.cat_code import make_coherent_gate_factory  # noqa: E402
from core.optimizer import make_smooth_warm_start, optimize_multi_state_pulse  # noqa: E402

N_T = 3
AMP_BOX = 25.0             # production hard_amp_limit (rad/us)
FTOL = 1e-12               # core/optimizer.py L-BFGS-B settings
GTOL = 1e-8
MAXCOR = 10
THRESHOLDS = (1e-2, 1e-3, 1e-4)
DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "fwm_runs", "qutip_bench")
DEFAULT_DEST = os.path.join(REPO_ROOT, "QuTip", "benchmark_results")

# --------------------------------------------------------------------------
# Benchmark settings. FULL is the PC run; QUICK is a few-minute smoke test.
# --------------------------------------------------------------------------
FULL = dict(
    agree_cases=[("fock1", 10, 250), ("X", 22, 550), ("X", 26, 550)],
    agree_seeds=[1, 2, 3],
    agree_saved_pulse="pulses/u_X_main.npy",
    grad_cases=[("fock1", 10, 250), ("X", 14, 550)],
    grad_entries=20,
    timing_nc=[6, 10, 14, 18, 22, 26, 30],
    timing_nc_N=550,
    timing_N=[100, 275, 550, 1100],
    timing_N_nc=14,
    timing_repeats=5,
    timing_repeats_slow=3,          # qtrl repeats when one call takes > 10 s
    timing_prop_types=["DIAG", "FRECHET"],
    opt_problems=[
        dict(name="A_fock1", problem="fock1", n_c=10, N=250, seeds=list(range(42, 52)),
             maxiter=1000, wall_cap_s=1800),
        # maxiter 1500 = main.py --max-iter (production)
        dict(name="B_Xgate", problem="X", n_c=22, N=550, seeds=list(range(42, 50)),
             maxiter=1500, wall_cap_s=7200),
    ],
)
QUICK = dict(
    agree_cases=[("fock1", 6, 60), ("X", 8, 60)],
    agree_seeds=[1],
    agree_saved_pulse=None,
    grad_cases=[("fock1", 6, 40), ("X", 8, 40)],
    grad_entries=6,
    timing_nc=[4, 6, 8],
    timing_nc_N=100,
    timing_N=[50, 100],
    timing_N_nc=6,
    timing_repeats=2,
    timing_repeats_slow=1,
    timing_prop_types=["DIAG", "FRECHET"],
    opt_problems=[
        dict(name="A_fock1", problem="fock1", n_c=6, N=150, seeds=[42],
             maxiter=60, wall_cap_s=120),
        dict(name="B_Xgate", problem="X", n_c=8, N=100, seeds=[42],
             maxiter=20, wall_cap_s=120),
    ],
)


# --------------------------------------------------------------------------
# Problem definitions (shared by both codes)
# --------------------------------------------------------------------------
def targets(problem, n_c):
    """Input and target state lists for a problem at truncation n_c."""
    if problem == "fock1":            # |g,0> -> |g,1>
        return [basis_state(N_T, n_c, 0, 0)], [basis_state(N_T, n_c, 0, 1)]
    if problem == "X":                # cat-code logical X, production pairs
        pairs = make_coherent_gate_factory("X")(n_c=n_c, n_t=N_T)
        return [p[0] for p in pairs], [p[1] for p in pairs]
    raise ValueError(problem)


def repo_fid(u, H0, Hc, psi_i, psi_f, want_grad=True):
    """Repo objective F (and dF/du, shape (N, 4))."""
    if len(psi_i) == 1:
        return fidelity_grad(u, H0, Hc, psi_i[0], psi_f[0], DT, want_grad=want_grad)
    return coherent_fidelity_multi_state(u, H0, Hc, psi_i, psi_f, DT, want_grad=want_grad)


def start_pulse(N, seed):
    """Same smooth random start the repo's optimizer uses (peak 4 rad/us)."""
    return make_smooth_warm_start(N, amp_max=4.0, cutoff_frac=0.04, seed=seed)


# --------------------------------------------------------------------------
# QuTiP (qutip-qtrl) side
# --------------------------------------------------------------------------
def qtrl_optimizer(H0, Hc, psi_i, psi_f, n_c, N, prop_type="DEF",
                   fid_err_targ=0.0, min_grad=0.0, max_iter=10**6,
                   max_wall_time=10**9, gen_stats=False, float64_tau=True):
    """Build a qutip-qtrl GRAPE optimizer for exactly the repo's problem.

    float64_tau: qutip-qtrl 0.2.0 builds its time-step array in float32 when it
    is given num_tslots + evo_time (dynamics.init_timeslots, np.ones(..., dtype="f")),
    so each step is 2 ns - 2.3e-10 ns and every propagator is off by ~1e-7
    relative. Passing an explicit float64 `tau` removes that; B1 reports both.
    """
    import qutip as qt
    import qutip_qtrl.pulseoptim as cpo

    dims = [[N_T, n_c], [N_T, n_c]]
    drift = qt.Qobj(H0, dims=dims)
    ctrls = [qt.Qobj(h, dims=dims) for h in Hc]
    k = len(psi_i)
    V = qt.Qobj(np.column_stack(psi_i), dims=[[N_T, n_c], [k]])
    W = qt.Qobj(np.column_stack(psi_f), dims=[[N_T, n_c], [k]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optim = cpo.create_pulse_optimizer(
            drift, ctrls, V, W,
            **(dict(tau=np.full(N, DT)) if float64_tau else dict(num_tslots=N, evo_time=N * DT)),
            amp_lbound=-AMP_BOX, amp_ubound=AMP_BOX,
            fid_err_targ=fid_err_targ, min_grad=min_grad, max_iter=max_iter,
            max_wall_time=max_wall_time, alg="GRAPE", optim_method="FMIN_L_BFGS_B",
            method_params={"accuracy_factor": FTOL / np.finfo(float).eps,
                           "max_metric_corr": MAXCOR},
            dyn_type="UNIT", prop_type=prop_type, phase_option="PSU",
            init_pulse_type="ZERO", log_level=logging.WARN, gen_stats=gen_stats,
        )
    return optim


def qtrl_set(optim, u):
    optim.dynamics.initialize_controls(np.asarray(u, dtype=float).copy())


def qtrl_fid(optim, u, want_grad=True):
    """Return (F, dF/du) from qtrl, in the repo's F = f^2 convention."""
    x = np.asarray(u, dtype=float).ravel()
    err = optim.fid_err_func_wrapper(x)
    f = 1.0 - err
    if not want_grad:
        return f ** 2, None
    g_err = optim.fid_err_grad_wrapper(x).reshape(u.shape)   # d(1 - f)/du
    return f ** 2, -2.0 * f * g_err


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------
def environment():
    import scipy
    info = dict(
        platform=platform.platform(), machine=platform.machine(),
        processor=platform.processor(), cpu_count=os.cpu_count(),
        python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, blas_threads=THREADS,
    )
    try:
        import qutip, qutip_qtrl
        info["qutip"] = qutip.__version__
        info["qutip_qtrl"] = qutip_qtrl.__version__
    except ImportError:
        info["qutip"] = info["qutip_qtrl"] = "missing"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            np.show_config()
        except Exception:
            pass
    txt = buf.getvalue()
    for name in ("accelerate", "openblas", "mkl", "blis"):
        if name in txt.lower():
            info["blas"] = name
            break
    else:
        info["blas"] = "unknown"
    return info


class Store:
    def __init__(self, out):
        self.raw = os.path.join(out, "raw")
        os.makedirs(self.raw, exist_ok=True)

    def path(self, key):
        return os.path.join(self.raw, key + ".json")

    def has(self, key):
        return os.path.exists(self.path(key))

    def save(self, key, obj):
        tmp = self.path(key) + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1, default=float)
        os.replace(tmp, self.path(key))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --------------------------------------------------------------------------
# B1: agreement
# --------------------------------------------------------------------------
def run_agree(cfg, store):
    for problem, n_c, N in cfg["agree_cases"]:
        H0, Hc = make_hamiltonian(N_T, n_c)
        psi_i, psi_f = targets(problem, n_c)
        pulses = [(f"random_s{s}", start_pulse(N, s) * 3.0) for s in cfg["agree_seeds"]]
        sp = cfg["agree_saved_pulse"]
        if problem == "X" and sp and os.path.exists(os.path.join(REPO_ROOT, sp)):
            u = np.load(os.path.join(REPO_ROOT, sp))
            if u.shape[0] == N:
                pulses.append((os.path.basename(sp), u))
        optim = qtrl_optimizer(H0, Hc, psi_i, psi_f, n_c, N)
        optim32 = qtrl_optimizer(H0, Hc, psi_i, psi_f, n_c, N, float64_tau=False)
        for label, u in pulses:
            key = f"b1_{problem}_nc{n_c}_N{N}_{label}"
            if store.has(key):
                continue
            F_r, _ = repo_fid(u, H0, Hc, psi_i, psi_f, want_grad=False)
            qtrl_set(optim, u)
            F_q, _ = qtrl_fid(optim, u, want_grad=False)
            qtrl_set(optim32, u)
            F_32, _ = qtrl_fid(optim32, u, want_grad=False)
            store.save(key, dict(bench="B1", problem=problem, n_c=n_c, N=N, pulse=label,
                                 F_repo=float(F_r), F_qutip=float(F_q),
                                 abs_diff=float(abs(F_r - F_q)),
                                 F_qutip_float32_tau=float(F_32),
                                 abs_diff_float32_tau=float(abs(F_r - F_32))))
            log(f"B1 {key}: F_repo {F_r:.12f}  F_qutip {F_q:.12f}  diff {abs(F_r - F_q):.2e}"
                f"  (float32 tau: {abs(F_r - F_32):.2e})")


# --------------------------------------------------------------------------
# B2: gradient accuracy
# --------------------------------------------------------------------------
def run_grad(cfg, store, h=1e-6):
    for problem, n_c, N in cfg["grad_cases"]:
        key = f"b2_{problem}_nc{n_c}_N{N}"
        if store.has(key):
            continue
        H0, Hc = make_hamiltonian(N_T, n_c)
        psi_i, psi_f = targets(problem, n_c)
        u = start_pulse(N, 7) * 3.0
        rng = np.random.default_rng(0)
        entries = [(int(rng.integers(N)), int(rng.integers(4))) for _ in range(cfg["grad_entries"])]

        def F_only(v):
            return repo_fid(v, H0, Hc, psi_i, psi_f, want_grad=False)[0]

        fd = []
        for k, j in entries:
            up, um = u.copy(), u.copy()
            up[k, j] += h
            um[k, j] -= h
            fd.append((F_only(up) - F_only(um)) / (2 * h))
        fd = np.array(fd)
        scale = np.max(np.abs(fd))
        res = dict(bench="B2", problem=problem, n_c=n_c, N=N, h=h,
                   n_entries=len(entries), fd_max_abs=float(scale))
        _, g_r = repo_fid(u, H0, Hc, psi_i, psi_f)
        g_sel = np.array([g_r[k, j] for k, j in entries])
        res["repo_max_rel_err"] = float(np.max(np.abs(g_sel - fd)) / scale)
        for pt in ("DIAG", "FRECHET"):
            optim = qtrl_optimizer(H0, Hc, psi_i, psi_f, n_c, N, prop_type=pt)
            qtrl_set(optim, u)
            _, g_q = qtrl_fid(optim, u)
            g_sel = np.array([g_q[k, j] for k, j in entries])
            res[f"qutip_{pt}_max_rel_err"] = float(np.max(np.abs(g_sel - fd)) / scale)
            res[f"repo_vs_qutip_{pt}_max_rel_diff"] = float(
                np.max(np.abs(g_q - g_r)) / np.max(np.abs(g_r)))
        store.save(key, res)
        log(f"B2 {key}: " + ", ".join(f"{k}={v:.2e}" for k, v in res.items()
                                       if k.endswith("err") or k.endswith("diff")))


# --------------------------------------------------------------------------
# B3: cost per evaluation
# --------------------------------------------------------------------------
def _time_calls(fn, xs):
    ts = []
    for x in xs:
        t0 = time.perf_counter()
        fn(x)
        ts.append(time.perf_counter() - t0)
    return ts


def run_timing(cfg, store):
    points = [("nc", n_c, cfg["timing_nc_N"]) for n_c in cfg["timing_nc"]]
    points += [("N", cfg["timing_N_nc"], N) for N in cfg["timing_N"]]
    for sweep, n_c, N in points:
        key = f"b3_{sweep}_nc{n_c}_N{N}"
        if store.has(key):
            continue
        H0, Hc = make_hamiltonian(N_T, n_c)
        psi_i, psi_f = targets("X", n_c)
        base = start_pulse(N, 11) * 3.0
        rng = np.random.default_rng(1)

        def fresh(n):   # a new pulse every call, so no cache can be hit
            return [base + 1e-3 * rng.standard_normal(base.shape) for _ in range(n + 1)]

        res = dict(bench="B3", sweep=sweep, n_c=n_c, d=N_T * n_c, N=N, problem="X")
        R = cfg["timing_repeats"]
        # repo: warm-up call, then repeats
        xs = fresh(R)
        repo_fid(xs[0], H0, Hc, psi_i, psi_f)
        res["repo_fg_s"] = float(np.median(_time_calls(lambda v: repo_fid(v, H0, Hc, psi_i, psi_f), xs[1:])))
        res["repo_f_s"] = float(np.median(_time_calls(
            lambda v: repo_fid(v, H0, Hc, psi_i, psi_f, want_grad=False), fresh(R)[1:])))
        for pt in cfg["timing_prop_types"]:
            optim = qtrl_optimizer(H0, Hc, psi_i, psi_f, n_c, N, prop_type=pt)
            xs = fresh(R)
            qtrl_set(optim, xs[0])
            t0 = time.perf_counter()
            qtrl_fid(optim, xs[0])
            first = time.perf_counter() - t0
            Rq = R if first < 10 else cfg["timing_repeats_slow"]
            res[f"qutip_{pt}_fg_s"] = float(np.median(_time_calls(lambda v: qtrl_fid(optim, v), xs[1:Rq + 1])))
            res[f"qutip_{pt}_f_s"] = float(np.median(_time_calls(
                lambda v: qtrl_fid(optim, v, want_grad=False), fresh(Rq)[1:])))
        res["speedup_fg_DIAG"] = res["qutip_DIAG_fg_s"] / res["repo_fg_s"]
        store.save(key, res)
        log(f"B3 {key}: repo {res['repo_fg_s']:.4f} s, qutip DIAG {res['qutip_DIAG_fg_s']:.4f} s "
            f"(x{res['speedup_fg_DIAG']:.1f})")

    # Where QuTiP spends its time: its own stats at one mid-size point.
    n_c = cfg["timing_nc"][-2]
    N = cfg["timing_nc_N"]
    key = f"b3_qutip_profile_nc{n_c}_N{N}"
    if not store.has(key):
        H0, Hc = make_hamiltonian(N_T, n_c)
        psi_i, psi_f = targets("X", n_c)
        optim = qtrl_optimizer(H0, Hc, psi_i, psi_f, n_c, N, gen_stats=True)
        u = start_pulse(N, 11) * 3.0
        qtrl_set(optim, u)
        for r in range(3):
            qtrl_fid(optim, u + 1e-3 * r)
        st = optim.dynamics.stats
        parts = dict(hamiltonians=st.wall_time_dyn_gen_compute,
                     propagators=st.wall_time_prop_compute,
                     forward=st.wall_time_fwd_prop_compute,
                     onward=st.wall_time_onwd_prop_compute,
                     gradient=st.wall_time_gradient_compute)
        store.save(key, dict(bench="B3profile", n_c=n_c, N=N, calls=3,
                             **{k: float(v) for k, v in parts.items()}))
        log(f"B3 {key}: " + ", ".join(f"{k} {v:.2f} s" for k, v in parts.items()))


# --------------------------------------------------------------------------
# B4: end-to-end optimisation
# --------------------------------------------------------------------------
def run_repo_opt(H0, Hc, psi_i, psi_f, N, u0, maxiter, wall_cap):
    trace = []
    t0 = time.perf_counter()

    def objective(x):
        F, g = repo_fid(x.reshape(N, 4), H0, Hc, psi_i, psi_f)
        trace.append((time.perf_counter() - t0, float(F)))
        return -F, -g.ravel()

    def callback(xk):
        if time.perf_counter() - t0 > wall_cap:
            raise StopIteration

    res = minimize(objective, u0.ravel(), jac=True, method="L-BFGS-B",
                   bounds=[(-AMP_BOX, AMP_BOX)] * (4 * N), callback=callback,
                   options=dict(maxiter=maxiter, ftol=FTOL, gtol=GTOL, maxcor=MAXCOR))
    wall = time.perf_counter() - t0
    F_final = repo_fid(res.x.reshape(N, 4), H0, Hc, psi_i, psi_f, want_grad=False)[0]
    return dict(wall_s=wall, iterations=int(res.nit), evaluations=int(res.nfev),
                F_final=float(F_final), message=str(res.message), trace=trace)


def run_qtrl_opt(H0, Hc, psi_i, psi_f, n_c, N, u0, maxiter, wall_cap):
    optim = qtrl_optimizer(H0, Hc, psi_i, psi_f, n_c, N, fid_err_targ=1e-12,
                           min_grad=GTOL, max_iter=maxiter, max_wall_time=wall_cap)
    qtrl_set(optim, u0)
    trace = []
    t0 = [None]
    fid = optim.dynamics.fid_computer
    f_wrap, g_wrap = optim.fid_err_func_wrapper, optim.fid_err_grad_wrapper

    def f_traced(*args):
        if t0[0] is None:
            t0[0] = time.perf_counter()
        try:
            return f_wrap(*args)
        except Exception:      # goal reached etc.: the gradient call will not come
            trace.append((time.perf_counter() - t0[0], float(fid.get_fidelity() ** 2)))
            raise

    def g_traced(*args):
        try:
            return g_wrap(*args)
        finally:
            trace.append((time.perf_counter() - t0[0], float(fid.get_fidelity() ** 2)))

    optim.fid_err_func_wrapper = f_traced      # instance attributes win the lookup
    optim.fid_err_grad_wrapper = g_traced
    t_start = time.perf_counter()
    result = optim.run_optimization()
    wall = time.perf_counter() - t_start
    return dict(wall_s=wall, iterations=int(result.num_iter),
                evaluations=int(optim.num_fid_func_calls),
                F_final=float((1.0 - result.fid_err) ** 2),
                message=str(result.termination_reason), trace=trace)


def run_production_entry(problem, n_c, N, seed, maxiter):
    """The repo's own entry point with every constraint switched off."""
    factory = (make_coherent_gate_factory("X") if problem == "X"
               else (lambda n_c, n_t: list(zip(*targets("fock1", n_c)))))
    fid_fn = coherent_fidelity_multi_state if problem == "X" else fidelity_multi_state
    t0 = time.perf_counter()
    u, info = optimize_multi_state_pulse(
        factory, trunc_list=[n_c], n_t=N_T, N=N, dt=DT,
        penalties={"deriv": 0.0, "amp": 0.0, "amp_max": AMP_BOX, "disc": 0.0},
        warm_start_seed=seed, n_jobs=1, maxiter=maxiter, ramp_ns=None,
        cav_band=None, tra_band=None, hard_amp_limit=AMP_BOX,
        fidelity_fn=fid_fn, verbose=False)
    return dict(wall_s=time.perf_counter() - t0, iterations=int(info["iterations"]),
                F_final=float(info["final_fidelity"]), message=str(info["message"]))


def run_opt(cfg, store):
    for P in cfg["opt_problems"]:
        n_c, N = P["n_c"], P["N"]
        H0, Hc = make_hamiltonian(N_T, n_c)
        psi_i, psi_f = targets(P["problem"], n_c)
        for seed in P["seeds"]:
            u0 = start_pulse(N, seed)
            for code in ("repo", "qutip"):
                key = f"b4_{P['name']}_nc{n_c}_N{N}_s{seed}_{code}"
                if store.has(key):
                    continue
                log(f"B4 {key}: start")
                if code == "repo":
                    r = run_repo_opt(H0, Hc, psi_i, psi_f, N, u0, P["maxiter"], P["wall_cap_s"])
                else:
                    r = run_qtrl_opt(H0, Hc, psi_i, psi_f, n_c, N, u0, P["maxiter"], P["wall_cap_s"])
                r.update(bench="B4", name=P["name"], problem=P["problem"], n_c=n_c, N=N,
                         seed=seed, code=code, maxiter=P["maxiter"], wall_cap_s=P["wall_cap_s"])
                store.save(key, r)
                log(f"B4 {key}: F {r['F_final']:.8f}, {r['iterations']} it, "
                    f"{r['evaluations']} ev, {r['wall_s']:.1f} s ({r['message']})")
        seed = P["seeds"][0]
        key = f"b4_{P['name']}_nc{n_c}_N{N}_s{seed}_repo_entry"
        if not store.has(key):
            r = run_production_entry(P["problem"], n_c, N, seed, P["maxiter"])
            r.update(bench="B4entry", name=P["name"], n_c=n_c, N=N, seed=seed)
            store.save(key, r)
            log(f"B4 {key}: F {r['F_final']:.8f}, {r['iterations']} it, {r['wall_s']:.1f} s")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _load(out):
    """Raw JSON rows from <out>/raw (written by `run`), and the run's environment."""
    raw = os.path.join(out, "raw")
    rows = []
    for name in sorted(os.listdir(raw)):
        if name.endswith(".json"):
            with open(os.path.join(raw, name)) as fh:
                rows.append(json.load(fh))
    env_path = os.path.join(out, "environment.json")
    env = json.load(open(env_path)) if os.path.exists(env_path) else environment()
    return rows, env


def _md_table(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt and v == v and v is not None:
                cells.append(fmt[c].format(v))
            elif isinstance(v, float):
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _time_to(trace, thr):
    best = 0.0
    for t, F in trace:
        best = max(best, F)
        if 1.0 - best <= thr:
            return t
    return float("nan")


# Report names of the two test problems (B4 names; B1-B3 use the `problem` key).
PROBLEMS = {
    "A_fock1": dict(problem="fock1", k=1,
                    title=r"Fock state $|g,0\rangle \to |g,1\rangle$"),
    "B_Xgate": dict(problem="X", k=2, title=r"Cat-code logical X"),
}
CODE_STYLE = {   # same colours as the B3 figure (repo C0, QuTiP DIAG C1)
    "repo": dict(color="C0", label="repo (grape_core)"),
    "qutip": dict(color="C1", label="QuTiP qtrl, DIAG"),
}


def _sci(x, digits=1):
    """x as LaTeX scientific notation, e.g. 1.5 \\times 10^{-13} (for use inside $...$)."""
    if x != x:
        return r"\text{nan}"
    if x == 0:
        return "0"
    m, e = f"{x:.{digits}e}".split("e")
    if m == "1":
        return rf"10^{{{int(e)}}}"
    return rf"{m} \times 10^{{{int(e)}}}"


def _fill(template, **kw):
    """Replace <<key>> placeholders (LaTeX braces make str.format awkward)."""
    for k, v in kw.items():
        template = template.replace(f"<<{k}>>", str(v))
    assert "<<" not in template, template
    return template


def export_tables(out, dest):
    """Stage 1 (needs <out>/raw, i.e. the machine that ran the benchmark): raw JSON -> CSVs."""
    import pandas as pd

    rows, env = _load(out)
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "environment.json"), "w") as fh:
        json.dump(env, fh, indent=1)
    by = lambda b: [r for r in rows if r.get("bench") == b]  # noqa: E731

    b1 = pd.DataFrame(by("B1"))
    if len(b1):
        b1[["problem", "n_c", "N", "pulse", "F_repo", "F_qutip", "abs_diff",
            "abs_diff_float32_tau"]].to_csv(os.path.join(dest, "b1_agreement.csv"), index=False)
    b2 = pd.DataFrame(by("B2"))
    if len(b2):
        b2.to_csv(os.path.join(dest, "b2_gradient.csv"), index=False)
    b3 = pd.DataFrame(by("B3"))
    if len(b3):
        b3.sort_values(["sweep", "n_c", "N"]).to_csv(os.path.join(dest, "b3_timing.csv"), index=False)
    prof = by("B3profile")
    if prof:
        p = prof[0]
        parts = ("hamiltonians", "propagators", "forward", "onward", "gradient")
        tot = sum(p[k] for k in parts)
        pd.DataFrame([dict(n_c=p["n_c"], N=p["N"], calls=p["calls"], part=k, seconds=p[k],
                           share_percent=100 * p[k] / tot) for k in parts]).to_csv(
            os.path.join(dest, "b3_qutip_profile.csv"), index=False)
    b4 = by("B4")
    if b4:
        trace_rows, summ = [], []
        for r in b4:
            for i, (t, F) in enumerate(r["trace"]):
                trace_rows.append(dict(name=r["name"], seed=r["seed"], code=r["code"],
                                       evaluation=i + 1, time_s=t, F=F, infidelity=1 - F))
            s = {k: r[k] for k in ("name", "n_c", "N", "seed", "code", "iterations",
                                   "evaluations", "wall_s", "F_final", "message")}
            s["infidelity_final"] = 1 - r["F_final"]
            s["s_per_evaluation"] = r["wall_s"] / max(r["evaluations"], 1)
            for thr in THRESHOLDS:
                s[f"t_to_{thr:g}_s"] = _time_to(r["trace"], thr)
            summ.append(s)
        pd.DataFrame(summ).sort_values(["name", "code", "seed"]).to_csv(
            os.path.join(dest, "b4_optimisation.csv"), index=False)
        pd.DataFrame(trace_rows).to_csv(os.path.join(dest, "b4_traces.csv"), index=False)
    entry = pd.DataFrame(by("B4entry"))
    if len(entry):
        entry[["name", "n_c", "N", "seed", "iterations", "wall_s", "F_final", "message"]].to_csv(
            os.path.join(dest, "b4_entry.csv"), index=False)


def _read_csv(dest, name):
    import pandas as pd
    path = os.path.join(dest, name)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def _plot_b3(b3, path):
    import matplotlib
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, sweep, xcol, xlabel in ((axes[0], "nc", "d", "Hilbert-space dimension d = 3 n_c"),
                                    (axes[1], "N", "N", "number of time steps N")):
        s = b3[b3.sweep == sweep]
        ax.loglog(s[xcol], s["repo_fg_s"], "o-", label="repo (grape_core)")
        ax.loglog(s[xcol], s["qutip_DIAG_fg_s"], "s-", label="QuTiP qtrl, DIAG (default)")
        if "qutip_FRECHET_fg_s" in s:
            ax.loglog(s[xcol], s["qutip_FRECHET_fg_s"], "^--", label="QuTiP qtrl, FRECHET")
        ax.set_xlabel(xlabel)
        ax.set_xticks(sorted(s[xcol].unique()))
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel("seconds per fidelity + gradient")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title("N = %d" % s.N.iloc[0] if sweep == "nc" else "n_c = %d" % s.n_c.iloc[0])
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


T_DROP = 0.2     # B4 wall-time axis starts at half the first time any run reaches 1 - F < T_DROP


def _best_curves(traces, name, code):
    """Best-so-far infidelity of every seed: list of (seed, time_s, eval_index, 1 - max F)."""
    out = []
    sub = traces[(traces.name == name) & (traces.code == code)]
    for seed, g in sub.groupby("seed"):
        g = g.sort_values("evaluation")
        best = 1.0 - np.maximum.accumulate(g.F.values)
        out.append((seed, g.time_s.values, g.evaluation.values, best))
    return out


def _band(curves, x_of, grid):
    """Median and quartiles of the curves on a common grid; a finished run keeps its last value."""
    vals = np.full((len(curves), len(grid)), np.nan)
    for i, c in enumerate(curves):
        x, best = x_of(c), c[3]
        idx = np.searchsorted(x, grid, side="right") - 1
        ok = idx >= 0
        vals[i, ok] = best[idx[ok]]
        vals[i, ~ok] = 1.0
    return np.percentile(vals, [25, 50, 75], axis=0)


def _plot_b4(summ, traces, path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    names = [n for n in PROBLEMS if n in set(summ.name)] + sorted(set(summ.name) - set(PROBLEMS))
    y_floors = []
    with plt.rc_context({"font.size": 11, "axes.titlesize": 12, "legend.fontsize": 11}):
        fig, axes = plt.subplots(len(names), 2, figsize=(12, 4.3 * len(names)), squeeze=False)
        for i, name in enumerate(names):
            info = PROBLEMS.get(name, dict(title=name))
            row = summ[summ.name == name].iloc[0]
            n_seeds = summ[summ.name == name].groupby("code").seed.nunique().max()
            curves = {code: _best_curves(traces, name, code) for code in CODE_STYLE}
            # wall-time axis: start just before the first clear drop
            t_drop = min(c[1][np.argmax(c[3] < T_DROP)] for cs in curves.values() for c in cs
                         if np.any(c[3] < T_DROP))
            t_lo = 0.5 * t_drop
            # y axis: down to the decade below the lowest quartile band (computed first)
            q_low = min(_band(cs, lambda c: c[2], np.arange(1, max(c[2][-1] for c in cs) + 1))[0].min()
                        for cs in curves.values())
            y_floor = 10.0 ** np.floor(np.log10(q_low))
            y_floors.append(y_floor)
            below = []
            for code, cs in curves.items():
                st = CODE_STYLE[code]
                for c in cs:
                    axes[i, 0].plot(c[1], c[3], color=st["color"], lw=0.7, alpha=0.25)
                    axes[i, 1].plot(c[2], c[3], color=st["color"], lw=0.7, alpha=0.25)
                    if c[3][-1] < y_floor:
                        below.append((code, c[0], c[3][-1]))
                t_max = max(c[1][-1] for c in cs)
                tg = np.logspace(np.log10(t_lo), np.log10(t_max), 600)
                eg = np.arange(1, max(c[2][-1] for c in cs) + 1)
                for ax, x_of, grid in ((axes[i, 0], lambda c: c[1], tg),
                                       (axes[i, 1], lambda c: c[2], eg)):
                    q25, q50, q75 = _band(cs, x_of, grid)
                    ax.fill_between(grid, q25, q75, color=st["color"], alpha=0.2, lw=0)
                    ax.plot(grid, q50, color=st["color"], lw=2.2)
            title = (f"{info['title']}   ($n_c = {row.n_c}$, $N = {row.N}$, "
                     f"{n_seeds} seeds per code)")
            for j, ax in enumerate(axes[i]):
                ax.set_yscale("log")
                ax.set_ylim(y_floor, 1.5)
                for thr in THRESHOLDS:
                    ax.axhline(thr, color="0.45", ls=":", lw=1)
                ax.grid(True, which="major", alpha=0.25)
                ax.set_ylabel(r"best-so-far infidelity $1 - F$")
                ax.text(0.015, 0.03, "(%s)" % "abcdefgh"[2 * i + j], transform=ax.transAxes,
                        fontsize=12, fontweight="bold", va="bottom")
                if below:
                    ax.text(0.98, 0.03, "\n".join(
                        f"{CODE_STYLE[c]['label'].split(' (')[0].split(',')[0]} seed {s} goes on to "
                        f"$1 - F = {_sci(v)}$ (off scale)" for c, s, v in below),
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color="0.25")
            axes[i, 0].set_xscale("log")
            axes[i, 0].set_xlim(t_lo, None)
            axes[i, 0].set_xlabel("wall time since start (s)")
            axes[i, 1].set_xlim(0, None)
            axes[i, 1].set_xlabel("fidelity + gradient evaluations")
            axes[i, 0].set_title(title, loc="left")
        handles = [Line2D([], [], color=CODE_STYLE[c]["color"], lw=2.2,
                          label=CODE_STYLE[c]["label"] + ", median") for c in CODE_STYLE]
        handles += [Patch(color="0.5", alpha=0.3, lw=0, label="middle 50 % of seeds"),
                    Line2D([], [], color="0.5", lw=0.7, alpha=0.5, label="one seed"),
                    Line2D([], [], color="0.45", ls=":", lw=1,
                           label=r"thresholds $10^{-2}, 10^{-3}, 10^{-4}$")]
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.93 if len(names) > 1 else 0.88), h_pad=2.0)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return y_floors


def _median_inf(x):
    """Median that counts NaN (threshold never reached) as infinitely slow."""
    v = np.where(np.isnan(x), np.inf, np.asarray(x, dtype=float))
    return float(np.median(v))


def _fmt_s(x):
    if not np.isfinite(x):
        return "never"
    return f"{x:.0f}" if x >= 100 else f"{x:.3g}"


INTRO = r"""# Repo GRAPE vs QuTiP GRAPE: benchmark results

This folder compares two implementations of GRAPE (gradient ascent pulse engineering) for closed-system,
unitary dynamics:

- **repo**: `core/grape_core.py`, driven by scipy's L-BFGS-B optimiser with the settings of `core/optimizer.py`;
- **QuTiP**: the GRAPE of the `qutip-qtrl` package (unitary dynamics, `dyn_type="UNIT"`).

Both codes get the same Hamiltonian, the same time steps and the same targets. QuTiP's default propagator
(`DIAG`) also diagonalises each step's Hamiltonian and computes an exact gradient, so the two codes do the
**same maths**: the benchmarks compare implementations, not algorithms. B1 checks that both solve the same
problem, B2 that their gradients are right, B3 how long one evaluation takes, and B4 how long a whole
optimisation takes.

Generated by `python QuTip/benchmark_qtrl.py report`; do not edit by hand.
"""

PROBLEM_SECTION = r"""## The shared problem

**Hamiltonian.** A transmon (3 levels, lowering operator $b$) coupled to a cavity ($n_c$ levels, lowering
operator $a$), so the state space has dimension $d = 3n_c$. In the frame rotating with both modes,

$$
H(t) = H_0 + \sum_{j=1}^{4} u_j(t)\thinspace H_j ,
\qquad
H_0 = \chi\thinspace a^\dagger a\thinspace b^\dagger b
    + \frac{K}{2}\thinspace a^{\dagger 2} a^{2}
    + \frac{\chi'}{2}\thinspace b^\dagger b\thinspace a^{\dagger 2} a^{2}
    + \frac{\alpha}{2}\thinspace b^{\dagger 2} b^{2} ,
$$

with the four controls $H_1 = a + a^\dagger$ and $H_2 = i(a - a^\dagger)$ (cavity I and Q), $H_3 = b + b^\dagger$
and $H_4 = i(b - b^\dagger)$ (transmon I and Q). Parameters (`core/grape_core.py`):
$\chi/2\pi = <<chi>>$ MHz (dispersive shift), $K/2\pi = <<K>>$ MHz (cavity Kerr),
$\chi'/2\pi = <<chip>>$ MHz (second-order dispersive shift), $\alpha/2\pi = <<alpha>>$ MHz (transmon
anharmonicity). The amplitudes $u_j$ are in rad/μs.

**Time steps.** The controls are constant on each of $N$ steps of length $\Delta t = <<dt_ns>>$ ns. On step $n$ the
Hamiltonian is $H_n = H_0 + \sum_j u_{nj} H_j$, and the gate is the ordered product

$$
U = U_N\thinspace U_{N-1} \cdots U_1 ,
\qquad
U_n = e^{-i H_n \Delta t} = V_n\thinspace e^{-i \Lambda_n \Delta t}\thinspace V_n^\dagger ,
$$

where $H_n = V_n \Lambda_n V_n^\dagger$ is the eigendecomposition both codes use. The pulse is the $N \times 4$
array $u = (u_{nj})$.

**Fidelity.** A problem is a list of $k$ input states $|i_m\rangle$ and their targets $|f_m\rangle$. Every
fidelity in this folder is the repo's

$$
F = \left| \frac{1}{k} \sum_{m=1}^{k} \langle f_m | U | i_m \rangle \right|^2 ,
$$

which is 1 only if $U$ sends every input to its target with one common phase. QuTiP's phase-insensitive
fidelity (`PSU`) is

$$
f = \frac{1}{k} \left| \mathrm{Tr} \negthinspace \left( W^\dagger U V \right) \right| ,
$$

with the inputs as the columns of $V$ and the targets as the columns of $W$. So $F = f^2$ and
$\partial F / \partial u_{nj} = 2f\thinspace \partial f / \partial u_{nj}$; every QuTiP number below is converted this way.

| problem (B1-B3) | name (B4) | inputs $\to$ targets | $k$ |
|---|---|---|---|
| `fock1` | `A_fock1` | $\lvert g,0\rangle \to \lvert g,1\rangle$: put one photon into the empty cavity | 1 |
| `X` | `B_Xgate` | logical X on the cat code with $\alpha_\text{cat} = \sqrt{3}$: $\lvert 0_L\rangle \to \lvert 1_L\rangle$ and $\lvert 1_L\rangle \to \lvert 0_L\rangle$, transmon in $g$ | 2 |

**Table 1.** The two test problems. `X` is the repo's production gate (the trained pulse
`pulses/u_X_main.npy` solves it). $\lvert g,n\rangle$ = transmon in its ground state, $n$ photons in the
cavity. Each benchmark states its own $n_c$ and $N$.
"""


def write_report(dest, quick=False):
    """Stage 2: CSVs + environment.json in `dest` -> README.md and both figures (no raw data needed)."""
    import pandas as pd
    import matplotlib
    import matplotlib.ticker  # noqa: F401
    matplotlib.use("Agg")
    import core.grape_core as gc

    env_path = os.path.join(dest, "environment.json")
    env = json.load(open(env_path)) if os.path.exists(env_path) else environment()
    b1 = _read_csv(dest, "b1_agreement.csv")
    b2 = _read_csv(dest, "b2_gradient.csv")
    b3 = _read_csv(dest, "b3_timing.csv")
    prof = _read_csv(dest, "b3_qutip_profile.csv")
    summ = _read_csv(dest, "b4_optimisation.csv")
    traces = _read_csv(dest, "b4_traces.csv")
    entry = _read_csv(dest, "b4_entry.csv")
    if len(b3):
        b3 = b3.sort_values(["sweep", "n_c", "N"])
        _plot_b3(b3, os.path.join(dest, "b3_timing.png"))
    if len(summ):
        summ = summ.sort_values(["name", "code", "seed"])
        if len(traces):
            y_floors = _plot_b4(summ, traces, os.path.join(dest, "b4_convergence.png"))

    md = [INTRO]
    if quick:
        md.append("> **Quick smoke-test run** (tiny sizes). Numbers here are not the benchmark.\n")
    tp = 2 * np.pi
    md.append(_fill(PROBLEM_SECTION, chi=f"{gc.chi / tp:g}", K=f"{gc.Kerr / tp:g}",
                    chip=f"{gc.chip / tp:g}", alpha=f"{gc.alpha / tp:g}", dt_ns=f"{DT * 1e3:g}"))

    # ---- B4 medians (used by the summary and by B4)
    med_rows = []
    if len(summ):
        for name in [n for n in PROBLEMS if n in set(summ.name)] + sorted(set(summ.name) - set(PROBLEMS)):
            g = summ[summ.name == name]
            per = {}
            for code in ("repo", "qutip"):
                s = g[g.code == code]
                if not len(s):
                    continue
                r = dict(name=name, code=code, runs=len(s))
                for thr in THRESHOLDS:
                    col = f"t_to_{thr:g}_s"
                    r[thr] = _median_inf(s[col].values)
                    r[f"n_{thr}"] = int(np.isfinite(s[col].values).sum())
                r["F_final"] = float(np.median(s.F_final))
                r["s_eval"] = float(np.median(s.s_per_evaluation))
                per[code] = r
                med_rows.append(r)
            if {"repo", "qutip"} <= set(per):
                ratio = dict(name=name, code="ratio", runs=None, F_final=None,
                             s_eval=per["qutip"]["s_eval"] / per["repo"]["s_eval"])
                for thr in THRESHOLDS:
                    ratio[thr] = per["qutip"][thr] / per["repo"][thr]
                med_rows.append(ratio)
    med = {(r["name"], r["code"]): r for r in med_rows}

    # ---- Summary
    md.append("## Summary\n")
    summary = []
    if len(b1):
        summary.append(
            rf"- **B1 same physics:** the largest $\lvert F_\text{{repo}} - F_\text{{QuTiP}}\rvert$ is "
            rf"${_sci(b1.abs_diff.max())}$ over {len(b1)} pulses (${_sci(b1.abs_diff_float32_tau.max())}$ with "
            "QuTiP's default single-precision time steps).")
    if len(b2):
        summary.append(
            rf"- **B2 gradients:** largest relative error against finite differences: repo "
            rf"${_sci(b2.repo_max_rel_err.max())}$, QuTiP DIAG ${_sci(b2.qutip_DIAG_max_rel_err.max())}$, "
            rf"QuTiP FRECHET ${_sci(b2.qutip_FRECHET_max_rel_err.max())}$, all at the finite-difference "
            rf"floor; the two codes' gradients differ by at most "
            rf"${_sci(b2.repo_vs_qutip_DIAG_max_rel_diff.max())}$.")
    if len(b3):
        s = b3[b3.sweep == "nc"]
        fr = s.qutip_FRECHET_fg_s / s.repo_fg_s
        summary.append(
            rf"- **B3 cost per evaluation:** QuTiP DIAG takes {s.speedup_fg_DIAG.min():.2f} to "
            rf"{s.speedup_fg_DIAG.max():.2f} times as long as the repo ($n_c$ = {s.n_c.min()} to "
            rf"{s.n_c.max()}, $N$ = {s.N.iloc[0]}); QuTiP FRECHET {fr.min():.1f} to {fr.max():.1f} times.")
    for name in PROBLEMS:
        if (name, "repo") in med and (name, "qutip") in med:
            r, q = med[(name, "repo")], med[(name, "qutip")]
            summary.append(
                rf"- **B4 `{name}`:** median time to reach $1 - F \le 10^{{-3}}$: repo "
                rf"{_fmt_s(r[1e-3])} s, QuTiP {_fmt_s(q[1e-3])} s ({q[1e-3] / r[1e-3]:.2f}×); "
                rf"median final $F$: repo {r['F_final']:.6f}, QuTiP {q['F_final']:.6f}.")
    md.append("\n".join(summary) + "\n")

    # ---- B1
    md.append("## B1. Same physics: same pulse, same fidelity?\n")
    md.append(
        "**What it tests.** Two calculators given the same numbers must print the same answer. If the two "
        "codes disagree on the fidelity of a fixed pulse, they are solving different problems (a different "
        "Hamiltonian, operator ordering, time step or fidelity), and every speed comparison below would be "
        "meaningless.\n")
    md.append(
        r"**How.** For each fixed pulse $u$ both codes compute $F(u)$, and the table reports "
        r"$\lvert F_\text{repo}(u) - F_\text{QuTiP}(u)\rvert$. Pass: a difference at rounding level, "
        r"$\lesssim 10^{-10}$." + "\n")
    if len(b1):
        md.append(_md_table(b1, {"F_repo": "{:.12f}", "F_qutip": "{:.12f}", "abs_diff": "{:.1e}",
                                 "abs_diff_float32_tau": "{:.1e}"}) + "\n")
        md.append(
            rf"**Table 2.** The same pulse scored by both codes: they agree to rounding (largest difference "
            rf"${_sci(b1.abs_diff.max())}$)." + "\n")
        md.append(
            "*Columns:* `n_c` = cavity levels; `N` = time steps; `pulse`: `random_s<s>` = the repo's smooth "
            r"random start (`make_smooth_warm_start`, seed $s$) scaled by 3 to a peak of 12 rad/μs, "
            "`u_X_main.npy` = the trained production X pulse; `F_repo`, `F_qutip` = $F$ from each code; "
            r"`abs_diff` = $\lvert F_\text{repo} - F_\text{QuTiP}\rvert$; `abs_diff_float32_tau` = the same "
            "with QuTiP at its default time-step precision (next paragraph).\n")
        md.append(
            "*Why the last column is larger.* When given `num_tslots` and `evo_time`, qutip-qtrl 0.2.0 stores "
            "the step lengths in single precision (`dynamics.init_timeslots`, `np.ones(..., dtype=\"f\")`). "
            r"Each step is then $\Delta t = 2\ \text{ns} - 2.3 \times 10^{-10}\ \text{ns}$, a relative error of "
            r"about $10^{-7}$ in every step. Everywhere else in this benchmark QuTiP is given an explicit "
            "double-precision `tau`, which removes the difference.\n")

    # ---- B2
    md.append("## B2. Gradient accuracy\n")
    md.append(
        "**What it tests.** The optimiser climbs $F$, and the gradient is its compass. L-BFGS-B also builds "
        "its picture of the landscape's curvature from successive gradients, so a slightly wrong compass "
        "makes it slow down or stall even when each step is cheap. This test checks that each code's "
        "gradient is right.\n")
    md.append(
        r"**How.** At one random pulse $u$, pick a set $S$ of " +
        (f"{int(b2.n_entries.iloc[0])} " if len(b2) else "") +
        r"random entries $(n, j)$ (time step $n$, control $j$) and compare each code's gradient "
        r"$g_{nj} = \partial F / \partial u_{nj}$ with the central finite difference" + "\n\n"
        r"$$" + "\n"
        r"g^\text{FD}_{nj} = \frac{F(u + h\thinspace e_{nj}) - F(u - h\thinspace e_{nj})}{2h} , \qquad h = 10^{-6} ," + "\n"
        r"$$" + "\n\n"
        r"where $e_{nj}$ is the unit pulse that is 1 at entry $(n, j)$ and 0 elsewhere. The error is relative to "
        r"the largest finite difference:" + "\n\n"
        r"$$" + "\n"
        r"\varepsilon = \frac{\max_{(n,j) \in S} \left| g_{nj} - g^\text{FD}_{nj} \right|}"
        r"{\max_{(n,j) \in S} \left| g^\text{FD}_{nj} \right|} ." + "\n"
        r"$$" + "\n\n"
        r"The two codes are also compared directly, over all $4N$ entries:" + "\n\n"
        r"$$" + "\n"
        r"\delta = \frac{\max_{n,j} \left| g^\text{repo}_{nj} - g^\text{QuTiP}_{nj} \right|}"
        r"{\max_{n,j} \left| g^\text{repo}_{nj} \right|} ." + "\n"
        r"$$" + "\n\n"
        "QuTiP is tested with both of its exact gradient options: `DIAG` (eigendecomposition; its default for "
        "unitary problems) and `FRECHET` (scipy's `expm_frechet`, the derivative of the matrix exponential; "
        "its default for open systems). Its third option, `APPROX`, works only with QuTiP's finite-difference "
        "fidelity (`TDAPPROX`), not with the unitary fidelity used here, so it is left out.\n")
    if len(b2):
        cols = ["problem", "n_c", "N", "repo_max_rel_err", "qutip_DIAG_max_rel_err",
                "qutip_FRECHET_max_rel_err", "repo_vs_qutip_DIAG_max_rel_diff"]
        md.append(_md_table(b2[[c for c in cols if c in b2]], {c: "{:.1e}" for c in cols[3:]}) + "\n")
        md.append(
            r"**Table 3.** Gradient error of each code against finite differences ($\varepsilon$) and between "
            r"the two codes ($\delta$)." + "\n")
        md.append(
            r"*Columns:* `repo_max_rel_err`, `qutip_DIAG_max_rel_err`, `qutip_FRECHET_max_rel_err` = $\varepsilon$ "
            r"for that code; `repo_vs_qutip_DIAG_max_rel_diff` = $\delta$ (the CSV also holds the same for "
            "FRECHET).\n")
        errs = b2[["repo_max_rel_err", "qutip_DIAG_max_rel_err", "qutip_FRECHET_max_rel_err"]].values
        spread = float(np.max((errs.max(axis=1) - errs.min(axis=1)) / errs.max(axis=1)))
        digits = int(np.floor(-np.log10(spread))) if spread > 0 else 15
        md.append(
            rf"*Reading it.* In each row the three values of $\varepsilon$ agree to {digits} significant digits. "
            r"So $\varepsilon$ measures the finite difference itself (mostly rounding, about "
            r"$\epsilon_\text{mach} F / h$ in absolute terms, plus a truncation term of order $h^2$), not any "
            r"of the codes: all three gradients are exact as far as finite differences can tell. The sharper "
            rf"test is $\delta$: the two codes' full gradients agree to ${_sci(b2.repo_vs_qutip_DIAG_max_rel_diff.max())}$, "
            "which is rounding.\n")

    # ---- B3
    md.append("## B3. Cost of one fidelity + gradient evaluation\n")
    md.append(
        r"**What it tests.** Each L-BFGS-B iteration needs about one evaluation of $F$ and its gradient "
        "(sometimes a few, see B4), so the cost of one evaluation sets the training time. This is the price "
        "of a single step, measured on its own.\n")
    md.append(
        r"**How.** The median wall time $t$ of one call to exactly the functions the optimiser calls, on the "
        r"cat-code X problem ($k = 2$), with a new pulse every call so nothing is cached, and one BLAS thread. "
        r"Two sweeps: the state-space size $d = 3n_c$ at fixed $N$, and the number of steps $N$ at fixed $n_c$. "
        r"The speed-up is $s = t_\text{QuTiP} / t_\text{repo}$, so $s > 1$ means the repo is faster." + "\n")
    md.append(
        r"*What differs between the codes.* The repo diagonalises all $N$ step Hamiltonians in one batched "
        r"`eigh` call and carries only the $k = 2$ input states forward and the targets backward. qtrl loops "
        r"over the steps in Python, calls `eigh` once per step, and builds the full $d \times d$ propagators "
        "and their derivatives. Both spend most of their time in the per-step eigendecomposition, so they end "
        "up within a small factor of each other.\n")
    if len(b3):
        cols = ["sweep", "n_c", "d", "N", "repo_fg_s", "qutip_DIAG_fg_s", "qutip_FRECHET_fg_s",
                "speedup_fg_DIAG", "repo_f_s", "qutip_DIAG_f_s"]
        md.append(_md_table(b3[[c for c in cols if c in b3]], {c: "{:.4g}" for c in cols[4:]}) + "\n")
        md.append(r"**Table 4.** Median seconds per evaluation, X problem, one BLAS thread." + "\n")
        md.append(
            r"*Columns:* `sweep` = which size varies (`nc`: $n_c$ at fixed $N$; `N`: $N$ at fixed $n_c$); "
            r"`d` = $3n_c$; `*_fg_s` = fidelity + gradient, `*_f_s` = fidelity alone (seconds); "
            r"`speedup_fg_DIAG` = $s$ for fidelity + gradient against QuTiP `DIAG`." + "\n")
        md.append("![B3 timing](b3_timing.png)\n")
        md.append(
            r"**Figure 1.** Seconds per fidelity + gradient evaluation for the repo (blue), QuTiP `DIAG` "
            r"(orange, its default) and QuTiP `FRECHET` (green, dashed), log-log. Left: against the "
            rf"state-space size $d = 3n_c$ at $N = {b3[b3.sweep == 'nc'].N.iloc[0]}$. Right: against the "
            rf"number of steps $N$ at $n_c = {b3[b3.sweep == 'N'].n_c.iloc[0]}$." + "\n")
        fits = {}
        for nm in ("nc", "N"):
            s = b3[b3.sweep == nm]
            if len(s) >= 2:
                x = np.log(s["d" if nm == "nc" else "N"].values.astype(float))
                for col in ("repo_fg_s", "qutip_DIAG_fg_s"):
                    fits[(nm, col)] = np.polyfit(x[-3:], np.log(s[col].values[-3:]), 1)[0]
        if len(fits) == 4:
            md.append(
                r"**Scaling.** Fitting $t \propto d^{p}$ and $t \propto N^{q}$ to the last three points of "
                "each sweep:\n")
            md.append("| code | $p$ (against $d$) | $q$ (against $N$) |\n|---|---|---|\n"
                      f"| repo | {fits[('nc', 'repo_fg_s')]:.2f} | {fits[('N', 'repo_fg_s')]:.2f} |\n"
                      f"| QuTiP DIAG | {fits[('nc', 'qutip_DIAG_fg_s')]:.2f} | "
                      f"{fits[('N', 'qutip_DIAG_fg_s')]:.2f} |\n")
            md.append(
                r"**Table 5.** Fitted scaling exponents. An eigendecomposition costs $O(d^3)$ and there is one "
                r"per step, so for large $d$ one expects $t \propto N d^3$ ($p = 3$, $q = 1$). Both codes scale "
                r"alike. $p$ is still below 3, most likely because at $d \le " +
                f"{int(b3.d.max())}" + r"$ the work that grows more slowly than $d^3$ (matrix-vector "
                r"products, per-call overhead) still takes a visible share." +
                "\n")
    if len(prof):
        p = prof.sort_values("share_percent", ascending=False)
        md.append(
            rf"**Where QuTiP's time goes** (its own timing statistics, $n_c = {int(p.n_c.iloc[0])}$, "
            rf"$N = {int(p.N.iloc[0])}$): " +
            ", ".join(f"`{r.part}` {r.share_percent:.0f} %" for r in p.itertuples()) +
            ". `propagators` is the per-step propagator computation, which holds the eigendecompositions; "
            "`hamiltonians` builds each $H_n$; `forward` and `onward` are the running products of propagators "
            "from the start and from the end; `gradient` combines them into $\\partial f / \\partial u$.\n")

    # ---- B4
    md.append("## B4. End-to-end optimisation: time to solution\n")
    md.append(
        "**What it tests.** The race. A faster evaluation (B3) does not guarantee faster training: the codes "
        "optimise different functions of the same overlap, pay different overheads around each call, and "
        "their line searches may need different numbers of calls. Here both codes train the same problem from "
        "the same start, and we time how long each takes to reach a given fidelity.\n")
    md.append(_fill(
        r"""**How.** Both run scipy's L-BFGS-B from the same start (the repo's `make_smooth_warm_start`, peak
4 rad/μs, same seed), inside the box $\lvert u_{nj} \rvert \le <<box>>$ rad/μs, with the same stopping rules
(`ftol` $<<ftol>>$ = QuTiP `accuracy_factor` <<acc>>, gradient tolerance $<<gtol>>$, <<maxcor>> correction
pairs) and the same iteration and wall-time caps. They minimise different functions of the same overlap:

$$
C_\text{repo} = -F , \qquad C_\text{QuTiP} = 1 - f = 1 - \sqrt{F} .
$$

The time to reach an infidelity threshold $\epsilon$ uses the best fidelity seen so far,

$$
t_\epsilon = \min \left\lbrace t \thickspace : \thickspace 1 - \max_{t' \le t} F(t') \le \epsilon \right\rbrace ,
\qquad \epsilon \in \lbrace 10^{-2},\ 10^{-3},\ 10^{-4} \rbrace ,
$$

where $F(t')$ is the fidelity of the evaluation that finished $t'$ seconds after the start. No band filter,
ramp or penalties: QuTiP has nothing like the repo's constraint chain, so the production recipe cannot be
compared like for like.
""", box=f"{AMP_BOX:g}", ftol=_sci(FTOL, 0), acc=f"{FTOL / np.finfo(float).eps:.0f}",
        gtol=_sci(GTOL, 0), maxcor=MAXCOR))
    if med_rows:
        lines = ["| problem | code | runs | $t_{10^{-2}}$ (s) | $t_{10^{-3}}$ (s) | $t_{10^{-4}}$ (s) "
                 "| final $F$ | s per evaluation |", "|---|---|---|---|---|---|---|---|"]
        for r in med_rows:
            if r["code"] == "ratio":
                cells = [f"`{r['name']}`", "*QuTiP / repo*", ""]
                cells += [f"*{r[t]:.2f}*" if np.isfinite(r[t]) else "" for t in THRESHOLDS]
                cells += ["", f"*{r['s_eval']:.2f}*"]
            else:
                cells = [f"`{r['name']}`", r["code"], str(r["runs"])]
                for t in THRESHOLDS:
                    c = _fmt_s(r[t])
                    if r[f"n_{t}"] < r["runs"]:
                        c += f" ({r[f'n_{t}']}/{r['runs']})"
                    cells.append(c)
                cells += [f"{r['F_final']:.6f}", f"{r['s_eval']:.3g}"]
            lines.append("| " + " | ".join(cells) + " |")
        md.append("\n".join(lines) + "\n")
        md.append(
            r"**Table 6.** Medians over seeds of the time to each threshold, the final fidelity and the cost "
            r"per evaluation; the *QuTiP / repo* rows divide the two medians (above 1 = the repo is faster). "
            r"$(n/m)$ = only $n$ of $m$ runs reached the threshold; a run that never did counts as infinitely "
            "slow in the median.\n")
        early = summ[~summ.message.str.contains("limit", case=False)]
        if len(early):
            md.append(
                "*Stopping.* Every run used its full iteration budget except " +
                "; ".join(f"`{r.name}` {r.code} seed {r.seed}, which met the convergence test "
                          f"(`{r.message}`) after {r.iterations} iterations at $1 - F = {_sci(r.infidelity_final)}$"
                          for r in early.itertuples()) + ".\n")
        else:
            md.append("*Stopping.* Every run used its full iteration budget.\n")
    if len(traces):
        md.append("![B4 convergence](b4_convergence.png)\n")
        n_by = summ.groupby("name").seed.nunique().to_dict()
        md.append(
            "**Figure 2.** Convergence of the best fidelity seen so far, repo (blue) against QuTiP DIAG "
            "(orange). Rows: the two problems of Table 1 (" +
            ", ".join(f"`{n}` {n_by[n]} seeds" for n in n_by) +
            "). Left column (a, c): against wall time, log axis, starting just before the first drop. Right "
            "column (b, d): against the number of fidelity + gradient evaluations. Thick line = median over "
            "seeds, band = middle 50 % of seeds, thin lines = single seeds, dotted lines = the thresholds of "
            "Table 6. The two codes converge alike per evaluation (right), so the gap in wall time (left) comes "
            "mostly from the cost per evaluation (Table 6, last column; B3). Each row's y axis ends one decade below its lowest band (" +
            ", ".join(f"${_sci(y, 0)}$" for y in y_floors) + "); a run that goes lower is named in the "
            "panel.\n")
    if len(summ):
        cols = ["name", "n_c", "N", "seed", "code", "iterations", "evaluations", "wall_s",
                "s_per_evaluation", "F_final", "t_to_0.01_s", "t_to_0.001_s", "t_to_0.0001_s", "message"]
        md.append("<details>\n<summary><b>Table 7.</b> Every run, one row per seed and code "
                  "(click to open).</summary>\n")
        md.append(_md_table(summ[cols], {"wall_s": "{:.1f}", "s_per_evaluation": "{:.3g}",
                                         "F_final": "{:.8f}", "t_to_0.01_s": "{:.3g}",
                                         "t_to_0.001_s": "{:.3g}", "t_to_0.0001_s": "{:.3g}"}) + "\n")
        md.append(
            "*Columns:* `code` = which optimiser; `iterations` = L-BFGS-B iterations; `evaluations` = calls of "
            "fidelity + gradient (more than the iterations, because a line search sometimes needs more than "
            "one call); `wall_s` = total seconds; `s_per_evaluation` = `wall_s` / `evaluations`; `F_final` = "
            r"$F$ of the returned pulse; `t_to_<ε>_s` = $t_\epsilon$ in seconds, `nan` = never reached; "
            "`message` = why the optimiser stopped: the iteration cap (`STOP: TOTAL NO. of ITERATIONS REACHED "
            "LIMIT` from scipy, `Iteration or fidelity function call limit reached` from QuTiP) or convergence "
            r"(`CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`: the cost changed by less than `ftol` "
            "between iterations).\n")
        md.append("</details>\n")
    if len(entry):
        e = entry.copy()
        bare = summ[summ.code == "repo"].set_index(["name", "seed"]).wall_s
        e["bare_wall_s"] = [bare.get((r.name, r.seed), np.nan) for r in e.itertuples()]
        e["overhead"] = [f"{100 * (r.wall_s / r.bare_wall_s - 1):+.0f} %" for r in e.itertuples()]
        md.append(
            "**Overhead check.** The same problem through the repo's production entry point "
            "`optimize_multi_state_pulse` (penalties 0, no ramp, no band, one truncation, one job), first "
            "seed only. It builds its start the same way and returns the best pulse seen.\n")
        md.append(_md_table(e[["name", "n_c", "N", "seed", "iterations", "wall_s", "bare_wall_s", "overhead",
                               "F_final"]],
                            {"wall_s": "{:.1f}", "bare_wall_s": "{:.1f}", "F_final": "{:.8f}"}) + "\n")
        md.append(
            "**Table 8.** The production entry point against the bare L-BFGS-B loop of Table 7 for the same "
            "seed. *Columns:* `wall_s` = entry point, `bare_wall_s` = bare loop (Table 7), `overhead` = "
            "extra time of the entry point." +
            (" The final $F$ is the same as the bare loop's." if all(
                abs(r.F_final - summ[(summ.code == "repo") & (summ.name == r.name) & (summ.seed == r.seed)]
                    .F_final.iloc[0]) < 1e-8 for r in e.itertuples()) else "") + "\n")

    note = os.path.join(REPO_ROOT, "QuTip", "benchmark_interpretation.md")
    if os.path.exists(note) and not quick:
        md.append("## Interpretation\n")
        md.append("(Hand-written in `QuTip/benchmark_interpretation.md` after looking at the numbers above.)\n")
        md.append(open(note).read().strip() + "\n")
    md.append("## Not benchmarked, and why\n")
    md.append(
        "- **Memory:** below about 100 MB for both codes at these sizes; not a constraint.\n"
        "- **qutip-qoc:** its GRAPE calls qutip-qtrl; its other methods (GOAT, JOPT) are different algorithms.\n"
        "- **CRAB:** a different algorithm (few basis functions, gradient-free).\n"
        "- **Open systems (Lindblad GRAPE vs DGRAPE):** out of scope by decision (2026-11-23).\n"
        "- **Production recipe** (band filter, ramp, penalties, three truncations): no QuTiP equivalent.\n")
    md.append("## Files\n")
    md.append(
        "| file | contents |\n|---|---|\n"
        "| `b1_agreement.csv` | Table 2 |\n"
        "| `b2_gradient.csv` | Table 3, plus the FRECHET-vs-repo difference and the finite-difference scale |\n"
        "| `b3_timing.csv`, `b3_timing.png` | Table 4, Figure 1 |\n"
        "| `b3_qutip_profile.csv` | QuTiP's own timing split (B3) |\n"
        "| `b4_optimisation.csv` | Table 7 (Table 6 is computed from it) |\n"
        "| `b4_traces.csv`, `b4_convergence.png` | every evaluation of every B4 run (time, $F$); Figure 2 |\n"
        "| `b4_entry.csv` | Table 8 |\n"
        "| `environment.json` | the machine and package versions below |\n")
    md.append(
        "**Table 9.** Files in this folder. Rerun: `python QuTip/benchmark_qtrl.py run --part all`, then "
        "`report`. The raw results (one JSON per case) stay in `~/fwm_runs/qutip_bench/raw` on the machine "
        "that ran them; without them, `report` rebuilds this README and both figures from the CSVs here.\n")
    md.append("## Environment\n")
    md.append(", ".join(f"{k} {env.get(k)}" for k in (
        "platform", "processor", "cpu_count", "python", "numpy", "scipy", "qutip", "qutip_qtrl",
        "blas", "blas_threads")) + ".\n")
    with open(os.path.join(dest, "README.md"), "w") as fh:
        fh.write("\n".join(md))
    log(f"report written to {dest}")


def report(out, dest):
    """CSVs from the raw results when <out>/raw exists, then README + figures from the CSVs."""
    quick = os.path.basename(os.path.normpath(out)).endswith("quick")
    if os.path.isdir(os.path.join(out, "raw")):
        export_tables(out, dest)
    else:
        log(f"{os.path.join(out, 'raw')} not found: rebuilding the README and figures from the CSVs in {dest}")
    write_report(dest, quick=quick)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--part", default="all", choices=["agree", "grad", "timing", "opt", "all"])
    r.add_argument("--quick", action="store_true", help="tiny sizes, a few minutes")
    r.add_argument("--out", default=None)
    r.add_argument("--threads", type=int, default=1, help="BLAS threads (set before numpy loads)")
    p = sub.add_parser("report")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--dest", default=DEFAULT_DEST)
    p.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()

    if a.cmd == "report":
        report(a.out, a.dest)
        return
    cfg = QUICK if a.quick else FULL
    out = a.out or (DEFAULT_OUT + ("_quick" if a.quick else ""))
    store = Store(out)
    env_path = os.path.join(out, "environment.json")
    if not os.path.exists(env_path):
        with open(env_path, "w") as fh:
            json.dump(environment(), fh, indent=1)
    log(f"output: {out}  (BLAS threads {THREADS}, {'quick' if a.quick else 'full'})")
    parts = ["agree", "grad", "timing", "opt"] if a.part == "all" else [a.part]
    for part in parts:
        {"agree": run_agree, "grad": run_grad, "timing": run_timing, "opt": run_opt}[part](cfg, store)
    log("done")


if __name__ == "__main__":
    main()
