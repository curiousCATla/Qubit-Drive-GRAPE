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
(a half-finished case restarts). `report` reads DIR/raw and writes the readable
report, CSVs and figures into --dest. Timings use single-threaded BLAS by
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


def report(out, dest):
    import pandas as pd
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, env = _load(out)
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "environment.json"), "w") as fh:
        json.dump(env, fh, indent=1)
    by = lambda b: [r for r in rows if r.get("bench") == b]  # noqa: E731
    md = []
    quick = os.path.basename(os.path.normpath(out)).endswith("quick")

    # ---- B1
    b1 = pd.DataFrame(by("B1"))
    if len(b1):
        b1 = b1[["problem", "n_c", "N", "pulse", "F_repo", "F_qutip", "abs_diff",
                 "abs_diff_float32_tau"]]
        b1.to_csv(os.path.join(dest, "b1_agreement.csv"), index=False)
    # ---- B2
    b2 = pd.DataFrame(by("B2"))
    if len(b2):
        b2.to_csv(os.path.join(dest, "b2_gradient.csv"), index=False)
    # ---- B3
    b3 = pd.DataFrame(by("B3"))
    prof = by("B3profile")
    if len(b3):
        b3 = b3.sort_values(["sweep", "n_c", "N"])
        b3.to_csv(os.path.join(dest, "b3_timing.csv"), index=False)
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
        fig.savefig(os.path.join(dest, "b3_timing.png"), dpi=150)
        plt.close(fig)
    # ---- B4
    b4 = by("B4")
    entry = pd.DataFrame(by("B4entry"))
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
        summ = pd.DataFrame(summ).sort_values(["name", "code", "seed"])
        summ.to_csv(os.path.join(dest, "b4_optimisation.csv"), index=False)
        pd.DataFrame(trace_rows).to_csv(os.path.join(dest, "b4_traces.csv"), index=False)
        names = sorted(summ.name.unique())
        fig, axes = plt.subplots(len(names), 2, figsize=(11, 3.8 * len(names)), squeeze=False)
        colors = {"repo": "C0", "qutip": "C3"}
        for i, name in enumerate(names):
            for r in b4:
                if r["name"] != name or not r["trace"]:
                    continue
                tr = np.array(r["trace"])
                best = 1 - np.maximum.accumulate(tr[:, 1])
                lab = ("repo" if r["code"] == "repo" else "QuTiP qtrl") if r["seed"] == min(
                    x["seed"] for x in b4 if x["name"] == name) else None
                axes[i, 0].semilogy(np.maximum(tr[:, 0], 1e-3), np.maximum(best, 1e-12),
                                    color=colors[r["code"]], alpha=0.8, label=lab)
                axes[i, 1].semilogy(np.arange(1, len(best) + 1), np.maximum(best, 1e-12),
                                    color=colors[r["code"]], alpha=0.8, label=lab)
            axes[i, 0].set_xscale("log")
            axes[i, 0].set_xlabel("wall time (s)")
            axes[i, 1].set_xlabel("fidelity evaluations")
            for ax in axes[i]:
                ax.set_ylabel("1 - F (best so far)")
                ax.set_title(name)
                ax.grid(True, which="both", alpha=0.3)
            axes[i, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(dest, "b4_convergence.png"), dpi=150)
        plt.close(fig)

    # ---- write README.md
    md.append("# Repo GRAPE vs QuTiP GRAPE: benchmark results\n")
    if quick:
        md.append("> **Quick smoke-test run** (tiny sizes). Numbers here are not the benchmark.\n")
    md.append(
        "This folder compares the repo's closed-system GRAPE (`core/grape_core.py` + scipy "
        "L-BFGS-B) with QuTiP's GRAPE (the `qutip-qtrl` package, unitary dynamics). Both get the "
        "same transmon-cavity Hamiltonian, the same 2 ns piecewise-constant time steps and the "
        "same targets. QuTiP's default propagator (`DIAG`) also uses an eigendecomposition and an "
        "exact gradient, so the two codes do the **same maths**: what differs is how it is "
        "implemented. Generated by `python QuTip/benchmark_qtrl.py report`; do not edit by hand.\n")
    md.append("**Machine:** " + ", ".join(f"{k} {env.get(k)}" for k in (
        "platform", "processor", "cpu_count", "python", "numpy", "scipy", "qutip", "qutip_qtrl",
        "blas", "blas_threads")) + "\n")
    md.append(
        "**Fidelity convention.** All fidelities are the repo's F = |sum_m <f_m|U|i_m>|^2 / k^2 "
        "(k = number of input states; k = 1 for the Fock problem, k = 2 for the cat-code X gate). "
        "QuTiP's phase-insensitive fidelity is f = |Tr(W^dag U V)| / k, so F = f^2.\n")

    # Summary
    md.append("## Summary\n")
    summary = []
    if len(b1):
        summary.append(f"- **B1 same physics:** largest |F_repo - F_QuTiP| = {b1.abs_diff.max():.1e} "
                       f"over {len(b1)} pulses ({b1.abs_diff_float32_tau.max():.1e} with QuTiP's "
                       f"default single-precision time steps).")
    if len(b2):
        summary.append(f"- **B2 gradients:** largest error against finite differences: repo "
                       f"{b2.repo_max_rel_err.max():.1e}, QuTiP DIAG {b2.qutip_DIAG_max_rel_err.max():.1e}, "
                       f"QuTiP FRECHET {b2.qutip_FRECHET_max_rel_err.max():.1e}.")
    if len(b3):
        s = b3[b3.sweep == "nc"]
        summary.append(f"- **B3 cost per evaluation:** QuTiP DIAG takes {s.speedup_fg_DIAG.min():.2f} to "
                       f"{s.speedup_fg_DIAG.max():.2f} times as long as the repo "
                       f"(n_c {s.n_c.min()} to {s.n_c.max()}, N {s.N.iloc[0]}); QuTiP FRECHET "
                       f"{(s.qutip_FRECHET_fg_s / s.repo_fg_s).min():.1f} to "
                       f"{(s.qutip_FRECHET_fg_s / s.repo_fg_s).max():.1f} times.")
    if b4:
        for name in sorted(summ.name.unique()):
            g = summ[summ.name == name].groupby("code")
            med = g[["t_to_0.001_s", "F_final", "wall_s"]].median()
            if {"repo", "qutip"} <= set(med.index):
                summary.append(
                    f"- **B4 {name}:** median time to 1 - F <= 1e-3: repo "
                    f"{med.loc['repo', 't_to_0.001_s']:.3g} s, QuTiP {med.loc['qutip', 't_to_0.001_s']:.3g} s; "
                    f"median final F repo {med.loc['repo', 'F_final']:.6f}, QuTiP {med.loc['qutip', 'F_final']:.6f}.")
    md.append("\n".join(summary) + "\n")

    md.append("## B1. Same physics: do both codes give the same fidelity for the same pulse?\n")
    md.append(
        "*Why:* every speed or convergence comparison below only means something if both codes "
        "solve exactly the same problem (same Hamiltonian, operator ordering, time stepping and "
        "fidelity). *What it measures:* the fidelity of fixed pulses (random smooth pulses and the "
        "production pulse `pulses/u_X_main.npy`) computed by each code. *Pass:* differences at "
        "rounding level (about 1e-10 or less).\n")
    if len(b1):
        md.append(_md_table(b1, {"F_repo": "{:.12f}", "F_qutip": "{:.12f}", "abs_diff": "{:.1e}",
                                 "abs_diff_float32_tau": "{:.1e}"}) + "\n")
        md.append(
            "`abs_diff_float32_tau` is QuTiP as it behaves by default when given `num_tslots` and "
            "`evo_time`: qutip-qtrl 0.2.0 stores the time steps in single precision "
            "(`dynamics.init_timeslots`, `np.ones(..., dtype=\"f\")`), so every step is 2 ns minus "
            "2.3e-10 ns and each propagator is off by about 1e-7 relative. Everywhere else in this "
            "benchmark QuTiP is given an explicit double-precision `tau`, which removes the "
            "difference.\n")

    md.append("## B2. Gradient accuracy\n")
    md.append(
        "*Why:* L-BFGS-B builds its curvature model from the gradient, so a wrong or approximate "
        "gradient slows or stalls the optimisation even if each evaluation is cheap. *What it "
        "measures:* the largest error of each code's gradient dF/du on random entries, relative to "
        "the largest central finite difference (h = 1e-6). QuTiP is tested with both of its exact "
        "propagator-gradient options: `DIAG` (default for unitary problems, eigendecomposition) and "
        "`FRECHET` (scipy's expm_frechet; QuTiP's default for open systems). Its third option, "
        "`APPROX`, only works with QuTiP's finite-difference fidelity (`TDAPPROX`), not with the "
        "unitary fidelity used here, so it is left out. The last column compares the full "
        "gradients of the two codes directly.\n")
    if len(b2):
        cols = ["problem", "n_c", "N", "repo_max_rel_err", "qutip_DIAG_max_rel_err",
                "qutip_FRECHET_max_rel_err", "repo_vs_qutip_DIAG_max_rel_diff"]
        md.append(_md_table(b2[[c for c in cols if c in b2]], {c: "{:.1e}" for c in cols[3:]}) + "\n")
        md.append("Finite differences carry their own error (rounding and the h^2 term), here "
                  "about 1e-7 to 1e-6 relative. A code whose error sits at that floor is exact as far "
                  "as finite differences can tell; the sharper test is the last column, the two "
                  "codes' full gradients against each other.\n")

    md.append("## B3. Cost of one fidelity + gradient evaluation\n")
    md.append(
        "*Why:* one L-BFGS-B iteration needs about one evaluation of the fidelity and its "
        "gradient, so this cost sets the training time. *What it measures:* the median wall time "
        "of one evaluation (the exact functions the optimiser calls; a new pulse each call so "
        "nothing is cached), single-threaded BLAS, cat-code X-gate problem (2 input states). Left: "
        "against the Hilbert-space size; right: against the number of time steps. `_f_s` columns "
        "are fidelity-only times; `speedup_fg_DIAG` = QuTiP DIAG time / repo time. *What differs:* the "
        "repo diagonalises all time steps in one batched `eigh` call and propagates only the 2 "
        "input states; qtrl loops over time steps in Python, calling `eigh` once per step, and "
        "forms d x d propagator gradients. Both are dominated by the per-step eigendecomposition, "
        "so they can end up close.\n")
    if len(b3):
        cols = ["sweep", "n_c", "d", "N", "repo_fg_s", "qutip_DIAG_fg_s", "qutip_FRECHET_fg_s",
                "speedup_fg_DIAG", "repo_f_s", "qutip_DIAG_f_s"]
        md.append(_md_table(b3[[c for c in cols if c in b3]],
                            {c: "{:.4g}" for c in cols[4:]}) + "\n")
        md.append("![B3 timing](b3_timing.png)\n")
        for nm in ("nc", "N"):
            s = b3[b3.sweep == nm]
            if len(s) >= 2:
                x = np.log(s["d" if nm == "nc" else "N"].values.astype(float))
                for col, lab in (("repo_fg_s", "repo"), ("qutip_DIAG_fg_s", "QuTiP DIAG")):
                    p = np.polyfit(x[-3:], np.log(s[col].values[-3:]), 1)[0]
                    md.append(f"- scaling of {lab} with {'d' if nm == 'nc' else 'N'} "
                              f"(last 3 points): exponent {p:.2f}")
        md.append("")
    if prof:
        p = prof[0]
        tot = sum(p[k] for k in ("hamiltonians", "propagators", "forward", "onward", "gradient"))
        md.append(f"Where QuTiP's time goes (its own statistics, n_c {p['n_c']}, N {p['N']}): " +
                  ", ".join(f"{k} {100 * p[k] / tot:.0f} %" for k in
                            ("propagators", "gradient", "hamiltonians", "forward", "onward")) + ".\n")

    md.append("## B4. End-to-end optimisation: time to solution\n")
    md.append(
        "*Why:* a faster evaluation does not automatically mean faster training: the two codes "
        "optimise different functions of the same overlap (repo -F, QuTiP 1 - sqrt(F)) and have "
        "different wrapper overheads. *What it measures:* both run scipy L-BFGS-B from the same "
        "smooth random start (`make_smooth_warm_start`, peak 4 rad/us, same seed), same box "
        f"+-{AMP_BOX:g} rad/us, same stopping rules (ftol {FTOL:g} = QuTiP accuracy_factor "
        f"{FTOL / np.finfo(float).eps:.0f}, gradient tolerance {GTOL:g}, {MAXCOR} correction pairs, "
        "same iteration cap and wall-time cap). Reported: wall time until the best-so-far "
        "infidelity first reaches 1e-2 / 1e-3 / 1e-4 (NaN = never), final F, iterations, "
        "evaluations. No band filter, ramp or penalties: QuTiP has no equivalent of the repo's "
        "constraint chain, so the production recipe cannot be compared like for like.\n"
        "Problems: **A_fock1** = |g,0> -> |g,1>; **B_Xgate** = cat-code logical X (alpha = sqrt 3), "
        "the production gate, at one truncation.\n")
    if b4:
        cols = ["name", "n_c", "N", "seed", "code", "iterations", "evaluations", "wall_s",
                "s_per_evaluation", "F_final", "t_to_0.01_s", "t_to_0.001_s", "t_to_0.0001_s", "message"]
        md.append(_md_table(summ[cols], {"wall_s": "{:.1f}", "s_per_evaluation": "{:.3g}",
                                         "F_final": "{:.8f}", "t_to_0.01_s": "{:.3g}",
                                         "t_to_0.001_s": "{:.3g}", "t_to_0.0001_s": "{:.3g}"}) + "\n")
        md.append("![B4 convergence](b4_convergence.png)\n")
    if len(entry):
        md.append(
            "Overhead check: the same problem through the repo's production entry point "
            "`optimize_multi_state_pulse` (penalties 0, no ramp, no band, one truncation, n_jobs 1), "
            "first seed only. Note it returns the best pulse seen, and builds its own start the same way.\n")
        md.append(_md_table(entry[["name", "n_c", "N", "seed", "iterations", "wall_s", "F_final"]],
                            {"wall_s": "{:.1f}", "F_final": "{:.8f}"}) + "\n")

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
    md.append("`b1_agreement.csv`, `b2_gradient.csv`, `b3_timing.csv`, `b4_optimisation.csv`, "
              "`b4_traces.csv` (every evaluation of every run), `b3_timing.png`, `b4_convergence.png`, "
              "`environment.json`. Rerun: `python QuTip/benchmark_qtrl.py run --part all`, then `report`.\n")
    with open(os.path.join(dest, "README.md"), "w") as fh:
        fh.write("\n".join(md))
    log(f"report written to {dest}")


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
