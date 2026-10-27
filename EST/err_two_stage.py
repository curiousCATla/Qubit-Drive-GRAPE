"""
C_err in stage 1 only, X, seed 0: est_optimization.ipynb section 7.

    python EST/err_two_stage.py run          # train both runs side by side (~2 h)
    python EST/err_two_stage.py status
    python EST/err_two_stage.py summarize    # score both stages of both runs, write the tables

The error-space fidelity term C_err (grape_eigh's `cerr`: C1's form on the error
cardinals at t = T) replaces nothing in the objective; it is added to stage 1 and
switched off for stage 2, which is what closes the gate:

    run      est_err, stage 1 (w1, w2, w3, w4, w_err, w5, w6) = (1, 0.7, 7, 1, 0.3, 0, 0)
                      stage 2                                 = (1, 0.1, 0, 1, 0,   0, 0)
    control  plain est, stage 1 (1, 0.7, 7, 1), stage 2 (1, 0.1, 0, 1)

Both: cold start at seed 0 (the seed of est_experiments.ipynb Figure 2a), eigh
pipeline, n_c 20, 2000 iterations per stage, one BLAS thread each, on one machine.
The control repeats est_experiments Appendix B's `est_appB_ctrl_seed0` command;
`summarize` reports how closely it reproduces that pulse.

Prediction, recorded before training: single-phase C_err at w_err 0.3, seed 0
(section 2.2) took F_err(T) 0.445 -> 0.931 and L(T) 0.513 -> 0.105. Stage 2 without
C_err may give most of that back, as stage 2 did to Appendix B's w2 change.
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from EST import tune_c2_c5 as t5
from EST.train_est_eigh import PULSE_DIR

REPO_ROOT = t5.REPO_ROOT
TABLE_DIR = t5.TABLE_DIR
RUN_DIR = os.path.join(os.path.expanduser("~"), "est_runs", "err2st")   # logs, score cache

GATE, SEED, MAXITER = "X", 0, 2000
W_ERR_S1, W_ERR_S2 = 0.3, 0.0
RUNS = {"control": "est_err2st_ctrl_seed0", "run": "est_err_err2st_seed0"}
REFERENCE = "est_appB_ctrl_seed0"      # the same control command, trained 2026-10-14
STAGES = ("stage1", "final")
WEIGHTS = {"control": [(1.0, 0.7, 7.0, 1.0), (1.0, 0.1, 0.0, 1.0)],
           "run": [(1.0, 0.7, 7.0, 1.0, W_ERR_S1, 0.0, 0.0),
                   (1.0, 0.1, 0.0, 1.0, W_ERR_S2, 0.0, 0.0)]}

RUNS_CSV = os.path.join(TABLE_DIR, "est_err2st_X_seed0.csv")
TRACES_NPZ = os.path.join(TABLE_DIR, "est_err2st_X_seed0_traces.npz")
CHECK_JSON = os.path.join(TABLE_DIR, "est_err2st_X_seed0_control_check.json")


def train_args(arm):
    base = ["EST/train_est_eigh.py", "--gate", GATE, "--seed", str(SEED),
            "--maxiter", str(MAXITER)]
    if arm == "control":
        return base + ["--variant", "est", "--tag", RUNS[arm][len("est_"):]]
    return base + ["--variant", "est_err", "--w-err", f"{W_ERR_S1:g}",
                   "--w-err-s2", f"{W_ERR_S2:g}", "--tag", RUNS[arm][len("est_err_"):]]


def stage_paths(name, stage):
    """(u, x) of one stage; the stage hook writes _stage1 and _stage2 (= final)."""
    k = 1 if stage == "stage1" else 2
    return (os.path.join(PULSE_DIR, f"u_{GATE}_{name}_stage{k}.npy"),
            os.path.join(PULSE_DIR, f"x_{GATE}_{name}_stage{k}.npy"))


def run():
    """Train every missing run, both at once, one thread each."""
    t5._keep_awake()
    os.makedirs(RUN_DIR, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    procs = {}
    for arm, name in RUNS.items():
        if t5.done(name):
            print(f"done already: {name}")
            continue
        log = open(os.path.join(RUN_DIR, f"{name[name.index('err2st'):]}.log"), "w")
        procs[name] = (subprocess.Popen([sys.executable] + train_args(arm), cwd=REPO_ROOT,
                                        env=env, stdout=log, stderr=subprocess.STDOUT), log)
        print(f"[{time.strftime('%H:%M:%S')}] start {name}", flush=True)
    for name, (p, log) in procs.items():
        p.wait()
        log.close()
        print(f"[{time.strftime('%H:%M:%S')}] "
              f"{'ok' if p.returncode == 0 and t5.done(name) else 'FAILED'} {name}", flush=True)


def status():
    for arm, name in RUNS.items():
        have = [st for st in STAGES if os.path.exists(stage_paths(name, st)[1])]
        print(f"{arm:8s} {name}: {'done' if t5.done(name) else 'running/missing'}; "
              f"stages on disk: {have}")


def _check_sidecar(arm):
    with open(t5.json_path(RUNS[arm])) as fh:
        info = json.load(fh)
    w = [tuple(s["weights"]) for s in info["stages"]]
    assert w == WEIGHTS[arm], (arm, w)
    assert info["seed"] == SEED and info["maxiter"] == MAXITER and info["init"] is None, arm
    assert info["n_stages"] == 2 and info["pipeline"] == "eigh_adjoint", arm
    return info


def _score_pulse(job):
    """compare_new_terms.score for one stage pulse, cached by the pre-image's mtime."""
    name, stage = job
    from EST import compare_new_terms
    upath, xpath = stage_paths(name, stage)
    cache = os.path.join(RUN_DIR, "scores", f"{name}_{stage}.npz")
    mtime = os.path.getmtime(xpath)
    if os.path.exists(cache):
        with np.load(cache) as c:
            if float(c["mtime"]) == mtime:
                return job, json.loads(str(c["row"])), {k[3:]: c[k] for k in c.files
                                                          if k.startswith("tr_")}
    row, tr = compare_new_terms.score(np.load(upath), np.load(xpath))
    row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    np.savez(cache, mtime=mtime, row=json.dumps(row),
             **{f"tr_{k}": np.asarray(v) for k, v in tr.items()})
    return job, row, tr


def control_check():
    """How closely the fresh control reproduces REFERENCE, stage by stage."""
    out = {"reference": REFERENCE, "control": RUNS["control"]}
    for st in STAGES:
        a = np.load(stage_paths(RUNS["control"], st)[1])
        b = np.load(stage_paths(REFERENCE, st)[1])
        ua = np.load(stage_paths(RUNS["control"], st)[0])
        ub = np.load(stage_paths(REFERENCE, st)[0])
        out[st] = {"max_abs_dx": float(np.abs(a - b).max()),
                   "max_abs_du_rad_per_us": float(np.abs(ua - ub).max()),
                   "identical": bool(np.array_equal(a, b))}
    return out


def summarize(jobs=4):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    infos = {arm: _check_sidecar(arm) for arm in RUNS}
    os.makedirs(os.path.join(RUN_DIR, "scores"), exist_ok=True)
    jobs_list = [(RUNS[arm], st) for arm in RUNS for st in STAGES]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
        scored = {job: (row, tr) for job, row, tr in ex.map(_score_pulse, jobs_list)}
    rows, traces = [], {}
    for arm, name in RUNS.items():
        for k, st in enumerate(STAGES):
            row, tr = scored[(name, st)]
            tier = ("parked" if row["progress"] <= 0.5 else
                    "closed" if row["F1"] >= 0.999 else "open")
            rows.append({"arm": arm, "stage": st, "tier": tier, "name": name,
                         "weights": str(WEIGHTS[arm][k]), **row,
                         "nit": infos[arm]["stages"][k]["nit"],
                         "hours": infos[arm]["stages"][k]["seconds"] / 3600})
            for key, v in tr.items():
                traces[f"{arm}/{st}/{key}"] = np.asarray(v)
    runs = pd.DataFrame(rows)
    runs.to_csv(RUNS_CSV, index=False)
    np.savez(TRACES_NPZ, **traces)
    check = control_check()
    with open(CHECK_JSON, "w") as fh:
        json.dump(check, fh, indent=2)
    print(f"tables -> {RUNS_CSV}\n          {TRACES_NPZ}\n          {CHECK_JSON}\n")
    print(runs[["arm", "stage", "tier", "F1", "F_ET", "F_err_T", "L_T", "dqec_mean", "L_mean",
                "eta_avg_mean", "dqec_tail", "L_tail", "eta_avg_tail"]]
          .to_string(index=False, float_format="%.4f"))
    print("\ncontrol vs", REFERENCE, json.dumps(check, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["run", "status", "summarize"])
    p.add_argument("--jobs", type=int, default=4, help="parallel scoring processes")
    a = p.parse_args()
    if a.action == "run":
        run()
    elif a.action == "status":
        status()
    else:
        summarize(jobs=a.jobs)


if __name__ == "__main__":
    main()
