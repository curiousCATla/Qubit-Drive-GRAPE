"""
Two-stage training at the section 5 lead cell, est_optimization.ipynb section 6
("Phase 2").

    python EST/c2c5_two_stage.py run --jobs 23     # train everything missing, in parallel
    python EST/c2c5_two_stage.py status            # how far the campaign has got
    python EST/c2c5_two_stage.py summarize         # score every run, write the tables

Section 5 scanned w2 x w5 in a single 1000-iteration phase. No cell passed its
F1 bar; (w2, w5) = (0.8, 3.5) was the post-hoc lead. A single phase never closes
the X gate (F1 ~ 0.94), so this campaign gives that cell the full two-stage
schedule and asks whether its transparency gain survives stage 2.

Pipeline: EST/train_est_eigh.py (numpy eigh + hand adjoint), two stages, n_c 20.

    tuned    est_dn, stage 1 (w1..w4, w_err, w5, w6) = (1, 0.8, 7, 1, 0, 3.5, 0)
                     stage 2                         = (1, 0.1, 0, 1, 0, 0.4, 0)
             warm start (--init-x) from the section 5 grid run at (0.8, 3.5)
    control  plain est, stage 1 (1, 0.7, 7, 1), stage 2 (1, 0.1, 0, 1)
             warm start from the section 5 control of the same seed

Both arms: 1000 more stage-1 iterations on top of the 1000 the warm start
already has (2000 in all), then 2000 stage-2 iterations.

Seeds (fixed before training): section 5's twelve minus seed 0, whose grid run
at (0.8, 3.5) parked. Its replacement is the first seed of REPLACEMENT_POOL, in
order, whose (0.8, 3.5) grid run -- trained here with section 5's exact command
-- drives (progress > 0.5). Every replacement control drives already.

Reading (fixed before training). A run is PARKED if progress <= 0.5 and is left
out of every paired comparison. For each metric, the paired difference tuned -
control is taken over seeds where both arms drive; it is RESOLVED when its mean
is more than two standard errors from zero. Tails are means over t >= 0.9 T.
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
RUN_DIR = os.path.join(os.path.expanduser("~"), "est_runs", "c2c5_2st")   # logs, score cache

GATE = t5.GATE
W2_S1, W5_S1 = 0.8, 3.5           # the section 5 lead cell
W2_S2, W5_S2 = 0.1, 0.4
MAXITER_S1, MAXITER_S2 = 1000, 2000
FIXED_SEEDS = (2, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16)   # section 5's seeds minus 0
DROPPED_SEED = 0
REPLACEMENT_POOL = (17, 18, 19, 20, 21)
ARMS = ("control", "tuned")
STAGES = ("stage1", "final")

RUNS_CSV = os.path.join(TABLE_DIR, "est_c2c5_2st_X_runs.csv")
PAIRED_CSV = os.path.join(TABLE_DIR, "est_c2c5_2st_X_paired.csv")
TRACES_NPZ = os.path.join(TABLE_DIR, "est_c2c5_2st_X_traces.npz")
SEED_JSON = os.path.join(TABLE_DIR, "est_c2c5_2st_X_seeds.json")

# (column, +1 if higher is better, -1 if lower is better)
METRICS = (("F1", +1), ("F_ET", +1),
           ("dqec_mean", -1), ("L_mean", -1), ("eta_avg_mean", -1),
           ("dqec_tail", -1), ("L_tail", -1), ("eta_avg_tail", -1),
           ("L_T", -1), ("F_err_T", +1), ("eta0_mean", -1), ("eta0_tail", -1),
           ("c5", -1), ("max_active_fock", -1))


# ---------------------------------------------------------------- names

def run_name(arm, seed):
    return (f"est_c2c5_2st_ctrl_seed{seed}" if arm == "control"
            else f"est_dn_c2c5_2st_seed{seed}")


def warm_name(arm, seed):
    """The section 5 run each arm starts from."""
    return t5.ctrl_name(seed) if arm == "control" else t5.cell_name(seed, W2_S1, W5_S1)


def stage_paths(name, stage):
    """(u, x) of one stage; the stage hook writes _stage1 and _stage2."""
    k = 1 if stage == "stage1" else 2
    return (os.path.join(PULSE_DIR, f"u_{GATE}_{name}_stage{k}.npy"),
            os.path.join(PULSE_DIR, f"x_{GATE}_{name}_stage{k}.npy"))


def train_args(name):
    arm = "control" if "_ctrl_" in name else "tuned"
    seed = int(name.rsplit("seed", 1)[1])
    base = ["EST/train_est_eigh.py", "--gate", GATE, "--stages", "2",
            "--maxiter-s1", str(MAXITER_S1), "--maxiter", str(MAXITER_S2),
            "--init-x", os.path.relpath(t5.x_path(warm_name(arm, seed)), REPO_ROOT),
            "--seed", str(seed)]
    if arm == "control":
        return base + ["--variant", "est", "--tag", name[len("est_"):]]
    return base + ["--variant", "est_dn", "--w2", f"{W2_S1:g}", "--w-dn", f"{W5_S1:g}",
                   "--w2-s2", f"{W2_S2:g}", "--w-dn-s2", f"{W5_S2:g}",
                   "--tag", name[len("est_dn_"):]]


# ---------------------------------------------------------------- seeds

def check_fixed_seeds():
    """FIXED_SEEDS must be section 5's selection minus DROPPED_SEED."""
    import pandas as pd
    sel = pd.read_csv(t5.SEEDS_CSV)
    chosen = tuple(int(s) for s in sel.seed[sel.selected.astype(bool)])
    expect = tuple(s for s in chosen if s != DROPPED_SEED)
    if expect != FIXED_SEEDS:
        raise SystemExit(f"section 5 selected {chosen}; FIXED_SEEDS {FIXED_SEEDS} "
                         f"is not that minus seed {DROPPED_SEED}")
    drives = dict(zip(sel.seed.astype(int), sel.drives.astype(bool)))
    bad = [s for s in REPLACEMENT_POOL if not drives.get(s, False)]
    if bad:
        raise SystemExit(f"replacement controls that do not drive: {bad}")


def replacement(score=True):
    """(seed or None, next grid run to train or None, walk log) for the swap rule."""
    walk = []
    for s in REPLACEMENT_POOL:
        name = t5.cell_name(s, W2_S1, W5_S1)
        if not t5.done(name):
            return None, name, walk
        if not score:
            return None, None, walk
        _, sc = t5._score_one(name)
        drives = sc["progress"] > 0.5
        walk.append({"seed": s, "grid_run": name, "progress": sc["progress"],
                     "F1": sc["F1"], "drives": bool(drives)})
        if drives:
            return s, None, walk
    return None, None, walk


def campaign_seeds():
    s, _, walk = replacement()
    return FIXED_SEEDS + ((s,) if s is not None else ()), s, walk


# ---------------------------------------------------------------- launcher

def run(jobs):
    """Train every missing run, `jobs` single-threaded processes at a time.

    Queue: the next replacement grid run first (one at a time, so the seed
    order of the swap rule is exact), then both arms of the fixed seeds, then
    both arms of the replacement once it is known. Runs whose pre-image and
    sidecar exist are skipped, so `run` resumes.
    """
    check_fixed_seeds()
    t5._keep_awake()
    os.makedirs(RUN_DIR, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    queued, queue, running = set(), [], {}

    def enqueue(names, front=False):
        for n in names:
            if n not in queued and not t5.done(n):
                queued.add(n)
                queue.insert(0, n) if front else queue.append(n)

    def args_for(name):
        return t5.train_args(name) if name.startswith("est_dn_c2c5_sp") else train_args(name)

    enqueue(run_name(a, s) for s in FIXED_SEEDS for a in ARMS)
    decided = False
    t_start, n_done = time.time(), 0
    while True:
        if not decided and not any(n.startswith("est_dn_c2c5_sp") for n in
                                   list(running) + queue):
            seed, nxt, walk = replacement()
            for w in walk:
                print(f"  replacement walk: seed {w['seed']} progress "
                      f"{w['progress']:.3f} -> {'drives' if w['drives'] else 'PARKS'}",
                      flush=True)
            if seed is not None:
                print(f"replacement seed: {seed}", flush=True)
                enqueue([run_name(a, seed) for a in ARMS])
                decided = True
            elif nxt is not None:
                enqueue([nxt], front=True)
            else:
                print("replacement pool exhausted: every grid run parks; "
                      "the campaign runs on the fixed seeds only", flush=True)
                decided = True
        while queue and len(running) < jobs:
            name = queue.pop(0)
            log = open(os.path.join(RUN_DIR, f"{name}.log"), "w")
            p = subprocess.Popen([sys.executable] + args_for(name), cwd=REPO_ROOT,
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
            running[name] = (p, log, time.time())
            print(f"[{time.strftime('%H:%M:%S')}] start {name}  "
                  f"(running {len(running)}, queued {len(queue)})", flush=True)
        for name, (p, log, t0) in list(running.items()):
            if p.poll() is None:
                continue
            log.close()
            del running[name]
            n_done += 1
            status = ("ok" if p.returncode == 0 and t5.done(name)
                      else f"FAILED rc={p.returncode}")
            print(f"[{time.strftime('%H:%M:%S')}] {status} {name}  "
                  f"{(time.time() - t0) / 60:.1f} min  (done {n_done}, running "
                  f"{len(running)}, queued {len(queue)})", flush=True)
        if not running and not queue and decided:
            break
        time.sleep(5)
    print(f"campaign finished in {(time.time() - t_start) / 3600:.2f} h")


def status():
    seed, nxt, _ = replacement(score=False)
    print(f"replacement: {'decided' if seed is not None else 'next grid run ' + str(nxt)}")
    names = [run_name(a, s) for s in FIXED_SEEDS + REPLACEMENT_POOL for a in ARMS]
    finished = [n for n in names if t5.done(n)]
    print(f"two-stage runs done: {len(finished)}")
    for n in finished:
        with open(t5.json_path(n)) as fh:
            st = json.load(fh)["stages"]
        print(f"  {n}: nit {[s['nit'] for s in st]}, "
              f"{sum(s['seconds'] for s in st) / 3600:.2f} h")


# ---------------------------------------------------------------- scoring

def _check_sidecar(name, arm, seed):
    """The run trained what this campaign says it trained."""
    with open(t5.json_path(name)) as fh:
        info = json.load(fh)
    w = [tuple(s["weights"]) for s in info["stages"]]
    expect = ([(1.0, 0.7, 7.0, 1.0), (1.0, 0.1, 0.0, 1.0)] if arm == "control" else
              [(1.0, W2_S1, 7.0, 1.0, 0.0, W5_S1, 0.0), (1.0, W2_S2, 0.0, 1.0, 0.0, W5_S2, 0.0)])
    assert w == expect, (name, w)
    assert info["maxiter_stage1"] == MAXITER_S1 and info["maxiter"] == MAXITER_S2, name
    assert os.path.samefile(info["init"]["path"], t5.x_path(warm_name(arm, seed))), name
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


def paired_table(runs):
    """Paired differences over seeds where both sides drive, for four comparisons."""
    import pandas as pd
    key = runs.set_index(["arm", "stage", "seed"])
    comps = [("tuned - control", ("tuned", "final"), ("control", "final")),
             ("tuned - control", ("tuned", "stage1"), ("control", "stage1")),
             ("final - stage1", ("control", "final"), ("control", "stage1")),
             ("final - stage1", ("tuned", "final"), ("tuned", "stage1"))]
    rows = []
    for label, a, b in comps:
        seeds = sorted(set(runs.seed))
        ok = [s for s in seeds if (a + (s,)) in key.index and (b + (s,)) in key.index
              and not key.loc[a + (s,), "parked"] and not key.loc[b + (s,), "parked"]]
        where = (f"{a[1]}" if label.startswith("tuned") else f"{a[0]}")
        for col, sign in METRICS:
            d = np.array([key.loc[a + (s,), col] - key.loc[b + (s,), col] for s in ok],
                         dtype=float)
            mean = float(d.mean()) if len(d) else np.nan
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
            rows.append({"comparison": label, "at": where, "metric": col,
                         "n": len(d), "mean_a": float(np.mean([key.loc[a + (s,), col] for s in ok])),
                         "mean_b": float(np.mean([key.loc[b + (s,), col] for s in ok])),
                         "diff_mean": mean, "diff_se": se,
                         "n_better": int((sign * d > 0).sum()),
                         "resolved": bool(abs(mean) > 2 * se) if len(d) > 1 else False})
    return pd.DataFrame(rows)


def summarize(jobs=8):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    check_fixed_seeds()
    seeds, repl, walk = campaign_seeds()
    with open(SEED_JSON, "w") as fh:
        json.dump({"fixed": list(FIXED_SEEDS), "dropped": DROPPED_SEED,
                   "replacement": repl, "walk": walk}, fh, indent=2)
    os.makedirs(os.path.join(RUN_DIR, "scores"), exist_ok=True)
    present, missing, infos = [], [], {}
    for s in seeds:
        for arm in ARMS:
            name = run_name(arm, s)
            if not t5.done(name):
                missing.append(name)
                continue
            infos[name] = _check_sidecar(name, arm, s)
            present += [(arm, s, name, st) for st in STAGES]
    if missing:
        print(f"{len(missing)} runs missing, e.g. {missing[:3]}")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
        scored = {job: (row, tr) for job, row, tr in
                  ex.map(_score_pulse, [(n, st) for _, _, n, st in present])}
    rows, traces = [], {}
    for arm, s, name, st in present:
        row, tr = scored[(name, st)]
        info = infos[name]
        k = 0 if st == "stage1" else 1
        tier = ("parked" if row["progress"] <= 0.5 else
                "closed" if row["F1"] >= 0.999 else "open")
        rows.append({"arm": arm, "seed": s, "stage": st, "tier": tier,
                     "parked": tier == "parked", **row,
                     "nit": info["stages"][k]["nit"],
                     "hours": info["stages"][k]["seconds"] / 3600})
        for key, v in tr.items():
            traces[f"{arm}/seed{s}/{st}/{key}"] = np.asarray(v)
    runs = pd.DataFrame(rows).sort_values(["stage", "arm", "seed"]).reset_index(drop=True)
    runs.to_csv(RUNS_CSV, index=False)
    np.savez(TRACES_NPZ, **traces)
    paired = paired_table(runs)
    paired.to_csv(PAIRED_CSV, index=False)
    print(f"tables -> {RUNS_CSV}\n          {PAIRED_CSV}\n          {TRACES_NPZ}\n")
    print(f"seeds {list(seeds)} (replacement for seed {DROPPED_SEED}: {repl})")
    fin = runs[runs.stage == "final"]
    print(fin[["arm", "seed", "tier", "F1", "F_ET", "dqec_mean", "L_mean", "eta_avg_mean",
               "dqec_tail", "L_tail", "eta_avg_tail", "L_T"]]
          .to_string(index=False, float_format="%.4f"))
    print()
    print(paired[["comparison", "at", "metric", "n", "mean_a", "mean_b", "diff_mean",
                  "diff_se", "n_better", "resolved"]]
          .to_string(index=False, float_format="%.4f"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["run", "status", "summarize"])
    p.add_argument("--jobs", type=int, default=23,
                   help="parallel single-threaded processes (training or scoring)")
    a = p.parse_args()
    if a.action == "run":
        run(a.jobs)
    elif a.action == "status":
        status()
    else:
        summarize(jobs=min(a.jobs, 8))


if __name__ == "__main__":
    main()
