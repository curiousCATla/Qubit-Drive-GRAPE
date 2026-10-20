"""
Joint tuning of the C2 weight (w2) and the C5 weight (w5) on the X gate,
est_optimization.ipynb section 5 ("Phase 1": a 1000-iteration scan).

    python EST/tune_c2_c5.py run --jobs 16     # train everything missing, in parallel
    python EST/tune_c2_c5.py status            # how far the campaign has got
    python EST/tune_c2_c5.py seeds             # score the controls, pick the seeds
    python EST/tune_c2_c5.py summarize         # score every run, apply the rule

Pipeline: the numpy eigh + hand-adjoint driver EST/train_est_eigh.py (not JAX),
single phase (`--stages 1`), 1000 iterations, `--variant est_dn` with `--w2`
and `--w-dn`. Each run is one single-threaded process (OMP/OpenBLAS/MKL threads
= 1); `run` keeps `--jobs` of them going at once.

Grid: w2 in {0.6, 0.7, 0.8, 0.9, 1.0} x w5 in {2, 2.5, 3, 3.5, 4}. The control
of each seed is plain `est` (w2 = 0.7, w5 = 0), same seed, same budget.

Seeds (fixed before training): train controls on seeds 0, 2, 4, 5, 6, 8 (the
section 9 seeds that reach a gate) and 10-21; keep the seeds whose control
DRIVES (progress > 0.5), in seed order, until there are N_SEEDS = 12. If fewer
than 12 drive, controls on 22-29 are added. Every control is retrained on this
machine: the Mac controls of section 2.1 used another BLAS and are not
comparable run for run.

Score. For each run, the three time-averaged error-transparency metrics of
Eqs. 6-8 (mean Delta_QEC, mean leakage L, mean six-cardinal eta) are divided by
the same seed's control value; `score` is the mean of the three ratios (< 1 is
more transparent than the control). L(T) is reported, never scored.

Rule (fixed before training). A run is PARKED if progress <= 0.5; parked runs
are left out of every mean. A driving run passes the fidelity bar if
F1 >= its control's F1 - F1_DROP_TOL (0.01). A cell is ELIGIBLE if at most
MAX_PARKED (1) of its seeds park AND every driving seed passes the bar. The pick
is the eligible cell with the lowest seed-mean score. The runner-up is printed,
and the pick is flagged when the gap is within one standard error.
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from EST.train_est_eigh import LOG_DIR, PULSE_DIR

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_DIR = os.path.join(REPO_ROOT, "tables")
RUN_DIR = os.path.join(os.path.expanduser("~"), "est_runs", "c2c5")   # stdout logs, score cache

GATE, MAXITER, N_C, N_STAGES = "X", 1000, 20, 1
W2_GRID = (0.6, 0.7, 0.8, 0.9, 1.0)
W5_GRID = (2.0, 2.5, 3.0, 3.5, 4.0)
CTRL_W2 = 0.7                     # the `est` stage-1 weight
SEED_POOL = (0, 2, 4, 5, 6, 8) + tuple(range(10, 22))
SEED_EXTENSION = tuple(range(22, 30))
N_SEEDS = 12
F1_DROP_TOL = 0.01
MAX_PARKED = 1
ET_METRICS = ("delta_qec_mean", "L_mean", "eta_avg_mean")
RATIO_COLS = {"delta_qec_mean": "r_dqec", "L_mean": "r_L", "eta_avg_mean": "r_eta"}

SEEDS_CSV = os.path.join(TABLE_DIR, "est_c2c5_seeds_X_sp.csv")
SCAN_CSV = os.path.join(TABLE_DIR, "est_c2c5_scan_X_sp.csv")
CELLS_CSV = os.path.join(TABLE_DIR, "est_c2c5_cells_X_sp.csv")


# ---------------------------------------------------------------- names

def ctrl_name(seed):
    return f"est_c2c5_sp{MAXITER}_ctrl_seed{seed}"


def cell_name(seed, w2, w5):
    return f"est_dn_c2c5_sp{MAXITER}_seed{seed}_w2_{w2:g}_w5_{w5:g}"


def x_path(name):
    return os.path.join(PULSE_DIR, f"x_{GATE}_{name}.npy")


def json_path(name):
    return os.path.join(LOG_DIR, f"est_{GATE}_{name}.json")


def done(name):
    return os.path.exists(x_path(name)) and os.path.exists(json_path(name))


def train_args(name):
    """Command-line arguments of train_est_eigh.py for one run name."""
    base = ["EST/train_est_eigh.py", "--gate", GATE, "--stages", str(N_STAGES),
            "--maxiter", str(MAXITER)]
    if name.startswith("est_c2c5"):
        seed = int(name.rsplit("seed", 1)[1])
        tag = name[len("est_"):]
        return base + ["--variant", "est", "--seed", str(seed), "--tag", tag]
    # est_dn_c2c5_sp1000_seed<s>_w2_<a>_w5_<b>
    parts = name.split("_")
    seed = int(parts[4][len("seed"):])
    w2, w5 = parts[6], parts[8]
    tag = name[len("est_dn_"):]
    return base + ["--variant", "est_dn", "--seed", str(seed), "--w2", w2,
                   "--w-dn", w5, "--tag", tag]


# ---------------------------------------------------------------- scoring

def _require_single_phase(jpath):
    with open(jpath) as fh:
        info = json.load(fh)
    n = info.get("n_stages", len(info.get("stages", [])))
    if n != N_STAGES:
        raise SystemExit(f"{jpath} was trained with {n} stage(s); this campaign is "
                         f"single phase. Retrain it with --stages {N_STAGES}.")
    return info


def _score_one(name):
    """score_x for one run, cached on disk by the pre-image's mtime."""
    from EST import smoke_new_terms
    os.makedirs(os.path.join(RUN_DIR, "scores"), exist_ok=True)
    cache = os.path.join(RUN_DIR, "scores", f"{name}.json")
    mtime = os.path.getmtime(x_path(name))
    if os.path.exists(cache):
        with open(cache) as fh:
            c = json.load(fh)
        if c.get("mtime") == mtime:
            return name, c["score"]
    info = _require_single_phase(json_path(name))
    s = smoke_new_terms.score_x(np.load(x_path(name)))
    s["nit"] = int(info["stages"][0]["nit"])
    s["seconds"] = float(info["stages"][0]["seconds"])
    s["w2_trained"] = float(info["stages"][0]["weights"][1])
    s["w5_trained"] = float(info["stages"][0]["weights"][5]) \
        if len(info["stages"][0]["weights"]) == 7 else 0.0
    with open(cache, "w") as fh:
        json.dump({"mtime": mtime, "score": s}, fh)
    return name, s


def score_many(names, jobs):
    names = [n for n in names if done(n)]
    if not names:
        return {}
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
        return dict(ex.map(_score_one, names))


# ---------------------------------------------------------------- seeds

def select_seeds(scores):
    """(selected seeds, table rows, all controls done?) from control scores.

    Seeds are taken in pool order; a seed whose control is missing stops the
    walk, so the selection never depends on which runs happened to finish first.
    """
    rows, chosen, complete = [], [], True
    for seed in SEED_POOL + SEED_EXTENSION:
        name = ctrl_name(seed)
        if name not in scores:
            if seed in SEED_POOL or len(chosen) < N_SEEDS:
                complete = False
            break
        s = scores[name]
        drives = s["progress"] > 0.5
        take = drives and len(chosen) < N_SEEDS
        if take:
            chosen.append(seed)
        rows.append({"seed": seed, "drives": drives, "selected": take, **s})
        if len(chosen) >= N_SEEDS and seed >= max(SEED_POOL):
            break
    return chosen, rows, complete and len(chosen) >= N_SEEDS


def seeds(jobs=8, write=True):
    import pandas as pd
    names = [ctrl_name(s) for s in SEED_POOL + SEED_EXTENSION]
    scores = score_many(names, jobs)
    chosen, rows, complete = select_seeds(scores)
    df = pd.DataFrame(rows)
    if write and len(df):
        df.to_csv(SEEDS_CSV, index=False)
        print(f"table -> {SEEDS_CSV}")
    if len(df):
        print(df[["seed", "F1", "progress", "drives", "selected", "F_ET",
                  "delta_qec_mean", "L_mean", "eta_avg_mean", "L_T"]]
              .to_string(index=False, float_format="%.4f"))
    print(f"selected seeds ({len(chosen)}): {chosen}"
          + ("" if complete else "   [INCOMPLETE: controls still missing]"))
    return chosen, complete


# ---------------------------------------------------------------- launcher

def _keep_awake():
    if os.name == "nt":
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no sleep while this process lives.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def run(jobs):
    """Train every missing run, `jobs` single-threaded processes at a time.

    Queue order: missing controls of the seed pool, then the grid for the six
    section 9 seeds (they come first in seed order, so they are selected if their
    controls drive; their grid runs only fill slots the controls leave free).
    When every control is done, the seeds are selected and the rest of the grid
    is queued. Runs whose pre-image already exists are skipped, so `run` resumes.
    """
    _keep_awake()
    os.makedirs(RUN_DIR, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    queued, queue, running = set(), [], {}

    def enqueue(names):
        for n in names:
            if n not in queued and not done(n):
                queued.add(n)
                queue.append(n)

    def grid(seed_list):
        return [cell_name(s, w2, w5) for s in seed_list
                for w2 in W2_GRID for w5 in W5_GRID]

    enqueue(ctrl_name(s) for s in SEED_POOL)
    enqueue(grid(SEED_POOL[:6]))
    selected_final = False
    t_start, n_done = time.time(), 0
    while True:
        # Seed selection once every needed control has finished.
        if not selected_final:
            pool_running = any(n.startswith("est_c2c5") for n in running)
            pool_queued = any(n.startswith("est_c2c5") for n in queue)
            if not pool_running and not pool_queued:
                chosen, complete = seeds(jobs=min(jobs, 8), write=True)
                if complete:
                    enqueue(grid(chosen))
                    selected_final = True
                else:
                    ext = [ctrl_name(s) for s in SEED_EXTENSION if not done(ctrl_name(s))]
                    if not ext:
                        print("seed pool exhausted with fewer than "
                              f"{N_SEEDS} driving controls; grid runs on {chosen}")
                        enqueue(grid(chosen))
                        selected_final = True
                    else:
                        enqueue(ext[:1])     # one at a time, keeps seed order exact
        while queue and len(running) < jobs:
            name = queue.pop(0)
            log = open(os.path.join(RUN_DIR, f"{name}.log"), "w")
            p = subprocess.Popen([sys.executable] + train_args(name), cwd=REPO_ROOT,
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
            status = "ok" if p.returncode == 0 and done(name) else f"FAILED rc={p.returncode}"
            print(f"[{time.strftime('%H:%M:%S')}] {status} {name}  "
                  f"{(time.time() - t0) / 60:.1f} min  (done {n_done}, running "
                  f"{len(running)}, queued {len(queue)})", flush=True)
        if not running and not queue and selected_final:
            break
        time.sleep(5)
    print(f"campaign finished in {(time.time() - t_start) / 3600:.2f} h")


def status():
    ctrl = [ctrl_name(s) for s in SEED_POOL + SEED_EXTENSION]
    print(f"controls done: {sum(done(n) for n in ctrl)}")
    cells = [cell_name(s, a, b) for s in SEED_POOL + SEED_EXTENSION
             for a in W2_GRID for b in W5_GRID]
    finished = [n for n in cells if done(n)]
    print(f"grid runs done: {len(finished)}")
    secs = []
    for n in finished + [c for c in ctrl if done(c)]:
        with open(json_path(n)) as fh:
            secs.append(json.load(fh)["stages"][0]["seconds"])
    if secs:
        print(f"seconds per run: median {np.median(secs):.0f}, max {np.max(secs):.0f}")


# ---------------------------------------------------------------- rule

def apply_rule(scan):
    """Per-cell table and the pick from a `summarize` scan table (or its CSV)."""
    import pandas as pd
    rows = []
    n_expected = int((scan.term == "control").sum())   # one control per selected seed
    for (w2, w5), d in scan[scan.term == "cell"].groupby(["w2", "w5"]):
        drove = d[~d.parked.astype(bool)]
        n_parked = int(d.parked.astype(bool).sum())
        n_f1_ok = int(drove.f1_ok.astype(bool).sum())
        r = {"w2": w2, "w5": w5, "n_seeds": len(d), "n_parked": n_parked,
             "n_f1_ok": n_f1_ok, "n_drove": len(drove)}
        for col in ("score",) + tuple(RATIO_COLS.values()) + ET_METRICS + (
                "L_T", "F1", "dF1", "F_ET", "c5", "F_err_T"):
            r[col] = float(drove[col].mean()) if len(drove) else np.nan
        r["score_se"] = (float(drove.score.std(ddof=1) / np.sqrt(len(drove)))
                         if len(drove) > 1 else np.nan)
        r["dF1_min"] = float(drove.dF1.min()) if len(drove) else np.nan
        # A cell missing seeds is never eligible: the rule is stated over all of them.
        r["eligible"] = bool(len(d) == n_expected and n_parked <= MAX_PARKED
                             and len(drove) > 0 and n_f1_ok == len(drove))
        rows.append(r)
    cells = pd.DataFrame(rows).sort_values(["w2", "w5"]).reset_index(drop=True)
    ok = cells[cells.eligible].sort_values("score")
    pick = runner = None
    if len(ok):
        pick = ok.iloc[0]
        runner = ok.iloc[1] if len(ok) > 1 else None
    cells["pick"] = False
    if pick is not None:
        cells.loc[(cells.w2 == pick.w2) & (cells.w5 == pick.w5), "pick"] = True
    return cells, pick, runner


def summarize(jobs=8):
    import pandas as pd
    chosen, complete = seeds(jobs=jobs, write=True)
    if not complete:
        print("WARNING: seed selection is not final; summarizing what exists.")
    ctrl_names = [ctrl_name(s) for s in chosen]
    cell_names = [cell_name(s, a, b) for s in chosen for a in W2_GRID for b in W5_GRID]
    scores = score_many(ctrl_names + cell_names, jobs)
    rows, missing = [], []
    for seed in chosen:
        c = scores[ctrl_name(seed)]
        rows.append({"term": "control", "seed": seed, "w2": CTRL_W2, "w5": 0.0,
                     **c, "parked": c["progress"] <= 0.5, "dF1": 0.0, "f1_ok": True,
                     **{RATIO_COLS[m]: 1.0 for m in ET_METRICS}, "score": 1.0})
        for w2 in W2_GRID:
            for w5 in W5_GRID:
                name = cell_name(seed, w2, w5)
                if name not in scores:
                    missing.append(name)
                    continue
                s = dict(scores[name])
                assert abs(s["w2_trained"] - w2) < 1e-12 and abs(s["w5_trained"] - w5) < 1e-12, name
                ratios = {RATIO_COLS[m]: s[m] / c[m] for m in ET_METRICS}
                s.update(ratios)
                s["score"] = float(np.mean(list(ratios.values())))
                s["parked"] = s["progress"] <= 0.5
                s["dF1"] = s["F1"] - c["F1"]
                s["f1_ok"] = bool(s["dF1"] >= -F1_DROP_TOL)
                rows.append({"term": "cell", "seed": seed, "w2": w2, "w5": w5, **s})
    if missing:
        print(f"{len(missing)} grid runs missing, e.g. {missing[:3]}")
    scan = pd.DataFrame(rows)
    scan.to_csv(SCAN_CSV, index=False)
    print(f"table -> {SCAN_CSV}")
    cells, pick, runner = apply_rule(scan)
    cells.to_csv(CELLS_CSV, index=False)
    print(f"table -> {CELLS_CSV}\n")
    cols = ["w2", "w5", "n_parked", "n_f1_ok", "n_drove", "eligible", "score",
            "score_se", "r_dqec", "r_L", "r_eta", "L_T", "dF1", "dF1_min", "F_ET"]
    print(cells[cols].to_string(index=False, float_format="%.4f"))
    ctrl = scan[scan.term == "control"]
    print(f"\ncontrol means: F1 {ctrl.F1.mean():.4f}  Delta_QEC {ctrl.delta_qec_mean.mean():.4f}"
          f"  L {ctrl.L_mean.mean():.4f}  eta {ctrl.eta_avg_mean.mean():.4f}"
          f"  L(T) {ctrl.L_T.mean():.4f}")
    if pick is None:
        print("NO cell is eligible under the rule; nothing is picked.")
        return
    print(f"pick: w2 = {pick.w2:g}, w5 = {pick.w5:g}, score {pick.score:.4f} "
          f"+- {pick.score_se:.4f}")
    if runner is not None:
        gap = runner.score - pick.score
        print(f"runner-up: w2 = {runner.w2:g}, w5 = {runner.w5:g}, score "
              f"{runner.score:.4f} (gap {gap:.4f})"
              + ("   [WITHIN ONE STANDARD ERROR: not resolved]"
                 if not gap >= pick.score_se else ""))   # NaN SE: not resolved either


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["run", "status", "seeds", "summarize"])
    p.add_argument("--jobs", type=int, default=16,
                   help="parallel single-threaded processes (training or scoring)")
    a = p.parse_args()
    if a.action == "run":
        run(a.jobs)
    elif a.action == "status":
        status()
    elif a.action == "seeds":
        seeds(jobs=min(a.jobs, 8))
    else:
        summarize(jobs=min(a.jobs, 8))


if __name__ == "__main__":
    main()
