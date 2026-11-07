"""
Section 7's C_err-in-stage-1-only pair over 13 seeds, X: est_optimization.ipynb section 7.2.

    python EST/err_two_stage_seeds.py run --jobs 24   # train everything missing, in parallel
    python EST/err_two_stage_seeds.py status
    python EST/err_two_stage_seeds.py summarize       # score every run, write the tables

Section 7 (EST/err_two_stage.py) trained one pair at seed 0. Its run is more
transparent than its control but has one Delta_QEC spike to 3.3 at 0.65 us. This
campaign trains the same two commands at more seeds and looks for a closed gate
with no such spike. The commands are err_two_stage.train_args with --seed s:

    run      est_err, stage 1 (w1, w2, w3, w4, w_err, w5, w6) = (1, 0.7, 7, 1, 0.3, 0, 0)
                      stage 2                                 = (1, 0.1, 0, 1, 0,   0, 0)
    control  plain est, stage 1 (1, 0.7, 7, 1), stage 2 (1, 0.1, 0, 1)

Cold starts, 2000 iterations per stage, n_c 20, one BLAS thread each, all on one
machine (the Windows PC, like seed 0). Logs and the score cache are section 7's,
~/est_runs/err2st/, so seed 0 is already trained and already scored.

Why not the photon-number term C5: split by Eq. A11, the seed-0 spike is almost all
the y part of the (a, a) and (I, a) pairs; the (a, a) z part, the only one C5 sees,
is 0.013 of 3.32. `spike_split` below reports that split for every final gate.

Fixed before training (2026-10-27):
  Seeds   0 plus section 6's twelve (tables/est_c2c5_2st_X_seeds.json).
  Pick    a candidate is a RUN-arm final that is closed (F1 >= 0.999), has peak
          Delta_QEC(t) <= 0.6 over the whole gate, and has F_ET >= its own seed's
          control. The pick is the candidate with the highest F_ET. Control finals
          that pass the first two conditions are reported, not picked. No candidate:
          report the closed run gate with the lowest peak and stop.
  Paired  run - control over seeds where both drive; resolved when |mean| > 2 SE.
  Prediction: 0 to 3 candidates (section 6 has 1 to 3 finals in 12 at <= 0.6);
          section 7's gains in F_ET, mean L and mean eta keep their sign but shrink.
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from EST import c2c5_two_stage as c2s
from EST import err_two_stage as e2s
from EST import tune_c2_c5 as t5

REPO_ROOT = e2s.REPO_ROOT
TABLE_DIR = e2s.TABLE_DIR
RUN_DIR = e2s.RUN_DIR

SEEDS = (0, 2, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16, 17)
ARMS = ("control", "run")
STAGES = e2s.STAGES
PEAK_MAX = 0.6            # the spike bar of the pick rule
F1_CLOSED = 0.999

RUNS_CSV = os.path.join(TABLE_DIR, "est_err2st_X_seeds_runs.csv")
PAIRED_CSV = os.path.join(TABLE_DIR, "est_err2st_X_seeds_paired.csv")
PICK_CSV = os.path.join(TABLE_DIR, "est_err2st_X_seeds_pick.csv")
SPLIT_CSV = os.path.join(TABLE_DIR, "est_err2st_X_seeds_spike_split.csv")
TRACES_NPZ = os.path.join(TABLE_DIR, "est_err2st_X_seeds_traces.npz")

METRICS = c2s.METRICS + (("dqec_peak", -1),)


def check_seeds():
    """SEEDS must be 0 plus section 6's campaign seeds, and seed 0 must be section 7's pair."""
    with open(c2s.SEED_JSON) as fh:
        s6 = json.load(fh)
    expect = (0,) + tuple(s6["fixed"]) + (s6["replacement"],)
    if tuple(sorted(expect)) != SEEDS:
        raise SystemExit(f"section 6 seeds give {sorted(expect)}; SEEDS is {SEEDS}")
    for arm in ARMS:
        assert e2s.run_name(arm, 0) == e2s.RUNS[arm], arm
        assert e2s.train_args(arm, 0) == e2s.train_args(arm), arm


# ---------------------------------------------------------------- launcher

def run(jobs):
    """Train every missing run, `jobs` single-threaded processes at a time (resumable)."""
    check_seeds()
    t5._keep_awake()
    os.makedirs(RUN_DIR, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    queue = [(arm, s) for s in SEEDS for arm in ARMS if not t5.done(e2s.run_name(arm, s))]
    running, n_done, t_start = {}, 0, time.time()
    print(f"{len(queue)} runs to train", flush=True)
    while queue or running:
        while queue and len(running) < jobs:
            arm, s = queue.pop(0)
            name = e2s.run_name(arm, s)
            log = open(os.path.join(RUN_DIR, f"{name}.log"), "w")
            p = subprocess.Popen([sys.executable] + e2s.train_args(arm, s), cwd=REPO_ROOT,
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
            ok = "ok" if p.returncode == 0 and t5.done(name) else f"FAILED rc={p.returncode}"
            print(f"[{time.strftime('%H:%M:%S')}] {ok} {name}  "
                  f"{(time.time() - t0) / 60:.1f} min  (done {n_done}, running "
                  f"{len(running)}, queued {len(queue)})", flush=True)
        time.sleep(5)
    print(f"campaign finished in {(time.time() - t_start) / 3600:.2f} h")


def status():
    check_seeds()
    for s in SEEDS:
        cells = []
        for arm in ARMS:
            name = e2s.run_name(arm, s)
            have = [st for st in STAGES if os.path.exists(e2s.stage_paths(name, st)[1])]
            cells.append(f"{arm} {'done' if t5.done(name) else 'stages ' + str(have)}")
        print(f"seed {s:2d}: " + "; ".join(cells))


# ---------------------------------------------------------------- scoring

def _check_sidecar(name, arm, seed):
    """The run trained what this campaign says it trained."""
    with open(t5.json_path(name)) as fh:
        info = json.load(fh)
    w = [tuple(s["weights"]) for s in info["stages"]]
    assert w == e2s.WEIGHTS[arm], (name, w)
    assert info["seed"] == seed and info["maxiter"] == e2s.MAXITER and info["init"] is None, name
    assert info["n_stages"] == 2 and info["pipeline"] == "eigh_adjoint", name
    return info


def spike_split(job):
    """
    Eq. A11's Delta_QEC split by error pair (I, a) x (I, a) and Pauli part x, y, z,
    at the time of the peak. (I, I) is exactly zero and (a, I) equals (I, a), so the
    columns are (I,a) and (a,a), each counted once; `total` adds (I,a) twice.
    The total is checked against EST/diagnostics.delta_qec at every time step.
    """
    name, stage = job
    from core.grape_core import make_ops
    from EST import diagnostics, kitten_code
    from EST.device import DT, N_T, make_hamiltonian_est
    n_c = t5.N_C
    H0, Hc = make_hamiltonian_est(N_T, n_c)
    A, _ = make_ops(N_T, n_c)
    u = np.load(e2s.stage_paths(name, stage)[0])
    W = np.asarray(diagnostics.propagate_states(u, H0, Hc,
                                                kitten_code.logical_basis(N_T, n_c), DT))
    pauli = {"x": np.array([[0, 1], [1, 0]], complex),
             "y": np.array([[0, -1j], [1j, 0]]),
             "z": np.diag([1.0, -1.0]).astype(complex)}
    parts = {}
    for label, op in (("Ia", A), ("aa", A.conj().T @ A)):
        M = np.einsum('tim,ij,tjn->tmn', np.conj(W), op, W)
        for p, P in pauli.items():
            parts[f"{label}_{p}"] = 2.0 * np.abs(0.5 * np.einsum('ij,tji->t', P, M)) ** 2
    total = 2 * sum(parts[f"Ia_{p}"] for p in pauli) + sum(parts[f"aa_{p}"] for p in pauli)
    ref = diagnostics.delta_qec(W, A)
    assert np.max(np.abs(total - ref)) < 1e-10, (name, np.max(np.abs(total - ref)))
    k = int(np.argmax(total))
    return job, {"peak": float(total[k]), "peak_t": k / len(total),
                 **{c: float(v[k]) for c, v in parts.items()}}


def paired_table(runs):
    """run - control at each stage, and final - stage1 per arm, over seeds where both drive."""
    import pandas as pd
    key = runs.set_index(["arm", "stage", "seed"])
    comps = [("run - control", ("run", "final"), ("control", "final")),
             ("run - control", ("run", "stage1"), ("control", "stage1")),
             ("final - stage1", ("control", "final"), ("control", "stage1")),
             ("final - stage1", ("run", "final"), ("run", "stage1"))]
    rows = []
    for label, a, b in comps:
        ok = [s for s in sorted(set(runs.seed))
              if (a + (s,)) in key.index and (b + (s,)) in key.index
              and not key.loc[a + (s,), "parked"] and not key.loc[b + (s,), "parked"]]
        where = a[1] if label.startswith("run") else a[0]
        for col, sign in METRICS:
            d = np.array([key.loc[a + (s,), col] - key.loc[b + (s,), col] for s in ok], float)
            mean = float(d.mean()) if len(d) else np.nan
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
            rows.append({"comparison": label, "at": where, "metric": col, "n": len(d),
                         "mean_a": float(np.mean([key.loc[a + (s,), col] for s in ok])),
                         "mean_b": float(np.mean([key.loc[b + (s,), col] for s in ok])),
                         "diff_mean": mean, "diff_se": se,
                         "n_better": int((sign * d > 0).sum()),
                         "resolved": bool(abs(mean) > 2 * se) if len(d) > 1 else False})
    return pd.DataFrame(rows)


def pick_table(runs):
    """The pick rule of the module docstring, one row per seed's final run gate."""
    import pandas as pd
    fin = runs[runs.stage == "final"].set_index(["arm", "seed"])
    rows = []
    for s in sorted(set(runs.seed)):
        if ("run", s) not in fin.index or ("control", s) not in fin.index:
            continue
        r, c = fin.loc[("run", s)], fin.loc[("control", s)]
        closed, low = bool(r.F1 >= F1_CLOSED), bool(r.dqec_peak <= PEAK_MAX)
        beats = bool(r.F_ET >= c.F_ET)
        rows.append({"seed": s, "F1": r.F1, "F_ET": r.F_ET, "control_F_ET": c.F_ET,
                     "dqec_peak": r.dqec_peak, "dqec_peak_t": r.dqec_peak_t,
                     "control_dqec_peak": c.dqec_peak, "closed": closed,
                     "peak_ok": low, "F_ET_ok": beats, "candidate": closed and low and beats,
                     "control_closed_peak_ok": bool(c.F1 >= F1_CLOSED and c.dqec_peak <= PEAK_MAX)})
    pick = pd.DataFrame(rows)
    pick["picked"] = False
    cand = pick[pick.candidate]
    if len(cand):
        pick.loc[cand.F_ET.idxmax(), "picked"] = True
    return pick


def summarize(jobs=8):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    check_seeds()
    os.makedirs(os.path.join(RUN_DIR, "scores"), exist_ok=True)
    present, missing, infos = [], [], {}
    for s in SEEDS:
        for arm in ARMS:
            name = e2s.run_name(arm, s)
            if not t5.done(name):
                missing.append(name)
                continue
            infos[name] = _check_sidecar(name, arm, s)
            present += [(arm, s, name, st) for st in STAGES]
    if missing:
        print(f"{len(missing)} runs missing, e.g. {missing[:3]}")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    finals = [(n, "final") for _, _, n, st in present if st == "final"]
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
        scored = {job: (row, tr) for job, row, tr in
                  ex.map(e2s._score_pulse, [(n, st) for _, _, n, st in present])}
        split = dict(ex.map(spike_split, finals))
    rows, traces = [], {}
    for arm, s, name, st in present:
        row, tr = scored[(name, st)]
        info = infos[name]
        k = 0 if st == "stage1" else 1
        tier = ("parked" if row["progress"] <= 0.5 else
                "closed" if row["F1"] >= F1_CLOSED else "open")
        dq = np.asarray(tr["dqec"])
        rows.append({"arm": arm, "seed": s, "stage": st, "tier": tier,
                     "parked": tier == "parked", "name": name, **row,
                     "dqec_peak": float(dq.max()), "dqec_peak_t": int(dq.argmax()) / len(dq),
                     "dqec_frac_gt_peak_max": float(np.mean(dq > PEAK_MAX)),
                     "nit": info["stages"][k]["nit"],
                     "hours": info["stages"][k]["seconds"] / 3600})
        for key, v in tr.items():
            traces[f"{arm}/seed{s}/{st}/{key}"] = np.asarray(v)
    runs = pd.DataFrame(rows).sort_values(["stage", "arm", "seed"]).reset_index(drop=True)
    runs.to_csv(RUNS_CSV, index=False)
    np.savez(TRACES_NPZ, **traces)
    paired = paired_table(runs)
    paired.to_csv(PAIRED_CSV, index=False)
    pick = pick_table(runs)
    pick.to_csv(PICK_CSV, index=False)
    arm_of = {e2s.run_name(a, s): (a, s) for a in ARMS for s in SEEDS}
    spl = pd.DataFrame([{"arm": arm_of[n][0], "seed": arm_of[n][1], **v}
                        for (n, _), v in split.items()]).sort_values(["arm", "seed"])
    spl.to_csv(SPLIT_CSV, index=False)
    print(f"tables -> {RUNS_CSV}\n          {PAIRED_CSV}\n          {PICK_CSV}\n"
          f"          {SPLIT_CSV}\n          {TRACES_NPZ}\n")
    fin = runs[runs.stage == "final"]
    print(fin[["arm", "seed", "tier", "F1", "F_ET", "dqec_mean", "dqec_peak", "dqec_peak_t",
               "L_mean", "eta_avg_mean", "L_T"]].to_string(index=False, float_format="%.4f"))
    print()
    print(pick.to_string(index=False, float_format="%.4f"))
    print()
    print(spl.to_string(index=False, float_format="%.3f"))
    print()
    print(paired[["comparison", "at", "metric", "n", "mean_a", "mean_b", "diff_mean",
                  "diff_se", "n_better", "resolved"]]
          .to_string(index=False, float_format="%.4f"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["run", "status", "summarize"])
    p.add_argument("--jobs", type=int, default=24,
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
