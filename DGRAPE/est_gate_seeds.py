#!/usr/bin/env python3
"""
Seed selection for a more transparent EsT gate on the Shirol chip: est_gate_selection.ipynb.

    python DGRAPE/est_gate_seeds.py run --wave 1 --jobs 25   # train every missing seed of the wave
    python DGRAPE/est_gate_seeds.py status
    python DGRAPE/est_gate_seeds.py summarize                # score every run, apply the rule, write tables
    python DGRAPE/est_gate_seeds.py open_score --top 3       # open-system score of the top candidates
    python DGRAPE/est_gate_seeds.py run --gate H --wave 1    # the same for the H gate (notebook §7)

WHAT IS TRAINED. DGRAPE/est_gate.py's variant est_err: the EsT gate of FWM_tails.ipynb §3 (closed system,
n_r = 1, n_c 20, 1.1 us, d2's drive limits and smooth start), with est_optimization.ipynb §7's cost, i.e.
C_err (C1's form on the error cardinals at t = T) added in stage 1 only:
    stage 1 (w1, w2, w3, w4, w_err, w5, w6) = (1, 0.7, 7, 1, 0.3, 0, 0)
    stage 2                                 = (1, 0.1, 0, 1, 0,   0, 0)
2000 iterations per stage, one BLAS thread per run, all on the Windows PC. Runs go to paths(gate)["run_dir"]
(outside iCloud); the picks are copied into pulses/dgrape/ by hand.

REFERENCE. The current EsT gate, pulses/dgrape/u_X_est_closed.npy (variant est, seed 42), and the two other
samples of that training (_s43, _s44), scored the same way. H has no reference (user, 2026-11-01).

RULE (fixed before training, 2026-10-29). Every pulse is scored closed at n_c 24 (EST/diagnostics, via
est_gate.metrics). eta is the six-cardinal mean; "tail" is the mean over t >= 0.9 T (the repo convention,
EST/compare_new_terms.TAIL_FRAC); the value at T is reported only. A final gate is a CANDIDATE when
    F1 >= 0.99 and not parked;
    peaks   max_t Delta_QEC <= 0.5,  max_t L <= 0.5,  max_t eta <= 0.5;
    tails   Delta_QEC_tail <= 0.1,  L_tail <= 0.2,  eta_tail <= 0.2;
    |F1(n_c 28) - F1(n_c 24)| < 1e-4 (the pulse does not lean on the truncation).
Candidates are ranked by score = (mean Delta_QEC + mean L + mean eta) / 3, lowest first; "more transparent
than the current gate" means a lower score than X_est_closed's.
WAVES: wave 1 = seeds 0-24; wave 2 = seeds 25-49 only if wave 1 gives fewer than 3 candidates. Wave 1 gave 3, but
wave 2 was run anyway at the user's request (2026-10-29); the rule and the ranking are unchanged and cover all 50.

H (est_gate_selection.ipynb §7, user, 2026-11-01). The same cost, seeds, limits and bars, for the H gate. Both
waves are run (no stop rule). Candidates are ranked by peak_score = (peak Delta_QEC + peak L + peak eta) / 3,
lowest first ("looking for small peaks"); the mean score is reported too. No reference gate, so no beats_current.
Result: 0 candidates of 50 (28 parked, 1 wall, 21 gates); every gate fails the L and eta bars, eta rising to ~0.9 at
the end of stage 2, where C_err is off.

VARIANT est_err2 (user, 2026-11-02). est_err with C_err kept on in stage 2: (1, 0.1, 0, 1, 0.3). Its stage 1 is
est_err's, so every gate of the est_err search (tier open or closed; resume_seeds) is resumed at stage 2 from its
stage-1 pre-image (--start-stage 2). Same rule and ranking; its own run directory and tables (paths()).
    python DGRAPE/est_gate_seeds.py run --gate H --variant est_err2 --jobs 25

H AT 2 us (user, 2026-11-02). The same est_err cost, limits and bars, with the gate almost twice as long:
--duration-us 2 (N = 1000; the tail is then the last 0.2 us). Seeds 0-24 (wave 1 only). Candidates are ranked by the
mean score, as for X ("low averages"); peak_score is reported too. Runs, tags and tables carry a _T2 suffix (paths()).
    python DGRAPE/est_gate_seeds.py run --gate H --duration-us 2 --wave 1 --jobs 25
After the first look (3 candidates: 7, 17, 23) the user lowered the peak Delta_QEC bar to 0.4 for this search only
(2026-11-02; peak_max()): one candidate, seed 17. Seeds 7 and 23 are still scored in the open system for comparison:
    python DGRAPE/est_gate_seeds.py open_score --gate H --duration-us 2 --top 3 --extra-seeds 7 23
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np                                    # noqa: E402

GATE, VARIANT, MAXITER = "X", "est_err", 2000
DURATION = 1.1                          # us, est_gate.DURATION_US
GATES = ("X", "H")
VARIANTS = ("est_err", "est_err2")
DURATIONS = (1.1, 2.0)
WAVES = {1: tuple(range(0, 25)), 2: tuple(range(25, 50))}
TABLE_DIR = os.path.join(REPO_ROOT, "tables")
PULSE_DIR = os.path.join(REPO_ROOT, "pulses", "dgrape")
REFERENCES = {"X_est_closed": 42, "X_est_closed_s43": 43, "X_est_closed_s44": 44}
CURRENT = "X_est_closed"
REFERENCES_BY_GATE = {"X": REFERENCES, "H": {}}
CURRENT_BY_GATE = {"X": CURRENT, "H": None}
RANK_BY = {("X", 1.1): "score", ("H", 1.1): "peak_score", ("H", 2.0): "score"}

N_C_SCORE = 24
TRUNC = (20, 24, 28)
TAIL_FRAC = 0.9
F1_MIN = 0.99
PEAK_MAX = {"dqec": 0.5, "L": 0.5, "eta": 0.5}
PEAK_MAX_BY = {("H", 2.0): {"dqec": 0.4}}  # user, 2026-11-02: peak Delta_QEC <= 0.4 for the 2 us H search
TAIL_MAX = {"dqec": 0.1, "L": 0.2, "eta": 0.2}
TRUNC_TOL = 1e-4
WALL_TOL = 1e-2                         # |F1(20) - F1(24)| above this: tier "wall" (label only; fails trunc_ok)
MIN_CANDIDATES = 3                      # X: wave 2 runs only if wave 1 gives fewer


def _dsuffix(duration):
    """'' at the 1.1 us default, else e.g. '_T2'."""
    return "" if duration == DURATION else f"_T{duration:g}"


def rank_by(gate=GATE, duration=DURATION):
    return RANK_BY.get((gate, duration), "score")


def peak_max(gate=GATE, duration=DURATION):
    """The peak bars of one search: PEAK_MAX with that search's overrides."""
    return {**PEAK_MAX, **PEAK_MAX_BY.get((gate, duration), {})}


def paths(gate=GATE, variant=VARIANT, duration=DURATION):
    """Run directory and output tables of one gate's (and variant's, and duration's) search."""
    d = _dsuffix(duration)
    if variant == VARIANT:
        sub, stem = ("est_err_seeds" if gate == "X" else f"est_err_seeds_{gate}") + d, f"est_gate_seeds_{gate}{d}"
    else:
        sub, stem = f"{variant}_seeds_{gate}{d}", f"est_gate_seeds_{gate}{d}_{variant.replace('est_', '')}"
    return {"run_dir": os.path.join(os.path.expanduser("~"), "fwm_runs", sub),
            "runs_csv": os.path.join(TABLE_DIR, f"{stem}_runs.csv"),
            "pick_csv": os.path.join(TABLE_DIR, f"{stem}_pick.csv"),
            "open_csv": os.path.join(TABLE_DIR, f"{stem}_open.csv"),
            "traces_npz": os.path.join(TABLE_DIR, f"{stem}_traces.npz")}


def resume_seeds(gate, duration=DURATION):
    """est_err2: the seeds whose est_err run made a gate (final tier open or closed), from its runs table."""
    import pandas as pd
    runs = pd.read_csv(paths(gate, VARIANT, duration)["runs_csv"])
    fin = runs[(runs.stage == "final") & (runs.kind == "run") & runs.tier.isin(["open", "closed"])]
    return tuple(int(s) for s in sorted(fin.seed))


def seeds_of(gate, variant, wave=None, duration=DURATION):
    if variant == VARIANT:
        return WAVES[wave] if wave is not None else WAVES[1] + WAVES[2]
    return resume_seeds(gate, duration)


_X = paths("X")
RUN_DIR, RUNS_CSV, PICK_CSV, OPEN_CSV, TRACES_NPZ = (_X[k] for k in ("run_dir", "runs_csv", "pick_csv", "open_csv",
                                                                     "traces_npz"))


# ---------------------------------------------------------------- names and launcher

def tag(seed, gate=GATE, variant=VARIANT, duration=DURATION):
    return f"{gate}_{variant}_closed{_dsuffix(duration)}_s{seed}"


def done(name, where=RUN_DIR):
    return (os.path.exists(os.path.join(where, f"u_{name}.npy"))
            and os.path.exists(os.path.join(where, f"{name}.json")))


def stage1_preimage(seed, gate, duration=DURATION):
    """The est_err run's stage-1 pre-image, where an est_err2 run starts."""
    return os.path.join(paths(gate, VARIANT, duration)["run_dir"],
                        f"x_{tag(seed, gate, VARIANT, duration)}_stage1.npy")


def train_args(seed, gate=GATE, variant=VARIANT, duration=DURATION):
    args = ["DGRAPE/train_est_gate.py", "--gate", gate, "--variant", variant, "--seed", str(seed),
            "--maxiter", str(MAXITER), "--tag", tag(seed, gate, variant, duration),
            "--outdir", paths(gate, variant, duration)["run_dir"]]
    if duration != DURATION:
        args += ["--duration-us", str(duration)]
    if variant != VARIANT:
        args += ["--init-x", stage1_preimage(seed, gate, duration), "--start-stage", "2"]
    return args


def _keep_awake():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def run(wave, jobs, gate=GATE, variant=VARIANT, duration=DURATION):
    """Train every missing seed of the wave (est_err2: of resume_seeds), `jobs` single-threaded processes at a
    time (resumable)."""
    _keep_awake()
    run_dir = paths(gate, variant, duration)["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    queue = [s for s in seeds_of(gate, variant, wave, duration) if not done(tag(s, gate, variant, duration), run_dir)]
    running, n_done, t_start = {}, 0, time.time()
    print(f"{gate} {variant} {duration} us wave {wave}: {len(queue)} runs to train", flush=True)
    while queue or running:
        while queue and len(running) < jobs:
            s = queue.pop(0)
            log = open(os.path.join(run_dir, f"{tag(s, gate, variant, duration)}.log"), "w")
            p = subprocess.Popen([sys.executable] + train_args(s, gate, variant, duration), cwd=REPO_ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
            running[s] = (p, log, time.time())
            print(f"[{time.strftime('%H:%M:%S')}] start seed {s}  (running {len(running)}, queued {len(queue)})",
                  flush=True)
        for s, (p, log, t0) in list(running.items()):
            if p.poll() is None:
                continue
            log.close()
            del running[s]
            n_done += 1
            ok = ("ok" if p.returncode == 0 and done(tag(s, gate, variant, duration), run_dir)
                  else f"FAILED rc={p.returncode}")
            print(f"[{time.strftime('%H:%M:%S')}] {ok} seed {s}  {(time.time() - t0) / 60:.1f} min  "
                  f"(done {n_done}, running {len(running)}, queued {len(queue)})", flush=True)
        time.sleep(5)
    print(f"{gate} {variant} wave {wave} finished in {(time.time() - t_start) / 3600:.2f} h", flush=True)


def status(gate=GATE, variant=VARIANT, duration=DURATION):
    run_dir = paths(gate, variant, duration)["run_dir"]
    groups = WAVES.items() if variant == VARIANT else [("resumed", resume_seeds(gate, duration))]
    for w, seeds in groups:
        have = [s for s in seeds if done(tag(s, gate, variant, duration), run_dir)]
        stage1 = [s for s in seeds if s not in have
                  and os.path.exists(os.path.join(run_dir, f"u_{tag(s, gate, variant, duration)}_stage1.npy"))]
        print(f"{gate} {variant} {duration} us wave {w}: {len(have)}/{len(seeds)} done; in stage 2: {stage1}")


# ---------------------------------------------------------------- scoring

def _pulse_path(name, stage, gate=GATE, variant=VARIANT, duration=DURATION):
    """(u path, sidecar json) of a run or reference; stage 'final' or 'stage1'."""
    where = PULSE_DIR if name in REFERENCES_BY_GATE[gate] else paths(gate, variant, duration)["run_dir"]
    suffix = "" if stage == "final" else "_stage1"
    return os.path.join(where, f"u_{name}{suffix}.npy"), os.path.join(where, f"{name}.json")


def score_pulse(job):
    """One pulse -> (job, row, traces): closed metrics at N_C_SCORE, peaks, tails, truncation scan.
    job = (name, stage) for X, or (name, stage, gate[, variant[, duration]])."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from DGRAPE import est_gate as eg
    name, stage = job[:2]
    gate = job[2] if len(job) > 2 else GATE
    variant = job[3] if len(job) > 3 else VARIANT
    duration = job[4] if len(job) > 4 else DURATION
    u = np.load(_pulse_path(name, stage, gate, variant, duration)[0])
    m = eg.metrics(u, n_c=N_C_SCORE, gate=gate)
    T = len(m["t"])
    tail = slice(int(TAIL_FRAC * T), None)
    row = {"F1": m["F1"], "tier": eg.tier(m["F1"], gate), "max_active_fock": m["max_active_fock"]}
    for key, col in (("delta_qec", "dqec"), ("leakage", "L"), ("eta_avg", "eta")):
        x = np.asarray(m[key])
        row.update({f"{col}_mean": float(x.mean()), f"{col}_peak": float(x.max()),
                    f"{col}_peak_t": float(m["t"][int(x.argmax())]), f"{col}_tail": float(x[tail].mean()),
                    f"{col}_T": float(x[-1])})
    row["score"] = (row["dqec_mean"] + row["L_mean"] + row["eta_mean"]) / 3.0
    row["peak_score"] = (row["dqec_peak"] + row["L_peak"] + row["eta_peak"]) / 3.0
    if stage == "final":
        scan = eg.truncation_scan(u, TRUNC, gate=gate)
        row.update({f"F1_nc{n}": v for n, v in scan.items()})
        row["trunc_spread"] = abs(scan[28] - scan[24])
        if abs(scan[20] - scan[24]) > WALL_TOL:     # trained against the n_c 20 wall (seed 9), not parked
            row["tier"] = "wall"
    traces = {"t": m["t"], "dqec": m["delta_qec"], "L": m["leakage"], "eta": m["eta_avg"]}
    return job, row, traces


def _passes(r, peak=PEAK_MAX):
    checks = {"F1_ok": r["F1"] >= F1_MIN and r["tier"] != "parked",
              "trunc_ok": r["trunc_spread"] < TRUNC_TOL}
    for k in peak:
        checks[f"{k}_peak_ok"] = r[f"{k}_peak"] <= peak[k]
        checks[f"{k}_tail_ok"] = r[f"{k}_tail"] <= TAIL_MAX[k]
    return checks


def pick_table(runs, gate=GATE, duration=DURATION):
    """The rule applied to the final pulses; candidates ranked by rank_by(gate, duration), lowest first."""
    import pandas as pd
    fin = runs[runs.stage == "final"]
    key = rank_by(gate, duration)
    peak = peak_max(gate, duration)
    rows = []
    for _, r in fin.iterrows():
        checks = _passes(r, peak)
        row = {"name": r["name"], "kind": r["kind"], "seed": r["seed"], "F1": r["F1"], "score": r["score"]}
        if "peak_score" in r:
            row["peak_score"] = r["peak_score"]
        rows.append({**row, **checks, "n_fail": sum(not v for v in checks.values()),
                     "candidate": all(checks.values())})
    pick = pd.DataFrame(rows)
    cand = pick[(pick.kind == "run") & pick.candidate].sort_values(key)
    pick["rank"] = np.nan
    pick.loc[cand.index, "rank"] = np.arange(1, len(cand) + 1)
    current = CURRENT_BY_GATE[gate]
    if current is not None:
        cur = float(pick.loc[pick.name == current, "score"].iloc[0])
        pick["beats_current"] = (pick.score < cur) & pick.F1_ok      # a parked gate's ~0 score beats nothing
    return pick.sort_values(["rank", "n_fail", key], na_position="last").reset_index(drop=True)


def summarize(jobs=8, gate=GATE, variant=VARIANT, duration=DURATION):
    import pandas as pd
    P = paths(gate, variant, duration)
    present, infos = [], {}
    for s in seeds_of(gate, variant, duration=duration):
        if done(tag(s, gate, variant, duration), P["run_dir"]):
            present.append((tag(s, gate, variant, duration), "run", s))
    refs = REFERENCES_BY_GATE[gate] if variant == VARIANT and duration == DURATION else {}
    for name, s in refs.items():
        present.append((name, "reference", s))
    for name, kind, s in present:
        with open(_pulse_path(name, "final", gate, variant, duration)[1]) as fh:
            infos[name] = json.load(fh)
        assert infos[name].get("gate", "X") == gate, name
        if kind == "run":
            info = infos[name]
            assert info["variant"] == variant and info["seed"] == s and info["maxiter"] == MAXITER, name
            assert abs(info["duration_us"] - duration) < 1e-9, name
            if variant == VARIANT:
                assert info["init"] == "d2 smooth start", name
            else:                                   # resumed at stage 2 from the est_err stage-1 pre-image
                assert info["start_stage"] == 2, name
                assert os.path.samefile(info["init_x_path"], stage1_preimage(s, gate, duration)), name
    # the s43/s44 references kept only their final pulse; est_err2 runs have no stage-1 pulse of their own
    job_list = [(n, st, gate, variant, duration) for n, _, _ in present for st in ("stage1", "final")
                if os.path.exists(_pulse_path(n, st, gate, variant, duration)[0])]
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
        scored = {job[:2]: (row, tr) for job, row, tr in ex.map(score_pulse, job_list)}
    rows, traces = [], {}
    for name, kind, s in present:
        for st in ("stage1", "final"):
            if (name, st) not in scored:
                continue
            row, tr = scored[(name, st)]
            info = infos[name]
            k = 0 if st == "stage1" else -1
            rows.append({"name": name, "kind": kind, "seed": s, "stage": st, **row,
                         "frac_preimage_pinned": info["stages"][k]["frac_preimage_pinned"],
                         "nit": info["stages"][k]["nit"], "minutes": info["stages"][k]["seconds"] / 60})
            for key, v in tr.items():
                traces[f"{name}/{st}/{key}"] = np.asarray(v)
    runs = pd.DataFrame(rows).sort_values(["stage", "kind", "seed"]).reset_index(drop=True)
    runs.to_csv(P["runs_csv"], index=False)
    np.savez(P["traces_npz"], **traces)
    pick = pick_table(runs, gate, duration)
    pick.to_csv(P["pick_csv"], index=False)
    print(f"tables -> {P['runs_csv']}\n          {P['pick_csv']}\n          {P['traces_npz']}\n")
    fin = runs[runs.stage == "final"]
    n_run = int((fin.kind == "run").sum())
    n_cand = int(pick[pick.kind == "run"].candidate.sum())
    print(f"{gate} {variant} {duration} us: {n_run} runs scored; "
          f"parked {int(((fin.kind == 'run') & (fin.tier == 'parked')).sum())}; candidates {n_cand}\n")
    cols = ["name", "tier", "F1", "score", "peak_score", "dqec_mean", "L_mean", "eta_mean", "dqec_peak", "L_peak",
            "eta_peak", "dqec_tail", "L_tail", "eta_tail", "trunc_spread", "max_active_fock"]
    print(fin[cols].to_string(index=False, float_format="%.4f"))
    print()
    print(pick.to_string(index=False, float_format="%.4f"))


def open_score(top=3, gate=GATE, variant=VARIANT, duration=DURATION, extra_seeds=()):
    """Open-system score (FWM_tails.ipynb §3.2's quantities) of the top candidates and the current gate, plus the
    runs of extra_seeds (rank NaN unless they are candidates), for comparison."""
    import pandas as pd
    from DGRAPE import fwm_tail as ft
    from DGRAPE.code import process_targets
    P = paths(gate, variant, duration)
    pick = pd.read_csv(P["pick_csv"])
    current = CURRENT_BY_GATE[gate] if variant == VARIANT and duration == DURATION else None
    names = list(pick[pick["rank"] <= top].sort_values("rank").name)
    names += [n for n in (tag(s, gate, variant, duration) for s in extra_seeds) if n not in names]
    names += [current] if current else []
    _, T24, W24 = process_targets(gate, N_C_SCORE, 1)
    rows = []
    for name in names:
        u = np.load(_pulse_path(name, "final", gate, variant, duration)[0])
        t0 = time.time()
        R1 = ft.gate_output(u, N_C_SCORE, 1)
        F_alone = ft.fidelity_from_outputs(R1, T24, W24)
        rep = ft.ideal_repair_fidelity(R1, T24, W24, N_C_SCORE, 1)
        ph = ft.ideal_repair_fidelity(ft.gate_output(u, N_C_SCORE, 1, {"a"}), T24, W24, N_C_SCORE, 1)["phases"]
        prof = ft.loss_time_profile(ph, N_C_SCORE, gate=gate, pulse=u)
        p = prof["p"]
        F_all = ft.score(np.zeros((0, 2)), pulse=u, n_c=N_C_SCORE, gate=gate, dephasing=True,
                         heating=True)["F_pro"]
        rows.append({"name": name, "rank": pick.loc[pick.name == name, "rank"].iloc[0],
                     "F_pro_decay": F_alone, "F_pro_all": F_all, "F_repaired": rep["F"],
                     "repair_gain": rep["F"] - F_alone, "losses_per_gate": float(p.sum()),
                     "f_midgate_loss_repaired": float(np.sum(p * prof["f_repaired"]) / p.sum())})
        print(f"{name}: " + "  ".join(f"{k} {v:.5f}" for k, v in rows[-1].items()
                                      if isinstance(v, float)) + f"  ({time.time() - t0:.0f} s)", flush=True)
    pd.DataFrame(rows).to_csv(P["open_csv"], index=False)
    print(f"-> {P['open_csv']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["run", "status", "summarize", "open_score"])
    p.add_argument("--gate", choices=GATES, default=GATE)
    p.add_argument("--variant", choices=VARIANTS, default=VARIANT)
    p.add_argument("--duration-us", type=float, choices=DURATIONS, default=DURATION)
    p.add_argument("--wave", type=int, choices=sorted(WAVES), default=1, help="est_err only")
    p.add_argument("--jobs", type=int, default=25)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--extra-seeds", type=int, nargs="*", default=[], help="open_score: also score these runs")
    a = p.parse_args()
    if a.action == "run":
        run(a.wave, a.jobs, a.gate, a.variant, a.duration_us)
    elif a.action == "status":
        status(a.gate, a.variant, a.duration_us)
    elif a.action == "summarize":
        summarize(min(a.jobs, 8), a.gate, a.variant, a.duration_us)
    else:
        open_score(a.top, a.gate, a.variant, a.duration_us, a.extra_seeds)


if __name__ == "__main__":
    main()
