"""
Weight scan (the one-factor-at-a-time "OFAT" ladders) for the extra EsT cost
terms on the eigh pipeline (EST/grape_eigh.py): ballpark the weights of the
error-space infidelity `cerr` (w_err), of the Eq. A7 photon-number mismatch C5
(w_dn) and of the leakage out of the instantaneous error space C_L (w_leak)
before committing to the full runs.

    python EST/smoke_new_terms.py run --jobs 20  # train everything missing (resumable)
    python EST/smoke_new_terms.py smoke          # C_L smoke screen -> its ladder
    python EST/smoke_new_terms.py summarize      # score the ladders, apply the rule
    python EST/smoke_new_terms.py status         # what is done
    python EST/smoke_new_terms.py calibrate      # score existing pulses, no training
    python EST/smoke_new_terms.py commands       # print shell training commands

SINGLE PHASE. Every run here is `train_est_eigh.py --stages 1`: stage 1's
weights (w1,w2,w3,w4) = (1, 0.7, 7, 1) held for the whole run, no stage 2. The
objective that is reported is therefore the objective that was optimized. Stage
2 is what CLOSES the gate, so expect more parked cells than the two-stage scan
this replaced -- that is a property of the protocol and is reported, not fixed.

Budget: X gate, MAXITER = 1000 iterations in the one phase. Six seeds (0, 2, 4,
5, 6, 8), the ones the section 9 two-stage screen reached a gate on
(tables/est_multiseed_X.csv); whether they still do under one phase is exactly
what this scan measures. Each seed has its own freshly trained single-phase
control (`--variant est`, same seed, same budget).

MACHINE. Training is deterministic on one machine but not across machines, so
every run name and the table carry MACHINE_SUFFIX. The live scan ("_win") was
trained on the Windows PC, single-threaded per run (`run`). The first scan was
trained on the Mac (no suffix) and is frozen in
tables/est_newterms_smoke_X_6seeds_sp.csv; set MACHINE_SUFFIX = "" to rebuild
its table. The older two-stage tables (est_newterms_smoke_X_6seeds.csv,
est_newterms_smoke_X_seed6.csv) are frozen too; nothing here writes to them.

Grids. `cerr` is O(0.5) on the delivered X pulses, the same order as w2*c2 ~ 0.15,
so w_err spans {0.01, 0.1, 0.3, 0.5, 1, 3}. C5 is only ~1e-2: both code words
have <a> = 0, so a linear displacement shifts n0 and n1 equally and
Delta_nbar_L comes from the nonlinear terms alone. w_dn spans {1, 3, 6, 10, 15,
30}, i.e. w5*c5 ~ 0.01-0.3. Both are carried over from the two-stage scan.

C_L's ladder is NOT fixed by hand: it is set by a SMOKE SCREEN (`smoke`). C_L is
~0.25 on the controls and ~0.94 of C2 (C2 = C_L + C_mis exactly; see
grape_eigh.leakage_cost), so its useful range is not obvious in advance. The
screen trains SMOKE_LEAK (8 weights, 0.01-30) on SMOKE_SEEDS (0 and 4) and
applies a range rule fixed before it ran (`leak_range`):
  w_lo = the smallest weight whose mean L over the driving smoke seeds is at
         least LEAK_EFFECT below their controls' mean;
  w_hi = the smallest weight at which a smoke seed parks or has
         c1 > C1_TOL_FACTOR x its control's (condition (1) below); if none
         breaks, the largest smoke weight;
  ladder = LADDER_RUNGS weights from one rung below w_lo up to w_hi, from the
         {1, 3} x 10^k sequence (filled from {1, 2, 3, 5} x 10^k if too few fit).
The result is written to LEAK_RANGE_PATH, and the C_L ladder is read from it.
Smoke cells at ladder weights are reused, not retrained (same names).

Selection rule. Per term, the LARGEST weight that
  (1) reaches a gate -- progress = (F1 - F1_parked)/(1 - F1_parked) > 0.5 -- and
      keeps final c1 <= C1_TOL_FACTOR x the control's c1, and
  (2) improves its own target over the control:
        err  : F_err(T) = 1 - cerr higher
        dn   : mean c5 lower AND mean Delta_QEC lower
        leak : mean L lower
  (3) costs at most ET_DROP_TOL in F_ET against the control.
Condition (3) is the REVISED rule: on the two-stage scan the first two alone
picked w_err = 3 and w_dn = 100, both of which damage transparency. `summarize`
prints the pick with and without it. (1) must hold on EVERY seed, each against
its own control; (2) and (3) compare the seed MEAN against the controls' mean.
C_L is part of C2, so (3) is expected to bind rarely for leak; (1) is its guard.
Single budget: a ballpark, not an optimum.
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

from EST import diagnostics, grape_eigh, kitten_code
from EST.train_est_eigh import LOG_DIR, PULSE_DIR

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_DIR = os.path.join(REPO_ROOT, "tables")
RUN_DIR = os.path.join(os.path.expanduser("~"), "est_runs", "newterms_win")

GATE, MAXITER, N_C = "X", 1000, 20
N_STAGES = 1           # single phase at stage 1's weights; see the docstring
MACHINE_SUFFIX = "_win"
# The section 9 X seeds with progress > 0.5 (tables/est_multiseed_X.csv) under
# the TWO-stage screen. Whether one phase still reaches a gate on them is what
# this scan measures, so a parked cell here is data, not a bad seed.
SEEDS = (0, 2, 4, 5, 6, 8)
GRIDS = {"err": ("est_err", "--w-err", (0.01, 0.1, 0.3, 0.5, 1.0, 3.0)),
         "dn": ("est_dn", "--w-dn", (1.0, 3.0, 6.0, 10.0, 15.0, 30.0))}
# C_L: the smoke screen, and its ladder once `smoke` has fixed it.
LEAK_VARIANT, LEAK_FLAG = "est_lk", "--w-leak"
SMOKE_LEAK = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
SMOKE_SEEDS = (0, 4)
LEAK_EFFECT = 0.01     # range rule: "visibly lowers mean L"
LADDER_RUNGS = 6
LEAK_RANGE_PATH = os.path.join(TABLE_DIR,
                               f"est_newterms_leak_range_X{MACHINE_SUFFIX}.json")
C1_TOL_FACTOR = 3.0
ET_DROP_TOL = 0.05     # revised rule (3); see module docstring
# Three tiers, not two. `progress > 0.5` only says a pulse escaped the parked
# trap at F1 = 1/3; F1_MIN says it is a gate. Single-phase runs are expected to
# land BETWEEN them. Reporting only -- the SELECTION rule stays control-relative
# (c1 <= C1_TOL_FACTOR * the control's), so a uniform stall leaves it well-posed.
F1_MIN = 0.999
STAGE1_WEIGHTS = (1.0, 0.7, 7.0, 1.0)
BATCH = 6              # concurrent training jobs per `wait` in `commands`
CSV_PATH = os.path.join(TABLE_DIR, f"est_newterms_smoke_X_6seeds_sp{MACHINE_SUFFIX}.csv")
MEAN_COLS = ["F1", "F_ET", "F_err_T", "c5", "cleak", "delta_qec_mean", "L_mean", "L_T",
             "eta0_mean", "eta_avg_mean"]


def leak_grid():
    """The C_L ladder fixed by `smoke`, or () before the smoke screen has run."""
    if not os.path.exists(LEAK_RANGE_PATH):
        return ()
    with open(LEAK_RANGE_PATH) as fh:
        return tuple(float(w) for w in json.load(fh)["ladder"])


def grids():
    """GRIDS plus the C_L ladder once it is known."""
    g = dict(GRIDS)
    ladder = leak_grid()
    if ladder:
        g["leak"] = (LEAK_VARIANT, LEAK_FLAG, ladder)
    return g


def _variant_flag(term):
    if term == "leak":
        return LEAK_VARIANT, LEAK_FLAG
    return GRIDS[term][:2]


def _tag(w, seed):
    return f"sp{MAXITER}_seed{seed}_w{w:g}{MACHINE_SUFFIX}"


def _ctrl_tag(seed):
    return f"sp{MAXITER}_ctrl_seed{seed}{MACHINE_SUFFIX}"


def _paths(variant, tag):
    name = f"{variant}_{tag}"
    return (os.path.join(PULSE_DIR, f"x_{GATE}_{name}.npy"),
            os.path.join(LOG_DIR, f"est_{GATE}_{name}.json"))


def _names(term, w, seed):
    """(x path, json path) for one scan cell.

    No legacy fallback. The old scan let seed 6 reuse the original untagged
    `*_smoke500_w<w>` files; those are TWO-stage runs, so reusing them here
    would silently mix protocols.
    """
    return _paths(_variant_flag(term)[0], _tag(w, seed))


def _ctrl_names(seed):
    """(x path, json path) for a seed's freshly trained single-phase control."""
    return _paths("est", _ctrl_tag(seed))


def _require_phases(jpath):
    """Refuse to score a pulse trained with a different number of phases.

    pulses/est/ holds 600+ files and the two-stage `*_seed<s>` / `*_smoke500_*`
    artifacts all look plausible next to these. `n_stages` is absent from JSONs
    written before the flag existed, so fall back to counting stages.
    """
    if not os.path.exists(jpath):
        return {}
    with open(jpath) as fh:
        info = json.load(fh)
    n = info.get("n_stages", len(info.get("stages", [])))
    if n != N_STAGES:
        raise SystemExit(
            f"{jpath} was trained with {n} stage(s); this scan is the "
            f"{N_STAGES}-phase campaign. Retrain it with --stages {N_STAGES}.")
    return info


def score_x(x, gate=GATE):
    """Extended report at n_c=20 plus Eqs. 6-8 on the physical pulse."""
    x = np.asarray(x, dtype=np.float64)
    _, report, constrain_np = grape_eigh.build_gate_objective(
        gate, len(x), trunc_list=[N_C], extra_terms=True)
    t = report(x.ravel())[N_C]
    u = constrain_np(x.ravel())
    r = diagnostics.analyze(u, gate=gate, n_c=N_C)
    parked = kitten_code.parked_fidelity(gate)
    T = len(r["leakage"])
    tail = slice(int(0.9 * T), None)
    return {
        "c1": t["c1"], "F1": 1 - t["c1"],
        "progress": (1 - t["c1"] - parked) / (1 - parked),
        "F_ET": 1 - t["c2"], "c3": t["c3"],
        "cerr": t["cerr"], "F_err_T": 1 - t["cerr"],
        "c5": t["c5"], "c6": t["c6"], "cleak": t["cleak"],
        "delta_qec_mean": float(r["delta_qec"].mean()),
        "delta_qec_tail": float(r["delta_qec"][tail].mean()),
        "L_mean": float(r["leakage"].mean()),
        "L_T": float(r["leakage"][-1]),
        "L_tail": float(r["leakage"][tail].mean()),
        "L_max": float(r["leakage"].max()),
        "eta0_mean": float(r["eta_0L"].mean()),
        "eta0_tail": float(r["eta_0L"][tail].mean()),
        # Eq. 8 is state-dependent; eta_0L is one cardinal, eta_avg the mean over
        # all six. est_optimization.ipynb section 2 reports the six-cardinal mean.
        "eta_avg_mean": float(r["eta_avg"].mean()),
        "eta_avg_tail": float(r["eta_avg"][tail].mean()),
        "max_active_fock": int(r["max_active_fock"]),
        "max_abs_preimage": float(np.abs(x).max()),
    }


# ---------------------------------------------------------------- scoring cache

def _score_one(xpath):
    """score_x for one pre-image, cached on disk by its mtime."""
    os.makedirs(os.path.join(RUN_DIR, "scores"), exist_ok=True)
    cache = os.path.join(RUN_DIR, "scores",
                         os.path.basename(xpath).replace(".npy", ".json"))
    mtime = os.path.getmtime(xpath)
    if os.path.exists(cache):
        with open(cache) as fh:
            c = json.load(fh)
        if c.get("mtime") == mtime:
            return xpath, c["score"]
    s = score_x(np.load(xpath))
    with open(cache, "w") as fh:
        json.dump({"mtime": mtime, "score": s}, fh)
    return xpath, s


def score_many(xpaths, jobs=8):
    """{x path: score} for every existing pre-image, scored in parallel."""
    xpaths = [p for p in dict.fromkeys(xpaths) if os.path.exists(p)]
    if not xpaths:
        return {}
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
        return dict(ex.map(_score_one, xpaths))


# ---------------------------------------------------------------- smoke screen

def _ladder(w_lo, w_hi):
    """
    LADDER_RUNGS weights from one rung below w_lo up to w_hi (see docstring).

    Both ends are kept. Among the candidate subsets, the pick is the one whose
    widest gap in log(w) is smallest (then the most even), preferring {1, 3} x
    10^k rungs over the fill values.
    """
    from itertools import combinations

    def seq(mants):
        return sorted({round(m * 10.0 ** k, 12) for k in range(-4, 4) for m in mants})

    base = seq((1, 3))
    below = [w for w in base if w < w_lo]
    bottom = below[-1] if below else w_lo
    inside = lambda ws: [w for w in ws if bottom - 1e-12 <= w <= w_hi + 1e-12]
    cand = []
    for mants in ((1, 3), (1, 2, 3, 5), (1, 1.5, 2, 3, 5, 7)):
        cand = inside(seq(mants))
        if len(cand) >= LADDER_RUNGS:
            break
    if len(cand) <= LADDER_RUNGS:
        return cand                # fewer than LADDER_RUNGS fit; reported as such
    is_base = set(inside(base))
    best, best_key = None, None
    for mid in combinations(cand[1:-1], LADDER_RUNGS - 2):
        pick = [cand[0], *mid, cand[-1]]
        gaps = np.diff(np.log10(pick))
        key = (round(gaps.max(), 9), sum(w not in is_base for w in pick),
               round(float(np.sum(gaps ** 2)), 9))
        if best_key is None or key < best_key:
            best, best_key = pick, key
    return best


def leak_range(scores):
    """Apply the range rule to the smoke screen. Returns the record, or None."""
    rows, ctrl = [], {}
    for seed in SMOKE_SEEDS:
        cx = _ctrl_names(seed)[0]
        if cx not in scores:
            return None
        ctrl[seed] = scores[cx]
    for w in SMOKE_LEAK:
        for seed in SMOKE_SEEDS:
            x = _names("leak", w, seed)[0]
            if x not in scores:
                return None
            s = scores[x]
            rows.append({"weight": w, "seed": seed, "F1": s["F1"],
                         "progress": s["progress"], "c1": s["c1"],
                         "ctrl_c1": ctrl[seed]["c1"], "L_mean": s["L_mean"],
                         "ctrl_L_mean": ctrl[seed]["L_mean"], "L_T": s["L_T"],
                         "L_max": s["L_max"], "F_ET": s["F_ET"],
                         "ctrl_F_ET": ctrl[seed]["F_ET"], "cleak": s["cleak"],
                         "delta_qec_mean": s["delta_qec_mean"],
                         "eta_avg_mean": s["eta_avg_mean"],
                         "parked": s["progress"] <= 0.5,
                         "c1_bad": s["c1"] > C1_TOL_FACTOR * ctrl[seed]["c1"]})
    ctrl_L = float(np.mean([ctrl[s]["L_mean"] for s in SMOKE_SEEDS]))
    w_lo = w_hi = None
    per_w = []
    for w in SMOKE_LEAK:
        d = [r for r in rows if r["weight"] == w]
        drove = [r for r in d if not r["parked"]]
        drop = (ctrl_L - float(np.mean([r["L_mean"] for r in drove]))) if drove else np.nan
        breaks = any(r["parked"] or r["c1_bad"] for r in d)
        per_w.append({"weight": w, "L_drop": drop, "breaks": breaks,
                      "n_parked": sum(r["parked"] for r in d)})
        if w_lo is None and drove and drop >= LEAK_EFFECT:
            w_lo = w
        if w_hi is None and breaks:
            w_hi = w
    notes = []
    if w_hi is None:
        w_hi = SMOKE_LEAK[-1]
        notes.append("no smoke weight breaks; ladder top = largest smoke weight")
    if w_lo is None:
        w_lo = SMOKE_LEAK[0]
        notes.append(f"no smoke weight lowers mean L by {LEAK_EFFECT}; "
                     "ladder starts at the smallest smoke weight")
    if w_lo > w_hi:
        notes.append("C_L breaks the gate before it visibly lowers L (w_lo > w_hi)")
    ladder = _ladder(min(w_lo, w_hi), w_hi)
    if len(ladder) < LADDER_RUNGS:
        notes.append(f"only {len(ladder)} rungs fit between the bounds")
    return {"smoke_seeds": list(SMOKE_SEEDS), "smoke_weights": list(SMOKE_LEAK),
            "ctrl_L_mean": ctrl_L, "w_lo": w_lo, "w_hi": w_hi, "ladder": ladder,
            "per_weight": per_w, "rows": rows, "notes": notes}


def smoke(jobs=8, write=True):
    """Score the smoke screen, apply the range rule, write LEAK_RANGE_PATH."""
    paths = ([_ctrl_names(s)[0] for s in SMOKE_SEEDS]
             + [_names("leak", w, s)[0] for w in SMOKE_LEAK for s in SMOKE_SEEDS])
    rec = leak_range(score_many(paths, jobs))
    if rec is None:
        missing = [p for p in paths if not os.path.exists(p)]
        print(f"smoke screen incomplete: {len(missing)} of {len(paths)} runs missing")
        return None
    print(f"C_L smoke screen, seeds {SMOKE_SEEDS}, {MAXITER} it; control mean L "
          f"{rec['ctrl_L_mean']:.4f}")
    print(f"{'w_L':>6s} {'seed':>4s} {'F1':>7s} {'prog':>5s} {'c1/ctrl':>7s} "
          f"{'L_mean':>7s} {'L(T)':>6s} {'L_max':>6s} {'F_ET':>6s} {'dQEC':>6s}")
    for r in rec["rows"]:
        print(f"{r['weight']:6g} {r['seed']:4d} {r['F1']:7.4f} {r['progress']:5.2f} "
              f"{r['c1'] / r['ctrl_c1']:7.2f} {r['L_mean']:7.4f} {r['L_T']:6.3f} "
              f"{r['L_max']:6.3f} {r['F_ET']:6.3f} {r['delta_qec_mean']:6.3f}"
              + ("  PARKED" if r["parked"] else "") + ("  c1>3x" if r["c1_bad"] else ""))
    print(f"\nw_lo = {rec['w_lo']:g}  w_hi = {rec['w_hi']:g}  ->  ladder "
          f"{tuple(rec['ladder'])}")
    for n in rec["notes"]:
        print(f"NOTE: {n}")
    if write:
        if os.path.exists(LEAK_RANGE_PATH):
            with open(LEAK_RANGE_PATH) as fh:
                old = json.load(fh)["ladder"]
            if old != rec["ladder"]:
                raise SystemExit(f"{LEAK_RANGE_PATH} already fixes the ladder {old}; "
                                 f"refusing to change it to {rec['ladder']}")
        with open(LEAK_RANGE_PATH, "w") as fh:
            json.dump(rec, fh, indent=1, default=float)
        print(f"range -> {LEAK_RANGE_PATH}")
    return rec


# ---------------------------------------------------------------- launcher

def _train_args(variant, flag, w, seed, tag):
    args = ["EST/train_est_eigh.py", "--gate", GATE, "--stages", str(N_STAGES),
            "--maxiter", str(MAXITER), "--variant", variant, "--seed", str(seed),
            "--tag", tag]
    return args + ([flag, f"{w:g}"] if flag else [])


def _todo():
    """(name, args) of every missing run, in queue order.

    Order: the smoke screen's two controls and cells first (the C_L ladder waits
    on them), then the other controls, the C_err and C5 ladders, and the C_L
    ladder once `smoke` has fixed it.
    """
    out = []

    def add(variant, flag, w, seed, tag):
        x, _ = _paths(variant, tag)
        if not os.path.exists(x):
            out.append((f"{variant}_{tag}", _train_args(variant, flag, w, seed, tag)))

    for s in SMOKE_SEEDS:
        add("est", None, None, s, _ctrl_tag(s))
    for w in SMOKE_LEAK:
        for s in SMOKE_SEEDS:
            add(LEAK_VARIANT, LEAK_FLAG, w, s, _tag(w, s))
    for s in SEEDS:
        add("est", None, None, s, _ctrl_tag(s))
    for term, (variant, flag, grid) in grids().items():
        for w in grid:
            for s in SEEDS:
                add(variant, flag, w, s, _tag(w, s))
    seen, uniq = set(), []
    for name, args in out:
        if name not in seen:
            seen.add(name)
            uniq.append((name, args))
    return uniq


def run(jobs):
    """Train every missing run, `jobs` single-threaded processes at a time.

    Resumable: a run whose pre-image exists is skipped. When the smoke screen is
    complete and no ladder is fixed yet, `smoke` is applied here (the rule is
    fixed in advance, so this needs no judgement) and the C_L ladder is queued.
    """
    if os.name == "nt":
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no sleep while this process lives.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    os.makedirs(RUN_DIR, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    running, failed = {}, set()
    t_start, n_done = time.time(), 0
    while True:
        if not leak_grid():
            smoke_names = {f"{LEAK_VARIANT}_{_tag(w, s)}" for w in SMOKE_LEAK
                           for s in SMOKE_SEEDS} | {f"est_{_ctrl_tag(s)}" for s in SMOKE_SEEDS}
            pending = [n for n, _ in _todo() if n in smoke_names] + \
                      [n for n in running if n in smoke_names]
            if not pending:
                print(f"[{time.strftime('%H:%M:%S')}] smoke screen complete; "
                      "applying the range rule", flush=True)
                smoke(jobs=min(8, max(1, jobs - len(running))))
        queue = [(n, a) for n, a in _todo() if n not in running and n not in failed]
        while queue and len(running) < jobs:
            name, args = queue.pop(0)
            log = open(os.path.join(RUN_DIR, f"{name}.log"), "w")
            p = subprocess.Popen([sys.executable] + args, cwd=REPO_ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
            running[name] = (p, log, time.time())
            print(f"[{time.strftime('%H:%M:%S')}] start {name}  "
                  f"(running {len(running)}, queued {len(queue)})", flush=True)
        for name, (p, log, t0) in list(running.items()):
            if p.poll() is None:
                continue
            log.close()
            del running[name]
            n_done += 1
            ok = p.returncode == 0
            if not ok:
                failed.add(name)
            print(f"[{time.strftime('%H:%M:%S')}] {'ok' if ok else f'FAILED rc={p.returncode}'} "
                  f"{name}  {(time.time() - t0) / 60:.1f} min  (done {n_done}, "
                  f"running {len(running)})", flush=True)
        if not running and not [n for n, _ in _todo() if n not in failed] and leak_grid():
            break
        if not running and not queue and not leak_grid():
            print("nothing left to run but no C_L ladder is fixed; stopping", flush=True)
            break
        time.sleep(5)
    print(f"campaign finished in {(time.time() - t_start) / 3600:.2f} h"
          + (f"; FAILED: {sorted(failed)}" if failed else ""))


def status():
    todo = _todo()
    print(f"missing runs: {len(todo)}   C_L ladder: {leak_grid() or 'not fixed yet'}")
    for name, _ in todo[:10]:
        print(f"  {name}")


def commands():
    """Shell training commands for every missing run (POSIX; `run` is the launcher)."""
    todo = _todo()
    for i, (name, args) in enumerate(todo, start=1):
        print("OMP_NUM_THREADS=1 python3 " + " ".join(args)
              + f" > logs/sp_{name}.log 2>&1 &")
        if i % BATCH == 0 or i == len(todo):
            print("wait")
    print(f"# {len(todo)} runs to train", file=sys.stderr)


def calibrate():
    # Reference pulses only, for ballparking term sizes. All TWO-stage; they are
    # not controls for this scan (see the docstring) and nothing below uses them.
    pulses = [("best2000 stage 1", "x_X_est_best2000_stage1.npy"),
              ("best2000 final", "x_X_est_best2000.npy"),
              ("seed-0 JAX est", "x_X_est.npy"),
              ("screen seed 6 (2-stage)", "x_X_est_seed6.npy")]
    print(f"{'pulse':<26s} {'w2*c2':>8s} {'cerr':>8s} {'c5':>9s} {'c6':>9s} "
          f"{'5e-6*c6':>8s} {'F_err(T)':>9s} {'cleak':>7s}")
    for label, fn in pulses:
        x = np.load(os.path.join(PULSE_DIR, fn))
        _, report, _ = grape_eigh.build_gate_objective(
            GATE, len(x), trunc_list=[N_C], extra_terms=True)
        t = report(x.ravel())[N_C]
        print(f"{label:<26s} {0.7 * t['c2']:8.4f} {t['cerr']:8.4f} {t['c5']:9.2e} "
              f"{t['c6']:9.1f} {5e-6 * t['c6']:8.4f} {1 - t['cerr']:9.4f} "
              f"{t['cleak']:7.4f}")
    print("\nsanity gate: F_err(T) on seed-0 JAX est must read ~0.589 "
          "(limitation 3's error-basis overlap).")


# ---------------------------------------------------------------- rule

def _improves(term, m, ref):
    """Condition (2): does mean row `m` improve the term's target over `ref`?"""
    if term == "err":
        return m["F_err_T"] > ref["F_err_T"]
    if term == "dn":
        return m["c5"] < ref["c5"] and m["delta_qec_mean"] < ref["delta_qec_mean"]
    return m["L_mean"] < ref["L_mean"]


def apply_rule(df):
    """
    Seed-mean selection over a `summarize` table (or its CSV).

    Returns (per-weight table, pick, revised pick), the picks as {term: weight or
    None}. Condition (1) must hold on every seed in SEEDS; (2) and (3) compare the
    run's seed mean against the controls' seed mean. The terms are read from the
    table, so a frozen table without a C_L ladder still works.
    """
    import pandas as pd

    ctrl = df[df.term == "control"]
    cols = [c for c in MEAN_COLS if c in df.columns]
    ctrl_mean = ctrl[cols].mean()
    rows = [{"term": "control", "weight": 0.0, "n_seeds": len(ctrl),
             "closed_seeds": int(ctrl.gate_closed.astype(bool).sum()),
             **ctrl_mean.to_dict()}]
    chosen, chosen_rev = {}, {}
    for term in [t for t in ("err", "dn", "leak") if (df.term == t).any()]:
        best = best_rev = None
        for w in np.sort(df[df.term == term].weight.unique()):
            d = df[(df.term == term) & (df.weight == w)]
            m = d[cols].mean()
            gate_ok = len(d) == len(SEEDS) and bool(d.gate_ok.astype(bool).all())
            improves = _improves(term, m, ctrl_mean)
            et_ok = m.F_ET >= ctrl_mean.F_ET - ET_DROP_TOL
            rows.append({"term": term, "weight": w, "n_seeds": len(d), **m.to_dict(),
                         "gate_ok_seeds": int(d.gate_ok.astype(bool).sum()),
                         "closed_seeds": int(d.gate_closed.astype(bool).sum()),
                         "improves_seeds": int(d.improves.astype(bool).sum()),
                         "et_ok_seeds": int(d.et_ok.astype(bool).sum()),
                         "gate_ok": gate_ok, "improves": bool(improves),
                         "et_ok": bool(et_ok)})
            if gate_ok and improves:
                best = float(w)                # ascending: keep the largest
                if et_ok:
                    best_rev = float(w)
        chosen[term] = best
        chosen_rev[term] = best_rev
    return pd.DataFrame(rows), chosen, chosen_rev


def summarize(jobs=8):
    import pandas as pd

    g = grids()
    if "leak" not in g:
        print("WARNING: the C_L ladder is not fixed yet (run `smoke`); "
              "summarizing C_err and C5 only")
    paths = [_ctrl_names(s)[0] for s in SEEDS]
    paths += [_names(t, w, s)[0] for t, (_, _, grid) in g.items()
              for w in grid for s in SEEDS]
    scores = score_many(paths, jobs)

    rows, missing = [], []
    for seed in SEEDS:
        ctrl_x, ctrl_j = _ctrl_names(seed)
        if ctrl_x not in scores:
            missing.append(ctrl_x)
            print(f"missing control {ctrl_x}; seed {seed} skipped entirely "
                  "(its cells have nothing to be scored against)")
            continue
        _require_phases(ctrl_j)
        ctrl = scores[ctrl_x]
        rows.append({"term": "control", "weight": 0.0, "seed": seed, **ctrl,
                     "gate_ok": ctrl["progress"] > 0.5,
                     "gate_closed": ctrl["F1"] >= F1_MIN})
        for term, (_, _, grid) in g.items():
            for w in grid:
                xpath, jpath = _names(term, w, seed)
                if xpath not in scores:
                    missing.append(xpath)
                    continue
                info = _require_phases(jpath)
                s = dict(scores[xpath])
                s["gate_closed"] = s["F1"] >= F1_MIN
                s["nit"] = "+".join(str(st["nit"]) for st in info["stages"])
                # Condition (1) is per seed, against that seed's own control.
                s["gate_ok"] = (s["progress"] > 0.5
                                and s["c1"] <= C1_TOL_FACTOR * ctrl["c1"])
                s["improves"] = _improves(term, s, ctrl)
                s["et_ok"] = s["F_ET"] >= ctrl["F_ET"] - ET_DROP_TOL
                rows.append({"term": term, "weight": w, "seed": seed, **s})
    for m in missing:
        print(f"missing {m}; skipped")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"table -> {CSV_PATH}\n")

    parked = df[(df.term == "control") & ~df.gate_ok.astype(bool)]
    if len(parked):
        print("WARNING: control seeds PARKED at F1 = 1/3:", parked.seed.tolist())
    open_gate = df[(df.term == "control") & df.gate_ok.astype(bool)
                   & ~df.gate_closed.astype(bool)]
    if len(open_gate):
        print(f"WARNING: control seeds that drove but did NOT close the gate "
              f"(F1 < {F1_MIN}): "
              + ", ".join(f"{int(r.seed)}:{r.F1:.4f}" for r in open_gate.itertuples()))
        print("  Single-phase holds w2=0.7 and w3=7 to the end; stage 2 is what "
              "finishes the gate. Selection below is control-relative and "
              "stays well-posed; the absolute claim 'these are gates' does not.")

    agg, chosen, chosen_rev = apply_rule(df)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("seed means (the *_seeds columns count seeds passing each condition):")
        print(agg.to_string(index=False, float_format="%.4g"))
    fmt = lambda w: f"{w:g}" if w is not None else "NONE qualifies -- do not promote"
    print()
    for term in chosen:
        print(f"selected {term}: pre-registered rule {fmt(chosen[term])}   "
              f"revised rule (F_ET drop <= {ET_DROP_TOL}) {fmt(chosen_rev[term])}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["run", "smoke", "summarize", "status",
                                      "calibrate", "commands"])
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()
    if a.action == "run":
        run(a.jobs)
    elif a.action == "smoke":
        smoke(jobs=a.jobs)
    elif a.action == "summarize":
        summarize(jobs=a.jobs)
    else:
        {"status": status, "calibrate": calibrate, "commands": commands}[a.action]()


if __name__ == "__main__":
    main()
