"""
Weight scan for the extra EsT cost terms on the eigh pipeline (EST/grape_eigh.py):
ballpark the weights of the error-space infidelity `cerr` (w_err) and of the
Eq. A7 photon-number mismatch C5 (w_dn) before committing to the full runs.

    python EST/smoke_new_terms.py calibrate      # score existing pulses, no training
    python EST/smoke_new_terms.py commands       # print the training commands
    python EST/smoke_new_terms.py summarize      # score the scan runs, apply the rule

SINGLE PHASE. Every run here is `train_est_eigh.py --stages 1`: stage 1's
weights (w1,w2,w3,w4) = (1, 0.7, 7, 1) held for the whole run, no stage 2. The
objective that is reported is therefore the objective that was optimized. Stage
2 is what CLOSES the gate, so expect more parked cells than the two-stage scan
this replaced -- that is a property of the protocol and is reported, not fixed.

Budget: X gate, MAXITER = 1000 iterations in the one phase, which is
compute-matched to the two-stage scan's 500 it/stage x 2. Six seeds (0, 2, 4, 5,
6, 8), the ones the section 9 two-stage screen reached a gate on
(tables/est_multiseed_X.csv); whether they still do under one phase is exactly
what this scan measures.

CONTROLS ARE TRAINED HERE. The old scan reused each seed's section 9 screen
pulse `x_X_est_seed<s>.npy` as its no-new-terms control at no cost. Those are
TWO-STAGE pulses and are not comparable to anything below, so `commands` now
emits six fresh single-phase control runs (`--variant est`, same seed, same
budget) and `summarize` reads those. 72 weight cells + 6 controls = 78 runs.

The two-stage tables this replaces stay frozen on disk:
tables/est_newterms_smoke_X_6seeds.csv (six seeds) and
tables/est_newterms_smoke_X_seed6.csv (the original seed-6-only scan). Nothing
here writes to either, and the seed-6 untagged-file fallback they relied on is
gone -- under one phase it would have mixed two-stage pulses into the scan.

Grids. `cerr` is O(0.5) on the delivered X pulses, the same order as w2*c2 ~ 0.15,
so w_err spans {0.01, 0.1, 0.3, 0.5, 1, 3}. C5 is only ~1e-2 (7.0e-3 on
u_X_est_best2000, 9.6e-3 on its stage-1 pulse): both code words have <a> = 0, so
a linear displacement shifts n0 and n1 equally and Delta_nbar_L comes from the
nonlinear terms alone. w_dn spans {1, 3, 6, 10, 15, 30}, i.e. w5*c5 ~ 0.01-0.3.
Both grids are carried over unchanged from the two-stage scan so the two are
comparable cell for cell.

Selection rule (unchanged in logic; only the control source moved). Per term, the
LARGEST weight that
  (1) reaches a gate -- progress = (F1 - F1_parked)/(1 - F1_parked) > 0.5 -- and
      keeps final c1 <= C1_TOL_FACTOR x the control's c1, and
  (2) improves its own target over the control:
        err : F_err(T) = 1 - cerr higher
        dn  : mean c5 lower AND mean Delta_QEC lower, and
  (3) costs at most ET_DROP_TOL in F_ET against the control.
Condition (3) is the REVISED rule: on the two-stage scan the first two alone
picked w_err = 3 and w_dn = 100, both of which damage transparency. `summarize`
prints the pick with and without it. (1) must hold on EVERY seed, each against
its own control; (2) and (3) compare the seed MEAN against the controls' mean.
Single budget: a ballpark, not an optimum.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from EST import diagnostics, grape_eigh, kitten_code
from EST.train_est_eigh import LOG_DIR, PULSE_DIR

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_DIR = os.path.join(REPO_ROOT, "tables")

GATE, MAXITER, N_C = "X", 1000, 20
N_STAGES = 1           # single phase at stage 1's weights; see the docstring
# The section 9 X seeds with progress > 0.5 (tables/est_multiseed_X.csv) under
# the TWO-stage screen. Whether one phase still reaches a gate on them is what
# this scan measures, so a parked cell here is data, not a bad seed.
SEEDS = (0, 2, 4, 5, 6, 8)
GRIDS = {"err": ("est_err", "--w-err", (0.01, 0.1, 0.3, 0.5, 1.0, 3.0)),
         "dn": ("est_dn", "--w-dn", (1.0, 3.0, 6.0, 10.0, 15.0, 30.0))}
C1_TOL_FACTOR = 3.0
ET_DROP_TOL = 0.05     # revised rule (3); see module docstring
# Three tiers, not two. `progress > 0.5` only says a pulse escaped the parked
# trap at F1 = 1/3; F1_MIN says it is a gate. Single-phase runs are expected to
# land BETWEEN them. Reporting only -- the SELECTION rule stays control-relative
# (c1 <= C1_TOL_FACTOR * the control's), so a uniform stall leaves it well-posed.
F1_MIN = 0.999
STAGE1_WEIGHTS = (1.0, 0.7, 7.0, 1.0)
BATCH = 6              # concurrent training jobs per `wait` in `commands`
CSV_PATH = os.path.join(TABLE_DIR, "est_newterms_smoke_X_6seeds_sp.csv")
MEAN_COLS = ["F1", "F_ET", "F_err_T", "c5", "delta_qec_mean", "L_mean", "L_T",
             "eta0_mean", "eta_avg_mean"]


def _tag(w, seed):
    return f"sp{MAXITER}_seed{seed}_w{w:g}"


def _ctrl_tag(seed):
    return f"sp{MAXITER}_ctrl_seed{seed}"


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
    return _paths(GRIDS[term][0], _tag(w, seed))


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
        "c5": t["c5"], "c6": t["c6"],
        "delta_qec_mean": float(r["delta_qec"].mean()),
        "delta_qec_tail": float(r["delta_qec"][tail].mean()),
        "L_mean": float(r["leakage"].mean()),
        "L_T": float(r["leakage"][-1]),
        "L_tail": float(r["leakage"][tail].mean()),
        "eta0_mean": float(r["eta_0L"].mean()),
        "eta0_tail": float(r["eta_0L"][tail].mean()),
        # Eq. 8 is state-dependent; eta_0L is one cardinal, eta_avg the mean over
        # all six. est_optimization.ipynb section 1 reports the six-cardinal mean.
        "eta_avg_mean": float(r["eta_avg"].mean()),
        "eta_avg_tail": float(r["eta_avg"][tail].mean()),
        "max_active_fock": int(r["max_active_fock"]),
        "max_abs_preimage": float(np.abs(x).max()),
    }


def calibrate():
    # Reference pulses only, for ballparking term sizes. All TWO-stage; they are
    # not controls for this scan (see the docstring) and nothing below uses them.
    pulses = [("best2000 stage 1", "x_X_est_best2000_stage1.npy"),
              ("best2000 final", "x_X_est_best2000.npy"),
              ("seed-0 JAX est", "x_X_est.npy"),
              ("screen seed 6 (2-stage)", "x_X_est_seed6.npy")]
    print(f"{'pulse':<26s} {'w2*c2':>8s} {'cerr':>8s} {'c5':>9s} {'c6':>9s} "
          f"{'5e-6*c6':>8s} {'F_err(T)':>9s}")
    for label, fn in pulses:
        x = np.load(os.path.join(PULSE_DIR, fn))
        _, report, _ = grape_eigh.build_gate_objective(
            GATE, len(x), trunc_list=[N_C], extra_terms=True)
        t = report(x.ravel())[N_C]
        print(f"{label:<26s} {0.7 * t['c2']:8.4f} {t['cerr']:8.4f} {t['c5']:9.2e} "
              f"{t['c6']:9.1f} {5e-6 * t['c6']:8.4f} {1 - t['cerr']:9.4f}")
    print("\nsanity gate: F_err(T) on seed-0 JAX est must read ~0.589 "
          "(limitation 3's error-basis overlap).")


def commands():
    """Every run still missing, controls first. All single phase (--stages 1)."""
    base = (f"OMP_NUM_THREADS=1 python3 EST/train_est_eigh.py --gate {GATE} "
            f"--stages {N_STAGES} --maxiter {MAXITER}")
    todo = []
    # Controls first: summarize() needs a seed's control to score its cells.
    for seed in SEEDS:
        if os.path.exists(_ctrl_names(seed)[0]):
            continue
        todo.append(f"{base} --variant est --seed {seed} "
                    f"--tag {_ctrl_tag(seed)} "
                    f"> logs/sp_ctrl_seed{seed}.log 2>&1 &")
    for seed in SEEDS:
        for term, (variant, flag, grid) in GRIDS.items():
            for w in grid:
                if os.path.exists(_names(term, w, seed)[0]):
                    continue
                todo.append(f"{base} --variant {variant} {flag} {w:g} "
                            f"--seed {seed} --tag {_tag(w, seed)} "
                            f"> logs/sp_{variant}_seed{seed}_w{w:g}.log 2>&1 &")
    for i, cmd in enumerate(todo, start=1):
        print(cmd)
        if i % BATCH == 0 or i == len(todo):
            print("wait")
    print(f"# {len(todo)} runs to train", file=sys.stderr)


def apply_rule(df):
    """
    Seed-mean selection over a `summarize` table (or its CSV).

    Returns (per-weight table, pick, revised pick), the picks as {term: weight or
    None}. Condition (1) must hold on every seed in SEEDS; (2) and (3) compare the
    run's seed mean against the controls' seed mean.
    """
    import pandas as pd

    ctrl = df[df.term == "control"]
    ctrl_mean = ctrl[MEAN_COLS].mean()
    rows = [{"term": "control", "weight": 0.0, "n_seeds": len(ctrl),
             "closed_seeds": int(ctrl.gate_closed.astype(bool).sum()),
             **ctrl_mean.to_dict()}]
    chosen, chosen_rev = {}, {}
    for term, (_, _, grid) in GRIDS.items():
        best = best_rev = None
        for w in grid:
            d = df[(df.term == term) & (df.weight == w)]
            if d.empty:
                continue
            m = d[MEAN_COLS].mean()
            gate_ok = len(d) == len(SEEDS) and bool(d.gate_ok.astype(bool).all())
            if term == "err":
                improves = m.F_err_T > ctrl_mean.F_err_T
            else:
                improves = (m.c5 < ctrl_mean.c5
                            and m.delta_qec_mean < ctrl_mean.delta_qec_mean)
            et_ok = m.F_ET >= ctrl_mean.F_ET - ET_DROP_TOL
            rows.append({"term": term, "weight": w, "n_seeds": len(d), **m.to_dict(),
                         "gate_ok_seeds": int(d.gate_ok.astype(bool).sum()),
                         "closed_seeds": int(d.gate_closed.astype(bool).sum()),
                         "improves_seeds": int(d.improves.astype(bool).sum()),
                         "et_ok_seeds": int(d.et_ok.astype(bool).sum()),
                         "gate_ok": gate_ok, "improves": bool(improves),
                         "et_ok": bool(et_ok)})
            if gate_ok and improves:
                best = w                       # grids are ascending: keep the largest
                if et_ok:
                    best_rev = w
        chosen[term] = best
        chosen_rev[term] = best_rev
    return pd.DataFrame(rows), chosen, chosen_rev


def summarize():
    import pandas as pd

    rows, missing = [], []
    for seed in SEEDS:
        ctrl_x, _ = _ctrl_names(seed)
        if not os.path.exists(ctrl_x):
            missing.append(ctrl_x)
            print(f"missing control {ctrl_x}; seed {seed} skipped entirely "
                  "(its cells have nothing to be scored against)")
            continue
        _require_phases(_ctrl_names(seed)[1])
        ctrl = score_x(np.load(ctrl_x))
        rows.append({"term": "control", "weight": 0.0, "seed": seed, **ctrl,
                     "gate_ok": ctrl["progress"] > 0.5,
                     "gate_closed": ctrl["F1"] >= F1_MIN})
        for term, (_, _, grid) in GRIDS.items():
            for w in grid:
                xpath, jpath = _names(term, w, seed)
                if not os.path.exists(xpath):
                    missing.append(xpath)
                    continue
                info = _require_phases(jpath)
                s = score_x(np.load(xpath))
                s["gate_closed"] = s["F1"] >= F1_MIN
                s["nit"] = "+".join(str(st["nit"]) for st in info["stages"])
                # Condition (1) is per seed, against that seed's own control.
                s["gate_ok"] = (s["progress"] > 0.5
                                and s["c1"] <= C1_TOL_FACTOR * ctrl["c1"])
                if term == "err":
                    s["improves"] = s["F_err_T"] > ctrl["F_err_T"]
                else:
                    s["improves"] = (s["c5"] < ctrl["c5"]
                                     and s["delta_qec_mean"] < ctrl["delta_qec_mean"])
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
              "finishes the gate. EST/CLAUDE.md records the same effect on T "
              "(est_c3s2, F1 ~ 0.98). Selection below is control-relative and "
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
    p.add_argument("action", choices=["calibrate", "commands", "summarize"])
    a = p.parse_args()
    {"calibrate": calibrate, "commands": commands, "summarize": summarize}[a.action]()


if __name__ == "__main__":
    main()
