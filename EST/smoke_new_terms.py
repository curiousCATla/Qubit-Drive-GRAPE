"""
Smoke scan for the extra EsT cost terms on the eigh pipeline (EST/grape_eigh.py):
ballpark the weights of the error-space infidelity `cerr` (w_err) and of the
Eq. A7 photon-number mismatch C5 (w_dn) before committing to 2000-it runs.

    python EST/smoke_new_terms.py calibrate      # score existing pulses, no training
    python EST/smoke_new_terms.py commands       # print the training commands
    python EST/smoke_new_terms.py summarize      # score the smoke runs, apply the rule

Budget: X gate, 500 it/stage, both stages, over the SIX seeds of the section 9
screen that reach a gate (0, 2, 4, 5, 6, 8; seeds 1, 3, 7, 9 park at F1 = 1/3,
tables/est_multiseed_X.csv). That is the section 9 screen budget, so each seed's
screen pulse `x_X_est_seed<s>.npy` (tables/est_multiseed_cache/X_seed<s>.json)
is that seed's no-new-terms CONTROL at no cost.

The scan was first run at seed 6 alone, with w_dn in {3, 10, 30, 100}; those
files carry no seed in their tag (`*_smoke500_w<w>`) and their table
tables/est_newterms_smoke_X_seed6.csv is kept frozen. Seed 6 reuses them where
the weight is still on the grid; every other cell is tagged
`smoke500_seed<s>_w<w>`, so the two can never overwrite each other.

Grids. `cerr` is O(0.5) on the delivered X pulses, the same order as w2*c2 ~ 0.15,
so w_err spans {0.3, 1, 3}. C5 is only ~1e-2 (7.0e-3 on u_X_est_best2000,
9.6e-3 on its stage-1 pulse): both code words have <a> = 0, so a linear
displacement shifts n0 and n1 equally and Delta_nbar_L comes from the nonlinear
terms alone. The seed-6 scan showed w_dn = 100 raising L(T) and w_dn = 30 already
halving Delta_QEC, so w_dn now spans {1, 3, 6, 10, 15, 30}, i.e. w5*c5 ~ 0.01-0.3.

Selection rule, fixed before looking at the seed-6 results. Per term, the
LARGEST weight that
  (1) reaches a gate -- progress = (F1 - F1_parked)/(1 - F1_parked) > 0.5 -- and
      keeps final c1 <= C1_TOL_FACTOR x the control's c1, and
  (2) improves its own target over the control:
        err : F_err(T) = 1 - cerr higher
        dn  : mean c5 lower AND mean Delta_QEC lower.
If no weight qualifies for a term, the summary says so and that term is not
promoted to a full run. Single budget: a ballpark, not an optimum.

REVISED AFTER SEEING THE SEED-6 RESULTS -- recorded, not hidden. The rule above
selected w_err = 3 and w_dn = 100, and both damage the overall goal: rule
(2) scores each term only on its own target. w_err = 3 reached the same
L(T) as 0.3 (0.013 vs 0.011) at F_ET 0.445 vs 0.687; w_dn = 100 raised L(T) to
0.845 against the control's 0.549. The full runs use the REVISED rule, which
adds
  (3) F_ET drops by at most ET_DROP_TOL = 0.05 below the control,
and selects w_err = 0.3, w_dn = 30. `summarize` prints both selections.

Over six seeds (`apply_rule`): (1) must hold on EVERY seed, each against its own
control; (2) and (3) compare the seed MEAN of the run against the seed mean of
the controls. `summarize` also prints how many seeds pass each condition.
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

GATE, MAXITER, N_C = "X", 500, 20
# The section 9 X seeds with progress > 0.5 (tables/est_multiseed_X.csv).
SEEDS = (0, 2, 4, 5, 6, 8)
LEGACY_SEED = 6        # the original single-seed scan; its files carry no seed tag
GRIDS = {"err": ("est_err", "--w-err", (0.3, 1.0, 3.0)),
         "dn": ("est_dn", "--w-dn", (1.0, 3.0, 6.0, 10.0, 15.0, 30.0))}
C1_TOL_FACTOR = 3.0
ET_DROP_TOL = 0.05     # revised rule (3); see module docstring
STAGE1_WEIGHTS = (1.0, 0.7, 7.0, 1.0)
BATCH = 6              # concurrent training jobs per `wait` in `commands`
CSV_PATH = os.path.join(TABLE_DIR, "est_newterms_smoke_X_6seeds.csv")
MEAN_COLS = ["F1", "F_ET", "F_err_T", "c5", "delta_qec_mean", "L_mean", "L_T",
             "eta0_mean"]


def _tag(w, seed):
    return f"smoke500_seed{seed}_w{w:g}"


def _paths(variant, tag):
    name = f"{variant}_{tag}"
    return (os.path.join(PULSE_DIR, f"x_{GATE}_{name}.npy"),
            os.path.join(LOG_DIR, f"est_{GATE}_{name}.json"))


def _names(term, w, seed):
    """(x path, json path) for one scan cell; seed 6 falls back to the legacy files."""
    variant = GRIDS[term][0]
    if seed == LEGACY_SEED:
        legacy = _paths(variant, f"smoke500_w{w:g}")
        if os.path.exists(legacy[0]):
            return legacy
    return _paths(variant, _tag(w, seed))


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
        "max_active_fock": int(r["max_active_fock"]),
        "max_abs_preimage": float(np.abs(x).max()),
    }


def calibrate():
    pulses = [("best2000 stage 1", "x_X_est_best2000_stage1.npy"),
              ("best2000 final", "x_X_est_best2000.npy"),
              ("seed-0 JAX est", "x_X_est.npy"),
              ("screen seed 6 (control)", "x_X_est_seed6.npy")]
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
    todo = []
    for seed in SEEDS:
        for term, (variant, flag, grid) in GRIDS.items():
            for w in grid:
                if os.path.exists(_names(term, w, seed)[0]):
                    continue
                todo.append(f"OMP_NUM_THREADS=1 python3 EST/train_est_eigh.py --gate {GATE} "
                            f"--variant {variant} {flag} {w:g} --seed {seed} "
                            f"--maxiter {MAXITER} --tag {_tag(w, seed)} "
                            f"> logs/smoke_{variant}_seed{seed}_w{w:g}.log 2>&1 &")
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
        ctrl = score_x(np.load(os.path.join(PULSE_DIR, f"x_{GATE}_est_seed{seed}.npy")))
        rows.append({"term": "control", "weight": 0.0, "seed": seed, **ctrl,
                     "gate_ok": ctrl["progress"] > 0.5})
        for term, (_, _, grid) in GRIDS.items():
            for w in grid:
                xpath, jpath = _names(term, w, seed)
                if not os.path.exists(xpath):
                    missing.append(xpath)
                    continue
                s = score_x(np.load(xpath))
                with open(jpath) as fh:
                    info = json.load(fh)
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
        print("WARNING: control seeds that do not reach a gate:", parked.seed.tolist())

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
