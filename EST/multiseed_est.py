"""
Multi-seed sampling for EsT gate training, on the eigh + analytic-adjoint
pipeline (EST/grape_eigh.py via EST/train_est_eigh.py).

    python EST/multiseed_est.py --gate X --seeds 0 1 2 3 4

EsT training is otherwise single-seed: one cold-start draw
`amp0 * default_rng(seed).standard_normal(N*4)`, one two-stage run. This script
runs the `est` variant from several independent seeds at a reduced budget
(500 iterations PER STAGE by default), scores each pulse, and keeps the best.

It is the EsT analogue of `analysis/penalty_sweep.py`: a thin orchestration
layer over the existing trainer, with the same per-item cache + `--shard I/K`
/ `--merge` split so seeds can run as concurrent processes.

The screening idea is section 11 of `penalty_optimization.ipynb`
("does a cheap screen preserve the ranking?"): a partial run already settles the
top of the ranking, so it is cheap to sample many starts and commit to the
winner. The selection is optimistic by construction -- see SELECTION_BIAS_NOTE
and the per-seed table `tables/est_multiseed_<gate>.csv`.

Ranking metric (fidelity + transparency):

    composite = F_heldout_mean + w_et * F_ET_trained

  F_heldout_mean  mean C1 gate fidelity re-scored (eigh) over the HELD-OUT
                  truncations {16,18,22,24,28}; training is at n_c=20 only.
  F_ET_trained    1 - c2 at n_c=20, the error-transparency fidelity (C2).

`--et-floor` optionally drops seeds whose F_ET_trained is below a threshold
before ranking (feasible-filter-then-sort, as in
visualization/penalty_viz.find_best_trial). Default 0.0 = no gate.

Outputs (unless --no-save):

    pulses/est/u_<gate>_est_multiseed.npy      winner pulse
    pulses/est/x_<gate>_est_multiseed.npy      winner raw pre-image
    logs/est_<gate>_est_multiseed.json         winner info + `multiseed` block
    pulses/est/u_<gate>_est_seed<seed>.npy     every seed's pulse (warm-start source)
    pulses/est/x_<gate>_est_seed<seed>.npy
    tables/est_multiseed_<gate>.csv            one row per seed
    tables/est_multiseed_cache/<gate>_seed<seed>.json   per-seed cache
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from EST import kitten_code
from EST.device import DT, N_T
from EST.diagnostics import rescore, truncation_scan
from EST.train_est import TRAIN_TRUNC          # [20]
from EST.train_est_eigh import PULSE_DIR, save, train

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_DIR = os.path.join(REPO_ROOT, "tables")
CACHE_DIR = os.path.join(TABLE_DIR, "est_multiseed_cache")

# C1 is re-scored at these truncations; the ones not in TRAIN_TRUNC are the
# held-out set that F_heldout_mean averages over.
EVAL_TRUNCS = (16, 18, 20, 22, 24, 28)
HELDOUT_TRUNCS = tuple(n for n in EVAL_TRUNCS if n not in TRAIN_TRUNC)

# Held-out C1 spread below this is "converged" (the same 1e-3 threshold
# EST/diagnostics.py:main uses for its truncation scan).
SPREAD_TOL = 1e-3

SELECTION_BIAS_NOTE = (
    "Picking the best of several independent seeds is optimistic by "
    "construction (selection bias): the winner's headline score sits above "
    "what a fresh seed would give in expectation. The per-seed table is the "
    "honest picture of seed-to-seed spread; read the winner as 'the best draw "
    "we have', not as evidence this seed is genuinely superior to the others. "
    "See visualization/penalty_viz.find_best_trial for the same caveat in the "
    "cat-code sweep."
)

CSV_FIELDS = [
    "seed", "F1_trained", "F_ET_trained", "C3_trained",
    "F_heldout_mean", "F_heldout_spread", "heldout_converged",
    "composite", "nit_stage1", "nit_stage2", "seconds", "chosen",
]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _score_pulse(u, gate, terms_trained):
    """
    Score one trained pulse. `terms_trained` is info["stages"][-1]["terms"],
    i.e. {n_c: {c1,c2,c3,c4}} from the final report(x); it carries only the
    trained truncation (TRAIN_TRUNC == [20]).
    """
    terms = {int(k): v for k, v in terms_trained.items()}
    t = terms[TRAIN_TRUNC[0]]

    scan = truncation_scan(u, gate, n_t=N_T, trunc_list=EVAL_TRUNCS, dt=DT)
    f1_trained = float(scan[TRAIN_TRUNC[0]])
    heldout = [float(scan[n]) for n in HELDOUT_TRUNCS]
    heldout_mean = float(np.mean(heldout))
    heldout_spread = float(max(heldout) - min(heldout))

    # The eigh re-score of C1 at n_c=20 should reproduce the training cost's
    # own 1-c1 -- both are the same propagator. A gap means the wiring is off.
    f1_from_cost = 1.0 - float(t["c1"])
    if abs(f1_from_cost - f1_trained) > 1e-5:
        print(f"  WARNING: F1 re-score {f1_trained:.8f} disagrees with "
              f"training cost 1-c1 {f1_from_cost:.8f} (gap "
              f"{abs(f1_from_cost - f1_trained):.2e})")

    return {
        "F1_trained": f1_trained,
        "F_ET_trained": 1.0 - float(t["c2"]),
        "C3_trained": float(t["c3"]),
        "F_heldout_mean": heldout_mean,
        "F_heldout_spread": heldout_spread,
        "heldout_converged": bool(heldout_spread < SPREAD_TOL),
        "trunc_scan": {int(k): float(v) for k, v in scan.items()},
    }


# ---------------------------------------------------------------------------
# per-seed run + cache
# ---------------------------------------------------------------------------
def _cache_path(gate, seed):
    return os.path.join(CACHE_DIR, f"{gate}_seed{seed}.json")


def _seed_pulse_paths(gate, seed):
    return (os.path.join(PULSE_DIR, f"u_{gate}_est_seed{seed}.npy"),
            os.path.join(PULSE_DIR, f"x_{gate}_est_seed{seed}.npy"))


def run_seed(gate, seed, maxiter, amp0, hard_bound, force=False):
    """
    Train one seed (or load its cache) and return a record dict. The pulse and
    pre-image are always written to pulses/est/u_<gate>_est_seed<seed>.npy so
    --merge (and any later warm start) can recover them.
    """
    cpath = _cache_path(gate, seed)
    if os.path.exists(cpath) and not force:
        with open(cpath) as fh:
            rec = json.load(fh)
        print(f"[cache] seed {seed}: F1_trained={rec['F1_trained']:.6f} "
              f"F_ET_trained={rec['F_ET_trained']:.6f}")
        return rec

    print(f"\n{'=' * 70}\n[train] {gate} / est / seed {seed}  "
          f"maxiter={maxiter}/stage\n{'=' * 70}")
    t0 = time.time()
    u, x, info = train(gate=gate, variant="est", maxiter=maxiter, seed=seed,
                       amp0=amp0, hard_bound=hard_bound, verbose=True)
    wall = time.time() - t0

    stages = info["stages"]
    scores = _score_pulse(u, gate, stages[-1]["terms"])

    upath, xpath = _seed_pulse_paths(gate, seed)
    os.makedirs(PULSE_DIR, exist_ok=True)
    np.save(upath, u)
    np.save(xpath, x)

    rec = {
        "gate": gate,
        "seed": seed,
        "maxiter": maxiter,
        "amp0": amp0,
        "hard_bound": hard_bound,
        "nit_stage1": int(stages[0]["nit"]),
        "nit_stage2": int(stages[1]["nit"]),
        "seconds": float(sum(s["seconds"] for s in stages)),
        "wall_seconds": float(wall),
        "u_path": os.path.relpath(upath, REPO_ROOT),
        "x_path": os.path.relpath(xpath, REPO_ROOT),
        **scores,
        "info": info,
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cpath, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"[done] seed {seed}: F1_trained={rec['F1_trained']:.6f} "
          f"F_ET_trained={rec['F_ET_trained']:.6f} "
          f"F_heldout_mean={rec['F_heldout_mean']:.6f} ({wall:.0f}s)")
    return rec


# ---------------------------------------------------------------------------
# ranking + outputs
# ---------------------------------------------------------------------------
def rank(records, w_et, et_floor):
    """Feasible-filter-then-sort. Returns (ranked_feasible, winner)."""
    for r in records:
        r["composite"] = r["F_heldout_mean"] + w_et * r["F_ET_trained"]

    feasible = [r for r in records if r["F_ET_trained"] >= et_floor]
    if not feasible:
        best = max(records, key=lambda r: r["F_ET_trained"])
        raise SystemExit(
            f"error: every seed has F_ET_trained < --et-floor={et_floor} "
            f"(best is seed {best['seed']} at {best['F_ET_trained']:.6f}); "
            "lower --et-floor or train more seeds.")

    ranked = sorted(feasible, key=lambda r: r["composite"], reverse=True)
    return ranked, ranked[0]


def write_outputs(gate, records, ranked, winner, w_et, et_floor, seeds,
                  no_save=False):
    winner_seed = winner["seed"]

    # --- per-seed CSV -------------------------------------------------------
    os.makedirs(TABLE_DIR, exist_ok=True)
    csv_path = os.path.join(TABLE_DIR, f"est_multiseed_{gate}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in sorted(records, key=lambda r: r["seed"]):
            w.writerow({
                "seed": r["seed"],
                "F1_trained": f"{r['F1_trained']:.8f}",
                "F_ET_trained": f"{r['F_ET_trained']:.8f}",
                "C3_trained": f"{r['C3_trained']:.6e}",
                "F_heldout_mean": f"{r['F_heldout_mean']:.8f}",
                "F_heldout_spread": f"{r['F_heldout_spread']:.3e}",
                "heldout_converged": r["heldout_converged"],
                "composite": f"{r['composite']:.8f}",
                "nit_stage1": r["nit_stage1"],
                "nit_stage2": r["nit_stage2"],
                "seconds": f"{r['seconds']:.1f}",
                "chosen": (r["seed"] == winner_seed),
            })
    print(f"\nwrote {csv_path}")

    # --- console ranking --------------------------------------------------
    print(f"\n--- ranking ({gate}, w_et={w_et}, et_floor={et_floor}) ---")
    print(f"{'seed':>5} {'F1_trained':>12} {'F_ET':>10} {'F_held_mean':>12} "
          f"{'spread':>10} {'composite':>12}")
    for r in ranked:
        mark = "  <- winner" if r["seed"] == winner_seed else ""
        print(f"{r['seed']:>5} {r['F1_trained']:>12.8f} "
              f"{r['F_ET_trained']:>10.6f} {r['F_heldout_mean']:>12.8f} "
              f"{r['F_heldout_spread']:>10.2e} {r['composite']:>12.8f}{mark}")
    dropped = [r["seed"] for r in records if r not in ranked]
    if dropped:
        print(f"dropped by --et-floor={et_floor}: seeds {dropped}")
    if not winner["heldout_converged"]:
        print(f"WARNING: winner seed {winner_seed} held-out C1 spread "
              f"{winner['F_heldout_spread']:.2e} >= {SPREAD_TOL:.0e} -- pulse "
              "may be exploiting the truncation wall; inspect the scan.")

    if no_save:
        print("\n--no-save: winner pulse not written")
        return

    # --- winner artifacts ------------------------------------------------
    upath, xpath = _seed_pulse_paths(gate, winner_seed)
    u = np.load(upath)
    x = np.load(xpath)

    info = dict(winner["info"])
    info["multiseed"] = {
        "seeds": list(seeds),
        "w_et": w_et,
        "et_floor": et_floor,
        "winner_seed": winner_seed,
        "eval_truncs": list(EVAL_TRUNCS),
        "heldout_truncs": list(HELDOUT_TRUNCS),
        "selection_bias_note": SELECTION_BIAS_NOTE,
        "ranking": [
            {
                "seed": r["seed"],
                "F1_trained": r["F1_trained"],
                "F_ET_trained": r["F_ET_trained"],
                "C3_trained": r["C3_trained"],
                "F_heldout_mean": r["F_heldout_mean"],
                "F_heldout_spread": r["F_heldout_spread"],
                "heldout_converged": r["heldout_converged"],
                "composite": r["composite"],
                "trunc_scan": r["trunc_scan"],
                "dropped_by_et_floor": r not in ranked,
            }
            for r in sorted(records, key=lambda r: r["seed"])
        ],
    }

    pulse_path, x_path, json_path = save(u, x, info, gate, "est_multiseed")
    print(f"\nwinner = seed {winner_seed}")
    print(f"saved {pulse_path}\n      {x_path}\n      {json_path}")


# ---------------------------------------------------------------------------
def _parse_shard(spec):
    try:
        i_str, k_str = spec.split("/")
        i, k = int(i_str), int(k_str)
    except ValueError:
        raise SystemExit(f"error: --shard must look like 'I/K', got {spec!r}")
    if k < 1 or not (0 <= i < k):
        raise SystemExit(
            f"error: --shard I/K needs K >= 1 and 0 <= I < K, got {spec!r}")
    return i, k


def _load_cached(gate, seeds):
    """Load every seed's cache for --merge; refuse a partial set."""
    records, missing = [], []
    for s in seeds:
        cpath = _cache_path(gate, s)
        if not os.path.exists(cpath):
            missing.append(s)
            continue
        with open(cpath) as fh:
            records.append(json.load(fh))
    if missing:
        raise SystemExit(
            f"error: no cache for seeds {missing} -- run them first, e.g.\n"
            f"  python EST/multiseed_est.py --gate {gate} --seeds "
            + " ".join(str(s) for s in missing))
    return records


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate", default="X",
                   choices=sorted(kitten_code.IDEAL_LOGICAL_U))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--maxiter", type=int, default=500,
                   help="L-BFGS-B iterations PER STAGE (default 500; two stages)")
    p.add_argument("--w-et", type=float, default=0.5,
                   help="weight on F_ET_trained in the composite score")
    p.add_argument("--et-floor", type=float, default=0.0,
                   help="drop seeds with F_ET_trained below this before ranking")
    p.add_argument("--amp0", type=float, default=20.0)
    p.add_argument("--hard-bound", type=float, default=60.0)
    p.add_argument("--shard", default=None, metavar="I/K",
                   help="train only this shard's slice of --seeds, into the "
                        "cache; combine with --merge afterwards")
    p.add_argument("--merge", action="store_true",
                   help="rank the cached seeds and write the winner + CSV; "
                        "trains nothing")
    p.add_argument("--force", action="store_true",
                   help="retrain seeds even if a cache entry exists")
    p.add_argument("--no-save", action="store_true",
                   help="rank and print, but do not write the winner pulse")
    args = p.parse_args()

    if args.shard and args.merge:
        raise SystemExit("error: --shard trains, --merge ranks; not both")

    seeds = args.seeds

    if args.merge:
        records = _load_cached(args.gate, seeds)
        ranked, winner = rank(records, args.w_et, args.et_floor)
        write_outputs(args.gate, records, ranked, winner, args.w_et,
                      args.et_floor, seeds, no_save=args.no_save)
        return

    if args.shard:
        i, k = _parse_shard(args.shard)
        todo = seeds[i::k]                       # seeds cost ~the same; round-robin
        print(f"shard {i}/{k}: seeds {todo}")
    else:
        todo = seeds

    for s in todo:
        run_seed(args.gate, s, args.maxiter, args.amp0, args.hard_bound,
                 force=args.force)

    if args.shard:
        print(f"\nshard {args.shard} done ({len(todo)} seeds cached). "
              f"Run:\n  python EST/multiseed_est.py --gate {args.gate} "
              f"--seeds {' '.join(str(s) for s in seeds)} --merge")
        return

    records = _load_cached(args.gate, seeds)
    ranked, winner = rank(records, args.w_et, args.et_floor)
    write_outputs(args.gate, records, ranked, winner, args.w_et,
                  args.et_floor, seeds, no_save=args.no_save)


if __name__ == "__main__":
    main()
