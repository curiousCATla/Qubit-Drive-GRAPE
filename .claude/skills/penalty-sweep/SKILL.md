---
name: penalty-sweep
description: Running, sharding, merging, or interpreting the GRAPE penalty-weight sweep (analysis/penalty_sweep.py, penalty_optimization.ipynb, tables/penalty_sweep_*, results/penalty_sweep_cache/) - commands, two-phase protocol, cache-hash rules, noise floor, selection rule, and pre-image caveats.
---

# Penalty-weight sweep

All commands run from the repository root.

Penalty-weight sweep (`analysis/penalty_sweep.py` + `penalty_optimization.ipynb`). A config
costs ~1250 s median at `maxiter=1500` (826-1449 s over 45 measured rows) and occupies ~2.4
cores, so run it sharded. **Always pass `--tag`** unless you mean to overwrite
`tables/penalty_sweep_X_ofat.csv`, which is the frozen pre-ramp table that
`penalty_optimization.ipynb` §11-§15 read and `validation/test_pulse_metrics.py` pins a
recommendation against:

```bash
# The post-ramp OFAT sweep: 17 configs (deriv x6, amp_max x6, ramp_ns x5) x 10 seeds = 170 runs.
# Seeds 42-51 are all in cache. To add more, shard only the NEW seeds then merge every seed.
# 4 shards is ~9.6 busy cores on a 10-core box; the 68 runs for seeds 48-51 took 7.5 h measured
# (449 min/shard, 1553 s median per run) -- budget ~26 min per new (config, seed) pair.
for i in 0 1 2 3; do
  python analysis/penalty_sweep.py --gate X --mode ofat --tag ramp \
      --seeds 48 49 50 51 --shard $i/4 &
done; wait
python analysis/penalty_sweep.py --gate X --mode ofat --tag ramp \
    --seeds 42 43 44 45 46 47 48 49 50 51 --merge

# Section 4.3's continuation table (~15 min); needs the sweep's cache to exist first.
python analysis/penalty_convergence_check.py --gate X --extra 750 \
       --out tables/penalty_convergence_check_ramp.csv
```

**Two-phase protocol (`--protocol two_phase`)**, an alternative to the single-continuous-run
recipe above: phase 1 cold-starts at `trunc_list=[22,26]` for `maxiter=1000`, phase 2
warm-starts from phase 1's output at `trunc_list=[26]` only for `maxiter=1500`. Both phases'
settings live in `analysis.penalty_sweep.TWO_PHASE`, not on the CLI (`--maxiter` is ignored
under this protocol). It occupies a distinct cache/hash space from the default `single_phase`
protocol (see the `_config_hash` note below), so switching `--protocol` never touches existing
pulses in either direction — re-running the command above with no `--protocol` flag stays a
100% cache hit. Same `--tag`/shard/merge mechanics as any other run:

```bash
for i in 0 1 2; do
  python analysis/penalty_sweep.py --gate X --mode ofat --protocol two_phase --tag two_phase \
      --seeds 42 43 44 45 46 --shard $i/3 &
done; wait
python analysis/penalty_sweep.py --gate X --mode ofat --protocol two_phase --tag two_phase \
    --seeds 42 43 44 45 46 --merge
```

**`--phase {1,2}`** splits that chain into two independently-schedulable stages instead of one
process per item, because phase 1 and phase 2 want different outer concurrency: phase 1 is
truncation-parallel (`n_jobs=2`, `trunc_list=[22,26]`) so a shard needs ~2 cores, while phase 2
has only one truncation (`trunc_list=[26]`) and gets nothing from `n_jobs>1`, so a shard needs
only 1 core. Chaining both phases per item under one flat shard count (the form above) sizes
that count for phase 1's heavier demand and leaves the machine under-used for the phase-2
portion of every item's run. Run Stage A (phase 1) to completion for the whole sweep, then
Stage B (phase 2) — Stage B errors immediately, before training anything, if it finds an item
Stage A hasn't finished, rather than silently training phase 1 inline under phase 2's
concurrency budget:

```bash
# Stage A: phase 1, n_jobs=2/item -> shard count ~= cores/2 (10-core machine -> 5)
for i in 0 1 2 3 4; do
  python analysis/penalty_sweep.py --gate X --mode ofat --protocol two_phase --phase 1 \
      --tag two_phase --seeds 42 43 44 45 46 --shard $i/5 &
done; wait

# Stage B: phase 2, n_jobs=1/item -> shard count can go up to ~cores (10)
for i in $(seq 0 9); do
  python analysis/penalty_sweep.py --gate X --mode ofat --protocol two_phase --phase 2 \
      --tag two_phase --seeds 42 43 44 45 46 --shard $i/10 &
done; wait

python analysis/penalty_sweep.py --gate X --mode ofat --protocol two_phase --tag two_phase \
    --seeds 42 43 44 45 46 --merge
```

`--merge` is phase-agnostic — it reads final scored rows regardless of whether they were
produced by the chained (no `--phase`) path or the split one, so no `--phase` flag is needed on
the `--merge` invocation itself (and `--phase 1 --merge` is rejected outright: Stage A never
writes a scored row, so there is nothing to merge). `--n-jobs` is ignored whenever `--phase` is
given — the per-phase truncation count already determines the correct value, and passing a
different one would either under-parallelize phase 1 or waste a pool on phase 2's single task.
Omitting `--phase` entirely keeps the original chained, single-process-per-item behavior, which
is still the simpler choice for small/interactive runs. The speedup comes entirely from phase
2's shard count no longer being capped by phase 1's core demand, not from any change in what
gets computed — `analysis/penalty_sweep.py`'s `_get_or_train_phase1` helper guarantees the
split path trains byte-identical pulses to the chained path, given the same config/seed
(verified directly at small scale: same warm-start, same `optimize_multi_state_pulse` calls,
only the process that runs them differs, and the resulting pulses were bit-identical).

Measured on a 10-core machine, 6 new OFAT configs: chained 4-way-shard baseline (seed 47) took
9419 s; Stage A(5-way) + Stage B(6-way) (seed 48) took 4179 s (A=1478 s, B=2701 s) -- a 2.25x
speedup. Take the exact multiplier with a grain of salt, not the direction: both runs' absolute
times were far above the ~22 min a single two-phase config costs when trained alone (one run hit
a joblib "worker stopped ... memory leak" warning), so real-machine memory/process contention is
adding overhead beyond what the simple "busy-core-count" model above predicts, on top of
whatever the split saves. Re-measure before quoting a specific number in a report.

That same run surfaced a real, separate finding, worth recording here rather than mistaking for
a bug in this split: seed 48's OFAT configs showed a MUCH larger overfit gap
(`F_coh_train - F_ped_heldout_mean`, up to 0.88) than seed 47's (<=0.0002) on the identical 6
configs. `F_coh_train` -- the trained-truncation fidelity, computed identically regardless of
which process trained the pulse -- was healthy in both groups, so this is not a broken
optimization or a scheduling artifact; it is the "Truncation convergence and wall exploitation"
failure mode from the paragraph below, manifesting hard for one seed's warm start and barely at
all for another's. A useful, if uncomfortable, data point on how seed-dependent two-phase's
known risk actually is in practice.

Phase 2 is structurally the pattern README.md's "Truncation convergence and wall exploitation"
section warns against (warm start + a single training truncation) — no extra guard was added
against it; the existing held-out scoring over `DEFAULT_EVAL_TRUNCS` (a strict superset of
either phase's `trunc_list`) is what would reveal a wall-exploiting pulse, and it runs
regardless of protocol. Read a two-phase config's held-out fidelity collapsing away from
n_c=26 as exactly that failure mode, not a bug in the harness.

Read before changing anything here:

- **The cached sweep predates the ramp.** Removing the `boundary` penalty took it out of
  `PENALTY_NAMES` and added `ramp_ns` to `FIXED`, which changes both `_label` and
  `_config_hash`, so every one of the ~60 pre-ramp cached pulses under
  `results/penalty_sweep_cache/` is stale (verified: 0 of 30 cache hits for the current config
  list). This was deliberate and accepted. Do **not** delete that cache: the existing
  `tables/penalty_sweep_X_*.csv` (all of which still carry a `lambda_boundary` column), the
  figures built from them, and `penalty_optimization.ipynb` §11-§15 all still reference those
  labels and remain a valid record of the **pre-ramp** regime. `validation/test_pulse_metrics.py`
  also pins two pre-ramp cache hashes (`INCUMBENT_HASH`, `CONTROL_HASH`) as metric fixtures.
  The 3- and 6-seed post-ramp tables are archived as `tables/penalty_sweep_X_ofat_ramp_n3.csv`
  and `tables/penalty_sweep_X_ofat_ramp_n6.csv` (the latter with its own
  `penalty_sweep_summary_ramp_n6.{csv,tex}`). They are supersets-in-time of the live table, not
  separate regimes: seeds 42-47 in the live 170-row table are byte-identical rows.
- **`penalty_optimization.ipynb` is now split at §11.** §1-§10 are **post-ramp and live**, built on
  `tables/penalty_sweep_X_ofat_ramp.csv` (17 recipes × seeds 42–51 = 170 pulses) and the three
  current axes. §9 is the post-ramp summary (`tables/penalty_sweep_summary_ramp.csv`; do not
  overwrite the unsuffixed pre-ramp `penalty_sweep_summary.csv` the report `\input`s). §10 is
  mid-training screening on the post-ramp snapshots. §11-§15 are **pre-ramp / two-phase and frozen**
  behind a banner, kept because the two-phase protocol comparison (§11/§13) has not been repeated
  post-ramp and cannot be regenerated (`boundary` is not a legal grid axis; `--mode disc-null` is
  cache-only). The two halves use disjoint variable names on purpose —
  `df_ramp`/`df_rank`/`agg_ramp`/`base_ramp`/`front`/`rec` in §1-§10, `df`/`combined`/`agg`/`base` in
  §11-§15, rebound by a restore cell immediately under the banner — because §11-§15 read globals
  the live half no longer defines. Do not "tidy" that by reusing one set of names.
  `df_rank` is `df_ramp` minus basin-collapse rows (`overfit_gap > 0.01`); still exactly one row
  at n=10 (seed 47, `ramp=50`, $F_{\mathrm{held}}=0.882$) — four extra seeds added six more
  clipped rows and no new collapse. It stays in `df_ramp` for the §4 audit. The notebook's load
  guards now derive from `SEEDS` and `N_RECIPES` in cells 10/11 rather than from literal counts,
  so adding seeds needs no code edit — only the markdown transcription.
- **The pre-ramp `deriv x boundary` grid section was deleted, not archived.** Its heatmap
  (Figure 3, `figures/penalty_grid_heatmap.*`) is not comparable to anything post-ramp, its axis
  no longer exists, and its one durable finding — the response was one-dimensional in
  `lambda_deriv` — is now a cited sentence in §7.2, where the 2-D verdict actually uses it. The
  CSV stays on disk and `grid` is still loaded, because §11/§13/§14 pool it.
  `figures/penalty_grid_heatmap.*` is now regenerated by nothing but is kept, because
  `penalty_optimization_report.tex` still `\includegraphics` it; do not treat its presence as
  evidence the notebook still produces it. `figures/penalty_disc_null_ofat.*` was likewise
  orphaned and has been deleted.
- **`--mode disc-null` and the `boundary` grid can no longer be regenerated.** `disc-null` is
  cache-only and recomputes its hashes under the post-ramp `FIXED`, so it reads all six configs
  as untrained and exits; `boundary` is not a legal grid axis any more. The CSVs are the
  artifact. §6.2 of the notebook therefore builds its floor from the `amp_max` ladder's
  non-binding rungs instead — see the note below.
- **`_config_hash` is the cache key, and the only-when-it-differs rule is what protects it.**
  It covers `FIXED`, the penalty weights, seed, maxiter, and — only when it is not
  `"single_phase"` — `protocol` plus the full `TWO_PHASE` dict. Every `FIXED_AXES` knob
  (`amp_max`, `ramp_ns`) enters the payload **only when it differs from `FIXED[axis]`**, which
  is what lets a new sweep axis be added without invalidating anything: verified directly when
  `ramp_ns` was added as an axis, the baseline hash was byte-identical before and after. Note
  the rule applies to top-level keys only — `fixed` embeds the whole `FIXED` dict verbatim, so
  changing a *default* in `FIXED` still invalidates everything, which is exactly what adding
  `ramp_ns: 48.0` to `FIXED` did. A third knob axis (`hard_amp_limit` is the obvious candidate)
  is a one-line addition to `FIXED_AXES` plus a ladder constant; the `knobs()` dict threads it
  everywhere else. Touching any of these
  invalidates all ~60 cached pulses and forces a multi-hour retrain. This is why
  `BASELINE["disc"]` stays at 0.5 even though `disc` is provably inert and is no longer a sweep
  axis — setting it to 0.0 would be a numerically null change that costs 10.6 h. `SNAPSHOT_ITERS`
  is deliberately *outside* the hash for the same reason, and `two_phase` runs skip snapshots
  entirely (`snapshot_iters=None` on both phases) rather than trying to map the single-run
  screening concept onto two separate short runs.
- **`disc`, `amp` and `boundary` are retired axes, not forgotten ones.** `disc` and `amp` are
  inert under `FIXED` (see the note above `PENALTY_NAMES`); `boundary` is gone because the
  penalty it weighted no longer exists — the endpoint condition is now structural (see the `core/ramp.py`
  note in the root `CLAUDE.md`). `penalty_viz.AXIS_NAMES` is deliberately a *superset* of
  `penalty_sweep.AXIS_NAMES` so historical CSVs still plot and grid rows are still classified
  correctly — do not "sync" the two lists.
- **There is exactly ONE post-ramp noise floor: `FLOOR = 1.11e-3` at n=10.** Notebook §6.2 derives
  it in three steps and nothing downstream is allowed a second one. (1) The `amp_max` 30 rung is
  **bit-identical** to the incumbent at all ten seeds, 26 at 9 of 10, and 22 at 5 of 10 (shaped
  otherwise) — same objective, different hash, separate run — so the pipeline is deterministic and
  contributes zero noise of its own. That is 24 null cells of 30, up from 16 of 18 at n=6.
  (2) Seed choice is therefore the only source: pooled cross-seed sd over the 13 distinct configs
  (null `amp_max` rungs and the collapsed `ramp=50` label dropped) is `s = 1.26e-3` (117 dof).
  (3) `FLOOR = t(117) * s * sqrt(2/10) = 1.11e-3` for a difference of two seed-means; ladder
  *ranges* are compared against `d_k * s / sqrt(10)` instead (Hartley's d2: 9.26e-4 at k=5,
  1.01e-3 at k=6, 1.08e-3 at k=7). **Pairing by seed does not help** — the pooled
  paired-difference sd is 1.39e-3, *larger* than the unpaired 1.26e-3, so basin choice is
  config-specific rather than a shared per-seed offset. Only `deriv` clears the floor (1.03e-2,
  9.3x). Do not report a sub-1.11e-3 fidelity difference as a ranking, and do not resurrect the
  old `floor_bitwise`/`floor_basin` pair — the pre-ramp 1.03e-3 came from the retired `disc`
  ladder and is a different regime.
  At the **historical 70% budget** the recommended-vs-incumbent gap was $+1.74\times10^{-3}$ =
  1.56× FLOOR, paired t(9)=+6.23, p=0.000, all ten seeds same sign — resolved, and by a wider
  margin than at n=6 (1.31×), on the same recipe at every sample size (`deriv=1e-6`). **That is
  no longer the shipped rule** — see the selection-rule note below.
- **The selection rule is TWO filters, and the recommendation it produces is not resolvable.**
  `visualization/penalty_viz.py` now ships `BUDGET_FRAC = 0.50` (was 0.70) **and**
  `MIN_FIDELITY = 0.995`, a floor on `F_ped_heldout_mean` — the same column `recommend` ranks by
  and `plot_pareto` puts on its y-axis, which is why Figure 4 can draw it as a horizontal dotted
  line. `recommend`, `find_best_trial`, `summary_table`, `plot_pareto` and `plot_all_trials` all
  take `min_fidelity=None` (→ the default); pass `0.0` for the historical cost-only rule.
  Under the new rule, on `agg_ramp`: 12 of 17 configs feasible (budget rejects 4, floor rejects 1
  more), and the recommendation is **`ampmax=22`** — the incumbent's recipe with a tighter
  amplitude cap, `deriv` unchanged at 1e-5. Its gap over the incumbent is $+5.3\times10^{-6}$ =
  0.005× FLOOR, t(9)=+0.41, p=0.70, signs flipping, and **five of ten seeds are bit-identical to
  the incumbent**. Read that as "no evidence for changing the production recipe at a 50% budget",
  not as an improvement; `deriv=1e-6` costs 57.2% of the transmon's allowance and is now
  out of budget. Two knock-ons worth knowing before quoting either section: §8.3's per-drive
  normalisation counterfactual **stops discriminating** at 50% (the cavity-binding `deriv=0`
  control is over the line on its transmon share too, so a transmon-only rule admits the identical
  13 configs), and §11/§13's "both protocols pick the same recipe" verdict **reverses** (they now
  pick different, mutually unresolvable recipes). Both are documented in the notebook prose.
  `validation/test_pulse_metrics.py::SelectionTest` pins the **pre-ramp** fixtures at
  `PINNED_BUDGET = 0.70` with `min_fidelity=0.0` on purpose — those assert the frozen pre-ramp
  recommendation, and two of them fail at the new default; `test_current_default_rule_is_live`
  covers the shipped rule instead. `penalty_optimization_report.tex` still states 70% throughout
  and is now out of step with the notebook.
- **Two caveats on that floor, both new at n=10 and both load-bearing.** (a) **Levene now rejects
  homogeneity** (p = 0.02, against 0.09 at n=6; Bartlett p = 0.000 as before). §6.2's
  pre-registered rule was to let Levene decide, so the pooled `s` is no longer defensible on the
  notebook's own terms. It is kept anyway because every §7/§8 claim was pre-registered against it
  and swapping rulers mid-campaign would re-rank results by changing the instrument. Read `FLOOR`
  as an *average* resolution limit across configs whose true spreads differ — conservative for
  tight configs, permissive for loose ones. A per-config Welch contrast is the honest successor
  and is the open methodological item. (b) **The floor has stopped falling.** 1.48e-3 (n=3,
  `s`=8.82e-4, 28 dof) → 1.15e-3 (n=6, `s`=9.95e-4, 65 dof) → 1.11e-3 (n=10, `s`=1.26e-3,
  117 dof). Six→ten seeds should have bought sqrt(6/10)=0.77 (→8.9e-4); it bought 3%, because `s`
  rose 26% and cancelled the sqrt(n) gain. `s` has risen at every campaign, so the extra seeds are
  **finding new basins, not measuring a fixed spread more precisely**. Halving this floor by seeds
  alone would need ~40 seeds and might not converge there. Do not propose more seeds as the fix;
  attack basin choice (warm start, restart policy) instead.
- **`amplitude_penalty` charges element-wise on the (N,4) quadratures, not on the complex
  envelope.** The `peak_amp` column is `max(|eps_C|, |eps_T|)`; the threshold `amp_max` acts on
  `max|u|` element-wise. They are different numbers, and reading one against the other is why the
  old `amp_max=20` rung looked like it should bind and mostly did not. Measured element-wise peak
  of an unconstrained converged pulse: **15.4-21.8 depending on seed** (widened by seeds 45 and
  47 at the two extremes), so the `AMP_MAX_VALUES` ladder `(10, 14, 18, 22, 26, 30)` is live at
  10/14, mixed at 18/22, and **provably inert at 30** (and at 26 for 9 seeds of 10). The n=6 note
  that 22/26/30 are all inert everywhere no longer holds — only 30 is null at every seed. Those inert rungs are the point, not waste: they optimize the identical objective
  as the incumbent under different hashes, which makes them independent replicates and the
  post-ramp replacement for both retired noise-floor instruments — they are step 1 of the single
  floor above. Do not trim them.
- **Pre-image inflation: `lambda_deriv` is the better-supported driver, but the ramp claim has
  REVERSED on seed means and is now only unresolved, not refuted.** Over the 30-70 ns ladder,
  `max_abs_preimage` vs `ramp_ns` has Spearman **rho = +0.83** at seeds 42-51, having climbed
  +0.03 (n=3) → +0.54 (n=6) → +0.83 (n=10) — monotone in sample size, which is what an
  under-sampled real effect looks like. It reaches 79% of the box, so Gate 0b's reach condition
  now PASSES and the gate fails on the correlation alone (0.83 vs the pre-registered 0.90).
  Against `lambda_deriv` it is **rho = -0.83**, the same magnitude. **The discriminator is no
  longer magnitude but per-seed sign consistency**: `lambda_deriv` keeps its sign in all ten
  seeds (min |rho_s| = 0.14), while `ramp_ns` ranges -0.37 to +0.77 and changes sign
  (min |rho_s| = 0.00). Quote directions, not strengths. Read `core/ramp.py`'s authority-loss
  argument as ramp-vs-no-ramp at the fixed 48 ns default; on **duration** the evidence has moved
  from refuting it to not yet resolving it, and the right instrument is a `hard_amp_limit` ladder,
  not more seeds. The physically coupled pair for a future grid is still `deriv x hard_amp_limit`
  — but `ramp_ns x hard_amp_limit` is no longer safely dismissed.
- **A ramp sweep is only interpretable because of the pre-image columns.** `max_abs_preimage`
  and `preimage_at_bound_frac` measure the raw L-BFGS-B variable against `hard_amp_limit=40`.
  When the box binds, the box chose the pulse and the row's fidelity is not comparable to the
  rest of its ladder (the U_Y seed-42 failure mode). **Seven rows of 170** pin it, and 16 sit
  above 99% of the box. Six are clipped-but-healthy ($F_{\mathrm{held}}$ 0.995-0.998): `ramp=60`
  seeds 42/49, `ramp=50` seed 45, `ramp=70` seed 48, and — new at n=10, on a ladder that had never
  clipped — `ampmax=10` seed 48 and `ampmax=14` seed 50. A binding *soft* cap inflates the
  pre-image exactly as the ramp does, so clipping is not a ramp-specific symptom. `ramp=50`
  seed 47 remains the only basin collapse ($F_{\mathrm{held}}=0.882$, `overfit_gap=0.098`) and is
  dropped from `df_rank` / the FLOOR pool. By every metric that existed before these columns the
  collapsed row would have been a ranking member. They are not
  recomputable from the waveform, so `run_one` now writes `x_<hash>.npy` beside `u_<hash>.npy` —
  outside `_config_hash`, for the same reason `SNAPSHOT_ITERS` is. The `constraint_report`
  metrics (`endpoint_rel_to_peak`, `out_of_band_{cav,tra}`) live in `pulse_metrics` instead,
  because those *are* recomputable and so get the normal cache backfill.
