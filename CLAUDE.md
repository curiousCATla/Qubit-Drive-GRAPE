# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two related tracks sharing one repository.

**1. Cat-code GRAPE (the main pipeline, `core/` + everything downstream).** Gradient Ascent
Pulse Engineering optimal control for a dispersively coupled transmon–cavity circuit-QED
system. Optimized waveforms prepare Fock states, implement even-parity cat-code
encode/decode, and realize single-qubit logical gates (X, Y, Z, H, T, I) on the resulting
cat-code subspace, following Heeres et al., *Nature Communications* 8, 94 (2017).

**2. Error-semitransparent (EsT) gates (`EST/`).** A separate replication of Roy, Wetherbee &
Fatemi, arXiv:2603.15356 — logical gates on a binomial *kitten* code that stay
error-transparent mid-gate. Different chip, different code, JAX autodiff instead of
hand-derived adjoint gradients. It reuses only `core.grape_core.make_ops`/`basis_state` and
`core.fourier_cutoff.make_band_mask`; it does **not** modify anything under `core/`,
`validation/`, or `analysis/`. See "EsT module" below and `EST/README.md`.

Full physics/math background for track 1 (Hamiltonian, cat-code definition, adjoint gradients,
frequency-band-limited controls, penalty terms, truncation-convergence protocol, decoherence
model) is documented in `README.md` — read it before making non-trivial changes to
`core/grape_core.py`, `core/optimizer.py`, or `core/cat_code.py`; it is not repeated here.
Track 2's equivalent is `EST/README.md`.

## Setup

```bash
pip install -r requirements.txt   # numpy, scipy, joblib, matplotlib, pandas, jax
pip install qutip                 # QuTip/qutip_validate.py cross-check only
```

`requirements.txt` covers everything except qutip. `jax` is needed only by `EST/`; the
`core/` pipeline remains pure numpy/scipy.

Runs on Python 3.9. Note: on Apple's Accelerate BLAS backend, `core/grape_core.py` deliberately
sets `np.seterr(divide='ignore', over='ignore', invalid='ignore')` to silence spurious
RuntimeWarnings from ordinary complex matmuls (verified benign — no actual NaN/Inf).

## Commands

There is no build step and no pytest config — tests are plain `unittest` files run directly.

`main.py` is the production training entry point for a single operation, and its CLI defaults
**are** the production recipe (`analysis/penalty_sweep.py`'s `FIXED` and `experiments.ipynb`'s
`OPTIMIZATION_RECIPE` both mirror it). Training one gate takes on the order of an hour:

```bash
# Train one gate at the production recipe. All of these are already the defaults:
#   --trunc-list 22 24 26  --max-iter 1500  --n-jobs 3  --seed 42
#   --ramp-ns 48.0  --hard-amp-limit 40.0  --amp-max 40.0
#   --lambda-deriv 1e-5  --lambda-amp 8e-5  --lambda-disc 0.5
#   --cav-band -27 27  --tra-band -33 33  --fidelity-fn auto (=> coherent, every gate)
python main.py --gate X

# U_Y is the exception: seed 42 drives max|x| into the hard_amp_limit box and
# converges into a 44%-leakage basin. pulses/u_Y_main.npy is a seed-44 pulse.
python main.py --gate Y --seed 44

# Disable the ramp (0 means off, not "zero-width"); needed for pulses too short
# to hold a flat top. 'none' likewise disables a band.
python main.py --gate X --ramp-ns 0 --cav-band none --tra-band none
```

```bash
# Regression suite for the batched fidelity core (pre-refactor equivalence,
# finite-difference gradient check, optimizer smoke tests). Run from repo root.
python validation/test_grape_core_perf.py -v

# Run a single test case / method
python -m unittest validation.test_grape_core_perf.TimingBenchmarkTest -v
python -m unittest validation.test_grape_core_perf.TimingBenchmarkTest.test_refine_dt_shaped_timing -v

# Acceptance checks for core/propagator.py (propagator order, logical-block index
# convention, Pedersen formula). Read the module docstring before adding checks here:
# two obvious-looking assertions are tautologies and are deliberately written otherwise.
python validation/test_phase0.py -v

# Re-score saved logical-gate pulses with the propagator-based Pedersen fidelity,
# over trained AND held-out truncations -> tables/phase0_corrected_fidelities.csv
python analysis/rescore_saved_pulses.py

# Five-tier validation suite over all saved pulses in pulses/
python validation/validate_logical_gates.py

# Truncation-convergence sweep (Heeres Eqs. 23-24 validity criterion)
python validation/truncation_convergence.py

# Independent fidelity cross-check via qutip.sesolve (does not import core.grape_core.make_ops)
python QuTip/qutip_validate.py
```

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
  CSV stays on disk and `grid` is still loaded, because §11/§13/§14 pool it. Two figure files are
  now orphaned and regenerated by nothing: `figures/penalty_grid_heatmap.*` and
  `figures/penalty_disc_null_ofat.*`. Left in place rather than deleted; do not treat their
  presence as evidence the notebook still produces them.
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
  penalty it weighted no longer exists — the endpoint condition is now structural (see the ramp
  note below). `penalty_viz.AXIS_NAMES` is deliberately a *superset* of
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

EsT module (`EST/`) — all from the repo root:

```bash
# 32-test correctness suite. Run before trusting any EsT training result (~80 s).
python EST/test_grape_jax.py -v

# Train one gate variant. ~1 h each at the default N=1000, dt=1 ns, 600 it/stage.
python EST/train_est.py --gate X --variant est   # full C1+C2+C3+C4
python EST/train_est.py --gate X --variant ord   # control: C1+C4 only

# Warm restart. Prefer --init-x (exact, loads the saved raw pre-image) over
# --init (inverts the constraint chain and usually needs a raised --hard-bound).
python EST/train_est.py --gate X --variant est --maxiter 1000 \
       --init-x pulses/est/x_X_est.npy --tag warm

# Eqs. 6-8 metrics, truncation scan, eigh re-score, Fig. 1d-f figure (log + linear)
python EST/diagnostics.py               # X; --gate H for another trained gate
python EST/diagnostics.py --gate X --suffix _it600   # an archived/alternate pulse pair

# Fig. 1a: <n>(t) for the code space vs the error space
python EST/subspace_evolution.py --gate H
```

All scripts assume they are run from the repository root (several insert `REPO_ROOT` onto
`sys.path` explicitly, e.g. `analysis/logical_gate_analysis1.py`).

## Architecture

**Data flow:** `core/grape_core.py` (Hamiltonian + propagation + fidelity/gradient) is the
computational kernel. `core/cat_code.py` builds physics-specific targets (cat states, logical
gate state pairs) on top of it. `core/optimizer.py` drives L-BFGS-B optimization using those
targets and saves results to `pulses/*.npy`. Everything downstream (`analysis/`, `validation/`,
`visualization/`, `wigner/`) consumes saved pulses rather than re-deriving them.

- **`core/grape_core.py`** — `make_ops`/`make_hamiltonian` build drift/control Hamiltonians;
  `_fidelity_core` is the shared batched kernel behind `fidelity_grad` and
  `fidelity_multi_state` (propagates all target states jointly per time step, amortizing one
  eigendecomposition per step across the batch — this is the main performance-sensitive path).
  Propagators use eigendecomposition (`U_k = V diag(e^{-i dt ω}) V†`), not matrix exponentiation.
  Penalty terms (`derivative_penalty`, `amplitude_penalty`) and the
  lower-level `optimize_controls` also live here.
- **`core/cat_code.py`** — cat-state construction and per-gate `get_*_state_pairs()` factories
  (encode, decode, X, Y, Z, H, T, I) that `optimizer.py` consumes; `validate_pulse_truncations`
  implements the Heeres truncation-convergence check.
- **`core/fourier_cutoff.py`** — `project_bandlimit` implements the hard frequency-band
  constraint as an orthogonal projection (`IFFT ∘ mask ∘ FFT`), applied identically to pulses
  and gradients since the projection is idempotent/self-adjoint. Enabled via `cav_band`/
  `tra_band` kwargs on `optimize_multi_state_pulse`; both must be given or both `None`
  (exactly one raises). The default is `None` — the band limit is **off** unless the caller
  asks for it, unlike `ramp_ns`, which is on by default at the `optimizer.py` entry points.
  Every production caller passes the bands explicitly (`-27/27`, `-33/33`).
- **`core/ramp.py`** — the 48 ns pedestal-subtracted Gaussian rise/fall (`ramp_envelope`) and
  the full raw-variable→physical-pulse chain `make_constraint_chain`, which returns the
  `(to_physical, to_preimage_grad)` pair every cost assembly uses. Order is **band-limit then
  ramp**, so `u = env * P(x)` and the adjoint is `P(env * g)` — the envelope multiplies the
  gradient *before* the projection. Getting that backwards still produces a plausible-looking
  gradient, so `validation/test_grape_core_perf.py` pins the adjoint identity with a negative
  control; do not "simplify" the order. This module **replaced `boundary_penalty`**, which was
  removed outright — passing `penalties={'boundary': ...}` now raises. `EST/device.py`
  re-exports `ramp_envelope` from here (core never imports from EST), so the two tracks share
  one implementation. Also holds `deramp` (chain inverse, for physical-pulse warm starts) and
  `constraint_report` (per-pulse endpoint + residual out-of-band measurement, surfaced as
  `info['constraints']`). When reporting endpoints prefer `endpoint_rel_to_peak`;
  `endpoint_rel_to_mid` divides a max by an RMS, and the two are not interchangeable — the
  module docstring's white-noise ordering benchmark ("0.84% of mid") quotes the latter.
- **The ramp defaults ON at the `optimizer.py` entry points and OFF everywhere else.**
  `ramp_envelope` and the four `optimizer.py` entry points default to `DEFAULT_RAMP_NS = 48.0`,
  but `make_constraint_chain` itself defaults to `ramp_ns=None`, as do the three legacy
  `grape_core` objective makers (`make_objective_with_pen`, `make_objective_multi_trunc`,
  `optimize_controls`). That is deliberate — silently reshaping the older notebooks' pulses
  would change historical results — but it means calling the legacy makers produces *unramped*
  pulses with no warning. `ramp_ns=0` (or `0.0`) also disables, which is what `main.py
  --ramp-ns 0` relies on.
- **`hard_amp_limit`, not the envelope, is what bounds the endpoint amplitude — and it can
  bind.** The ramp cuts `max|u[0]|/peak|u|` on every operation (typically ~3x, up to 11x: Z
  3.64%->0.32%, T 4.98%->0.47%, I 3.87%->0.34%, opt 6.06%->1.13%), but it does not reach zero,
  and how close it gets tracks how amplitude-hungry the gate is (X only 4.75%->2.39%). The
  reason: the envelope removes control authority over the first/last 24 steps and L-BFGS-B buys
  it back by inflating the *pre-image* there — on `u_X_main`, `|P(x)[0]| = 30.6` against a
  mid-pulse RMS of 5.03 (6x), so `u[0] = 0.0135 * 30.6 = 0.42` survives. What stops that
  inflation is the box on the raw variable, so raising `hard_amp_limit` would quietly undo part
  of the ramp. **Mind which default you are getting**: `optimize_multi_state_pulse` defaults to
  `hard_amp_limit=50.0`, `refine_pulse`/`refine_pulse_dt`/`refine_pulse_dt_light` to 40.0, and
  every production path (`main.py`, `analysis/penalty_sweep.py` `FIXED`, the notebook recipe)
  pins **40.0**. All the numbers below are against 40, so comparing them to a bare
  `optimize_multi_state_pulse` call's box is comparing to the wrong number.
  **Y is the cautionary case**: under a cold start at seed 42 it drove
  `max|x|` to exactly 40.0 (0.46% of entries pinned at the bound) and converged into a
  44%-leakage basin, `F_ped` held-out 0.9973 -> 0.7491. Seeds 43 and 45 also bound the box
  (`F_coh` 0.9976 / 0.8908); seed 44 stayed clear at `max|x| = 26.59` and reached
  `F_coh = 0.9988`. **`pulses/u_Y_main.npy` is therefore a seed-44 pulse** — every other
  operation is a seed-42 cold start — and is reproduced with `python main.py --gate Y
  --seed 44`, not by the bare recipe. Note that `experiments.ipynb`'s `OPTIMIZATION_RECIPE`
  carries no seed key, so re-running its Section 5 loop with `RUN_OPTIMIZATION=True` would
  silently regenerate the broken seed-42 Y. Always check `info['max_abs_preimage']` against
  `hard_amp_limit` before trusting a retrained pulse: a pulse at the bound has been clipped,
  not converged (for reference, X sits at 32.3 and enc at 32.7).
  Midpoint sampling (EST's convention, `env[0]=0.0135` not 0) is the other half of the gap.
- **`core/optimizer.py`** — `optimize_multi_state_pulse()` (top-level entry point: averaged
  fidelity over `trunc_list`, optional discrepancy penalty, joblib-parallel truncation
  evaluation, saves to `save_path`), `refine_pulse()`, `refine_pulse_dt()` /
  `refine_pulse_dt_light()` (finer-timestep refinement of an existing pulse).
- **`core/compare_pulses.py`** — shared fidelity + shape-metrics table reused by multiple
  visualization/validation scripts; not a standalone entry point for new comparisons.
- **`core/propagator.py`** — measurement-only: builds the full D×D propagator for a saved
  pulse (`full_propagator`, LEFT-multiply `U = Uk @ U`) and extracts the effective 2×2
  logical map by explicit projection (`logical_block` = `B† U B`, **column per input** —
  the convention `IDEAL_LOGICAL_U` is written in). Provides `pedersen_gate_fidelity`
  (leakage enters via `Tr(MM†)`, not an assumed `+d`), `leakage_L1`, and `seepage_report`
  for non-logical inputs. Deliberately does not import from `validation/` (ideal targets
  are always arguments) so `validate_logical_gates.py` can import it without a cycle.
  Nothing here feeds any training objective.
- **`validation/`** — `test_grape_core_perf.py` is the correctness/regression suite (equivalence
  vs. a frozen pre-refactor reference + finite-difference gradient checks); `validate_logical_gates.py`
  runs the five-tier protocol (fidelity-vs-truncation, logical action/leakage, gate algebra,
  encode-gate-decode pipeline, effective-unitary extraction) against `pulses/`;
  `truncation_convergence.py` checks the Eqs. 23-24 plateau criterion in isolation.
- **`QuTip/`** — fully independent re-implementation of the physics (operators, Hamiltonians,
  propagation via `qutip.sesolve`) used to cross-check `core/grape_core.py`'s hand-rolled
  eigendecomposition propagator; deliberately does not import from `core/`.
- **`analysis/`** — one-off/ad-hoc optimization and comparison scripts built on `core/` and
  `pulses/` (e.g. `decoherence.py` for Lindblad decoherence-limited fidelity, `pulse_analysis.py`
  for trajectory/Fock-population inspection). Not a stable API — read the specific script before
  reusing it.
- **`visualization/`** — waveform/spectrum plots (`pulse_viz.py`), Wigner tomography
  (`wigner_viz.py`), QuTiP cross-check figures (`plot_qutip_validation.py`).
- **`pulses/`** — saved control sequences, `u_*.npy`, shape `(N, 4)` = `(cavity I, cavity Q,
  transmon I, transmon Q)` per time step at `dt = 0.002 μs`. Treat as generated artifacts;
  `pulses/u_*_main.npy` are the current canonical logical-gate pulses (retrained under the
  cold-start + Eqs. 23/24 protocol — see README "Truncation convergence" section before
  reintroducing warm-started multi-truncation training, which is known to produce pulses that
  exploit the Hilbert-space truncation wall). Every `u_*_main.npy` is now accompanied by its
  raw optimizer pre-image `x_*_main.npy`, same shape, written by `save_preimage=True`. Only
  `u` is physical and only `u` is ever scored; `x` exists so a run resumes *exactly* via
  `init_x`. Prefer `init_x` over `warm_start` — a physical `warm_start` has to be inverted
  through the chain by `deramp`, which divides by an envelope that floors at 0.0135 at core
  geometry, and a pulse not produced by this chain (anything pre-ramp) is refused rather than
  silently mis-resumed unless you pass `warm_start_strict=False`. Same rule as `pulses/est/`.
- **`figures/`, `tables/`, `wigner/`, `results/`, `logs/`** — generated outputs (figures, CSV
  summaries, campaign metadata). Do not hand-edit; regenerate via the corresponding script.
- **The notebook-owned half of `tables/` is now post-ramp; the script-owned half is not.**
  The `pulses/u_*_main.npy` set was retrained on 2026-09-09. `experiments.ipynb` was then
  re-executed top to bottom on 2026-09-10 when Section 8 was added (43 code cells, ~7 min,
  `RUN_OPTIMIZATION=False`, zero errors, no pulse rewritten), so its **stored cell outputs are
  post-ramp**, and so is everything it writes: `validation_master_summary.csv`,
  `gate_campaign_summary.csv`, `pulse_characterization.csv`, `decoherence_*.csv`,
  `unitary_*.csv`, `process_tomography_*.csv`, and `results/gate_campaign_info.json` (whose
  stored recipe no longer lists the removed `boundary` penalty), alongside
  `phase0_corrected_fidelities.csv` and `phase2_summary.csv`, which were already current.
  Post-ramp `F_avg_gate`/`Pipeline_avg` now exist in both the notebook markdown (Section 6) and
  the CSVs. **This does not extend to tables the notebook never writes** — the
  `penalty_sweep_*` family above in particular keeps its own, separately documented provenance,
  and the pre-ramp cache warnings there still stand. Check mtimes before quoting a number from
  anything outside the notebook-owned list.

### EsT module (`EST/`)

Self-contained replication of arXiv:2603.15356; parallel structure to `core/` but for a
different device and code. Full details in `EST/README.md`.

- **`EST/device.py`** — Table I constants (rad/μs, same `two_pi *` convention as
  `grape_core`) and `make_hamiltonian_est`. Two differences from
  `grape_core.make_hamiltonian` beyond the constants: a second-order cavity self-Kerr term
  `K'` that the main pipeline's `H0` does not have, and `n_t >= 3` is **enforced with a
  raise** (at `n_t=2` the main pipeline silently drops the anharmonicity term, which here
  would discard `K_q/2pi = -180 MHz`). Also holds `ramp_envelope` and `band_mask`.
- **`EST/kitten_code.py`** — binomial kitten code `|0_L>=(|0>+|4>)/sqrt2`, `|1_L>=|2>`;
  error words `|3>`, `|1>`. Gate targets are built by applying the ideal 2x2 to *coefficient
  vectors* (reusing `core.propagator.PAULI_EIGENSTATE_COEFFS`), never by hand-permuting
  states. `error_cardinals` carries a live assertion on a code-specific coincidence — read
  its docstring before reusing it for a different code.
- **`EST/grape_jax.py`** — the only module that replaces existing machinery. Constraint chain
  (band-limit -> ramp -> control mask) applied *inside* the cost so autodiff covers it; `expm`
  propagation (**not** `eigh` — autodiff through `eigh` hits the same degeneracy singularity
  `grape_core` hand-fixes); cost terms C1-C4; `build_gate_objective` returns a scipy-ready
  `(cost, grad)` callable.
- **`EST/train_est.py`** — two-stage L-BFGS-B driver. `--variant ord` is the same code path
  with `w2=w3=0`; keep it that way, the whole result is a comparison against it. `train()`
  returns `(u, x, info)` and `save()` writes both `u` and `x` — see `pulses/est/` below.
- **`EST/diagnostics.py`** — Eqs. 6-8 metrics + Fig. 1d-f. Propagates with
  `grape_core.step_data`'s **eigh** propagator, deliberately not the JAX `expm` path, so
  re-scoring is an independent check. `propagate_states` is the only trajectory function in
  the repo that returns state *vectors* rather than populations. Only `main()` is gate-
  specific; everything below it already takes `gate` as an argument; `--suffix` scores a
  non-default pulse pair (`_it600`, `_warm`) into correspondingly suffixed outputs.
  **`delta_qec`, `leakage_Ej` and `eta_mismatch` all take the two evolved code WORDS**, not
  the six-cardinal stack: `a` on the code space is rank 2, so a wider array yields basis
  vectors outside the error space and silently understates `L` (it read 0.148 against the
  correct 0.258 on `u_X_est`). `_require_code_words` raises rather than allowing it.
  `c2_integrand` is **not** a paper metric — it is the C2 training cost's integrand, kept
  only so the numpy and JAX paths can be cross-checked, and must never be reported as a
  transparency number.
- **`EST/subspace_evolution.py`** — Fig. 1a. Per Bloch cardinal, plots `<n>(t)` of the
  evolved code state, of the evolved error state, and of the normalized photon-loss image
  `a|psi_C(t)>` that the error state must match under transparency. Propagates only the
  four basis columns and rebuilds every cardinal by linearity (`cardinal_states`), reusing
  `diagnostics.propagate_states`. **`<n>` is exactly conserved under the drift** (`[H0, n] = 0`),
  so it is blind to App. A's obstruction — agreement is necessary, not sufficient, which is
  why the figure also carries `map_mismatch`, a phase-insensitive `U_L`-vs-`U_E` distance on
  the **fixed** code/error bases. That metric's normalization is this module's choice, so its
  EsT:Ord ratio is meaningful and its absolute value is not. That caveat no longer extends to
  Eqs. 6-8, which are transcribed — `map_mismatch` is now the only quantity here carrying it.
- **`EST/test_grape_jax.py`** — 38 tests. The anchor is the C1 gradient checked against
  `grape_core.fidelity_multi_state`'s analytic adjoint at rtol 1e-6 (sign-flipped: JAX
  returns cost, numpy returns fidelity).
- **`pulses/est/`** — EsT pulses `u_<gate>_<variant>.npy` **and** their raw optimizer
  pre-images `x_<gate>_<variant>.npy`, both `(N,4)`, final and per stage. Only `u` is
  physical and only `u` is ever scored; `x` exists so a run can be resumed exactly via
  `--init-x`. Do not reconstruct `x` from `u` with `deramp` when an `x_*.npy` exists — the
  inversion divides by a ramp envelope that reaches 0.0066 at the pulse edges, so the
  pre-image it returns can fall outside the default box (81.0 for `u_X_est`, 158.5 for
  `u_X_est_warm`, against `hard_bound=60`). Kept out of `pulses/` on purpose; see
  conventions below.

## Working conventions

- Physical parameters (`chi`, `Kerr`, `chip`, `alpha`, `dt`) are defined in lab units (MHz, μs)
  at the top of `core/grape_core.py` and converted to rad/μs via `two_pi = 2*np.pi`. Follow this
  convention rather than hand-converting elsewhere.
- Fidelity/gradient function signatures in `grape_core.py` are considered a stable interface
  (`fidelity_grad`, `fidelity_multi_state` are thin wrappers over `_fidelity_core`); if you
  change `_fidelity_core`, `test_grape_core_perf.py` must still pass its pre-refactor
  equivalence and finite-difference checks.
- When adding a new logical gate, follow the pattern in `core/cat_code.py`
  (`get_logical_*_state_pairs`) and register it the way `analysis/logical_gate_analysis1.py`
  does, rather than special-casing it inside `optimizer.py`.

### EsT-specific conventions

- **Never mutate the `chi`/`Kerr`/`chip`/`alpha` globals in `core/grape_core.py`** to run EsT
  work. Every pulse in `pulses/` and the whole `validation/`+`analysis/` layer is scored
  against them. `EST/device.py` builds its own Hamiltonian for exactly this reason.
- **EsT pulses go to `pulses/est/`, never `pulses/`.** `validation/validate_logical_gates.py`'s
  `GATE_PULSE_MAP` and every `analysis/` script locate pulses by the hardcoded name
  `u_<gate>_main.npy` and assume the alpha=sqrt(3) four-component cat code. A kitten-code
  pulse in `pulses/` would be silently mis-scored, not rejected.
- **Amplitude is capped by the C4 penalty, not by the L-BFGS-B box.** Because the constraint
  chain lives inside the cost, `bounds` constrain the raw pre-image, and the cap is circular
  (`sqrt(u0^2+u1^2) <= eps_max`) which a per-element box cannot express. The default
  `hard_bound=60` permits `|eps|` up to 84.9 rad/us, 3.4x the 25.13 rad/us physical cap, so it
  never binds. Do not "tighten" it expecting the waveform to follow.
- **C2 and C3 deviate deliberately from the paper's printed equations** (C2 divides by
  `N_norm^2`; C3 is normalized by `mean(v)^2` because the literal form is dimensionful and
  swamps the fidelity terms by ~700x at dt=1 ns). Both are documented in
  `EST/grape_jax.py` and pinned by tests. Do not "correct" them back without reading those.
- Eqs. 6-8 in `EST/diagnostics.py` are **transcribed** from the paper, so both absolute values
  and EsT:Ord ratios are comparable to it. `Delta_QEC` is App. A's closed form
  (`sum_ik 2(|x|^2+|y|^2+|z|^2)`, error set `{I,a}`) and is **unnormalized** — do not
  reintroduce the `/nbar` division that made it incomparable. `eta` is Eq. 8's Bloch-vector
  distance in **[0,2]** on the error state *projected* into the instantaneous error space,
  which makes it leakage-insensitive, so **`L <= eta` is no longer true by construction** and
  `mean(eta) != 1 - F_ET` (that identity moved to `c2_integrand`). One interpretive choice
  survives: Eq. 8 fixes `sigma_Ej ∝ Ej sigma_C Ej^dag` only up to a constant, resolved here by
  the polar isometry of `a P_C(t)`; sensitivity against the literal reading is 0.4%, pinned by
  `test_eta_basis_convention_is_pinned`. Where main-text Eq. 6 says "2-norm" and App. A says
  Frobenius, App. A wins.
- **These metrics are analysis-only and touch no objective.** `et_cost` (C2) lives in
  `EST/grape_jax.py` and is never imported by the analysis path, so changing Eqs. 6-8
  retrains nothing and leaves every pulse, `logs/est_*.json` `c2`, and multiseed
  `F_ET_trained` untouched. The reverse is not true: changing `et_cost` invalidates all of it.
- After changing anything in `EST/`, run `python EST/test_grape_jax.py` before training —
  a training run is ~1 h per variant.
