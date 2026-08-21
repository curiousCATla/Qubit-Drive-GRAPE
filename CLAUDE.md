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
costs ~850 s at `maxiter=1500` and occupies only ~2.4 cores, so run it sharded:

```bash
# 3 concurrent shards, then rebuild the combined table from the row cache.
for i in 0 1 2; do
  python analysis/penalty_sweep.py --gate X --mode ofat --seeds 42 43 44 --shard $i/3 &
done; wait
python analysis/penalty_sweep.py --gate X --mode ofat --seeds 42 43 44 --merge
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

- **`_config_hash` is the cache key.** It covers `FIXED`, the four penalty weights, seed,
  maxiter, and — only when it is not `"single_phase"` — `protocol` plus the full `TWO_PHASE`
  dict (mirroring how `amp_max` only enters the hash when it differs from its fixed default, so
  every hash computed before either axis existed is unchanged). Touching any of these
  invalidates all ~60 cached pulses and forces a multi-hour retrain. This is why
  `BASELINE["disc"]` stays at 0.5 even though `disc` is provably inert and is no longer a sweep
  axis — setting it to 0.0 would be a numerically null change that costs 10.6 h. `SNAPSHOT_ITERS`
  is deliberately *outside* the hash for the same reason, and `two_phase` runs skip snapshots
  entirely (`snapshot_iters=None` on both phases) rather than trying to map the single-run
  screening concept onto two separate short runs.
- **`disc` and `amp` are retired axes, not forgotten ones.** Both are inert under `FIXED` (see
  the note above `PENALTY_NAMES`). `penalty_viz.AXIS_NAMES` is deliberately a *superset* of
  `penalty_sweep.AXIS_NAMES` so historical CSVs still plot and grid rows are still classified
  correctly — do not "sync" the two lists.
- **The noise floor for comparing two configs is ~1e-3, not ~1.5e-4.** The retired `disc` ladder
  is a calibrated null: a dynamically negligible perturbation that still moves held-out fidelity
  by 1.03e-3 by pushing L-BFGS-B into a different basin. Only the `deriv` axis clears it
  decisively. Do not report a sub-1e-3 fidelity difference as a ranking.

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
  Penalty terms (`derivative_penalty`, `boundary_penalty`, `amplitude_penalty`) and the
  lower-level `optimize_controls` also live here.
- **`core/cat_code.py`** — cat-state construction and per-gate `get_*_state_pairs()` factories
  (encode, decode, X, Y, Z, H, T, I) that `optimizer.py` consumes; `validate_pulse_truncations`
  implements the Heeres truncation-convergence check.
- **`core/fourier_cutoff.py`** — `project_bandlimit` implements the hard frequency-band
  constraint as an orthogonal projection (`IFFT ∘ mask ∘ FFT`), applied identically to pulses
  and gradients since the projection is idempotent/self-adjoint. Enabled via `cav_band`/
  `tra_band` kwargs on `optimize_multi_state_pulse`; `None` disables it.
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
  exploit the Hilbert-space truncation wall).
- **`figures/`, `tables/`, `wigner/`, `results/`, `logs/`** — generated outputs (figures, CSV
  summaries, campaign metadata). Do not hand-edit; regenerate via the corresponding script.

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
  specific; everything below it already takes `gate` as an argument.
- **`EST/subspace_evolution.py`** — Fig. 1a. Per Bloch cardinal, plots `<n>(t)` of the
  evolved code state, of the evolved error state, and of the normalized photon-loss image
  `a|psi_C(t)>` that the error state must match under transparency. Propagates only the
  four basis columns and rebuilds every cardinal by linearity (`cardinal_states`), reusing
  `diagnostics.propagate_states`. **`<n>` is exactly conserved under the drift** (`[H0, n] = 0`),
  so it is blind to App. A's obstruction — agreement is necessary, not sufficient, which is
  why the figure also carries `map_mismatch`, a phase-insensitive `U_L`-vs-`U_E` distance on
  the **fixed** code/error bases. That metric's normalization is this module's choice, so its
  EsT:Ord ratio is meaningful and its absolute value is not (same caveat as Eqs. 6-8).
- **`EST/test_grape_jax.py`** — 32 tests. The anchor is the C1 gradient checked against
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
- Eqs. 6-8 in `EST/diagnostics.py` are **reconstructed** from the transparency conditions, not
  transcribed. `L` and `eta` are bounded in [0,1] by construction; `Delta_QEC`'s normalization
  is a choice, so EsT-vs-Ord ratios are meaningful but its absolute value is not.
- After changing anything in `EST/`, run `python EST/test_grape_jax.py` before training —
  a training run is ~1 h per variant.
