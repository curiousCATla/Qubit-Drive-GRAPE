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
`validation/`, or `analysis/`. See `EST/CLAUDE.md` and `EST/README.md`.

Full physics/math background for track 1 (Hamiltonian, cat-code definition, adjoint gradients,
frequency-band-limited controls, penalty terms, truncation-convergence protocol, decoherence
model) is documented in `README.md` — read it before making non-trivial changes to
`core/grape_core.py`, `core/optimizer.py`, or `core/cat_code.py`; it is not repeated here.
Track 2's equivalent is `EST/README.md`.

## Setup

```bash
pip install -r requirements.txt
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

Penalty-weight sweep (`analysis/penalty_sweep.py`, `penalty_optimization.ipynb`, `tables/penalty_sweep_*`,
`results/penalty_sweep_cache/`): commands, the two-phase / `--phase` protocols, cache-hash rules, the
noise floor, the selection rule and the pre-image caveats live in the `penalty-sweep` skill
(`.claude/skills/penalty-sweep/SKILL.md`). Read it before running, merging, or quoting anything from
the sweep. Two rules apply even without it: **always pass `--tag`** unless you mean to overwrite the
frozen pre-ramp `tables/penalty_sweep_X_ofat.csv`, and **never change `FIXED`, `BASELINE`, or anything
else `_config_hash` covers, and never delete `results/penalty_sweep_cache/`** — either invalidates the
cached pulses and forces a multi-hour retrain.

EsT module (`EST/`): commands, per-module notes and conventions live in `EST/CLAUDE.md`, which loads
when working with files under `EST/`. Read it explicitly before EsT work that starts elsewhere
(`est_experiments.ipynb`, `est_optimization.ipynb`, `pulses/est/`, `logs/est_*`).

All scripts assume they are run from the repository root (several insert `REPO_ROOT` onto
`sys.path` explicitly, e.g. `analysis/logical_gate_analysis1.py`).

## Architecture

- **`core/` module notes** (`grape_core.py`, `fourier_cutoff.py`, `ramp.py`, `compare_pulses.py`,
  `propagator.py`) live in `core/CLAUDE.md`, which loads when working under `core/`.
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
- **`QuTip/`** — fully independent re-implementation of the physics (operators, Hamiltonians,
  propagation via `qutip.sesolve`) used to cross-check `core/grape_core.py`'s hand-rolled
  eigendecomposition propagator; deliberately does not import from `core/`.
- **`analysis/`** — one-off/ad-hoc optimization and comparison scripts built on `core/` and
  `pulses/` (e.g. `decoherence.py` for Lindblad decoherence-limited fidelity, `pulse_analysis.py`
  for trajectory/Fock-population inspection). Not a stable API — read the specific script before
  reusing it.
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
- **Which `tables/` are post-ramp and which are not** is recorded in `tables/CLAUDE.md` (loads when
  working under `tables/`). Check mtimes before quoting a number from anything in `tables/`.

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
- Everything else EsT-specific (module notes, cost-function deviations, metric conventions) is in
  `EST/CLAUDE.md`.
