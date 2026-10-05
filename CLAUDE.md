# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Four related tracks sharing one repository.

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

**3. Passive AQEC (`AQEC/`, `aqec_experiments.ipynb`).** Write-up of the PReSPA tone combs of
Shirol et al., PRX 16, 021042 (2026) on the odd-parity n̄=3 code `(|1>+|5>)/√2, |3>`. It uses a
**third chip**, Shirol's Table I (storage/transmon/readout, `AQEC/device.py`), not the EsT or Heeres
device; the EsT Table I lacks χ_ef, κ_r and T1_ef. Sign convention differs from `EST/device.py`:
Shirol's χ and α are stored **positive** with explicit minus signs in H0. `AQEC/master_equation.py` holds both
frames of notebook §8.2 (pure numpy/scipy): frame (i) = Eq. (3) as a static Liouvillian (populations), frame (ii) =
dispersive terms explicit, time-dependent drive, `solve_ivp` (anything with phases between photon numbers; also the
`six_tones` drive), plus the Eq. (B2) rate. `AQEC/wigner.py` is a closed-form Wigner function. Notebook §10 reproduces
all of PRX Fig. 2; populations sit an n-dependent +0.04 to +0.09 above the paper's fit at 15 μs and the (2,4) coherence
factor 0.16 above (documented in §10.6, not tuned away). Tests: `python3 validation/test_aqec_master_equation.py -v`
(~1 min; the qutip frame-(ii) check is limited to 2 μs because qutip's python coefficients are slow) and
`python3 validation/test_aqec_wigner.py -v`.

**4. Exact open-system GRAPE (`DGRAPE/`, `dissipation_grape.ipynb`).** GRAPE whose cost is the process fidelity
of a logical gate on the Shirol odd kitten code, computed from the **exact Lindblad evolution of density matrices**
on the AQEC chip (imports `AQEC/device.py`, `AQEC/master_equation.py` helpers, `AQEC/odd_kitten.py`; same positive-χ/α
sign convention). Frame = rotating at the linear mode frequencies only (AQEC frame (ii) + `-alpha|f><f|`), so the jump
operators are exact. Space = storage × transmon (g,e,f) × readout (m=0,1), d = 6·n_c; the readout is inert during a gate
(n_r=2 equals n_r=1, tested). Jump operators default to **decay only** (κ_a a, κ_r r, split transmon T1);
dephasing/heating are opt-in flags. Step map = matrix-level Taylor action of exp(L_k Δt), never the d²×d²
superoperator; (s, m) from a norm bound at tol 1e-15. Three exact gradients of the SAME polynomial agree to ~1e-13:
`lindblad_np.fidelity_and_grad` (hand-derived adjoint, one W matrix per step, ~3 forward passes, the default and the
fastest), `fidelity_and_grad_augmented` (block augmented generator, reference) and `lindblad_jax` (autodiff,
`jax.checkpoint`ed scan). `tables/dgrape_gradient_benchmark.csv` holds the comparison. The training objective
(`DGRAPE/objective.py`, notebook §8) is the `experiments.ipynb` production cost with F_coh replaced by the Lindblad
F_pro: band filter → ramp chain, derivative, modulus amplitude and Eq. 24 discrepancy penalties, pre-image box, smooth
seeded cold start / `init_x` / `deramp` warm start, best-vs-final, `constraint_report`. Its **defaults are the
Heeres-chip recipe** via `DGRAPE/recipe.py`, pinned to `main.build_arg_parser()` by a test; change the production
recipe and that test fails, by design. Only the fidelity is backend-specific; penalties and the chain rule are numpy
for both backends. The default truncation list is **[12, 14, 16]** (the code needs only Fock 5; the cost is ∝
(6·n_c)³), deliberately NOT the Heeres `[22, 24, 26]` and not pinned to `main.py`. The truncations run as joblib
jobs (`n_jobs`, default 3 = `main.py --n-jobs`, pinned; loky for np, threading for jax). One job is single-threaded
(Accelerate does not thread d~100 products), so with np `n_jobs` > #truncations splits the largest truncations by
process input (E_00/E_11/E_01 add up; `_split_plan`). Per evaluation at N=200 on [12,14,16]: 1 job 21.6 s, 3 jobs 11.4 s,
5 jobs 9.1 s, 7 jobs 7.7 s, 9 jobs 6.9 s (pass `--n-jobs 9`; 4 P-cores + 6 E-cores here). A DGRAPE-only **forbidden-state penalty** (Heeres Supp.), `lambda_forbid` 1e-3 ×
mean_i Σ_{k=1..N} Σ_{b=0_L,1_L} Tr[P_F ρ_b(k)], P_F = storage Fock >= `forbid_fock_min` = 10, is a RAW sum over steps
(rescale λ if N changes a lot); its exact gradient shares the fidelity's backward pass (extra costate entries with a
P_F source per step, ~30% extra), and train.py refuses any training n_c <= forbid_fock_min. **First X run,
not usable**: X at 1.1 us on the old [10, 12] with `--higher-order` (`pulses/dgrape/*X_open_d1*`, notebook §10) reports
F_pro 0.951/0.937 but converges to 0.9154 at n_c >= 16, because the pulse reaches Fock 10+ (C_forbid ≈ 36, i.e. a
0.036 cost the new term would have charged). **[10, 12] is falsified**, and lambda_disc 0.5 is far too weak to catch
it; the readout is inert, so n_r=1 is the next speed-up (open decisions in notebook §12). **Second X run**
(`pulses/dgrape/*X_open_d2*`, notebook §11): same command at the new defaults ([12, 14, 16], lambda_forbid 1e-3,
`--n-jobs 9`, 10.3 h). Converged in truncation (n_c 16/18/20 agree to 1e-6) at **F_pro 0.9556**, C_forbid 10; the
n_c=12 entry reads 1.7e-3 high. Coherent error 0.9%, dissipation 3.6%. Still clipped (1.6% of x pinned at 25), and it
plateaued after ~500 iterations. The higher-order terms (Kerr, chi'_q) cost ~0.7% over a 1.1 us idle: train with them on. Duration
and dt are required arguments of `DGRAPE/train.py`, which saves to `pulses/dgrape/`, never `pulses/`. L-BFGS-B needs
tight `gtol` (default 1e-12): from a random start every gradient entry is ~1e-6 and scipy's default stops at
iteration 0. Tests: `python3 validation/test_dgrape.py -v` (29 tests, ~3 min).

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
#   --ramp-ns 48.0  --hard-amp-limit 25.0  --amp-max 25.0  --amp-norm modulus
#   (amp_max caps |I+iQ| per drive, Heeres Supp. Eq. 19; the box is per element on x)
#   --lambda-deriv 1e-5  --lambda-amp 8e-5  --lambda-disc 0.5
#   --cav-band -27 27  --tra-band -33 33  --fidelity-fn auto (=> coherent, every gate)
python main.py --gate X

# Seeds are per gate under u_max=25 (the box binds): enc 47, dec 43, X 43,
# Y 46, Z/H/T/I 42. Y at seed 42 pins 4.86% of the pre-image and collapses to
# F_ped = 0.8986; enc is clipped at every seed and 47 is kept deliberately.
python main.py --gate Y --seed 46
python main.py --gate enc --seed 47

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

The sweep's live tag is now **`amp25`** (`tables/penalty_sweep_X_ofat_amp25.csv`): `FIXED` was moved
onto the production amplitude recipe (`amp_max` = `hard_amp_limit` = 25, `amp_norm="modulus"`) and the
`amp_max` ladder densified to 9 rungs, so all 20 configs × 10 seeds were retrained from scratch. The
u_max=40 tables (`tables/penalty_sweep_X_ofat_ramp*.csv`) are now a **second frozen regime** alongside
the pre-ramp ones — same three axes, different amplitude settings, different pulses. Never concatenate
across regimes. `penalty_optimization.ipynb` §1–§10 has been rewritten against the new campaign;
§11–§15 remain the frozen pre-ramp record. Two artifacts are **overwritten in place** and now hold
u_max=25 data despite their names: `tables/penalty_sweep_summary_ramp.csv` and
`figures/penalty_pareto.*` (the latter is still `\includegraphics`'d by
`penalty_optimization_report.tex`, whose prose remains u_max=40 at a 70% budget — a known, unfixed
inconsistency).

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
- **`hard_amp_limit`, not the envelope, is what bounds the endpoint amplitude — and at the
  production `u_max = 25` it DOES bind.** The ramp cuts `max|u[0]|/peak|u|` on every operation
  (typically ~3x, up to 23x: Z 3.64%->0.16%, T 4.98%->0.47%, I 3.87%->0.34%, opt 6.06%->1.13%),
  but it does not reach zero,
  and how close it gets tracks how amplitude-hungry the gate is (X only 4.75%->1.29%). The
  reason: the envelope removes control authority over the first/last 24 steps and L-BFGS-B buys
  it back by inflating the *pre-image* there — on `u_X_main`, `|P(x)[0]| = 20.2` against a
  mid-pulse RMS of 6.22 (3x), so `u[0] = 0.0135 * 20.2 = 0.27` survives. What stops that
  inflation is the box on the raw variable, so raising `hard_amp_limit` would quietly undo part
  of the ramp. **Mind which default you are getting**: `optimize_multi_state_pulse` defaults to
  `hard_amp_limit=50.0`, `refine_pulse`/`refine_pulse_dt`/`refine_pulse_dt_light` to 40.0,
  and `analysis/penalty_sweep.py` `FIXED` and the production paths (`main.py`, the notebook
  recipe) all pin **25.0** — the sweep used to sit at 40.0 to protect its cache and no longer
  does. All the numbers below are against 25.
  **The box binds at 25, and seeds are therefore per gate** (`GATE_SEEDS` in the notebook;
  `--seed` on `main.py`): enc 47, dec 43, X 43, Y 46, and Z/H/T/I 42.
  `u_enc_main` sits at exactly `max|x| = 25.00` with 0.41% of entries pinned at *every* seed
  42-50 — accepted deliberately, since seed 47 is the best encode pulse to date (held-out
  0.999202 vs 0.998810) — and `u_Y_main` at 25.00 with 0.045% pinned.
  **Y remains the cautionary case**: at seed 42 it pinned 4.86% of entries and collapsed to
  `F_ped = 0.8986` (n_c=24); seed 45 likewise (0.8355, and `|eps| = 25.98` broke the amplitude
  limit too). Seed 46 is what is on disk. Check **both** numbers before trusting a retrained
  pulse: `info['max_drive_modulus']` against `amp_max` (the physical Eq. 19 constraint, which
  the penalty enforces) and `info['max_abs_preimage']` against `hard_amp_limit` (the numerical
  box; at the bound a pulse has been clipped, not converged — read its held-out fidelity).
  For reference X sits at 23.58 and Z at 22.71, both clear.
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
