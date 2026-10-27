# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Six related tracks sharing one repository.

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
`six_tones` drive), plus the Eq. (B2) rate. `AQEC/wigner.py` is a closed-form Wigner function. **`collapse_operators`
defaults to every channel** (decision D3): decay, thermal + drive-induced heating (one up/down ratio r on g→e and e→f,
0.77 ms⁻¹ on g→e from Table I + App. D's Δγ↑ = 0.42), storage heating, and pure dephasing. Storage φ_a is the
**residual** 1/T2a − κ_a/2 − γ↑_th = 0.23 ms⁻¹, because frame (ii)'s thermal jumps already dephase the storage through χ
(DGRAPE's opt-in φ_a is the full 0.58, so it double counts with heating on). `heating="none", dephasing=False` gives
the old decay-only set, which DGRAPE's decay set matches (tested). Rates: `me.jump_rates()`. Notebook §10 reproduces
all of PRX Fig. 2; populations sit an n-dependent +0.02 to +0.07 above the paper's fit at 15 μs (decay only: +0.04 to
+0.09) and the (2,4) coherence factor 0.13 above, with (0,2), (0,4) within 0.02 (documented in §10.6, not tuned away).
Tests: `python3 validation/test_aqec_master_equation.py -v`
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
P_F source per step, ~30% extra), and train.py refuses any training n_c <= forbid_fock_min. A **transmon-excitation
penalty** (`--lambda-transmon`, recipe key `transmon`, **default 0 = off**) is the same time sum with
P_T = |e><e| + |f><f|. The fidelity already pays the exact T1 price, about dt/(2·T1_ge) = 2e-5, so a positive weight
charges extra, standing in for the untrained dephasing and heating. Both occupation costs ride ONE extra costate
with the source diag(λ_F P_F + λ_T P_T) (`fidelity_and_grad(..., forbid_mask=(K,d) stack, occ_weights=...)`),
so the transmon term costs nothing extra per evaluation; terms and reports carry `C_forbid` and `C_transmon`
separately. **First X run,
not usable**: X at 1.1 us on the old [10, 12] with `--higher-order` (`pulses/dgrape/*X_open_d1*`, notebook §10) reports
F_pro 0.951/0.937 but converges to 0.9154 at n_c >= 16, because the pulse reaches Fock 10+ (C_forbid ≈ 36, i.e. a
0.036 cost the new term would have charged). **[10, 12] is falsified**, and lambda_disc 0.5 is far too weak to catch
it; the readout is inert, so n_r=1 is the next speed-up (open decisions in notebook §13). **Second X run**
(`pulses/dgrape/*X_open_d2*`, notebook §11): same command at the new defaults ([12, 14, 16], lambda_forbid 1e-3,
`--n-jobs 9`, 10.3 h). Converged in truncation (n_c 16/18/20 agree to 1e-6) at **F_pro 0.9556**, C_forbid 10; the
n_c=12 entry reads 1.7e-3 high. Coherent error 0.9%, dissipation 3.6%. Still clipped (1.6% of x pinned at 25), and it
plateaued after ~500 iterations. The higher-order terms (Kerr, chi'_q) cost ~0.7% over a 1.1 us idle: train with them on. Duration
and dt are required arguments of `DGRAPE/train.py`, which saves to `pulses/dgrape/`, never `pulses/`. L-BFGS-B needs
tight `gtol` (default 1e-12): from a random start every gradient entry is ~1e-6 and scipy's default stops at
iteration 0. **Third runs (notebook §12)**: the d2 command at a single `--n-c 16` (3 jobs, 6.8 h each, run side by side), without
(`X_open_d3a`) and with `--lambda-transmon 1e-4` (`X_open_d3b`, 5x the T1 price). **One truncation is enough**: A
scores 0.9550 against d2's 0.9556, n_c 16/18/20 agree to 3e-6, and it retraces d2's pulse (same seed); not yet
the default (one seed). **The penalty buys T1, not dephasing**: B cuts mean P_e+P_f 0.58 -> 0.42 and excess decay
by 32%, giving **F_pro 0.9589** (all channels 0.9479), the best X gate so far, with 0.45% pinned. Dephasing stays
~1.05% for every pulse (storage 0.65%, transmon 0.40%); heating is 0.03%. B had not plateaued (cost -10.5% over
its last 250 iterations): resume with `--init-x`. Next: train with `--dephasing`.
Tests: `python3 validation/test_dgrape.py -v` (34 tests, ~4 min).

**5. FWM GRAPE (`DGRAPE/fwm.py`, `fwm_grape.ipynb`).** Adds PReSPA pump tones (by default the six of Shirol, two combs, n = 1, 3, 5) as
GRAPE controls β_k(t) next to DGRAPE's linear drives, on the same Lindblad model. The tones run on a CF4 sub-stepped
propagator (`S` sub-steps per 2 ns step); `fwm.fidelity_and_grad` is the W-form exact gradient in β (and u, ready
for Stage 3), checked against an augmented-generator derivative and finite differences. Recipe constants live in
the module: per-tone cap `BETA_CAP` 0.1 and envelope band ±2 MHz, both set by the Stark limit (the comb-1 line moves by
2π·538 MHz·|ξ|², so a fixed carrier stays resonant only up to |ξ| ≈ 0.06√n). An opt-in pump-mixing a†r† term
(`mixing=1|2`, a convention factor) is used only in held-out scoring. Scripts: `DGRAPE/fwm_stage2_scan.py`
(forward-only start scan + time-resolved loss cost), `DGRAPE/train_fwm.py` (tones-only training, d2 frozen; writes
to `~/fwm_runs`, copy results to `pulses/dgrape/`), and `DGRAPE/fwm_score.py` (16 held-out conditions →
`tables/fwm_stage2_heldout.csv`). **Stage 2 result (2026-10-12): null, as predicted beforehand.** Four starts
(`pulses/dgrape/*X_fwm_s2_{a,b,c,d}*`, 300 iterations, ~6.5 h each at n_c 16, S 2, 3 jobs) all collapse to nearly
zero tones (max|β| 0.003, comb 1 off) at F_pro 0.955561 vs 0.955554 tones off. The +6e-6 is a Stark retuning, not
a repair (0 with `stark_off`; loss cost unchanged). **The tone set is a parameter**: `paths` (teeth n per comb) on
`fwm.fwm_terms`, `Stage2Objective`, `score`, and `--paths` on the three scripts; the default `dev.CORRECTED_FOCK` =
(1, 3, 5) is the six tones above and reproduces every Stage 2 number bit for bit. "Six PReSPA tones" in this paragraph
and in notebook §2–§6 means that default. `fwm.FULL_PATHS` = 1..9 is the **full comb** (18 tones, 9 -> 10 left out because
Fock 10 is `forbid_fock_min`), with band ±1 MHz (`BETA_BAND_FULL`; `default_band(paths)` picks ±2 only for (1, 3, 5)),
because the teeth are χ_gf = 2.07 MHz apart; `fwm.beta_table2(paths)` extends Table II. Cost at the cap: 138 s per
evaluation with 18 tones vs 62 s with six (Taylor m 46 vs 21; tones off 48 s for both). **Full-comb Stage 2 (notebook §7,
2026-10-12): also null.** Runs `pulses/dgrape/*X_fwm_s2full_{a,b,c,d}*` (`--paths 1 2 3 4 5 6 7 8 9 --band -1 1`, 300
iterations, ~7.3 h each) reach at best F_pro 0.955576, +2.2e-5 over tones off; tones stay ≤ 0.006, comb 1 on. The gain is
coherent (+3.7e-5 closed system, survives `stark_off`, halves at g/√2), not a repair (photon-loss cost unchanged).
Tables `fwm_stage2_full_scan.csv`, `fwm_stage2_full_heldout.csv`. Stage 3 (joint u + β) is not run; notebook §8 says
to run an idle-memory positive control first, with the six tones. Tests: `python validation/test_fwm_grape.py -v`
(15 tests, ~1.5 min).

**6. FWM tails (`DGRAPE/fwm_tail.py`, `FWM_tails.ipynb`).** The d2 X gate runs unchanged, then a **tail** of the two
FWM combs only (linear drives off) acts on its mixed output (`fwm_tail.d2_output`: d2 at n_r = 1, embedded), scored
against the ideal X at the end of the tail. Each comb is ONE complex waveform η_c(t) (control (N, 4) = η_1 I/Q, η_2
I/Q, η = I − iQ), in a frame that rotates **only |f⟩** at Δ_ref = the comb-1 tooth 2 → 3 (n_ref = 3, 2π·140.485 MHz):
ξ_1 = η_1 e^{+iΔ_ref t}, ξ_2 = η_2 e^{−iΔ_ref t}. With the transmon drive off this frame is exact (jump ops commute
with P_f or gain a global phase; targets have the transmon in g), so `fwm.propagate` with `scheme="midpoint", S=1`
and the carriers zeroed is exact: one exponential per step, no CF4. `fwm_tail` reuses `fwm.py` unchanged (W-form
gradient, `BetaChain` with 2 tones = one (N, 4) block, penalties). Decisions (2026-10-19): baseline **d2 alone**
(0.955554); combs only (the cavity drive would keep the frame exact; the transmon drive would not); |η_c| ≤ 0.1 per
comb; readout traced (`readout="empty"` available); tail **1 μs default** (3 μs until 2026-10-20), **a parameter**; **no free rotation** (the
storage Kerr phase, 5.1 % over 3 μs closed, must be cancelled by the tail). Upper limits recorded before training:
ideal instant FWM repair at T_g 0.9652, any recovery 0.9897, kitten limit 0.935 per repaired loss, d2 + idle
3 μs 0.8488; the optimistic estimate is ≈ −0.1 % vs d2 alone at 3 μs and +0.2–0.3 % at 0.5–1 μs.
Training: `DGRAPE/train_fwm_tail.py --start tableII --tag X_tail_a` (`--dry-run`: one timed evaluation,
~4 s at n_c 16, 1 μs, 3 jobs; ~12 s at 3 μs); outputs to `~/fwm_runs`, then `pulses/dgrape/` (never `pulses/`). Every scorer
takes the gate as `pulse=` (`fwm_tail.gate_output`; `--gate-pulse` on the trainer; None = d2, bit for bit).
`gate_n_c` (`--gate-n-c`; `fwm_tail.tail_input`, `cut_storage`) runs the gate at that truncation and cuts its output
to the tail's n_c; the closed-trained gates end with ≤ 2e-5 above Fock 16, so tails after them train at n_c 16 on the
n_c 24 output (cut F = uncut to 3e-16, ~3x faster). Held-out scorer: `DGRAPE/fwm_tail_score.py` (14 conditions, refs
gate alone / idle / Table II teeth per condition) → `tables/fwm_tail_heldout.csv`. **On Windows set
`OPENBLAS_NUM_THREADS=2` at launch** (numpy is OpenBLAS there; loky otherwise gives each worker 8 threads).
**Trained tails (notebook §6, 2026-10-21/22): no-go.** 15 runs `pulses/dgrape/*X_tail_{est,ord,d2}_T{05,1,2}_{t2,z}*`
(EsT/Ord/d2 × 0.5/1/2 μs, starts Table II and zero + noise, 300 it), the 8 still climbing resumed 300 more (`*_r*`).
Only EsT + 1 μs beats its gate alone: **+0.16 %** (0.96966 vs 0.96803; go bar +0.3 %); Ord −0.58 %, d2 −0.67 %
with the same tail shape. Real repair (0 closed; loss cost 2.22 → 1.81 % after EsT), paid for by transmon decay
(~0.2 %), all channels −0.11 %. The optimiser builds ONE ~600 ns repair shot (comb 1, then comb 2 at the cap), so
0.5 μs stalls near idle; 1 μs is best for every gate. The shot sits at the END of the tail because it repairs only
losses before it (to ~0.92) and none after: sliding it later gains 1.8 %/μs, 0 closed (§6.3, Figure 8,
`tables/fwm_tail_shot_timing.csv`); the unrepaired end of the shot (~490 ns weighted, ~1.0 %) is the floor. Fragile: η × 0.9 / 1.1 costs the EsT tail 8–10 %, Stark off
13–63 %, readout-empty scoring removes the gain. Estimate vs result (§6.5, Table 13): of the +1.09 % → +0.16 % gap,
the slow shot is ~−0.4 % and transmon decay ~−0.2 %. **Re-timed EsT tails (notebook §6.8, 2026-10-26): no-go.**
0.8 μs and 1.4 μs (`pulses/dgrape/*X_tail_est_T{08,14}_crop*`, started from the 1 μs / 2 μs tail with its front cut,
600 it): +0.19 % and +0.05 % over the gate alone (1 μs +0.16 %); 0.8 μs is twice as pump-sensitive (η × 0.9 −13.7 %).
The 1 μs tail's quiet front is not idle (zeroing it costs 2 % closed: Kerr-phase work); every tail ends in the same
~600 ns shot and leaves ~450–620 ns (~1 %) unrepaired. Tables `fwm_tail_heldout_retimed.csv`,
`fwm_tail_retimed_crop_check.csv`, `fwm_tail_shot_timing_retimed.csv`. Next: a faster shot (comb 2 above all) is the
biggest lever, not a shorter wait; robust training over pump scale; a gate trained for repairable losses.
**EsT gate (notebook §3, 2026-10-20).** `DGRAPE/est_gate.py` trains an error-semitransparent X gate with EST's cost
unchanged (`EST/grape_eigh.cost_and_grad`, C1–C4, weights (1, 0.7, 7, 1) then (1, 0.1, 0, 1), 2000 it per stage,
pinned to `EST/train_est.py`) in the CLOSED system on this chip (DGRAPE drift and controls at n_r = 1, n_c 20), with
d2's duration, dt, bands, ramp, cap, box and seed-42 smooth start; the three transparency measures are
`EST/diagnostics`' unchanged. `DGRAPE/train_est_gate.py --variant est|ord` (~30 min each, 2 side by side) →
`pulses/dgrape/*X_{est,ord}_closed*` (+ `_s43`, `_s44` seeds). **The closed-trained gates reach Fock 15–17 and need
n_c ≥ 20** (closed F1 at n_c 16 reads 0.006 / 0.016 low), so §3 scores at n_c 24 (28 held out). EsT: F1 0.99782
(just under EST's 0.999 "closed"), F_ET 0.726 (Ord 0.331); mean L 0.27 vs d2 0.59, L(T) 0.13 vs 0.93; Δ_QEC barely
moves (0.20 vs 0.24). Open: EsT alone 0.96803 > Ord 0.96380 > d2 0.95556; a mid-gate loss repairs ideally to 0.861 (d2
0.461), the ideal repair gains +1.85 % (d2 +0.96 %), §2.4's estimate for a 1 μs tail becomes +1.1 % (3 μs: +0.8 %). With d2's
gentle start stage 1 parks or nearly parks: seed 44 parked (zero drive), seeds 42/43 stall at F1 ≈ 0.49 in stage 1
and stage 2 closes from there (seed 43: F1 0.9985, F_ET 0.774, gain +1.96 %). The R-W best-recovery iteration stops
3e-4 low at n_c 24 (d2 0.98938 vs 0.98968 at 16): a lower bound.
Notebook order (rewritten 2026-10-25): Setup (glossary), §1 idea, §2 limits after d2, §3 EsT comparison, §4 method
(frame, controls, cost, baselines), §5 training as run, §6 results (6.5 = estimate vs result), §7 conclusions,
Appendix A checks and timing, Appendix B dated decisions, run logistics, seed 43. Every figure and table has a
markdown caption below it with a *Symbols* line; table gains are shown in %.
Tests: `python3 validation/test_fwm_tail.py -v` (13 tests, ~90 s on the Windows PC),
`python3 validation/test_est_gate.py -v` (7 tests, ~1 s).

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

### iCloud sync

The repo lives in `~/Documents`, which iCloud ("Desktop & Documents") syncs between the user's
devices, on purpose. iCloud does not understand git. When two writes to one file collide, it keeps
both and renames one to `<name> 2.<ext>` (or `<name> 2`). **Found 2026-10-19:** `.git/index` was
gone, replaced by `.git/index 2` and `.git/index 3`. With no index, git reported all 1230 tracked
files as staged-deleted and the same files as untracked, which looks like data loss but is not. There
were also 445 byte-identical `" 2"` copies in `logs/`, `pulses/est/`, `pulses/staging_umax25/`
(since renamed `pulses/archive_umax25_seeds/`), `pulses/archive_umax40/` and `tables/`.

- **Check:** a `git status` full of `D` plus `??` for the same paths, `ls .git/index*`, and
  `find . -path ./.git/objects -prune -o \( -name '* [0-9]' -o -name '* [0-9].*' \) -print`.
  Use exactly that pattern. A looser `'* [0-9]*'` also matches real names such as
  `sources/Gertler et al. - 2021 - ….pdf`, and since that name has no original to compare with,
  a `cmp` check passes against itself.
- **Fix the index:** `git reset` (mixed; it rebuilds the index from HEAD and leaves every working
  file alone), then confirm `git status` shows only real edits, then remove the `index N` copies.
- **Fix `" 2"` copies:** `cmp` each against its original first. Delete only identical ones. A copy
  that differs may be another device's newer version, so read both before choosing.
- **Avoid it:** let iCloud finish syncing before working on another device, never run git or a
  training job on two devices at once, and do not commit `" 2"` files (`.gitignore` ignores them).
  Training runs that write many files (`pulses/est/`, `logs/`) are the usual trigger.

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
`sys.path` explicitly, e.g. `analysis/pulse_analysis.py`).

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

- **Write in plain language** in notebooks, READMEs, docstrings and explanations: short sentences, everyday words,
  and define any jargon on first use.

- Physical parameters (`chi`, `Kerr`, `chip`, `alpha`, `dt`) are defined in lab units (MHz, μs)
  at the top of `core/grape_core.py` and converted to rad/μs via `two_pi = 2*np.pi`. Follow this
  convention rather than hand-converting elsewhere.
- Fidelity/gradient function signatures in `grape_core.py` are considered a stable interface
  (`fidelity_grad`, `fidelity_multi_state` are thin wrappers over `_fidelity_core`); if you
  change `_fidelity_core`, `test_grape_core_perf.py` must still pass its pre-refactor
  equivalence and finite-difference checks.
- When adding a new logical gate, follow the pattern in `core/cat_code.py`
  (`get_logical_*_state_pairs`) and register it in `main.py`'s `GATE_FACTORIES` and
  `validation/validate_logical_gates.py`'s `GATE_PULSE_MAP`, rather than special-casing it inside
  `optimizer.py`.

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
