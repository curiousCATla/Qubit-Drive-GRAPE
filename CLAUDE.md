# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Seven related tracks sharing one repository.

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

**3. Passive AQEC (`AQEC/`, `notebooks/aqec_experiments.ipynb`).** Write-up of the PReSPA tone combs of
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

**4. Exact open-system GRAPE (`DGRAPE/`, `notebooks/dissipation_grape.ipynb`).** GRAPE whose cost is the process fidelity
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
(`DGRAPE/objective.py`, notebook §8) is the `notebooks/grape_experiments.ipynb` production cost with F_coh replaced by the Lindblad
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

**5. FWM GRAPE (`DGRAPE/fwm.py`, `notebooks/fwm_grape.ipynb`).** Adds PReSPA pump tones (by default the six of Shirol, two combs, n = 1, 3, 5) as
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

**6. FWM tails (`DGRAPE/fwm_tail.py`, `notebooks/FWM_tails.ipynb`).** The d2 X gate runs unchanged, then a **tail** of the two
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
**The notebook's EsT gate is the "good" one (2026-11-01, user decision):** seed 24 of the C_err cost
(`u_X_est_err_closed_s24.npy`, see "Seed selection" below) is "EsT" in §1–§6, §8–§9 and Appendices B–C (the "X gate" in §7) (`EST_PULSE` in Setup);
its tails are `pulses/dgrape/*X_tail_estg_T{05,08,1,14,2}_{t2,z}*` (600 it, one leg) and its held-out scores
`tables/fwm_tail_heldout_good.csv` (gate `EsT-good`, renamed `EsT` on load). The earlier seed-42 "mid" gate
(`u_X_est_closed.npy`, tails `X_tail_est_*`) is no longer in the notebook (Appendix D removed 2026-11-09). §6 merges d2 and Ord rows of
`fwm_tail_heldout.csv` with the good table; the re-timed (cropped) mid tails and their crop check (old Table C8) are
dropped; `tables/fwm_tail_runs.csv` now holds the good rows. **Trained tails (notebook §6): one GO, at the bar.**
19 runs: d2/Ord × 0.5/1/2 μs (300 it; 5 resumed `*_r*`), EsT × 0.5/0.8/1/1.4/2 μs. EsT + 0.8 μs **+0.32 %** (0.97027
vs 0.96706, held-out n_c 28 the same; clears the +0.3 % bar by the spread between its two starts), 1 μs +0.28 %, 1.4 μs
+0.14 %, 2 μs −0.06 %; Ord −0.58 %, d2 −0.67 % at best (1 μs). Real repair (0 closed; loss cost 2.38 → 1.85 % at 1 μs),
paid for by transmon decay (~0.25 %). The optimiser builds ONE ~600 ns shot (comb 1, then comb 2 at the cap) at the END
of every tail (0.5 μs stays off): it repairs losses before it (to ~0.92) and none after; sliding the whole 1 μs tail later in a 2 μs window gains
1.77 %/μs, 0 closed (§6.2, `tables/fwm_tail_shot_timing_good.csv`); the unrepaired end of the shot (458–604 ns
weighted, 0.9–1.2 %) is the floor. Estimate vs result (§6.4, Table 9, 1 μs): +1.23 % → +0.28 %, slow shot −0.41 %,
non-loss −0.25 %, remainder −0.29 %. Fragile (§6.5): 0.8 μs η × 0.9 / 1.1 −10.3 / −12.8 %, Stark off −77 %, all
channels +0.03 %, readout-empty −1.2 %. Next: a faster shot (comb 2 above all); robust training over pump scale; a gate
trained for repairable losses.
**Tails alone (notebook §6.6, 2026-11-11, user).** Every EsT tail with NO gate in front, on the code words and on
clean one-loss error words a E a'/3, scored against I, closed and open, plus one decay channel at a time
(`fwm_tail.code_word_scores`, moved there from `fwm_tail_hgate.tail_on_code`, Table 16 bit for bit;
`DGRAPE/fwm_tail_code.py`, ~2 min -> `tables/fwm_tail_code_lengths.csv`). The 50/50 mix is the exact average (linear;
tested), so it is shown as such, with the break-even loss fraction p* = (1-F_code)/(1-F_code+F_err). Closed: code
words >= 0.9998 at every length (idle loses 0.15-2.3 % to Kerr; even the 0.5 us tail cancels it), error words 0.904 /
0.922 / 0.922 / 0.908 (0.8/1/1.4/2 us; 97-99 % of the kitten limit). Open: error words 0.820 -> 0.865, p* 1.8-2.2 %
vs the gate's 2.68 % losses. Readout decay is the largest cost of a short repair (6.2 % at 0.8 us, -0.4 % at 2 us; not
yet explained). A linear estimate from these numbers (losses scaled 0.851/0.935) overshoots Table 6 by a constant
0.23-0.24 % at every length. Mid's record (first campaign 2026-10-21/22: mid 1 μs +0.16 %; cropped 0.8/1.4 μs
+0.19/+0.05 %, `X_tail_est_T{08,14}_crop*`, `fwm_tail_heldout_retimed.csv`, `fwm_tail_retimed_crop_check.csv`,
`fwm_tail_shot_timing{,_retimed}.csv`) stays on disk, unused by the notebook.
**EsT gate, first version = mid (2026-10-20; no longer in the notebook).** `DGRAPE/est_gate.py` trains an error-semitransparent X gate with EST's cost
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
**Seed selection with C_err (`notebooks/est_gate_selection.ipynb`, 2026-10-29).** Variant `est_err` of `est_gate.py` =
est_optimization §7's cost ((1,0.7,7,1,0.3,0,0) then (1,0.1,0,1,0,0,0), pinned to `EST/err_two_stage.WEIGHTS["run"]`);
`DGRAPE/est_gate_seeds.py run|status|summarize|open_score` (runs in `~/fwm_runs/est_err_seeds`, 25 single-thread jobs,
1.5 h a wave). Rule fixed before training, scored closed at n_c 24: F1 >= 0.99, peaks of Delta_QEC, L, eta <= 0.5, tails
(mean t >= 0.9 T) <= 0.1 / 0.2 / 0.2, |F1(28) - F1(24)| < 1e-4; rank by (mean Delta_QEC + mean L + mean eta)/3. Wave 1
(seeds 0-24) met the stop rule (3 candidates); wave 2 (25-49) was run anyway at the user's request, rule unchanged.
Over 50 seeds: **pick still seed 24** (`pulses/dgrape/u_X_est_err_closed_s24.npy`, score 0.110 vs the current gate's
0.203, which fails peak Delta_QEC 0.61, peak L 0.52, tail L 0.27), then seeds 45 (0.127), 38 (0.130), 16, 7; picks
copied to `pulses/dgrape/`. C_err parks 22/50 starts (plain est 1/3); seed 9 trains against the n_c 20 wall (tier
"wall"); 27 make a gate, 5 pass. The L bars bind (15 fail peak L, 13 tail L). Plain-est seed 43 also passes every bar
(0.135), so the gain is the seed's, not the cost's. Open system (top 3): F_pro alone and mid-gate repairability
unchanged (0.967-0.968; 0.84-0.85 vs 0.86), ideal-repair gain +1.95-1.99 % vs +1.85 %. More seeds will not help; a
cost on L(t) would.
**H gate at 2 us (notebook §7, 2026-11-02): pick seed 17** (`pulses/dgrape/u_H_est_err_closed_T2_s17.npy`).
`--duration-us 2` on `train_est_gate.py` and
`est_gate_seeds.py` (N 1000; `est_gate.train(duration_us=)`; tags/dirs/tables carry `_T2`: `~/fwm_runs/est_err_seeds_H_T2`,
`tables/est_gate_seeds_H_T2_*`; 1.1 us names unchanged, tested). Same est_err cost and bars; user: rank by the MEAN score.
Wave 1 (seeds 0-24): 0 parked, 0 wall, 25 gates; at §1's bars 3 candidates 7 (score 0.084), 17 (0.086), 23 (0.105).
**User then lowered peak Delta_QEC to 0.4 for this search only** (`PEAK_MAX_BY[("H", 2.0)]`, `peak_max()`, tested): only
17 passes (peak dQEC 0.36; 7 is 0.44, 23 0.4015). **Wave 2 (seeds 25-49, 2026-11-02/03, user, rule unchanged; 7.5 h,
half the per-run speed of wave 1)**: 24 gates + seed 29 wall (F1 0.9997 at n_c 20, 0.016 at 24). Over 50: 49 gates (F1
0.9982-0.9998), 6 pass at 0.5 (+27, 35, 45), **2 at 0.4: 17 (pick, rank 1) and 27 (rank 2, score 0.111, peak dQEC 0.31,
L(T) 0.09 vs 0.16, eta tail 0.16 vs 0.07)**; lowest score of all, seed 32 (0.082), fails only tail L. 0.3 check over 50:
19 gates <= 0.3, none passes (4, 5, 21 fail only eta tail, 41 only L tail). Seeds 7/17/23/27 copied to `pulses/dgrape/`.
Binding bars over 49 gates: eta tail 39, L tail 39, dQEC peak (0.4) 22. Open (`open_score --top 3 --extra-seeds 7 23`,
decay): seed 17 F_pro 0.9501 vs X pick 0.967 (losses 0.044 vs 0.027 per gate), ideal repair +3.1 %, mid-gate loss repaired
0.75 vs 0.85; **seed 27 F_pro 0.9504 but repairs better (mid-gate 0.81, +3.4 %)**; 7 and 23 within 0.0015. All runs hit
2000 it. §7.2 = chip
table (chi 3.66 vs 1.12 MHz; chi*T 3.7 EST H vs 1.2 at 1.1 us, 2.2 at 2 us). §7 = 7.1 condensed 1.1 us, 7.2 chips, 7.3-7.8
the 2 us search; the 1.1 us record is Appendix B, commands/decisions Appendix A.
**H gate at 1.1 us (now notebook §7.1 + Appendix B, 2026-11-01/02): no candidate.** `--gate H` on `train_est_gate.py` and
`est_gate_seeds.py` (X defaults and X tables unchanged, tested); runs in `~/fwm_runs/est_err_seeds_H`, tables
`tables/est_gate_seeds_H_*`. Same cost, bars, 1.1 us; user: rank by peak score = mean of the three peaks, both waves, no
H reference. 50 seeds: 28 parked, seed 9 wall, 21 gates (F1 0.990-0.999), **0 candidates**: all 21 fail peak/tail L
and peak/tail eta (eta tail 0.68-0.98 vs X median 0.10). Stage 1 keeps eta low (F1 0.24-0.83); stage 2 (C_err off)
closes the gate and eta climbs to ~0.9 over the last third. Follow-up (user): variant `est_err2` = C_err kept in stage
2 ((1,0.1,0,1,0.3)), the 21 gates resumed from their stage-1 pre-images (`--variant est_err2`, `train --start-stage 2`;
`~/fwm_runs/est_err2_seeds_H`, `tables/est_gate_seeds_H_err2_*`): eta -0.47 but F1 0.935-0.980 (all fail), Delta_QEC up,
L still fails; 0 candidates; all hit 2000 it. H on the EST chip was the most transparent gate at **1.0 us** (not 2 us, as
written here until 2026-11-02), but with chi 3.66 MHz vs 1.12 here: chi*T = 3.7 cycles there vs 1.2 here, so the short gate
in units of 1/chi is the likely limit. Next options: longer H gate (duration is a module constant; chi*T = 3.7 needs ~3.3
us), a cost on L(t)/eta(t), longer est_err2 stage 2. Done: 2 us, above.
**Good vs mid EsT tails (2026-10-31; was notebook §8, then Appendix D, removed 2026-11-09).** "mid" = the seed-42 `u_X_est_closed.npy`; "good" =
seed 24 above, which replaced mid in the rest of the notebook on 2026-11-01. Tails after both, both starts at 0.5/0.8/1/1.4/2 us, 600 it (`pulses/dgrape/*X_tail_estg_*`, new mid
`*X_tail_est_T{08,14}_{t2,z}*`; ran 8 side by side x 1 BLAS thread, +12 % throughput over 4 x 2). Held-out tables
`fwm_tail_heldout_{good,mid_extra,ord_idle_T08}.csv`. Good gains +0.12-0.21 % more at every length: **0.8 us +0.32 %
(the first GO, just at the bar; mid +0.19 %)**, absolute only +0.03 %; same waveform; more pump-fragile (eta x 0.9:
-10 % vs -5 %). **Repeated X gates (notebook §8 since 2026-11-10; was §7, §8, before that §9; no mid line since 2026-11-01)**: `DGRAPE/fwm_repeat.py` (B1 = N gates + one tail, B2 = N gate +
tail pairs, frame hand-off after each tail, teeth reference), `DGRAPE/fwm_repeat_run.py` (~4 h on 16 cores) ->
`tables/fwm_tail_repeat_B{1,2}.csv`; tails `X_tail_{estg,ord,est,idle}_T08_*` (`train_fwm_tail.py --target I` after
the zero pulse `u_idle_X_duration.npy` for idle). N = 20: one tail after 20 good EsT gates +7.3 %; 20 good pairs
0.642 vs 0.520 back to back (p 0.979 per pair vs 0.968 per gate); Ord pairs gain nothing per gate. **Memory
references use a Kerr-corrected decoding** (2026-11-01, Gertler 2021 Methods: `fwm_repeat.kerr_decode` = cavity frame
shift 2.5 K + logical Z 2 K t, free; `fwm_repeat_run.py --tails idle=X_tail_idle_T08_t2 --kerr-decode --n-jobs 8`,
~15 min -> `tables/fwm_tail_repeat_kerr.csv`, rows `decode` none|kerr; B1 = idle with/without its tail and the Table II
teeth sampled every 1.1 us, B2 = plain idle and the teeth every 1.9 us; the two teeth samplings are one waveform and
agree exactly at 20.9 us). **§8's teeth are Stark-tuned (2026-11-09, user)**: each Table II tooth moved by its mean
pump Stark shift, sum|beta|^2 (S_t - S_s) = +204 kHz comb 1, -204 kHz comb 2, against comb 1's 57 kHz drive
(`fwm_tail.teeth_stark_shifts`, `eta_from_teeth(stark_tuned=True)`, default False so §2/§3/§6 teeth are unchanged;
`fwm_repeat.teeth_memory(stark_tuned=True)` default). The kerr table also holds the untuned rows, label
`idle + Table II teeth, bare carriers` (bit for bit the old numbers); `fwm_repeat_run.py --teeth-only` (main mode or
`--every-k`) reruns only the teeth and replaces their rows in B2/B3 (B3 parts are not on the Mac). Uncorrected, idle
fell to 0.13 by N = 20 on Kerr phase alone. Decoded: idle closed 1 exactly, open 0.624 at N = 20 (2.1 %/us =
nbar/T1a); idle + one tail 0.849 (+22.5 %); tuned teeth beat plain idle from 3.3 us (0.806 at 22 us; bare 4.4 us,
0.741) and **beat the EsT-good pairs from ~9 to ~41 us** (+4.3 % at 22.8 us; 0.654 vs 0.642 at 38 us; bare 0.613),
then fall below (0.354 vs 0.442 at 76 us, B3): they keep ~91 % on Fock 1/3/5 throughout, so the late loss is in the
repaired amplitudes/phases, not leakage. Closed tuned teeth 0.804 at 22 us (bare 0.540), best phase correction +3.2 %
(bare +1.4 %): a Raman leak |n,g,0> -> |n+1,g,1> (comb-1 + comb-2 tooth n; middle state off by chi_gf, two-tone
detuning only K, which the opposite-sign tuning keeps; 14 % in |n+1,g,1> at 22 us, bare 43 %), which kappa_r turns
into weak heating (kappa_r as the only jump: 0.948, bare 0.956); checked at n_c 12, notebook §8.1/§8.2. B2 "gate + idle" cannot be
corrected (gates in between). Shown in the notebook (user, 2026-11-03): Figure 13 (B1) = gates + idle, idle + tail and
teeth, all memory Kerr decoded, panel b at 0.9-1; Figure 14 (B2) = gate + tail pairs and gates back to back, plus
memory (idle, teeth; Kerr decoded, same time) in panel a only; gate + idle not shown (rows stay in the CSV); no time
panel. **A tail every k gates (notebook §8.4, 2026-11-09)**: `fwm_repeat.every_k` (B3; after every gate scored as
is and with a closing tail on a copy), `fwm_repeat_run.py --tails EsT-good=X_tail_estg_T08_t2 --every-k 1 2 3 4 5 6 7 8
--n-max 40 --n-jobs 20 --out-dir ~/fwm_runs/repeat_B3` (3.5 h; parts in `B3_parts`, reused on rerun) ->
`tables/fwm_tail_repeat_B3.csv`. Open, 40 gates: k = 2 best from N = 20 (0.451, 60 us), k = 2-5 within 0.6 %, k = 1
0.442 (76 us), k = 8 0.419, no tail 0.280; up to N = 5 the best k is N. n_c 28 check (k = 1, 8; `~/fwm_runs/repeat_B3_nc28`):
no F moves more than 5e-4. **Tails retrained after m gates (§8.5, 2026-11-09/10)**: `train_fwm_tail.py --n-gates m`
(gate pulse tiled m times = one propagation; default target X^m; json `n_gates`), tails
`pulses/dgrape/*X_tail_estg_T08_m{2..8}_{t2,z}*` (600 it, 15.7 h, 14 side by side under an 80 % CPU cap);
`fwm_repeat_run.py --matched` (block = `after_n_gates` row N = m, open + closed; kept start `every_k(k=m,
closing_tail=False)` to 40 gates) -> `tables/fwm_tail_matched.csv` (3 h, 17 jobs). One block: retrained - reused
-0.07 % (m 2, both starts) to +0.94 % (m 8); repeated to 40 gates +2.3-2.8 % (m 3: +0.76 %); best = retrained every 2
gates, 0.478 vs 0.451 reused. Training (n_c 16, cut) = n_c 24 score to 1e-15.
Tests: `python validation/test_fwm_repeat.py -v` (16, ~7 s).
**Tails after the H gate (notebook §7, 2026-11-10, user).** Gate = the EsT H pick, seed 17 at 2 us
(`u_H_est_err_closed_T2_s17.npy`, label `EsT-H` in the scorer); 0.8 us tails from the two cold starts
(`train_fwm_tail.py --gate-pulse ... --gate-n-c 24 --target H`, 600 it, 2.7 h each, 2 side by side x 2 BLAS threads)
-> `pulses/dgrape/*H_tail_esth_T08_{t2,z}*`. `DGRAPE/fwm_tail_hgate.py` (forward only): `baselines` | `profiles` |
`code` -> `tables/fwm_tail_H_{baselines,profiles,code}.csv`, `cross-json` writes the swapped pseudo-runs
`H_xtail_T08` (X tail after H) and `X_htail_T08` (H tail after X) for `fwm_tail_score.py` ->
`tables/fwm_tail_heldout_H.csv` (it re-scores the X-gate references; identical to the good table, deduplicated on
load). Ceilings (n_c 24): H alone 0.9501 vs X 0.9671, losses 0.044 vs 0.027 per gate, ideal repair +3.09 % vs
+1.99 %. **Result: the tail works after H and pays more**: H tail +1.19 % (t2 0.96200, z 0.96197, same optimum),
but the **X tail moved unchanged gains +1.30 %** (also lower by the H training cost, 0.0404 vs 0.0411), so both
cold starts sit in a worse early-shot optimum (comb 1 peaks at 50 ns vs 460 ns, comb 2 bursts from 300 ns; unrepaired window 613 vs
458 ns). H tail after X +0.13 % (X tail +0.32 %). All channels after H: +0.98 % (X tail) / +0.87 %. **The tail is
a gate-independent photon-loss corrector**: with no gate in front, closed, code words 0.9998/0.9999 vs I, clean
error words 0.904 (X tail) / 0.892 (H tail) vs kitten 0.935; loss-time repair profiles equal after both gates to
1e-3. Pump-fragile as before (eta x 0.9/1.1 -6 to -13 %, Stark off -61 to -77 %). Next: warm-start the H tail
from the X tail; one tail trained for several gates.
**Is the tail error transparent? (notebook §9 since 2026-11-10, was §8, §9, 2026-11-04; photon loss only since 2026-11-09, user).** One
photon loss put in by hand at time t of the EsT + 0.8 us tail (readout decay only, n_c 16; the next pair at n_c 24),
scored as is and + ideal repair.
**Semi-transparent to photon loss**: a late loss ends as a clean error word (0.000 as is, 0.93 after an ideal repair),
and the next gate + tail repairs it to 0.71; that is the B2 first-pair vs later-pair gap (2.97 %
vs 2.07 %, = Table 7's 0.91 %). The next gate is the leak (0.93 -> 0.78 repairable), plus readout decay during
comb 2 (an early loss 0.83 closed -> 0.77). f->e and e->g jumps were dropped from this section and its caches on
2026-11-09. **"Best recovery" (`ft.optimal_recovery_fidelity`, Reimpell-Werner) is no longer used anywhere in the
notebook (2026-11-09, user)**; the function and its `Ceilings` test stay in `fwm_tail.py`.
Caches `tables/fwm_tail_transparency_{jumps,summary,pair2}.csv` (~12.5 min with 8 jobs to rebuild).
**Naming (2026-11-10, user): the notebook calls Shirol's Table II teeth "QEC teeth"** everywhere it displays them
(text, tables, legends; start `t2` = "QEC teeth", T-II -> QEC). Only display labels changed: file tags (`t2`),
`start_library` key `tableII`, `fwm.beta_table2`, and the CSV/module labels (`gate + Table II teeth`,
`idle + Table II teeth[, bare carriers]`) are unchanged and are mapped at display time. Same day (user): Table 6's
gains are plain F_pro differences (+0.00321, no %), and the notebook no longer uses the go / no-go verdict anywhere
(no verdict column, no §5 rule 5, no bar in Figure 5, "the best tail" instead of "the GO"). The "GO" wording
elsewhere in this file is the campaign's record.
Notebook order (2026-11-10): Setup (glossary), §1 idea, §2 limits after the EsT gate (since 2026-11-10, user; was d2: Figure 1 = pie of its all-channel
infidelity 0.0427, one channel at a time, photon loss 2.40 %; Table 1 ideal repair +1.99 % (rows per channel, P(even, g), gain per unit); Table 1b (2026-11-11) = even-population
shares per channel and input, right/wrong error word, overlap of the two inputs (why the gain per unit varies); Figure 2 loss timing,
EsT only; Figure 3 (2026-11-11) = population at T_g per Table 1 row: right/wrong code word, odd outside the
code, right error word (correctable, 2.24 % all channels; x 0.935 = the +2.10 % gain), wrong error word, even
uncorrectable, transmon e/f (1.0 %), numbers in Table C10; Table 2 idle/teeth tails; gate n_c 24, tails 16 cut; Table C4 keeps d2), §3 EsT comparison, §4 method
(frame, controls, cost, baselines), §5 training as run, §6 results (6.4 = estimate vs result, 6.5 robustness, 6.6 tails alone: Figure 9, Tables 10-12), §7
tails after the H gate (new 2026-11-10; Figures 10-11, Tables 13-16), §8 repeated gates, §9 error transparency of the
tail, §10 conclusions and next steps (findings, then next steps; moved
to the end 2026-11-09), Appendix A checks and timing, Appendix B dated decisions, run logistics, checklist (B.4) and
open decisions (B.5), Appendix C
supporting tables (C1-C9; Appendix D, good vs mid, removed 2026-11-09 with a moderate cleanup, user). §8.4 = B3,
§8.5 = retrained tails (Tables 20-21, no figure); §9 figures are 16-18, tables 22-23 (renumbered 2026-11-11 when §6.6 was added). Runs top to bottom with `python3 -m nbconvert --execute --inplace`
(~10 min with the shot-timing cache). Every figure and table has a markdown caption below it with a *Symbols* line;
table gains are shown in %.
**Tail trained with all of Shirol's channels (`notebooks/tail_experiments.ipynb`, 2026-11-11/12, user).** New noise
option `noise="shirol"` (`DGRAPE.model.shirol_collapse_ops`, = `AQEC.master_equation.collapse_operators` default, tested to
1e-12): decay + heating g->e and e->f (with the pumps' Δγ↑, applied in gate AND tail) + storage heating + dephasing with
the residual φ_a 0.23 ms⁻¹. Jump names `a r s_ge s_ef up_ge up_ef up_a phi_e phi_f phi_a` (`keep=` filters them); the
|f> frame stays exact. Threaded through `fwm.model_with_channels`, `fwm_tail` (`tail_model`, `gate_output`, `tail_input`,
`score`, `TailObjective(noise=)` with `decay|dgrape_all|shirol`), `train_fwm_tail.py --channels` (NOT `--noise`, which is
the start noise), `fwm_tail_score.py` conditions `shirol`, `shirol_nc_a/b`; defaults bit for bit. Runs
`pulses/dgrape/*X_tail_estg_T08_shirol_{t2,z}*` (warm from the decay 0.8 us tails, 600 it, 3.9 h each): **F_pro never
rose (+7e-7 max)**; the cost fell only by smoothing (deriv 3.59e-3 -> 3.27e-3, F -0.006/-0.009 %), so best_seen =
iteration 1 = the decay tail (|Δη| <= 1e-4). **Breaks even, barely: +0.044 % (t2; z +0.040 %) vs gate alone 0.95993,
same at n_c 16/24/28 (to 2e-8)**; all_ch +0.027 %. Channels one at a time (`DGRAPE/tail_experiments_channels.py`):
photon loss +0.78 %, other decay -0.25 %, and the four non-decay groups -0.28 % (transmon dephasing -0.19 %), which equals
the decay -> Shirol drop. Tables `tables/tail_experiments_{baselines,heldout,channels}.csv` (now notebook Appendix A).
**Noise-start sweep, derivative weight / 10 (notebook §4-§7 since 2026-11-13, user)**: `train_fwm_tail.py --deriv-scale`
(multiplies `TAIL_PENALTIES["deriv"]`, default 1 = bit for bit; json `deriv_scale`; tested), runs
`pulses/dgrape/*X_tail_estg_T08_shirol_d01_s{1..15}*` (`--start zero --noise 0.005 --seed s --channels shirol
--deriv-scale 0.1`, 1000 it, 10.2-11.1 h each, 5 side by side x 3 jobs x 1 BLAS thread). Seed rule fixed before training
(F spread over seeds 1-5 > 1e-4 -> seeds 6-10) fired (7.5e-4); seeds 11-15 added at the user's request (2026-11-13/14,
same settings, run from a `..\GRAPE-run` worktree). **All fifteen break even: +0.055 to +0.131 % (seed 8, 0.961235),
n_c 16 = 28 to 2e-8; twelve within 0.007 %**; seeds 4, 5 still climbing at 1000 it. Seeds 11-15 span 7.8e-5, best
(s14) 4e-6 below seed 8: more seeds will not help. Same waveform as the warm
tail, only stronger (comb 1 shot at the cap, comb-2 bursts larger; roughness 1.5-1.7x). Decay-set gain +0.407 % vs the
decay-trained tail's +0.321 %: the old deriv weight was the limit, not the start. Extra gain = photon-loss repair
(+0.047 %); other decay -0.016 %; more pump-fragile (eta x 0.9/1.1 -13 to -22 % vs -10/-13 %). Tables
`tables/tail_experiments_cold_{heldout,channels}.csv` (both rewritten for 15 seeds; seed 1-10 rows bit for bit). Next
(not run): warm start at deriv / 10 to separate the two changes, resume seeds 4/5, deriv / 100 or 0, pump-robust
training, shorter tails. `validation/test_dgrape.py`
`TaylorStepTest.test_against_superoperator_expm` fails at 2.7e-9 vs 1e-12 on the Windows OpenBLAS install, with or
without this change (the committed model.py gives the same).
Tests: `python3 validation/test_fwm_tail.py -v` (22 tests incl. `NGates`, `ShirolNoise`, ~1.5 min on the Windows PC),
`python3 validation/test_est_gate.py -v` (15 tests, ~1 s).

**7. FWM tails with Gertler combs (`DGRAPE/fwm_tail_ge.py`, `notebooks/FWM_tails_ge.ipynb`, 2026-11-03/04).** Track 6 redone with
Gertler 2021 Eq. 4 instead of Shirol's |f> combs, gates EsT (s24) and Ord only (no d2). Comb T = a selective linear
transmon drive (u[:, 2:4]) exciting g -> e on the even error states; comb M = `g_m xi_M a'r'q + h.c.`,
g_m = sqrt(chi_ge chi_qr) = 2pi 1.125 MHz (11x weaker than Shirol's comb 2), |n,e,0> -> |n+1,g,1>. The tail runs on
transmon {g, e} (`restrict_model`, d = 2 n_c n_r = 64 at n_c 16; the gate runs on 3 levels at n_c 24, cut and
projected), DGRAPE frame, one exponential per step (`fwm.propagate` midpoint S=1, exact), pump Stark
`S = 2 alpha q'q + chi_ge n_a + chi_qr n_r` added by hand. Controls (N, 4) = eps_T I/Q, xi_M I/Q, band +-12 MHz, caps
2pi 1 MHz / xi_M 0.1 or 0.2. No free rotation in training; `frame_tracked_fidelity` (best e^{i theta n}) reported. Scripts:
`train_fwm_tail_ge.py`, `fwm_tail_ge_grid.py` (24 runs, 8 side by side, 12.9 h), `fwm_tail_ge_prescan.py`,
`fwm_tail_ge_score.py` (12 conditions incl. `three_level`) -> `tables/fwm_tail_ge_{prescan,heldout}.csv`; tails
`pulses/dgrape/*X_tailge_{est,ord}_T{1,2,3}_c{10,20}_{g,z}*` (b4 physical, x pre-image). **Result: no go.** Every tail
loses to its gate alone: −2.1 % at 1 us, −3.4/−3.7 % at 2 us (cap 0.2), −5.1 to −5.7 % at 3 us; EsT = Ord against their
own gates. At 1 us the mixing comb stays off and the transmon comb only cancels a cavity rotation (+0.00 % vs a
frame-tracked idle tail). Real repair needs >= 2 us at cap 0.2 (photon loss only: +1.0–2.3 % over tracked idle, readout
photons 0.01–0.03), less than the 2.2 %/us the tail costs. The pump's storage Stark shift chi_ge|xi|^2 is a logical
rotation on this code; repairing tails cancel it and are fragile (2 us cap 0.2: pump ±10 % −2 to −7 %, Stark off −43 to
−47 %). Dropping |f> is safe (3-level held out within 1e-4). Ceilings at n_c 24: ideal instant repair EsT +1.99 %,
Ord +1.07 %. Next options in notebook §7 (Gertler combs as a memory, §8-style; cap 0.3 forward check).
**Chip search (2026-11-04/05): no go, stopped by the user after the forward map.** Question: is there a realistic chip
(chi_ge <= 4, chi_qr <= 5 MHz, |xi_M| <= 0.2; alpha, kappa_r, T1/T2 at Shirol's) where a g-e tail clears +0.3 %?
`DGRAPE/ge_chip.py` builds a chip from (chi_ge, chi_qr): K and chi'_q scale as (chi_ge/1.12)^2, chi_gf as chi_ge/1.12,
new cross-Kerr chi_ar = chi_ge chi_qr/(2 alpha), comb-T cap and band scale with chi_ge; `SHIROL` = track-7 constants
(chi_ar 0), bit for bit. `chip=` (and `gate_chip=`) on `fwm_tail_ge`, `est_gate`, `train_est_gate.py`,
`train_fwm_tail_ge.py` (`--chi-ge/--chi-qr/--gate-chip`) and `fwm_tail_ge_score.py`; `fwm_tail_ge.score(decode=True)`
adds a Kerr-decoded F (free theta n + phi n(n-1)/2). Forward map `DGRAPE/fwm_tail_ge_chipmap.py` (7-parameter
sequential shot, Nelder-Mead, transplant EsT gate, n_c 12; 36 chips x T 0.4-1.4 us, 102 min on 20 jobs) ->
`tables/fwm_tail_ge_chipmap.csv`. No chip clears the bar: best decoded gain **-0.48 % at (4, 5) MHz, 0.6 us**; repair
(+0.5-1.1 % over a decoded idle tail) only at chi_ge >= 2 and chi_qr >= 3; Shirol and Gertler (1.31, 2.8) chips repair
nothing. The tail costs ~2.1 %/us on every chip (cavity T1 cancels), so +0.3 % at 0.6 us needs ~1.6 % repair, 80 % of
the EsT ideal ceiling. Without free rotation high-chi chips lose 16-52 % (Kerr ~ chi^2 plus pump Stark). Not run:
Phase-2 GRAPE grid (`DGRAPE/fwm_tail_ge_chip_grid.py`, ready), gate retraining on a chip, notebook
`notebooks/FWM_tails_ge_chips.ipynb`. Plan: `~/.claude/plans/be-my-research-partner-luminous-quill.md`.
Tests: `python validation/test_fwm_tail_ge.py -v` (12 tests, ~10 s), `python validation/test_fwm_tail_ge_chip.py -v`
(12 tests, ~5 s).

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

### Syncing between devices (GitHub only)

**Since 2026-11-13 GitHub (`origin`) is the only way the repo syncs between devices.** The user has a
Windows PC (`C:\Users\iando\Developer\Qubit-Drive-GRAPE`, runs the long simulations) and a Mac (the
user works there meanwhile), each with its own clone outside any cloud-synced folder.
`pull.rebase true` is set globally on both. Protocol (agreed with the user 2026-11-13):

1. **Start a run (PC):** `git pull`, confirm `git status` is clean, then launch in its own terminal
   with `$env:OPENBLAS_NUM_THREADS = 2`. Note the `--tag`. The PC must not sleep or install
   updates during a run (Windows power settings; pause Windows Update). On the PC, `python`/`python3` on
   PATH are Microsoft Store stubs: use `C:\Users\iando\anaconda3\python.exe`. Without an activated conda env,
   also put `anaconda3\Library\bin` (and `Library\mingw-w64\bin`, `Library\usr\bin`, `Scripts`) on PATH, or
   `nbconvert` fails with a zmq "DLL load failed". A keep-awake loop that watches PIDs can outlive the run,
   because Windows reuses PIDs; check that it released.
2. **While a run is going, never `git pull`, `git switch`, `git checkout <branch>` or `git rebase` in
   the clone the run started from.** loky workers can re-read modules from disk mid-run, so new code
   would silently mix into the run. Before any of those on the PC, check for running training jobs
   (`Get-Process python`, or ask the user). To keep using git on the PC during a run, start the run
   from a second worktree (`git worktree add ..\GRAPE-run`) instead.
3. **Meanwhile (Mac):** pull, edit, commit, push often. Don't create files with the running job's
   tag, and don't re-execute and commit a notebook that the PC will also re-execute with the run's
   results.
4. **Run finished (PC):** `git pull` first, copy the run's files from `~/fwm_runs` into
   `pulses/dgrape/` (and `tables/` if any), check `git status`, add files **by name** (never
   `git add .`), commit, push right away. Then `git pull` on the Mac.
5. **Check sync:** `git fetch && git status -sb`: `ahead` = push, `behind` = pull, both = pull then
   push.

Things that follow from this:

- **Training outputs in `~/fwm_runs` are not in git.** They exist only on the device that ran them
  until they are copied into `pulses/dgrape/` or `tables/` and pushed. Remind the user to push
  finished results promptly. The trainers checkpoint every iteration there, so a killed run resumes.
- **Generated files can't be merged.** For a conflict in a notebook, `.csv`, `.npy` or `.json`,
  keep one side (`git checkout --ours|--theirs <file>`, `git add`, `git rebase --continue`) and
  rerun. During a rebase `--theirs` is the local commit being replayed and `--ours` is the upstream
  one; read both before choosing. `git rebase --abort` undoes a pull that went wrong.
  `.py`/`.md` conflicts are resolved by hand as usual.
- **A rejected push** ("fetch first") means the other device pushed first: `git pull`, then push.
- **Line endings:** `.gitattributes` stores every text file with LF endings (`* text=auto eol=lf`)
  and marks `.npy`, `.npz`, `.pkl`, `.pdf` and images as binary. The 748 files committed earlier with
  Windows CRLF endings (logs, JSON, CSV, 3 notebooks) were converted in one commit on 2026-11-13.
  Python on Windows writes CRLF; git converts it on `git add`, so this needs no care. If a device
  shows unedited files as modified after a pull, `git add --renormalize .` clears it.

**iCloud history (until 2026-11-13 the Mac clone lived in iCloud's `~/Documents`).** iCloud does not
understand git: when two writes to one file collide it keeps both and renames one `<name> 2.<ext>`
(or `<name> 2`). On 2026-10-19 `.git/index` was replaced by `.git/index 2` and `index 3`, so git
showed every tracked file as staged-deleted and also untracked (no data was lost), and there were
445 byte-identical `" 2"` copies. `.gitignore` still ignores `" 2"` names. If old copies turn up:

- **Check:** `ls .git/index*`, and
  `find . -path ./.git/objects -prune -o \( -name '* [0-9]' -o -name '* [0-9].*' \) -print`.
  Use exactly that pattern. A looser `'* [0-9]*'` also matches real names such as
  `sources/Gertler et al. - 2021 - ….pdf`.
- **Fix the index:** `git reset` (mixed; it rebuilds the index from HEAD and leaves every working
  file alone), then confirm `git status` shows only real edits.
- **Fix `" 2"` copies:** `cmp` each against its original and delete only identical ones. A copy
  that differs may be another device's newer version, so read both before choosing.

Some script docstrings still say outputs go "outside iCloud" (`~/fwm_runs`); the default folder is
still right, only the reason is out of date.

## Commands

There is no build step and no pytest config — tests are plain `unittest` files run directly.

`main.py` is the production training entry point for a single operation, and its CLI defaults
**are** the production recipe (`analysis/penalty_sweep.py`'s `FIXED` and `notebooks/grape_experiments.ipynb`'s
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

Penalty-weight sweep (`analysis/penalty_sweep.py`, `notebooks/penalty_optimization.ipynb`, `tables/penalty_sweep_*`,
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
across regimes. `notebooks/penalty_optimization.ipynb` §1–§10 has been rewritten against the new campaign;
§11–§15 remain the frozen pre-ramp record. Two artifacts are **overwritten in place** and now hold
u_max=25 data despite their names: `tables/penalty_sweep_summary_ramp.csv` and
`figures/penalty_pareto.*`.

EsT module (`EST/`): commands, per-module notes and conventions live in `EST/CLAUDE.md`, which loads
when working with files under `EST/`. Read it explicitly before EsT work that starts elsewhere
(`notebooks/est_experiments.ipynb`, `notebooks/est_optimization.ipynb`, `pulses/est/`, `logs/est_*`).

All scripts assume they are run from the repository root (several insert `REPO_ROOT` onto
`sys.path` explicitly, e.g. `analysis/pulse_analysis.py`).

**Notebooks live in `notebooks/`** (moved there by the user 2026-11-11; `experiments.ipynb` was renamed
`notebooks/grape_experiments.ipynb` at the same time). Every notebook's first code cell does
`os.chdir("..")` when the working directory is `notebooks/`, so data paths (`tables/`, `pulses/`, `figures/`)
stay relative to the repository root; keep that cell in any new notebook. Run one with
`python3 -m nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`. Notebook names in docstrings and
notes carry the `notebooks/` prefix; inside a notebook, other notebooks are named without it.

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
