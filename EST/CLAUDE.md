# EST/CLAUDE.md

Guidance for track 2, the error-semitransparent gate replication (arXiv:2603.15356). The root `CLAUDE.md` still applies; its two EsT prohibitions (never mutate the
`core/grape_core.py` physical globals; EsT pulses go to `pulses/est/`, never `pulses/`) stay there.

## Commands (all from the repo root)

```bash
# Correctness suite. Run before trusting any EsT training result.
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

## Modules

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
  its docstring before reusing it for a different code. `error_gate_target` (the ideal gate
  applied in the error basis, target of `cerr`) leans on the same coincidence and asserts it.
- **`EST/grape_jax.py`** — the only module that replaces existing machinery. Constraint chain
  (band-limit -> ramp -> control mask) applied *inside* the cost so autodiff covers it; `expm`
  propagation (**not** `eigh` — autodiff through `eigh` hits the same degeneracy singularity
  `grape_core` hand-fixes); cost terms C1-C4; `build_gate_objective` returns a scipy-ready
  `(cost, grad)` callable.
- **`EST/train_est.py`** — two-stage L-BFGS-B driver. `--variant ord` is the same code path
  with `w2=w3=0`; keep it that way, the whole result is a comparison against it. `train()`
  returns `(u, x, info)` and `save()` writes both `u` and `x` — see `pulses/est/` below.
- **`--stages 1` (eigh driver only) is SINGLE-PHASE training**: `stage_weights` truncates the
  schedule to stage 1, so `(w1,w2,w3,w4) = (1, 0.7, 7, 1)` is held for the whole run and the
  gate-finishing stage 2 never happens. Default is still 2 and is bit-identical to before
  (`StageWeightScheduleTest` pins both). `STAGE2_VARIANTS` **raise** under `--stages 1` — they
  are defined as a change to stage 2, so truncating them would silently train plain `est`.
  Every sidecar now carries `n_stages`, and `smoke_new_terms`/`compare_new_terms` refuse to
  score a pulse whose `n_stages` does not match their campaign (`_require_phases`) — with 600+
  files in `pulses/est/` the two-stage artifacts all look plausible next to the new ones.
  **Single phase does not close the gate on X.** Stage 1 of any two-stage run *is* a
  single-phase run, so this was measurable for free: at 2000 it, seed 6, `F1` reads 0.9443
  (`est`), 0.9744 (`est_err`), 0.9601 (`est_dn`), 0.9721 (`est_all`), against 0.9991-0.9995 for
  the same runs after stage 2, and 4x the budget buys ~0.01. That is `est_c3s2`'s finding (below)
  on X rather than T: holding `w3 = 7` to the end dominates the objective. Both drivers
  therefore report three tiers, not two — **parked** (`progress <= 0.5`, F1 = 1/3, trivially
  transparent and excluded from every mean), **open** (drove, `F1 < 0.999`, not a gate), and
  **closed**. Selection stays control-relative, so a uniform stall leaves it well-posed; only
  the claim "these are gates" fails.
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
- **`EST/grape_eigh.py` / `EST/train_est_eigh.py`** — second pipeline (numpy eigh +
  hand-derived adjoint, notebook §8), tested by `EST/test_grape_eigh.py`. **Eigh-only extra
  terms**, selected by a 7-tuple `weights = (w1,w2,w3,w4,w_err,w5,w6)` (a 4-tuple is the
  original objective, bit-identical): `cerr` = C1's form on the error cardinals at t=T
  (1-cerr on `u_X_est` = 0.589, limitation 3's number); `c5` = mean_t (Δn̄_L/(n̄0+n̄1))², Eq.
  A7 with a repo-chosen normalization — **necessary, not sufficient** for Δ_QEC (it is only
  the Z part of the (a,a) term), and ~1e-2 on real pulses because a linear displacement shifts
  both words' n̄ equally, so w5 needs O(1-100) — w5=3 already drives c5 to 2e-4 and Δ_QEC to
  0.166 on X, so the earlier "O(10-100)" floor was too high; `c6` = `grape_core.derivative_penalty`, a
  SUM (~1e4 on X pulses, so 5e-6 contributes ~0.06); `cleak` = C_L = gate average of Eq. 7's
  leakage out of the INSTANTANEOUS error space span{a|0_L(t)>, a|1_L(t)>}, the 8th weight w_L
  (`leakage_cost`; projector G (G^dag G)^-1 G^dag with G = a C(t), two-word form, equal to the
  mean of `diagnostics.leakage_Ej` to 1e-12 -- tested). **C2 = C_L + C_mis exactly** (per
  cardinal 1 - F_ET = L + (1-L)(1-f), f = in-space overlap with C2's normalized image; tested),
  so C_L <= C2 and w_L reweights only C2's leakage part; C_L is ~0.94 C2 on X controls, so a
  w_L ladder behaves partly like a w2 ladder. A 7-tuple (or 8-tuple at w_L = 0) is bit-identical
  to before; `cleak` is then only reported. The JAX reference for these lives in
  the test file only; `grape_jax.py` has none. Driver variants `est_err`, `est_dn`,
  `est_err_dn`, `est_all`, `est_lk` (`--w-err`, `--w-dn`, `--w-smooth`, `--w-leak`, same weight
  in every stage; only `est_lk` builds 8-tuples and writes `w_leak` to its sidecar).
  **`WEIGHT_VARIANTS`** change only the `est` weights: `est_wB` = (1,1,7)/(1,0.2,0) (w2 raised in
  both stages), `est_experiments.ipynb` Appendix B, X seed 0, tag `seed0`, against a control trained
  alongside it (`est_appB_ctrl_seed0`). Result: more transparent after stage 1, but stage 2 gives it
  back and the gate stays open (F1 0.9956 vs 0.9990); L(T) 0.55 vs 0.34. **Cross-machine caveat**:
  the same `est` seed-0 command on the Windows PC (OpenBLAS) gives a different pulse from the Mac's
  `est_appA_seed0` (max|du| 47 rad/us), so compare only runs trained on one machine.
  `EST/smoke_new_terms.py` / `EST/compare_new_terms.py` are the scan and the ablation. **Both now
  run SINGLE PHASE** (`N_STAGES = 1`) for `est_optimization.ipynb` §2 (X) — see the single-phase entry
  at the end of this bullet for the live configuration and results. **The rest of this paragraph,
  down to that entry, is the FROZEN TWO-STAGE record** (git 4243344): its tables, tags and numbers
  are still on disk and still correct for what they are, but no code path produces them any more.
  Two-stage scan: X at 500 it/stage over the six §9 seeds that reach a gate (0, 2, 4, 5, 6, 8), w_err
  {0.01, 0.1, 0.3, 0.5, 1, 3} (extended below 0.3 after the first six-seed scan) and w5 {1, 3, 6, 10, 15, 30}, into `tables/est_newterms_smoke_X_6seeds.csv`;
  its selection rule (`apply_rule`) needs the gate on every seed and compares seed means.
  Each cell scores BOTH readings of Eq. 8: `eta_avg_mean`/`eta_avg_tail` (mean over the six
  cardinals -- what `est_optimization.ipynb` §2 plots throughout) and `eta0_mean`/`eta0_tail`
  (|0_L> alone, the paper's Fig. 1f state). `apply_rule`'s conditions read F_err_T / c5 /
  delta_qec_mean / F_ET only, so adding the columns left the picks unchanged (w_err 0.1
  pre-registered, 0.01 revised; w5 10). The two readings are NOT interchangeable: on the
  C_err grid the six-cardinal mean improves only at w_err=0.01 and is flat at 0.3, where
  eta0 reports 0.3 as the one improvement.
  Seed 6 reuses the original single-seed scan's untagged `*_smoke500_w<w>` files, whose table
  `est_newterms_smoke_X_seed6.csv` is frozen — new cells are tagged `smoke500_seed<s>_w<w>`.
  Measured (six seeds, 500 it/stage, `est_optimization.ipynb` Figure 1a): 6/72 runs **park** at
  F1=1/3 although all six controls reach the gate, so exclude parked seeds before averaging (a
  parked pulse reads F_ET ~0.997, L(T) ~0). On gate-reaching seeds C_err needs w_err >= 0.1:
  L(T) 0.65 -> 0.55 at 0.01, 0.061 at 0.1, 0.013-0.022 at 0.3-3. Its F_ET cost is **not
  monotone** (-0.004, -0.057, -0.030, -0.11, -0.14, -0.18 over the grid), so 0.3 beats 0.1.
  C5 lowers Δ_QEC by 0.05-0.14 without fixing L(T). Rule picks over six seeds: w_err 0.1
  (original) / 0.01 (revised; 0.1 drops F_ET 0.057 > 0.05, and 0.01 leaves L(T) at 0.55), since
  0.3 and 0.5 park seed 5; w5=10 (15 and 30 park seed 2).
  `STAGE2_VARIANTS` change **stage 2 only** (stage 1 = `est` stage 1, so same seed = same
  stage 1): `est_c3s2` keeps w3=7, `est_d2` adds C6, `est_c3s2_d2` both; `*_d2` **require**
  an explicit `--w-smooth` (no silent default). Weights per stage come from
  `train_est_eigh.stage_weights`, pinned by `StageWeightScheduleTest`. `EST/stage2_t.py` is
  the T-gate end-of-pulse drive study built on them, now a **FROZEN two-stage record** (no longer
  in any notebook; `drive_shape_t.py` imports its scoring and reads its table for a grey
  reference curve, so do not change its scoring without rerunning both). Measured (T, seed 1,
  2000 it/stage, single seed): `est_d2` at w6=1e-4 (the top of the screened grid) evens the drive
  (closing-48-ns energy 35.4% -> 7.1%, peak 25.1@584 ns -> 6.7@304 ns) at F1 0.99994, F_ET
  0.939 (vs 0.945), and *improves* the endpoint (L(T) 0.120 -> 0.048, F_err(T) 0.874 -> 0.950).
  Keeping w3=7 in stage 2 (`est_c3s2`, `est_c3s2_d2`) does **not** give a gate (F1 ~0.98 at
  maxiter; 7*C3 dominates stage 2) and front-loads the drive instead. C3 flattens state
  *speed*, not the waveform.
  **`est_optimization.ipynb` is single-stage throughout** (2026-10 rework: §1 shared Methods,
  §2 X results, §3 T results, §4 conclusions). **T study (§3): `EST/drive_shape_t.py`** with
  the variant **`est_sm`** (C6 alone, every stage; `--w-smooth` REQUIRED, `EXT_REQUIRE_W_SMOOTH`).
  Seed 1, 2000 it, w6 {5e-6, 2e-5, 1e-4}, tags `sp2000_seed1_w<w>`; tables
  `tables/est_drive_T_seed1_sp{.csv,_traces.npz}`, figures `fig7{a,b,c}_drive_T_sp.png`. The
  control is **reused**: `x_T_est_best2000_stage1` (stage 1 of a two-stage run IS a `--stages 1`
  run), and a fresh retrain `x_T_est_sp2000_seed1` reproduces it to 0.0 (`compare` checks this).
  Measured: the end spike is a stage-2 artifact (single-stage control 9.2% closing-window energy
  vs 35.4% two-stage; front-loaded instead, 52% in the first 100 ns; F1 0.9659, parked F1 on T is
  0.902). C6 cuts total transmon energy 41.3 -> 16.0 and C6 489 -> 116 (mostly the opening burst),
  so the closing SHARE rises 9.2% -> 17.5% though its absolute energy falls 27%: the
  pre-registered rule (share must fall) selects NO weight. F_ET flat (-0.0013), L(T) 0.121 ->
  0.092, F_err(T) 0.824 -> 0.876, F1 0.9607 at 1e-4. All four pin 87-112 transmon pre-image
  entries at 60.
  **X study (now §2, was §1) is the SINGLE-PHASE campaign** (`--stages 1`, X, 1000 it
  scan / 2000 it full runs), not the two-stage one. Artifacts: `tables/est_newterms_smoke_X_6seeds_sp.csv`
  (scan, 6 seeds x 12 cells + 6 freshly trained controls, tags `sp1000_ctrl_seed<s>` /
  `sp1000_seed<s>_w<w>`) and `tables/est_newterms_X_seed4_sp_win.csv` + `_traces.npz` (5 full runs
  at **seed 4**, tag `sp2000_seed4_win`, one row per run; `compare_new_terms.SEED` and
  `MACHINE_SUFFIX` name the paths). The full runs moved from seed 0 because seed-0 `est_dn` parked;
  seeds 4, 6 and 8 drive in all 12 scan cells. The seed-0 tables (`est_newterms_X_seed0_sp*`) were
  deleted on 2026-10-19 (never committed). The seed-0 pulses remain, so `compare_new_terms` with
  `SEED = 0` rebuilds them; notebook §2.4 / §5.4 still quote their numbers.
  **Seed 4 was trained twice, on the Mac (`sp2000_seed4`, tables `est_newterms_X_seed4_sp*`, now
  frozen, quoted in notebook §2.4 Table 2b) and on the Windows PC (`_win`, live, 2026-10-19).** The
  retrain was asked for because the Mac baseline's eta ended at 1.25; that baseline was already a
  cold start (`init: null`). The PC pulses are different waveforms (relative difference 0.6-1.2), so
  the pair is two samples of one experiment. What repeats: baseline L(T) 0.708/0.710 (eta(T)
  1.25/0.98, so the end rise is the C1-C4 objective's, not a bug; 6 of 17 PC 1000-it controls in §5
  end with tail eta >= 0.9), C_err's fix (L(T) <= 0.15, F_err(T) >= 0.92), F1 within 0.005, F_ET
  within 0.03. What does not: mean Delta_QEC moves up to 0.17 for the same command (`est_all`
  0.301 -> 0.133, `est_err_dn` 0.189 -> 0.312), so C5/C6 Delta_QEC effects at one seed are
  unresolved and "est_err_dn is the best trade" no longer stands. The
  "Measured" numbers below are the seed-0 campaign's. `apply_rule`'s seed means include PARKED
  seeds, whereas Figure 2a's means do not; quote the figure's (drove-only) means in prose. The
  two-stage seed-6 tables
  (`est_newterms_X_seed6*`, `est_newterms_smoke_X_6seeds.csv`, `est_newterms_smoke_X_seed6.csv`)
  are frozen on disk and in git at 4243344; nothing in the notebook reads them.
  Measured (single phase): **0 of 78 scan cells and 0 of 5 full runs close the gate.** The six
  controls read F1 0.9311-0.9560 (mean 0.9431), the seed-0 baseline 0.9581 -- see the `--stages`
  entry above. `C_err` survives intact: at seed 0 it takes F_err(T) 0.445 -> 0.931 and L(T)
  0.513 -> 0.105 at F_ET 0.7278 -> 0.7279, and *raises* F1 to 0.9792; the scan has L(T) falling
  0.639 -> 0.106 by w_err=0.3 with F_ET at or above the control throughout. `C5` alone drives its
  own term 6.2e-3 -> 8e-5 and mean Delta_QEC 0.265 -> 0.110 across the grid at almost no F_ET
  cost, but **never moves L(T)** (0.48-0.65 at every weight). Combined with C_err at full budget it
  inverts as before: Delta_QEC 0.298 -> 0.392, F_ET 0.728 -> 0.520. C6 cuts its own term 3.0x and
  the transmon rms step 2.9 -> 1.4 rad/us, but here it RAISES out-of-band fraction (5.6e-3 ->
  8.0e-3) and mean eta (0.475 -> 0.605), so it is no longer "nearly free".
  **Parking is the live hazard and the scan predicts it.** 6/72 scan cells park (`err` 0.3/0.5 at
  seed 5, `err` 1 at seed 0, `dn` 3 at seed 0, `dn` 15/30 at seed 2) though every control drives;
  a parked pulse barely drives at all (cavity +-1 rad/us vs +-20, Fock <= 4) so it reads F_ET ~
  0.99 and L(T) ~ 0.007 for free. `dn` w5=3 seed 0 parks in the scan at 1000 it AND as §2.2's
  `est_dn` at 2000 it -- the full runs used w_err=0.3 / w5=3, and the revised rule picks 0.1 / 10,
  so **neither §2.2 weight is one the scan selects**. Do not carry a weight that parks any scanned
  seed to a single-seed full run.
- **`--w2` / `EST/tune_c2_c5.py` (`est_optimization.ipynb` §5, "Phase 1")** — `--w2` on
  `train_est_eigh.py` replaces the C2 weight of the single phase; it **raises** with `--stages 2`
  (which stage would be ambiguous), `None` is bit-identical, the sidecar gets `w2_override` only when
  set (`StageWeightScheduleTest.test_w2_override_*`). The campaign: X, `est_dn`, 1000 it, single
  phase, w2 {0.6..1.0} x w5 {2..4} (5 x 5) over the first 12 seeds of 0, 2, 4, 5, 6, 8, 10-21 whose
  control (`est`, tag `c2c5_sp1000_ctrl_seed<s>`) drives; cells tagged
  `c2c5_sp1000_seed<s>_w2_<a>_w5_<b>`. **All trained on the Windows PC** (controls retrained there,
  not the Mac's `sp1000_ctrl_*`). `run --jobs N` is a resumable Python launcher (one thread per run,
  stdout logs + score cache in `~/est_runs/c2c5/`). Score = mean of (mean Delta_QEC, mean L, mean
  eta six-cardinal), each over the seed's own control; L(T) reported only. Rule: parked runs out of
  means; eligible = <= 1 parked seed and every driving seed F1 >= control F1 - 0.01; pick = lowest
  eligible seed-mean score. Tables `tables/est_c2c5_{seeds,scan,cells}_X_sp.csv`, figures
  `figures/est/fig8{a,b}_c2c5_*` (fig8c only if a cell is picked). 318 runs, 11.8 h at `--jobs 20`
  (~40 min each). **Result (2026-10-19): no cell is eligible, nothing picked.** Seed 15 loses
  0.014-0.073 in F1 in all 25 cells; without it only the w2 = 0.6 row passes, and that row is
  LESS transparent (score 1.03-1.13). w2 is the knob (score row means 1.08 -> 0.80 for w2
  0.6 -> 1.0, mean F1 -0.011 per +0.1); w5 in 2-4 has no visible effect on the score. Run-to-run F1
  noise (same seed and w2, different w5) is ~0.009, about the 0.01 bar. Post hoc only: w2 = 0.8,
  w5 = 3 / 3.5 gives score 0.93 / 0.91 at mean dF1 -0.004. The bar change is the user's call.
- **`EST/test_grape_jax.py`** — the anchor is the C1 gradient checked against
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

## Conventions

- **Amplitude is capped by the C4 penalty, not by the L-BFGS-B box.** Because the constraint
  chain lives inside the cost, `bounds` constrain the raw pre-image, and the cap is circular
  (`sqrt(u0^2+u1^2) <= eps_max`) which a per-element box cannot express. The default
  `hard_bound=60` permits `|eps|` up to 84.9 rad/us, 3.4x the 25.13 rad/us physical cap, so it
  never binds *the physical amplitude*. Do not "tighten" it expecting the waveform to follow.
  It **does** bind the raw pre-image on T: `x_T_est_best2000_stage1` has 127 transmon entries
  pinned at ±60 and `x_T_est_best2000` 55 (the band-limit projection lets a large `x` give a
  small `u`), so those are box-constrained optima. Stage-2 C6 (`est_d2`, w6=1e-4) takes it to 0.
  Count pinned transmon entries, not `max|x|` -- masked cavity columns sit at 60 as frozen junk.
- **C2 and C3 deviate deliberately from the paper's printed equations** (C2 divides by
  `N_norm^2`; C3 is normalized by `mean(v)^2`). The literal C3 is dimensionful, so its size
  depends on units the paper does not fully fix: 965 in (rad/us)^2 swamps C1/C2 at
  w3=5-10, 9.6e-4 in (rad/ns)^2 is negligible. "Swamping" is therefore a unit artifact; the
  unit-free reason is scale invariance -- raw `Var(v) = mean(v)^2 * CV^2` rewards slowing
  down, the normalized `CV^2` does not, and it is the only form that is both non-negligible
  and leaves the trained stage-1 X pulse below the parked trajectory. It does not forbid
  parking on its own (C1 does). Both are documented in
  `EST/grape_jax.py` and pinned by tests. Do not "correct" them back without reading those.
  The printed forms exist **opt-in only**: `--variant paper` in `train_est.py` uses `grape_jax.PAPER_COST_FORM`:
  - C2 over `N_norm`, excluding t=T;
  - raw C3 with v in rad/ns;
  - weights (1,0.6,6)/(1,0.1,0).

  Its C2 is unbounded below (it rewards photon number) and its C3 is ~1e-3, i.e. inert. Its logs are not comparable to `est` logs. `PaperCostFormTest` pins the defaults as unchanged.
  The eigh pipeline has the same form: `grape_eigh.PAPER_COST_FORM` (a copy, pinned equal to
  grape_jax's; grape_eigh must not import jax), passed as `**form` to `cost_and_grad` /
  `build_gate_objective`, with a hand adjoint for C2 over `N_norm` (and the zero injection at
  t=T). Driver variant **`est_paperform`** (`train_est_eigh.FORM_VARIANTS`) = that form at the
  **`est` weights** (1,0.7,7)/(1,0.1,0), so a paper-vs-repo pair differs in cost form only;
  only form variants write `cost_form` to the sidecar. `PaperFormTest` in
  `EST/test_grape_eigh.py` checks it against JAX autodiff and finite differences.
  Measured, the FAIR comparison (`est_experiments.ipynb` Appendix A, rewritten 2026-10-12): X,
  eigh, `est` vs `est_paperform` at equal weights, seeds 0 and 6, 2000 it/stage, tags
  `appA_seed{0,6}`, tables `tables/est_appA_X_*.csv`. The repo-form reruns reproduce
  `u_X_est_eigh` / `u_X_est_best2000` exactly (training here is deterministic). Verdict rule:
  resolved = same sign at both seeds AND both |paper - repo| > |repo_0 - repo_6|. **No
  transparency metric is resolved**; L(T) 0.400/0.392 vs 0.375/0.398 (seed effect 0.023), so the
  printed form does not explain the end-of-gate leak. Resolved: F1 worse (0.99814/0.99776, open
  gates), cavity drive at the cap 43-49% vs 5-22%, mean <n>_C up. Paper seed 0 reaches Fock 15,
  so its n_c 16-28 truncation spread is 7.3e-3, but it is flat to 2e-5 for n_c >= 20.
  The earlier JAX run (`u_X_paper.npy`, weights (1,0.6,6), seed 0: L(T) 0.470 vs 0.396) is
  superseded. Its pulses, `logs/est_X_paper.json` and `tables/est_paper_X_*.csv` were deleted on
  2026-10-19 and are in git at a04895f; its figures (`figures/est/*paper_X*`) are kept.
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
