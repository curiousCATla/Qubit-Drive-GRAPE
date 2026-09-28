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
  SUM (~1e4 on X pulses, so 5e-6 contributes ~0.06). The JAX reference for these lives in
  the test file only; `grape_jax.py` has none. Driver variants `est_err`, `est_dn`,
  `est_err_dn`, `est_all` (`--w-err`, `--w-dn`, `--w-smooth`, same weight in every stage).
  `EST/smoke_new_terms.py` / `EST/compare_new_terms.py` are the scan and the ablation. **Both now
  run SINGLE PHASE** (`N_STAGES = 1`) for `est_optimization.ipynb` §1 — see the single-phase entry
  at the end of this bullet for the live configuration and results. **The rest of this paragraph,
  down to that entry, is the FROZEN TWO-STAGE record** (git 4243344): its tables, tags and numbers
  are still on disk and still correct for what they are, but no code path produces them any more.
  Two-stage scan: X at 500 it/stage over the six §9 seeds that reach a gate (0, 2, 4, 5, 6, 8), w_err
  {0.01, 0.1, 0.3, 0.5, 1, 3} (extended below 0.3 after the first six-seed scan) and w5 {1, 3, 6, 10, 15, 30}, into `tables/est_newterms_smoke_X_6seeds.csv`;
  its selection rule (`apply_rule`) needs the gate on every seed and compares seed means.
  Each cell scores BOTH readings of Eq. 8: `eta_avg_mean`/`eta_avg_tail` (mean over the six
  cardinals -- what `est_optimization.ipynb` §1 plots throughout) and `eta0_mean`/`eta0_tail`
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
  the T-gate end-of-pulse drive study built on them (`est_optimization.ipynb` §2). Measured (T, seed 1,
  2000 it/stage, single seed): `est_d2` at w6=1e-4 (the top of the screened grid) evens the drive
  (closing-48-ns energy 35.4% -> 7.1%, peak 25.1@584 ns -> 6.7@304 ns) at F1 0.99994, F_ET
  0.939 (vs 0.945), and *improves* the endpoint (L(T) 0.120 -> 0.048, F_err(T) 0.874 -> 0.950).
  Keeping w3=7 in stage 2 (`est_c3s2`, `est_c3s2_d2`) does **not** give a gate (F1 ~0.98 at
  maxiter; 7*C3 dominates stage 2) and front-loads the drive instead. C3 flattens state
  *speed*, not the waveform.
  **`est_optimization.ipynb` §1 is now the SINGLE-PHASE campaign** (`--stages 1`, X, 1000 it
  scan / 2000 it full runs), not the two-stage one. Artifacts: `tables/est_newterms_smoke_X_6seeds_sp.csv`
  (scan, 6 seeds x 12 cells + 6 freshly trained controls, tags `sp1000_ctrl_seed<s>` /
  `sp1000_seed<s>_w<w>`) and `tables/est_newterms_X_seed0_sp.csv` + `_traces.npz` (5 full runs at
  seed 0, tag `sp2000_seed0`, one row per run). The two-stage seed-6 tables
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
  0.99 and L(T) ~ 0.007 for free. `dn` w5=3 seed 0 parks in the scan at 1000 it AND as §1.2's
  `est_dn` at 2000 it -- the full runs used w_err=0.3 / w5=3, and the revised rule picks 0.1 / 10,
  so **neither §1.2 weight is one the scan selects**. Do not carry a weight that parks any scanned
  seed to a single-seed full run.
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
  Measured (X, seed 0, 2000 it/stage, `u_X_paper.npy`, `est_experiments.ipynb` Appendix A): it is not more transparent and leaks MORE at the end (L(T) 0.470 vs 0.396, error-basis overlap 0.411 vs 0.589), at Fock 13 vs 9 and 2.9x the infidelity -- so the C2 normalisation is not the cause of the end-of-gate leakage. Single seed, and confounded with the weight change.
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
