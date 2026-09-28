# tables/CLAUDE.md

Provenance of the generated tables. Do not hand-edit anything here; regenerate via the script or
notebook that owns it.

- **Everything below is now post-`u_max=25`, except the `penalty_sweep_*` family.**
  The `pulses/u_*_main.npy` set was retrained on 2026-09-15 under the Heeres Supp. Eq. 19
  modulus amplitude penalty at `amp_max = hard_amp_limit = 25` (per-gate seeds: enc 47, dec 43,
  X 43, Y 46, Z/H/T/I 42; the `u_max=40` pulses they replace are kept in
  `pulses/archive_umax40/`). `experiments.ipynb` was re-executed top to bottom the same day
  (43 code cells, `RUN_OPTIMIZATION=False`, zero errors, no pulse rewritten), and the
  script-owned tables were regenerated alongside it: `phase0_corrected_fidelities.csv` and
  `phase1b_enc_dec_fidelities.csv` (`analysis/rescore_saved_pulses.py`),
  `validation_master_summary.csv` (`validation/validate_logical_gates.py`),
  `truncation_convergence.csv`, and `qutip_validation.csv` / `qutip_phase_validation.csv`.
  **`penalty_sweep_*` is the exception and is NOT post-`u_max=25`**: `analysis/penalty_sweep.py`
  deliberately keeps `FIXED` at the pre-change 40 / per-quadrature values so its pulse cache
  stays valid, so those tables describe a different recipe from every other table here.

- **Provenance of the previous (ramp) retrain, still accurate for the notebook-owned list.**
  The pulses were retrained on 2026-09-09 and `experiments.ipynb` re-executed on 2026-09-10 when
  Section 8 was added, so its **stored cell outputs are post-ramp**, and so is everything it
  writes: `validation_master_summary.csv`,
  `gate_campaign_summary.csv`, `pulse_characterization.csv`, `decoherence_*.csv`,
  `unitary_*.csv`, `process_tomography_*.csv`, and `results/gate_campaign_info.json` (whose
  stored recipe no longer lists the removed `boundary` penalty), alongside
  `phase0_corrected_fidelities.csv` and `phase2_summary.csv`, which were already current.
  Post-ramp `F_avg_gate`/`Pipeline_avg` now exist in both the notebook markdown (Section 6) and
  the CSVs. **This does not extend to tables the notebook never writes** — the
  `penalty_sweep_*` family (see the `penalty-sweep` skill) in particular keeps its own, separately documented provenance,
  and the pre-ramp cache warnings there still stand. Check mtimes before quoting a number from
  anything outside the notebook-owned list.
