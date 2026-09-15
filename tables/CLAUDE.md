# tables/CLAUDE.md

Provenance of the generated tables. Do not hand-edit anything here; regenerate via the script or
notebook that owns it.

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
  `penalty_sweep_*` family (see the `penalty-sweep` skill) in particular keeps its own, separately documented provenance,
  and the pre-ramp cache warnings there still stand. Check mtimes before quoting a number from
  anything outside the notebook-owned list.
