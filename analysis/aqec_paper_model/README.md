# Ideal AQEC on the Cornell paper's own model (context for notebooks/AQEC_compare.ipynb §4)

Scratch-quality scripts from 2026-11-12, kept so the context numbers can be reproduced. They rebuild Roy et al.'s gate
model (their notebook cells 152-155: 2-level transmon, their H0, their four collapse operators, their pulse envelope and
clipping) in numpy, check it against their printed closed fidelity (LE 0.99862293), and run idealised AQEC variants on
their even kitten code (|0>+|4>)/sqrt2, |2>.

- `aqec_sim.py`: the gate model and an exact sparse-Liouvillian propagator.
- `aqec_models.py <path to EST-paper-data/data_and_pulse>`: variants A (map, then 2.2 us lossy), B (1 us lossy, map,
  1.2 us lossy: the variant matching DGRAPE/aqec_ideal.py), C (1 us only), D (instant), B1 and B2 to N = 20. Writes
  `aqec_ideal_results.csv` in the working directory (copied to `tables/aqec_ideal_paper_chip.csv`).
- `aqec_summary.py <same path>`: prints the comparison and plots it.

Run them from this folder. `EST-paper-data/` is the paper's data release (not tracked by git).
`tables/aqec_paper_A6.csv` holds the paper's own Fig. A6 simulated points (B1 = A6b, B2 = A6d), from
`Nxgates_AQEC_sims_X_*.npz` and `Nx_gatesAQEC_sims_X_*.npz`; "ET" is their EsT gate, "Ord" their `fidelity_L`.

Indexing: in both npz files entry 0 is N = 0 (fidelity 1) and entry 1 is N = 1; the two files agree there (0.941365),
as they must, since one gate + one AQEC is the same in both protocols. (Their B1 plot, notebook cell 164, draws
entry k at N = k + 1, a one-gate shift; we use entry k = N.)
