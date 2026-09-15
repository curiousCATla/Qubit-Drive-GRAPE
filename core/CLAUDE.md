# core/CLAUDE.md

Module notes for the cat-code GRAPE kernel. The root `CLAUDE.md` still applies; the ramp-default
and `hard_amp_limit` warnings stay there because they govern `main.py` runs from the repo root.

- **`core/grape_core.py`** — `make_ops`/`make_hamiltonian` build drift/control Hamiltonians;
  `_fidelity_core` is the shared batched kernel behind `fidelity_grad` and
  `fidelity_multi_state` (propagates all target states jointly per time step, amortizing one
  eigendecomposition per step across the batch — this is the main performance-sensitive path).
  Propagators use eigendecomposition (`U_k = V diag(e^{-i dt ω}) V†`), not matrix exponentiation.
  Penalty terms (`derivative_penalty`, `amplitude_penalty`) and the
  lower-level `optimize_controls` also live here.
- **`core/fourier_cutoff.py`** — `project_bandlimit` implements the hard frequency-band
  constraint as an orthogonal projection (`IFFT ∘ mask ∘ FFT`), applied identically to pulses
  and gradients since the projection is idempotent/self-adjoint. Enabled via `cav_band`/
  `tra_band` kwargs on `optimize_multi_state_pulse`; both must be given or both `None`
  (exactly one raises). The default is `None` — the band limit is **off** unless the caller
  asks for it, unlike `ramp_ns`, which is on by default at the `optimizer.py` entry points.
  Every production caller passes the bands explicitly (`-27/27`, `-33/33`).
- **`core/ramp.py`** — the 48 ns pedestal-subtracted Gaussian rise/fall (`ramp_envelope`) and
  the full raw-variable→physical-pulse chain `make_constraint_chain`, which returns the
  `(to_physical, to_preimage_grad)` pair every cost assembly uses. Order is **band-limit then
  ramp**, so `u = env * P(x)` and the adjoint is `P(env * g)` — the envelope multiplies the
  gradient *before* the projection. Getting that backwards still produces a plausible-looking
  gradient, so `validation/test_grape_core_perf.py` pins the adjoint identity with a negative
  control; do not "simplify" the order. This module **replaced `boundary_penalty`**, which was
  removed outright — passing `penalties={'boundary': ...}` now raises. `EST/device.py`
  re-exports `ramp_envelope` from here (core never imports from EST), so the two tracks share
  one implementation. Also holds `deramp` (chain inverse, for physical-pulse warm starts) and
  `constraint_report` (per-pulse endpoint + residual out-of-band measurement, surfaced as
  `info['constraints']`). When reporting endpoints prefer `endpoint_rel_to_peak`;
  `endpoint_rel_to_mid` divides a max by an RMS, and the two are not interchangeable — the
  module docstring's white-noise ordering benchmark ("0.84% of mid") quotes the latter.
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
