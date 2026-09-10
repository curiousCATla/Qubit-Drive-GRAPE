# Qubit-Drive-GRAPE

This repository implements **GRAPE** (Gradient Ascent Pulse Engineering) for microwave control of a **transmon–cavity** circuit-QED system. Optimized waveforms prepare Fock states, realize **even-parity cat-code** encoding and decoding, and implement single-qubit logical gates on the resulting cat-code subspace.

The physical model, cat-code definition, and frequency-band-limited control constraint follow Heeres, R. W. *et al.*, ["Implementing a universal gate set on a logical qubit encoded in an oscillator"](https://www.nature.com/articles/s41467-017-00045-1), *Nature Communications* **8**, 94 (2017).

## Physics

### System model

We consider a dispersively coupled transmon and cavity, simulated in a rotating frame with truncated Fock bases. The joint Hilbert space is $\mathcal{H}_T \otimes \mathcal{H}_C$ (dimension $n_t \times n_c$), with annihilation operators $B$ (transmon) and $A$ (cavity).

**Drift Hamiltonian** (laboratory parameters in MHz, converted to rad/μs):

$$
H_0 = \chi\, n_A n_B + \frac{K}{2}\, A^{\dagger 2} A^2 + \frac{\chi'}{2}\, n_B\, A^{\dagger 2} A^2 + \frac{\alpha}{2}\, B^{\dagger 2} B^2
$$

| Term | Physical origin |
|------|-----------------|
| $\chi\, n_A n_B$ | Dispersive transmon–cavity coupling |
| Kerr $A^{\dagger 2}A^2$ | Cavity self-Kerr / oscillator anharmonicity |
| $\chi'\, n_B\, A^{\dagger 2}A^2$ | Second-order dispersive shift |
| $\alpha\, B^{\dagger 2}B^2$ | Transmon anharmonicity (included for $n_t \geq 3$) |

**Control Hamiltonian** — four real drive channels at each time step:

$$
H_c = \varepsilon_{C,I}(A+A^\dagger) + \varepsilon_{C,Q}\, i(A-A^\dagger) + \varepsilon_{T,I}(B+B^\dagger) + \varepsilon_{T,Q}\, i(B-B^\dagger)
$$

The total Hamiltonian at step $k$ is $H_k = H_0 + \sum_j u_{k,j}\, H_{c,j}$, with $dt = 2$ ns and $N \approx 550$ steps (total duration $\sim 1.1$ μs).

### Cat-code logical qubit

Logical states are defined by the **even four-component cat code** (Heeres *et al.*, 2017):

- $|{+Z_L}\rangle$: coherent superposition on Fock levels $n = 0, 4, 8, \ldots$
- $|{-Z_L}\rangle$: superposition on $n = 2, 6, 10, \ldots$

with amplitude $\alpha = \sqrt{3}$. Encoding maps the transmon computational states $|g,0\rangle$ and $|e,0\rangle$ onto $|g\rangle \otimes |{\pm Z_L}\rangle$; decoding implements the inverse map. Logical gates (X, Y, Z, H, T, I) act on the cat subspace while the transmon remains in $|g\rangle$.

## Optimization

### GRAPE objective

For a target state transfer $|i\rangle \to |f\rangle$, the unitary $U = U_N \cdots U_1$ is constructed from piecewise-constant controls. The state-transfer fidelity is

$$
F = \bigl|\langle f | U | i \rangle\bigr|^2.
$$

Every gate here (encode, decode, and the six single-qubit logical gates) is trained on two branches at once (e.g. `|+Z_L⟩`/`|-Z_L⟩`, or `|g,0⟩`/`|e,0⟩` for encode/decode), so `grape_core.py` provides two multi-state reductions of $F$:

- **`fidelity_multi_state`** — the plain per-branch average, $F = \frac{1}{M}\sum_m |\langle f_m | U | i_m \rangle|^2$. This is blind to global phase: $|\langle f|e^{i\varphi} U|i\rangle|^2$ is independent of $\varphi$, so it can be maximal even when the *relative* phase between branches — the quantity a superposition input actually depends on — is wrong.
- **`coherent_fidelity_multi_state`** — the phase-coherent (process) fidelity, $F = \bigl|\sum_m \langle f_m | U | i_m \rangle\bigr|^2 / M^2$, which sums the raw complex overlaps *before* squaring and is therefore only maximal when every branch matches its target under the *same* global phase.

Every logical gate's correctness depends on that relative phase (e.g. $T$'s $\pi/4$ relative phase, $H$'s self-inverse $H^2=I$ property, and X/Y/Z/enc/dec's correctness on superposition inputs), so **production training uses `coherent_fidelity_multi_state` for all eight gates by default** (`main.py`'s `resolve_fidelity_fn`, `--fidelity-fn auto`; `optimize_multi_state_pulse`'s `fidelity_fn` parameter). This was not a design choice made up front but a fix for a real defect: all eight pulses were originally trained with the phase-blind `fidelity_multi_state`, and six of them (`U_X`, `U_Y`, `U_Z`, `U_I`, `U_enc`, `U_dec`) reached excellent per-branch fidelity ($F_{\text{mean}} \geq 0.996$) while their extracted 2×2 logical-gate fidelity `F_avg_gate` was as low as 0.335–0.980 — undetectable by checks that only ever propagate one branch/trajectory at a time (`U^2=I`, `tier4_enc_gate_dec_pipeline`). (`U_H`/`U_T` had already been retrained with the coherent objective earlier and were unaffected.) All six were retrained with `coherent_fidelity_multi_state` (`analysis/retrain_phase_coherent_gates.py`), recovering `F_avg_gate` to $\geq 0.9979$ for every gate at the time. `validate_logical_gates.py`'s `tier3_gate_algebra` (X/Y/Z/H/T/I) and `tier4_enc_dec_relative_phase` (enc/dec) checks, both driven by `F_avg_gate`, now guard against this regressing silently again.

Those figures predate the endpoint ramp. On the current `pulses/u_*_main.npy` set, retrained under the ramp pipeline, `F_avg_gate` runs **0.99638 (dec) to 0.99979 (T)** against a $\geq 0.99$ target, and `Pipeline_avg` — the encode→gate→decode number, and the one that matters practically — runs **0.99391 (X) to 0.99576 (H)** against $\geq 0.985$. The pipeline figure improved for five of the six logical gates relative to the pre-ramp set (H $+2.9$, Y $+1.5$, X $+1.1$, I $+0.9$, T $+0.6$, all $\times 10^{-3}$) and slipped only for Z ($-1.0 \times 10^{-3}$).

### Analytic gradients

Gradients are obtained via the **adjoint (costate) method**:

1. **Forward pass** — propagate $|\phi_k\rangle = U_k \cdots U_1 |i\rangle$ and store the eigendecomposition of each $H_k$.
2. **Backward pass** — propagate costates $\langle\lambda_k| = \langle f| U_N^\dagger \cdots U_k^\dagger$.
3. **Per-step derivative** — evaluate $\partial U_k / \partial u_j = V \bigl(\Phi \circ V^\dagger H_{c,j} V\bigr) V^\dagger$, where $\Phi$ regularizes near-degenerate eigenvalue pairs (e.g. $|g,0\rangle$–$|g,1\rangle$ in the rotating frame).

### Frequency-band-limited controls

To respect realistic AWG and filter bandwidths, controls may be restricted to a hard frequency band following Heeres *et al.* (2017, Supplementary Eq. 22). Rather than penalizing out-of-band spectral content, `fourier_cutoff.py` enforces the constraint exactly: the raw L-BFGS-B variable $x$ is treated as a pre-image, and the physical pulse is the orthogonal projection

$$
u = P(x), \qquad P = \mathrm{IFFT} \circ \text{mask} \circ \mathrm{FFT}
$$

onto the band-limited subspace, applied separately to the cavity drive $\varepsilon_C = C_I + iC_Q$ and the transmon drive $\varepsilon_T = T_I + iT_Q$. Since $P$ is idempotent and self-adjoint with respect to the real inner product, the same operator projects both the pulse and its gradient. The constraint is enabled by passing `cav_band`/`tra_band` to `optimize_multi_state_pulse`; both must be given or both `None`, and the default is `None` (band limit off unless requested — every production caller passes them explicitly).

$P$ is exact *as a projection*, but it is not the last step in the chain: the endpoint ramp described in the next section is applied after it, so the physical pulse is band-limited only up to the envelope's own spectral leakage. See "Endpoint boundary condition" below for the measured residual and why this ordering was chosen.

### Endpoint boundary condition: a ramp, not a penalty

A realistic drive must start and end at zero amplitude. This was originally a soft penalty, $\lambda_{\text{boundary}}(\|u_0\|^2 + \|u_N\|^2)$, and it did not work. Its gradient had support on exactly 2 of 550 rows, and because every penalty acts on the *projected* pulse and the gradient is pushed back through $P$, that two-row gradient was smeared across all 550 rows before L-BFGS-B saw it. $P$ is a circular FFT operator and does not preserve endpoint zeros in the first place, so the objective and the reparametrization were in direct conflict. The penalty sweep measured the result: over a full $5\times5$ `deriv`$\times$`boundary` grid, the marginal range of held-out fidelity along `boundary` was $8.1\times10^{-4}$–$4.8\times10^{-3}$, straddling the $1.03\times10^{-3}$ basin-noise floor and non-monotonic in every row.

The penalty has been **removed** and replaced by a structural constraint borrowed from the EsT track (`core/ramp.py`, shared with `EST/device.py`). The full chain from the raw L-BFGS-B variable to the physical pulse is

$$
u = R\,P(x), \qquad R = \mathrm{diag}(\mathrm{env}), \qquad \frac{\partial\,\mathrm{cost}}{\partial x} = P\bigl(\mathrm{env}\odot \tfrac{\partial\,\mathrm{cost}}{\partial u}\bigr),
$$

where $\mathrm{env}$ is a pedestal-subtracted Gaussian rise/fall of 48 ns ($\sigma = T_{\text{ramp}}/2$), exactly 0 at $t=0$ and $t=T$ and exactly 1 across the flat top. Because $R$ is diagonal and $P$ self-adjoint, the adjoint of $R P$ is $P R$ — the envelope multiplies the gradient **before** the projection. Reversing the two still yields a plausible-looking gradient, so the adjoint identity is pinned by test with a negative control (`validation/test_grape_core_perf.py`, `RampedConstraintChainTest`).

Three consequences, all measured rather than assumed:

- **Endpoint amplitude drops on every operation measured, typically ~3× and up to 11×.** The metric is `constraint_report`'s `endpoint_rel_to_peak` — $\max\|u_0\|$ over the waveform's own peak amplitude, i.e. how far from *off* the drive starts. (Do not conflate it with `endpoint_rel_to_mid`, which divides by a mid-pulse RMS; the white-noise ordering benchmark quoted below uses that other metric.)

  | | $U_{\rm opt}$ | enc | dec | X | Z | H | T | I |
  |---|---|---|---|---|---|---|---|---|
  | pre-ramp | 6.06% | 4.44% | 3.17% | 4.75% | 3.64% | 2.98% | 4.98% | 3.87% |
  | post-ramp | 1.13% | 1.91% | 1.60% | 2.39% | 0.32% | 1.53% | 0.47% | 0.34% |

  $U_Y$ is absent from this table because it was not retrained on the same cold start as the others — see the seed caveat below.

  The spread tracks how amplitude-hungry the gate is. Because the envelope removes control authority over the first and last 24 steps, L-BFGS-B buys it back by inflating the pre-image there — on $U_X$, $|P(x)_0| = 30.6$ against a mid-pulse RMS of 5.03, so $u_0 = 0.0135 \times 30.6 = 0.42$ rad/μs survives. Hence high-amplitude gates land near 2% while low-amplitude, largely phase-only ones ($Z$, $T$, $I$) reach ~0.3%. Midpoint sampling is the other half of the gap: $\mathrm{env}[0] = 0.0135$, not 0. **What bounds that inflation is `hard_amp_limit` on the raw variable, not the envelope** — raising it would quietly undo part of the ramp, so `info['max_abs_preimage']` is worth watching.
- **Held-out fidelity improved, though not uniformly.** On $U_X$, mean Pedersen fidelity over the five untrained truncations went 0.99655 → 0.99815 ($+1.6\times10^{-3}$ against a $1.03\times10^{-3}$ basin-noise floor), the spread across held-out truncations tightened ~10× ($3.5\times10^{-3} \to 3.3\times10^{-4}$), and peak leakage $L_1$ fell ~3× ($6.1\times10^{-3} \to 2.1\times10^{-3}$) — at essentially unchanged peak amplitude (17.45 → 17.73 rad/μs). Across the full set at $n_c=24$, only four operations moved beyond the noise floor: enc ($+2.0\times10^{-3}$), Y ($+1.1\times10^{-3}$) and X improved, and **$U_{\rm dec}$ regressed** ($-1.6\times10^{-3}$, with $L_1$ rising $2.0 \to 3.6 \times10^{-3}$) — the one clear cost of the retrain. H, T, I and Z moved by less than the floor either way.
- **The pulse is no longer *exactly* band-limited.** Ramping after projecting reintroduces sub-percent out-of-band energy. The alternative ordering — projecting last — is exactly band-limited but smears the envelope until the endpoints sit at ~32% of mid-pulse amplitude, defeating the point. Both orderings are asserted in the test suite so the choice rests on evidence rather than a comment.

The width is set by `ramp_ns` on `optimize_multi_state_pulse` (default 48 ns; `None` disables it, which is required for pulses too short to hold a flat top). Note that `make_constraint_chain` itself defaults to `ramp_ns=None` — the 48 ns default lives on the `optimizer.py` entry points, not on the chain.

#### A seed caveat: $U_Y$ hit the box

The pre-image inflation described above is not merely a curiosity; on one gate it broke training outright. Under the ramp at the production `hard_amp_limit=40`, a seed-42 cold start drove $U_Y$'s raw variable to exactly $\max|x| = 40.000$, with 0.46% of its entries pinned at the bound. It converged into a badly leaky basin: held-out $F_{\rm pedersen}$ **0.9973 → 0.7491**, with $L_1$ reaching 0.44. Re-running the identical configuration at further seeds separates cause from coincidence:

| seed | $F_{\rm coh}$ | $\max\lvert x\rvert$ | box binding? |
|---|---|---|---|
| 42 | 0.9171 | 40.000 | **yes** |
| 43 | 0.9976 | 40.000 | **yes** |
| 44 | **0.9988** | 26.59 | no |
| 45 | 0.8908 | 40.000 | **yes** |

The correlation is exact — the three seeds that reached the bound produced the three worst pulses, and the one that stayed clear produced the best. `pulses/u_Y_main.npy` therefore holds the **seed-44** pulse; every other operation is a seed-42 cold start. Reproduce it with `python main.py --gate Y --seed 44`.

The practical rule: **check `info['max_abs_preimage']` against `hard_amp_limit` before trusting any retrained pulse.** A pulse sitting at the bound has not converged, it has been clipped. For reference, $U_X$ sits at 32.3 and $U_{\rm enc}$ at 32.7, both comfortably inside.

### Saved pre-images

Training now writes two arrays per pulse: the physical waveform `pulses/u_<gate>_main.npy` and the raw optimizer pre-image `pulses/x_<gate>_main.npy`. Only $u$ is physical and only $u$ is ever scored; $x$ exists so a run can be resumed *exactly*, via `init_x`. This matters more than it did before the ramp: feeding a saved physical pulse back in as `warm_start` would envelope it a second time, so a physical warm start is now inverted through the chain (`core.ramp.deramp`) and the inversion is checked. Pulses trained before the ramp existed are not in the chain's range and are refused rather than silently mis-resumed — pass `warm_start_strict=False` to override for legacy comparisons.

### Solver and regularization

| Component | Role |
|-----------|------|
| **L-BFGS-B** (`scipy.optimize.minimize`) | Bound-constrained quasi-Newton minimization of $-F$ |
| **Derivative penalty** $\sum_k \|u_{k+1} - u_k\|^2$ | Temporal smoothness of the control |
| **Gaussian rise/fall ramp** (48 ns) | Vanishing drive amplitude at the endpoints, enforced *structurally* rather than by a penalty |
| **Amplitude penalty** | Soft bound on $\|u\|_\infty$ (default 40 rad/μs) |
| **Hard amplitude bound** | Box constraint on the raw L-BFGS-B variable $x$. `optimize_multi_state_pulse` defaults to 50 rad/μs and `refine_pulse*` to 40; **production training uses 40** (`main.py`, `analysis/penalty_sweep.py`) |
| **Frequency band limit** | Optional hard spectral cutoff via orthogonal projection |
| **Multi-truncation training** | Averaged fidelity over several $n_c$ (e.g. 20, 24, 28) to improve robustness to Hilbert-space truncation |
| **Parallel evaluation** (`joblib`) | Concurrent fidelity and gradient evaluations over truncations |
| **Warm start** | Low-pass filtered random initialization, or loading of a saved `.npy` pulse |

Time-step propagators are constructed via eigendecomposition, $U_k = V \,\mathrm{diag}(e^{-i\,dt\,\omega})\, V^\dagger$, rather than direct matrix exponentiation.

### Performance

`grape_core.fidelity_grad` and `fidelity_multi_state` are thin wrappers around a shared `_fidelity_core`. Call signatures and return shapes are unchanged, so existing callers remain compatible. Two algorithmic improvements account for the observed speedup:

- **Shared eigendecomposition across state pairs.** The propagator $U_k$ and its spectral decomposition depend only on $(H_0, H_c, u_k)$, not on the particular state being transferred. `_fidelity_core` diagonalizes once per time step and propagates all $M$ states jointly as an $(n, M)$ batch. This amortization is most significant for multi-state gates ($M=2$ for the cat-code logical gates, encode, and decode).
- **Batched diagonalization and broadcast matrix products.** Hamiltonians are diagonalized in chunks of approximately 256 steps to bound peak memory. The four-control gradient-basis transformation uses broadcast BLAS products rather than a per-channel Python loop. Evaluations with `want_grad=False` omit trajectory and eigenvector storage required only for the adjoint pass.

**Benchmarks** (fidelity bit-identical to the pre-refactor implementation; regression tests in `test_grape_core_perf.py`):

- `compare_pulses.py` (nine production pulses, $n_c = 20$–$36$): **140.0 s → 69.9 s (2.0×)**
- Single `fidelity_multi_state` fidelity-and-gradient evaluation ($n_t=3$, $n_c=28$, $N=2500$, $M=2$): **5.57 s → 2.71 s (2.1×)**
- Single-state problems ($M=1$) exhibit a substantially smaller gain, as there is no redundant work to amortize; residual cost is dominated by the per-step $O(n^3)$ eigendecomposition required by the adjoint method.

`optimize_multi_state_pulse` reuses a single `joblib.Parallel` pool for the full optimization call (`parallel_backend`, default `'loky'`). On present workloads this yields no measurable change relative to joblib's global executor cache, but simplifies pool management and exposes `parallel_backend='threading'` when appropriate.

`test_grape_core_perf.py` comprises randomized equivalence tests against a frozen pre-refactor reference, an independent finite-difference gradient check, and end-to-end smoke tests of `optimize_multi_state_pulse`, `refine_pulse`, and `refine_pulse_dt`.

### Truncation convergence and wall exploitation

`validate_pulse_truncations` and `truncation_convergence.py` evaluate the bare fidelity $F_N$ of a trained pulse over a range of cavity truncations extending well beyond the training value. This implements the validity criterion of Heeres *et al.* (Supplementary Note 2, Eqs. 23–24): $F_N$ should plateau once $N$ exceeds the training truncation. A finite-dimensional cavity model is a faithful proxy for the infinite oscillator only when the dynamics remain interior to the truncated space.

Earlier multi-truncation pulses (`pulses/*_mt.npy`) frequently failed this test. Five of nine gates (`U_enc`, `U_X`, `U_Y`, `U_Z`, `U_T`) attained high fidelity (0.96–0.98) at the training truncation $n_c=26$, yet collapsed to 0.55–0.73 for $n_c > 26$. The optimizer had exploited artificial reflection at the Hilbert-space boundary—the failure mode anticipated by Heeres *et al*.

Two partial remedies proved insufficient:

- Enlarging `trunc_list` within the previous max-truncation architecture merely relocated wall exploitation to the new maximum truncation.
- Averaging fidelity at every iteration (Eq. 23 alone) while warm-starting from an already overfit pulse improved consistency *within* the trained range, but fidelity still degraded beyond that range.

Convergence was restored by implementing the full Heeres recipe with a cold start:

- **Eqs. 23 and 24, recomputed at every iteration.** `optimize_multi_state_pulse` optionally applies a discrepancy penalty $\sum_{k_1 \neq k_2} (F_{k_1} - F_{k_2})^2$ (`penalties['disc']`, default `0.0`), constructed from the same per-truncation fidelity evaluations used for the averaged objective—without additional propagation or stale penalty terms.
- **Cold start rather than warm start.** Because L-BFGS-B is a local method, refinement of a wall-exploiting pulse tends to remain in that basin. Optimization from random initialization yielded qualitatively different solutions; for `U_Z` and `U_T` in particular, nearly pure-phase gates driven primarily through the transmon with negligible cavity amplitude, consistent with the paper's own T and I controls (Supplementary Figure 4).

All eight logical-gate pulses (`U_enc`, `U_dec`, `U_X`, `U_Y`, `U_Z`, `U_H`, `U_T`, `U_I`) were retrained under this protocol and, subsequently, under the endpoint ramp: cold start, `trunc_list=[22,24,26]`, `maxiter=1500`, saved as `pulses/u_*_main.npy` (with `x_*_main.npy` pre-images). Across the six logical gates, $F_{\rm pedersen}$ over $n_c = 18$–32 now spans 0.9899–0.9998, flat above the trained range. The retired max-truncation module has been removed; only `pulses/u_opt_mt.npy` remains from that procedure.

One row deserves reading carefully rather than being taken as a failure: $U_Y$'s minimum, 0.98985, is the only value under 0.99, and the entire deficit sits at $n_c=18$ — the *lowest* truncation swept, and below the trained range. Y then reads 0.99645 at $n_c=20$, 0.99836, 0.99872, 0.99877, and is dead flat at 0.998767 from $n_c=26$ through 32. The Heeres criterion asks that fidelity plateau *above* the training truncation, which it does exactly. A dip *below* it means the pulse genuinely requires $n_c \geq 20$ to be represented — the opposite of exploiting the truncation wall. `analysis/rescore_saved_pulses.py` flags the held-out spread automatically and prints this caveat next to the flag.

### Validation

`validate_logical_gates.py` subjects each pulse to a five-tier protocol:

1. **Fidelity robustness** — fidelity as a function of cavity truncation.
2. **Logical action and leakage** — correct action on the cat subspace without population loss from the code space.
3. **Gate algebra** — consistency relations such as $X^2 \approx I$, $H^2 \approx I$, and the expected relative phase of $T$.
4. **Encode–gate–decode pipeline** — end-to-end fidelity relevant to a logical qubit.
5. **Effective unitary extraction** — reconstruction of the realized single-qubit unitary and comparison with the ideal target.

Results are summarized in tabular form via `pandas`.

`QuTip/qutip_validate.py` provides an independent cross-check. Most of this repository, including the optimizer, relies on a common hand-implemented eigendecomposition propagator; an error in that path could still yield self-consistently high fidelity. The validation script reconstructs operators and Hamiltonians independently with QuTiP (without importing `grape_core.make_ops`) and evolves each saved pulse with `qutip.sesolve`, using zero-order-hold coefficients to preserve the piecewise-constant control convention. Agreement at the level of $\lesssim 10^{-5}$ (within solver tolerance) supports correctness of the physical model, not merely internal consistency of the codebase.

### Decoherence simulation

`analysis/decoherence.py` re-propagates an optimized pulse under the **Lindblad master equation** to estimate decoherence-limited fidelity. Jump operators include cavity relaxation ($\kappa$), transmon relaxation ($\gamma$), and transmon pure dephasing ($\gamma_\phi$), with rates set by $T_1^C$, $T_1^T$, and $T_\phi$ (specified in seconds and converted to the simulation's microsecond time base).

The density matrix is vectorized to dimension $d^2$ and evolved with `scipy.sparse.linalg.expm_multiply`, which applies $e^{\mathcal{L}\,dt}$ to the state vector without forming the dense $d^2 \times d^2$ matrix exponential at each step.

```python
import numpy as np
from core.grape_core import basis_state
from analysis.decoherence import simulate_with_decoherence, compute_fidelity

u = np.load("pulses/u_opt.npy")
psi0 = basis_state(n_t=3, n_c=24, t_level=0, c_level=0)      # |g,0⟩
psi_target = basis_state(n_t=3, n_c=24, t_level=0, c_level=6) # |g,6⟩

rho_final = simulate_with_decoherence(u, psi0)
print(compute_fidelity(rho_final, psi_target))
```

## Project layout

| File | Description |
|------|-------------|
| `core/grape_core.py` | Hamiltonian construction, propagation, fidelity, gradients, and penalties |
| `core/cat_code.py` | Cat-state generation, encode/decode and logical-gate targets, truncation validation |
| `core/fourier_cutoff.py` | Hard frequency-band projection of controls and gradients (Heeres Supplementary Eq. 22) |
| `core/ramp.py` | Gaussian rise/fall envelope (`ramp_envelope`), the raw-variable→pulse constraint chain (`make_constraint_chain`), its inverse (`deramp`), and per-pulse endpoint/out-of-band measurement (`constraint_report`) |
| `core/optimizer.py` | `optimize_multi_state_pulse()`, `refine_pulse()` — averaged fidelity (Eq. 23), optional Eq. 24 discrepancy penalty, `ramp_ns` / `hard_amp_limit` |
| `core/compare_pulses.py` | Shared pulse-comparison table (fidelity + shape metrics) reused by visualization/validation scripts |
| `validation/test_grape_core_perf.py` | Regression suite for the batched fidelity core: pre-refactor equivalence, finite-difference gradients, optimizer smoke tests |
| `validation/truncation_convergence.py` | Truncation sweep of bare fidelity (Eqs. 23–24 validity criterion) |
| `validation/validate_logical_gates.py` | Five-tier validation suite for logical-gate pulses |
| `analysis/logical_gate_analysis.py` | Standalone encode-optimization script |
| `analysis/logical_gate_analysis1.py` | Gate-optimization examples and encode/decode round-trip checks |
| `analysis/refine_dt_and_compare.py` | Pulse refinement with pre-/post-refinement fidelity comparison |
| `QuTip/qutip_validate.py` | Independent fidelity cross-check via `qutip.sesolve` |
| `QuTip/qutip_grape_optimizer.py` | Pedagogical GRAPE implementation on QuTiP `Qobj` physics — adjoint/L-BFGS-B structure as in `optimizer.py`, without band limiting, discrepancy penalty, or joblib |
| `analysis/pulse_analysis.py` | Trajectory simulation, Fock populations, and basic optimization demonstration |
| `analysis/decoherence.py` | Lindblad evolution with $T_1$/$T_\phi$; decoherence-limited fidelity |
| `visualization/pulse_viz.py` | I/Q waveforms and complex-envelope FFT spectra |
| `visualization/wigner_viz.py` | Wigner tomography of Fock states, cat states, and pulse-propagated states |
| `visualization/plot_qutip_validation.py` | Small-multiples figure of the QuTiP cross-check results per gate |
| `pulses/` | Saved control sequences (`u_*.npy`, shape $(N, 4)$) and their raw optimizer pre-images (`x_*.npy`, same shape). Only $u$ is physical and only $u$ is scored; $x$ exists to resume a run exactly via `init_x` |
| `figures/` | Generated waveform and spectrum figures |
| `wigner/` | Generated Wigner-function figures |

## Quick start

```bash
pip install -r requirements.txt
pip install joblib matplotlib pandas   # analysis scripts
pip install qutip                      # QuTip/qutip_validate.py
```

**Logical-gate optimization.** This is the exact recipe that produced the
`pulses/u_*_main.npy` set — the library defaults differ (notably
`hard_amp_limit=50` and no band limit), so pass these explicitly to reproduce:

```python
from core.cat_code import make_coherent_gate_factory, validate_pulse_truncations
from core.grape_core import coherent_fidelity_multi_state
from core.optimizer import optimize_multi_state_pulse

# The coherent objective must be paired with the coherent state-pair factory
# (canonical pairs from IDEAL_LOGICAL_U), not with get_logical_X_state_pairs.
get_X_pairs = make_coherent_gate_factory("X")

u_X, info = optimize_multi_state_pulse(
    get_state_pairs=get_X_pairs,
    trunc_list=[22, 24, 26],
    penalties={"deriv": 1e-5, "amp": 8e-5, "amp_max": 40.0, "disc": 0.5},
    warm_start=None,              # cold start; seeded by warm_start_seed
    warm_start_seed=42,           # NOTE: U_Y requires seed 44 (see the ramp section)
    cav_band=(-27.0, 27.0),
    tra_band=(-33.0, 33.0),
    ramp_ns=48.0,                 # Gaussian rise/fall; replaces the old boundary penalty
    hard_amp_limit=40.0,
    n_jobs=3,
    maxiter=1500,
    fidelity_fn=coherent_fidelity_multi_state,
    save_path="pulses/u_X_logical_v1.npy",
)
assert info["max_abs_preimage"] < 40.0     # a pulse at the bound was clipped, not converged
validate_pulse_truncations(u_X, get_X_pairs)
```

The same run is available from the command line, where these values are already
the defaults:

```bash
python main.py --gate X                  # seed 42, ramp_ns 48, hard_amp_limit 40
python main.py --gate Y --seed 44        # Y only: seed 42 drives max|x| into the box
python main.py --gate X --ramp-ns 0      # disable the ramp (0 means off)
```

**Refine and compare** an existing pulse:

```bash
# Set INPUT_PULSE_PATH / GET_STATE_PAIRS in analysis/refine_dt_and_compare.py, then, from the repo root:
python analysis/refine_dt_and_compare.py
```

**Full validation suite** on pulses in `pulses/`:

```bash
python validation/validate_logical_gates.py
```

**Independent QuTiP cross-check** of all saved pulses:

```bash
python QuTip/qutip_validate.py
```

**Pulse visualization**:

```python
from visualization.pulse_viz import plot_pulse_waveforms, plot_pulse_spectrum
import numpy as np

u = np.load("pulses/u_enc_v2.npy")
plot_pulse_waveforms(u, title="U_enc")
plot_pulse_spectrum(u, title="U_enc Spectrum")
```

**Wigner tomography of a pulse-prepared state**:

```python
from visualization.wigner_viz import plot_wigner_from_pulse

# Transmon initial state is relevant when the output depends on it
# (e.g. encode: |g,0> -> +Z_L, |e,0> -> -Z_L)
plot_wigner_from_pulse("pulses/u_enc_refined_t3v2.npy", initial_state="vacuum",
                       transmon='g', title="+Z_L after encoding")
plot_wigner_from_pulse("pulses/u_enc_refined_t3v2.npy", initial_state="vacuum",
                       transmon='e', title="-Z_L after encoding")
```

## Default parameters

| Parameter | Value |
|-----------|-------|
| $\chi$ | $-2.194$ MHz |
| Kerr | $-0.0037$ MHz |
| $\chi'$ | $-0.019$ MHz |
| $\alpha$ | $-236$ MHz |
| $dt$ | 0.002 μs |
| $N$ | 550 |
| $\alpha_{\mathrm{cat}}$ | $\sqrt{3}$ |
| $n_t$ | 2–3 (use 3 when transmon leakage is relevant) |
| $n_c$ | 20–28 (training); validated up to 30 |

**Production training recipe** (`main.py` CLI defaults; mirrored by
`analysis/penalty_sweep.py`'s `FIXED` and by `experiments.ipynb`'s
`OPTIMIZATION_RECIPE`). Several differ from the library defaults of
`optimize_multi_state_pulse`, which are noted where they do:

| Parameter | Value | Library default |
|-----------|-------|-----------------|
| `trunc_list` | $[22, 24, 26]$ | $[20, 24, 28]$ |
| `maxiter` | 1500 | 2000 |
| `ramp_ns` | 48 ns | 48 ns (but `None` on `make_constraint_chain`) |
| `hard_amp_limit` | 40 rad/μs | 50 (`refine_pulse*`: 40) |
| `amp_max` (soft) | 40 rad/μs | 40 |
| `deriv` | $10^{-5}$ | $10^{-5}$ |
| `amp` | $8\times10^{-5}$ | $1.2\times10^{-4}$ |
| `disc` | 0.5 | 0.0 |
| `cav_band` | $\pm 27$ MHz | `None` (off) |
| `tra_band` | $\pm 33$ MHz | `None` (off) |
| `warm_start_seed` | 42 — **except $U_Y$, which uses 44** | 42 |
| `fidelity_fn` | `coherent_fidelity_multi_state` | `fidelity_multi_state` |

## References

- Heeres, R. W., Reinhold, P., Ofek, N., *et al.* Implementing a universal gate set on a logical qubit encoded in an oscillator. *Nature Communications* **8**, 94 (2017). [doi:10.1038/s41467-017-00045-1](https://www.nature.com/articles/s41467-017-00045-1) — **primary reference**
- Khaneja, N., Reiss, T., Kehlet, C., Schulte-Herbrüggen, T. & Glaser, S. J. Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent algorithms. *J. Magn. Reson.* **172**, 296–305 (2005). [doi:10.1016/j.jmr.2004.11.004](https://doi.org/10.1016/j.jmr.2004.11.004)
- Blais, A., Huang, R.-S., Wallraff, A., Girvin, S. M. & Schoelkopf, R. J. Cavity quantum electrodynamics for superconducting electrical circuits: an architecture for quantum computation. *Phys. Rev. A* **69**, 062320 (2004). [arXiv:cond-mat/0402216](https://arxiv.org/abs/cond-mat/0402216)
