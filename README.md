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

For multi-state gates, $F$ is averaged over all target pairs (e.g. both encoding branches, or both logical basis states).

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

onto the band-limited subspace, applied separately to the cavity drive $\varepsilon_C = C_I + iC_Q$ and the transmon drive $\varepsilon_T = T_I + iT_Q$. Since $P$ is idempotent and self-adjoint with respect to the real inner product, the same operator projects both the pulse and its gradient. The constraint is enabled by passing `cav_band`/`tra_band` to `optimize_multi_state_pulse`; setting them to `None` disables it.

### Solver and regularization

| Component | Role |
|-----------|------|
| **L-BFGS-B** (`scipy.optimize.minimize`) | Bound-constrained quasi-Newton minimization of $-F$ |
| **Derivative penalty** $\sum_k \|u_{k+1} - u_k\|^2$ | Temporal smoothness of the control |
| **Boundary penalty** $\|u_0\|^2 + \|u_N\|^2$ | Vanishing drive amplitude at the endpoints |
| **Amplitude penalty** | Soft bound on $\|u\|_\infty$ (default 40 rad/μs) |
| **Hard amplitude bound** | Box constraint on the raw L-BFGS-B variable (default 50 rad/μs) |
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

All eight logical-gate pulses (`U_enc`, `U_dec`, `U_X`, `U_Y`, `U_Z`, `U_H`, `U_T`, `U_I`) were retrained under this protocol (`trunc_list=[22,24,26]`, saved as `pulses/u_*_main.npy`). The five previously non-convergent gates now exhibit flat fidelity from $n_c=18$ to $n_c=40$ at 0.995–0.9998. `U_opt` was already truncation-convergent and was not retrained. The retired max-truncation module has been removed; only `pulses/u_opt_mt.npy` remains from that procedure.

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
| `core/optimizer.py` | `optimize_multi_state_pulse()`, `refine_pulse()` — averaged fidelity (Eq. 23), optional Eq. 24 discrepancy penalty |
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
| `pulses/` | Saved control sequences (`u_*.npy`, shape $(N, 4)$) |
| `figures/` | Generated waveform and spectrum figures |
| `wigner/` | Generated Wigner-function figures |

## Quick start

```bash
pip install -r requirements.txt
pip install joblib matplotlib pandas   # analysis scripts
pip install qutip                      # QuTip/qutip_validate.py
```

**Logical-gate optimization** (from `analysis/logical_gate_analysis1.py`):

```python
from core.cat_code import get_logical_X_state_pairs, validate_pulse_truncations
from core.optimizer import optimize_multi_state_pulse

u_X, info = optimize_multi_state_pulse(
    get_state_pairs=get_logical_X_state_pairs,
    trunc_list=[22, 24, 26],
    save_path="pulses/u_X_logical_v1.npy",
)
validate_pulse_truncations(u_X, get_logical_X_state_pairs)
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

## References

- Heeres, R. W., Reinhold, P., Ofek, N., *et al.* Implementing a universal gate set on a logical qubit encoded in an oscillator. *Nature Communications* **8**, 94 (2017). [doi:10.1038/s41467-017-00045-1](https://www.nature.com/articles/s41467-017-00045-1) — **primary reference**
- Khaneja, N., Reiss, T., Kehlet, C., Schulte-Herbrüggen, T. & Glaser, S. J. Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent algorithms. *J. Magn. Reson.* **172**, 296–305 (2005). [doi:10.1016/j.jmr.2004.11.004](https://doi.org/10.1016/j.jmr.2004.11.004)
- Blais, A., Huang, R.-S., Wallraff, A., Girvin, S. M. & Schoelkopf, R. J. Cavity quantum electrodynamics for superconducting electrical circuits: an architecture for quantum computation. *Phys. Rev. A* **69**, 062320 (2004). [arXiv:cond-mat/0402216](https://arxiv.org/abs/cond-mat/0402216)
