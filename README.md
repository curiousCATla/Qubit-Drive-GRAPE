# Qubit-Drive-GRAPE

GRAPE (Gradient Ascent Pulse Engineering) optimization of microwave drive waveforms for a **transmon–cavity** circuit-QED system. The pulses are designed for Fock-state preparation, **even-parity cat-code** encoding/decoding, and single-qubit logical gates on the resulting cat-code qubit.

## Physics

### System model

A dispersively coupled transmon and cavity, simulated in a rotating frame with truncated Fock bases. The joint Hilbert space is $\mathcal{H}_T \otimes \mathcal{H}_C$ (dimension $n_t \times n_c$), with annihilation operators $B$ (transmon) and $A$ (cavity).

**Drift Hamiltonian** (lab parameters in MHz, converted to rad/μs):

$$
H_0 = \chi\, n_A n_B + \frac{K}{2}\, A^{\dagger 2} A^2 + \frac{\chi'}{2}\, n_B\, A^{\dagger 2} A^2 + \frac{\alpha}{2}\, B^{\dagger 2} B^2
$$

| Term | Physical origin |
|------|-----------------|
| $\chi\, n_A n_B$ | Dispersive transmon–cavity coupling |
| Kerr $A^{\dagger 2}A^2$ | Cavity self-Kerr / oscillator anharmonicity |
| $\chi'\, n_B\, A^{\dagger 2}A^2$ | Second-order dispersive shift |
| $\alpha\, B^{\dagger 2}B^2$ | Transmon anharmonicity (included when $n_t \geq 3$) |

**Control Hamiltonian** — four real drive channels per time step:

$$
H_c = \varepsilon_{C,I}(A+A^\dagger) + \varepsilon_{C,Q}\, i(A-A^\dagger) + \varepsilon_{T,I}(B+B^\dagger) + \varepsilon_{T,Q}\, i(B-B^\dagger)
$$

Each step uses $H_k = H_0 + \sum_j u_{k,j}\, H_{c,j}$ with $dt = 2$ ns and $N \approx 550$ steps (~1.1 μs total).

### Cat-code logical qubit

Logical states follow the **even 4-component cat code** (Heeres *et al.*, 2017):

- $|{+Z_L}\rangle$: coherent superposition on Fock levels $n = 0, 4, 8, \ldots$
- $|{-Z_L}\rangle$: superposition on $n = 2, 6, 10, \ldots$

with amplitude $\alpha = \sqrt{3}$. Encoding maps transmon computational states $|g,0\rangle, |e,0\rangle$ to $|g\rangle \otimes |{\pm Z_L}\rangle$; decoding reverses this. Logical gates (X, Y, Z, H, T, I) act on the cat subspace while the transmon is held in $|g\rangle$.

## Optimization

### GRAPE objective

For each target state transfer $|i\rangle \to |f\rangle$, the unitary $U = U_N \cdots U_1$ is built from piecewise-constant controls. Fidelity is

$$
F = \bigl|\langle f | U | i \rangle\bigr|^2
$$

Multi-state gates average $F$ over all target pairs (e.g. both branches of encode, or both logical basis states).

### Analytic gradients

Gradients are computed via the **adjoint (costate) method**, not finite differences:

1. **Forward pass** — propagate $|\phi_k\rangle = U_k \cdots U_1 |i\rangle$ and store the eigen-decomposition of each $H_k$.
2. **Backward pass** — propagate costates $\langle\lambda_k| = \langle f| U_N^\dagger \cdots U_k^\dagger$.
3. **Per-step derivative** — use $\partial U_k / \partial u_j = V \bigl(\Phi \circ V^\dagger H_{c,j} V\bigr) V^\dagger$, where $\Phi$ handles near-degenerate eigenvalue pairs (e.g. $|g,0\rangle$–$|g,1\rangle$ in the rotating frame).

### Frequency-band-limited controls

To keep pulses within realistic AWG/filter bandwidths, controls can be constrained to a hard frequency band following Heeres *et al.* 2017 (Supplementary Eq. 22). Rather than penalizing out-of-band content, `fourier_cutoff.py` enforces it exactly: the raw L-BFGS-B variable $x$ is treated as a pre-image, and the physical pulse is the orthogonal projection

$$
u = P(x), \qquad P = \mathrm{IFFT} \circ \text{mask} \circ \mathrm{FFT}
$$

onto the band-limited subspace, applied separately to the cavity drive $\varepsilon_C = C_I + iC_Q$ and transmon drive $\varepsilon_T = T_I + iT_Q$. Because $P$ is idempotent and self-adjoint under the real inner product, the same function projects both the pulse and its gradient, so the chain rule needs no extra bookkeeping. Passing `cav_band`/`tra_band` to `optimize_multi_state_pulse` turns this on; leaving them `None` disables it.

### Solver and regularization

| Tool | Role |
|------|------|
| **L-BFGS-B** (`scipy.optimize.minimize`) | Bound-constrained quasi-Newton minimization of $-F$ |
| **Derivative penalty** $\sum_k \|u_{k+1} - u_k\|^2$ | Pulse smoothness |
| **Boundary penalty** $\|u_0\|^2 + \|u_N\|^2$ | Drives start and end at zero |
| **Amplitude penalty** | Soft limit on $\|u\|_\infty$ (default 40 rad/μs) |
| **Hard amplitude bound** | Box constraint on the raw L-BFGS-B variable (default 50 rad/μs), independent of the soft penalty above |
| **Frequency band limit** | Optional hard cutoff via orthogonal projection (see above) |
| **Multi-truncation training** | Average fidelity over several $n_c$ values (e.g. 20, 24, 28) to improve robustness to Hilbert-space truncation |
| **Parallel evaluation** (`joblib`) | Independent fidelity/gradient calls per truncation |
| **Warm start** | Low-pass random initialization or loading a saved `.npy` pulse |

Time-step propagators use `numpy.linalg.eigh` rather than matrix exponentials directly: $U_k = V \,\mathrm{diag}(e^{-i\,dt\,\omega})\, V^\dagger$.

### Validation

Once a pulse is optimized (and ideally refined — see below), `validate_logical_gates.py` runs it through a five-tier check before it's trusted as a usable logical gate:

1. **Fidelity robustness** — sweep fidelity across a wide range of cavity truncations.
2. **Logical action & leakage** — confirm the gate acts correctly on the cat subspace and doesn't leak population out of the code space.
3. **Gate algebra** — self-consistency checks such as $X^2 \approx I$, $H^2 \approx I$, and the correct relative phase for $T$.
4. **Encode–gate–decode pipeline** — the end-to-end fidelity that actually matters for a usable logical qubit.
5. **Effective unitary extraction** — recover the realized single-qubit unitary and compare it to the ideal target.

Results are summarized in `pandas` tables for quick inspection.

### Decoherence simulation

`decoherence.py` re-evolves an optimized pulse under the **Lindblad master equation** to estimate the realistic, decoherence-limited fidelity (vs. the closed-system fidelity GRAPE optimizes against). Jump operators are cavity relaxation ($\kappa$), transmon relaxation ($\gamma$), and transmon pure dephasing ($\gamma_\phi$), with rates set from $T_1^C$, $T_1^T$, $T_\phi$ (given in seconds, converted internally to the simulation's μs time base to match `dt`).

The density matrix is vectorized ($d^2$-dimensional, row-major/`order='C'` to match the Kronecker convention used to build the Liouvillian $\mathcal{L}$) and propagated with `scipy.sparse.linalg.expm_multiply`, which applies $e^{\mathcal{L}\,dt}$ directly to the state vector instead of forming the dense $d^2\times d^2$ matrix exponential at every step — the latter is intractable even for modest truncations (e.g. $n_c=24 \Rightarrow d^2 = 5184$).

```python
import numpy as np
from grape_core import basis_state
from decoherence import simulate_with_decoherence, compute_fidelity

u = np.load("pulses/u_opt.npy")
psi0 = basis_state(n_t=3, n_c=24, t_level=0, c_level=0)      # |g,0⟩
psi_target = basis_state(n_t=3, n_c=24, t_level=0, c_level=6) # |g,6⟩

rho_final = simulate_with_decoherence(u, psi0)
print(compute_fidelity(rho_final, psi_target))
```

## Project layout

| File | Purpose |
|------|---------|
| `grape_core.py` | Hamiltonian construction, propagation, fidelity, gradients, penalties |
| `cat_code.py` | Cat-state generation, encode/decode/logical-gate target factories, truncation validation |
| `fourier_cutoff.py` | Hard frequency-band projection for controls and gradients (Heeres Supplementary Eq. 22) |
| `optimizer.py` | `optimize_multi_state_pulse()`, `refine_pulse()` |
| `logical_gate_analysis.py` | Standalone encode optimization script |
| `logical_gate_analysis1.py` | Gate optimization examples + encode/decode round-trip check |
| `refine_and_compare.py` | Refine an existing pulse and compare fidelity before/after |
| `validate_logical_gates.py` | Five-tier validation suite for refined logical-gate pulses |
| `pulse_analysis.py` | Trajectory simulation, Fock populations, basic optimization demo |
| `decoherence.py` | Lindblad master-equation simulation with $T_1$/$T_\phi$ decoherence; reports decoherence-limited fidelity |
| `pulse_viz.py` | I/Q waveform and complex-envelope FFT spectrum plots |
| `wigner_viz.py` | Wigner-function tomography: Fock states, cat states, and states propagated through a saved pulse |
| `pulses/` | Saved control sequences (`u_*.npy`, shape `(N, 4)`) |
| `figures/` | Generated waveform/spectrum plots |
| `wigner/` | Generated Wigner-function plots |

## Quick start

```bash
pip install -r requirements.txt
pip install joblib matplotlib pandas   # used by analysis scripts
```

**Optimize a logical gate** (example from `logical_gate_analysis1.py`):

```python
from cat_code import get_logical_X_state_pairs, validate_pulse_truncations
from optimizer import optimize_multi_state_pulse

u_X, info = optimize_multi_state_pulse(
    get_state_pairs=get_logical_X_state_pairs,
    trunc_list=[22, 24, 26],
    save_path="pulses/u_X_logical_v1.npy",
)
validate_pulse_truncations(u_X, get_logical_X_state_pairs)
```

**Refine and compare** an existing pulse:

```bash
# Edit GATE in refine_and_compare.py, then:
python refine_and_compare.py
```

**Run the full validation suite** on refined pulses in `pulses/`:

```bash
python validate_logical_gates.py
```

**Visualize a pulse**:

```python
from pulse_viz import plot_pulse_waveforms, plot_pulse_spectrum
import numpy as np

u = np.load("pulses/u_enc_v2.npy")
plot_pulse_waveforms(u, title="U_enc")
plot_pulse_spectrum(u, title="U_enc Spectrum")
```

**Wigner tomography of a pulse-prepared state**:

```python
from wigner_viz import plot_wigner_from_pulse

# transmon initial state matters for pulses whose output depends on it
# (e.g. an encoder maps |g,0> -> +Z_L and |e,0> -> -Z_L)
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
| $n_t$ | 2–3 (3 when transmon leakage matters) |
| $n_c$ | 20–28 (training); validated up to 30 |

## References

- Khaneja, Reiss, Kehlet, *et al.* — GRAPE: optimal control of spin systems ([JMR 2005](https://doi.org/10.1016/j.jmr.2004.11.004))
- Heeres, *et al.* — Even-parity cat encoding ([PRL 2017](https://doi.org/10.1103/PhysRevLett.119.060501))
- Blais, Huang, Wallraff, Girvin — Circuit QED ([arxiv:cond-mat/0402216](https://arxiv.org/abs/cond-mat/0402216))
