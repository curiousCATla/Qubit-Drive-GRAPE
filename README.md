# Qubit-Drive-GRAPE

GRAPE (Gradient Ascent Pulse Engineering) optimization of microwave drive waveforms for a **transmon–cavity** circuit-QED system. Pulses are designed for Fock-state preparation, **even-parity cat-code** encoding/decoding, and single-qubit logical gates.

## Physics

### System model

A dispersively coupled transmon and cavity, simulated in a rotating frame with truncated Fock bases. The joint Hilbert space is \(\mathcal{H}_T \otimes \mathcal{H}_C\) (dimension \(n_t \times n_c\)), with annihilation operators \(B\) (transmon) and \(A\) (cavity).

**Drift Hamiltonian** (lab parameters in MHz, converted to rad/μs):

\[
H_0 = \chi\, n_A n_B + \tfrac{K}{2}\, A^{\dagger 2} A^2 + \tfrac{\chi'}{2}\, n_B\, A^{\dagger 2} A^2 + \tfrac{\alpha}{2}\, B^{\dagger 2} B^2
\]

| Term | Physical origin |
|------|-----------------|
| \(\chi\, n_A n_B\) | Dispersive transmon–cavity coupling |
| Kerr \(A^{\dagger 2}A^2\) | Cavity self-Kerr / oscillator anharmonicity |
| \(\chi'\, n_B\, A^{\dagger 2}A^2\) | Second-order dispersive shift |
| \(\alpha\, B^{\dagger 2}B^2\) | Transmon anharmonicity (included when \(n_t \geq 3\)) |

**Control Hamiltonian** — four real drive channels per time step:

\[
H_c = \varepsilon_{C,I}(A+A^\dagger) + \varepsilon_{C,Q}\, i(A-A^\dagger) + \varepsilon_{T,I}(B+B^\dagger) + \varepsilon_{T,Q}\, i(B-B^\dagger)
\]

Each step uses \(H_k = H_0 + \sum_j u_{k,j}\, H_{c,j}\) with \(dt = 2\) ns and \(N \approx 550\) steps (~1.1 μs total).

### Cat-code logical qubit

Logical states follow the **even 4-component cat code** (Heeres *et al.*, 2017):

- \(|{+Z_L}\rangle\): coherent superposition on Fock levels \(n = 0, 4, 8, \ldots\)
- \(|-Z_L\rangle\): superposition on \(n = 2, 6, 10, \ldots\)

with amplitude \(\alpha = \sqrt{3}\). Encoding maps transmon computational states \(|g,0\rangle, |e,0\rangle\) to \(|g\rangle \otimes |{\pm Z_L}\rangle\); decoding reverses this. Logical gates (X, Y, Z, H, T, I) act on the cat subspace with the transmon held in \(|g\rangle\).

## Optimization

### GRAPE objective

For each target state transfer \(|i\rangle \to |f\rangle\), the unitary \(U = U_N \cdots U_1\) is built from piecewise-constant controls. Fidelity is

\[
F = \bigl|\langle f | U | i \rangle\bigr|^2
\]

Multi-state gates average \(F\) over all target pairs (e.g. both branches of encode or both logical basis states).

### Analytic gradients

Gradients are computed via the **adjoint (costate) method**, not finite differences:

1. **Forward pass** — propagate \(|\phi_k\rangle = U_k \cdots U_1 |i\rangle\) and store eigen-decompositions of each \(H_k\).
2. **Backward pass** — propagate costates \(\langle\lambda_k| = \langle f| U_N^\dagger \cdots U_k^\dagger\).
3. **Per-step derivative** — use \(\partial U_k / \partial u_j = V \bigl(\Phi \circ V^\dagger H_{c,j} V\bigr) V^\dagger\), where \(\Phi\) handles near-degenerate eigenvalue pairs (e.g. \(|g,0\rangle\)–\(|g,1\rangle\) in the rotating frame).

### Solver and regularization

| Tool | Role |
|------|------|
| **L-BFGS-B** (`scipy.optimize.minimize`) | Bound-constrained quasi-Newton minimization of \(-F\) |
| **Derivative penalty** \(\sum_k \|u_{k+1} - u_k\|^2\) | Pulse smoothness |
| **Boundary penalty** \(\|u_0\|^2 + \|u_N\|^2\) | Drives start and end at zero |
| **Amplitude penalty** | Soft limit on \(\|u\|_\infty\) (default 40 rad/μs) |
| **Multi-truncation training** | Average fidelity over several \(n_c\) values (e.g. 20, 24, 28) to improve robustness to Hilbert-space truncation |
| **Parallel evaluation** (`joblib`) | Independent fidelity/gradient calls per truncation |
| **Warm start** | Low-pass random initialization or loading a saved `.npy` pulse |

Time-step propagators use `numpy.linalg.eigh` rather than matrix exponentials directly: \(U_k = V \,\mathrm{diag}(e^{-i\,dt\,\omega})\, V^\dagger\).

## Project layout

| File | Purpose |
|------|---------|
| `grape_core.py` | Hamiltonian construction, propagation, fidelity, gradients, penalties |
| `cat_code.py` | Cat-state generation, encode/decode/logical-gate target factories, truncation validation |
| `optimizer.py` | `optimize_multi_state_pulse()`, `refine_pulse()` |
| `logical_gate_analysis.py` | Standalone encode optimization script |
| `logical_gate_analysis1.py` | Gate optimization examples + encode/decode round-trip check |
| `refine_and_compare.py` | Refine an existing pulse and compare fidelity before/after |
| `pulse_analysis.py` | Trajectory simulation, Fock populations, basic optimization demo |
| `pulse_viz.py` | I/Q waveform and FFT spectrum plots |
| `pulses/` | Saved control sequences (`u_*.npy`, shape `(N, 4)`) |
| `figures/` | Generated plots |

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

**Visualize a pulse**:

```python
from pulse_viz import plot_pulse_waveforms, plot_pulse_spectrum
import numpy as np

u = np.load("pulses/u_enc_v2.npy")
plot_pulse_waveforms(u, title="U_enc")
plot_pulse_spectrum(u, title="U_enc Spectrum")
```

## Default parameters

| Parameter | Value |
|-----------|-------|
| \(\chi\) | \(-2.194\) MHz |
| Kerr | \(-0.0037\) MHz |
| \(\chi'\) | \(-0.019\) MHz |
| \(\alpha\) | \(-236\) MHz |
| \(dt\) | 0.002 μs |
| \(N\) | 550 |
| \(\alpha_{\mathrm{cat}}\) | \(\sqrt{3}\) |
| \(n_t\) | 2–3 (3 when transmon leakage matters) |
| \(n_c\) | 20–28 (training); validated up to 30 |

## References

- Khaneja, Reiss, Kehlet, *et al.* — GRAPE: optimal control of spin systems ([JMR 2005](https://doi.org/10.1016/j.jmr.2004.11.004))
- Heeres, *et al.* — Even-parity cat encoding ([PRL 2017](https://doi.org/10.1103/PhysRevLett.119.060501))
- Blais, Huang, Wallraff, Girvin — Circuit QED ([arxiv:cond-mat/0402216](https://arxiv.org/abs/cond-mat/0402216))