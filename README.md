# Qubit-Drive-GRAPE

GRAPE (Gradient Ascent Pulse Engineering) optimization of microwave drive waveforms for a **transmon–cavity** circuit-QED system. The pulses are designed for Fock-state preparation, **even-parity cat-code** encoding/decoding, and single-qubit logical gates on the resulting cat-code qubit.

This project is based entirely on the scheme demonstrated in Heeres, R. W. *et al.* ["Implementing a universal gate set on a logical qubit encoded in an oscillator"](https://www.nature.com/articles/s41467-017-00045-1), *Nature Communications* **8**, 94 (2017). All Hamiltonian conventions, the cat-code definition, and the frequency-band-limited pulse constraint reproduce that paper's approach.

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

### Max-truncation refinement (retired)

An earlier revision of this project used a second optimizer module, `optimizer_max_trunc.py`, for pulses that needed to hold up past their training truncation. Unlike `optimizer.py`'s multi-truncation mode (which averages fidelity and its gradient over every $n_c$ in `trunc_list` on **every** L-BFGS-B call), it drove L-BFGS-B primarily with the bare fidelity at only `max(trunc_list)`, with a separate **consistency penalty** $\sum_k (F_{\max} - F_k)^2$ against the lower truncations — refreshed (and held frozen between refreshes) only every `refresh_every` iterations rather than every call, to keep it cheap. Applied as a two-stage "refine + restart" recipe, this produced every `pulses/*_mt.npy` file, which `qutip_validate.py` (above) cross-checks.

That staleness turned out to be exactly what let the optimizer exploit the finite-truncation wall — see "Challenge" below. `optimizer_max_trunc.py` and the two scripts that depended on it (`refine_all_gates_max_trunc.py`, `refine_all_mt_with_leak.py`) have since been removed in favor of `optimizer.py`'s cold-start Eq. 23+24 recipe. `pulses/u_opt_mt.npy` is the one pulse still in active use that this retired recipe produced — `U_opt` was already fully convergent, so it was never retrained.

### Challenge: truncation wall-exploitation

`validate_pulse_truncations` / `truncation_convergence.py` sweep a trained pulse's bare fidelity $F_N$ across a wide range of truncations $N$, well past whatever the pulse was trained at. This is the paper's own validity criterion (Supplementary Note 2, Eq. 23–24): $F_N$ should plateau once $N$ exceeds the training truncation, since a finite-dimensional cavity truncation is only a valid stand-in for the real infinite-dimensional oscillator if none of the relevant dynamics reach the wall.

Sweeping the `pulses/*_mt.npy` gates this way turned up a real problem: 5 of 9 (`U_enc`, `U_X`, `U_Y`, `U_Z`, `U_T`) reported excellent fidelity (0.96–0.98) at their training truncation $n_c=26$, but collapsed to 0.55–0.73 once simulated at $n_c > 26$. The `optimizer_max_trunc.py` recipe that produced them drives L-BFGS-B with the bare fidelity at only `nc_max` as the primary objective, with the cross-truncation consistency penalty refreshed (and held stale) only every 10 iterations — enough slack for the optimizer to find a solution that exploits the artificial reflection off the $n_c=26$ Hilbert-space wall to reach the target state, rather than one that would behave the same in a real, untruncated cavity. This is exactly the failure mode Heeres *et al.* 2017 warn about: *"For generic applied drives this is not the case [dynamics staying within N]. In order to enforce this property, we modify the optimization problem..."*

Two fixes were tried and **didn't** solve it:
- Widening `trunc_list` within the same max-trunc architecture (e.g. `[20, 26, 32]`) — the wall-exploitation just relocated to the new `nc_max`.
- Switching to `optimizer.py`'s averaged-fidelity objective (the paper's Eq. 23 alone, fresh gradient every iteration, no staleness) while still warm-starting from the already-overfit pulse — fidelity became consistent *across the trained truncations* but still collapsed beyond them.

What actually fixed it was implementing the paper's **full** combined recipe and removing the warm start:
- **Eq. 23 + Eq. 24 together, both fresh every iteration** — `optimizer.py`'s `optimize_multi_state_pulse` now also computes the discrepancy penalty $\sum_{k_1 \neq k_2} (F_{k_1} - F_{k_2})^2$ (a `'disc'` key in `penalties`, defaulting to `0.0` so existing callers are unaffected) from the same per-truncation fidelity/gradient pairs it already evaluates for the Eq. 23 sum — no extra propagation needed, and no staleness, unlike the `optimizer_max_trunc.py` consistency term.
- **Cold start, not warm start.** L-BFGS-B is a local method: refining from a pulse already sitting in a wall-exploiting basin, no matter what penalty is added, tends to stay in that basin. Retraining from scratch let the optimizer find a fundamentally different, cheaper solution — for `U_Z` and `U_T` in particular, one that implements the (pure-phase) target almost entirely through the transmon with a nearly negligible cavity drive, sidestepping the truncation question altogether, mirroring how the paper's own `T`/`I` gates work (Supplementary Figure 4: *"grape finds a solution with a very small oscillator drive amplitude"*).

All 8 logical-gate pulses (`U_enc`, `U_dec`, `U_X`, `U_Y`, `U_Z`, `U_H`, `U_T`, `U_I`) have since been retrained this way (`trunc_list=[22,24,26]`, saved as new `pulses/u_*_eq23eq24_coldstart.npy` files, originals untouched). The 5 that were broken now show flat fidelity from $n_c=18$ all the way to $n_c=40$ — fully converged, at higher fidelity (0.995–0.9998) than the originals ever reached even at their own training truncation. `U_dec`, `U_H`, and `U_I` were already convergent and were redone purely for consistency, so every logical gate except `U_opt` (already fully convergent, never needed retraining) now comes from the same recipe.

### Validation

Once a pulse is optimized (and ideally refined — see below), `validate_logical_gates.py` runs it through a five-tier check before it's trusted as a usable logical gate:

1. **Fidelity robustness** — sweep fidelity across a wide range of cavity truncations.
2. **Logical action & leakage** — confirm the gate acts correctly on the cat subspace and doesn't leak population out of the code space.
3. **Gate algebra** — self-consistency checks such as $X^2 \approx I$, $H^2 \approx I$, and the correct relative phase for $T$.
4. **Encode–gate–decode pipeline** — the end-to-end fidelity that actually matters for a usable logical qubit.
5. **Effective unitary extraction** — recover the realized single-qubit unitary and compare it to the ideal target.

Results are summarized in `pandas` tables for quick inspection.

`qutip_validate.py` adds an independent cross-check on top of that: every propagator used elsewhere in this repo (`grape_core.fidelity_grad`/`fidelity_multi_state`) is the same hand-rolled `eigh`-and-exponentiate code the optimizer itself maximizes, so a bug there (wrong operator ordering, a missing `dt`/factor-of-2) could converge cleanly and still self-report a great fidelity. `qutip_validate.py` rebuilds the operators and Hamiltonian from scratch with QuTiP (not by importing `grape_core.make_ops`) and propagates each saved `pulses/*_mt.npy` waveform with `qutip.sesolve` — a separately implemented ODE integrator — using a zero-order-hold `Coefficient` (`order=0`) per control channel so the piecewise-constant pulse convention is reproduced exactly rather than smoothed by the default spline interpolation. Agreement between the two propagators (all pulses currently agree to $\lesssim 10^{-5}$, within `sesolve`'s solver tolerance) is evidence the physics is right, not just that the code is internally consistent.

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
| `optimizer.py` | `optimize_multi_state_pulse()`, `refine_pulse()` — averaged-fidelity (Eq. 23) objective, optionally with the fresh Eq. 24 discrepancy penalty (`penalties['disc']`) |
| `truncation_convergence.py` | Sweeps bare fidelity across a wide truncation range per gate to check the paper's Eq. 23–24 validity criterion (see "Challenge: truncation wall-exploitation") |
| `logical_gate_analysis.py` | Standalone encode optimization script |
| `logical_gate_analysis1.py` | Gate optimization examples + encode/decode round-trip check |
| `refine_and_compare.py` | Refine an existing pulse and compare fidelity before/after |
| `validate_logical_gates.py` | Five-tier validation suite for refined logical-gate pulses |
| `qutip_validate.py` | Independent fidelity cross-check of every saved pulse via `qutip.sesolve`, run in parallel with `grape_core`'s own propagator |
| `qutip_grape_optimizer.py` | Teaching implementation of the same GRAPE algorithm built directly on QuTiP `Qobj` physics (not just a cross-check) — same adjoint-gradient/L-BFGS-B structure as `optimizer.py`, deliberately simpler and without band-limiting, the discrepancy penalty, or `joblib` parallelism |
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
pip install qutip                      # used by qutip_validate.py
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

**Cross-check every saved pulse against an independent QuTiP propagator**:

```bash
python qutip_validate.py
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

- Heeres, Reinhold, Ofek, *et al.* — Implementing a universal gate set on a logical qubit encoded in an oscillator ([Nature Communications 8, 94 (2017)](https://www.nature.com/articles/s41467-017-00045-1)) — **primary reference for this project**
- Khaneja, Reiss, Kehlet, *et al.* — GRAPE: optimal control of spin systems ([JMR 2005](https://doi.org/10.1016/j.jmr.2004.11.004))
- Blais, Huang, Wallraff, Girvin — Circuit QED ([arxiv:cond-mat/0402216](https://arxiv.org/abs/cond-mat/0402216))
