# Error-Semitransparent (EsT) Gates

This module implements **error-semitransparent** logical gates on a binomial *kitten* bosonic code, following Roy, Wetherbee & Fatemi, "Error Semitransparent Universal Control of a Bosonic Logical Qubit," [arXiv:2603.15356](https://arxiv.org/abs/2603.15356).

An ordinary GRAPE gate constrains only where the code states land at $t = T$. If a photon is lost partway through, the pulse spends its remaining duration acting arbitrarily on the resulting state, and the error is generally no longer correctable. An **error-transparent** gate additionally requires that the same gate act identically on the error subspace, so that a loss event at *any* time remains correctable afterwards without knowing when it occurred.

`EST/` shares only `core.grape_core.make_ops` / `basis_state` (pure, constant-free) and `core.fourier_cutoff.make_band_mask` with the main pipeline. The device, the code, the gradient method, and the constraint handling all differ, so the module builds its own Hamiltonian rather than reusing `core.grape_core`'s module globals. **Nothing under `core/`, `validation/`, or `analysis/` is modified**; the root [`README.md`](../README.md) remains the reference for the cat-code pipeline, the adjoint gradient method, and the band-limit projection theory that this module reuses conceptually.

## Physics

### System model

The device is a dispersively coupled transmon–cavity system in a rotating frame, as in the root pipeline, but with a different chip and one additional term:

$$
H_0 = \chi\, n_A n_B + \frac{K_a}{2}\, A^{\dagger 2} A^2 + \frac{K'}{6}\, A^{\dagger 3} A^3 + \frac{\chi'}{2}\, n_B\, A^{\dagger 2} A^2 + \frac{K_q}{2}\, B^{\dagger 2} B^2
$$

| Term | Physical origin |
|------|-----------------|
| $\chi\, n_A n_B$ | Dispersive transmon–cavity coupling |
| $K_a\, A^{\dagger 2}A^2$ | Cavity self-Kerr |
| $K'\, A^{\dagger 3}A^3$ | **Second-order cavity self-Kerr — absent from the root pipeline's $H_0$** |
| $\chi'\, n_B\, A^{\dagger 2}A^2$ | Second-order dispersive shift |
| $K_q\, B^{\dagger 2}B^2$ | Transmon anharmonicity |

The control Hamiltonian, joint index convention $|t,c\rangle = t\,n_c + c$, and $(N,4)$ pulse layout $[\varepsilon_{C,I}, \varepsilon_{C,Q}, \varepsilon_{T,I}, \varepsilon_{T,Q}]$ are identical to the root pipeline, so all downstream shape conventions carry over.

Two departures are worth stating explicitly. First, `make_hamiltonian_est` **raises** for $n_t < 3$ rather than silently proceeding: the root's `make_hamiltonian` drops the anharmonicity term below $n_t = 3$, which for this device would discard $K_q/2\pi = -180$ MHz. Second, the $K'$ operator form is taken as $\tfrac{K'}{6} A^{\dagger 3}A^3$, the next term in the normal-ordered expansion; Table I supplies a magnitude, not an operator. Both are flagged in `device.py`.

### Device parameters (Table I)

| Parameter | Value | |
|-----------|-------|--|
| $\chi/2\pi$ | $-3.66$ MHz | dispersive coupling |
| $K_a/2\pi$ | $-22$ kHz | cavity self-Kerr |
| $K'/2\pi$ | $590$ Hz | second-order self-Kerr |
| $\chi'/2\pi$ | $39$ kHz | second-order dispersive shift |
| $K_q/2\pi$ | $-180$ MHz | transmon anharmonicity |
| $T_1^C,\ T_2^C$ | 180 μs, 290 μs | cavity coherence |
| $T_1^q,\ T_{2,\mathrm{Ramsey}}^q,\ T_{2,\mathrm{echo}}^q$ | 70 μs, 30 μs, 84 μs | transmon coherence |

Coherence times are recorded for completeness; the simulations producing the diagnostics below are lossless and unitary.

### The binomial kitten code

$$
|0_L\rangle = \frac{|0\rangle + |4\rangle}{\sqrt 2}, \qquad |1_L\rangle = |2\rangle,
\qquad\qquad
|0_E\rangle = |3\rangle, \qquad |1_E\rangle = |1\rangle
$$

The error words are the single-photon-loss images of the code words. Both bases are exactly orthonormal at every truncation, since their Fock supports are disjoint.

This code has a property the construction depends on: $a|0_L\rangle = \sqrt2\,|3\rangle$ and $a|1_L\rangle = \sqrt2\,|1\rangle$ carry **equal weight**. Restricted to the code space, $a$ is therefore $\sqrt 2$ times an isometry onto the error space, and it maps Bloch-sphere cardinal points to the corresponding cardinal points without distorting superpositions. This is what makes "the error partner of $|{+}X_L\rangle$" unambiguous. It is *not* a general fact — for a code with unequal weights, $a|{+}X_L\rangle$ would not be the error-space $|{+}X\rangle$ — so `kitten_code.error_cardinals` carries a live assertion pinning the coincidence.

Gate targets are constructed by applying the ideal $2\times2$ logical unitary to each cardinal point's **coefficient vector** and re-embedding, reusing `core.propagator.PAULI_EIGENSTATE_COEFFS`. Hand-permuting states is where sign and phase conventions are silently lost.

### Error transparency

The transparency conditions require that, at every instant $t$, (i) the evolved code words still satisfy the Knill–Laflamme conditions for the error set $\{I, a\}$; (ii) the evolved error space coincides with the instantaneous error space $\mathrm{span}\{a|\psi_0(t)\rangle, a|\psi_1(t)\rangle\}$; and (iii) the Hamiltonian acts identically on code and error subspaces. The three diagnostics in [Validation](#validation) measure these one-to-one.

Exact transparency is unreachable with linear drives. The obstruction is that

$$
\Bigl[\tfrac{K_a}{2} A^{\dagger 2}A^2 + \tfrac{\chi'}{2} A^{\dagger 2}A^2\, n_B,\ A\Bigr] = -\bigl(K_a + \chi'\, n_B\bigr) A^{\dagger}A^2
$$

is uncorrectable. Hence *semi*-transparent: a floor in these metrics is the physics of the device, not a convergence failure.

## Optimization

### Why automatic differentiation

The root pipeline obtains gradients from a hand-derived adjoint (root README, "Analytic gradients"). That is fast and validated, but **every new running-cost term requires its own hand-derived adjoint**. The EsT construction adds two. This module therefore replaces propagation and gradient with a JAX autodiff path (`grape_jax.py`) and leaves the physics, the state construction, the band-limit definition, and the L-BFGS-B driver shared.

One choice inside that path is load-bearing. `grape_core.step_data` diagonalizes $H_k$ and hand-derives an eigenbasis limit for near-degenerate eigenvalues, precisely because $\partial/\partial H$ of an eigendecomposition is singular at degeneracies. Autodiff through `jnp.linalg.eigh` hits **the same singularity**, so delegating the problem to autodiff via `eigh` would silently reintroduce the defect the root pipeline already fixed by hand. `grape_jax` uses `jax.scipy.linalg.expm` instead, which computes the identical propagator $e^{-i\,dt\,H_k}$ by scaling-and-squaring Padé without ever diagonalizing $H_k$.

Two implementation notes: the scan body is wrapped in `jax.checkpoint` (measured at $n_c=24$: 6.0 s/gradient at 0.32 GB, versus 4.78 s at 2.55 GB without), and the code and error cardinals are propagated as a single stacked $(n, 12)$ batch rather than two $(n,6)$ batches — exact, since propagation is column-wise, and it halves the `expm` work per gradient.

### Cost functions

$C_{\mathrm{tot}} = \sum_i w_i C_i$, with weights $(w_1, w_2, w_3, w_4)$.

| | Term | Type | Role |
|---|------|------|------|
| $C_1$ | `fidelity_cost` | terminal | ordinary gate infidelity |
| $C_2$ | `et_cost` | running | error-transparency infidelity |
| $C_3$ | `velocity_variance_cost` | running | velocity-variance regularizer |
| $C_4$ | `amplitude_cost` | running | circular drive-amplitude cap |

**$C_1$** is the ordinary infidelity, incoherently averaged over the six logical cardinal points $\{\pm Z, \pm X, \pm Y\}$:

$$
C_1 = 1 - \frac{1}{M}\sum_{j=1}^{6} \bigl|\langle \mathrm{target}_j | U(T) | \psi_j\rangle\bigr|^2 .
$$

The incoherent average is correct here, even though the root pipeline required `coherent_fidelity_multi_state` for its logical gates. That requirement arose from training on only two basis states, where the relative phase between branches is unconstrained. With all six cardinal points present — a spherical 2-design — linearity of $U$ pins every relative phase, leaving only the physically irrelevant global phase free.

**$C_2$** compares, at every time step, the normalized photon-loss image of the code state against the independently evolved error state:

$$
C_2 = 1 - \Bigl\langle \frac{\bigl|\langle \psi_{E_j}(t) | a | \psi_C(t)\rangle\bigr|^2}{\langle \psi_C(t)| a^\dagger a |\psi_C(t)\rangle} \Bigr\rangle_{t,\,j}
$$

**$C_3$** penalizes non-uniform speed through Hilbert space along the code trajectory, using the Fubini–Study distance between consecutive states, $v_i = \tfrac{2}{dt}\arccos|\langle\psi(t_i)|\psi(t_{i+1})\rangle|$:

$$
C_3 = \Bigl\langle \frac{\mathrm{Var}_t(v)}{\bigl(\mathrm{mean}_t\, v\bigr)^2} \Bigr\rangle_j
$$

This term exists only to stop the optimizer parking the dynamics in order to satisfy $C_2$ trivially — a state that does not move is transparent by default. Note that it penalizes *uneven* motion, not slow motion: a trajectory that sprints to the target and then sits still has high velocity variance, while a uniformly paced one does not. It does not itself improve transparency, which is why the second stage drops it to zero.

**$C_4$** is described under [Bounds versus penalties](#bounds-versus-penalties).

### Deviations from the published equations

Two cost terms depart from the paper as printed. Both are deliberate, documented at the point of definition, and **unverified against the authors' released code** — check them before comparing absolute numbers.

**1. $C_2$ normalization.** Eq. (C2) as printed divides the squared overlap by a single power of $N_{\mathrm{norm}} = \sqrt{\langle\psi_C|a^\dagger a|\psi_C\rangle}$, which does not bound $F_{\mathrm{ET}}$ in $[0,1]$. Dividing by $N_{\mathrm{norm}}^2$ does, via Cauchy–Schwarz:
$|\langle\psi_E|a|\psi_C\rangle|^2 \le \langle\psi_C|a^\dagger a|\psi_C\rangle\,\langle\psi_E|\psi_E\rangle$ with $\psi_E$ of unit norm. The implementation uses $N_{\mathrm{norm}}^2$, so $F_{\mathrm{ET}}$ is a genuine fidelity.

**2. $C_3$ normalization.** Eqs. (C4)–(C7) define $C_3$ as the raw variance of $v$; the implementation divides by $\mathrm{mean}(v)^2$. The rest of this subsection is the full argument, since this is the deviation most likely to be mistaken for a transcription error.

*Why $C_3$ is not in $[0,1]$ even though C6–C7 look normalized.* C6 and C7 are **sample averages, not normalizations**. C6 takes a variance over the $N$ time steps and C7 a mean over the six cardinal points $j$; both divide by a *count*, which removes extensivity — $C_3$ does not grow when you add time steps or cardinals — but neither divides by any *scale*. An average of a dimensionful quantity is still dimensionful. $v$ carries $\mathrm{rad}/\mu s$, so $\mathrm{Var}_t(v)$ carries $(\mathrm{rad}/\mu s)^2$ and $C_3$ inherits it, with no upper bound of any kind. The $[0,1]$ boundedness of $C_1$ and $C_2$ comes from somewhere else entirely: both are built from squared overlaps of *unit-norm* states, capped at 1 by Cauchy–Schwarz. There is no analogous inequality available for a velocity variance, because there is nothing in Eqs. (C4)–(C7) that a velocity is measured *against*.

*The $2/dt$ prefactor does not diverge.* This is the natural worry — $v_i = \tfrac{2}{dt}\arccos|\langle\psi_i|\psi_{i+1}\rangle|$ has $dt$ in a denominator — but the numerator cancels it exactly. For a smooth (here: band-limited) trajectory,

$$
|\langle\psi(t)|\psi(t+dt)\rangle| = 1 - \tfrac12(\Delta E\, dt)^2 + O(dt^4)
\;\Longrightarrow\;
2\arccos|\cdot| = 2\,\Delta E\, dt + O(dt^3),
$$

so $v_i \to 2\Delta E(t)$, the Anandan–Aharonov speed, with $\Delta E = \sqrt{\langle H^2\rangle - \langle H\rangle^2}$ the instantaneous energy uncertainty. The factor 2 in Eq. (C4) is precisely the convention that makes this identity hold. $v$ is therefore a well-defined continuum quantity and Eq. (C5) is a convergent finite-difference estimate of it, not a $dt$-amplified one — the Fubini–Study *distance* $d_{\mathrm{FS}}$ shrinks $\propto dt$ at exactly the rate the prefactor grows.

Measured directly, by sub-dividing the trained `u_X_est.npy` into $r$ identical sub-steps (which leaves the *continuous* evolution untouched and only samples the trajectory more finely) at $n_c = 20$:

| $r$ | $dt$ (ns) | $\mathrm{mean}(v)$ | $\mathrm{Var}(v)$ raw | $\mathrm{Var}/\mathrm{mean}^2$ | FS path length |
|---|---|---|---|---|---|
| 1 | 1.0 | 92.4116 | 964.70 | 0.11263 | 92.4116 |
| 2 | 0.5 | 92.4380 | 965.58 | 0.11267 | 92.4380 |
| 4 | 0.25 | 92.4447 | 965.81 | 0.11268 | 92.4447 |
| 8 | 0.125 | 92.4464 | 965.86 | 0.11268 | 92.4464 |
| 16 | 0.0625 | 92.4468 | 965.87 | 0.11268 | 92.4468 |

The raw variance moves by 0.12% over a 16× refinement, converging from below as the $O(dt^2)$ term dies — it is not $dt$-sensitive at all. Pointwise, $v_i$ agrees with $2\Delta E(t_i)$ computed from the instantaneous $H_k$ to a max relative deviation of $2.0\times10^{-3}$ at $dt = 1$ ns. **So the answer to "is the problem the $2/dt$?" is no** — the printed $C_3$ is a perfectly well-posed continuum functional, $\langle\mathrm{Var}_t\, 2\Delta E_j(t)\rangle_j$. Any $dt$-invariance claimed for the normalized form is a property both forms already have.

*The problem is that $v$ is physically large on this device.* $2\Delta E$ is set by the Hamiltonian, and on this chip the drives dominate it. On the trained X pulses at $dt=1$ ns, $n_c=20$:

| | $\mathrm{mean}(v)$ (rad/μs) | $\mathrm{std}(v)$ | $\mathrm{Var}(v)$ raw | $\mathrm{Var}/\mathrm{mean}^2$ |
|---|---|---|---|---|
| `u_X_est` | 92.41 | 31.06 | 964.7 | 0.1126 |
| `u_X_ord` | 85.29 | 29.65 | 878.9 | 0.1208 |

$\mathrm{mean}(v) = 92.4\ \mathrm{rad}/\mu s$ is $2\pi \times 14.7$ MHz — the state sweeps ~92 rad of Fubini–Study path in the 1 μs gate. For contrast, $2\Delta E$ of the code cardinals under $H_0$ *alone* is $0.56\ \mathrm{rad}/\mu s$: **99.4% of the trajectory speed is drive-induced**, which is what one expects when $\varepsilon_{\max} = 25.13\ \mathrm{rad}/\mu s$ enters $H$ multiplied by $\sqrt{n}$ matrix elements. So $v \sim 10^2$ and $\mathrm{Var}(v) \sim 10^3$, and with the paper's $w_3 = 5\text{–}10$ the raw $C_3$ contribution to $C_{\mathrm{tot}}$ is $\approx 5\times10^3$ to $10^4$, against $C_1, C_2 \in [0,1]$. **The fidelity terms would be numerically invisible and the optimizer would minimize the regularizer alone.** (An earlier revision of this section quoted $\mathrm{Var} \approx 100$ and a normalized 0.745; those were measured on an untrained pulse at a different $N$ and $n_c$ and do not reproduce. The conclusion is unchanged and the margin is an order of magnitude wider than stated.)

*A dimensionful term cannot carry a portable weight.* Because $\mathrm{Var}(v)$ has units of (time)$^{-2}$, its numerical value — and hence the meaning of $w_3$ — depends on the time unit. In rad/μs the raw variance is 965; write the identical physics in rad/ns and it is $9.7\times10^{-4}$, at which point $w_3 = 5\text{–}10$ is negligible rather than dominant. The unit that would make raw $C_3$ genuinely $O(1)$ is $\tau = 1/\mathrm{mean}(v) \approx 10.8$ ns, which is not a natural constant of anything — it is a property of this pulse. Supplying that scale from the trajectory itself is exactly what dividing by $\mathrm{mean}(v)^2$ does.

*Scale invariance is the real reason, not just the magnitude.* Under a uniform slowdown $v \to \lambda v$, raw $\mathrm{Var} \to \lambda^2 \mathrm{Var}$ while $\mathrm{Var}/\mathrm{mean}^2$ is invariant. So the raw form penalizes *slow* motion as well as *uneven* motion, and its steepest descent direction is "turn the drives down." Scaling the trained pulse by $\lambda$:

| $\lambda$ | 1.0 | 0.5 | 0.25 | 0.1 | 0.01 | 0.0 |
|---|---|---|---|---|---|---|
| $\mathrm{Var}(v)$ raw | 964.70 | 359.27 | 153.06 | 28.53 | 0.237 | **0.000** |
| $\mathrm{Var}/\mathrm{mean}^2$ | 0.1126 | 0.1213 | 0.1761 | 0.2000 | 0.1399 | 0.000 |

Raw $C_3$ falls by 4000× for a 100× amplitude reduction and reaches its **global minimum, exactly zero, at $\lambda = 0$** — the drift alone moves the code cardinals at a constant $0.57\ \mathrm{rad}/\mu s$, so the variance vanishes identically. The literal Eq. (C6)–(C7) term is therefore minimized by the parked trajectory, which is the precise failure mode $C_3$ was introduced to prevent, and with $w_3 \gg w_1$ that minimum is what the optimizer would find. The normalized form is flat in $\lambda$ over three decades (0.11–0.20), so it shapes the velocity *profile* and says nothing about its magnitude — which is the behaviour `velocity_variance_cost`'s docstring describes ("penalizes UNEVEN motion, not slow motion") and which is only literally true of the normalized form. Note the $\lambda = 0$ column is degenerate for both: with $v \equiv \mathrm{const}$ the numerator vanishes, so the normalized form is 0 there too via the $10^{-12}$ guard rather than being large. Its protection is against *near*-parking with any residual jitter, not against the measure-zero exactly-uniform case.

*Summary.* Dividing by $\mathrm{mean}(v)^2$ — the squared coefficient of variation — makes the term dimensionless, unit-independent, scale-invariant, and $O(0.1\text{–}1)$, i.e. commensurate with $C_1$ and $C_2$. This is the only reading under which the paper's stated weights $(1,\ 0.6\text{–}0.75,\ 5\text{–}10)$ form a coherent schedule. The literal form remains available via `normalize=False`, and three tests in `VelocityNormalizationTest` pin the argument in both directions. One caveat on those tests: `test_normalized_form_is_dt_stable` regenerates band-limited white noise at each $N = T/dt$, so the two sides are different *physical* pulses and the test does not isolate discretization; the refinement table above is what actually establishes $dt$-convergence, and it does so for both forms.

### The constraint chain

The band-limit, Gaussian ramp, and per-gate control mask are applied to the raw optimization variable **inside** the cost function, so autodiff differentiates the entire chain and there is no manual chain rule to derive:

$$
x \;\xrightarrow{\ P\ } \;\xrightarrow{\ \odot\, \mathrm{env}\ } \;\xrightarrow{\ \odot\, \mathrm{mask}\ }\; u_{\mathrm{phys}}
$$

This differs from the root pipeline, which applies `project_bandlimit` as a pre/post sandwich around the objective and relies on $P$ being idempotent and self-adjoint. The projection itself is the same operator, $P = \mathrm{IFFT}\circ\mathrm{mask}\circ\mathrm{FFT}$ on the complex drives, and the mask is built by the root's own `core.fourier_cutoff.make_band_mask` so the training constraint and the post-hoc `out_of_band_energy_fraction` check cannot drift apart. This module uses a single band, $\pm 50$ MHz, on both drives.

The ordering was chosen by measurement, not preference. On white-noise input at $N=1000$, $dt = 1$ ns, 48 ns ramp:

| Order | Out-of-band energy | Endpoint amplitude (relative to mid-pulse) |
|-------|-------------------|--------------------------------------------|
| project → ramp | 0.34% | 0.4% |
| ramp → project | 0% | 23% |

Projecting last gives exact band-limiting but smears the envelope so severely that the endpoints reach a quarter of full scale, defeating the purpose of the ramp. Ramping last is used; the residual sub-percent band violation is measured by the test suite rather than assumed.

### Bounds versus penalties

This is the most easily misread part of the setup. L-BFGS-B supports only **box bounds** — one independent $(\mathrm{lo}, \mathrm{hi})$ interval per variable, an axis-aligned feasible region. Two separate facts make box bounds unable to express this device's amplitude constraint.

**They act on the wrong variable.** Because the constraint chain lives inside the cost, the bounds constrain the raw pre-image $x$, not the physical pulse $u_{\mathrm{phys}}$ — the same caveat documented for the root pipeline at `core/optimizer.py:73-76`.

**They have the wrong shape.** With $H_c = [A + A^\dagger,\ i(A - A^\dagger),\ \ldots]$, the physical drive is $\varepsilon = u_0 - i u_1$, so Table I's $\varepsilon_{\max}/2\pi = 4$ MHz caps the *quadrature-pair norm* $\sqrt{u_0^2 + u_1^2} \le 25.13$ rad/μs. A per-element box at $\varepsilon_{\max}$ admits $|\varepsilon|$ up to $\sqrt2\,\varepsilon_{\max} \approx 35.5$.

The consequence is measurable: at `hard_bound = 60`, a raw variable pinned at the box corner yields $|\varepsilon| = 84.9$ rad/μs, **3.4× the physical cap**. The box therefore never binds *on a cold start* — see below for the warm-start case, where it does. What actually caps the waveform is $C_4$, a one-sided quadratic penalty on the pair norm:

$$
C_4 = \Bigl\langle \max\bigl(|\varepsilon_C| - \varepsilon_{\max},\, 0\bigr)^2 + \max\bigl(|\varepsilon_T| - \varepsilon_{\max},\, 0\bigr)^2 \Bigr\rangle_t
$$

Both trained pulses sit at 24.6–25.2 rad/μs, at the cap rather than at the box. Because $C_4$ is soft, it settles marginally *over* the cap where the penalty gradient balances the fidelity gradient — 0.2% over on 0.4–1.1% of steps for the EsT pulse — which is why `train_est.check_constraints` accepts a 1% tolerance. A hard guarantee would require either a larger $w_4$ or an explicit projection $\varepsilon \to \varepsilon\,\min(1, \varepsilon_{\max}/|\varepsilon|)$ as a fourth step in the chain.

The Gaussian ramp similarly replaces a penalty with structure: because the envelope vanishes at both ends by construction, the root pipeline's `boundary_penalty` is redundant here.

#### The box does bind on a warm start

Warm-starting from a saved pulse inverts the constraint chain, and that turns the harmless box into a live constraint. `train_est.deramp` recovers a pre-image by dividing out the envelope, which is exact — a saved pulse is $\mathrm{env} \odot P(x)$, and $u/\mathrm{env} = P(x)$ is already band-limited, so re-projecting is idempotent (verified to $1.8\times10^{-14}$). But the envelope reaches $0.0066$ on the first and last steps, so the division amplifies by up to **151×** in the ramp windows. For `u_X_est` the recovered pre-image peaks at $81.0$, with 0.6% of its components above the default bound of 60.

The pre-image coordinate system is thus *stretched by up to 151× in exactly the region where the box ends up binding*, and the largest excursions sit where the physical pulse is smallest: RMS $|x| = 28.6$ in the ramp windows against $12.6$ on the flat top, while RMS $|u|$ is $9.3$ against $12.6$. (The mechanism is this coordinate stretch, not gradient flatness — measured $\mathrm{RMS}|dC/dx|$ differs by only 1.3× between the two regions, because $P$ sits behind the ramp in the backward pass and its global FFT delocalizes the suppression.)

Two consequences:

* **scipy will not warn you.** L-BFGS-B silently projects an infeasible starting point into the box, so a warm start at the default bound would begin from an edge-truncated pulse while the log still reports the original filename. `train()` therefore raises rather than clipping, and records `max_abs_preimage` per stage so a run can be audited afterwards.
* **`max|deramp(u)| > hard_bound` is not evidence of a bound violation.** $P$ is norm-1 in $L^2$ but not in sup-norm, so it can raise a signal's peak while strictly lowering its energy — a saturated $\pm 60$ vector projects to a peak of $66.7$. Since L-BFGS-B is a feasible-point method, its output always satisfies the box; only `max_abs_preimage`, recorded from $x$ itself, can establish binding.

#### The fix: save the raw variable (`--init-x`)

The cleanest fix is to stop discarding the raw variable, and `save()` now writes it. Every run emits an `x_<gate>_<variant>.npy` beside the pulse, plus one per stage, and `--init-x` warm-starts from it directly.

This removes the problem rather than working around it. $x$ is the vector L-BFGS-B stopped on, so it is feasible in the box it ran under *by construction* — there is nothing to invert, no envelope division, no 151× amplification, and no roundtrip assertion to satisfy. The measured cost of not having it grows with each warm generation:

| pulse | $\max|\mathrm{deramp}(u)|$ | components outside the default box |
|---|---|---|
| `u_X_est` (cold) | 81.0 | 24 / 4000 |
| `u_X_est_warm` (one warm restart) | 158.5 | 66 / 4000 |

A third generation would need `--hard-bound 300`. With `--init-x` all of these start at whatever bound they were trained under.

`--init` is retained and unchanged, because pulses trained before this change have no saved pre-image and `EST/compare_warmstart.py` uses `deramp` directly. Note that `--init` is not *inaccurate* — the inversion is exact to $10^{-14}$ whenever it stays inside the box, and `train()` refuses to start rather than clipping when it does not. What `--init-x` buys is that the question never arises. The two flags are mutually exclusive.

One caveat when reading `max_abs_preimage` back out of a log: **a cold start pins the box on its own.** `amp0 * randn(4N)` at `amp0 = 20` has a maximum of roughly $3.5\sigma = 70$ over 4000 draws, so a handful of components sit outside `hard_bound = 60` at iteration 0 and L-BFGS-B clips them there. A cold run therefore reports `max_abs_preimage = 60.0` exactly, and that is the initialization, not the optimizer pressing against a constraint the way the warm X run did. The quantity that separates the two cases is the *fraction* of components at the bound, not the maximum.

### Weight schedule and truncation

Training runs in two stages, warm-starting the second from the first:

| Stage | $(w_1, w_2, w_3, w_4)$ | Purpose |
|-------|------------------------|---------|
| 1 | $(1,\ 0.7,\ 7,\ 1)$ | establish transparency |
| 2 | $(1,\ 0.1,\ 0,\ 1)$ | polish ordinary fidelity |

The **Ord** control variant is the *same code path* with $w_2 = w_3 = 0$. Since the entire result is a comparison against it, it is important that it differ from EsT by weights alone rather than by a separate implementation.

The whole two-stage schedule can also be re-entered from a finished pulse with `--init`, which is how the *EsT (warm)* result below was produced. Note what that means for stage 1: it re-imposes $w_3 = 7$ on a pulse that stage 2 had already released from it, so a warm restart does not resume where the previous run stopped — it re-traverses the fidelity-for-transparency trade from a better starting point.

Training uses a **single** truncation $n_c = 20$, unlike the root pipeline's multi-truncation default. The measured cost is 2.65 s per gradient at one truncation versus 9.9 s at three — a 3.7× tax on every optimizer step. The root pipeline needs multi-truncation training because $\alpha=\sqrt3$ cat states carry population out to $n \sim 15$, where a pulse can exploit the Hilbert-space wall; the kitten code's highest code word is $|4\rangle$ and the drive is capped at 4 MHz, so the wall is far from the dynamics. Convergence is therefore **verified** on held-out truncations rather than trained in, with retraining at `--trunc 16 20 24` prescribed if the check fails. It has not: the measured spread over $n_c = 16\ldots28$ is $2\times10^{-7}$ (EsT) and $2\times10^{-8}$ (Ord).

## Validation

### Test suite

`python EST/test_grape_jax.py -v` — 24 tests, plain `unittest`, matching the repository convention.

| Class | Tests | What it pins |
|-------|-------|--------------|
| `AnalyticGradientCrossCheck` | 2 | JAX gradient against the hand-derived adjoint; `expm` trajectory against `eigh` |
| `FiniteDifferenceTest` | 5 | full $C_{\mathrm{tot}}$, the chain-constrained cost, and $C_2$/$C_3$/$C_4$ individually |
| `ConstraintSatisfactionTest` | 5 | band-limit exactness, out-of-band leakage, endpoint ramp-down, T-gate masking, envelope shape |
| `VelocityNormalizationTest` | 3 | that the raw $C_3$ would swamp the fidelity terms, and the normalized form is $O(1)$ and $dt$-stable |
| `DiagnosticsTest` | 3 | Eqs. 6–8 against two cases with known answers |
| `KittenCodeTest` | 4 | code words, error words as photon-loss images, gate targets |
| `PreimagePersistenceTest` | 2 | that a saved $(u, x)$ pair still satisfies $u = \mathrm{constrain}(x)$, and that `deramp` inverts the chain only modulo the band-limit projection |

The anchor is `test_c1_gradient_matches_grape_core`: with $C_1$ alone and the constraint chain disabled, `jax.grad` is compared against `core.grape_core.fidelity_multi_state`'s hand-derived adjoint gradient at `rtol=1e-6`, with the expected sign flip (JAX returns the cost $1-F$, numpy returns the fidelity $F$). This ties the new gradient path to one already validated against QuTiP. The finite-difference tests then cover $C_2$, $C_3$ and $C_4$, which have no analytic counterpart to compare against.

### Independent cross-checks

`EST/diagnostics.py` propagates with `core.grape_core.step_data`'s **eigendecomposition** propagator, deliberately *not* the JAX `expm` path used for training. Re-scoring a trained pulse through a different algorithm is an independent check, in the same spirit as the root pipeline's QuTiP cross-check: a pulse that scores well under its own propagator and poorly under another has a numerics problem rather than a physics result. Measured agreement on $C_1$ for both trained pulses is $5\times10^{-14}$.

`diagnostics.propagate_states` also fills a genuine gap — all four `simulate_trajectory` variants elsewhere in the repository reduce to populations inside the loop and discard the state amplitudes, which the transparency metrics require.

### Diagnostics (Eqs. 6–8)

| Metric | Condition tested | Range |
|--------|------------------|-------|
| $\Delta_{\mathrm{QEC}}(t)$ | instantaneous Knill–Laflamme validity of the evolved code | $\ge 0$, normalization chosen |
| $L_{E_j}(t)$ | evolved error state remains inside the instantaneous error space | $[0,1]$ |
| $\eta_{E_j,\psi}(t)$ | evolved error state matches the photon-loss image of the code state | $[0,1]$ |

$\Delta_{\mathrm{QEC}}$ is computed from the two evolved code words alone — the error trajectory does not enter — as the RMS of the four independent KL residuals for $\{I, a\}$, normalized by mean photon number. $L$ measures the component of $|\psi_{E_j}(t)\rangle$ outside $\mathrm{span}\{a|\psi_0(t)\rangle, a|\psi_1(t)\rangle\}$; $\eta$ is strictly stronger, since one can be inside that subspace yet at the wrong point in it, so $L \le \eta$ is expected and observed.

**These three are reconstructed from the transparency conditions, not transcribed from the paper.** $L$ and $\eta$ are bounded in $[0,1]$ by construction and have no free normalization; $\Delta_{\mathrm{QEC}}$'s does, and is this module's choice. The EsT-versus-Ord *ratio* is robust to that choice; absolute values are not. Note also that $\eta$ is exactly the time-resolved integrand of $C_2$, so $\mathrm{mean}(\eta) \equiv 1 - F_{\mathrm{ET}}$ — verified across the two independent code paths to $3\times10^{-13}$.

One result corroborates the reconstruction. Under free Kerr evolution with no drive, $\Delta_{\mathrm{QEC}}$ and $L$ sit at machine zero — Kerr is diagonal in the Fock basis and preserves the code's photon-number structure — while $\eta = 7\times10^{-4}$. That is the App. A obstruction appearing in exactly the channel $C_2$ targets and nowhere else. It is now a regression test.

## Results

Two gates are trained, X and H. Both use $n_t = 3$, $n_c = 20$, $dt = 1$ ns, $N = 1000$, seed 0, and all other defaults; they differ only in the target logical unitary and in iteration count. Metrics are time-averaged over the six cardinal points; lower is better except for the two fidelities.

Headline comparison — the EsT : Ord separation each run achieves, against the same-gate control:

| run | it/stage | $F_1$ | $F_{\mathrm{ET}}$ | $L_{E_j}$ | Ord : EsT on $L$ | Ord : EsT on $\eta$ |
|---|---|---|---|---|---|---|
| X, cold | 600 | 0.99920 | 0.679 | 0.208 | 2.18× | 2.53× |
| X, warm | 1000 | 0.99904 | 0.736 | 0.149 | 3.05× | 3.07× |
| **H, cold** | **2000** | **0.99968** | **0.818** | **0.113** | **4.65×** | **4.58×** |

Every metric improves monotonically with iteration count, and the H run is the first to reach the paper's $F_{\mathrm{ET}} \approx 0.83$ (0.827 at stage 1). Note the confound: the H row changes both the iteration count *and* the gate, so it is not a controlled demonstration that iterations alone are responsible — see limitation 1.

### X gate (600–1000 iterations per stage)

Two EsT runs are reported:

* **EsT** — cold start from random controls, 600 iterations per stage.
* **EsT (warm)** — restarted from the EsT pulse and rerun through *both* stages at 1000 iterations each (`--init pulses/est/u_X_est.npy`). 91 min total.

| | $F_1$ | $F_{\mathrm{ET}}$ | $\Delta_{\mathrm{QEC}}$ | $L_{E_j}$ | $\eta$ | max active Fock |
|---|---|---|---|---|---|---|
| **EsT** | 0.99920 | 0.679 | $4.32\times10^{-2}$ | $2.08\times10^{-1}$ | $3.21\times10^{-1}$ | 9 |
| **EsT (warm)** | 0.99904 | **0.736** | $4.65\times10^{-2}$ | $\mathbf{1.49\times10^{-1}}$ | $\mathbf{2.64\times10^{-1}}$ | 10 |
| **Ord** | 0.99999 | 0.188 | $6.92\times10^{-2}$ | $4.54\times10^{-1}$ | $8.12\times10^{-1}$ | 8 |
| ratio, Ord : EsT | — | 3.6× | 1.60× | 2.18× | 2.53× | — |
| ratio, Ord : EsT (warm) | — | **3.9×** | 1.49× | **3.05×** | **3.07×** | — |

The EsT pulse improves every transparency metric while giving up ordinary fidelity, $0.99920$ against $0.99999$. That trade is the expected behaviour rather than a defect: Ord optimizes $C_1$ alone and drives it as far as it can, while EsT spends part of that budget on transparency. Time-resolved curves are in `figures/est/fig1def_est_vs_ord.png`; summary metrics in `tables/est_fig1_metrics.csv`.

The warm restart improves the two error-space metrics substantially — $\eta$ by 18% and $L$ by 29%, lifting both EsT : Ord ratios from ~2.2–2.5× to ~3.05× — at essentially no fidelity cost ($0.99920 \to 0.99904$). $\Delta_{\mathrm{QEC}}$ is the exception: it degrades 8%, the only transparency metric to move the wrong way, and its ratio falls from 1.60× to 1.49×. Note also that the restart did **not** merely polish the starting pulse: $\lVert u_{\mathrm{warm}} - u_{\mathrm{EsT}}\rVert / \lVert u_{\mathrm{EsT}}\rVert = 1.21$, so this is a different waveform, not a refinement of the old one.

Reproduce with `python EST/compare_warmstart.py` → `tables/est_warmstart_comparison.csv`, `figures/est/warmstart_est_vs_stages.png`. (`diagnostics.py --gate X` scores only `u_X_est` and `u_X_ord` — it has no `--tag`, so it does not pick the warm pulses up.)

### What the second stage does

Because the warm run saved both stages, the stage-1 → stage-2 transition can be read off directly. The stage-1 row *is* the warm-start point re-scored, so the first row below is also the cold EsT result:

| | $F_1$ | $F_{\mathrm{ET}}$ | $C_3$ (vel. var.) | $\Delta_{\mathrm{QEC}}$ | $L_{E_j}$ | $\eta$ |
|---|---|---|---|---|---|---|
| start (= cold EsT) | 0.99920 | 0.679 | 0.1126 | $4.32\times10^{-2}$ | $2.08\times10^{-1}$ | 0.3213 |
| after stage 1 | 0.96566 | **0.747** | **0.0072** | $3.99\times10^{-2}$ | $1.54\times10^{-1}$ | 0.2533 |
| after stage 2 | **0.99904** | 0.736 | 0.1203 | $4.65\times10^{-2}$ | $1.49\times10^{-1}$ | 0.2644 |

Stage 1 buys transparency with fidelity exactly as intended: infidelity worsens 43×, while $F_{\mathrm{ET}}$ climbs to 0.747 and velocity variance drops 16×. Stage 2 then recovers essentially all of the fidelity — infidelity $3.43\times10^{-2} \to 9.63\times10^{-4}$, a 36× improvement — and gives back only a small part of the transparency gain: $F_{\mathrm{ET}}$ falls 1.5% relative, $\eta$ rises 4.4%, $\Delta_{\mathrm{QEC}}$ rises 17%, while $L$ continues to *improve*.

One caveat on the schedule as documented: `train_est.py`'s docstring says stage 2 moves the ET **and velocity** metrics by ~1%. That holds for $F_{\mathrm{ET}}$ but not for $C_3$, which regresses 17× ($0.0072 \to 0.1203$, back to Ord's 0.1208) because $w_3 = 0$ in stage 2 and nothing then holds uniform speed. Velocity uniformity is not a property of the delivered pulse; it is scaffolding that stage 1 uses to stop the optimizer parking the dynamics to cheat $C_2$.

### H gate (2000 iterations per stage)

Both variants cold-started at `--maxiter 2000`, otherwise identical to the X runs:

```bash
python EST/train_est.py --gate H --variant est --maxiter 2000   # 5.1 h
python EST/train_est.py --gate H --variant ord --maxiter 2000   # 10.1 h (loaded machine)
```

| | $F_1$ | $F_{\mathrm{ET}}$ | $\Delta_{\mathrm{QEC}}$ | $L_{E_j}$ | $\eta$ | max active Fock |
|---|---|---|---|---|---|---|
| **EsT** | 0.99968 | **0.818** | $4.08\times10^{-2}$ | $1.13\times10^{-1}$ | $1.82\times10^{-1}$ | 10 |
| **Ord** | 1.00000 | 0.165 | $5.31\times10^{-2}$ | $5.28\times10^{-1}$ | $8.35\times10^{-1}$ | 8 |
| ratio, Ord : EsT | — | **5.0×** | 1.30× | **4.65×** | **4.58×** | — |

This is the strongest separation the replication has produced, and it beats the best X result on every error-space metric at *higher* ordinary fidelity: $L$ is $0.113$ against the warm X pulse's $0.149$, $\eta$ is $0.182$ against $0.264$, and $F_1$ is $0.99968$ against $0.99904$. The Ord baselines are comparable across the two gates ($L = 0.528$ here versus $0.454$ for X), so the improvement lives in the EsT pulse rather than in a shifted control.

$\Delta_{\mathrm{QEC}}$ is again the laggard at 1.30×, consistent with X's 1.49–1.60× and with limitation 4 below.

The stage decomposition:

| | $F_1$ | $F_{\mathrm{ET}}$ | $C_3$ (vel. var.) |
|---|---|---|---|
| after stage 1 | 0.96112 | **0.827** | **0.0063** |
| after stage 2 | **0.99968** | 0.818 | 0.1259 |

Stage 1 reaches $F_{\mathrm{ET}} = 0.827$ — the paper's $\approx 0.83$ — and stage 2 then buys a 120× infidelity improvement ($3.89\times10^{-2} \to 3.2\times10^{-4}$) for 1.1% relative of it. The same $C_3$ scaffolding behaviour as X: velocity variance drops 21× in stage 1 and returns to the Ord level in stage 2.

Two checks on the pulses. The independent `eigh` re-score reproduces the JAX training value to six digits ($0.999675$ against $1 - c_1 = 0.99968$), and the truncation scan is flat over $n_c = 16\ldots28$ at a spread of $6.9\times10^{-7}$ (EsT) and $9.7\times10^{-9}$ (Ord), so single-truncation training is again validated rather than assumed. Figures `figures/est/fig1def_est_vs_ord_H.png` (log axes, the paper's own view) and `figures/est/fig1def_est_vs_ord_H_linear.png` (linear axes, which show the gap at its true relative size for metrics that sit at $O(0.1)$); metrics `tables/est_fig1_metrics_H.csv`.

#### Fig. 1a: the gate as seen by each subspace

`python EST/subspace_evolution.py --gate H` draws the paper's Fig. 1a for this pulse pair — `figures/est/fig1a_subspace_H.png`/`.pdf`, `tables/est_fig1a_subspace_H.csv`. Three curves per Bloch cardinal, from one lossless unitary simulation:

| curve | meaning |
|---|---|
| $\langle n\rangle_C(t)$ | mean photon number of the evolved **code** cardinal |
| $\langle n\rangle_E(t)$ | mean photon number of the evolved **error** cardinal |
| $\langle n\rangle_{\rm loss}(t)$ | mean photon number of the photon-loss image $a\lvert\psi_C(t)\rangle/\lVert a\lvert\psi_C(t)\rangle\rVert$ |

The third is the prediction the second must match: error transparency says the evolved error state stays equal to the loss image of the evolved code state, so those two curves coincide for a transparent gate and peel apart for one that is not. Reported alongside is the map-level distance between the two *fixed*-basis blocks $U_L(t) = B^\dagger U(t) B$ and $U_E(t) = E^\dagger U(t) E$,

$$\mathcal{M}(t) = 1 - \frac{|\mathrm{Tr}(U_L^\dagger U_E)|^2}{\mathrm{Tr}(U_L^\dagger U_L)\,\mathrm{Tr}(U_E^\dagger U_E)} \in [0,1].$$

| | $\overline{|\Delta\langle n\rangle|}$ | $\overline{\mathcal{M}}$ | $\mathcal{M}(T)$ | $\langle n\rangle_E$ peak |
|---|---|---|---|---|
| **EsT** | **0.350** | **0.156** | **0.050** | 4.00 |
| **Ord** | 0.947 | 0.557 | 0.701 | 5.47 |
| ratio, Ord : EsT | **2.70×** | **3.57×** | 14× | — |

The $+Z_L$ panel is the clearest single view: the Ord error state's photon number runs away to $\approx 5$ and stays there while its loss-image target sits near 2, whereas the EsT error state tracks its target across the whole gate. Both variants' code curves start at exactly $\langle n\rangle = 2$, which is fixed by the code rather than by the pulse — both code words carry $\langle n\rangle = 2$ and $\langle 0_L|n|1_L\rangle = 0$, so no superposition of them can differ. The error cardinals start split 3 / 1 / 2 / 2 / 2 / 2 over $+Z$ / $-Z$ / $+X$ / $-X$ / $+Y$ / $-Y$, since $\lvert 0_E\rangle = \lvert3\rangle$ and $\lvert1_E\rangle = \lvert1\rangle$.

**What $\langle n\rangle$ cannot show.** $H_0$ is a function of the number operators alone, so $[H_0, n] = 0$ and $\langle n\rangle$ is *exactly* conserved under drift — every feature in these curves is drive-induced. It also means $\langle n\rangle$ is blind to App. A's obstruction: free Kerr evolution drives $\eta$ away from zero while leaving all three photon-number curves flat, because it rearranges phases within a fixed photon distribution. Two states can share $\langle n\rangle$ to machine precision and still be orthogonal. So $\langle n\rangle_E = \langle n\rangle_{\rm loss}$ is **necessary for transparency and not sufficient**; the sufficient statements are $\eta$ (Eq. 8) and $\mathcal{M}$, which is why the figure carries the mismatch panel beside the photon-number panels. This is pinned in `test_grape_jax.SubspaceEvolutionTest`.

$\mathcal{M}$ and $\overline{|\Delta\langle n\rangle|}$ are companions to the figure, not the paper's printed metrics: their normalizations are this module's choice, so EsT-vs-Ord ratios are meaningful and absolute values are not, exactly as for Eqs. 6–8.

The $r^2$ curves in the right-hand panel are the surviving weight inside each fixed span. Read $1 - r^2$ carefully: it is not all leakage in the usual sense, because $\lvert 0_L\rangle = (\lvert0\rangle + \lvert4\rangle)/\sqrt2$ has its two Fock components dephase against each other under Kerr and rotate toward $(\lvert0\rangle - \lvert4\rangle)/\sqrt2$, costing weight with nothing having left the $\{0,2,4\}$ sector. The error span $\{\lvert3\rangle, \lvert1\rangle\}$ has no internal superposition and so no such channel — under drift alone it is exactly invariant.

`max_abs_preimage` reads 60.0 on both stages, but this is the cold-start clipping described above, not a binding box: only 0.125% of components (5 of 4000) sit at the bound.

### Known limitations

1. **Iteration count was the binding constraint, and the H run resolved it.** Almost no stage-run has terminated on a convergence criterion; nearly all hit `maxiter`. The X warm restart tested running longer and suggested it helps without closing the gap — $F_{\mathrm{ET}}$ at stage 1 went $0.704$ (600 it) $\to 0.747$ (1000 it) against the paper's $\approx 0.83$ — which pointed suspicion at the $C_2$ normalization and the box artifact in item 3. The H run at 2000 iterations reaches $0.827$ at stage 1, so that reading was wrong: the schedule needed roughly 3× the iterations it was being given, and the normalization choices are not implicated.

   Two caveats. This is H, not X, so it is not a controlled comparison — a matching X rerun at `--maxiter 2000` would settle it and has not been done. And the H stage-1 run still stopped on `maxiter`, so 0.827 is a lower bound on what the schedule converges to, not the converged value.
2. **The comparison is not like-for-like.** Max active Fock level is 9 (X EsT), 10 (X EsT warm), 10 (H EsT) against 8 for both Ord pulses. The paper's fair-comparison criterion matches on this quantity rather than on gate duration, so every ratio above should be regarded as provisional — the EsT pulses are being scored across a wider Fock spread than the controls they are compared to. `diagnostics.py` reports the mismatch automatically.
3. **The X warm run trained against a binding box.** `max_abs_preimage` sat at exactly 120.0 — the `--hard-bound` value — at the end of both warm stages, so L-BFGS-B spent the whole run pressed against a constraint that has no physical meaning (see *Bounds versus penalties*). Side effects are visible but small: out-of-band energy rose from 0.50% to 0.70% on the cavity drive and the endpoint/mid amplitude ratio from 0.016 to 0.040, both indicating extra amplitude pushed into the 48 ns ramp windows. The amplitude cap is still satisfied (25.15 rad/μs, the same 0.2% $C_4$ overshoot as the cold pulse). This affects only that one pulse — the H runs are cold and show 0.125% of components at the bound — and `--init-x` prevents it recurring.
4. **The $\Delta_{\mathrm{QEC}}$ endpoint is an artifact.** At $t = T$, $\Delta_{\mathrm{QEC}}$ is $8\times10^{-5}$ for Ord against $5.3\times10^{-3}$ for EsT — purely because Ord's terminal fidelity is higher, and a gate landing exactly on the ideal code words has zero KL violation by construction. Mid-gate, where transparency matters, EsT leads 1.9×.
5. **Absolute transparency is improving but still short.** $L = 0.21$ (X cold), $0.15$ (X warm), $0.113$ (H) — so between a fifth and a ninth of the error state has left the instantaneous error space on average. The hierarchy is reproduced throughout and the longer H run roughly halves the X figure, but a residual floor is expected on principle: App. A rules out exact transparency with linear drives.
6. **Scope.** X and H are trained; **T is not**. T is a parameter change against the same pipeline, but not a free one — it is 0.6 μs rather than 1.0 ($N = 600$) and its `CONTROL_MASK` is transmon-only `[0,0,1,1]`, which is the one case where `deramp` cannot round-trip the masked-out cavity columns. The **LE** control variant is not implemented — it requires an error-space terminal-fidelity term, which is a new cost function rather than a reweighting — and neither is the AQEC recovery pulse needed for the paper's Figs. 4–5.

## Project layout

| File | Description |
|------|-------------|
| `EST/device.py` | Table I constants, `make_hamiltonian_est`, Gaussian ramp envelope, band mask |
| `EST/kitten_code.py` | Code and error bases, cardinal points, gate targets, per-gate control masks |
| `EST/grape_jax.py` | Constraint chain, `expm` propagation, cost terms $C_1$–$C_4$, scipy-ready objective |
| `EST/train_est.py` | Two-stage L-BFGS-B driver, warm restart (`--init-x` exact, `--init` via `deramp`), constraint verification, JSON metadata |
| `EST/diagnostics.py` | Eqs. 6–8, truncation scan, independent re-score, Fig. 1d–f figure; `--gate` selects which trained gate to score, `--scale` its y-axis |
| `EST/subspace_evolution.py` | Fig. 1a: mean photon number of the code and error subspaces against the photon-loss prediction, plus the map mismatch $\mathcal{M}(t)$ |
| `EST/compare_warmstart.py` | Cold vs. warm-restart comparison across both stages and both code paths |
| `EST/test_grape_jax.py` | 32-test correctness suite |
| `pulses/est/` | Trained pulses `u_<gate>_<variant>.npy` and their raw pre-images `x_<gate>_<variant>.npy`, both shape $(N,4)$ |
| `figures/est/`, `tables/`, `logs/` | Generated figure, summary CSV, per-run metadata |

Trained pulses are written to `pulses/est/` and deliberately **not** to `pulses/`. `validation/validate_logical_gates.py` and every script in `analysis/` locate pulses by the hard-coded name `u_<gate>_main.npy` and assume the $\alpha=\sqrt3$ four-component cat code; a kitten-code pulse placed in `pulses/` would be silently mis-scored rather than rejected.

## Quick start

```bash
pip install -r requirements.txt        # adds jax to the root dependencies
```

```bash
# Correctness suite -- run before trusting any training result (~80 s)
python EST/test_grape_jax.py -v

# Train the X gate. ~1 h per variant at N=1000, dt=1 ns.
python EST/train_est.py --gate X --variant est
python EST/train_est.py --gate X --variant ord

# Train the H gate, both variants, at 2000 iterations per stage (~3-4 h each)
python EST/train_est.py --gate H --variant est --maxiter 2000
python EST/train_est.py --gate H --variant ord --maxiter 2000

# Metrics, truncation scan, independent re-score, and the Fig. 1d-f figure.
# Writes both a log-axis and a linear-axis version; --scale picks just one.
python EST/diagnostics.py                # X (unsuffixed output names)
python EST/diagnostics.py --gate H       # -> ..._H.png / ..._H_linear.png / ..._H.csv

# Fig. 1a: <n>(t) for the code space vs the error space
python EST/subspace_evolution.py --gate H        # -> figures/est/fig1a_subspace_H.png/.pdf
python EST/subspace_evolution.py --gate H --cardinals +Z,-Z,+X,-X,+Y,-Y   # all six

# Warm restart. Prefer --init-x: it loads the raw pre-image the run stopped on,
# which is feasible at the bound it was trained under, so no --hard-bound bump.
python EST/train_est.py --gate X --variant est --maxiter 1000 \
       --init-x pulses/est/x_X_est.npy --tag warm

# --init is the older route, for pulses saved before x was persisted. It inverts
# the constraint chain, so --hard-bound must exceed max|deramp(u)| or train()
# refuses to start (see "The box does bind on a warm start").
python EST/train_est.py --gate X --variant est --maxiter 1000 \
       --init pulses/est/u_X_est.npy --hard-bound 300 --tag warm

# Cold vs. warm-stage-1 vs. warm-stage-2 vs. Ord, scored through both code paths
python EST/compare_warmstart.py
```

All commands are run from the repository root. Useful flags: `--maxiter` (default 600 per stage), `--trunc 16 20 24` for multi-truncation training, `--init-x`/`--init`/`--tag`/`--hard-bound` for warm restarts, `--gate`, `--dt`, `--seed`, `--no-save`.

Every run writes **both** the physical pulse and the raw optimizer pre-image, at the end and after each stage:

| | final | per stage |
|---|---|---|
| physical pulse | `pulses/est/u_<gate>_<variant>[_<tag>].npy` | `..._stage<i>.npy` |
| raw pre-image | `pulses/est/x_<gate>_<variant>[_<tag>].npy` | `..._stage<i>.npy` |
| metadata | `logs/est_<gate>_<variant>[_<tag>].json` | — |

Only `u` is physically meaningful — it is what `diagnostics.py` and every other consumer scores — but only `x` resumes a run exactly, which is why both are kept. Keeping the per-stage pair means the stage-1 → stage-2 delta stays measurable *and* either stage can be restarted from. `--tag` keeps a rerun from clobbering the pulse it started from.

## Default parameters

| Parameter | Value |
|-----------|-------|
| $dt$ | 0.001 μs (1 ns) |
| $N$ | 1000 (X, H); 600 (T) |
| Band limit | $\pm 50$ MHz, both drives |
| $\varepsilon_{\max}$ | $2\pi \times 4$ MHz $= 25.13$ rad/μs |
| Gaussian ramp | 48 ns rise and fall |
| $n_t$ | 3 (enforced) |
| $n_c$ | 20 (training); validated 16–28 |
| `hard_bound` | 60 rad/μs (box on the raw variable; not the physical cap) |
| Stage weights | $(1, 0.7, 7, 1)$ then $(1, 0.1, 0, 1)$ |

## References

- Roy, A., Wetherbee, C. & Fatemi, F. K. Error semitransparent universal control of a bosonic logical qubit. [arXiv:2603.15356](https://arxiv.org/abs/2603.15356) — **primary reference**
- Heeres, R. W., Reinhold, P., Ofek, N., *et al.* Implementing a universal gate set on a logical qubit encoded in an oscillator. *Nature Communications* **8**, 94 (2017). [doi:10.1038/s41467-017-00045-1](https://www.nature.com/articles/s41467-017-00045-1) — band-limited controls and the truncation-convergence criterion reused here
- Michael, M. H., Silveri, M., Brierley, R. T., *et al.* New class of quantum error-correcting codes for a bosonic mode. *Phys. Rev. X* **6**, 031006 (2016). [doi:10.1103/PhysRevX.6.031006](https://doi.org/10.1103/PhysRevX.6.031006) — binomial codes
- Bradbury, J., Frostig, R., Hawkins, P., *et al.* JAX: composable transformations of Python+NumPy programs (2018). [github.com/jax-ml/jax](https://github.com/jax-ml/jax) — autodiff backend
