# Phase 2 — Leakage / Seepage Report

Measurement only. No pulses were changed. No ideal full-space unitary was invented.
Parameters: n_t=3, dt=0.002 μs, α=√3, main pulses 550×4, headline n_c=24.

## Probe set

Fixed in `validation/outside_inputs.py`, identical for every gate:

- **A** Transmon-excited: `|e,0>`, `|e,1>`, `|e,2>`
- **B** Opposite parity: `|g,1>`, `|g,3>`, `|g,5>`
- **C** Same-parity orthogonal-to-code: residuals of `|g,4>`, `|g,6>`, `|g,8>`
  after projecting out the logical cats (Gram–Schmidt; true outside probes).

## Metrics

- **L1** (code → out): average leakage on the gate’s natural 2×2 block
  (`leakage_L1`). For I/X/Y/Z/H/T this is the cat-subspace block; for enc/dec
  it is the encode/decode map of Phase-0 tier4.
- **L2** (out → code): mean cat-subspace population after the pulse, averaged
  over true-outside probes (P_logical_initial < 1e-12). Localization diagnostic;
  do not add to L1 as a double-counted error budget.

## Summary at n_c=24

| Gate | L1 (code→out) | L2 (out→code) | Dominant leak destination | Flag |
|------|---------------|---------------|---------------------------|------|
| enc | 3.096013e-03 | 9.760694e-02 | g, odd photon (wrong parity) | watch+seepage_note |
| X | 2.580545e-03 | 1.110431e-04 | transmon excited, odd photon | dangerous |
| Y | 2.348341e-03 | 3.874921e-05 | g, odd photon (wrong parity) | watch |
| dec | 2.001333e-03 | 7.696970e-02 | g, even photon (outside code / radial) | watch |
| H | 1.668596e-03 | 8.417685e-05 | g, odd photon (wrong parity) | watch |
| Z | 7.606574e-04 | 1.003622e-06 | transmon excited, even photon | watch |
| I | 4.755784e-04 | 7.877761e-07 | g, even photon (outside code / radial) | document |
| T | 2.091518e-04 | 2.588655e-06 | g, even photon (outside code / radial) | document |

## Three highest-L1 gates — plain language

### 1. U_enc  (L1 = 3.0960e-03)

When amplitude leaves the intended subspace, it lands predominantly in **g, odd photon (wrong parity)** (`P_g_odd`). Seepage L2 = 9.7607e-02 (max single-probe P_logical from outside = 7.7069e-01). Flag: **watch+seepage_note**.

### 2. U_X  (L1 = 2.5805e-03)

When amplitude leaves the intended subspace, it lands predominantly in **transmon excited, odd photon** (`P_e_odd`). Seepage L2 = 1.1104e-04 (max single-probe P_logical from outside = 3.1025e-04). Flag: **dangerous**.

### 3. U_Y  (L1 = 2.3483e-03)

When amplitude leaves the intended subspace, it lands predominantly in **g, odd photon (wrong parity)** (`P_g_odd`). Seepage L2 = 3.8749e-05 (max single-probe P_logical from outside = 1.3571e-04). Flag: **watch**.

## Single-photon-loss error subspace

The even cat code has one physically distinguished outside subspace:

    E = span{ a|+Z_L>, a|-Z_L> }   (`core.propagator.error_basis`)

exactly orthonormal and exactly orthogonal to the code space (the cats
live on Fock n ≡ 0, 2 mod 4, their loss images on n ≡ 3, 1). Group **B**
above samples the odd manifold generically; E *is* the state a real loss
produces.

**Is a loss exactly correctable to begin with?** Knill–Laflamme for
E = {I, a} needs `<i_L|a†a|j_L> = c·δ_ij`. The cross term `||B†aB||` and the
off-diagonal are exactly zero (0.0e+00, 0.0e+00), but the diagonal is not:

| n̄(+Z_L) | n̄(−Z_L) | relative mismatch |
|----------|----------|-------------------|
| 3.2641 | 2.7562 | 0.169 |

So the α=√3 four-component cat is an **approximate** single-loss code, not
an exact one. The exact version is the binomial kitten code in
`EST/kitten_code.py`, where `a|0_L> = √2|3>` and `a|1_L> = √2|1>` carry equal
weight — the coincidence its `error_cardinals` assertion pins.

### Metrics

- **L1_E** (E → out): `leakage_L1(E† U E)`. Population the pulse removes from
  the error subspace, i.e. logical content of the loss that does not survive.
- **L2_E→C** (E → code): mean `||B† U|e_j>||²`. The dangerous direction —
  amplitude back in the *even* code space reads as error-free to a parity
  check, converting a flagged, correctable loss into a silent logical error.
- **F_ET**: `pedersen_gate_fidelity(U_ideal, E† U E)` — did the gate perform
  its intended rotation on E as well? This is error transparency. No training
  objective in this repo ever referenced E, so it is unconstrained, not
  regressed.

### Idle baseline (drive off, T = 1.100 μs)

`L1_E = 2.0108e-02`, `L2_E→C = 0.00e+00`, `|M_E offdiag| = 0.00e+00`.

H0 is exactly diagonal, so drift cannot move amplitude across parity
sectors: seepage back into the code space is zero to machine precision and
the loss stays flagged. The residual L1_E is Kerr n² dephasing *within* the
error word, and the block is diagonal — a deterministic, known rotation, so
a frame update rather than lost information. **This is what "correctable in
idle" means quantitatively**, and it is the reference the gate rows below
are measured against.

### Under the trained pulses (n_c=24)

| Gate | L1_E (E→out) | L2_E→C (E→code) | F_ET | Dominant destination |
|------|--------------|-----------------|------|----------------------|
| enc | 9.4073e-01 | 7.9425e-05 | — | `P_g_even_nonlogical` |
| H | 9.3515e-01 | 2.0603e-05 | 0.0299 | `P_e_even` |
| dec | 9.0792e-01 | 3.3344e-02 | — | `P_g_odd` |
| X | 8.4053e-01 | 5.8259e-05 | 0.0748 | `P_e_odd` |
| Y | 6.9838e-01 | 1.5669e-05 | 0.1016 | `P_g_odd` |
| Z | 3.0342e-01 | 5.9165e-07 | 0.3423 | `P_g_odd` |
| I | 6.3513e-02 | 1.7505e-08 | 0.9360 | `P_g_odd` |
| T | 3.7955e-02 | 9.5114e-08 | 0.9466 | `P_g_odd` |

Read enc's row as a robustness probe only: enc's input subspace is
{|g,0>, |e,0>}, so a post-loss state is not an input it legitimately sees.
dec is the opposite — its input *is* the code space, so E is exactly "a
photon was lost before decode fired".

### Reading

The trained gates are **not error-transparent**: on the six logical gates L1_E runs 3.80e-02–9.35e-01 and F_ET falls as low as 0.030, while the *code-space* L1 for the same gates stays at 2.58e-03 or below — the pulses protect C between 134× and 560× better than they protect E. Expected, since E was never in any objective, but it is the gap the `EST/` track's C2 cost is built to close on a different code.

Seepage back into the code space stays small for the logical gates (L2_E→C ≤ 5.83e-05): a loss is still *detected*, it just stops being *correctable*. The exception is **U_dec** at L2_E→C = 3.33e-02 — that fraction of a post-loss state re-enters the even manifold, where the parity check reports no error at all.

## Truncation cross-check

Top-L1 gates re-scored at n_c ∈ {22, 24, 26}. Quantitative bin weights may drift; destination **class** should not.

- **U_enc**: fine bins {22: 'P_g_odd', 24: 'P_g_odd', 26: 'P_g_odd'} (STABLE); coarse class {22: 'g_odd_parity', 24: 'g_odd_parity', 26: 'g_odd_parity'} — STABLE
- **U_X**: fine bins {22: 'P_e_odd', 24: 'P_e_odd', 26: 'P_e_even'} (shifts within class); coarse class {22: 'transmon_excited', 24: 'transmon_excited', 26: 'transmon_excited'} — STABLE
- **U_Y**: fine bins {22: 'P_g_odd', 24: 'P_g_odd', 26: 'P_g_odd'} (STABLE); coarse class {22: 'g_odd_parity', 24: 'g_odd_parity', 26: 'g_odd_parity'} — STABLE

## Decision

At least one gate is flagged **dangerous** (leaked amplitude prefers excited-transmon or truncation-edge subspaces). Consider a Phase-3 targeted constraint; do **not** retrain blindly.

Seepage L2 is non-negligible on at least one gate: outside population can enter the code space under the pulse. For protocols that assume the code space is only entered via encode, this is more harmful than ordinary leakage — note for system-level design.

No pulses were modified in Phase 2.
