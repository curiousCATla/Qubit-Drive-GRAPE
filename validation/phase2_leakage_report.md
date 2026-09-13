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
| dec | 3.617338e-03 | 9.623468e-02 | g, even photon (outside code / radial) | watch+seepage_note |
| X | 1.781625e-03 | 2.348177e-05 | g, even photon (outside code / radial) | document |
| H | 1.697523e-03 | 2.417924e-05 | g, even photon (outside code / radial) | document |
| Y | 1.267731e-03 | 1.777679e-05 | g, odd photon (wrong parity) | watch |
| enc | 1.143546e-03 | 9.725052e-02 | g, even photon (outside code / radial) | watch+seepage_note |
| Z | 5.788654e-04 | 5.024072e-06 | transmon excited, even photon | watch |
| I | 4.930576e-04 | 8.792398e-07 | g, even photon (outside code / radial) | document |
| T | 2.103567e-04 | 2.667797e-06 | g, even photon (outside code / radial) | document |

## Three highest-L1 gates — plain language

### 1. U_dec  (L1 = 3.6173e-03)

When amplitude leaves the intended subspace, it lands predominantly in **g, even photon (outside code / radial)** (`P_g_even_nonlogical`). Seepage L2 = 9.6235e-02 (max single-probe P_logical from outside = 3.4972e-01). Flag: **watch+seepage_note**.

### 2. U_X  (L1 = 1.7816e-03)

When amplitude leaves the intended subspace, it lands predominantly in **g, even photon (outside code / radial)** (`P_g_even_nonlogical`). Seepage L2 = 2.3482e-05 (max single-probe P_logical from outside = 6.5585e-05). Flag: **document**.

### 3. U_H  (L1 = 1.6975e-03)

When amplitude leaves the intended subspace, it lands predominantly in **g, even photon (outside code / radial)** (`P_g_even_nonlogical`). Seepage L2 = 2.4179e-05 (max single-probe P_logical from outside = 7.1077e-05). Flag: **document**.

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
| dec | 9.2593e-01 | 6.7498e-02 | — | `P_g_odd` |
| X | 9.2467e-01 | 3.0460e-05 | 0.0606 | `P_e_even` |
| enc | 9.1853e-01 | 7.4310e-05 | — | `P_g_even_nonlogical` |
| Y | 8.6572e-01 | 1.7781e-05 | 0.0551 | `P_g_odd` |
| H | 8.4617e-01 | 1.3488e-05 | 0.0661 | `P_g_odd` |
| Z | 2.3463e-01 | 3.5387e-06 | 0.3327 | `P_g_odd` |
| I | 8.9601e-02 | 1.1198e-07 | 0.9101 | `P_g_odd` |
| T | 4.8343e-02 | 3.4942e-07 | 0.9382 | `P_g_odd` |

Read enc's row as a robustness probe only: enc's input subspace is
{|g,0>, |e,0>}, so a post-loss state is not an input it legitimately sees.
dec is the opposite — its input *is* the code space, so E is exactly "a
photon was lost before decode fired".

### Reading

The trained gates are **not error-transparent**: on the six logical gates L1_E runs 4.83e-02–9.25e-01 and F_ET falls as low as 0.055, while the *code-space* L1 for the same gates stays at 1.78e-03 or below — the pulses protect C between 182× and 683× better than they protect E. Expected, since E was never in any objective, but it is the gap the `EST/` track's C2 cost is built to close on a different code.

Seepage back into the code space stays small for the logical gates (L2_E→C ≤ 3.05e-05): a loss is still *detected*, it just stops being *correctable*. The exception is **U_dec** at L2_E→C = 6.75e-02 — that fraction of a post-loss state re-enters the even manifold, where the parity check reports no error at all.

## Truncation cross-check

Top-L1 gates re-scored at n_c ∈ {22, 24, 26}. Quantitative bin weights may drift; destination **class** should not.

- **U_dec**: fine bins {22: 'P_g_odd', 24: 'P_g_even_nonlogical', 26: 'P_g_even_nonlogical'} (shifts within class); coarse class {22: 'g_odd_parity', 24: 'g_even_radial', 26: 'g_even_radial'} — UNSTABLE
- **U_X**: fine bins {22: 'P_g_even_nonlogical', 24: 'P_g_even_nonlogical', 26: 'P_g_even_nonlogical'} (STABLE); coarse class {22: 'g_even_radial', 24: 'g_even_radial', 26: 'g_even_radial'} — STABLE
- **U_H**: fine bins {22: 'P_g_even_nonlogical', 24: 'P_g_even_nonlogical', 26: 'P_g_even_nonlogical'} (STABLE); coarse class {22: 'g_even_radial', 24: 'g_even_radial', 26: 'g_even_radial'} — STABLE

## Decision

Some gates are flagged **watch** (non-negligible excited-transmon component of leakage and/or elevated seepage). Documented here; no pulse change in Phase 2. Revisit only if a protocol is sensitive to that landing.

Seepage L2 is non-negligible on at least one gate: outside population can enter the code space under the pulse. For protocols that assume the code space is only entered via encode, this is more harmful than ordinary leakage — note for system-level design.

No pulses were modified in Phase 2.
