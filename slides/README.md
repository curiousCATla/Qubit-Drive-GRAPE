# EsT slides: theory and pulse optimisation

Beamer deck and speaker script for presenting Roy, Wetherbee & Fatemi,
*Error Semitransparent Universal Control of a Bosonic Logical Qubit*
(`arXiv:2603.15356`) at group meeting, together with our simulation replication (`EST/`).
45 main slides plus 22 backup slides:

- **Opening** (1): title.
- **The basics** (2–13, for an experimental audience, pictures first): what a bosonic qubit is,
  how we steer a cavity through the transmon and χ, how the gates are made (GRAPE only: a
  computer-shaped waveform next to a familiar π pulse), what a gate does to the code space (a TikZ
  cartoon, then the paper's Fig. A5), correcting one photon loss at rest (parity check, the QEC
  cycle), the problem (a loss mid-gate), the fix the paper aims for (its Fig. 1a–b, the headline
  result, attributed), and the hardware: Fig. A1 (package), Fig. A2 (wiring), Table I (retyped).
- **Theory, intuitively** (14–24, no formulas apart from slides 14 and 19): the cavity–transmon
  Hamiltonian (14); Knill–Laflamme as two rules, "a loss must not mix the codewords" and "must not
  reveal which codeword" (15–17); error transparency as two tracks turning in step (18), then the
  paper's commuting square (19); why linear drives give only *semi*-transparency (20–21: which terms
  let a loss pass, Kerr makes the logical phase drift); the instantaneous code and error spaces as moving
  ribbons (22–23); the three checks and the paper's metric names (24).
- **The paper's experiment** (25–31): our X gate's three metrics in time against the paper's
  Fig. 1d–f (25) and the numbers table, agrees / differs (26), both moved up from the results; the
  test sequence (Fig. 3a, 27); the X gate (Fig. 3b–c), H and T (Fig. A7), the per-gate ~5× summary
  (Fig. 3d), and AQEC measured (Fig. 4a–b).
- **Pulse optimisation** (32–42, App. C): what GRAPE does, the model (drift, truncation), the
  I/Q rewrite of the drive, the piecewise-constant propagator, the inputs and targets, the cost
  function (C1, C2, C3, two-stage weights, Ord as C1 alone), the hardware constraints.
- **Results vs the paper** (43): slide 25's EsT and Ord pulses (`est_experiments.ipynb` Figure 2a)
  against the EsT gate with an end-of-gate error-space term (C_err in stage 1) that the pick rule
  of `est_optimization.ipynb` §7.2 chose from 13 seeds (seed 6, final gate; Figure 7c), with
  caveats.
- **EsT gate simulation** (44, our extension, not in the paper): `FWM_tails.ipynb` Figure 8a redrawn,
  N X gates then one 0.8 μs AQEC tail on Shirol's chip (EsT and Ord with and without the tail, idle + tail;
  idle without a tail and Shirol's teeth left off).
- **Closing** (45): our replication's status and open questions.
- **Backup** (B1–B22, after `\appendix`, numbered separately by `appendixnumberbeamer`): the formal
  theory, moved out of the main deck. Error correction and Knill–Laflamme with its derivation
  (B1–B6), the kitten code (B7–B8), why a gate breaks QEC, error closure and the Kerr obstruction
  (B9–B11), the instantaneous subspaces (B12–B13), the three conditions (B14–B17), the three metrics
  (B18–B21), reading the Hamiltonian term by term (B22).

Other results (the simulated AQEC with EsT vs Ord vs LE, Fig. A6, the Wigner and composite-sequence data) are out of
scope.

## Files

| File | Role |
|---|---|
| `preamble.tex` | Theme, colours, math macros, TikZ libraries, note-page template |
| `content.tex` | All 67 frames (45 + 22 backup), each with a `\note{}` speaker block |
| `make_comparison_figs.py` | Builds `img/paper_fig1def.png`, `img/ours_fig2a.png` and `img/ours_err_pick.png` for the results slides |
| `make_fwm_repeat_fig.py` | Builds `img/ours_fwm_repeat_B1.png` (slide 44) from `tables/fwm_tail_repeat_B1.csv` and `tables/fwm_tail_repeat_kerr.csv`; asserts the N = 20 values |
| `digitise_paper_fig1.py` | Digitises the paper's Fig. 1d–f curves and prints each one's time average, peak and end value (slide 26's table); `--overlay` saves a check image |
| `make_paper_crops.py` | Crops the paper's Figs. 1a–b, 3, 4a–b, A1, A2, A5, A6 and A7 into `img/` for the basics and experiment slides |
| `img/` | Generated images for the basics and results slides; do not hand-edit (except `kitten.png`, a hand-added photo beside the kitten-code bullet on slide 2) |
| `est_theory_slides.tex` | Driver → `est_theory_slides.pdf` (the deck) |
| `est_theory_notes.tex` | Same body, `notes=only` → `est_theory_notes.pdf` (script PDF) |
| `est_theory_script.md` | Markdown speaker script, **generated** |
| `make_script.py` | Regenerates `est_theory_script.md` from `content.tex` |

The deck and the script share one source: the `\note{}` blocks in `content.tex`. Edit those,
never `est_theory_script.md`.

## Build

```bash
python3 slides/make_paper_crops.py       # only if the paper PDF changed
python3 slides/make_comparison_figs.py   # only if the paper PDF or our Figure 2a changed
python3 slides/make_fwm_repeat_fig.py    # only if the fwm_tail_repeat tables changed
cd slides
tectonic est_theory_slides.tex
tectonic est_theory_notes.tex
python3 make_script.py
```

`tectonic` is the only LaTeX engine on this machine (there is no `pdflatex` or `latexmk`).
The **first** compile downloads `beamer`, `pgf`/`tikz` and the Latin Modern fonts from the
TeX Live bundle, so it needs network access; later builds are offline. Tectonic reruns the
passes itself, so the footline and frame counts resolve in one invocation.

Both files must compile with **no** `Overfull \hbox`/`\vbox` warnings — on a Beamer frame an
overfull vbox means text ran off the bottom of the slide, which is exactly the failure mode
this deck is trying to avoid.

**Cross-references.** Slide text refers to other slides with `\slideref{name}`, resolved from
a `\slidelabel{name}` placed in the target frame (macros in `preamble.tex`), so reordering
never leaves a stale number. Speaker notes cannot use them (`make_script.py` does not resolve
labels), so notes name the slide instead ("the device slide").

## Figures

The intuitive theory slides (15–24) are TikZ too: good and bad codes on the Fock ladder, the
leaking-bucket and tilted-arrow picture for rule 2, the rigid / overlapping / squashed copies of
the code, the two-track Bloch arrows for transparency, the three-term pass / fail panel, the
Kerr fan of arrows, and the moving code and error ribbons with cartoon Fock bars (made-up
populations, as in the gate cartoon). The experiment slides (25–31) show more crops from
`make_paper_crops.py`: `img/paper_fig3a.png`, `paper_fig3b.png`, `paper_fig3c.png`,
`paper_fig3d.png` (page 5), `paper_fig4ab.png` (page 6) and `paper_figA7.png` (page 19).
`paper_figA6.png` (page 18, the simulated AQEC) is still cropped but no longer on a slide.

The basics slides draw their own TikZ pictures (the transmon and cavity ladders, the three-mode
block diagram and number-split transmon line, the π-pulse vs GRAPE waveform picture, the Fock-bar gate
cartoon, the parity-check circuit, the QEC-cycle and loss-time timelines) and show four crops of
the paper, made by `make_paper_crops.py` (600 dpi, each cut above its caption, which the slide
retells and cites): `img/paper_figA5.png` (Fig. A5, page 18), `img/paper_fig1ab.png` (Fig. 1a–b,
page 2), `img/paper_figA1.png` (Fig. A1, page 11) and `img/paper_figA2.png` (Fig. A2, page 13).
Table I is retyped, not cropped; its values were checked against the PDF text. The gate cartoon's
populations are made up, and the slide says so.

All pictures in the theory and optimisation slides are TikZ, drawn inline in `content.tex`:
the Fock ladder, scattered loss endpoints, the commuting square, the swept subspace tubes, the
GRAPE loop, the I–Q phasor with the amplitude circle, the piecewise-constant staircase and the
two-trajectory C2 picture.

The results slides use three raster images, all made by `make_comparison_figs.py`, which
writes only into `slides/img/` and never into `figures/`:

- `img/paper_fig1def.png`: **cropped from the paper** (page 2 of the PDF in `sources/`, 600
  dpi), panels d, e, f laid out in one row and cited on the slide. The crop boxes are hard-coded page pixels, so re-check the output by
  eye if the PDF changes.
- `img/ours_fig2a.png`: `../figures/est/fig1def_est_vs_ord_linear.png` (our Figure 2a,
  regenerated by `python3 EST/diagnostics.py`) with its title strip trimmed.
- `img/ours_err_pick.png` (slide 43): slide 25's EsT and Ord pulses plus
  `pulses/est/u_X_est_err_err2st_seed6.npy`, the gate picked in `est_optimization.ipynb` §7.2 (seed 6,
  final two-stage gate; the blue curve of Figure 7c), scored with `EST/diagnostics.analyze` at
  n_c = 20. The script checks the green traces against `run/seed6/final/*` in
  `tables/est_err2st_X_seeds_traces.npz` and stops if they differ.

Slide 44's `img/ours_fwm_repeat_B1.png` is made by `make_fwm_repeat_fig.py` from the cached tables only (no
simulation): panel (a) of `FWM_tails.ipynb` Figure 8 with five of its seven curves, "+ one 0.8 μs tail" renamed
"+ AQEC tail", "EsT-good" renamed "EsT", and the deck's colours (EsT blue, Ord red). The y axis starts at 0.4.

The paper's numbers on the results slides are either digitised (the time averages of all
three metrics and L(T), by `digitise_paper_fig1.py`; its leakage averages reproduce
`est_experiments.ipynb` §10) or read by eye from that crop (marked ≈ on the slide).

## Equation numbering

Slides cite the **paper's** numbers, because that is what the audience can look up.
`sources/My notes.pdf` numbers its equations independently; the mapping is:

| Notes | Paper | Content |
|---|---|---|
| (1) | Sec. II, unnumbered | ET commuting square |
| (2) | — (paper's static form is Sec. II, unnumbered) | `[H, E] = λ E` (shown as the special case of error closure, App. A) |
| (3) | (10), (A5) | Linear drive Hamiltonian |
| (4) | (A6) | The uncorrectable commutator |
| (5) | (1) | Instantaneous code space |
| (6) | (2) | Instantaneous error space |
| (7) | (3) | Condition 1, instantaneous correctability |
| (9) | (4) | Condition 2, subspace tracking |
| (10) | (5) | Condition 3, matched internal dynamics |
| (11), (12) | (6), (A8)–(A12) | QEC violation Δ_QEC |
| (13) | (7) | Instantaneous leakage L |
| (14) | (8) | Trajectory mismatch η |
| — | (9), (A1)–(A5), Table I | System Hamiltonian (slide 14, backup B22) |
| — | (10), App. B | I/Q rewrite of the drive (slide 34) |
| — | App. C, unnumbered | Constraints: 50 MHz, 4 MHz, 48 ns ramp, 1 ns steps (slides 36, 42) |
| — | (C1) | Gate infidelity C1 (slide 38) |
| — | (C2)–(C3) | Transparency cost C2 (slide 39) |
| — | (C4)–(C7) | Velocity-variance regulariser C3 (slide 40) |
| — | (C8) | Weighted total cost (slide 37) |
| — | Fig. 1d–f | Paper's X-gate metrics, compared with ours (slides 25–26) |

## Where the slides deviate from `sources/My notes.pdf`

The slides follow the paper; each deviation is flagged in the corresponding speaker note.

1. **The obstruction slide (B11), the commutator.** The notes write
   `−(K − χ q†q) a†a²`. Paper Eq. (A6) is `−(K + χ′ q†q) a†a²` — **plus** sign, and `χ′`
   (the higher-order dispersive coupling), not `χ`.
2. **Metric 1 (B19), the norm.** The notes give `‖A‖₂ = √(λ_max(AᵀA))` (spectral) but then quote
   the remainder as `2(|x|²+|y|²+|z|²)`, which is the Frobenius result. App. A defines
   `‖A‖₂ = Tr(A†A)` — squared Frobenius, no square root — and that is where the factor of 2
   comes from. Same adjudication as `EST/diagnostics.py:179` ("App. A wins").
3. **Condition 1 (B15), the Condition-1 matrix.** The notes write a 2×2 array of *operators*
   (`P_C I†I P_C`, …) set equal to `diag(1, 2)`. The slide shows the matrix of scalar
   *coefficients* `c_ij(0) = diag(1, 2)`, with the operator equation stated separately.

## Where the optimisation slides differ from what we train with

The optimisation half is **paper first**: slides show the printed App. C equations. Where our
replication (`EST/`) differs, the slide carries a small grey footnote and the speaker note
explains. Only choices the paper does not state (truncation n_c = 20, transmon g/e/f, the C4
amplitude penalty) appear on a slide as ours, tagged `[ours]` / `[our truncation]`.

1. **C2 (slide 39).** Printed: divide by 𝒩. Ours: divide by 𝒩², so F_ET ≤ 1 (Cauchy–Schwarz);
   the printed form is unbounded and rewards photon number. See `EST/README.md`.
2. **C3 (slide 40).** Printed: raw velocity variance (units of frequency²). Ours: divided by
   mean(v)², unit-free and not minimised by parking.
3. **GRAPE and the staircase (32, 35), optimiser.** Paper: Dynamiqs + JAX + Adam. Ours: JAX `expm` autodiff (and
   a numpy hand-adjoint pipeline) + L-BFGS-B. Mentioned in notes only.
4. **I/Q (34), sign of Q.** Slides use the paper's `ε â + h.c.`; our code uses `ε â† + h.c.`,
   so there ε = u₀ − iu₁. Only the sign of Q differs.

## Mentor-review corrections (2026-10-18)

Statements tightened after a critical read, so that no slide claims more than the paper or
our data support. Keep them this way:

- Slide B10 states **error closure** (`[H, â]` in the correctable span), not `[H, E] = λE`:
  the linear drive and χ fail the strict form yet are correctable (App. A).
- "Exact ET is **not expected**" (the paper's words), never "impossible".
- Idling is **nearly** transparent (residual Kerr; η ≈ 0.03 with no drive in our check).
- Condition 1 is **not** just "matched n̄" mid-gate; matched n̄ is necessary, not sufficient.
- d_j in Eq. 5 is real by Hermiticity (the paper's text says complex; the slide says both).
- EsT vs Ord gaps are pinned on the cost only if they beat the seed-to-seed spread; our
  headline pairs are mostly single-seed.
- The paper's Table I gives the K′ operator form (`K′a†³a³/6`); it is not our reading.
