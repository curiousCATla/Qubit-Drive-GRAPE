# Speaker script — EsT theory and pulse optimisation

Generated from `content.tex` by `make_script.py`; edit the `\note{}` blocks
there, not this file. One section per slide, numbered to match
`est_theory_slides.pdf`.

Paper: Roy, Wetherbee & Fatemi, *Error Semitransparent Universal Control of a
Bosonic Logical Qubit*, arXiv:2603.15356. Equation numbers are the paper's.

---

## 1. Title slide

- Paper: Roy, Wetherbee & Fatemi, arXiv:2603.15356 (7 Apr 2026), Cornell + Berkeley.
- Headline claim: a logical gate set {X,H,T} on a binomial bosonic qubit that stays *almost* error-transparent mid-gate, using only ordinary linear drives.
- Result to quote: five-fold reduction in logical infidelity conditioned on a photon loss during the gate.
- Two halves today. First the paper's theory, up to the three metrics. Then how the pulses are found: GRAPE, the model it runs on, the cost function and the constraints, with the places our replication differs marked "ours". The results, simulated and experimental, are a separate talk.
- Be explicit that this is the paper's work presented by me, plus our simulation replication — not my result.
- Framing sentence to open with: bosonic memories already beat break-even while *idling*. This paper is about what breaks when you try to do a *gate*.

## 2. Why gates are the hard part

- One minute, to answer "why should I care" before any equations.
- Memories: cavity-encoded qubits (binomial, GKP) have been run past break-even — corrected, they live longer than the best uncorrected encoding in the same hardware. That is idling only.
- Gates: the moment you drive the system, an error that happens mid-gate is transformed by the remaining pulse. Unless the gate is designed for it, the error that comes out is not the one the code knows how to fix.
- Error transparency is the known fix; the paper's abstract notes experiments so far have only done phase gates. Their contribution is non-phase gates (X, H) with only standard linear drives, at the price of being only *semi*-transparent.
- Quote the headline and attribute it: about five-fold lower infidelity conditioned on a photon loss mid-gate, measured. We do not reproduce the experiment.
- Then: the first half is the theory of why this is hard and how they measure it; the second half is how the pulses are optimised, with our replication marked.

## 3. Where we are going

- Five ideas, one chain. Signpost the argument, not the list of slides.
- **1. Correct a loss at rest.** The device, the Knill–Laflamme condition, and the kitten code that meets it. QEC while idling is solved and demonstrated.
- **2. Keep that through a gate.** The natural demand is error transparency: a loss during the gate should come out the same as a loss after it. The punchline arrives early: with linear drives, exact transparency is not expected, only *semi*-transparency.
- **3. Measure the shortfall.** So the paper lets the code and error spaces move in time, states three conditions, and gives one number for each.
- **4. Optimise the pulse.** GRAPE finds the drives, with a cost that stands in for those conditions and stays inside the hardware limits.
- **5. Check against the paper.** Our simulated X gate against the paper's Fig. 1d–f, what agrees and what does not, and what none of this covers.
- Emphasise: nothing here makes exact transparency possible. The contribution is that the failure becomes *quantifiable and optimisable*.

## 4. The device: a cavity dispersively coupled to a transmon

- Two modes. The cavity a is where the logical qubit lives; the transmon q is a nonlinear ancilla we use to steer the cavity and to read out parity.
- H_a: the cavity's bare frequency, plus self-Kerr K and a smaller second-order self-Kerr K'. Both make the photon energy depend on how many photons are already there.
- H_q: transmon frequency plus anharmonicity K_q. At -180 MHz this is what makes the transmon usable as a qubit at all — and it is why the simulation keeps three transmon levels, not two.
- H_{int}: the dispersive coupling chi, plus a second-order correction chi'. Both are photon-number dependent shifts conditioned on the transmon.
- H_{drive}: the only thing we control. Two complex amplitudes, linear in the mode operators.
- These are exactly the numbers and operator forms in `EST/device.py`.
- If asked why the main text shows fewer terms: Eq. 9 is the low-order version; App. A adds K' and chi', and chi' matters later — it is one of the two terms that make exact transparency impossible.

## 5. Reading the Hamiltonian

- The key term is chi. Written as (omega_a + chi q^dagq)a^daga it is a cavity frequency that depends on the transmon; written as (omega_q + chi a^daga)q^dagq it is a transmon frequency that depends on photon number. |chi|/2pi = 3.66 MHz is far larger than the transmon linewidth, so the photon-number peaks are resolved: a transmon pulse longer than ~ 1/|chi| ( ≈ 270 ns) can address individual photon numbers.
- The Kerr-type terms are tiny (K/2pi = -22 kHz), but they do not commute nicely with a. Plant the seed: this is where "semi" will come from.
- Rotating frame: in the simulation there is no omega_a or omega_q. What remains is the static nonlinearity the pulse has to work with and against.
- Photon loss, jump operator a, is the error the kitten code is designed to correct. It is the E_j that will appear in the Kraus form on the next slide.
- Flag the symbol clash out loud: drive epsilon(t) versus channel varepsilon(rho).

## 6. Setup: code space, projector, errors

- Two objects, then one warning.
- The projector: P is Hermitian and idempotent — the linear-algebra projection onto the code subspace. U^dag U = I on the logical space, but P = UU^dag is *not* the identity on the full Hilbert space. That asymmetry is the whole point of encoding.
- The Kraus operators E_j are what we casually call "the errors". Same maths as simulating decoherence.
- The warning is load-bearing, not a footnote: {E_j} is only defined up to a unitary recombination W. The proof of Knill–Laflamme *uses* this to pick a basis in which the error matrix is diagonal.
- If asked: this is why "correcting three errors" and "correcting a three-dimensional span of errors" are the same statement.

## 7. The Knill–Laflamme theorem

- Read the equation aloud in words before unpacking it: sandwich any product of two errors between the code projectors, and you must get back the projector itself times a number.
- "Proportional to the logical identity" is the phrase to repeat — it recurs verbatim as Condition 1 later in the talk.
- The codeword form makes the two demands visible. Spend time here.
- delta_{ij}: no leakage *between* codewords. If E_a^dag E_b had an off-diagonal piece, an error would rotate |0_L> toward |1_L> and no measurement could undo it.
- alpha_{ab} not depending on i: no which-path information. This is the subtler half and the one the kitten code buys us on the next slides.
- Hermiticity of alpha is automatic — it comes out of the proof as a Gram matrix, hence also positive semidefinite.
- Do NOT go through the full proof. The next slide sketches why the condition is enough; the derivation is in backup.

## 8. Why that condition is enough (sketch)

- This replaces three derivation slides; they are in backup if someone asks.
- Step 1 uses the freedom from the setup slide: any unitary recombination of the Kraus operators is the same channel. Pick the one that diagonalises alpha.
- Step 2: after that, F_k restricted to the code is sqrt(d_k) times a rigid move onto a copy C_k, and different k give orthogonal copies.
- Step 3: orthogonal copies can be told apart by a projective measurement; rigid copies can be rotated back. The outcome probability d_k is the same for every code state, so the measurement reveals nothing logical. That is the "alpha independent of i" half of Knill–Laflamme, seen as a measurement property.
- Kitten preview: the copies are the even and odd parity sectors, so the syndrome is a parity measurement.

## 9. The binomial "kitten" code

- One logical qubit in one cavity mode. Two codewords, two error words, five Fock levels in total — hence "kitten", a small sibling of the four-legged cat code.
- Parity is the organising principle. Codewords are even, loss images are odd, so a single parity measurement detects that a loss happened.
- Walk the arrows on the diagram. Note a|0>=0, so only the |4> component of |0_L> survives the loss: a|0_L>=(0+2|3>)/sqrt(2) = sqrt(2)|3>.
- The sqrt(2) appearing in *both* lines is not a coincidence and it is not cosmetic. It is the whole reason this code works — next slide.
- If asked why |4> and not |6>: binomial codes are built so the codewords' photon-number distributions match to the order of loss you want to correct. This one corrects single loss.

## 10. Why the kitten code satisfies Knill–Laflamme

- Map the two facts onto the two halves of Knill–Laflamme explicitly — this is the payoff of having read the equation carefully earlier.
- Fact 1 is the delta_{ij} half. The images are orthogonal, so the loss is an isometry onto the error space; logical information is preserved, just relocated.
- Fact 2 is the "alpha independent of i" half. Both codewords have nbar = 2: for |0_L> it is (0+4)/2 = 2, for |1_L> it is just 2. Photon loss is an amplitude-damping process whose *rate* depends on nbar, so matched nbar means matched loss rate, which means the leak of information to the environment carries no logical content.
- Say the punchline: the syndrome is the parity flip, even -> odd. Detectable and recoverable. This is how bosonic memories beat break-even while idling.
- Remember the number 2 — it comes back as c_{aa}=2 in Condition 1.
- If asked about dephasing: what would break Fact 2 is the two codewords damping at different rates. Then the environment's record correlates with the logical bit and you get irreversible logical dephasing, which no end-of-gate recovery can undo. App. A raises exactly this as the reason Delta_QEC is worth minimising.

## 11. Why a gate breaks all of that

- This slide is the hinge of the talk. Do not rush it — the next equation is unmotivated without it.
- Idling QEC works because the codewords are nearly *static*. The photon leaves while the state sits in the configuration the code was designed around, so the loss image is one of two known error words.
- Say "nearly" on purpose. Kerr is on while idling too, so the error words pick up a small phase that depends on when the photon left. Our own check: with no drive at all, the trajectory-mismatch metric is already about 0.03. It is the same Kerr effect that comes back as the obstruction — just small when nothing moves.
- During a gate that premise is gone. At an intermediate time the state is not a codeword at all, just some vector the drive swept it to.
- Point at the figure: same ideal trajectory, three loss times, three endpoints. The final state is t-dependent, and t is a random variable we do not measure.
- So end-of-gate recovery has no single target to undo. A syndrome measurement can still tell you *that* a photon left, but not how to put the state back.
- Set up the next slide as the obvious fix: demand that the jump time stop mattering.

## 12. Error transparency: the jump commutes through the gate

- State the equation in words first: the two routes around the square agree, up to an overall phase that may depend on t.
- The phase e^{iphi_j(t)} is free because a global phase is physically invisible — it does not appear in rho.
- What this buys: collapse a continuum of possible outcomes (one per jump time) into a single outcome. The error becomes *transparent* to the gate — it commutes through, up to a phase.
- Concretely: after the gate, you measure parity, find you are in the error space, and apply the same recovery you would have used for an idling qubit. The gate did not scramble the error.
- Heads-up for later: this exact square, with the subspaces allowed to move, is what the three conditions in a few slides are designed to enforce.
- If asked about multiple losses: the framework is built for single loss, the dominant channel for this code. Two-photon events are not covered.

## 13. When does a loss commute through? Error closure

- Reframe the commuting square as a statement about the Hamiltonian. Moving the jump a from before H to after it produces the commutator [H,a]. If that commutator is itself a correctable error, a loss at any time still ends as a correctable error at t=T. That is error closure (App. A, citing the error-closure literature); exact transparency, [H,a] = lambdaa, is the strict special case.
- Walk the table. The bare frequencies and the leading dispersive term give a back, times at most a transmon number operator: a deterministic phase on the error state (the paper's words).
- The linear drives: [epsilona + epsilon^{*}a^dag, a] = -epsilon^{*}, a plain number, and the transmon drive commutes with a. A number times the identity is in the correctable set {I, a}, so linear drives are fine. This is why the paper can use the standard cQED toolkit at all.
- Then the two that fail: self-Kerr K and the higher-order dispersive chi'. Their commutator is n a, a jump weighted by photon number — next slide.
- If pressed on the chi row: the phase depends on the transmon state, and mid-gate the transmon is driven. The paper counts it as correctable; the honest statement is "correctable as long as the transmon path is the planned one".
- Why defend linear drives: strong nonlinear drives have been proposed for exact ET, but they need strong nonlinearities and tight confinement to the static code space, and bring their own spurious processes. The paper's bet is that simpler wins.

## 14. The obstruction: nonlinear terms that will not commute

- Recall the table on the last slide: the frequencies, the leading chi and the linear drives all pass the error-closure test. Only the nonlinear cavity terms fail.
- K' also fails, with a -(K'/2)a^dag{}^2a^3 term. At the photon numbers used it is much smaller than the K term (about 40 times at n=4), and Eq. A6 leaves it out.
- The two that break it are the self-Kerr K and the higher-order dispersive chi'. Note for yourself: it is chi', the higher-order coupling, not the leading chi — and the sign in front of chi' is PLUS.
- Why a^daga^2 is fatal: write it as n a. The a is the ordinary jump; n then weights the result by the photon number left behind. On the kitten, |4> -> |3> gets weight 3 and |2> -> |1> gets weight 1. Same error space, so the delta_{ij} demand still holds; but i_L|(na)^dagna|i_L is 18 for |0_L> and 2 for |1_L>, so the error learns which codeword it hit.
- Over a gate this shows up as a relative phase of about 2K(T-t) between the two error words: a logical Z rotation by a random angle, since t is random. Averaged over t, that is logical dephasing. Roughly 0.28 rad at most for a 1 mus gate.
- Numbers if asked: K_a/2pi = -22 kHz for the cavity self-Kerr, chi/2pi = -3.66 MHz for the dispersive coupling.
- Wording: the paper says exact ET is "not expected" here, not that it is proven impossible. The commutator shows the operator-level route fails; say that, no more.
- Land the word: this is what "semi" in error semitransparent names. It is a deliberate trade, not a failure to converge. Keep repeating this — it explains why the metrics on trained pulses plateau at a floor.
- App. A's suggested escape routes, if asked: engineer smaller K and chi, keep intermediate photon number low, or go to strong nonlinear drives and pay the complexity cost.

## 15. Letting the code space move

- This is the paper's central conceptual move, and it is the reason the title says "dynamic encoding subspaces".
- The old framing fails for a trivial reason: a gate has to move the state, so insisting it stay inside a fixed two-dimensional subspace rules out the gates we want. Earlier exact-ET proposals accepted that and paid for it with strong nonlinear drives and very tight confinement.
- The fix: define the code space to be wherever the ideal evolution has carried the codewords by time t. By construction the ideal, lossless trajectory never leaves C(t).
- Emphasise that this is not a weaker demand in a hand-wavy sense — it is a different, well-posed demand. We still ask for consistency; we just ask for consistency with the intended trajectory rather than with a static subspace.
- Underline that U(t,0) is the ideal propagator. All three metrics are computed from lossless unitary simulation; the loss operator appears only as the algebraic object E_j, never as an actual quantum jump.

## 16. And the error space with it

- Same move, applied to the error space: push the error space along with the code space, so at each instant it is the image of the *current* code space under the error.
- For this code and this error the error space is again 2-dimensional — a restricted to the code space has rank 2. Worth flagging: in the repo's own implementation this is a real trap; feeding six cardinal states instead of the two code words yields four spurious basis vectors outside the error space and silently understates leakage.
- The proportionality in P_(E_j) ∝ E_jP_C E_j^dag is not an oversight. E_jP_C(t)E_j^dag is not a projector — a is not unitary, so it does not preserve norms. It has to be renormalised, and the paper fixes it only up to a constant. At t=0 this is exact for the kitten code, because both a|0_L> and a|1_L> have norm sqrt(2) — the coincidence from the Knill–Laflamme slide. Mid-gate that coincidence fails by up to about 20%.
- Point at the figure: two tubes swept along the ideal trajectory. At any time slice you have a code 2-space and the error 2-space it maps onto under a.
- Now we have the objects. The next four slides are the conditions that keep the commuting square closed on these moving subspaces.

## 17. The three conditions

- Give the audience the map before the algebra. Three conditions, and later exactly one scalar metric per condition — that pairing is the cleanest way to hold the rest of the talk in your head.
- One-line intuitions to say out loud:
- (1) is a *static* property checked at a moving location — freeze time, ask whether the code you are sitting on could still correct a photon loss.
- (2) is about consistency of *history* with *present* — an error that already happened and got carried forward must land in the same place as an error happening now.
- (3) is about *rates* — even if the two subspaces coincide, the drive must not rotate the state differently inside them.
- Note that (2) and (3) are genuinely independent: (2) is about which subspace you are in, (3) is about where inside it you are.

## 18. Condition 1: instantaneous correctability

- Point out that this is literally the Knill–Laflamme equation from earlier, with P_C -> P_C(t). Nothing new mathematically; the content is the t.
- The error set is {I, a} — identity plus single photon loss — so there are four ordered pairs and hence four 2x2 blocks.
- Walk the matrix. c_{II}=1: trivial, the code words stay orthonormal under unitary evolution, so that block is exactly the identity and contributes nothing. Off-diagonals vanish: <0_L|a|1_L>=0 because a|2> ∝ |1> which is orthogonal to both code words, and likewise the other way. So the only live entry is c_{aa}.
- c_{aa} = 2 is the number from the kitten-code slide: nbar = 2 for both code words. Call back to it explicitly — the audience saw it twenty minutes ago.
- Do not oversell the t=0 picture. Matched nbar is only one component of one block. Mid-gate the drive mixes parities, so <0_L(t)|a|1_L(t)> and the off-diagonal parts of the (a,a) block need not vanish. Matched nbar is necessary, not sufficient.
- Why it can fail: photon loss is amplitude damping, and its rate scales with nbar. If the drive pushes |0_L> to a higher nbar than |1_L>, they damp at different rates, the environment's record correlates with the logical bit, and you get logical dephasing — irreversible, and not something the standard end-of-gate recovery can undo. App. A makes exactly this point.
- Presentational note to self: the paper's Eq. A7 defines an optional penalty Deltanbar_L(t) that enforces just this. It is available but not part of the default objective — that comes up again on the cost-function overview.

## 19. Condition 2: subspace tracking

- The equation is short; the content is in distinguishing the two objects. Spend the time there and use the tube picture from two slides ago.
- "Evolved" is a statement about history: the jump happened at t=0 and the drive has been pushing the resulting state around ever since.
- "Instantaneous" is a statement about the present: if the jump happened right now, from the current code space, where would it put you?
- Condition 2 says these are the same subspace. If they are not, then whether a recovery works depends on when the jump happened — which is exactly the t-dependence we set out to eliminate.
- Anticipate the obvious objection, because someone will raise it: the equation only ever compares time t against time 0, but jumps happen at all times.
- Walk the block left to right. Eq. 4 at time s rewrites the space the jump lands in as an evolved copy of E_j(0). The two propagators then merge into one, U(T,s)U(s,0) = U(T,0). Eq. 4 at time T turns that back into E_j(T).
- The punchline is that s drops out. Where the state ends up does not depend on when the jump happened, so the final recovery does not need to know it. t=0 is a sufficient anchor because the propagator composes.
- Connect forward: this is the condition that the leakage metric L_(E_j) measures.

## 20. Condition 3: matched internal dynamics

- Conditions 1 and 2 are about *which subspace* you occupy. Condition 3 is about *where inside it* you are. That distinction is the reason all three are needed.
- Even if the error space tracks perfectly, the drive could rotate the state within the error space differently from how it rotates the code state. Then the logical content of the two would drift apart, and the error would no longer commute through the gate.
- P_j is not a projector: it maps one space to another, like the undistorted-copy maps in the Knill–Laflamme sketch. So P_j^dagHP_j is the error-space Hamiltonian pulled back onto the code space, which is what lets the two sides be compared.
- d_j is real: both sides of Eq. 5 are Hermitian, so their difference is too. An imaginary part would change the norm of the state, not just its phase.
- d_j(t)I is the allowed slack, and it is exactly the right amount of slack: a term proportional to the identity generates a global phase, which is invisible in rho. Tie this back to e^{iphi_j(t)} on the commuting-square slide — it is the same freedom, expressed at the level of the generator instead of the state.
- Anything beyond that — any relative rotation — is a real, physical mismatch. That is what the trajectory-mismatch metric eta_(E_j,psi) will measure.
- If asked how this connects to the App. A obstruction: the uncorrectable a^daga^2 commutator is precisely a term that fails Condition 3, because it generates a photon-number-dependent *relative* rotation, not a multiple of the identity.

## 21. Three metrics, one per condition

- The pairing is the thing to sell here. Three conditions, three numbers, one-to-one.
- All three are computed from *lossless* unitary simulation. No quantum jumps are simulated; E_j enters only as an algebraic object. That is worth saying explicitly because it surprises people — these are properties of the ideal evolution, not of a stochastic trajectory.
- Time dependence matters: a pulse can look perfect at t=T and be badly non-transparent in the middle, and the middle is exactly where a random photon loss might strike. This is why the cost function has to sum over the whole trajectory rather than just score the endpoint.
- Flag now, expand in the cost-function section: these are *diagnostics*. The optimiser's actual cost function is a different, differentiable object built from state overlaps.

## 22. Metric 1: QEC violation Delta_QEC(t)

- Structure the explanation as three moves: build the matrix, expand it, throw away the harmless part.
- Build: M_{ik} is Condition 1's left-hand side. Condition 1 wanted every M_{ik} proportional to P_C; this metric measures the failure of exactly that.
- Expand: P_C(t) acts as the identity inside the 2-dimensional logical space, so {P_C, X_C, Y_C, Z_C} is a complete basis and the expansion is unique.
- Strip: Tr[M_{ik}] = 2c_{ik} because the logical Paulis are traceless and TrP_C = 2. So subtracting (1/2)Tr[M_{ik}]P_C removes precisely the correctable piece, leaving only uncorrectable generators.
- Note to self on the norm — this is the one place my own notes are wrong. I wrote ||A||_2 = sqrt(lambda_{max}(A^dag A)), the spectral norm, but then quoted the answer 2(|x|^2+|y|^2+|z|^2), which is the Frobenius result. The main text says "2-norm" but sends you to App. A, and App. A defines ||A||_2 = Tr(A^dag A) with no square root. Frobenius is the right reading, and it is what our own implementation uses.
- The (I,I) pair contributes exactly zero: M_{II} = P_C is pure identity, because unitary evolution keeps the code words orthonormal.
- Not normalised by anything — so absolute values are directly comparable to the paper's.

## 23. Metric 2: instantaneous leakage L_(E_j)(t)

- Unpack why the maximally mixed state is the right initial condition: it is proportional to the projector, so Tr[P_(E_j)(t)rho(t)] is an average over all logical inputs rather than a statement about one particular state. That is what makes this a property of the *subspace* and not of a chosen vector.
- For the kitten code, P_(E_j)(0) = |1><1| + |3><3|, so rho(0) = (1/2)(|1><1|+|3><3|).
- The key identity that makes this a subspace-overlap: for any state, Tr[P|phi><phi|] = <phi|P|phi>. So the trace expands into the mean of <0_E(t)|P_(E_j)(t)|0_E(t)> and the same for |1_E(t)> — the alignment of the two error subspaces.
- Emphasise the evolved-versus-instantaneous framing; it is the same distinction as Condition 2 and this is its scalar.
- Worth stating: no loss is ever applied during the evolution. The error is applied once, at t=0, and then everything is unitary.
- If asked how this differs from eta_(E_j,psi): leakage asks whether you are in the right *subspace*; eta_(E_j,psi) asks, given that you are, whether you are at the right *place inside it*. eta_(E_j,psi) projects the error state into the instantaneous error space first, which makes it deliberately blind to leakage. So the two carry genuinely independent information, and neither bounds the other.

## 24. Metric 3: trajectory mismatch eta_(E_j,psi)(t)

- The picture to describe: two Bloch spheres side by side, one for the code space and one for the error space, each carried along by the drive. Put a state on the code sphere and its error partner on the error sphere. At t=0 they sit at the same point. Transparency says they must stay at the same point.
- Stress "its own instantaneous frame". The logical Pauli operators are themselves being carried along by the drive; we are not comparing against a fixed lab frame. That is what makes this a statement about *relative* rotation.
- Why a global phase does not show up: Bloch vectors come from rho, and rho is insensitive to global phase. So the d_j(t)I freedom of Condition 3 is automatically forgiven by this metric — which is exactly what we want.
- Six cardinal points, ± X, ± Y, ± Z, give six code/error pairs. Testing a single direction would not pin the map: two basis states fix a complex 2-dimensional map only up to relative phases, and all six fix it completely.
- Range is [0,2] — the diameter of the Bloch sphere — so the numbers are easy to read: 0.2 is a modest rotation, 1.0 is severe.
- One interpretive wrinkle if anyone presses: the paper fixes sigma_{E_j} only up to a proportionality constant, because E_jP_C E_j^dag is not a projector. Our implementation resolves it with the polar isometry of aP_C(t); the sensitivity to that choice is about 0.4%.

## 25. What GRAPE does (Gradient Ascent Pulse Engineering)

- GRAPE is Gradient Ascent Pulse Engineering (Khaneja et al., 2005). The name says "ascent" because it was written to maximise a fidelity; minimising a cost is the same thing with the sign flipped.
- It is *open loop and model based*: the pulse is designed entirely on a computer model of the device, then sent to the hardware as is. Nothing is learned from the experiment during the optimisation.
- This opens the second half. The theory told us what a good gate looks like; now we need a way to actually find a pulse that does it.
- The representation first. The control hardware plays one value per nanosecond, so the pulse is a staircase of N steps. On each step there are four real numbers: the I and Q amplitudes of the cavity drive and of the transmon drive (the I/Q slide explains them). Stacked end to end that is 4N numbers — 4000 for a 1 mus gate. To the optimiser, "the pulse" is just this vector u.
- Then walk the loop. Simulate, score, take the gradient, step. The trick that makes it practical: the gradient with respect to *all* the numbers costs only about one extra simulation, not 4000.
- The paper's tools (App. C): Dynamiqs to integrate the Schrödinger equation, JAX for automatic differentiation, and the Adam optimiser from Optax.
- Ours, if asked: JAX autodiff for the gradient, but scipy's L-BFGS-B as the optimiser, plus a second, pure-numpy pipeline with a hand-derived gradient that agrees with it. Same cost, different step rule.
- The next four slides fill in the loop: the model that is simulated, the controls as real numbers, how a staircase pulse becomes a gate and a gradient, and which states go in. Then the cost.

## 26. The model the optimiser simulates

- "Characterising the system for optimisation" means: decide exactly what the computer simulates. Three decisions — the frame, the truncation, and what to leave out.
- The frame. App. C says both drives are applied on resonance. In the frame rotating at omega_a and omega_q the two big frequencies vanish. Every other term commutes with both photon-number operators, so it survives unchanged. That leftover is the drift H_0: always on, not under our control.
- The truncation is *ours*; the paper does not state one. Twenty cavity levels is generous — the code words stop at |4> and the trained pulses reach about Fock 9 or 10. We checked it on held-out truncations from 16 to 28: the fidelity moves by less than 10^{-5}.
- Three transmon levels, not two, because the anharmonicity K_q is finite: a strong drive on g -> e can also push population to f, and a two-level model would hide that.
- The K' term (K'/6)a^dag{}^3a^3 is exactly as Table I writes it. It is a 590 Hz term and barely matters.
- The key point to repeat: the optimisation is lossless. No jump is ever simulated. The annihilation operator appears only as an algebraic object inside C_2, just as it did in the metrics. Decoherence is assessed afterwards, not trained against.

## 27. Rewriting the drive as real controls: I and Q

- The Hamiltonian on the device slide has complex drives epsilon(t) and Omega(t). An optimiser wants a plain vector of real numbers, so split each into a real part I (in-phase) and an imaginary part Q (quadrature).
- Check the identity in one line: (epsilon_I + iepsilon_Q)a + (epsilon_I - iepsilon_Q)a^dag = epsilon_I(a+a^dag) + iepsilon_Q(a-a^dag). Both operators on the right are Hermitian — they are, up to sqrt(2), the position-like and momentum-like quadratures of the mode.
- What this buys. One: four real control functions instead of two complex ones. Two: four *fixed* Hermitian operators H_j, with the pulse entering only as real coefficients, so H is linear in the controls. That makes the derivative with respect to each control trivial: d H/d u_j = H_j.
- Three: it is how the hardware works. App. B: the pulses come out of an AWG and go through IQ mixers. The two numbers per drive per nanosecond are literally the two voltages the AWG plays.
- So a pulse is an N x 4 real array: 4000 numbers for X and H. This is the same column order as our saved files in `pulses/est/`.
- Point at the picture: the drive is a vector in the I–Q plane. The amplitude limit is a *circle* in that plane. Hold that thought for the constraints slide.
- If asked about the sign of Q: it is a convention. Our code writes the drive as epsilon a^dag + h.c. instead of epsilon a + h.c., so there epsilon = u_0 - i u_1. Same operators, same pulse array; only the sign of Q flips, and the amplitude cap depends only on |epsilon|.

## 28. Piecewise-constant pulses and the gradient

- This is the mechanics inside the GRAPE loop's "simulate" and "gradient" boxes: how a pulse turns into a gate, and why its gradient is cheap.
- Within one nanosecond the controls are constant, so the Hamiltonian is constant and the evolution over that step is a single matrix exponential. The whole gate is the product of N of them, latest on the left.
- The states the cost needs — the six code states, their six error partners, at every step — are all obtained by applying these U_k one after another.
- The gradient. The cost depends on u_j(k) only through U_k, so the chain rule needs the states just before step k and a "backward" state coming from the cost side. One forward sweep and one backward sweep give the derivative for all 4000 numbers. In cost, that is about two to three simulations, not 4000.
- The paper lets JAX do this automatically. So do we, with one care point: we take the exponential with `expm`, not by diagonalising H_k, because differentiating an eigendecomposition blows up when eigenvalues are degenerate.
- Then the optimiser takes the step: Adam in the paper, L-BFGS-B in our code.

## 29. What goes in, and what should come out

- The optimiser does not work with the full 60x60 propagator. It evolves a handful of states and compares them with targets. App. C says so explicitly: simulating states is cheaper than simulating the propagator.
- Why six and not just |0_L> and |1_L>: two basis states fix where each one goes but not the relative phase between them. The six cardinal points pin that phase down; together they are what is called a 2-design. Only the physically meaningless global phase is left free.
- The error partners are the same six states pushed through a and normalised. Because a maps the kitten code onto the error space without distortion (the equal-sqrt(2) fact), the partner of |+X_L> really is the error-space |+X>. They are used by C_2.
- Gate lengths are the paper's: X and H use both drives for 1 mus; T is a phase gate and uses the transmon only, for 0.6 mus.
- One nanosecond is the time resolution of the control hardware, so the simulation uses the same grid the AWG will play.
- Fairness note from the main text: EsT and Ord pulses are compared at the same maximum active Fock level, the highest n that reaches population above 0.01.

## 30. The cost function: one weighted sum

- One number for the optimiser to push down: a weighted sum of separate costs. The weights are tuned by hand (App. C says "manually tuned").
- Read down the table. C_1 is the only one that looks at the end of the pulse. C_2 and C_3 are *running* costs: they look at the whole trajectory. That is the structural difference from ordinary gate optimisation — a loss can strike at any time, so the middle of the pulse has to be scored too.
- C_4 is ours. The paper states a 4 MHz amplitude limit but not how it was imposed. We impose it as a penalty; the constraints slide says why.
- The block is the caveat that used to come later: the three metrics we just defined are *not* what is optimised. The paper's own words (p. 4): "to facilitate differentiability, we utilize a cost function based on state overlaps and fidelities rather than commutator norms and projectors." The metrics are then used to check the result.
- App. C lists optional extra costs: (a) the mean-photon-number difference between the evolving code words (Eq. A7, aimed at Condition 1), (b) smoothness, (c) robustness. None is in the default recipe.
- The last sentence of the block heads off a natural misreading: three conditions, three metrics, three costs does *not* mean one cost per condition. C_2 is one overlap that pushes on tracking and matched dynamics together; Condition 1 is only touched indirectly; C_1 is the gate and C_3 is scaffolding.
- If asked whether we tried (a): yes. It more than halves Delta_QEC, but makes end-of-pulse leakage and eta_(E_j,psi) worse. The three conditions trade against each other.

## 31. C_1: did the gate happen?

- The standard gate cost. Each term is the overlap between where a cardinal state actually ends up and where it should end up; |.|^2 makes it a probability. Average over six, subtract from one.
- Because all six cardinal states are in the average, a pulse cannot score well by getting |0_L> and |1_L> right with the wrong relative phase.
- This is the only cost an Ord gate uses. It says nothing about the middle of the pulse, so an Ord pulse is free to wander anywhere mid-gate — exactly the behaviour the "why a gate breaks all of that" slide warned about.
- The trade in one number, from our own X pulses: Ord reaches F_{target} ≈ 0.99999; EsT stops at about 0.9993, because part of its budget goes into transparency.

## 32. C_2: transparency at every step

- This is the cost that makes a gate error-transparent, and it is the commuting square written as one overlap per time step.
- Two routes, as on the commuting-square slide. Route 1: the photon is lost at t=0 and the error state is carried forward by the drive — that is psi_{E_j}(t), evolved on its own. Route 2: no loss until now, then the loss — that is apsi_C(t). Transparency says the two routes give the same state.
- N is there because a is not unitary: a|psi_C> has length sqrt(nbar), not 1, so the overlap has to be normalised.
- The paper's claim: this single overlap "partially" enforces all three conditions. It most directly enforces tracking and matched dynamics (2 and 3): a mismatch in either subspace or position inside it lowers the overlap. Condition 1, and so Delta_QEC, is only weakly targeted — App. A says as much, and it is why Delta_QEC improves least.
- The footnote, our deviation: the paper prints a single power of N. With that, F_{ET} is not bounded by 1 — it can reach about sqrt(nbar) — so the cost rewards simply adding photons. Dividing by N^2 makes it a true fidelity between 0 and 1. We kept the printed form as an option and tested it: it is not more transparent.
- The sum stops at N-1, so the final time is not included; the endpoint is covered by C_1.

## 33. C_3: keep the dynamics spread out

- The loophole is real and the optimiser finds it. C_2 is an average over every step. If the gate happens in a short burst and the state sits still for the rest of the pulse, most steps are trivially transparent and score well, while the few steps that do the gate — where a photon loss would actually hurt — barely count. App. C: the regulariser ensures "the optimizer does not simply minimize the pulse dynamics duration to artificially inflate F_{ET}".
- Why "at the end": a burst early on would leave the code and error states mismatched for the rest of the pulse, and every later step would score badly. Only a late burst keeps the mismatch confined to a few steps. "Nearly" because Kerr still acts on an idle state.
- Consistent with this, not proof of it: in our T-gate runs, the stage with C_3 switched off put about a third of the transmon drive energy into the last 48 ns.
- The paper's fix: look at how fast the code state moves at each step, and penalise the variance of that speed over the pulse. The distance used is the Fubini–Study distance, the natural distance between two quantum states.
- Why a variance is the right cure: a burst followed by stillness has a very uneven speed — fast for a moment, then zero — so its variance is large. A gate spread over the whole duration moves at a steady pace and pays little.
- The paper is explicit (App. C) that this term "does not improve ET performance directly". It is scaffolding. It also makes EsT and Ord gates use their time in a comparable way, so incoherent errors affect both alike.
- The footnote, our deviation. The printed C_3 has units of frequency squared, so its size depends on the time unit: about 965 in rad per microsecond, about 10^{-3} in rad per nanosecond, for the same pulse. It is also smallest when the drives are switched off — it rewards slowing the whole pulse down, which is not what it is for. Dividing by the mean speed squared removes the units and the bias towards slowing down.
- Honest statement if pressed: C_3 does not stop a pulse that barely drives at all. C_1 does, because such a pulse does not do the gate.

## 34. Putting it together: two stages

- Why two stages: asking for high fidelity and high transparency at once from a random start tends to get neither. So the paper first builds transparency with a strong C_2 and C_3, then turns C_3 off and C_2 down to finish the gate.
- Stage 1 numbers from App. C: transparency fidelity up to 0.83, gate fidelity only 0.9–0.99 — not yet a usable gate. Stage 2 lifts the gate fidelity while the transparency moves by about a percent.
- Our weights, if asked: (w_1,w_2,w_3,w_4) = (1, 0.7, 7, 1) then (1, 0.1, 0, 1), inside the paper's ranges, with C_4 on in both stages.
- The block is the experimental design. Ord runs the same code with only C_1. That is the comparison every EsT result is quoted against, so it matters that the two differ in the cost and nothing else.
- The caveat in the block: each run starts from a random pulse and lands in a local optimum, so one EsT–Ord pair could differ by luck. Our comparisons are mostly single-seed so far. Where we checked two seeds, the seed-to-seed change in end-of-gate leakage was about 0.02, far below the EsT–Ord gap in mean leakage (about 0.26 vs 0.74 on X). Say that, and say it is two seeds.
- We also found that a single stage — stage-1 weights all the way — does not finish the X gate: holding the velocity term on to the end dominates the cost. That is good evidence that the second stage is doing real work.

## 35. Hardware constraints on the pulse

- Keep this brief. These are the rules that make a pulse playable on real hardware and keep the simulation honest.
- Bandwidth: the AWG cannot produce frequencies above about 50 MHz, so the paper Fourier transforms the pulse and removes everything outside that band. App. C: it keeps the pulses reliable and the simulation close to the experiment.
- Amplitude: at most 4 MHz on each drive, the same for cavity and transmon.
- Ramp: a 48 ns Gaussian rise and fall, so the pulse switches on from zero and off to zero smoothly, instead of jumping.
- The 1 ns step and the transmon-only T gate we have already seen.
- How, in our code. The optimiser works on a raw vector x; we band-limit it, then multiply by the ramp envelope, then zero any drive that is switched off, and only the result u is physical. The mask and the ramp hold exactly. The band limit holds only to about 0.3% of the energy, because the ramp is applied after it. That order was chosen by measurement: the other order is exactly band-limited but smears the ramp so the ends reach a quarter of full scale.
- The amplitude cap cannot be done that way. The optimiser's bounds are a box, one interval per number, but |epsilon| <= epsilon_{max} is a circle in the I–Q plane. So it is a penalty, C_4, that switches on only above the cap. Being soft, it lets the pulse sit about 0.2% over the cap on under 1% of the steps.

## 36. Our X gate vs the paper: the three metrics in time

- Same three metrics, same gate, same duration, same linear axes. Top row is the paper's Fig. 1d–f, cut from the PDF; bottom row is our own Figure 2a from `est\_experiments.ipynb`, recomputed from our trained pulses.
- Read it column by column. Left, QEC violation: both start and end near zero, because both pulses bring the code words back into the code at t=T; Ord peaks mid-pulse in both, EsT stays lower through the middle.
- Middle, leakage: Ord climbs towards 0.8–1 in both. The EsT curves are at a similar level mid-pulse. The visible difference is the very end: the paper's EsT drops to zero at t=T, ours stays near 0.4.
- Right, trajectory mismatch for |0_L>: Ord reaches the maximum of 2 in both, EsT stays well below. At the very end the paper's Ord falls back to about 0.2; ours stays near 1.5.
- Our legend says "Bloch-vector mismatch" and the paper says "trajectory mismatch"; it is the same Eq. 8.
- The next slide puts numbers on this.

## 37. What agrees, and what does not

- Where the numbers come from. Paper leakage: digitised from Fig. 1e in the notebook (400 and 600 dpi agree to 0.002). Paper endpoint values and peaks: read off the figure by eye, good to about 0.05–0.1. Ours: computed from the trained pulses with the eigh propagator.
- The good news first. Time-averaged leakage is 0.258 against their 0.22 for EsT, and the Ord-to-EsT ratio is 2.87 against 2.74. Same metric, same gate, same ballpark: the replication is in the right place on average.
- Peaks: our Ord QEC violation tops out at 1.5 against their 1.9; EsT is about the same height, 0.68 vs 0.6, but ours peaks late, at 0.78 mus, where theirs peaks at 0.40 mus.
- The real difference is the endpoint, in bold. Their EsT gate finishes with leakage 0.013, essentially transparent. Ours finishes at 0.396. Their Ord pulse also returns to a small trajectory mismatch at t=T (about 0.2); ours ends at 1.55.
- Why the endpoint matters: QEC runs at t=T. A gate that is transparent on average but leaks at the end hands the recovery an error state in the wrong place.

## 38. Why the endpoint differs, and caveats

- We tested the obvious explanation first. Our C_2 and C_3 differ from the printed equations, so we retrained X with the printed forms at the same weights, at two seeds. L(T) came out 0.40 and 0.39 against 0.38 and 0.40 with ours: no change (notebook Appendix A).
- Then the constructive test: add a cost on the error-space fidelity at the final time. On a comparable run (seed 6, baseline L(T) = 0.398) it brings L(T) to 0.005, with gate fidelity 0.9995 and the time-averaged transparency unchanged. One seed (`est\_optimization.ipynb`, section 1).
- So the leak is a property of the objective, not of the device. The paper's text says the EsT gate optimises fidelity in both code and error space, but its printed cost (Eqs. C1–C8) has no error-space fidelity term. Our guess is that their code has one. Present this as a hypothesis we cannot check without their code.
- Caveats, plainly. Each comparison is one EsT and one Ord pulse. The paper compares pulses matched in maximum active Fock level; ours are 9 and 8, so our ratios are provisional. And their numbers come from digitising or reading a figure.

## 39. What this approach does not cover

- Say these before anyone asks; the first one is the question a cQED PI asks first.
- Transmon errors: the whole framework is about the cavity jump a. A transmon T_1 event or dephasing mid-gate is not transparent. The numbers: a 1 mus gate against T_2 (Ramsey) of 30 mus and T_1 of 70 mus, versus cavity loss probability nbar T/T_1 ≈ 2/180 to 4/180. Same order. The paper notes its approach is compatible with detectable-ancilla schemes; that is future work.
- Single loss: the code corrects one photon loss. Two in one gate is second order and outside the framework.
- Lossless optimisation: decoherence enters only when a pulse is scored afterwards. The optimiser has no reason to keep the transmon in |g> when it does not need it.
- Model: our simulation drives the ideal Hamiltonian. The real pulse passes through mixers and lines that distort it. And chi a^daga q^dagq makes the |f> level shift by exactly 2chi per photon, which a real transmon need not do.
- Last: the floor in the metrics is expected, not a bug.

## 40. Summary

- Walk the chain once, fast — one sentence per item. The audience should hear it as a single argument: what is wrong, why it cannot be fully fixed, how to measure it, and how to optimise against it.
- Items 1–4 are the theory; items 5–7 are the optimisation. The bridge between them: the metrics measure transparency, but the optimiser uses a cheaper overlap, C_2, that pushes in the same direction.
- Land on the trade. The authors are not claiming exact transparency; they are claiming a favourable exchange rate.
- The number that makes the trade look good, as a teaser for the results talk: a five-fold reduction in logical infidelity conditioned on a photon loss during the gate, and a seventeen-fold reduction in per-gate error-space infidelity for X.
- Results are a separate talk: the metrics on real pulses, EsT versus Ord versus LE, and the experimental Wigner tomography.
- Invite questions on the two deviations we made from the printed equations (C_2 and C_3 normalisation) — they are the most likely thing a careful reader asks.

## 41. Our replication: status and open questions

- Keep this to what we have shown. Everything here is simulation; we have not run hardware.
- Two gradient codes: a JAX autodiff path and a numpy path with a hand-derived gradient. They are checked against each other and against finite differences.
- The two cost changes are the N^2 in C_2 and the unit-free C_3, both explained on their slides.
- The 3–7 times is time-averaged leakage and Bloch mismatch, EsT against Ord. X: leakage 2.9 times, mismatch 6.6 times on |0_L>. H: 4.5 and 5.2 times. Single seed for these headline pairs.
- Extra costs: the mean-photon-number penalty halves Delta_QEC but hurts leakage; an end-of-gate error-space fidelity term cuts L(T) but can stall some seeds; a smoothness term raised the mismatch. The metrics trade against each other.
- Open: the end-of-gate leakage of about 0.4, against the paper's 0.013. The results slides show a terminal error-space term removes it; whether that is what the paper did is the question. And we need more seeds before quoting any EsT–Ord ratio as a result.

## B1. Why that condition is enough (1/2): untangle the errors

- Use the non-uniqueness from two slides ago. That freedom is the whole trick.
- Step 1: alpha is Hermitian (in fact a Gram matrix, so positive semidefinite), so it has a unitary eigenbasis and real, non-negative eigenvalues d_k.
- Step 2: rotate the Kraus set by that same unitary. Same channel, new labels.
- Step 3: substitute. P F_k^dagF_l P = sum_{a,b} conj u_{ak} alpha_{ab} u_{bl} P, and that sum is just (u^dagalpha u)_{kl} = d_kdelta_{kl}. Only the diagonal is left, hence the delta_{kl}.
- What the delta_{kl} buys: the original E_a could overlap on the code, so you could not cleanly tell them apart. The new F_k cannot.
- What the d_k means: ||F_k|psi>||^2 = d_k for every code state, so learning "k happened" reveals nothing about the logical state.
- If some d_k = 0, that F_k annihilates the code: it never happens on code states, so drop it.

## B2. Why that condition is enough (2/2): undistorted copies

- Both identities are the previous slide's equation divided by d_k: set k=l for the first, k != l for the second.
- Partial isometry: V_k preserves every inner product inside C and kills everything outside it. So on the code, F_k is sqrt(d_k) times a rigid move. The image C_k is an undistorted copy of C, and different k give orthogonal copies.
- Set up the next slide: that geometry *is* what makes recovery possible. Orthogonal copies can be told apart by a measurement; undistorted copies can be rotated back.
- Closing remark on linearity: if R corrects {E_a} it corrects any F_r = sum_a M_{ra}E_a, because R_kF_rP ∝ P still. That is why we may restrict attention to a discrete error set.

## B3. Recovery: find the copy, rotate it back

- The previous slide built the geometry. This one runs the protocol on it.
- Starting point: F_lP = sqrt(d_l) V_l, so the noisy state is a mixture. With probability d_l the code was moved rigidly onto copy l; we do not yet know which.
- Step 1: P_k is a projector because V_k^dagV_k = P and V_kP = V_k, so P_k^2 = V_kPV_k^dag = P_k. Different copies are orthogonal because V_k^dagV_l = 0. Add I - sum_k P_k to complete the measurement; that outcome never fires.
- Step 2: on a single branch, P_kF_l|psi> = delta_{kl}F_l|psi>. So the outcome is certain and the state is not disturbed — it was already an eigenstate of the projector.
- The key point: p_k = d_k does not depend on rho. The measurement asks only a question whose answer is the same for every code state, so it learns the error and nothing about the logical information. That is the "alpha_{ab} independent of i" half of Knill–Laflamme, seen as a property of the measurement.
- Step 3: V_k^dag is not unitary on the whole space. In the lab you apply any unitary that agrees with V_k^dag on C_k; one always exists because C_k and C have the same dimension.
- Order matters: measure first, then pick V_k^dag (feedback). Averaged over outcomes, sum_k V_k^dagvarepsilon(rho)V_k = sum_k d_k rho = rho, since completeness of the Kraus set gives sum_k d_k = 1.
- Kitten preview: the copies are the even and odd parity sectors, so the syndrome is a parity measurement.
- Key sentence: the parity check we use in the lab is not an extra ingredient — it is the projective measurement that Knill–Laflamme forces into existence.
