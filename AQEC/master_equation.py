"""
Lindblad master equation for PReSPA -- Shirol, van Geldern, Xi & Wang, PRX 16, 021042
(2026) -- in the two rotating frames of aqec_experiments.ipynb §8.2.

    d rho/dt = -i[H, rho] + sum_k D[L_k] rho

Frame (i), rotating with the full H0 of Eq. (2): H = Eq. (3) (+ the D1 shifts), time-
independent. Exact for photon-number populations (Fig. 2a/c), not for phases between
photon numbers. Solved with the Liouvillian below.

Frame (ii), rotating with the uncoupled modes only: H(t) = H_disp + sum (e^{iwt} A + h.c.)
with the dispersive terms explicit, period 2 pi / chi_gf. Needed for coherences between
photon numbers (Fig. 2d): a transmon decay at a random time leaves a photon-number-
dependent phase that frame (i) hides inside its frame transformation. Solved by
integrating the density-matrix ODE (solve_ivp, DOP853).

Jump operators (collapse_operators): decay, thermal + drive-induced heating and pure
dephasing, all on by default. The dephasing operators are diagonal, so frame (i) is exact for
them; the heating jumps, like the decay jumps, lose a photon-number phase in frame (i).

Space: storage (Fock 0..n_c-1) x transmon (g, e, f) x readout (m = 0, 1), storage
outermost, so |n, j, m> sits at index (n*3 + j)*2 + m. Eq. (3) never creates a second
readout photon, so m <= 1 is exact for it.

The Liouvillian is column-stacking: vec(A rho B) = (B^T kron A) vec(rho). It is
non-normal, so propagation uses expm(L dt) on a uniform grid rather than an
eigendecomposition.
"""

import numpy as np
import scipy.linalg as sla
from scipy.integrate import solve_ivp

from AQEC import device as dev
from AQEC.odd_kitten import MIN_N_C, annihilation

N_Q = 3                      # transmon levels g, e, f
N_R = 2                      # readout photon number 0, 1
G, E, F = 0, 1, 2
FIG2_INITIAL = (0, 2, 4)     # even Fock states the paper starts from (Fig. 2a)

# Apple's Accelerate BLAS raises spurious divide/overflow RuntimeWarnings on ordinary
# complex matmuls (see core/grape_core.py's np.seterr); scoped here, not global.
_QUIET = dict(divide="ignore", over="ignore", invalid="ignore")


def dim(n_c=MIN_N_C):
    return n_c * N_Q * N_R


def index(n, j, m, n_c=MIN_N_C):
    """Flat index of |n, j, m>."""
    if not (0 <= n < n_c and 0 <= j < N_Q and 0 <= m < N_R):
        raise IndexError(f"|{n},{j},{m}> outside n_c={n_c}")
    return (n * N_Q + j) * N_R + m


def ket(n, j, m, n_c=MIN_N_C):
    v = np.zeros(dim(n_c), dtype=complex)
    v[index(n, j, m, n_c)] = 1.0
    return v


def _embed(op_a, op_q, op_r):
    return np.kron(np.kron(op_a, op_q), op_r)


def build_operators(n_c=MIN_N_C):
    """Storage a, transmon ladder q (<e|q|f> = sqrt2), readout r, transmon projectors
    Pg/Pe/Pf, and the single-transition lowering operators s_ge = |g><e|, s_ef = |e><f|."""
    Ia, Iq, Ir = np.eye(n_c), np.eye(N_Q), np.eye(N_R)
    q = np.diag(np.sqrt(np.arange(1, N_Q)), 1).astype(complex)
    r = np.diag(np.sqrt(np.arange(1, N_R)), 1).astype(complex)

    def proj(j):
        P = np.zeros((N_Q, N_Q), dtype=complex)
        P[j, j] = 1.0
        return P

    s_ge = np.zeros((N_Q, N_Q), dtype=complex)
    s_ge[G, E] = 1.0
    s_ef = np.zeros((N_Q, N_Q), dtype=complex)
    s_ef[E, F] = 1.0
    a = _embed(annihilation(n_c), Iq, Ir)
    with np.errstate(**_QUIET):
        n_a = a.conj().T @ a             # same construction as DGRAPE.model (compared exactly)
    return {
        "a": a,
        "q": _embed(Ia, q, Ir),
        "r": _embed(Ia, Iq, r),
        "Pg": _embed(Ia, proj(G), Ir),
        "Pe": _embed(Ia, proj(E), Ir),
        "Pf": _embed(Ia, proj(F), Ir),
        "s_ge": _embed(Ia, s_ge, Ir),
        "s_ef": _embed(Ia, s_ef, Ir),
        "n_a": n_a,
    }


def drive_hamiltonian(n_c=MIN_N_C, omega_1=dev.OMEGA_1, omega_2=dev.OMEGA_2,
                      paths=dev.CORRECTED_FOCK):
    """PRX Eq. (3): sum_n Omega_1 |n,f,0><n-1,g,0| + Omega_2 |n,g,1><n,f,0| + h.c."""
    H = np.zeros((dim(n_c), dim(n_c)), dtype=complex)
    for n in paths:
        H[index(n, F, 0, n_c), index(n - 1, G, 0, n_c)] += omega_1
        H[index(n, G, 1, n_c), index(n, F, 0, n_c)] += omega_2
    return H + H.conj().T


def higher_order_shifts(n_c=MIN_N_C, K=dev.K_a, chi_q_p=dev.chi_q_p):
    """Decision D1: -(K/2) a'^2 a^2 + (chi'_q/2) q'q a'^2 a^2, diagonal.

    The tones stay set by the Eq. (2) H0, so in its rotating frame these terms remain
    as static, path-dependent detunings; the drive stays time-independent.
    chi'_q is assumed to scale with q'q (|f> gets twice the |e> value), as in §3.2.
    """
    diag = np.zeros(dim(n_c))
    for n in range(n_c):
        for j in range(N_Q):
            for m in range(N_R):
                diag[index(n, j, m, n_c)] = (-0.5 * K + 0.5 * chi_q_p * j) * n * (n - 1)
    return np.diag(diag).astype(complex)


def jump_rates(kappa_a=dev.kappa_a, drive_heating=True):
    """Every rate the jump operators use, in 1/us (notebook Table 5).

    gamma_ge, gamma_ef -- transmon decay, 1/T1_GE and 1/T1_EF (Table I)
    r_thermal          -- up/down ratio at equilibrium, P_e/(1 - P_e) with P_e = 1.7% (Table I)
    r                  -- up/down ratio used: r_thermal, plus DELTA_GAMMA_UP_DRIVE * T1_GE when
                          drive_heating (App. D: 0.42 ms^-1 extra g->e under the pumps)
    up_ge, up_ef       -- transmon heating g->e and e->f, r * gamma_ge and r * gamma_ef. App. D
                          uses one ratio r for both transitions
    up_a               -- storage heating, detailed balance at P_1 = 0.6% against the kappa_a
                          passed in (kappa_a = 0 switches storage heating off too)
    phi_ge, phi_gf     -- transmon pure dephasing from Ramsey: 1/T2R - gamma_ge/2, 1/T2gf - gamma_ef/2
    phi_a              -- storage pure dephasing NOT already produced by thermal transmon heating:
                          1/T2a - kappa_a/2 - r_thermal * gamma_ge (measured kappa_a, not the argument).
                          T2a was measured at equilibrium, where each thermal g->e jump shifts the
                          storage by chi_ge for ~T1_GE and so dephases it at ~the jump rate. Frame (ii)
                          simulates that explicitly, so it is subtracted (0.58 -> 0.23 ms^-1); the full
                          drives-off model then reproduces T2a to ~2% (tested)
    Thermal heating and the transmon dephasing rates equal DGRAPE.model.rates(); DGRAPE's phi_a is
    the unsubtracted 1/T2a - kappa_a/2 (tested).
    """
    g_ge, g_ef = 1.0 / dev.T1_GE, 1.0 / dev.T1_EF
    r_th = dev.P_E_THERMAL / (1.0 - dev.P_E_THERMAL)
    r = r_th + (dev.DELTA_GAMMA_UP_DRIVE * dev.T1_GE if drive_heating else 0.0)
    return {
        "gamma_ge": g_ge, "gamma_ef": g_ef, "r_thermal": r_th, "r": r,
        "up_ge": r * g_ge, "up_ef": r * g_ef,
        "up_a": kappa_a * dev.P_CAVITY_THERMAL / (1.0 - dev.P_CAVITY_THERMAL),
        "phi_ge": 1.0 / dev.T2_RAMSEY - 0.5 * g_ge,
        "phi_gf": 1.0 / dev.T2_GF_RAMSEY - 0.5 * g_ef,
        "phi_a": 1.0 / dev.T2_CAVITY - 0.5 * dev.kappa_a - r_th * g_ge,
    }


def collapse_operators(n_c=MIN_N_C, transmon_decay="split", gamma=None,
                       kappa_a=dev.kappa_a, kappa_r=dev.kappa_r, heating="driven",
                       dephasing=True):
    """Jump operators of the notebook's Table 5. Default: every row (decay, heating, dephasing).

    transmon_decay:
      "split"    -- sqrt(1/T1_GE)|g><e| and sqrt(1/T1_EF)|e><f| (notebook §8.1; both
                    measured lifetimes reproduced)
      "single_q" -- sqrt(gamma) q, the paper's literal D[q]. gamma is the e->g rate and,
                    since <e|q|f> = sqrt2, f->e runs at 2 gamma. Default 1/T1_GE gives
                    T1_ef = 25 us instead of the measured 31 us (§8.1)
      "none"     -- no transmon decay
    heating (independent of transmon_decay; rates from jump_rates):
      "driven"   -- sqrt(up_ge)|e><g|, sqrt(up_ef)|f><e| at thermal + drive-induced ratio r,
                    and sqrt(up_a) a'
      "thermal"  -- the same at the equilibrium ratio only
      "none"     -- no heating
    dephasing:  sqrt(2 phi_ge)|e><e|, sqrt(2 phi_gf)|f><f|, sqrt(2 phi_a) n_a. D[sqrt(G)|j><j|]
                damps the (g, j) coherence at G/2, so G = 2 phi gives phi exactly.
    heating="none", dephasing=False is the decay-only set used before these were added.
    """
    if heating not in ("driven", "thermal", "none"):
        raise ValueError(f"unknown heating={heating!r}")
    ops = build_operators(n_c)
    c_ops = [np.sqrt(kappa_a) * ops["a"], np.sqrt(kappa_r) * ops["r"]]
    if transmon_decay == "split":
        c_ops += [np.sqrt(1.0 / dev.T1_GE) * ops["s_ge"],
                  np.sqrt(1.0 / dev.T1_EF) * ops["s_ef"]]
    elif transmon_decay == "single_q":
        g = 1.0 / dev.T1_GE if gamma is None else gamma
        c_ops.append(np.sqrt(g) * ops["q"])
    elif transmon_decay != "none":
        raise ValueError(f"unknown transmon_decay={transmon_decay!r}")
    R = jump_rates(kappa_a, drive_heating=(heating == "driven"))
    if heating != "none":
        c_ops += [np.sqrt(R["up_ge"]) * ops["s_ge"].conj().T,
                  np.sqrt(R["up_ef"]) * ops["s_ef"].conj().T,
                  np.sqrt(R["up_a"]) * ops["a"].conj().T]
    if dephasing:
        c_ops += [np.sqrt(2 * R["phi_ge"]) * ops["Pe"],
                  np.sqrt(2 * R["phi_gf"]) * ops["Pf"],
                  np.sqrt(2 * R["phi_a"]) * ops["n_a"]]
    return [c for c in c_ops if np.any(c)]


def liouvillian(H, c_ops):
    """Column-stacking superoperator: d vec(rho)/dt = L vec(rho)."""
    d = H.shape[0]
    Id = np.eye(d)
    with np.errstate(**_QUIET):
        L = -1j * (np.kron(Id, H) - np.kron(H.T, Id))
        for c in c_ops:
            cdc = c.conj().T @ c
            L += np.kron(c.conj(), c) - 0.5 * np.kron(Id, cdc) - 0.5 * np.kron(cdc.T, Id)
    return L


def propagate(L, rho0, t_grid):
    """rho(t) on a uniform grid starting at t_grid[0]; returns (len(t_grid), d, d)."""
    t_grid = np.asarray(t_grid, dtype=float)
    dts = np.diff(t_grid)
    if dts.size and not np.allclose(dts, dts[0], rtol=1e-9, atol=1e-12):
        raise ValueError("propagate needs a uniform time grid")
    d = rho0.shape[0]
    out = np.empty((t_grid.size, d, d), dtype=complex)
    v = rho0.reshape(-1, order="F").astype(complex)
    out[0] = rho0
    if dts.size:
        with np.errstate(**_QUIET):
            P = sla.expm(L * dts[0])
            for k in range(1, t_grid.size):
                v = P @ v
                out[k] = v.reshape(d, d, order="F")
    return out


def fig2_observables(rhos, n_target, n_c=MIN_N_C):
    """Populations traced over the readout.

    Png, Pne -- P_{n,g}, P_{n,e} at n = n_target; Fig. 2(a) plots Png - Pne, the signal
                of the number-selective g->e pi pulse (App. A.6).
    Pe, Pf   -- total transmon |e>, |f> population (Fig. 2c).
    """
    p = np.real(np.einsum("tii->ti", rhos)).reshape(len(rhos), n_c, N_Q, N_R).sum(axis=3)
    return {
        "Png": p[:, n_target, G],
        "Pne": p[:, n_target, E],
        "Pe": p[:, :, E].sum(axis=1),
        "Pf": p[:, :, F].sum(axis=1),
    }


def simulate_fig2(t_max=25.0, dt=0.05, include_higher_order=False, transmon_decay="split",
                  n_c=MIN_N_C, omega_1=dev.OMEGA_1, omega_2=dev.OMEGA_2, **collapse_kw):
    """Fig. 2(a)/(c): start in |n0, g, 0> for n0 = 0, 2, 4 and drive both combs.

    Returns {n0: dict(t, Png, Pne, Pe, Pf)}, with Png/Pne evaluated at n = n0 + 1.
    """
    H = drive_hamiltonian(n_c, omega_1, omega_2)
    if include_higher_order:
        H = H + higher_order_shifts(n_c)
    L = liouvillian(H, collapse_operators(n_c, transmon_decay, **collapse_kw))
    t = np.arange(0.0, t_max + 0.5 * dt, dt)
    out = {}
    for n0 in FIG2_INITIAL:
        psi = ket(n0, G, 0, n_c)
        rhos = propagate(L, np.outer(psi, psi.conj()), t)
        out[n0] = {"t": t, **fig2_observables(rhos, n0 + 1, n_c)}
    return out


# ---------------------------------------------------------------------------
# Frame (ii): dispersive terms explicit, drives time-dependent (notebook §8.2)
# ---------------------------------------------------------------------------

def dispersive_hamiltonian(n_c=MIN_N_C, chi_ge=dev.chi_ge, chi_gf=dev.chi_gf, chi_qr=dev.chi_qr):
    """Static part of H^(ii): -chi_ge|e><e| n - chi_gf|f><f| n - chi_qr|e><e| m (diagonal)."""
    diag = np.zeros(dim(n_c))
    chi_j = (0.0, chi_ge, chi_gf)
    for n in range(n_c):
        for j in range(N_Q):
            for m in range(N_R):
                diag[index(n, j, m, n_c)] = -n * chi_j[j] - m * chi_qr * (j == E)
    return np.diag(diag).astype(complex)


def frame_ii_drive_terms(n_c=MIN_N_C, omega_1=dev.OMEGA_1, omega_2=dev.OMEGA_2,
                         chi_gf=dev.chi_gf, six_tones=False):
    """[(A, w)] with H_drive(t) = sum_k e^{i w_k t} A_k + h.c. in frame (ii).

    Eq. (3) (default): only the resonant pairing, tone n on path n,
        Omega_1 |n,f,0><n-1,g,0| at w = +n chi_gf,  Omega_2 |n,g,1><n,f,0| at w = -n chi_gf.
    six_tones=True: every tone acts on every transition of its FWM process,
        comb 1, tone n:  (Omega_1/sqrt n) a' (x) |f><g| (x) 1   at w = +n chi_gf
        comb 2, tone n:   Omega_2 1 (x) |g><f| (x) r'           at w = -n chi_gf
    The frame removes only the linear frequencies, so a tone's phase does not depend on
    the transition it drives; which pairing is resonant is decided by H_disp. The Eq. (3)
    terms are the resonant matrix elements of these (the 1/sqrt n cancels <n|a'|n-1>).
    Off-resonant pairings include comb 1 acting on the odd code states, i.e. the
    'unwanted corrections' of App. E.2.
    """
    terms = []
    if not six_tones:
        for n in dev.CORRECTED_FOCK:
            A1 = np.zeros((dim(n_c),) * 2, dtype=complex)
            A1[index(n, F, 0, n_c), index(n - 1, G, 0, n_c)] = omega_1
            A2 = np.zeros((dim(n_c),) * 2, dtype=complex)
            A2[index(n, G, 1, n_c), index(n, F, 0, n_c)] = omega_2
            terms += [(A1, n * chi_gf), (A2, -n * chi_gf)]
        return terms
    fg = np.zeros((N_Q, N_Q), dtype=complex)
    fg[F, G] = 1.0
    r_dag = np.diag(np.sqrt(np.arange(1, N_R)), -1).astype(complex)
    B1 = _embed(annihilation(n_c).conj().T, fg, np.eye(N_R))
    B2 = _embed(np.eye(n_c), fg.T, r_dag)
    for n in dev.CORRECTED_FOCK:
        terms += [(omega_1 / np.sqrt(n) * B1, n * chi_gf), (omega_2 * B2, -n * chi_gf)]
    return terms


def simulate_frame_ii(rho0, t_grid, include_higher_order=False, six_tones=False,
                      transmon_decay="split", n_c=MIN_N_C, omega_1=dev.OMEGA_1,
                      omega_2=dev.OMEGA_2, chi_ge=dev.chi_ge, chi_gf=dev.chi_gf,
                      chi_qr=dev.chi_qr, rtol=1e-9, atol=1e-11, **collapse_kw):
    """rho(t) in frame (ii) at the times t_grid (starting at t_grid[0]); (len(t), d, d)."""
    H0 = dispersive_hamiltonian(n_c, chi_ge, chi_gf, chi_qr)
    if include_higher_order:
        H0 = H0 + higher_order_shifts(n_c)
    terms = frame_ii_drive_terms(n_c, omega_1, omega_2, chi_gf, six_tones)
    ops = [A for A, _ in terms]
    ws = np.array([w for _, w in terms])
    c_ops = collapse_operators(n_c, transmon_decay, **collapse_kw)
    c_dag = [c.conj().T for c in c_ops]
    with np.errstate(**_QUIET):
        cdc = sum((cd @ c for c, cd in zip(c_ops, c_dag)), np.zeros_like(H0))
    d = H0.shape[0]

    def rhs(t, y):
        rho = y.reshape(d, d)
        X = sum(np.exp(1j * w * t) * A for A, w in zip(ops, ws))
        H = H0 + X + X.conj().T
        out = -1j * (H @ rho - rho @ H) - 0.5 * (cdc @ rho + rho @ cdc)
        for c, cd in zip(c_ops, c_dag):
            out += c @ rho @ cd
        return out.ravel()

    t_grid = np.asarray(t_grid, dtype=float)
    with np.errstate(**_QUIET):
        sol = solve_ivp(rhs, (t_grid[0], t_grid[-1]), rho0.astype(complex).ravel(),
                        method="DOP853", t_eval=t_grid, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T.reshape(-1, d, d)


def storage_reduced(rhos, n_c=MIN_N_C):
    """Partial trace over transmon and readout; works on (d, d) or (T, d, d)."""
    r = np.asarray(rhos)
    shape = r.shape[:-2] + (n_c, N_Q * N_R, n_c, N_Q * N_R)
    return np.trace(r.reshape(shape), axis1=-3, axis2=-1)


FIG2D_PAIRS = ((0, 2), (0, 4), (2, 4))


def simulate_fig2d(pairs=FIG2D_PAIRS, t_final=15.0, include_higher_order=True,
                   six_tones=False, frame="ii", n_c=MIN_N_C, **kw):
    """Fig. 2(d): start in (|m> + |n>)/sqrt2 (x) |g,0>, drive for t_final.

    Returns {(m, n): dict(rho0, rho, coherence, phase)}: rho0/rho are the storage-reduced
    states before and after, coherence = |rho_{m+1,n+1}(t)| / |rho_{mn}(0)| (the paper's
    coherence-preservation factor) and phase = arg rho_{m+1,n+1}(t).
    frame='i' uses the time-independent Eq. (3) model instead (no six-tone option).
    """
    out = {}
    for m, n in pairs:
        psi = (ket(m, G, 0, n_c) + ket(n, G, 0, n_c)) / np.sqrt(2.0)
        rho0 = np.outer(psi, psi.conj())
        t = np.array([0.0, t_final])
        if frame == "ii":
            rho_t = simulate_frame_ii(rho0, t, include_higher_order, six_tones, n_c=n_c, **kw)[-1]
        elif frame == "i":
            if six_tones:
                raise ValueError("the six-tone drive exists only in frame (ii)")
            H = drive_hamiltonian(n_c)
            if include_higher_order:
                H = H + higher_order_shifts(n_c)
            rho_t = propagate(liouvillian(H, collapse_operators(n_c, **kw)), rho0, t)[-1]
        else:
            raise ValueError(f"frame must be 'i' or 'ii', not {frame!r}")
        s0, s = storage_reduced(rho0, n_c), storage_reduced(rho_t, n_c)
        out[(m, n)] = {"rho0": s0, "rho": s,
                       "coherence": abs(s[m + 1, n + 1]) / abs(s0[m, n]),
                       "phase": float(np.angle(s[m + 1, n + 1]))}
    return out


# ---------------------------------------------------------------------------
# Fig. 2(b): the 3x3 non-Hermitian model of one correction path, Eq. (B2)
# ---------------------------------------------------------------------------

def b2_correction_rate(omega_1, omega_2, kappa_r=dev.kappa_r):
    """(|Im lam|, |Re lam|) of the Eq. (B2) eigenvalue whose eigenvector overlaps most
    with |n-1, g, 0> (App. B). Over the Fig. 2(b) range this is also the eigenvalue with
    the smallest |Im| (the caption's wording). |Re lam| > 0 means the decay oscillates."""
    H = np.array([[0, omega_1, 0], [omega_1, 0, omega_2], [0, omega_2, -0.5j * kappa_r]])
    with np.errstate(**_QUIET):
        w, v = np.linalg.eig(H)
    lam = w[np.argmax(np.abs(v[0]) ** 2)]
    return abs(lam.imag), abs(lam.real)


def correction_rate_map(omega_1_grid, omega_2_grid, kappa_r=dev.kappa_r):
    """(rate, osc) arrays of shape (len(omega_1_grid), len(omega_2_grid))."""
    rate = np.empty((len(omega_1_grid), len(omega_2_grid)))
    osc = np.empty_like(rate)
    for i, o1 in enumerate(omega_1_grid):
        for k, o2 in enumerate(omega_2_grid):
            rate[i, k], osc[i, k] = b2_correction_rate(o1, o2, kappa_r)
    return rate, osc
