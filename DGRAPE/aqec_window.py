"""
A realistic step 2 of the Cornell AQEC on Shirol's chip: readout through the readout resonator, a decision, a
conditional pi pulse and ring-down, with every error channel. It replaces the idealised "1.2 us lossy idle + perfect
reset" of DGRAPE/aqec_ideal.aqec_step (steps 4-5), DGRAPE/aqec_pulse.open_step and DGRAPE/aqec_open.window_targets.
Readout physics: DGRAPE/readout.py. Notebook: notebooks/AQEC_compare.ipynb §6.3.

TIMELINE of one window (all on the 1 ns grid, t = 0 is the end of step 1):

    [0, t_drive_end)       readout drive on (ring-up, flat, optional kick, optional CLEAR depletion)
    [0, t_int)             the record is integrated (matched weights)
    t_dec = t_int + latency    decision; if "e", the pi pulse starts (the drive must have ended by then)
    [t_dec, t_dec + t_pi)  pi pulse (Gaussian + DRAG on the transmon drive), only on the "e" branch
    [.., t_w)              ring-down; the next gate starts at t_w. NO ideal reset: leftover e and f go on.

THE MAP. Averaged over outcomes the window is one CPTP map W on storage x transmon (n_r = 1, readout traced):
W = sum_d Post_d o M_d, d in {g, e}. It does not depend on the step-1 pulse, so it drops into the ideal chain, the
B1/B2 sequences and the backward ("folded target") trick of open training unchanged.

WHY IT IS EXACT AND CHEAP: the transmon-diagonal sector. Outside the pi pulse nothing drives the transmon. Coherences
between transmon levels never feed back into the diagonal blocks rho_j = <j|rho|j> (j = g, e, f), the score reads only
the g block, and the measurement destroys them anyway. So the window acts on the three storage blocks rho_j. Every
generator term keeps the storage coherence order k = n - n' (loss and heating shift n and n' together; transmon jumps
move a whole block j -> j'; everything else is diagonal), so each map is a set of small blocks, one per k (at most
3 n_c on a side; ~88k entries in all at n_c 24). Time steps are exact exponentials of the static part, Strang-split
around the time-dependent readout terms. Transmon coherences present at the window's input are dropped (the
measurement kills them); those an imperfect pi pulse creates are dropped too, and their size is reported
(`coh_dropped`).

THE READOUT ACTS ON THE STORAGE through chi_ar (the storage-readout cross-Kerr). For a linear readout the readout-traced
coherence of storage levels n, n' in transmon block j obeys exactly (Gambetta et al., PRA 77, 012112, 2008, generalised)

    d ln rho_{nn'} / dt = -i (Delta_{j,n} - Delta_{j,n'}) alpha_{j,n} alpha*_{j,n'} = +i chi_ar (n - n') alpha_{j,n} alpha*_{j,n'}:

its real part is the measurement-induced storage dephasing, its imaginary part the (branch-dependent) Stark phase.
These terms use the pointer of the level the readout remembers: after the pi pulse (decision e) g and e swap labels.
Approximations (each tiny): after a transmon or storage jump the term switches to the new level's no-jump pointer
(the readout really relaxes to it over ~1/kappa_r), and chi_ar dephasing during the pi pulse itself is left out (its
Stark phase is in).

DECISION-RESOLVED INTEGRATION. Split the generator over [0, t_int) into the part without transmon jumps (U0) and the
transmon jumps J_c (decay e->g, f->e; heating g->e, e->f). With jump-time bins b (default 10 ns) of width h_b,

    M_d = sum_j P(d|j) Pi_j U0 + sum_{c,b} P(d | c at t_b) h_b U0(t_int, t_b) J_c U0(t_b, 0) + two-jump remainder,

the remainder (full - none - one jump, trace ~1e-6) assigned by its final level. P(d|.) comes from the record
(DGRAPE/readout.py: matched filter, efficiency eta, threshold bias). A jump after t_int does not change the decision;
it is inside Post_d (the full generator).

POST_d. "g": ring-down from t_int to t_w. "e": latency, the pi pulse, ring-down. The pi pulse runs on the full space
(lindblad_np, 3 transmon levels, chi_ge spread, Kerr, every channel) and is averaged over the readout photon number at
t_dec: Poisson(|alpha_j(t_dec)|^2) per block, each photon number adding -n_r (chi_qr |e><e| + chi_fr |f><f| +
chi_ar n_a) to the drift. Its amplitude, DRAG and carrier are calibrated closed (`calibrate_pi`, on the code-word Fock
numbers 1, 3, 5, carrier tracking the mean readout Stark shift at t_dec by default).

CHEAP RE-SCORING. F is linear in the outcome probabilities: fold the targets backwards through Post_d once, then
F(eta, bias) = sum over terms of P(d | term) <T_d, term> (`Terms`). eta, the threshold and the weights cost no
propagation.

PRE-TWIST. aqec_ideal's D = exp(+i H_d tau2) generalises to `Window.pretwist()`: a diagonal phase per (n, branch) read
off the closed, no-jump, perfect-decision window (the chi_ar Stark phase, the time in e before the pi pulse, the pi
pulse's n-dependent phase). With zero drive, perfect decision, an instant pi pulse at t_w and a final ideal reset the
window reproduces aqec_ideal.aqec_step and aqec_pulse.open_step to round-off (validation/test_aqec_window.py).
"""

import copy
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

np.seterr(divide="ignore", over="ignore", invalid="ignore")   # spurious Accelerate matmul warnings (loky workers)

from DGRAPE import aqec_ideal as aq
from DGRAPE import fwm
from DGRAPE import fwm_repeat as fr
from DGRAPE import lindblad_np as lnp
from DGRAPE import readout as ro
from DGRAPE.model import _QUIET, E, G, N_Q, monomial

DT = ro.DT
LEVEL_NAMES = ("g", "e", "f")


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------

@dataclass
class PiSpec:
    """kind "gauss" (Gaussian + DRAG, 4 sigma long, calibrated) or "ideal" (instant g <-> e swap)."""
    kind: str = "gauss"
    sigma_ns: float = 8.0
    track: bool = True              # carrier follows the mean readout Stark shift of e at t_dec


@dataclass
class WindowSpec:
    """One readout window. t_int None = the drive end. t_w None = the first time after the pi pulse at which every
    pointer holds at most nres_target photons. decision "signal" (the record) or "ideal" (the level at t_int).
    readout_on False drops the chi_ar terms (a check)."""
    drive: ro.DriveSpec = field(default_factory=ro.DriveSpec)
    t_int: Optional[float] = None
    latency: float = 0.3
    pi: PiSpec = field(default_factory=PiSpec)
    t_w: Optional[float] = None
    nres_target: float = 0.01
    eta: float = 0.3
    bias: float = 0.0
    weights: str = "matched"
    decision: str = "signal"
    chi_ar: float = ro.CHI_AR
    chi_fr: float = ro.CHI_FR
    bin_ns: int = 10
    readout_on: bool = True

    def to_dict(self):
        d = dict(self.__dict__)
        d["drive"] = self.drive.to_dict()
        d["pi"] = dict(self.pi.__dict__)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["drive"] = ro.DriveSpec.from_dict(d["drive"])
        d["pi"] = PiSpec(**d["pi"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def replace(self, **kw):
        out = copy.deepcopy(self)
        for k, v in kw.items():
            setattr(out, k, v)
        return out


def save_spec(spec, path, **extra):
    with open(path, "w") as f:
        json.dump({"window": spec.to_dict(), **extra}, f, indent=1, default=float)


def load_spec(path):
    with open(path) as f:
        d = json.load(f)
    return WindowSpec.from_dict(d["window"] if "window" in d else d)


# ---------------------------------------------------------------------------
# The transmon-diagonal sector, split by storage coherence order
# ---------------------------------------------------------------------------

class Sector:
    """Index maps and generator blocks of the transmon-diagonal sector of a DGRAPE model at n_r = 1.

    States are (K, M, b) arrays: K = 2 n_c - 1 coherence orders k = n - n', M = 3 n_c padded positions (block j first,
    then n), b process inputs. Padded positions stay zero (their generator rows and columns are zero)."""

    def __init__(self, m):
        if m.n_r != 1:
            raise ValueError("the window model must have n_r = 1 (the readout is treated by its pointers)")
        self.m = m
        n_c = self.n_c = m.n_c
        self.d = m.d
        self.K, self.M = 2 * n_c - 1, N_Q * n_c
        j, n, n2 = np.meshgrid(np.arange(N_Q), np.arange(n_c), np.arange(n_c), indexing="ij")
        j, n, n2 = j.ravel(), n.ravel(), n2.ravel()
        k = n - n2
        lo = np.maximum(0, k)
        L = n_c - np.abs(k)
        self.j, self.n, self.n2 = j, n, n2
        self.a, self.b = n * N_Q + j, n2 * N_Q + j
        self.kb = k + n_c - 1
        self.pos = j * L + (n - lo)
        self.flat = self.kb * self.M + self.pos
        lookup = -np.ones((self.d, self.d), dtype=np.int64)
        lookup[self.a, self.b] = self.flat
        self._lookup = lookup
        h = np.real(np.diag(m.H_d))
        G0 = np.zeros((self.K, self.M, self.M), complex)
        G0[self.kb, self.pos, self.pos] = -1j * (h[self.a] - h[self.b]) - 0.5 * (m.Gdiag[self.a] + m.Gdiag[self.b])
        self.jumps, self.transitions, self.jump_names = [], [], []
        for name, c in m.c_ops:
            Gc, trans = self._sandwich(c)
            if trans is None:
                G0 += Gc
            else:
                self.jumps.append(Gc)
                self.transitions.append(trans)
                self.jump_names.append(name)
        self.G0 = G0
        self.Gfull = G0 + sum(self.jumps, np.zeros_like(G0))
        self.mask = np.zeros((N_Q, self.K * self.M))
        for jj in range(N_Q):
            self.mask[jj, self.flat[self.j == jj]] = 1.0
        self.mask = self.mask.reshape(N_Q, self.K, self.M, 1)
        perm = np.arange(self.K * self.M)
        for jj, jp in ((G, E), (E, G)):
            src = self.flat[self.j == jj]
            dst = self._lookup[self.a[self.j == jj] - jj + jp, self.b[self.j == jj] - jj + jp]
            perm[dst] = src
        self.swap_perm = perm
        self._exp = {}

    def _sandwich(self, c):
        """Generator block of the sandwich c rho c' on the sector, and (j_from, j_to) if c moves the transmon."""
        src, w = monomial(c)
        coef = w[self.a] * np.conj(w[self.b])
        nz = coef != 0
        sa, sb = src[self.a[nz]], src[self.b[nz]]
        sflat = self._lookup[sa, sb]
        if np.any(sflat < 0):
            raise ValueError("jump operator leaves the transmon-diagonal sector")
        skb, spos = sflat // self.M, sflat % self.M
        if np.any(skb != self.kb[nz]):
            raise ValueError("jump operator changes the storage coherence order")
        Gc = np.zeros((self.K, self.M, self.M), complex)
        np.add.at(Gc, (self.kb[nz], self.pos[nz], spos), coef[nz])
        j_src, j_dst = sa % N_Q, self.a[nz] % N_Q
        if np.all(j_src == j_dst):
            return Gc, None
        pairs = set(zip(j_src.tolist(), j_dst.tolist()))
        if len(pairs) != 1:
            raise ValueError("a transmon jump operator must drive one transition")
        return Gc, pairs.pop()

    def exps(self, h):
        """(exp(G0 h), exp(Gfull h)) per coherence order, cached per step length."""
        key = round(h, 12)
        if key not in self._exp:
            with np.errstate(**_QUIET):
                self._exp[key] = (np.stack([expm(g * h) for g in self.G0]),
                                  np.stack([expm(g * h) for g in self.Gfull]))
        return self._exp[key]

    def to_sector(self, R):
        R = np.asarray(R)
        X = np.zeros((self.K * self.M, R.shape[0]), complex)
        X[self.flat] = R[:, self.a, self.b].T
        return X.reshape(self.K, self.M, R.shape[0])

    def from_sector(self, X):
        nb = X.shape[-1]
        R = np.zeros((nb, self.d, self.d), complex)
        R[:, self.a, self.b] = X.reshape(self.K * self.M, nb)[self.flat].T
        return R

    def swap(self, X):
        nb = X.shape[-1]
        return X.reshape(self.K * self.M, nb)[self.swap_perm].reshape(X.shape)

    def readout_rate(self, alpha, lab, chi_ar):
        """(K, M, 1) diagonal readout terms +i chi_ar (n - n') alpha_{lab j, n} alpha*_{lab j, n'} (alpha (3, n_c))."""
        lj = np.asarray(lab)[self.j]
        v = 1j * chi_ar * (self.n - self.n2) * alpha[lj, self.n] * np.conj(alpha[lj, self.n2])
        out = np.zeros(self.K * self.M, complex)
        out[self.flat] = v
        return out.reshape(self.K, self.M, 1)


def inner(X, Y):
    """Re <X, Y> summed over the sector and the process inputs."""
    return float(np.real(np.vdot(X, Y)))


def _matmul(A, X):
    with np.errstate(**_QUIET):
        return A @ X


# ---------------------------------------------------------------------------
# The pi pulse
# ---------------------------------------------------------------------------

def gauss_envelope(sigma_ns, dt=DT):
    """(g, dg/dt) at the step midpoints of a 4 sigma Gaussian, shifted to zero at its ends (us units)."""
    n = int(round(4 * sigma_ns * 1e-3 / dt))
    t = (np.arange(n) + 0.5) * dt
    s, T = sigma_ns * 1e-3, n * dt
    ex = np.exp(-(t - T / 2) ** 2 / (2 * s * s))
    return ex - np.exp(-(T / 2) ** 2 / (2 * s * s)), -(t - T / 2) / (s * s) * ex


def pi_waveform(params, sigma_ns, dt=DT):
    """u (n, 4) of the pi pulse: transmon quadratures only, eps = A (g - i drag dg/dt) exp(-i delta t)."""
    A, drag, delta = params
    g, dg = gauss_envelope(sigma_ns, dt)
    t = (np.arange(g.size) + 0.5) * dt
    eps = A * (g - 1j * drag * dg) * np.exp(-1j * delta * t)
    u = np.zeros((g.size, 4))
    u[:, 2], u[:, 3] = eps.real, -eps.imag
    return u


def _transmon_levels(m, n):
    h = np.real(np.diag(m.H_d))
    return np.array([h[(n * N_Q + j) * m.n_r] for j in range(N_Q)])


@lru_cache(maxsize=256)
def _calibrate(levels_key, sigma_ns, stark_e, stark_f):
    levels = [np.array(lv) for lv in levels_key]
    q = np.diag(np.sqrt(np.arange(1, N_Q)), 1).astype(complex)
    qd = q.conj().T

    def transfer(params):
        u = pi_waveform(params, sigma_ns)
        out = []
        for lv in levels:
            H0 = np.diag(lv - lv[0] - np.array([0.0, stark_e, stark_f]))
            U = np.eye(N_Q, dtype=complex)
            for uk in u:
                eps = uk[2] - 1j * uk[3]
                U = expm(-1j * DT * (H0 + eps * qd + np.conj(eps) * q)) @ U
            out.append(abs(U[G, E]) ** 2)
        return np.array(out)

    g, _ = gauss_envelope(sigma_ns)
    A0 = 0.5 * np.pi / (np.sum(g) * DT)
    delta0 = float(np.mean([lv[E] - lv[G] - stark_e for lv in levels]))
    s_ns = sigma_ns * 1e-3
    # search in scaled units: amplitude / A0, DRAG / sigma, carrier offset in MHz
    unscale = lambda z: (A0 * z[0], z[1] * s_ns, delta0 + 2 * np.pi * z[2])  # noqa: E731
    best = None
    for z0 in ([1.0, 0.0, 0.0], [1.0, 0.3, 0.0], [1.0, -0.3, 0.0]):
        r = minimize(lambda z: 1.0 - transfer(unscale(z)).mean(), z0, method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-13, "maxiter": 6000, "initial_simplex":
                              np.array(z0) + np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.2, 0], [0, 0, 0.5]])})
        if best is None or r.fun < best.fun:
            best = r
    res = type("R", (), {"x": np.array(unscale(best.x))})
    return tuple(float(x) for x in res.x), float(1.0 - transfer(res.x).mean())


def calibrate_pi(m, sigma_ns, stark_photons=0.0, chi_fr=ro.CHI_FR, fock=(1, 3, 5)):
    """((A, drag, delta), closed infidelity): the pi pulse that best moves |n, e> -> |n, g> for n in fock (the corrected
    code words), closed 3-level transmon with the model's drift, e (f) shifted by stark_photons chi_qr (chi_fr).
    Phases are free (the pre-twist takes them)."""
    key = tuple(tuple(np.round(_transmon_levels(m, n), 9)) for n in fock)
    return _calibrate(key, float(sigma_ns), round(float(stark_photons) * ro.CHI_QR, 3),
                      round(float(stark_photons) * chi_fr, 3))


def shifted_model(m, extra_diag):
    """Copy of model m with H_d + diag(extra_diag) (a frozen readout photon number)."""
    m2 = copy.copy(m)
    m2.H_d = m.H_d + np.diag(extra_diag).astype(complex)
    hd = np.real(np.diag(m2.H_d))
    m2.shift = 0.5 * (hd.max() + hd.min())
    m2.H_d_shifted = m2.H_d - m2.shift * np.eye(m.d)
    m2.half_width_d = 0.5 * (hd.max() - hd.min())
    return m2


def propagate_adjoint(model, u, X, dt=DT, spec=None):
    spec = lnp.spec_for_pulse(model, u, dt) if spec is None else spec
    for k in range(u.shape[0] - 1, -1, -1):
        X = lnp.step_adjoint(model, u[k], X, dt, spec)
    return X


_CLOSED = {}


def _closed(n_c):
    """(model, Sector) of the closed system at n_c, n_r = 1, cached (the pre-twist calibration)."""
    if n_c not in _CLOSED:
        mc = fwm.model_with_channels(n_c, 1, frozenset(), True)
        _CLOSED[n_c] = (mc, Sector(mc))
    return _CLOSED[n_c]


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------

class Window:
    """The realistic readout window on the n_r = 1 model m (module docstring)."""

    def __init__(self, spec, m, sector=None):
        self.spec, self.m = spec, m
        self.sec = sector if sector is not None and sector.m is m else Sector(m)
        n_c = m.n_c
        self.eps_drive = ro.build_drive(spec.drive, DT, spec.chi_fr)
        n_drive = self.eps_drive.size
        n_int = n_drive if spec.t_int is None else ro.n_steps(spec.t_int)
        n_dec = n_int + ro.n_steps(spec.latency)
        if n_drive > n_dec:
            raise ValueError(f"the readout drive ({n_drive} ns) must end by the decision ({n_dec} ns)")
        if spec.pi.kind == "gauss":
            n_pi = int(round(4 * spec.pi.sigma_ns))
        elif spec.pi.kind == "ideal":
            n_pi = 0
        else:
            raise ValueError(f"pi kind must be 'gauss' or 'ideal', not {spec.pi.kind!r}")
        self.delta = ro.detunings(n_c, spec.drive.detuning_mhz, spec.chi_ar, spec.chi_fr)   # (3, n_c)
        n_free = n_dec + n_pi
        if spec.t_w is None:
            a_end = ro.pointers(np.r_[self.eps_drive, np.zeros(max(n_free - n_drive, 0))], self.delta[:, 0])[0][-1]
            n_w = n_free + int(np.ceil(ro.ring_down_time(float(np.max(np.abs(a_end) ** 2)), spec.nres_target) / DT))
        else:
            n_w = ro.n_steps(spec.t_w)
            if n_w < n_free:
                raise ValueError(f"t_w = {spec.t_w} us ends before the pi pulse does ({n_free} ns)")
        self.n_drive, self.n_int, self.n_dec, self.n_pi, self.n_w = n_drive, n_int, n_dec, n_pi, n_w
        self.duration = n_w * DT
        eps = np.r_[self.eps_drive, np.zeros(n_w - n_drive)]
        grid, mid = ro.pointers(eps, self.delta)
        self.alpha_grid, self.alpha_mid = grid, mid                                       # (n_w + 1 / n_w, 3, n_c)
        trans = self.sec.transitions
        self.record = ro.record(eps, n_int, trans, spec.bin_ns, spec.drive.detuning_mhz, spec.weights, DT,
                                spec.chi_fr)
        self.bin_centres = self.record.bin_centres
        self.bin_widths = self.record.bin_widths
        # the pi pulse: photon-number mixture per transmon block at t_dec
        self.nbar_dec = np.abs(grid[n_dec, :, 0]) ** 2
        self.coh_dropped = 0.0
        if spec.pi.kind == "gauss":
            stark = float(self.nbar_dec[E]) if spec.pi.track else 0.0
            self.pi_params, self.pi_closed_infidelity = calibrate_pi(m, spec.pi.sigma_ns, stark, spec.chi_fr)
            self.u_pi = pi_waveform(self.pi_params, spec.pi.sigma_ns)
            nmax = int(max(poisson.ppf(1 - 1e-10, max(nb, 1e-12)) for nb in self.nbar_dec))
            nr = np.arange(nmax + 1)
            self.pi_weights = np.stack([poisson.pmf(nr, max(nb, 0.0)) if nb > 1e-14 else (nr == 0).astype(float)
                                        for nb in self.nbar_dec])                                     # (3, n_r)
            ops = np.zeros((N_Q, n_c))
            for j, chi in ((E, ro.CHI_QR), (2, spec.chi_fr)):
                ops[j] += chi
            ops += spec.chi_ar * np.arange(n_c)[None, :]
            per_photon = -ops.T.ravel()                                                       # index n*3 + j
            self._pi_models = [shifted_model(m, k * per_photon) for k in nr]
            self._pi_spec = lnp.spec_for_pulse(self._pi_models[-1], self.u_pi, DT)
        self.set_decision(spec.eta, spec.bias)

    # -- decision probabilities ------------------------------------------------

    def p_e(self, eta=None, bias=None):
        """(P(e | level) (3,), P(e | jump c at bin b) (C, B))."""
        eta = self.spec.eta if eta is None else eta
        bias = self.spec.bias if bias is None else bias
        if self.spec.decision == "ideal":
            lvl = np.array([0.0, 1.0, 0.0])
            jmp = np.zeros((len(self.sec.transitions), self.bin_centres.size))
            for c, (_, jp) in enumerate(self.sec.transitions):
                jmp[c] = 1.0 if jp == E else 0.0
            return lvl, jmp
        if self.spec.decision != "signal":
            raise ValueError(f"decision must be 'signal' or 'ideal', not {self.spec.decision!r}")
        return self.record.p_e_level(eta, bias), self.record.p_e_jump(eta, bias)

    def set_decision(self, eta, bias):
        self.eta, self.bias = eta, bias
        self.P_level, self.P_jump = self.p_e(eta, bias)

    # -- sector time stepping ----------------------------------------------------

    def _steps(self, X, k0, k1, full, lab, adjoint=False, inject=None):
        """Propagate X over steps [k0, k1) (backwards if adjoint). inject(k, X) is called at the boundary k (before step
        k forward, after it backward) for every k in the bin centres."""
        E0, Ef = self.sec.exps(DT)
        Ek = Ef if full else E0
        if adjoint:
            Ek = np.conj(np.swapaxes(Ek, 1, 2))
        on = self.spec.readout_on and self.spec.chi_ar != 0
        centres = set(int(c) for c in self.bin_centres) if inject is not None else ()
        ks = range(k1 - 1, k0 - 1, -1) if adjoint else range(k0, k1)
        for k in ks:
            if not adjoint and k in centres:
                inject(k, X)
            if on:
                D = np.exp(0.5 * DT * self.sec.readout_rate(self.alpha_mid[k], lab, self.spec.chi_ar))
                if adjoint:
                    D = np.conj(D)
                X = D * _matmul(Ek, D * X)
            else:
                X = _matmul(Ek, X)
            if adjoint and k in centres:
                inject(k, X)
        return X

    def _bin_index(self):
        return {int(c): i for i, c in enumerate(self.bin_centres)}

    # -- integration [0, t_int) --------------------------------------------------

    def integrate_forward(self, X):
        """(M_g, M_e): the decision-resolved states at t_int (sector)."""
        sec, lab = self.sec, (0, 1, 2)
        Pl, Pj = self.P_level, self.P_jump
        bi = self._bin_index()
        nb = X.shape[-1]
        # the no-jump state and three accumulators (decision g, decision e, unweighted) share U0: one stacked array
        Z = np.concatenate([X, np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)], axis=-1)

        def inject(k, Zk):
            b = bi[k]
            for c, J in enumerate(sec.jumps):
                JX = _matmul(J, Zk[..., :nb]) * self.bin_widths[b]
                Zk[..., nb:2 * nb] += (1.0 - Pj[c, b]) * JX
                Zk[..., 2 * nb:3 * nb] += Pj[c, b] * JX
                Zk[..., 3 * nb:] += JX

        Z = self._steps(Z, 0, self.n_int, False, lab, inject=inject)
        X0, Yg, Ye, Y1 = (Z[..., i * nb:(i + 1) * nb] for i in range(4))
        Xf = self._steps(X, 0, self.n_int, True, lab)
        R2 = Xf - X0 - Y1                                   # two or more jumps
        self.remainder_trace = self._trace(R2)
        Mg, Me = Yg.copy(), Ye.copy()
        for j in range(N_Q):
            Mg += (1.0 - Pl[j]) * sec.mask[j] * (X0 + R2)
            Me += Pl[j] * sec.mask[j] * (X0 + R2)
        return Mg, Me

    def _trace(self, X):
        """Largest |trace| over the process inputs of a sector state."""
        diag = self.sec.n == self.sec.n2
        return float(np.abs(X.reshape(self.sec.K * self.sec.M, -1)[self.sec.flat[diag]].sum(axis=0)).max())

    def integrate_adjoint(self, Tg, Te):
        """Adjoint of integrate_forward: targets Tg, Te at t_int -> the target at t = 0 (sector).

        <Tg, M_g> + <Te, M_e> = <Q, Xf> - <Q, Y1> + <Tg, Y_g> + <Te, Y_e>, Q = sum_j Pi_j((1 - P_j) Tg + P_j Te), because
        X0 + R2 = Xf - Y1. The full part is Full'(Q); each one-jump part is h_b J_c' of a U0'-propagated target at t_b."""
        sec, lab = self.sec, (0, 1, 2)
        Pl, Pj = self.P_level, self.P_jump
        bi = self._bin_index()
        nb = Tg.shape[-1]
        Q = sum(sec.mask[j] * ((1.0 - Pl[j]) * Tg + Pl[j] * Te) for j in range(N_Q))
        Z = np.concatenate([np.zeros_like(Tg), Tg, Te, Q], axis=-1)     # main, Lambda_g, Lambda_e, Lambda_Q

        def inject(k, Zk):
            b = bi[k]
            Lg, Le, LQ = (Zk[..., i * nb:(i + 1) * nb] for i in (1, 2, 3))
            for c, J in enumerate(sec.jumps):
                Jd = np.conj(np.swapaxes(J, 1, 2))
                Zk[..., :nb] += self.bin_widths[b] * _matmul(Jd, (1.0 - Pj[c, b]) * Lg + Pj[c, b] * Le - LQ)

        Z = self._steps(Z, 0, self.n_int, False, lab, adjoint=True, inject=inject)
        return Z[..., :nb] + self._steps(Q, 0, self.n_int, True, lab, adjoint=True)

    # -- after the decision --------------------------------------------------------

    def _pi_forward(self, X):
        out = np.zeros((X.shape[-1], self.m.d, self.m.d), complex)
        blocks = [self.sec.from_sector(self.sec.mask[j] * X) for j in range(N_Q)]
        for i, mod in enumerate(self._pi_models):
            if self.pi_weights[:, i].max() < 1e-13:
                continue
            Rin = sum(self.pi_weights[j, i] * blocks[j] for j in range(N_Q))
            out += lnp.propagate(mod, self.u_pi, Rin, DT, spec=self._pi_spec)
        S = self.sec.to_sector(out)
        back = self.sec.from_sector(S)
        # largest Frobenius norm, over the process inputs, of the transmon coherences the pi pulse made (dropped)
        self.coh_dropped = float(np.max(np.linalg.norm(out - back, axis=(1, 2))))
        return S

    def _pi_adjoint(self, Y):
        T = self.sec.from_sector(Y)
        out = np.zeros_like(Y)
        for i, mod in enumerate(self._pi_models):
            if self.pi_weights[:, i].max() < 1e-13:
                continue
            Z = self.sec.to_sector(propagate_adjoint(mod, self.u_pi, T, DT, spec=self._pi_spec))
            out += sum(self.pi_weights[j, i] * self.sec.mask[j] * Z for j in range(N_Q))
        return out

    def post_forward(self, d, X):
        ident, swapped = (0, 1, 2), (1, 0, 2)
        if d == "g":
            return self._steps(X, self.n_int, self.n_w, True, ident)
        X = self._steps(X, self.n_int, self.n_dec, True, ident)
        X = self.sec.swap(X) if self.spec.pi.kind == "ideal" else self._pi_forward(X)
        return self._steps(X, self.n_dec + self.n_pi, self.n_w, True, swapped)

    def post_adjoint(self, d, Y):
        ident, swapped = (0, 1, 2), (1, 0, 2)
        if d == "g":
            return self._steps(Y, self.n_int, self.n_w, True, ident, adjoint=True)
        Y = self._steps(Y, self.n_dec + self.n_pi, self.n_w, True, swapped, adjoint=True)
        Y = self.sec.swap(Y) if self.spec.pi.kind == "ideal" else self._pi_adjoint(Y)
        return self._steps(Y, self.n_int, self.n_dec, True, ident, adjoint=True)

    # -- the whole window ------------------------------------------------------------

    def apply(self, R):
        """W(R) for a stack R (b, d, d): the state at t_w, transmon as it is (no reset), readout traced."""
        X = self.sec.to_sector(R)
        Mg, Me = self.integrate_forward(X)
        return self.sec.from_sector(self.post_forward("g", Mg) + self.post_forward("e", Me))

    def adjoint(self, T):
        """W'(T) for a stack T (b, d, d) (block-diagonal in the transmon)."""
        Y = self.sec.to_sector(T)
        return self.sec.from_sector(self.integrate_adjoint(self.post_adjoint("g", Y), self.post_adjoint("e", Y)))

    def terms(self, R, T, w):
        """Terms for cheap re-scoring of F = sum_b w_b Re Tr[T_b' W(R)_b] / 4 (module docstring)."""
        sec, lab = self.sec, (0, 1, 2)
        X = sec.to_sector(R)
        Y = sec.to_sector(np.asarray(w)[:, None, None] * T / 4.0)
        Td = {"g": self.post_adjoint("g", Y), "e": self.post_adjoint("e", Y)}
        bi = self._bin_index()
        C, B, nb = len(sec.jumps), self.bin_centres.size, X.shape[-1]
        X0_bins = np.zeros((B,) + X.shape, complex)
        Z = np.concatenate([X, np.zeros_like(X)], axis=-1)          # no-jump state, unweighted one-jump sum

        def grab(k, Zk):
            b = bi[k]
            X0_bins[b] = Zk[..., :nb]
            for J in sec.jumps:
                Zk[..., nb:] += _matmul(J, Zk[..., :nb]) * self.bin_widths[b]

        Z = self._steps(Z, 0, self.n_int, False, lab, inject=grab)
        X0, Y1 = Z[..., :nb], Z[..., nb:]
        Xf = self._steps(X, 0, self.n_int, True, lab)
        R2 = Xf - X0 - Y1
        f1 = np.zeros((2, C, B))
        Jd = [np.conj(np.swapaxes(J, 1, 2)) for J in sec.jumps]
        for i, dd in enumerate(("g", "e")):
            def score(k, Lk, i=i):
                b = bi[k]
                for c in range(C):
                    f1[i, c, b] = self.bin_widths[b] * inner(_matmul(Jd[c], Lk), X0_bins[b])

            self._steps(Td[dd], 0, self.n_int, False, lab, adjoint=True, inject=score)
        f0 = np.array([[inner(sec.mask[j] * Td[dd], X0) for j in range(N_Q)] for dd in ("g", "e")])
        f2 = np.array([[inner(sec.mask[j] * Td[dd], R2) for j in range(N_Q)] for dd in ("g", "e")])
        return Terms(self, f0, f1, f2, remainder_trace=self._trace(R2))

    def pretwist(self):
        """Diagonal v (d,) of the pre-twist: rotate(R, v) before the window cancels its deterministic storage phases on
        the g branch (no pi pulse) and the e branch (pi pulse), read off the closed, perfect-decision window."""
        mc, sc = _closed(self.m.n_c)
        closed = Window(self.spec.replace(decision="ideal"), mc, sc)
        n_c = self.m.n_c
        v = np.ones(self.m.d, complex)
        for j in (G, E):
            psi = np.zeros(self.m.d, complex)
            psi[np.arange(n_c) * N_Q + j] = 1.0 / np.sqrt(n_c)
            out = closed.apply(np.outer(psi, psi.conj())[None])[0]
            col = out[np.arange(n_c) * N_Q + G, G]
            phi = -np.angle(col / col[0]) if abs(col[0]) > 0 else np.zeros(n_c)
            v[np.arange(n_c) * N_Q + j] = np.exp(1j * phi)
        return v

    # -- diagnostics ---------------------------------------------------------------------

    def transmon_rates(self):
        """Classical transmon jump rates (1/us) at storage n = 0, from the model's operators: {(j, j'): rate}."""
        k0, n_c = self.m.n_c - 1, self.m.n_c
        return {(jf, jt): float(np.real(Gc[k0, jt * n_c, jf * n_c]))
                for (jf, jt), Gc in zip(self.sec.transitions, self.sec.jumps)}

    def confusion(self, eta=None, bias=None):
        """P(decide e | start g), P(decide g | start e): noise only, and with the transmon jumps during [0, t_int)."""
        Pl, Pj = self.p_e(eta, bias)
        rates = self.transmon_rates()
        t = self.bin_centres * DT
        out = {"P_e_given_g_noise": float(Pl[G]), "P_g_given_e_noise": float(1 - Pl[E])}
        for j0, key in ((G, "P_e_given_g"), (E, "P_g_given_e")):
            gam = sum(r for (jf, _), r in rates.items() if jf == j0)
            tint = self.n_int * DT
            p = np.exp(-gam * tint) * Pl[j0]
            for c, (jf, jt) in enumerate(self.sec.transitions):
                if jf == j0:
                    p += np.sum(rates[(jf, jt)] * self.bin_widths * np.exp(-gam * t) * Pj[c])
            out[key] = float(p if j0 == G else 1 - p)
        return out

    def metrics(self):
        a = np.abs(self.alpha_grid[:, :2, 0]) ** 2
        return {"t_drive_us": self.n_drive * DT, "t_int_us": self.n_int * DT, "t_dec_us": self.n_dec * DT,
                "t_pi_us": self.n_pi * DT, "t_w_us": self.duration, "n_peak": float(a.max()),
                "n_dec": float(self.nbar_dec[:2].max()), "n_res": float(a[-1].max()),
                "snr": float(ro.snr(self.record.s1_level[G], self.record.s1_level[E], self.eta))
                if self.n_int else 0.0,
                "pi_closed_infidelity": getattr(self, "pi_closed_infidelity", 0.0),
                "coh_dropped": self.coh_dropped, "remainder_trace": getattr(self, "remainder_trace", float("nan")),
                **self.confusion()}


@dataclass
class Terms:
    """F(eta, bias) without propagation (Window.terms)."""
    window: Window
    f0: np.ndarray           # (2, 3)    decision g/e x no-jump level
    f1: np.ndarray           # (2, C, B) decision x jump type x bin
    f2: np.ndarray           # (2, 3)    two-jump remainder by final level
    remainder_trace: float = 0.0

    def fidelity(self, eta=None, bias=None):
        Pl, Pj = self.window.p_e(eta, bias)
        lvl = (1 - Pl) * (self.f0[0] + self.f2[0]) + Pl * (self.f0[1] + self.f2[1])
        jmp = (1 - Pj) * self.f1[0] + Pj * self.f1[1]
        return float(lvl.sum() + jmp.sum())

    def best_bias(self, eta=None, lo=-6.0, hi=6.0):
        """(bias, F) maximising F at efficiency eta (bounded scalar search)."""
        if self.window.spec.decision == "ideal" or np.isinf(eta if eta is not None else self.window.spec.eta):
            return 0.0, self.fidelity(eta, 0.0)
        res = minimize_scalar(lambda b: -self.fidelity(eta, b), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-4})
        return float(res.x), -float(res.fun)


# ---------------------------------------------------------------------------
# The AQEC step with the realistic window
# ---------------------------------------------------------------------------

def real_step(window, R, K, mode="forgive", tau1=aq.TAU1, twist=None):
    """aqec_ideal.aqec_step with its steps 3-5 (pre-twist, 1.2 us idle, ideal reset) replaced by the calibrated
    pre-twist and the realistic window. Returns the state at the end of the window (no reset)."""
    m = window.m
    R0 = aq.forgive(R, m.n_c, m.n_r) if mode == "forgive" else aq.strict(R, m.n_c, m.n_r)
    R1 = aq.rotate(aq.idle(m, R0, tau1), aq.phase(m, tau1))
    R2 = aq.apply_map(R1, K)
    v = window.pretwist() if twist is None else twist
    return window.apply(aq.rotate(R2, v))


def open_step_real(window, R, pulse, dt=1e-3):
    """aqec_pulse.open_step with the 1.2 us idle + ideal reset replaced by the realistic window."""
    R1 = lnp.propagate(window.m, np.asarray(pulse, float), R, dt)
    return window.apply(R1)


def gate_with_residual_photons(pulse, nbar, n_c=24, n_r=6, channels="shirol", target="X"):
    """F_pro (and the score_state row) of one gate whose readout starts in Poisson(nbar) photons (diagonal: the gate
    never drives the readout, so its coherences do not matter), n_r levels, readout traced at the end."""
    m = aq.model(n_c, n_r, channels)
    p = poisson.pmf(np.arange(n_r), nbar) if nbar > 0 else (np.arange(n_r) == 0).astype(float)
    p = p / p.sum()
    E1 = fr.inputs(n_c, 1)
    R = np.einsum("bij,mn->bimjn", E1, np.diag(p)).reshape(E1.shape[0], m.d, m.d)
    out = fr.gate_step(m, np.asarray(pulse, float), R)
    return fr.score_state(out, n_c, n_r, target)
