"""
Dispersive transmon readout on Shirol's chip: the classical pointer states of the readout resonator, the drive shapes,
the integrated record and the g/e decision. The physics behind DGRAPE/aqec_window.py (the realistic step 2 of the
Cornell AQEC). Notebook: notebooks/AQEC_compare.ipynb §6.3.

POINTERS. Drive the readout at omega_d. In the frame of the drive, with the transmon in level j and the storage
holding n photons, the readout is a coherent state |alpha_{j,n}(t)> with

    d alpha / dt = -i eps(t) - (kappa_r/2 + i Delta_{j,n}) alpha,
    Delta_{j,n}  = omega_r - chi_j - chi_ar n - omega_d,      chi_g = 0, chi_e = chi_qr, chi_f = CHI_FR.

(DGRAPE.model writes the readout shift as -chi_qr |e><e| r'r, so e pulls the readout DOWN by chi_qr.) The default drive
sits at the midpoint omega_r - chi_qr/2, so Delta_g = +chi_qr/2 and Delta_e = -chi_qr/2; `detuning_mhz` moves it.
eps is piecewise constant on a DT grid and every step is integrated exactly.

ASSUMPTIONS. CHI_FR = 0, as DGRAPE.model (no f readout shift: f reads like g). CHI_AR = 2 pi 9.3 kHz (Shirol's text;
sources/device_comparison.md), sign convention -chi_ar n_a r'r like the other chis; its sign only changes the
deterministic storage phase, which the pre-twist removes. N_CAP caps the peak photon number of every drive
(measurement-induced state transitions, MIST, are out of scope; n_crit ~ 58 on this chip).

THE RECORD. The amplifier chain records the outgoing field sqrt(kappa_r) alpha with efficiency eta. With integration
weights w(t) on [0, t_int] the integrated, projected signal is

    S = Re int w*(t) [2 sqrt(eta kappa_r) alpha(t) dt + dW_c(t)],      E|dW_c|^2 = 2 dt,

Gaussian with mean mu = 2 sqrt(eta kappa_r) Re int w* alpha dt and sigma^2 = int |w|^2 dt. Matched weights
w = alpha_e - alpha_g give SNR^2 = (mu_e - mu_g)^2 / sigma^2 = 4 eta kappa_r int |alpha_e - alpha_g|^2 dt. eta is the
EFFECTIVE efficiency (a phase-preserving amplifier caps it at 0.5). Here everything is written per unit sqrt(eta):
s1 = mu / (sqrt(eta) sigma), so the decision variable is s = sqrt(eta) s1 + unit Gaussian noise.

THE DECISION. "e" when s > theta, theta = sqrt(eta) (s1_g + s1_e)/2 + bias (bias in units of sigma: a positive bias
asks for more evidence before flagging e, which protects the ~97 % g branch from a wrong pi pulse). For a trajectory
with mean s: P(e) = Q(theta - s) = erfc((theta - s)/sqrt 2)/2. eta = inf is the noiseless limit (a step function).

A transmon jump j -> j' at time t during the integration changes the record: the pointer follows Delta_j up to t and
Delta_j' from alpha_j(t) on. `jump_means` gives s1 for every jump type and jump time (bin centres), with storage n = 0
(chi_ar / chi_qr = 0.8 %: the storage does not change the record).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import erfc

from AQEC import device as dev

two_pi = 2 * np.pi
DT = 0.001                      # us, 1 ns grid (an AWG at 1 GS/s)
CHI_QR = dev.chi_qr
KAPPA_R = dev.kappa_r
CHI_AR = two_pi * 0.0093        # rad/us, storage-readout cross-Kerr (assumption, see docstring)
CHI_FR = 0.0                    # rad/us, f readout shift (assumption: none, as DGRAPE.model)
N_CAP = 10.0                    # default cap on the peak photon number
LEVELS = (0, 1, 2)              # g, e, f


def level_shift(j, chi_fr=CHI_FR):
    return (0.0, CHI_QR, chi_fr)[j]


def detunings(n_c=1, detuning_mhz=0.0, chi_ar=CHI_AR, chi_fr=CHI_FR):
    """Delta_{j,n} (3, n_c) in rad/us for the drive at the g/e midpoint + detuning_mhz."""
    base = 0.5 * CHI_QR - two_pi * detuning_mhz
    n = np.arange(n_c)
    return np.array([[base - level_shift(j, chi_fr) - chi_ar * k for k in n] for j in LEVELS])


def midpoint_amplitude(nbar):
    """|eps| that holds nbar photons in steady state at the midpoint (either level)."""
    return np.sqrt(nbar) * abs(0.5 * KAPPA_R + 0.5j * CHI_QR)


# ---------------------------------------------------------------------------
# Pointer integration (exact for piecewise-constant eps)
# ---------------------------------------------------------------------------

def _step_coeffs(delta, h):
    lam = 0.5 * KAPPA_R + 1j * np.asarray(delta)
    e = np.exp(-lam * h)
    return e, (1.0 - e) / lam


def pointers(eps, delta, alpha0=None, dt=DT):
    """alpha at the grid points (N + 1, *delta.shape) and at the step midpoints (N, *delta.shape) for the drive
    eps (N,) complex and detunings delta (any shape). alpha0: start (default empty)."""
    eps = np.asarray(eps, dtype=complex)
    delta = np.asarray(delta, dtype=float)
    a = np.zeros(delta.shape, complex) if alpha0 is None else np.array(alpha0, complex) * np.ones(delta.shape)
    e1, g1 = _step_coeffs(delta, dt)
    eh, gh = _step_coeffs(delta, 0.5 * dt)
    grid = np.empty((eps.size + 1,) + delta.shape, complex)
    mid = np.empty((eps.size,) + delta.shape, complex)
    grid[0] = a
    for k, ek in enumerate(eps):
        mid[k] = a * eh - 1j * ek * gh
        a = a * e1 - 1j * ek * g1
        grid[k + 1] = a
    return grid, mid


# ---------------------------------------------------------------------------
# Drive shapes ("composition")
# ---------------------------------------------------------------------------

@dataclass
class DriveSpec:
    """A readout drive. shape:
        square  amp for t_drive
        kick    kick * amp for t_kick, then amp up to t_drive (faster ring-up)
        clear   square, then two depletion segments of lengths t_dep1, t_dep2 whose complex amplitudes are solved so
                that alpha_g = alpha_e = 0 at the end (McClure et al. 2016's CLEAR, exact for the linear model)
        free    t_drive split into len(segments) equal pieces, amplitudes amp * segments[i] (complex); with t_dep1 and
                t_dep2 > 0 the CLEAR depletion is added after it, as for "clear"
    amp = midpoint_amplitude(nbar). Every time in us, rounded to the DT grid. kick and clear combine (a kick inside a
    clear drive)."""
    shape: str = "square"
    nbar: float = 5.0
    t_drive: float = 1.0
    kick: float = 1.0
    t_kick: float = 0.0
    t_dep1: float = 0.0
    t_dep2: float = 0.0
    segments: tuple = ()
    detuning_mhz: float = 0.0

    def to_dict(self):
        d = dict(self.__dict__)
        d["segments"] = [[float(np.real(c)), float(np.imag(c))] for c in self.segments]
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["segments"] = tuple(complex(re, im) for re, im in d.get("segments", ()))
        return cls(**d)


def n_steps(t, dt=DT):
    return int(round(t / dt))


def depletion_solve(eps_main, n1, n2, delta_ge, dt=DT):
    """(c1, c2): amplitudes of two depletion segments (n1, n2 steps) after eps_main that empty the g and e pointers
    (delta_ge = (Delta_g, Delta_e)) at the end. alpha is linear in eps, so this is a 2 x 2 linear solve."""
    if n1 < 1 or n2 < 1:
        raise ValueError("CLEAR needs two depletion segments of at least one step each")
    delta_ge = np.asarray(delta_ge, float)
    end = lambda eps, a0: pointers(eps, delta_ge, a0, dt)[0][-1]  # noqa: E731
    a_main = pointers(eps_main, delta_ge, None, dt)[0][-1]
    A = end(np.zeros(n1 + n2), a_main)                                  # free ring-down of the main part
    B1 = end(np.r_[np.ones(n1), np.zeros(n2)], np.zeros(2))             # response to unit segment 1
    B2 = end(np.r_[np.zeros(n1), np.ones(n2)], np.zeros(2))
    c = np.linalg.solve(np.stack([B1, B2], axis=1), -A)
    return complex(c[0]), complex(c[1])


def build_drive(spec, dt=DT, chi_fr=CHI_FR):
    """eps (N,) complex on the DT grid, N = drive end (depletion included)."""
    amp = midpoint_amplitude(spec.nbar)
    nd = n_steps(spec.t_drive, dt)
    if spec.shape == "free":
        segs = np.asarray(spec.segments, complex)
        if segs.size == 0:
            raise ValueError("shape 'free' needs segments")
        idx = np.minimum((np.arange(nd) * segs.size) // max(nd, 1), segs.size - 1)
        eps = amp * segs[idx]
    else:
        eps = np.full(nd, amp, complex)
        nk = min(n_steps(spec.t_kick, dt), nd)
        if spec.kick != 1.0 and nk > 0:
            eps[:nk] *= spec.kick
    if spec.shape == "clear" or (spec.shape == "free" and spec.t_dep1 > 0 and spec.t_dep2 > 0):
        n1, n2 = n_steps(spec.t_dep1, dt), n_steps(spec.t_dep2, dt)
        dge = detunings(1, spec.detuning_mhz, 0.0, chi_fr)[:2, 0]
        c1, c2 = depletion_solve(eps, n1, n2, dge, dt)
        eps = np.r_[eps, np.full(n1, c1), np.full(n2, c2)]
    elif spec.shape not in ("square", "kick", "free"):
        raise ValueError(f"unknown drive shape {spec.shape!r}")
    return eps


# ---------------------------------------------------------------------------
# Record and decision
# ---------------------------------------------------------------------------

def matched_weights(mid_g, mid_e, kind="matched"):
    """Integration weights on the integration steps from the no-jump midpoint pointers (n = 0)."""
    dalpha = np.asarray(mid_e) - np.asarray(mid_g)
    if kind == "matched":
        return dalpha
    if kind == "boxcar":
        ph = np.sum(dalpha)
        return np.full(dalpha.shape, ph / abs(ph) if abs(ph) > 0 else 1.0)
    raise ValueError(f"weights must be 'matched' or 'boxcar', not {kind!r}")


def unit_mean(w, alpha_mid, dt=DT):
    """s1 = mu / (sqrt(eta) sigma) for pointer midpoints alpha_mid (n_int, ...) and weights w (n_int,)."""
    sigma = np.sqrt(np.sum(np.abs(w) ** 2) * dt)
    if sigma == 0:
        return np.zeros(np.shape(alpha_mid)[1:])
    mu = 2.0 * np.sqrt(KAPPA_R) * np.real(np.tensordot(w.conj(), alpha_mid, axes=(0, 0))) * dt
    return mu / sigma


def p_excited(s1, s1_g, s1_e, eta, bias=0.0):
    """P(decide e) for trajectories with unit means s1 (any shape)."""
    s1 = np.asarray(s1, dtype=float)
    mid = 0.5 * (s1_g + s1_e)
    if np.isinf(eta):
        return np.where(s1 > mid, 1.0, np.where(s1 < mid, 0.0, 0.5))
    r = np.sqrt(eta)
    return 0.5 * erfc((r * mid + bias - r * s1) / np.sqrt(2.0))


def snr(s1_g, s1_e, eta):
    return np.sqrt(eta) * abs(s1_e - s1_g)


@dataclass
class Record:
    """No-jump and one-jump unit means of the record over [0, t_int]."""
    s1_level: np.ndarray                  # (3,)  no-jump, per level g, e, f
    s1_jump: np.ndarray                   # (C, B) jump c at bin centre b
    transitions: list                     # [(j, j')] per jump type c
    bin_centres: np.ndarray               # (B,) step index of each bin centre
    bin_widths: np.ndarray                # (B,) us
    weights: np.ndarray = field(repr=False, default=None)

    def p_e_level(self, eta, bias=0.0):
        return p_excited(self.s1_level, self.s1_level[0], self.s1_level[1], eta, bias)

    def p_e_jump(self, eta, bias=0.0):
        return p_excited(self.s1_jump, self.s1_level[0], self.s1_level[1], eta, bias)


def bins(n_int, bin_steps):
    """(centres, widths in steps) of the jump-time bins over [0, n_int)."""
    starts = np.arange(0, n_int, bin_steps)
    ends = np.minimum(starts + bin_steps, n_int)
    return starts + (ends - starts) // 2, ends - starts


def record(eps_full, n_int, transitions, bin_steps=10, detuning_mhz=0.0, weights="matched", dt=DT, chi_fr=CHI_FR):
    """Record of the integration window [0, n_int steps) for the drive eps_full (covering at least n_int steps)."""
    eps = np.asarray(eps_full, complex)[:n_int]
    if eps.size < n_int:
        eps = np.r_[eps, np.zeros(n_int - eps.size)]
    dl = detunings(1, detuning_mhz, 0.0, chi_fr)[:, 0]                  # (3,)
    grid, mid = pointers(eps, dl, None, dt)
    centres, widths = bins(n_int, bin_steps)
    if n_int == 0:
        return Record(np.zeros(3), np.zeros((len(transitions), 0)), list(transitions), centres, widths * dt, None)
    w = matched_weights(mid[:, 0], mid[:, 1], weights)
    s_level = unit_mean(w, mid, dt)
    # one jump j -> j' at the start of step centres[b]: pointer j before, j' after
    C, B = len(transitions), centres.size
    a = np.zeros((C, B), complex)
    mids = np.empty((n_int, C, B), complex)
    d_new = np.array([dl[jp] for _, jp in transitions], dtype=float).reshape(C, 1) * np.ones((1, B))
    e1, g1 = _step_coeffs(d_new, dt)
    eh, gh = _step_coeffs(d_new, 0.5 * dt)
    j_old = np.array([j for j, _ in transitions], dtype=int)
    for k in range(n_int):
        before = centres[None, :] > k                                    # still on the old level
        start = centres[None, :] == k
        a = np.where(start, grid[k][j_old][:, None], a)
        m_new = a * eh - 1j * eps[k] * gh
        mids[k] = np.where(before, mid[k][j_old][:, None], m_new)
        a = a * e1 - 1j * eps[k] * g1
    s_jump = unit_mean(w, mids, dt)
    return Record(s_level, s_jump, list(transitions), centres, widths * dt, w)


def ring_down_time(nbar_now, target):
    """Time for a free (undriven) pointer to decay from nbar_now to target photons: |alpha|^2 decays at kappa_r."""
    if nbar_now <= target:
        return 0.0
    return float(np.log(nbar_now / target) / KAPPA_R)
