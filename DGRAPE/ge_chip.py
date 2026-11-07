"""
Chips for the Gertler-comb tails (FWM_tails_ge_chips.ipynb). Track 7 ran on Shirol's chip only
(AQEC/device.py). This module lets the g -> e tail, and the gate before it, run on a different chip:
the dispersive shifts chi_ge (storage-transmon) and chi_qr (readout-transmon) are the scan axes, and the
terms that grow with them are derived from them. Everything else (alpha, kappa_r, every T1/T2, the
thermal populations, the code) stays at Shirol's Table I.

THE DERIVED TERMS (a transmon of fixed anharmonicity; r = chi_ge / chi_ge_Shirol):
  K        storage self-Kerr          K_Shirol r^2      (K ~ chi^2 / 4 alpha: both scale as phi_a^4)
  chi_q'   6th-order storage shift    chi_q'_Shirol r^2 (~ phi_a^4 phi_q^2, so ~ K at a fixed transmon)
  chi_gf   storage shift of |f>       chi_gf_Shirol r   (keeps chi_ef / chi_ge; the 3-level gate only)
  chi_ar   storage-readout cross-Kerr chi_ge chi_qr / (2 alpha)  (Gertler 2021: 9.1 kHz by this rule
           against the 7 kHz he derived). A NEW drift term -chi_ar n_a n_r: it moves the storage
           frequency only while a readout photon is present, i.e. on the repaired branch.
  cap_T    transmon-comb cap          CAP_T_SHIROL r    (the same cap / chi ratio as track 7)
  band     both combs, MHz            +-max(12, 5 chi_ge + 6) (teeth reach 4 chi_ge, plus the pump Stark shift)
These are modelling assumptions, stated in the notebook; they are not measured values.

SHIROL is the chip of track 7 exactly (chi_ar = 0, the measured K and chi_q'), so chip=SHIROL or None
reproduces every track-7 number bit for bit. derived(1.12, 1.13) differs from it only by chi_ar
(4.7 kHz) and is the anchor of the scan.

Constants are angular frequencies in rad/us (AQEC/device.py convention), positive with explicit minus
signs in the drift (DGRAPE/model.py).
"""

import math
from dataclasses import dataclass

import numpy as np

from AQEC import device as dev
from DGRAPE.model import N_Q, dim, drift_hamiltonian, index

TWO_PI = 2 * np.pi
CHI_GE_SHIROL_MHZ = 1.12
CAP_T_SHIROL = TWO_PI * 1.0         # = fwm_tail_ge.CAP_T (tested)
BAND_SHIROL = (-12.0, 12.0)         # = fwm_tail_ge.BAND (tested)


@dataclass(frozen=True)
class Chip:
    """One chip for the g -> e tail. Frequencies in rad/us; band in MHz."""
    chi_ge: float
    chi_qr: float
    chi_gf: float
    alpha: float
    K: float
    chi_q_p: float
    chi_ar: float
    cap_T: float
    band: tuple
    name: str

    @property
    def chi_ge_MHz(self):
        return self.chi_ge / TWO_PI

    @property
    def chi_qr_MHz(self):
        return self.chi_qr / TWO_PI

    @property
    def g_m(self):
        """a' r' q coupling per unit xi_M, rad/us: sqrt(chi_ge chi_qr) (the direct cosine count)."""
        return np.sqrt(self.chi_ge * self.chi_qr)

    @property
    def tag(self):
        """File-name tag, e.g. ch300x500 for chi_ge 3.00, chi_qr 5.00 MHz."""
        return f"ch{int(round(100 * self.chi_ge_MHz)):03d}x{int(round(100 * self.chi_qr_MHz)):03d}"

    def table_row(self):
        """Lab units: MHz for the shifts and alpha, kHz for the small terms."""
        return {"chip": self.name, "chi_ge_MHz": self.chi_ge_MHz, "chi_qr_MHz": self.chi_qr_MHz,
                "chi_gf_MHz": self.chi_gf / TWO_PI, "alpha_MHz": self.alpha / TWO_PI,
                "K_kHz": 1e3 * self.K / TWO_PI, "chi_q_p_kHz": 1e3 * self.chi_q_p / TWO_PI,
                "chi_ar_kHz": 1e3 * self.chi_ar / TWO_PI, "g_m_MHz": self.g_m / TWO_PI,
                "cap_T_MHz": self.cap_T / TWO_PI, "band_MHz": self.band[1]}


SHIROL = Chip(chi_ge=dev.chi_ge, chi_qr=dev.chi_qr, chi_gf=dev.chi_gf, alpha=dev.alpha_q, K=dev.K_a,
              chi_q_p=dev.chi_q_p, chi_ar=0.0, cap_T=CAP_T_SHIROL, band=BAND_SHIROL, name="Shirol")


def derived(chi_ge_MHz, chi_qr_MHz, name=None):
    """The chip with these dispersive shifts (MHz) and every other term derived (module docstring)."""
    r = chi_ge_MHz / CHI_GE_SHIROL_MHZ
    chi_ge, chi_qr = TWO_PI * chi_ge_MHz, TWO_PI * chi_qr_MHz
    half = max(12.0, 5.0 * chi_ge_MHz + 6.0)
    return Chip(chi_ge=chi_ge, chi_qr=chi_qr, chi_gf=dev.chi_gf * r, alpha=dev.alpha_q,
                K=dev.K_a * r ** 2, chi_q_p=dev.chi_q_p * r ** 2,
                chi_ar=chi_ge * chi_qr / (2.0 * dev.alpha_q), cap_T=CAP_T_SHIROL * r,
                band=(-half, half), name=name or f"chi {chi_ge_MHz:g}/{chi_qr_MHz:g}")


def resolve(chip):
    """None -> SHIROL; a Chip as is; a (chi_ge_MHz, chi_qr_MHz) pair -> derived(...)."""
    if chip is None:
        return SHIROL
    if isinstance(chip, Chip):
        return chip
    ge, qr = chip
    return derived(float(ge), float(qr))


def drift(chip, n_c, n_r=2, include_higher_order=True):
    """H_d (d, d) of the full (g, e, f) DGRAPE space on this chip: DGRAPE.model.drift_hamiltonian with
    the chip's constants, plus -chi_ar n_a n_r."""
    ch = resolve(chip)
    H = drift_hamiltonian(n_c, n_r, include_higher_order, alpha=ch.alpha, chi_ge=ch.chi_ge,
                          chi_gf=ch.chi_gf, chi_qr=ch.chi_qr, K=ch.K, chi_q_p=ch.chi_q_p)
    if ch.chi_ar != 0.0:
        d = np.real(np.diag(H)).copy()
        for n in range(n_c):
            for j in range(N_Q):
                for m in range(n_r):
                    d[index(n, j, m, n_c, n_r)] -= ch.chi_ar * n * m
        H = np.diag(d).astype(complex)
    return H


def apply_chip(model, chip, include_higher_order=True):
    """Put the chip's drift into a built (g, e, f) DGRAPE Model, in place, and return it. Hc and the
    jump operators do not depend on the chip (fixed rates). chip None/SHIROL leaves the model as built."""
    ch = resolve(chip)
    if ch is SHIROL:
        return model
    if model.d != dim(model.n_c, model.n_r):
        raise ValueError("apply_chip needs the full three-level model (apply it before restricting)")
    model.H_d = drift(ch, model.n_c, model.n_r, include_higher_order)
    hd = np.real(np.diag(model.H_d))
    model.shift = 0.5 * (hd.max() + hd.min())
    model.H_d_shifted = model.H_d - model.shift * np.eye(model.d)
    model.half_width_d = 0.5 * (hd.max() - hd.min())
    return model


def repair_time_estimate(chip, xi=0.2, n=2):
    """us: the back-of-envelope shortest repair, a selective pi pulse (square, Rabi zero on the code
    states chi_ge away: sqrt(3)/(2 chi_ge)) then a full swap on path n at |xi|."""
    ch = resolve(chip)
    t_sel = math.sqrt(3) / (2 * ch.chi_ge_MHz)
    t_swap = math.pi / (2 * ch.g_m * abs(xi) * math.sqrt(n + 1))
    return t_sel, t_swap
