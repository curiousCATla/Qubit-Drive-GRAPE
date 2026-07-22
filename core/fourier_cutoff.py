#!/usr/bin/env python3
"""
fourier_cutoff.py

Hard frequency cutoff for GRAPE control pulses, implementing the
reparametrization of Heeres et al. 2017, Supplementary Eq. (22):

    maximize F(eps)  subject to  eps_tilde(omega) = 0  for  omega < w_min or omega > w_max

We enforce this as an ORTHOGONAL PROJECTION P onto the band-limited
subspace, applied inside the objective:

    physical pulse  u = P(x)          (x = raw L-BFGS-B variable)
    gradient        dF/dx = P(dF/du)  (P is self-adjoint & idempotent)

The projection acts on the COMPLEX drives eps_C = C_I + i*C_Q and
eps_T = T_I + i*T_Q, so the allowed band may be asymmetric about zero
(the physical drive spectrum generally is). Frequencies are ordinary
frequency in MHz (NOT angular), matching Supplementary Fig. 4's axis.

Key mathematical facts (why the code is this short):
  * P = IFFT . mask . FFT  is idempotent  -> it is a true projection.
  * With a real 0/1 mask, P is self-adjoint under the real inner product,
    so the chain-ruled gradient is just P applied to the fidelity gradient.
    => `project_bandlimit` is used for BOTH the controls and the gradient.
"""

import numpy as np


def make_band_mask(N, dt, f_lo, f_hi):
    """
    Boolean mask over the N FFT bins of a COMPLEX signal, True where
    f_lo <= f <= f_hi.

    Parameters
    ----------
    N : int
        Number of time samples.
    dt : float
        Time step in microseconds (your pulses use dt = 0.002 us).
        Then np.fft.fftfreq(N, d=dt) is in cycles/us = MHz directly.
    f_lo, f_hi : float
        Band edges in MHz (ordinary frequency). For a symmetric low-pass
        at cutoff f_c, use f_lo = -f_c, f_hi = +f_c.

    Returns
    -------
    mask : (N,) bool array, aligned with np.fft.fft bin ordering.
    """
    f = np.fft.fftfreq(N, d=dt)          # two-sided, MHz
    return (f >= f_lo) & (f <= f_hi)


def project_bandlimit(u, dt, cav_band, tra_band):
    """
    Apply the hard frequency cutoff (Eq. 22) as an orthogonal projection.

    Because P is self-adjoint and idempotent, calling this on the GRADIENT
    array (same shape (N,4)) yields the correctly chain-ruled gradient.
    That is the whole trick: use this one function for controls AND grad.

    Parameters
    ----------
    u : (N, 4) real array, columns [C_I, C_Q, T_I, T_Q].
    dt : float, time step in microseconds.
    cav_band : (f_lo, f_hi) tuple in MHz, applied to eps_C = C_I + i C_Q.
    tra_band : (f_lo, f_hi) tuple in MHz, applied to eps_T = T_I + i T_Q.

    Returns
    -------
    u_proj : (N, 4) real array, provably zero spectral content outside
             the two bands (up to floating-point).
    """
    N = u.shape[0]
    out = np.empty_like(u)

    mC = make_band_mask(N, dt, *cav_band)
    zC = np.fft.ifft(np.fft.fft(u[:, 0] + 1j * u[:, 1]) * mC)
    out[:, 0] = zC.real
    out[:, 1] = zC.imag

    mT = make_band_mask(N, dt, *tra_band)
    zT = np.fft.ifft(np.fft.fft(u[:, 2] + 1j * u[:, 3]) * mT)
    out[:, 2] = zT.real
    out[:, 3] = zT.imag

    return out


# ----------------------------------------------------------------------
# Verification helpers -- run these once on a pulse before trusting anything
# ----------------------------------------------------------------------

def out_of_band_energy_fraction(u, dt, cav_band, tra_band):
    """
    Fraction of each complex drive's spectral energy lying OUTSIDE its band.
    For a freshly projected pulse this should be ~1e-30 (float noise).
    For an unprojected optimized pulse it tells you how much the paper's
    "99% within 33/27 MHz" statement is (or isn't) already satisfied.
    """
    N = u.shape[0]
    def frac(I, Q, band):
        Z = np.fft.fft(I + 1j * Q)
        m = make_band_mask(N, dt, *band)
        tot = np.sum(np.abs(Z) ** 2)
        out = np.sum(np.abs(Z[~m]) ** 2)
        return out / tot if tot > 0 else 0.0
    return {
        'cavity':   frac(u[:, 0], u[:, 1], cav_band),
        'transmon': frac(u[:, 2], u[:, 3], tra_band),
    }


def check_projection_properties(u, dt, cav_band, tra_band):
    """
    Sanity checks you should run once:
      1. Idempotency:  P(P(u)) == P(u)   (defining property of a projection)
      2. Self-adjointness: <a, P b> == <P a, b>  for random a, b
         (this is what makes 'apply P to the gradient' the correct chain rule)
    Prints the residuals; both should be ~1e-12 or smaller.
    """
    Pu  = project_bandlimit(u, dt, cav_band, tra_band)
    PPu = project_bandlimit(Pu, dt, cav_band, tra_band)
    idem = np.max(np.abs(PPu - Pu))

    rng = np.random.default_rng(0)
    a = rng.standard_normal(u.shape)
    b = rng.standard_normal(u.shape)
    Pa = project_bandlimit(a, dt, cav_band, tra_band)
    Pb = project_bandlimit(b, dt, cav_band, tra_band)
    lhs = np.sum(a * Pb)     # <a, P b>
    rhs = np.sum(Pa * b)     # <P a, b>
    adj = abs(lhs - rhs)

    print(f"idempotency  max|P(Pu) - Pu| = {idem:.3e}   (want ~0)")
    print(f"self-adjoint |<a,Pb>-<Pa,b>| = {adj:.3e}   (want ~0)")
    return idem, adj


if __name__ == "__main__":
    # Quick self-test on a random pulse
    N, dt = 550, 0.002
    cav_band = (-27.0, 27.0)
    tra_band = (-33.0, 33.0)
    u = np.random.default_rng(1).standard_normal((N, 4))
    u_I = np.load("pulses/u_I_refined_t3v2.npy")

    check_projection_properties(u_I, dt, cav_band, tra_band)
    up = project_bandlimit(u_I, dt, cav_band, tra_band)
    print("out-of-band energy after projection:",
          out_of_band_energy_fraction(up, dt, cav_band, tra_band))
