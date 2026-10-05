"""
Wigner function of a Fock-basis density matrix, for PRX Fig. 2(d) (aqec_experiments.ipynb §10.5).

Closed form (Cahill & Glauber), exact for any finite n_c -- no displacement operator, so
no truncation error from displacing a truncated state:

    W(alpha) = (2/pi) e^{-2|alpha|^2} [ sum_m rho_mm (-1)^m L_m(4|alpha|^2)
               + 2 Re sum_{m<n} rho_mn (-1)^m sqrt(m!/n!) (2 alpha)^{n-m} L_m^{n-m}(4|alpha|^2) ]

alpha is the phase-space coordinate itself (axes Re alpha, Im alpha, as in the paper),
and W integrates to 1 over d^2 alpha = d(Re alpha) d(Im alpha). The paper's colour bar is
the displaced parity (pi/2) W in [-1, 1], returned by parity().
"""

import numpy as np
from scipy.special import eval_genlaguerre, gammaln


def wigner(rho, xvec, yvec):
    """W(alpha) on the grid alpha = x + i y; returns shape (len(yvec), len(xvec))."""
    rho = np.asarray(rho)
    X, Y = np.meshgrid(np.asarray(xvec, float), np.asarray(yvec, float))
    alpha = X + 1j * Y
    r2 = 4.0 * np.abs(alpha) ** 2
    W = np.zeros(alpha.shape)
    for m in range(rho.shape[0]):
        if rho[m, m] != 0:
            W += np.real(rho[m, m]) * (-1) ** m * eval_genlaguerre(m, 0, r2)
        for n in range(m + 1, rho.shape[0]):
            if rho[m, n] == 0:
                continue
            k = n - m
            coef = (-1) ** m * np.exp(0.5 * (gammaln(m + 1) - gammaln(n + 1)))
            W += 2.0 * np.real(rho[m, n] * coef * (2.0 * alpha) ** k * eval_genlaguerre(m, k, r2))
    return (2.0 / np.pi) * np.exp(-0.5 * r2) * W


def parity(rho, xvec, yvec):
    """Displaced photon-number parity <D(alpha) P D(alpha)'> = (pi/2) W(alpha), in [-1, 1]."""
    return 0.5 * np.pi * wigner(rho, xvec, yvec)
