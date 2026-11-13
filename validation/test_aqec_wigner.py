#!/usr/bin/env python3
"""
test_aqec_wigner.py

Checks for AQEC/wigner.py, the closed-form Wigner function behind Fig. 2(d) of
notebooks/aqec_experiments.ipynb (§10.5):

  * qutip       -- matches qutip.wigner (with g=2, so qutip's x + iy is alpha itself)
  * normalised  -- integrates to 1 over d^2 alpha
  * origin      -- W(0) = (2/pi) sum_n (-1)^n rho_nn (the parity of rho)
  * coherent    -- a coherent state |beta> peaks at alpha = beta (pins the alpha vs
                   alpha* convention, which a Hermitian-symmetric check would miss)

Run:
    python3 validation/test_aqec_wigner.py [-v]
"""

import os
import sys
import unittest

import numpy as np
from scipy.special import gammaln

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from AQEC.wigner import parity, wigner          # noqa: E402

try:
    import qutip
    HAVE_QUTIP = True
except ImportError:
    HAVE_QUTIP = False


def random_rho(n, seed):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = M @ M.conj().T
    return rho / np.trace(rho)


class WignerTest(unittest.TestCase):
    @unittest.skipUnless(HAVE_QUTIP, "qutip not installed")
    def test_matches_qutip(self):
        x = np.linspace(-2.5, 2.5, 41)
        y = np.linspace(-2.0, 2.2, 37)
        for seed in range(3):
            rho = random_rho(8, seed)
            ref = qutip.wigner(qutip.Qobj(rho), x, y, g=2)
            np.testing.assert_allclose(wigner(rho, x, y), ref, atol=1e-10)

    def test_normalised(self):
        x = np.linspace(-6, 6, 601)
        W = wigner(random_rho(7, 4), x, x)
        self.assertAlmostEqual(W.sum() * (x[1] - x[0]) ** 2, 1.0, places=6)

    def test_origin_is_parity(self):
        rho = random_rho(7, 5)
        expected = np.sum((-1) ** np.arange(7) * np.real(np.diag(rho)))
        self.assertAlmostEqual(parity(rho, [0.0], [0.0])[0, 0], expected, places=12)

    def test_coherent_state_peaks_at_beta(self):
        beta, n_c = 0.8 - 0.5j, 25
        k = np.arange(n_c)
        c = np.exp(-0.5 * abs(beta) ** 2 + k * np.log(beta + 0j) - 0.5 * gammaln(k + 1))
        rho = np.outer(c, c.conj())
        x = np.linspace(-2, 2, 81)
        W = wigner(rho, x, x)
        iy, ix = np.unravel_index(np.argmax(W), W.shape)
        self.assertAlmostEqual(x[ix], beta.real, places=6)
        self.assertAlmostEqual(x[iy], beta.imag, places=6)


if __name__ == "__main__":
    unittest.main()
