"""
Device parameters and Hamiltonian for the error-semitransparent (EsT) gate
replication -- Roy, Wetherbee & Fatemi, arXiv:2603.15356, Table I.

This is a DIFFERENT chip from the one hard-coded in core/grape_core.py:12-17
(chi/2pi = -2.194 MHz etc., the Heeres-style cat-code device). Those globals are
deliberately left alone: every pulse in pulses/ and the whole validation//analysis/
layer is scored against them, so this module builds its own Hamiltonian rather
than mutating them.

Two differences from core.grape_core.make_hamiltonian beyond the constants:

  1. A second-order cavity self-Kerr term K' is present here and absent there.
  2. n_t >= 3 is required, not optional -- K_q/2pi = -180 MHz is a headline
     parameter of this device, and core/grape_core.py:614 silently drops the
     anharmonicity term at n_t = 2.

Units follow the repo convention (core/grape_core.py:12-17): constants are
written in lab units (MHz, us) and pre-multiplied by two_pi, so the stored
values are angular frequency in rad/us and `dt` is in us.

ASSUMPTIONS FLAGGED FOR CHECKING AGAINST THE PAPER
--------------------------------------------------
* K' normal-ordered form. Table I gives a magnitude, not an operator. This
  module uses (K'/6) a^dag^3 a^3, the next term in the normal-ordered expansion
  after (K/2) a^dag^2 a^2. If the paper defines it with a different combinatorial
  prefactor, the K' contribution rescales by a constant -- it is ~600 Hz on a
  n<=4 code, so this is a small absolute effect either way.
* Gaussian ramp shape. App. C specifies "48 ns Gaussian rise/fall" without a
  sigma convention. `ramp_envelope` uses sigma = T_ramp/2, pedestal-subtracted so
  the envelope is exactly 0 at both endpoints and exactly 1 across the flat top.

Neither assumption affects the EsT-vs-Ord *comparison* that is the replication
target, since both variants are trained and scored under identical settings.
"""

import numpy as np

two_pi = 2 * np.pi

# ---------------------------------------------------------------------------
# Table I -- Hamiltonian parameters (lab units in the comments, rad/us stored)
# ---------------------------------------------------------------------------

chi = two_pi * (-3.66)       # MHz    dispersive qubit-cavity coupling
K_a = two_pi * (-0.022)      # MHz    cavity self-Kerr            (-22 kHz)
K_p = two_pi * (0.000590)    # MHz    2nd-order cavity self-Kerr  (590 Hz)
chi_p = two_pi * (0.039)     # MHz    2nd-order dispersive shift  (39 kHz)
K_q = two_pi * (-180.0)      # MHz    qubit (transmon) anharmonicity

# ---------------------------------------------------------------------------
# Table I -- coherence times (us). Not used by the lossless unitary sims that
# produce Fig. 1d-f; recorded here so the decoherence study has one source.
# ---------------------------------------------------------------------------

T1_CAVITY = 180.0
T2_CAVITY = 290.0
T1_QUBIT = 70.0
T2_RAMSEY_QUBIT = 30.0
T2_ECHO_QUBIT = 84.0

# ---------------------------------------------------------------------------
# App. C -- pulse constraints
# ---------------------------------------------------------------------------

DT = 0.001                   # us, 1 ns piecewise-constant steps
BAND_MHZ = (-50.0, 50.0)     # single hard cutoff at 50 MHz, both drives.
                             # Ordinary (not angular) frequency, matching
                             # core.fourier_cutoff.make_band_mask, which reads
                             # np.fft.fftfreq(N, d=dt) directly.
EPS_MAX = two_pi * 4.0       # rad/us. Table I gives Omega_max/2pi = eps_max/2pi
                             # = 4 MHz. With the repo's control convention
                             # Hc = [A+Ad, 1j*(A-Ad), ...] the physical drive is
                             # eps = u[0] - 1j*u[1], so this caps the QUADRATURE
                             # PAIR NORM sqrt(u0^2 + u1^2), not each column
                             # separately -- see EST/grape_jax.amplitude_cost.
RAMP_NS = 48.0               # ns Gaussian rise and fall on the envelope

GATE_DURATION_US = {         # App. C
    "X": 1.0,
    "H": 1.0,
    "T": 0.6,
}

N_T = 3                      # transmon levels; >= 3 so K_q is present at all
TRUNC_LIST = [16, 20, 24]    # cavity truncations trained jointly (Heeres Eq. 23)


def make_hamiltonian_est(n_t, n_c):
    """
    Drift and control Hamiltonians for this device.

    Mirrors core.grape_core.make_hamiltonian's interface and conventions exactly
    -- joint index |t,c> = t*n_c + c, and Hc in the canonical column order
    [C_I, C_Q, T_I, T_Q] -- so every downstream shape/ordering assumption in the
    repo (fourier_cutoff, propagator, the (N,4) pulse layout) carries over.

    Parameters
    ----------
    n_t : int
        Transmon levels. Must be >= 3; see module docstring.
    n_c : int
        Cavity levels.

    Returns
    -------
    H0 : (n, n) complex, n = n_t * n_c
    Hc : list of four (n, n) complex arrays
    """
    if n_t < 3:
        raise ValueError(
            f"n_t={n_t} < 3 would drop the K_q anharmonicity term, which is a "
            "headline parameter of this device (K_q/2pi = -180 MHz). Use n_t >= 3."
        )

    from core.grape_core import make_ops   # pure, reads no module constants

    A, B = make_ops(n_t, n_c)
    Ad, Bd = A.conj().T, B.conj().T
    nA, nB = Ad @ A, Bd @ B

    a2 = Ad @ Ad @ A @ A          # a^dag^2 a^2
    a3 = Ad @ Ad @ Ad @ A @ A @ A  # a^dag^3 a^3

    H0 = (chi * (nA @ nB)
          + (K_a / 2) * a2
          + (K_p / 6) * a3
          + (chi_p / 2) * (nB @ a2)
          + (K_q / 2) * (Bd @ Bd @ B @ B))

    Hc = [A + Ad, 1j * (A - Ad), B + Bd, 1j * (B - Bd)]
    return H0, Hc


def ramp_envelope(N, dt, ramp_ns=RAMP_NS):
    """
    Gaussian rise/fall envelope, (N,) real in [0, 1].

    Pedestal-subtracted so the underlying continuous envelope is exactly 0 at
    t = 0 and t = T, and exactly 1 across the flat top. Samples are taken at the
    MIDPOINT of each piecewise-constant step, which is the honest convention for
    a sampled-and-held drive but means the first and last steps sit at a small
    nonzero value (~0.7% of full scale at N=1000, dt=1 ns) rather than at 0.

    Multiplying a pulse by this envelope therefore makes
    core.grape_core.boundary_penalty near-redundant: the boundary condition is
    enforced structurally by the envelope shape rather than by a penalty term.

    Parameters
    ----------
    N : int          number of time steps
    dt : float       step size in us
    ramp_ns : float  rise/fall duration in ns

    Returns
    -------
    env : (N,) float64
    """
    t_ramp = ramp_ns * 1e-3          # ns -> us
    t = (np.arange(N) + 0.5) * dt    # midpoint of each piecewise-constant step
    T = N * dt

    if 2 * t_ramp >= T:
        raise ValueError(
            f"ramp_ns={ramp_ns} ns leaves no flat top in a {T*1e3:.1f} ns pulse."
        )

    sigma = t_ramp / 2.0
    pedestal = np.exp(-t_ramp ** 2 / (2 * sigma ** 2))   # value of the raw Gaussian at t=0

    def lifted(x):
        """Raw Gaussian shifted to peak at x = t_ramp, rescaled to hit 0 at x = 0."""
        g = np.exp(-(x - t_ramp) ** 2 / (2 * sigma ** 2))
        return (g - pedestal) / (1.0 - pedestal)

    env = np.ones(N)
    rise = t < t_ramp
    fall = t > (T - t_ramp)
    env[rise] = lifted(t[rise])
    env[fall] = lifted(T - t[fall])
    return np.clip(env, 0.0, 1.0)


def band_mask(N, dt=DT, band=BAND_MHZ):
    """
    (N,) bool FFT mask for the 50 MHz cutoff, built by the repo's own
    core.fourier_cutoff.make_band_mask so the training-time constraint and the
    post-hoc check in core.fourier_cutoff.out_of_band_energy_fraction cannot
    drift apart.
    """
    from core.fourier_cutoff import make_band_mask
    return make_band_mask(N, dt, *band)


def n_steps(gate, dt=DT):
    """Number of piecewise-constant steps for a named gate."""
    return int(round(GATE_DURATION_US[gate] / dt))
