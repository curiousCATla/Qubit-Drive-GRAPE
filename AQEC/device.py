"""
Device parameters for the passive (continuous) AQEC reconstruction -- Shirol,
van Geldern, Xi & Wang, "Passive Quantum Error Correction of Photon Loss at
Breakeven", PRX 16, 021042 (2026), Table I (system) and Table II (FWM drives).

This is a THIRD chip, distinct from both the Heeres-style cat-code device hard-coded
in core/grape_core.py and the EsT device in EST/device.py. Neither of those modules
is imported or mutated here; this module only defines its own constants.

Three modes: storage cavity a, transmon q (levels g, e, f), readout r. The readout
doubles as the low-Q reservoir that PReSPA dumps entropy into.

Units follow the repo convention (core/grape_core.py, EST/device.py): constants are
written in lab units (MHz, us) and multiplied by two_pi, so stored values are angular
frequencies in rad/us.

SIGN CONVENTION -- differs from EST/device.py
---------------------------------------------
Shirol writes every nonlinearity with an explicit minus sign and a POSITIVE
coefficient (Eq. 2):
    H0/hbar = w_a a'a + w_q q'q + w_r r'r - (alpha/2) q'^2 q^2
              - chi_ge |e><e| a'a - chi_gf |f><f| a'a - chi_qr |e><e| r'r
so alpha, chi_ge, chi_ef, chi_gf, chi_qr are all stored POSITIVE here. EST/device.py
instead stores signed coefficients with a plus sign (chi/2pi = -3.66 MHz,
K_q/2pi = -180 MHz). Do not mix the two without flipping signs.
"""

import numpy as np

two_pi = 2 * np.pi

# ---------------------------------------------------------------------------
# Table I -- mode frequencies (rad/us)
# ---------------------------------------------------------------------------

omega_q = two_pi * 3482.9    # MHz   transmon g-e frequency
omega_a = two_pi * 4657.9    # MHz   storage cavity frequency
omega_r = two_pi * 8725.0    # MHz   readout / reservoir frequency

# ---------------------------------------------------------------------------
# Table I -- nonlinearities and couplings (rad/us), all positive (see docstring)
# ---------------------------------------------------------------------------

alpha_q = two_pi * 134.28    # MHz   transmon anharmonicity
K_a = two_pi * 0.0033        # MHz   storage self-Kerr (3.3 kHz); omitted from Eq. 2
chi_ge = two_pi * 1.12       # MHz   storage-transmon g-e dispersive shift
chi_ef = two_pi * 0.95       # MHz   storage-transmon e-f dispersive shift
chi_q_p = two_pi * 0.0019    # MHz   6th-order storage-transmon shift chi'_q (1.9 kHz)
chi_qr = two_pi * 1.13       # MHz   readout-transmon dispersive shift
kappa_r = two_pi * 0.58      # MHz   readout (reservoir) energy decay rate

# Derived
chi_gf = chi_ge + chi_ef                 # 2.07 MHz: storage shift of |f> relative to |g>
omega_gf = 2 * omega_q - alpha_q         # g -> f two-photon transition frequency

# ---------------------------------------------------------------------------
# Table I -- coherence (us) and thermal populations
# ---------------------------------------------------------------------------

T1_GE = 50.0                 # transmon |e> -> |g>
T1_EF = 31.0                 # transmon |f> -> |e>
T2_RAMSEY = 53.0             # transmon g-e T2*
T2_ECHO = 70.0               # transmon g-e T2E
T2_GF_RAMSEY = 30.0          # transmon g-f T2*
P_E_THERMAL = 0.017          # transmon |e> equilibrium population
T1_CAVITY = 136.0            # storage |1> -> |0>
T2_CAVITY = 235.0            # storage T2
P_CAVITY_THERMAL = 0.006     # storage |1> equilibrium population

kappa_a = 1.0 / T1_CAVITY    # 1/us, single-photon loss rate

# ---------------------------------------------------------------------------
# Table II / Sec. III -- the two FWM combs as operated (rad/us)
# ---------------------------------------------------------------------------

CORRECTED_FOCK = (1, 3, 5)   # photon number n reached after addition (odd code words)

OMEGA_1 = two_pi * 0.055     # MHz   measured stage-1 rate |n-1,g,0> -> |n,f,0>
OMEGA_2 = two_pi * 0.160     # MHz   measured stage-2 rate |n,f,0>   -> |n,g,1>
STARK_1 = two_pi * 0.003     # MHz   transmon Stark shift, one comb-1 tone (n = 1)
STARK_2 = two_pi * 0.032     # MHz   transmon Stark shift, one comb-2 tone (n = 1)
EPS_1 = two_pi * 26.0        # MHz   comb-1 drive strength (App. D)
EPS_2 = two_pi * 11.5        # MHz   comb-2 drive strength (App. D)
TAU_COR = 4.0                # us    dissipation half-time (Sec. III)
