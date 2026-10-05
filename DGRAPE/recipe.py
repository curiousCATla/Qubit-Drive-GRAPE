"""
Default training recipe for open-system GRAPE (DGRAPE/objective.py, DGRAPE/train.py).

Copied from the closed-system production recipe: experiments.ipynb `OPTIMIZATION_RECIPE`
and the `main.py` CLI defaults (validation/test_dgrape.py pins RECIPE to
`main.build_arg_parser()` so the two cannot silently drift).

    cost = 1 - mean_i F_pro(n_c_i)                                     (Eq. 23)
           + disc  * sum_{i != j} (F_i - F_j)^2                        (Eq. 24)
           + deriv * sum_k |u_k - u_{k-1}|^2
           + amp   * sum_k sum_{d in C,T} (|eps_d(k)| - amp_max)_+^2   (Supp. Eq. 19, 'modulus')
           + forbid * mean_i sum_{k=1}^{N} sum_{b=0_L,1_L} Tr[P_F rho_b(k)]   (forbidden states)
    u = ramp(bandlimit(x)),  |x| <= hard_amp_limit elementwise         (Supp. Eq. 22, core/ramp.py)

THESE NUMBERS WERE TUNED ON THE HEERES CHIP (core/grape_core.py), not on Shirol's. They are
defaults to validate on this device (dissipation_grape.ipynb §8 and §12), not results. In
particular the 25 rad/us cap is the Heeres-recipe drive limit; Shirol et al. publish none.

TRUNCATIONS. `trunc_list = [12, 14, 16]` is NOT the Heeres recipe (main.py trains on
[22, 24, 26], sized for a code reaching Fock ~20) and is deliberately not pinned to it. The code
reaches only Fock 5 (MIN_N_C = 7), but the first trained X pulse (dissipation_grape.ipynb §10)
climbed to Fock 10+ on the earlier guess [10, 12] and lost 2.1 % at the converged n_c >= 16, so
[10, 12] is falsified. The Lindblad cost grows as d^3 = (6 n_c)^3; the three truncations are
evaluated in parallel (`n_jobs`, joblib, as main.py). Score a trained pulse on a held-out n_c
(e.g. 18) before trusting it.

FORBIDDEN STATES (DGRAPE-specific, not in main.py). Heeres' "cost for occupation of certain
higher states |F>", C = sum_n |<psi_F|psi_n>|^2 summed over every time step, here the Lindblad
version Tr[P_F rho_b(k)] for the two code basis inputs, with P_F the projector on storage Fock
>= forbid_fock_min. It keeps the pulse away from the truncation wall directly, where Eq. 24 only
notices it once the truncations disagree (lambda_disc 0.5 did not, §10). It is a raw sum over
steps: lambda_forbid = 1e-3 makes 1 % forbidden population held for N = 550 steps cost ~0.01, the
size of the infidelity; rescale it if N changes a lot. Every training n_c must exceed
forbid_fock_min, or the term is identically zero (the objective refuses).

Deliberately ABSENT: gate duration and dt, which depend on this chip (chi_ge/2pi = 1.12 MHz,
half of Heeres') and stay required arguments of train.py.
"""

RECIPE = dict(
    trunc_list=[12, 14, 16],       # see TRUNCATIONS above; not the Heeres [22, 24, 26]
    n_jobs=3,                      # parallel truncations (main.py --n-jobs)
    cav_band=(-27.0, 27.0),        # MHz, on eps_C = C_I - i C_Q (DGRAPE/model.py)
    tra_band=(-33.0, 33.0),        # MHz, on eps_T = T_I - i T_Q
    ramp_ns=48.0,                  # Gaussian rise/fall, core.ramp.ramp_envelope
    penalties={"deriv": 1e-5, "amp": 8e-5, "amp_max": 25.0, "disc": 0.5,
               "forbid": 1e-3},    # forbid: DGRAPE only, see FORBIDDEN STATES above
    forbid_fock_min=10,            # P_F projects on storage Fock >= this
    amp_norm="modulus",            # amp_max caps |I + iQ| per drive
    hard_amp_limit=25.0,           # L-BFGS-B box on the raw pre-image x
    warm_start_amp=4.0,            # cold start: core.optimizer.make_smooth_warm_start
    warm_start_cutoff_frac=0.04,
)

VALID_PENALTY_KEYS = frozenset(RECIPE["penalties"])
HEERES_PENALTY_KEYS = ("deriv", "amp", "amp_max", "disc")   # the ones pinned to main.py


def penalties_with_defaults(penalties=None):
    """Recipe penalties overridden by `penalties`; unknown keys raise (as core.optimizer)."""
    out = dict(RECIPE["penalties"])
    if penalties:
        unknown = set(penalties) - VALID_PENALTY_KEYS
        if unknown:
            raise ValueError(f"unknown penalty key(s) {sorted(unknown)}; "
                             f"expected a subset of {sorted(VALID_PENALTY_KEYS)}")
        out.update(penalties)
    return out
