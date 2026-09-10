#!/usr/bin/env python3
"""
ramp.py

Gaussian rise/fall envelope, and the full raw-variable -> physical-pulse
constraint chain that composes it with the band-limit projection.

This REPLACES the old soft `boundary_penalty` (a weighted |u_0|^2 + |u_N|^2
term). That penalty could not work: its gradient had support on exactly 2 of
550 rows, and because the band-limit projection is a circular FFT operator
that does not preserve endpoint zeros, the goal and the reparametrization were
in direct conflict. The penalty sweep measured it as dead -- variation along
the `boundary` axis straddled the basin-noise floor and was non-monotonic
(see analysis/penalty_sweep.py's grid note). Here the boundary condition is
instead STRUCTURAL: the envelope vanishes at both ends by construction, so
there is nothing for the optimizer to trade away.

The chain (order matters, see below):

    physical pulse  u = R(P(x)),   R = diag(env),  P = IFFT . mask . FFT
    gradient        dCost/dx = P(env * dCost/du)

P is self-adjoint and idempotent (core/fourier_cutoff.py); R is a real
diagonal scaling, hence self-adjoint. The Jacobian of the chain is R.P, so
its adjoint is P.R -- multiply by `env` FIRST, then project. Note the composed
chain is NOT itself a projection and NOT self-adjoint, which is why this lives
here rather than in fourier_cutoff.py: that module's central claim ("the same
function projects both the pulse and the gradient") is true of P alone.

Getting the order backwards still produces a plausible-looking gradient, so
validation/test_grape_core_perf.py checks the adjoint identity exactly, with a
negative control asserting the reversed composition fails it.

WHAT THIS ACTUALLY DELIVERS -- and what still bounds it. Measured as
max|u[0]| / peak|u| ("how far from off does the drive start") across the nine
retrained production pulses, pre-ramp -> post-ramp:

    opt 6.06%->1.13%   enc 4.44%->1.91%   dec 3.17%->1.60%   X 4.75%->2.39%
    Z   3.64%->0.32%   H   2.98%->1.53%   T   4.98%->0.47%   I 3.87%->0.34%

Every operation improves, typically ~3x and up to 11x, but none reaches zero
and the spread is systematic rather than noise: it tracks how amplitude-hungry
the gate is. Two causes. (1) Midpoint sampling leaves env[0] = 0.0135, not 0.
(2) More importantly, the envelope removes control authority over the first and
last 24 steps, and L-BFGS-B buys it back by inflating the PRE-IMAGE there: on
u_X_main, |P(x)[0]| reaches 30.6 against a mid-pulse RMS of 5.03, a factor of
six, so u[0] = 0.0135 * 30.6 = 0.42 survives. Hence the high-amplitude gates
(X, enc, H) land near 2% while the near-phase-only ones (Z, T, I) reach ~0.3%.

The consequence worth remembering: that inflation is bounded by
`hard_amp_limit` on the raw variable, NOT by the envelope, so raising it would
quietly let the endpoints creep back up. It can also BIND: retraining Y cold at
seed 42 drove max|x| to exactly 40.0 and landed in a 44%-leakage basin. Check
`info['max_abs_preimage']` against the limit before trusting a pulse;
`constraint_report` below exists so all of this is measured, not assumed.

WHY project THEN ramp. Measured at core geometry (N=550, dt=0.002 us,
cav (-27,27) MHz, tra (-33,33) MHz, 48 ns ramp) on white noise -- the worst
case, a trained pulse does far better:

    project -> ramp :  0.60% / 0.53% out-of-band,  endpoints at 0.84% of mid
    ramp -> project :  ~1e-31 out-of-band,         endpoints at 32%   of mid

Projecting last gives exact band-limiting but smears the envelope until the
endpoints are a third of full scale, defeating the entire point of the ramp.
Ramping last is therefore used, and the residual band violation is sub-percent
and measured by the test suite rather than assumed. Same conclusion EST
reached independently at its own geometry (EST/grape_jax.py:make_constrainer).
"""

import numpy as np

from core.fourier_cutoff import project_bandlimit, out_of_band_energy_fraction

# 48 ns Gaussian rise/fall. Originally from Roy, Wetherbee & Fatemi App. C via
# EST/device.py; adopted here as the core pipeline's default.
DEFAULT_RAMP_NS = 48.0


def ramp_envelope(N, dt, ramp_ns=DEFAULT_RAMP_NS):
    """
    Gaussian rise/fall envelope, (N,) real in [0, 1].

    Pedestal-subtracted so the underlying continuous envelope is exactly 0 at
    t = 0 and t = T, and exactly 1 across the flat top. Samples are taken at the
    MIDPOINT of each piecewise-constant step, which is the honest convention for
    a sampled-and-held drive but means the first and last steps sit at a small
    nonzero value rather than at 0 (0.7% of full scale at EST's N=1000/dt=1 ns;
    1.35% at the core pipeline's N=550/dt=2 ns, since a coarser grid puts the
    first midpoint further up the Gaussian).

    The paper specifies "48 ns Gaussian rise/fall" without a sigma convention;
    sigma = T_ramp/2 is this repo's reading of it, shared with EST/device.py.

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


def make_constraint_chain(N, dt, cav_band=None, tra_band=None, ramp_ns=None):
    """
    Build the (to_physical, to_preimage_grad) pair for one optimization run.

    to_physical(x)        : (N,4) raw pre-image -> (N,4) physical pulse
    to_preimage_grad(g)   : (N,4) dCost/du     -> (N,4) dCost/dx

    Both are always returned, degrading to identity when every constraint is
    disabled, so callers never branch on `if bandlimit:` and the composition
    order exists in exactly one place.

    cav_band/tra_band : (f_lo, f_hi) MHz tuples, or both None to disable the
        band limit. Passing exactly one is an error (same rule as
        optimize_multi_state_pulse).
    ramp_ns : float, or None to disable the ramp.
    """
    if (cav_band is None) != (tra_band is None):
        raise ValueError("cav_band and tra_band must both be given or both be None")

    bandlimit = cav_band is not None
    env = None if not ramp_ns else ramp_envelope(N, dt, ramp_ns)

    def to_physical(x):
        u = np.asarray(x, dtype=np.float64).reshape(N, 4)
        if bandlimit:
            u = project_bandlimit(u, dt, cav_band, tra_band)
        if env is not None:
            u = u * env[:, None]
        return u

    def to_preimage_grad(g):
        # Adjoint of (R . P) is (P . R): scale by env first, then project.
        g = np.asarray(g, dtype=np.float64).reshape(N, 4)
        if env is not None:
            g = g * env[:, None]
        if bandlimit:
            g = project_bandlimit(g, dt, cav_band, tra_band)
        return g

    return to_physical, to_preimage_grad


def deramp(u, dt, cav_band=None, tra_band=None, ramp_ns=None, atol=1e-10):
    """
    Invert the constraint chain: recover a raw pre-image x with
    to_physical(x) == u, for warm-starting from a SAVED (physical) pulse.

    Only a pseudo-inverse: it undoes the envelope exactly (u/env is already
    band-limited, and P is idempotent, so P(u/env) == u/env) but cannot undo P
    itself. Returns (x, roundtrip_err) and raises if the roundtrip exceeds
    `atol` -- which is the correct behavior for a pulse that was never ramped,
    since such a pulse is genuinely not in the range of the current chain.

    Note the amplification. The envelope floors at 0.0135 at core geometry
    (N=550, dt=2 ns), so deramping a pulse that was NOT produced by this chain
    inflates its edges by up to ~74x and will usually blow past hard_amp_limit.
    A pulse that WAS produced by this chain has small edges to begin with, so
    the division is benign. Prefer resuming from a saved x_*.npy pre-image
    (`init_x`) over deramping a u_*.npy whenever one exists.
    """
    u = np.asarray(u, dtype=np.float64)
    N = u.shape[0]
    to_physical, _ = make_constraint_chain(N, dt, cav_band, tra_band, ramp_ns)

    if ramp_ns:
        x = u / ramp_envelope(N, dt, ramp_ns)[:, None]
    else:
        x = u.copy()

    err = float(np.abs(to_physical(x) - u).max())
    if err > atol:
        raise ValueError(
            f"constraint-chain inversion failed: roundtrip error {err:.3e} > {atol:.1e}. "
            "The saved pulse is not in the range of the current constraint chain "
            "(trained before the ramp existed, or with a different band/ramp_ns/dt?). "
            "Resume from a saved x_*.npy pre-image via init_x, or pass "
            "warm_start_strict=False to proceed anyway."
        )
    return x, err


def constraint_report(u, dt, cav_band=None, tra_band=None):
    """
    Measure what the chain actually delivered on a finished pulse: endpoint
    amplitude relative to mid-pulse, peak amplitude, and residual out-of-band
    energy per drive.

    This is how the ramp's cost gets REPORTED rather than assumed. Ramping
    after projecting leaves a sub-percent band violation by construction; these
    numbers are what make that visible per gate instead of once in a test.
    """
    u = np.asarray(u, dtype=np.float64)
    N = u.shape[0]
    lo, hi = N // 3, 2 * N // 3
    mid_rms = float(np.sqrt(np.mean(u[lo:hi] ** 2)))

    report = {
        'peak_amp': float(np.abs(u).max()),
        'mid_rms': mid_rms,
        'endpoint_start': float(np.abs(u[0]).max()),
        'endpoint_end': float(np.abs(u[-1]).max()),
    }
    endpoint = max(report['endpoint_start'], report['endpoint_end'])
    report['endpoint_rel_to_mid'] = endpoint / mid_rms if mid_rms > 0 else 0.0
    # The interpretable one: "how far from off does the drive start", as a
    # fraction of the waveform's own full scale. Prefer this when reporting --
    # endpoint_rel_to_mid divides a max by an RMS and also moves when the
    # mid-pulse RMS changes, which makes it awkward to compare across pulses.
    report['endpoint_rel_to_peak'] = (
        endpoint / report['peak_amp'] if report['peak_amp'] > 0 else 0.0
    )
    if cav_band is not None and tra_band is not None:
        oob = out_of_band_energy_fraction(u, dt, cav_band, tra_band)
        report['out_of_band_cavity'] = float(oob['cavity'])
        report['out_of_band_transmon'] = float(oob['transmon'])
    return report
