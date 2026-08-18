#!/usr/bin/env python3
"""
analysis/penalty_sweep.py

Multi-objective (Pareto) sweep over the four GRAPE soft-penalty weights.

The cost function in `core/optimizer.optimize_multi_state_pulse` carries four
regularizer weights -- `deriv`, `boundary`, `amp`, `disc`. Because they are
regularizers, maximizing the *training* objective F_coh drives all four toward
zero and yields rough, high-bandwidth pulses that overfit the training
truncations `trunc_list`. Choosing them is therefore not a scalar optimization:
it is a trade-off between the reported fidelity on *held-out* cavity
truncations and the pulse's hardware cost (bandwidth, peak amplitude) and
robustness.

This harness sweeps the weights, retrains a pulse per configuration, and scores
each on metrics that training never sees:

  * `F_ped_heldout_{mean,min}` -- Pedersen gate fidelity (main.report_pedersen_
    gate_fidelity) averaged over cavity truncations NOT in `trunc_list`. This is
    the reported figure of merit, distinct from the optimizer's F_coh.
  * `robustness_spread`        -- max-min of F_ped across the held-out set.
  * `leakage_L1_mean`          -- mean population leaving the logical subspace.
  * `peak_amp`, `roughness`, `occ99_frac` -- pulse cost. `occ99_frac` is the
    99% occupied spectral width as a FRACTION of each drive's own band mask,
    maxed over the two drives; the hardware budget is a percentage of it. The
    same widths in MHz are reported as `occ99_MHz` and per drive, but the raw
    max is not scored on because the two masks differ (54 vs 66 MHz). The older
    RMS-about-zero `bandwidth_MHz` is reported and not scored either: it charges
    the drive's carrier offset as width, and that offset is a gauge choice. See
    `pulse_metrics` and `BAND_LIMITS_MHz`.

Only the penalty weights vary. Everything else (warm start, `trunc_list`,
frequency bands, amplitude limits, N, dt) is pinned in `FIXED` below so the
comparison is clean.

Both the trained pulse and the scored metric row are disk-cached under
`results/penalty_sweep_cache/`, keyed by a hash of the full configuration, so an
interrupted sweep resumes for free and re-scoring never retrains.

Usage
-----
    python analysis/penalty_sweep.py --gate X --mode ofat --seeds 42 --maxiter 1500
    python analysis/penalty_sweep.py --gate X --mode grid --grid-x deriv --grid-y boundary

Config-level parallelism
------------------------
One config occupies only ~2.4 cores (joblib fans the `trunc_list` evaluation out
`n_jobs`-wide inside the optimizer, and each worker is single-threaded), so a
serial sweep leaves most of a 10-core machine idle. Run K shards concurrently,
then merge:

    for i in 0 1 2; do
        python analysis/penalty_sweep.py --gate X --seeds 42 43 44 --shard $i/3 &
    done; wait
    python analysis/penalty_sweep.py --gate X --seeds 42 43 44 --merge

Shards partition `configs x seeds` and never collide: every row lands in its own
`row_<hash>.json`. The split balances the configs that actually need training
rather than the raw item count -- on a resumed sweep the cached rows are free,
and a naive round-robin can hand one shard all of them (see `_shard_filter`).
`--merge` rebuilds the full CSV + manifest from that cache in seconds and
refuses to guess if a row is missing.

Sharding is used in preference to wrapping the config loop in an outer
`joblib.Parallel`, because joblib silently degrades a nested `Parallel` inside a
loky worker to sequential -- which would cancel the speedup it was added for.

Outputs
-------
    tables/penalty_sweep_<gate>_ofat.csv
    tables/penalty_sweep_<gate>_grid_<x>_<y>.csv
    results/penalty_sweep_<gate>_<mode>.json      (manifest: configs + provenance)
    ..._shard<I>of<K>.csv                          (per-shard partial tables)
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.grape_core import chi, coherent_fidelity_multi_state, two_pi
from core.optimizer import optimize_multi_state_pulse
from core.propagator import leakage_L1
from main import GATE_FACTORIES, report_pedersen_gate_fidelity


# ============================================================
# Fixed experimental conditions -- everything that is NOT swept
# ============================================================
#
# These mirror main.py's production defaults. They are pinned here rather than
# exposed as sweep axes so that every row in the output CSV differs ONLY in the
# penalty weights. Changing anything here invalidates the cache (it is part of
# the config hash), which is the intended behavior.

FIXED = {
    "trunc_list": [22, 24, 26],
    "n_t": 3,
    "N": 550,
    "dt": 0.002,
    "cav_band": (-27.0, 27.0),
    "tra_band": (-33.0, 33.0),
    "hard_amp_limit": 40.0,
    "amp_max": 40.0,
    "warm_start": None,          # smooth random warm start, seeded per-run
}

# --- Two-phase training protocol ("--protocol two_phase") --------------
#
# An alternative to FIXED["trunc_list"]/maxiter's single continuous run:
#   phase 1 -- cold start, trunc_list=[22, 26], maxiter=1000
#   phase 2 -- warm-started from phase 1's output, trunc_list=[26] only,
#              maxiter=1500, to refine fidelity at the single truncation.
#
# This is a real deviation from the cold-start, all-truncations-at-once recipe
# README.md's "Truncation convergence and wall exploitation" section documents
# as necessary: phase 2 here is exactly warm-start + single-truncation, the
# pattern that let past pulses overfit the Hilbert-space truncation wall. No
# extra guard is added against that here -- the existing held-out scoring over
# DEFAULT_EVAL_TRUNCS (trained truncations are only a subset of what is scored)
# is the mechanism that would reveal it, and is applied to every pulse
# regardless of protocol. Treat a two_phase config whose held-out fidelity
# collapses away from n_c=26 as exactly that failure mode, not a bug here.
TWO_PHASE = {
    "phase1_trunc_list": [22, 26],
    "phase1_maxiter": 1000,
    "phase2_trunc_list": [26],
    "phase2_maxiter": 1500,
}

# Per-drive spectral allowance: the width of each drive's own hard band mask.
# The soft budget is a percentage OF THIS, so the two drives are compared on
# equal terms even though their masks differ.
#
# Collapsing them into max(occ99_cav_MHz, occ99_tra_MHz) <= tau was a conjunction
# of two constraints forced to share one threshold, which the drives do not
# share: they are separate hardware chains (own LO, mixer, DAC channel, line)
# and nothing physically experiences the max of the two. Normalising each by its
# own allowance restores the correct conjunction,
#
#     max(x/a, y/b) <= t   <=>   (x <= t*a) AND (y <= t*b),
#
# and makes the reported scalar "share of its own allowance consumed by the
# worse drive" -- dimensionless, so the budget cannot be mis-ported.
BAND_LIMITS_MHz = {"cav": 54.0, "tra": 66.0}

# Tie to FIXED so the two cannot silently drift apart.
assert BAND_LIMITS_MHz["cav"] == FIXED["cav_band"][1] - FIXED["cav_band"][0]
assert BAND_LIMITS_MHz["tra"] == FIXED["tra_band"][1] - FIXED["tra_band"][0]

# Incumbent recipe = main.py's current defaults. Every OFAT ladder holds the
# other three weights here.
BASELINE = {
    "deriv": 1e-5,
    "boundary": 2e-5,
    "amp": 8e-5,
    "disc": 0.5,
}

PENALTY_NAMES = ("deriv", "boundary", "amp", "disc")

# --- Why `disc` is pinned here and NOT swept ---------------------------
#
# `lambda_disc` is inert under these fixed conditions, on three independent
# counts, so sweeping it buys nothing but wall-clock:
#
#   1. Cost.     disc_cost = 5.81e-08 at the incumbent, so its contribution is
#                lambda * 5.81e-08 <= 2.9e-07 even at lambda = 5, against an
#                infidelity of O(2.4e-03). Cost share below 1e-04.
#   2. Gradient. disc_grad = sum_{i!=j} 2*delta*(g_i - g_j) with
#                delta = F_i - F_j ~ 1e-04 (implied by disc_cost over 6 ordered
#                pairs) -- about four orders of magnitude below the fidelity
#                gradient it is added to. Inert in gradient, not merely in cost.
#   3. Measured. The retired ladder spanned lambda in {0, .05, .15, .5, 1.5, 5},
#                a 100x range INCLUDING exactly zero, and F_ped_heldout_mean
#                wandered 0.99595 -> 0.99698 NON-MONOTONICALLY with no trend.
#
# Point 3 is worth more than the axis it retired: an objective perturbation
# that is provably dynamically null still moves the held-out fidelity by
# 1.03e-03. That is L-BFGS-B landing in a different local basin, and it is this
# pipeline's real noise floor for comparing two different configs (the 1.5e-04
# replicate-pair figure measures only reduction-order nondeterminism). Those 5
# rows are kept in tables/penalty_sweep_X_ofat.csv as the basin-noise estimate.
#
# Scope: `disc` is inert BECAUSE the per-truncation fidelities already agree to
# ~1e-04 under band-limited, penalized, cold-started training at [22, 24, 26].
# It is not inert in general -- Heeres Supp. Eq. 24 exists for a reason. Do not
# generalize this past these conditions.
#
# IMPORTANT: `disc` stays in PENALTY_NAMES and stays at 0.5 in BASELINE. It is
# part of the `_config_hash` payload, so setting it to 0.0 would invalidate all
# 60 cached pulses and force a ~10.6 h retrain for a numerically null change.

# OFAT ladder: multiples of the incumbent value, plus a 0 negative control.
# The 0 row is the point of the exercise -- it should show HIGHER training
# fidelity but worse bandwidth/robustness, demonstrating the trade-off is real.
OFAT_MULTIPLIERS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)

# --- The amp_max axis --------------------------------------------------
#
# Sweeping `lambda_amp` is a NO-OP under these fixed conditions, and measurably
# so: `amplitude_penalty(u, amp_max=40)` returns exactly 0.0 with an exactly
# zero gradient, because converged pulses here peak near 17 rad/us -- well under
# the 40 rad/us soft threshold. Multiplying an identically-zero term by any
# weight changes nothing (verified: F_coh range 0.0e+00 across lambda_amp
# 0 -> 8e-4).
#
# The live knob is the THRESHOLD, not the weight. This ladder brackets the
# observed peak amplitude, so the low end binds hard and the high end never
# binds -- which is what makes it a real peak-power constraint.
AMP_MAX_VALUES = (12.0, 16.0, 20.0, 28.0, 40.0)

# Axis names that are valid sweep targets. "amp_max" is not a penalty weight,
# so it is handled separately from PENALTY_NAMES throughout. `disc` and `amp`
# are both absent by design -- see the PENALTY_NAMES note above for `disc` and
# AMP_MAX_VALUES for `amp`.
AXIS_NAMES = ("deriv", "boundary", "amp_max")

# Focused 2-D grid axes (absolute values, not multipliers).
#
# deriv x boundary was run as a full 5x5 (see the retained
# tables/penalty_sweep_X_grid_deriv_boundary.csv). It cost 5.9 h and returned
# one durable finding: the response is 1-D in `deriv`. Variation along deriv was
# 6.3e-03 .. 1.21e-02 at every fixed boundary, while variation along boundary
# was 8.1e-04 .. 4.81e-03 -- straddling the 1.03e-03 basin-noise floor and
# non-monotonic in all five rows.
#
# So the grid is trimmed to the low-deriv 3x3 corner that brackets the
# recommended deriv = 1e-06. All nine cells are already in the cache, so the
# grid now costs ZERO new compute and the freed budget goes to seed replication
# on the OFAT ladders, which is what actually puts error bars on the claims.
GRID_VALUES = {
    "deriv": [0.0, 3e-6, 1e-5],
    "boundary": [0.0, 6e-6, 2e-5],
    "amp_max": list(AMP_MAX_VALUES),
}

# Cavity truncations used for SCORING. Those in FIXED["trunc_list"] were trained
# on; the rest are held out and carry the headline metric.
DEFAULT_EVAL_TRUNCS = list(range(16, 41, 2))

# --- Mid-training snapshots --------------------------------------------
#
# Iterations at which to also keep the pulse a *truncated* run would have
# returned (core.optimizer.optimize_multi_state_pulse's `snapshot_iters`). The
# optimizer-side cost is a callback that copies an array; the sweep-side cost is
# one extra score_pulse per snapshot, ~7 s against ~850 s of training, so this
# is ~1.7% overhead for a maxiter-sensitivity curve on every trained config.
#
# The question it answers: would a cheap screening stage preserve the RANKING of
# configs? If Spearman rho between the @500 ordering and the final ordering is
# ~1, a short screen becomes a defensible way to extend this sweep from X to all
# six logical gates at ~6x the cost. Given that the pipeline's basin-noise floor
# is 1.03e-03 (see the note above PENALTY_NAMES) it may well not hold -- which is
# the point of measuring it rather than assuming it.
#
# NOT part of `_config_hash`: snapshots are an additive artifact of a run, they
# do not change the pulse it produces, and hashing them would invalidate every
# cached pulse. Rows and pulses trained before this existed simply have no
# snapshot files, and the columns come back absent.
SNAPSHOT_ITERS = (500, 1000)

CACHE_DIR = os.path.join(REPO_ROOT, "results", "penalty_sweep_cache")
TABLE_DIR = os.path.join(REPO_ROOT, "tables")
RESULT_DIR = os.path.join(REPO_ROOT, "results")


# ============================================================
# Configuration enumeration
# ============================================================

def _label(penalties, amp_max):
    """
    Compact human-readable label for a recipe. `amp_max` is appended only when
    it differs from the fixed default, so the common case stays readable and
    labels from before the amp_max axis existed are unchanged.
    """
    s = "_".join(f"{k}={penalties[k]:.3g}" for k in PENALTY_NAMES)
    if amp_max != FIXED["amp_max"]:
        s += f"_ampmax={amp_max:.3g}"
    return s


def _make_config(pen, amp_max, swept, multiplier=np.nan):
    return {
        "penalties": pen,
        "amp_max": amp_max,
        "swept": swept,
        "multiplier": multiplier,
        "label": _label(pen, amp_max),
        "is_baseline": (
            all(pen[k] == BASELINE[k] for k in PENALTY_NAMES)
            and amp_max == FIXED["amp_max"]
        ),
    }


def build_ofat_configs():
    """
    One-factor-at-a-time: vary each axis while the others sit at BASELINE. The
    baseline point appears once, deduplicated.

    Neither `amp` nor `disc` is an axis -- both are provably inert here. `amp`'s
    slot is taken by `amp_max`, the threshold that actually determines whether
    the amplitude penalty ever activates (see AMP_MAX_VALUES); `disc`'s ladder
    was retired outright (see the note above PENALTY_NAMES).
    """
    configs, seen = [], set()

    for name in ("deriv", "boundary"):
        for mult in OFAT_MULTIPLIERS:
            pen = dict(BASELINE)
            pen[name] = BASELINE[name] * mult
            key = (tuple(pen[k] for k in PENALTY_NAMES), FIXED["amp_max"])
            if key in seen:
                continue
            seen.add(key)
            configs.append(_make_config(pen, FIXED["amp_max"], name, mult))

    for amp_max in AMP_MAX_VALUES:
        pen = dict(BASELINE)
        key = (tuple(pen[k] for k in PENALTY_NAMES), amp_max)
        if key in seen:
            continue
        seen.add(key)
        configs.append(_make_config(pen, amp_max, "amp_max", np.nan))

    return configs


def build_disc_null_configs():
    """
    The retired `disc` ladder, kept as a regenerable artifact.

    These six configurations are no longer a sweep axis (see the note above
    PENALTY_NAMES) but they are not scratch data: `lambda_disc` spans a
    hundredfold range including exactly zero under a perturbation that is
    provably negligible in both cost and gradient, so their spread in held-out
    fidelity is a *calibrated null* -- this pipeline's basin-noise floor, and the
    yardstick every cross-config claim is measured against.

    Kept as its own mode rather than left in `build_ofat_configs` because it is
    a null experiment, not an axis: including it in the OFAT ladder invited
    exactly the misreading that its 1.03e-3 spread was a 7x-above-noise effect.
    All six are already trained, so `--mode disc-null` is a cache read.
    """
    return [
        _make_config(dict(BASELINE, disc=BASELINE["disc"] * mult),
                     FIXED["amp_max"], "disc", mult)
        for mult in OFAT_MULTIPLIERS
    ]


def _axis_apply(pen, amp_max, name, value):
    """Set one axis, returning the updated (penalties, amp_max) pair."""
    if name == "amp_max":
        return pen, value
    pen = dict(pen)
    pen[name] = value
    return pen, amp_max


def build_grid_configs(x_name, y_name):
    """Full factorial over two axes; everything else stays at BASELINE."""
    configs = []
    for xv in GRID_VALUES[x_name]:
        for yv in GRID_VALUES[y_name]:
            pen, amp_max = dict(BASELINE), FIXED["amp_max"]
            pen, amp_max = _axis_apply(pen, amp_max, x_name, xv)
            pen, amp_max = _axis_apply(pen, amp_max, y_name, yv)
            configs.append(_make_config(pen, amp_max, f"{x_name}x{y_name}"))
    return configs


# ============================================================
# Pulse-cost metrics
# ============================================================

def occupied_width(power, freqs, frac=0.99):
    """
    Minimum-width frequency window holding `frac` of the spectral power.

        W = min over [a, b] of (b - a)   s.t.   sum_{a <= f <= b} p(f) >= frac

    Returns `(width_MHz, f_lo_MHz, f_hi_MHz)`.

    Two properties are the reason this exists (see `pulse_metrics`):

      * **Translation-invariant.** Shifting the spectrum shifts the window and
        leaves its width alone, so the width is a property of the waveform and
        not of the drive LO.
      * **Tail-robust.** Every bin contributes its own power and nothing more,
        unlike an f^2-weighted moment where the outermost decade of a
        hard-masked spectrum dominates the statistic.

    Taking the *minimum-width* window rather than a window centred on the
    centroid matters here: these spectra are strongly skewed, and a symmetric
    99% window about the centroid comes out wider than the band mask itself
    (82.3 MHz against a 66 MHz mask for the lambda_deriv = 0 transmon drive),
    because half of it sits over empty spectrum.

    The edges land on bin centres, so the width is quantised to the FFT bin
    spacing 1/(N*dt) = 0.909 MHz at N = 550, dt = 0.002 us.
    """
    power = np.asarray(power, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    total = power.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0

    # np.fft.fftfreq is NOT monotonic (it wraps to negative frequencies at the
    # halfway point), so the axis has to be sorted before any window is a
    # contiguous slice.
    order = np.argsort(freqs)
    f = freqs[order]
    c = np.cumsum(power[order] / total)

    # Power in the closed window [f[a], f[b]] is c[b] - c[a-1]. For each left
    # edge, the narrowest admissible right edge is the first b clearing the
    # target -- one searchsorted over the (monotone) cumulative sum. O(N log N).
    before = np.concatenate(([0.0], c[:-1]))
    b = np.searchsorted(c, before + frac, side="left")
    ok = b < c.size
    if not ok.any():
        return float(f[-1] - f[0]), float(f[0]), float(f[-1])

    a_idx = np.nonzero(ok)[0]
    b_idx = b[ok]
    i = int(np.argmin(f[b_idx] - f[a_idx]))
    lo, hi = float(f[a_idx[i]]), float(f[b_idx[i]])
    return hi - lo, lo, hi


def pulse_metrics(u, dt, band_limits=None):
    """
    Hardware-cost and smoothness metrics for a pulse, computed on the physical
    (band-limited, as-returned) waveform.

    Drives are complex: eps_C = u[:,0] + i u[:,1], eps_T = u[:,2] + i u[:,3].

    The **scored** spectral cost is `occ99_frac`: for each drive, the width of
    the narrowest window carrying 99% of its power, divided by that drive's OWN
    band-mask width, then the max over drives. It is a dimensionless "share of
    its own allowance consumed by the worse drive", so a budget on it is a
    percentage. `band_limits` defaults to `BAND_LIMITS_MHz`; pass a dict to score
    a pulse trained under a different mask.

    `occ99_MHz` -- the same widths in MHz, maxed across drives without
    normalising -- is still reported and is readable, but nothing selects on it:
    the cavity is allowed 54 MHz and the transmon 66, so their raw widths are not
    comparable quantities (see `BAND_LIMITS_MHz`).

    `occ99_{cav,tra}_{lo,hi}_MHz` give the window edges, i.e. *where in the
    allowed band* the 99% of power sits. That is a real diagnostic and not
    decoration: on the transmon drive the incumbent occupies [-25.5, +8.2] --
    pushed hard against the lower mask edge -- while lambda_deriv = 0 occupies
    [-32.7, +29.1], filling the mask almost symmetrically. The width statistic
    alone does not distinguish those.

    `bandwidth_MHz` -- the RMS frequency about f = 0, i.e. sqrt of the second
    *raw* moment -- is still reported, for continuity with earlier tables, but
    nothing selects on it. By Koenig-Huygens it splits exactly as

        <f^2> = <f>^2 + sigma_f^2

    so it folds in the spectral centroid <f>, and <f> is a gauge choice rather
    than a property of the pulse: the lab-frame field is Re[eps(t) e^{i w_LO t}]
    and the pair (eps, w_LO) is redundant, since (eps(t) e^{-2 pi i f0 t},
    w_LO + 2 pi f0) is the identical physical field. Retuning the drive LO --
    routine and free -- moves <f> and therefore `bandwidth_MHz` without touching
    the waveform. It also double-counts `core.fourier_cutoff.project_bandlimit`,
    which already pins the spectral *support* exactly (power outside the mask is
    ~1e-31 of the total).

    `sigma_f_MHz` (the second *central* moment) fixes the gauge problem but not
    the tail problem: any f^2-weighted statistic is dominated by the band edges,
    where the mask hard-truncates. `occ99_MHz` fixes both.

    Diagnostics, reported and never scored: `centroid_{cav,tra}_MHz`, and
    `n_bar_drive = centroid_tra_MHz / (chi/2pi)` -- the cavity photon number
    whose number-split transmon peak the drive is centred on.
    """
    band_limits = BAND_LIMITS_MHz if band_limits is None else band_limits

    u = np.asarray(u)
    eps_c = u[:, 0] + 1j * u[:, 1]

    # The transmon drive in normal-ordered form is eps b^dag + h.c. with
    #     eps = u[:,2] - 1j*u[:,3]
    # (from Hc = [.., B+Bd, 1j*(B-Bd)], grape_core.py:44). The array below is
    # eps.conj(), so its DFT is mirrored: P_code(f) = |eps~(-f)|^2.
    # First-order response addresses photon-number peak n at f_n = -Delta_n/2pi
    # with Delta_n = n*chi, and chi < 0, so peak n>0 sits at NEGATIVE code-frequency.
    # The two sign flips cancel, making n_bar_drive = <f>_code / (chi/2pi) correct
    # as written. Do not "fix" one of them in isolation.
    eps_t = u[:, 2] + 1j * u[:, 3]

    # dt is in microseconds, so fftfreq returns cycles/us == MHz directly.
    freqs = np.fft.fftfreq(u.shape[0], d=dt)

    def spectral_stats(eps):
        power = np.abs(np.fft.fft(eps)) ** 2
        total = power.sum()
        # Every branch must return the SAME key set: `_METRIC_KEYS` is probed
        # with an all-zero pulse, which lands here, and a reduced dict there
        # would drop the missing columns from the cache-backfill guard in
        # `run_one` -- silently, and forever.
        if total <= 0:
            return {"rms": 0.0, "centroid": 0.0, "sigma": 0.0,
                    "occ99": 0.0, "lo": 0.0, "hi": 0.0}
        p = power / total
        centroid = float((freqs * p).sum())
        m2 = float((freqs ** 2 * p).sum())
        occ99, lo, hi = occupied_width(power, freqs, 0.99)
        return {
            "rms": float(np.sqrt(m2)),
            "centroid": centroid,
            # max(., 0) guards the exactly-monochromatic case, where the two
            # terms cancel to a small negative float.
            "sigma": float(np.sqrt(max(m2 - centroid ** 2, 0.0))),
            "occ99": occ99,
            "lo": lo,
            "hi": hi,
        }

    s_c, s_t = spectral_stats(eps_c), spectral_stats(eps_t)

    frac_c = s_c["occ99"] / band_limits["cav"]
    frac_t = s_t["occ99"] / band_limits["tra"]

    peak_amp = float(max(np.abs(eps_c).max(), np.abs(eps_t).max()))
    # Per-step RMS first difference, in rad/us: a direct proxy for how hard the
    # waveform is to synthesize.
    roughness = float(np.sqrt(np.mean(np.sum(np.diff(u, axis=0) ** 2, axis=1))))

    return {
        "peak_amp": peak_amp,
        "roughness": roughness,
        # Retained for continuity; superseded as the scored cost by occ99_MHz.
        "bandwidth_MHz": float(max(s_c["rms"], s_t["rms"])),
        "bandwidth_cav_MHz": s_c["rms"],
        "bandwidth_tra_MHz": s_t["rms"],
        # Scored spectral cost: dimensionless, each drive against its own mask.
        "occ99_frac": float(max(frac_c, frac_t)),
        "occ99_frac_cav": float(frac_c),
        "occ99_frac_tra": float(frac_t),
        # Which drive the budget actually binds on. A string -- do not try to
        # average it; `penalty_viz.aggregate_seeds` recomputes it from the
        # averaged fractions.
        "binding_drive": "cav" if frac_c > frac_t else "tra",
        # The same widths in MHz. Readable, and cited in the report, but the raw
        # max across drives is not the scored quantity -- the allowances differ.
        "occ99_MHz": float(max(s_c["occ99"], s_t["occ99"])),
        "occ99_cav_MHz": s_c["occ99"],
        "occ99_tra_MHz": s_t["occ99"],
        # Window edges: WHERE in the allowed band the 99% of power sits.
        "occ99_cav_lo_MHz": s_c["lo"],
        "occ99_cav_hi_MHz": s_c["hi"],
        "occ99_tra_lo_MHz": s_t["lo"],
        "occ99_tra_hi_MHz": s_t["hi"],
        # Echoed for provenance: a row is then self-describing about what
        # allowance its fractions were taken against.
        "band_limit_cav_MHz": float(band_limits["cav"]),
        "band_limit_tra_MHz": float(band_limits["tra"]),
        # Gauge-invariant second moment: reported for the Koenig-Huygens split.
        "sigma_f_MHz": float(max(s_c["sigma"], s_t["sigma"])),
        "sigma_f_cav_MHz": s_c["sigma"],
        "sigma_f_tra_MHz": s_t["sigma"],
        # Gauge quantities: diagnostics only, never a cost.
        "centroid_cav_MHz": s_c["centroid"],
        "centroid_tra_MHz": s_t["centroid"],
        "n_bar_drive": float(s_t["centroid"] / (chi / two_pi)),
    }


# Keys `pulse_metrics` returns. Used to detect cached rows written before a
# metric was added, so `run_one` can backfill them without retraining.
_METRIC_KEYS = tuple(pulse_metrics(np.zeros((8, 4)), 0.002))


def score_pulse(gate, u, eval_truncs, trained_truncs, n_t, dt):
    """
    Score a trained pulse on the reported (Pedersen) fidelity across cavity
    truncations, separating trained from held-out.

    The held-out aggregate is the headline: a pulse that exploits the Hilbert
    space truncation wall scores well on `trained` and poorly here.
    """
    per_trunc, leaks = {}, {}
    for nc in eval_truncs:
        F, U_log = report_pedersen_gate_fidelity(gate, u, n_c=nc, n_t=n_t, dt=dt)
        per_trunc[nc] = float(F)
        leaks[nc] = float(leakage_L1(U_log))

    held = [nc for nc in eval_truncs if nc not in trained_truncs]
    trained = [nc for nc in eval_truncs if nc in trained_truncs]

    F_held = np.array([per_trunc[nc] for nc in held], dtype=float)
    F_train = np.array([per_trunc[nc] for nc in trained], dtype=float) if trained else np.array([np.nan])

    out = {
        "F_ped_heldout_mean": float(F_held.mean()),
        "F_ped_heldout_min": float(F_held.min()),
        "F_ped_heldout_std": float(F_held.std()),
        "robustness_spread": float(F_held.max() - F_held.min()),
        "F_ped_trained_mean": float(np.nanmean(F_train)),
        "leakage_L1_mean": float(np.mean([leaks[nc] for nc in held])),
        "leakage_L1_max": float(np.max([leaks[nc] for nc in held])),
    }
    # Overfit gap: positive means the pulse does better where it was trained.
    out["overfit_gap"] = out["F_ped_trained_mean"] - out["F_ped_heldout_mean"]
    out["_per_trunc"] = {str(k): v for k, v in per_trunc.items()}
    return out


# ============================================================
# Caching
# ============================================================

def _config_hash(gate, penalties, seed, maxiter, amp_max=None, extra=None,
                 protocol="single_phase"):
    """
    Stable hash over everything that affects the trained pulse.

    `amp_max` enters the payload only when it differs from the fixed default.
    That keeps hashes computed before amp_max became a sweep axis valid, so an
    existing cache is not invalidated by adding the axis. `protocol` follows the
    same rule: it enters the payload (along with the actual TWO_PHASE settings,
    mirroring how `fixed` embeds FIXED) only when it is not "single_phase", so
    every hash computed before the two-phase protocol existed -- i.e. every
    hash in the current cache -- is unchanged.

    `SNAPSHOT_ITERS` deliberately does NOT enter the payload -- see its
    definition. Anything added here invalidates every cached pulse, so add only
    things that change the pulse.
    """
    payload = {
        "gate": gate,
        "penalties": {k: float(penalties[k]) for k in PENALTY_NAMES},
        "seed": int(seed),
        "maxiter": int(maxiter),
        "fixed": {
            k: (list(v) if isinstance(v, (list, tuple)) else v)
            for k, v in FIXED.items()
        },
    }
    if amp_max is not None and float(amp_max) != float(FIXED["amp_max"]):
        payload["amp_max"] = float(amp_max)
    if protocol != "single_phase":
        payload["protocol"] = protocol
        payload["two_phase"] = TWO_PHASE
    if extra:
        payload["extra"] = extra
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _cache_paths(gate, h):
    d = os.path.join(CACHE_DIR, gate)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"u_{h}.npy"), os.path.join(d, f"row_{h}.json")


def _snapshot_path(gate, h, k):
    return os.path.join(CACHE_DIR, gate, f"u_{h}_it{k}.npy")


def _phase1_meta_path(gate, h):
    """Sidecar for phase 1's `info` dict, since a phase-split run reads phase
    1's outcome in a DIFFERENT process than the one that trained it -- unlike
    the chained path, which still has `info1` in local memory. Not part of
    `_config_hash` (same reasoning as SNAPSHOT_ITERS above: this is metadata
    about a run, not something that changes the pulse it produces)."""
    return os.path.join(CACHE_DIR, gate, f"u_{h}_itphase1_meta.json")


def _get_or_train_phase1(gate, cfg, seed, h, n_t, N, dt, verbose, force):
    """
    Reuse phase 1's on-disk snapshot (and its metadata sidecar) if present;
    otherwise cold-start it. Used identically by the chained (`phase=None`)
    path and by Stage A (`phase=1`), so both ways of running phase 1 write
    and read the exact same files and can resume each other.

    `n_jobs` is fixed at `len(TWO_PHASE["phase1_trunc_list"])` -- always
    correct for this call, not a caller-supplied knob, since it must match
    the truncation count.

    Returns `(u1, meta, trained)` where `meta = {"final_fidelity",
    "train_time_s"}` and `trained` is False on a cache hit.
    """
    snap_path = _snapshot_path(gate, h, "phase1")
    meta_path = _phase1_meta_path(gate, h)
    if not force and os.path.exists(snap_path):
        u1 = np.load(snap_path)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            # A snapshot from before this sidecar existed. Not fatal: the
            # snapshot itself is the only thing phase 2 actually needs to
            # warm-start from.
            meta = {"final_fidelity": float("nan"), "train_time_s": float("nan")}
        return u1, meta, False

    penalties = cfg["penalties"]
    amp_max = cfg.get("amp_max", FIXED["amp_max"])
    pen = dict(penalties)
    pen["amp_max"] = amp_max
    t0 = time.time()
    u1, info1 = optimize_multi_state_pulse(
        GATE_FACTORIES[gate],
        trunc_list=TWO_PHASE["phase1_trunc_list"],
        warm_start=None, warm_start_seed=seed,
        maxiter=TWO_PHASE["phase1_maxiter"],
        snapshot_iters=None,
        n_t=n_t, N=N, dt=dt, penalties=pen,
        save_path=None, n_jobs=len(TWO_PHASE["phase1_trunc_list"]),
        cav_band=FIXED["cav_band"], tra_band=FIXED["tra_band"],
        hard_amp_limit=FIXED["hard_amp_limit"],
        fidelity_fn=coherent_fidelity_multi_state, verbose=verbose,
    )
    train_time = time.time() - t0
    np.save(snap_path, u1)
    meta = {"final_fidelity": float(info1.get("final_fidelity", np.nan)),
            "train_time_s": train_time}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return u1, meta, True


def snapshot_scores(gate, h, eval_truncs):
    """
    Score whichever mid-training snapshots exist on disk for config hash `h`.

    Returns `{"F_ped_heldout_mean@500": ..., "occ99_frac@500": ..., ...}`,
    or `{}` when no snapshot files are present -- which is the normal case for
    every pulse trained before SNAPSHOT_ITERS existed. Absent columns, not NaN
    columns: a missing snapshot is missing data, and pandas will represent it as
    NaN in the CSV only for rows that coexist with rows that have it.
    """
    out = {}
    for k in SNAPSHOT_ITERS:
        path = _snapshot_path(gate, h, k)
        if not os.path.exists(path):
            continue
        u_k = np.load(path)
        sc = score_pulse(
            gate, u_k, eval_truncs, set(FIXED["trunc_list"]),
            FIXED["n_t"], FIXED["dt"],
        )
        met = pulse_metrics(u_k, FIXED["dt"])
        out[f"F_ped_heldout_mean@{k}"] = sc["F_ped_heldout_mean"]
        out[f"F_ped_heldout_min@{k}"] = sc["F_ped_heldout_min"]
        out[f"robustness_spread@{k}"] = sc["robustness_spread"]
        out[f"occ99_frac@{k}"] = met["occ99_frac"]
    return out


# ============================================================
# One run
# ============================================================

def run_one(gate, cfg, seed, maxiter, eval_truncs, n_jobs=3, verbose=False,
            force=False, protocol="single_phase", phase=None):
    """
    Train (or load from cache) one pulse for one penalty recipe + seed, then
    score it. Returns a flat dict = one CSV row.

    `protocol="two_phase"` replaces the single continuous
    `optimize_multi_state_pulse` call with the two-phase chain (see
    TWO_PHASE): phase 1 cold-starts at trunc_list=[22,26] for 1000 iters,
    phase 2 warm-starts from phase 1's output at trunc_list=[26] for 1500
    more. The trained-truncation set used for the held-out/trained split in
    `score_pulse` is therefore the union of both phases' truncations, NOT
    FIXED["trunc_list"] -- n_c=24 was never trained on under this protocol
    and must be scored as held-out, not silently dropped from either bucket.

    `phase` (only meaningful when `protocol="two_phase"`) splits that chain
    across two independently-schedulable stages, so a sweep-wide driver can
    give phase 1 (truncation-parallel, n_jobs=2/item) and phase 2
    (n_jobs=1/item, no truncation parallelism to speak of) different outer
    concurrency instead of one flat shard count sized for the worse of the
    two -- see CLAUDE.md's two-phase section.

      * `None` (default) -- chained, single call, unchanged behavior.
      * `1` -- Stage A. Trains (or reuses) only phase 1 and returns a small
        status dict, not a scored row -- there is no final pulse yet.
      * `2` -- Stage B. Requires phase 1's snapshot already on disk (raises
        if missing, rather than silently training it inline under the
        phase-2 concurrency budget); trains phase 2 and returns the full
        scored row, identically shaped to the chained path's.
    """
    if phase is not None and protocol != "two_phase":
        raise ValueError("phase is only meaningful for protocol='two_phase'")

    penalties = cfg["penalties"]
    amp_max = cfg.get("amp_max", FIXED["amp_max"])
    h = _config_hash(gate, penalties, seed, maxiter, amp_max=amp_max,
                     protocol=protocol)
    pulse_path, row_path = _cache_paths(gate, h)

    row_extra_key = _config_hash(
        gate, penalties, seed, maxiter, amp_max=amp_max,
        extra={"eval": list(eval_truncs)}, protocol=protocol,
    )

    if protocol == "two_phase":
        trained_truncs = (set(TWO_PHASE["phase1_trunc_list"])
                          | set(TWO_PHASE["phase2_trunc_list"]))
    else:
        trained_truncs = set(FIXED["trunc_list"])

    # Fully cached row (same pulse AND same scoring set)?
    if not force and os.path.exists(row_path):
        with open(row_path) as f:
            cached = json.load(f)
        if cached.get("_score_key") == row_extra_key:
            # A row written before a metric existed is still a valid SCORE --
            # `_score_key` covers the pulse and the truncation set, neither of
            # which a metric definition touches. So backfill the missing columns
            # from the cached waveform instead of invalidating the row, which
            # would force the expensive Pedersen re-scoring for nothing.
            dirty = False
            if not set(_METRIC_KEYS) <= cached.keys() and os.path.exists(pulse_path):
                cached.update(pulse_metrics(np.load(pulse_path), FIXED["dt"]))
                dirty = True
            # Same argument for snapshots: they are scored from waveforms on
            # disk, so a row predating them can gain the columns without
            # retraining. Only pay the extra score_pulse calls once.
            snap_keys = {f"F_ped_heldout_mean@{k}" for k in SNAPSHOT_ITERS
                         if os.path.exists(_snapshot_path(gate, h, k))}
            if snap_keys and not snap_keys <= cached.keys():
                cached.update(snapshot_scores(gate, h, eval_truncs))
                dirty = True
            if dirty:
                with open(row_path, "w") as f:
                    json.dump(cached, f, indent=2)
            cached["cached"] = True
            return cached

    # Stage A (phase 1 only): train/reuse phase 1's snapshot and stop -- there
    # is no final (phase-2) pulse yet to score, so this returns a lightweight
    # status dict rather than a CSV row.
    if phase == 1:
        if not force and os.path.exists(pulse_path):
            # Some other invocation already produced the final pulse; phase 1
            # has nothing left to do for this item.
            return {"status": "final_cached", "config_hash": h}
        u1, meta1, trained1 = _get_or_train_phase1(
            gate, cfg, seed, h, FIXED["n_t"], FIXED["N"], FIXED["dt"],
            verbose, force,
        )
        return {
            "status": "trained" if trained1 else "phase1_cached",
            "config_hash": h,
            "phase1_train_time_s": meta1["train_time_s"],
            "phase1_F_coh_train": meta1["final_fidelity"],
        }

    # Trained pulse cached?
    if not force and os.path.exists(pulse_path):
        u = np.load(pulse_path)
        info = {"iterations": np.nan, "success": None, "final_fidelity": np.nan}
        train_time = np.nan
        reused_pulse = True
    else:
        pen = dict(penalties)
        pen["amp_max"] = amp_max
        common = dict(
            n_t=FIXED["n_t"], N=FIXED["N"], dt=FIXED["dt"], penalties=pen,
            save_path=None, n_jobs=n_jobs, cav_band=FIXED["cav_band"],
            tra_band=FIXED["tra_band"], hard_amp_limit=FIXED["hard_amp_limit"],
            fidelity_fn=coherent_fidelity_multi_state, verbose=verbose,
        )
        if protocol == "two_phase":
            # phase is None (chained) or 2 (Stage B). Stage B must NEVER
            # retrain phase 1 even under --force -- its contract is "only
            # touch phase 2" -- so force only propagates to phase 1 when this
            # is the chained (phase=None) path. And unlike the chained path,
            # Stage B must never silently cold-start a missing phase 1 --
            # that would put an n_jobs=2 job on a shard sized for phase 2's
            # n_jobs=1, defeating the point of scheduling them separately.
            if phase == 2 and not os.path.exists(_snapshot_path(gate, h, "phase1")):
                raise SystemExit(
                    f"error: no phase-1 snapshot for config_hash={h} "
                    f"(label={cfg['label']}, seed={seed}). Run "
                    f"'--phase 1' for this sweep before '--phase 2'."
                )
            u1, meta1, _ = _get_or_train_phase1(
                gate, cfg, seed, h, FIXED["n_t"], FIXED["N"], FIXED["dt"],
                verbose, force=(force if phase is None else False),
            )
            t0 = time.time()
            u, info = optimize_multi_state_pulse(
                GATE_FACTORIES[gate],
                trunc_list=TWO_PHASE["phase2_trunc_list"],
                warm_start=u1, warm_start_seed=seed,
                maxiter=TWO_PHASE["phase2_maxiter"],
                snapshot_iters=None,
                n_t=FIXED["n_t"], N=FIXED["N"], dt=FIXED["dt"], penalties=pen,
                save_path=None, n_jobs=len(TWO_PHASE["phase2_trunc_list"]),
                cav_band=FIXED["cav_band"], tra_band=FIXED["tra_band"],
                hard_amp_limit=FIXED["hard_amp_limit"],
                fidelity_fn=coherent_fidelity_multi_state, verbose=verbose,
            )
            phase2_time = time.time() - t0
            train_time = meta1["train_time_s"] + phase2_time
            info["phase1_final_fidelity"] = meta1["final_fidelity"]
            info["snapshots"] = {}
        else:
            t0 = time.time()
            u, info = optimize_multi_state_pulse(
                GATE_FACTORIES[gate],
                trunc_list=FIXED["trunc_list"],
                warm_start=FIXED["warm_start"], warm_start_seed=seed,
                maxiter=maxiter,
                snapshot_iters=[k for k in SNAPSHOT_ITERS if k < maxiter],
                **common,
            )
            train_time = time.time() - t0
        np.save(pulse_path, u)
        for k, u_k in info.get("snapshots", {}).items():
            np.save(_snapshot_path(gate, h, k), u_k)
        reused_pulse = False

    scores = score_pulse(
        gate, u, eval_truncs, trained_truncs, FIXED["n_t"], FIXED["dt"]
    )
    metrics = pulse_metrics(u, FIXED["dt"])
    snaps = snapshot_scores(gate, h, eval_truncs) if protocol != "two_phase" else {}

    row = {
        "gate": gate,
        "label": cfg["label"],
        "swept": cfg["swept"],
        "multiplier": cfg["multiplier"],
        "is_baseline": cfg["is_baseline"],
        "seed": seed,
        "maxiter": (TWO_PHASE["phase1_maxiter"] + TWO_PHASE["phase2_maxiter"]
                    if protocol == "two_phase" else maxiter),
        "protocol": protocol,
        **({"phase1_maxiter": TWO_PHASE["phase1_maxiter"],
            "phase2_maxiter": TWO_PHASE["phase2_maxiter"],
            "phase1_F_coh_train": float(info.get("phase1_final_fidelity", np.nan))}
           if protocol == "two_phase" else {}),
        **{f"lambda_{k}": float(penalties[k]) for k in PENALTY_NAMES},
        "amp_max": float(amp_max),
        "F_coh_train": float(info.get("final_fidelity", np.nan)),
        "iterations": info.get("iterations", np.nan),
        "converged": info.get("success", None),
        "train_time_s": train_time,
        **scores,
        **metrics,
        **snaps,
        "config_hash": h,
        "_score_key": row_extra_key,
        "reused_pulse": reused_pulse,
        "cached": False,
    }

    with open(row_path, "w") as f:
        json.dump(row, f, indent=2)
    return row


# ============================================================
# Sweep drivers
# ============================================================

def _work_items(configs, seeds):
    """
    Flatten the sweep into an ordered (index, cfg, seed) list.

    The order is fixed and shard-independent, so `--merge` can rebuild the
    combined table in exactly the order a serial run would have produced.
    """
    return [
        (i, cfg, seed)
        for i, (cfg, seed) in enumerate(
            (cfg, seed) for cfg in configs for seed in seeds
        )
    ]


def _shard_filter(gate, items, maxiter, s_idx, s_cnt, force=False,
                  protocol="single_phase", phase=None):
    """
    Split work items across shards, balancing the items that actually cost
    something.

    `phase=1` counts an item as "done" once EITHER the final pulse exists
    (some other run already finished it, phase 1 is moot) OR phase 1's own
    snapshot exists -- unlike `phase=2`/`None`, which only look at the final
    pulse, since that is the only file Stage A is responsible for producing.

    Round-robin on the raw work-item index is wrong here. `_work_items` orders
    by (config, seed), so the index stride through seeds is `len(seeds)` -- and
    when that equals the shard count, `i % s_cnt` hands every shard exactly one
    seed. On a resumed sweep (seed 42 cached, 43 and 44 new) that put all the
    free work in shard 0, which finished instantly, and split the real work
    across the remaining two. Measured, not hypothetical.

    So partition the untrained and the already-cached items separately, each
    round-robin. Cached rows cost a file read, so their distribution is
    irrelevant; what has to come out even is the training.
    """
    todo, cached = [], []
    for it in items:
        _, cfg, seed = it
        h = _config_hash(gate, cfg["penalties"], seed, maxiter,
                         amp_max=cfg.get("amp_max", FIXED["amp_max"]),
                         protocol=protocol)
        pulse_path, _ = _cache_paths(gate, h)
        done = os.path.exists(pulse_path)
        if phase == 1:
            done = done or os.path.exists(_snapshot_path(gate, h, "phase1"))
        (cached if (done and not force) else todo).append(it)

    keep = [it for k, it in enumerate(todo) if k % s_cnt == s_idx]
    keep += [it for k, it in enumerate(cached) if k % s_cnt == s_idx]
    # Restore the canonical order so a shard's own partial CSV stays readable.
    return sorted(keep, key=lambda it: it[0]), len(todo)


def run_sweep(gate, configs, seeds, maxiter, eval_truncs, out_csv, manifest_path,
              n_jobs=3, verbose=False, force=False, shard=None,
              protocol="single_phase", phase=None):
    """
    shard : (index, count) or None
        Run this shard's slice of the work, balanced by `_shard_filter`. Shards
        share the on-disk cache but write disjoint rows (one `row_<hash>.json`
        per config+seed), so K shards can run as K concurrent processes with no
        locking. Rebuild the combined table afterwards with `merge_sweep`.

    phase : {None, 1, 2}
        Only meaningful for `protocol="two_phase"`. `phase=1` (Stage A) trains
        only phase 1 for every item and returns without writing a CSV/manifest
        -- there is no scored row yet, only phase-1 snapshots on disk (the
        actual completion record `--phase 2` reads). `phase=2` (Stage B) and
        the default `phase=None` both produce the usual CSV + manifest.
    """
    import pandas as pd

    items = _work_items(configs, seeds)
    if shard is not None:
        s_idx, s_cnt = shard
        items, n_todo = _shard_filter(gate, items, maxiter, s_idx, s_cnt,
                                      force=force, protocol=protocol, phase=phase)
        print(f"shard {s_idx} of {s_cnt}: {len(items)} items "
              f"({n_todo} of {len(_work_items(configs, seeds))} need training, "
              f"split evenly across shards)", flush=True)

    total = len(items)
    t_start = time.time()

    if phase == 1:
        # Stage A: no CSV/manifest -- the phase-1 snapshot + meta files this
        # writes to results/penalty_sweep_cache/ ARE the completion record.
        n_trained = n_cached = 0
        for i, (_, cfg, seed) in enumerate(items, start=1):
            status = run_one(
                gate, cfg, seed, maxiter, eval_truncs,
                n_jobs=n_jobs, verbose=verbose, force=force,
                protocol=protocol, phase=1,
            )
            # Three possible shapes: a fresh/cached phase-1-only status dict
            # (has "status"), or -- if the FINAL scored row already existed
            # -- the full row dict from run_one's top-level cache shortcut
            # (has "cached" instead, no "status" key).
            label = status.get("status") or (
                "final_cached" if status.get("cached") else "unknown")
            if label == "trained":
                n_trained += 1
            else:
                n_cached += 1
            print(f"[{i}/{total}] gate={gate} seed={seed} {cfg['label']} "
                  f"-> {label}", flush=True)
        elapsed = time.time() - t_start
        print(f"\nStage A (phase 1) done: {n_trained} trained, "
              f"{n_cached} already done. Wall time: {elapsed/60:.1f} min")
        print("Run --phase 2 next to train phase 2 for these items.")
        return []

    rows = []
    done = 0

    for _, cfg, seed in items:
        done += 1
        print(
            f"[{done}/{total}] gate={gate} seed={seed} {cfg['label']}",
            flush=True,
        )
        row = run_one(
            gate, cfg, seed, maxiter, eval_truncs,
            n_jobs=n_jobs, verbose=verbose, force=force, protocol=protocol,
            phase=phase,
        )
        tag = "cache" if row.get("cached") else (
            "rescored" if row.get("reused_pulse") else "trained"
        )
        print(
            f"    [{tag}] F_held={row['F_ped_heldout_mean']:.6f} "
            f"min={row['F_ped_heldout_min']:.6f} "
            f"spread={row['robustness_spread']:.3e} "
            f"occ99={row.get('occ99_frac', float('nan')):.0%} of own mask "
            f"({row.get('binding_drive', '?')}; "
            f"{row.get('occ99_MHz', float('nan')):.1f} MHz) "
            f"peak={row['peak_amp']:.1f}",
            flush=True,
        )
        rows.append(row)

        # Write incrementally so an interrupted sweep still leaves a usable
        # table behind.
        df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                           for r in rows])
        os.makedirs(TABLE_DIR, exist_ok=True)
        df.to_csv(out_csv, index=False)

    elapsed = time.time() - t_start

    trained_truncs = sorted(
        set(TWO_PHASE["phase1_trunc_list"]) | set(TWO_PHASE["phase2_trunc_list"])
        if protocol == "two_phase" else set(FIXED["trunc_list"])
    )

    os.makedirs(RESULT_DIR, exist_ok=True)
    manifest = {
        "gate": gate,
        "n_configs": len(configs),
        "seeds": list(seeds),
        "maxiter": maxiter,
        "protocol": protocol,
        "eval_truncs": list(eval_truncs),
        "trained_truncs": trained_truncs,
        "heldout_truncs": [nc for nc in eval_truncs if nc not in trained_truncs],
        "fixed": {k: (list(v) if isinstance(v, tuple) else v) for k, v in FIXED.items()},
        **({"two_phase": TWO_PHASE} if protocol == "two_phase" else {}),
        "baseline": BASELINE,
        "csv": os.path.relpath(out_csv, REPO_ROOT),
        "elapsed_s": elapsed,
        "shard": list(shard) if shard is not None else None,
        "n_rows": len(rows),
        "configs": [
            {"label": c["label"], "swept": c["swept"], "penalties": c["penalties"],
             "amp_max": c.get("amp_max", FIXED["amp_max"])}
            for c in configs
        ],
        "per_trunc": {
            f"{r['label']}|seed={r['seed']}": r.get("_per_trunc", {}) for r in rows
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {out_csv}")
    print(f"Wrote {manifest_path}")
    print(f"Total wall time: {elapsed/60:.1f} min")
    return rows


def merge_sweep(gate, configs, seeds, maxiter, eval_truncs, out_csv, manifest_path,
                protocol="single_phase"):
    """
    Rebuild the combined CSV + manifest from the row cache after a sharded run.

    Reads `row_<hash>.json` rather than concatenating the shard CSVs: the cache
    is the authority, it already carries `_per_trunc` for the manifest, and
    reading it restores the serial work-item order regardless of which shard
    produced which row or in what sequence the shards finished.

    Refuses to write a partial table -- a missing or stale row means a shard
    died or was run with different settings, and silently emitting a short CSV
    is exactly how a truncated sweep gets mistaken for a complete one.
    """
    import pandas as pd

    rows, missing = [], []
    for _, cfg, seed in _work_items(configs, seeds):
        amp_max = cfg.get("amp_max", FIXED["amp_max"])
        h = _config_hash(gate, cfg["penalties"], seed, maxiter, amp_max=amp_max,
                         protocol=protocol)
        want_key = _config_hash(
            gate, cfg["penalties"], seed, maxiter, amp_max=amp_max,
            extra={"eval": list(eval_truncs)}, protocol=protocol,
        )
        _, row_path = _cache_paths(gate, h)
        if not os.path.exists(row_path):
            missing.append((cfg["label"], seed, "no cached row"))
            continue
        with open(row_path) as f:
            row = json.load(f)
        if row.get("_score_key") != want_key:
            missing.append((cfg["label"], seed, "scored on a different trunc set"))
            continue
        row["cached"] = True
        rows.append(row)

    if missing:
        print(f"error: {len(missing)} of {len(rows) + len(missing)} rows unavailable:")
        for label, seed, why in missing[:20]:
            print(f"  seed={seed} {label}  ({why})")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        raise SystemExit(
            "refusing to write a partial table -- rerun the shards that failed"
        )

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in rows])
    os.makedirs(TABLE_DIR, exist_ok=True)
    df.to_csv(out_csv, index=False)

    trained_truncs = sorted(
        set(TWO_PHASE["phase1_trunc_list"]) | set(TWO_PHASE["phase2_trunc_list"])
        if protocol == "two_phase" else set(FIXED["trunc_list"])
    )

    os.makedirs(RESULT_DIR, exist_ok=True)
    manifest = {
        "gate": gate,
        "n_configs": len(configs),
        "seeds": list(seeds),
        "maxiter": maxiter,
        "protocol": protocol,
        "eval_truncs": list(eval_truncs),
        "trained_truncs": trained_truncs,
        "heldout_truncs": [nc for nc in eval_truncs if nc not in trained_truncs],
        "fixed": {k: (list(v) if isinstance(v, tuple) else v) for k, v in FIXED.items()},
        **({"two_phase": TWO_PHASE} if protocol == "two_phase" else {}),
        "baseline": BASELINE,
        "csv": os.path.relpath(out_csv, REPO_ROOT),
        "elapsed_s": float("nan"),   # merged from shards; no single wall time
        "shard": "merged",
        "n_rows": len(rows),
        "configs": [
            {"label": c["label"], "swept": c["swept"], "penalties": c["penalties"],
             "amp_max": c.get("amp_max", FIXED["amp_max"])}
            for c in configs
        ],
        "per_trunc": {
            f"{r['label']}|seed={r['seed']}": r.get("_per_trunc", {}) for r in rows
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Merged {len(rows)} rows from the cache")
    print(f"Wrote {out_csv}")
    print(f"Wrote {manifest_path}")
    return rows


def _parse_shard(spec):
    """Parse a '--shard I/K' spec into a validated (index, count) pair."""
    try:
        i_str, k_str = spec.split("/")
        i, k = int(i_str), int(k_str)
    except ValueError:
        raise SystemExit(f"error: --shard must look like 'I/K', got {spec!r}")
    if k < 1 or not (0 <= i < k):
        raise SystemExit(f"error: --shard I/K needs K >= 1 and 0 <= I < K, got {spec!r}")
    return i, k


def build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gate", default="X", choices=sorted(GATE_FACTORIES))
    p.add_argument("--mode", default="ofat", choices=["ofat", "grid", "disc-null"])
    # Default grid is deriv x boundary. `disc` is no longer a legal axis at all
    # (provably inert -- see the note above PENALTY_NAMES), so the original
    # deriv x disc plan is not merely discouraged, it cannot be selected.
    p.add_argument("--grid-x", default="deriv", choices=AXIS_NAMES)
    p.add_argument("--grid-y", default="boundary", choices=AXIS_NAMES)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--maxiter", type=int, default=1500)
    p.add_argument("--protocol", default="single_phase",
                   choices=["single_phase", "two_phase"],
                   help="single_phase = FIXED['trunc_list']/--maxiter, one "
                        "continuous run (the production recipe). two_phase = "
                        "TWO_PHASE's cold-start [22,26]@1000-iter phase "
                        "followed by a warm-started [26]-only 1500-iter phase "
                        "(--maxiter is ignored). Distinct cache/hash space from "
                        "single_phase, so switching never touches existing "
                        "pulses.")
    p.add_argument("--phase", type=int, default=None, choices=[1, 2],
                   help="Only valid with --protocol two_phase. Splits the "
                        "phase1+phase2 chain into two independently-"
                        "schedulable stages instead of one process per item: "
                        "1 (Stage A) trains only phase 1 (n_jobs=2/item, so "
                        "outer --shard concurrency should be ~cores/2) and "
                        "writes no CSV -- the phase-1 snapshot on disk is the "
                        "completion record. 2 (Stage B) trains only phase 2 "
                        "(n_jobs=1/item, so --shard concurrency can go up to "
                        "~cores) and requires Stage A to have already run for "
                        "every item (errors otherwise, rather than silently "
                        "training phase 1 inline). --n-jobs is ignored either "
                        "way. Omit --phase entirely to keep the original "
                        "chained single-process-per-item behavior.")
    p.add_argument("--eval-truncs", type=int, nargs="+", default=DEFAULT_EVAL_TRUNCS,
                   help="Cavity truncations to SCORE on. Those also in "
                        "trunc_list are reported as 'trained'; the rest are held out.")
    p.add_argument("--n-jobs", type=int, default=3)
    p.add_argument("--tag", default="", help="Suffix for output filenames, e.g. 'demo'.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only run the first K configs (smoke tests).")
    p.add_argument("--force", action="store_true", help="Ignore cache and retrain.")
    p.add_argument("--shard", default=None, metavar="I/K",
                   help="Run shard I of K so K shards can run concurrently. The "
                        "split balances the configs that actually need training, "
                        "not the raw item count. Writes <stem>_shard<I>of<K>.csv; "
                        "combine with --merge afterwards.")
    p.add_argument("--merge", action="store_true",
                   help="Rebuild the combined CSV + manifest from the row cache "
                        "after a sharded run. Trains nothing.")
    p.add_argument("--verbose", action="store_true", help="Optimizer chatter.")
    return p


def main():
    args = build_arg_parser().parse_args()

    if args.mode == "ofat":
        configs = build_ofat_configs()
        stem = f"penalty_sweep_{args.gate}_ofat"
    elif args.mode == "disc-null":
        configs = build_disc_null_configs()
        stem = f"penalty_sweep_{args.gate}_disc_null"
        # Cache-only by design. This mode reconstructs a RETIRED null experiment
        # from pulses that are already on disk; it is not a live axis, and
        # training it at a seed it was never run at would silently spend hours
        # extending an experiment whose whole point is that it measures nothing.
        # (Asking for `--seeds 42 43 44` here once cost 10 minutes before it was
        # caught -- hence the guard rather than a comment.)
        untrained = [
            (c["label"], s) for c in configs for s in args.seeds
            if not os.path.exists(_cache_paths(
                args.gate,
                _config_hash(args.gate, c["penalties"], s, args.maxiter,
                             amp_max=c.get("amp_max", FIXED["amp_max"]),
                             protocol=args.protocol))[0])
        ]
        if untrained and not args.force:
            print(f"error: --mode disc-null is cache-only, but {len(untrained)} "
                  f"config/seed pairs have no trained pulse:")
            for label, s in untrained[:6]:
                print(f"  seed={s} {label}")
            raise SystemExit(
                "The disc ladder is a retired null experiment kept for its noise "
                "estimate, not an axis to extend. Re-run with only the seeds it "
                "was originally trained at (--seeds 42), or pass --force if you "
                "really intend to train new ones."
            )
    else:
        if args.grid_x == args.grid_y:
            raise SystemExit("error: --grid-x and --grid-y must differ")
        configs = build_grid_configs(args.grid_x, args.grid_y)
        stem = f"penalty_sweep_{args.gate}_grid_{args.grid_x}_{args.grid_y}"
        # The grid shape is part of the filename. GRID_VALUES was trimmed from
        # 5x5 to 3x3 (see its definition), and without this the trimmed run
        # would silently overwrite the retained 25-row 5x5 table with 9 rows --
        # the table the notebook loads and the report's one-dimensionality
        # finding rests on. The historical file keeps its unsuffixed name.
        shape = f"{len(GRID_VALUES[args.grid_x])}x{len(GRID_VALUES[args.grid_y])}"
        if shape != "5x5":
            stem = f"{stem}_{shape}"

    if args.tag:
        stem = f"{stem}_{args.tag}"
    if args.limit:
        configs = configs[: args.limit]

    if args.shard and args.merge:
        raise SystemExit("error: --shard and --merge are mutually exclusive")
    if args.phase is not None and args.protocol != "two_phase":
        raise SystemExit("error: --phase requires --protocol two_phase")
    if args.phase == 1 and args.merge:
        raise SystemExit(
            "error: --phase 1 --merge is meaningless -- Stage A never writes "
            "a scored row, so there is nothing for --merge to combine. Merge "
            "after --phase 2 instead."
        )
    shard = _parse_shard(args.shard) if args.shard else None

    # Shards write their own partial tables; only the base stem is the
    # authoritative combined output, and only --merge may write it.
    stem_out = f"{stem}_shard{shard[0]}of{shard[1]}" if shard else stem
    out_csv = os.path.join(TABLE_DIR, f"{stem_out}.csv")
    manifest = os.path.join(RESULT_DIR, f"{stem_out}.json")

    if args.protocol == "two_phase":
        trained_truncs = sorted(set(TWO_PHASE["phase1_trunc_list"])
                                | set(TWO_PHASE["phase2_trunc_list"]))
    else:
        trained_truncs = FIXED["trunc_list"]
    held = [nc for nc in args.eval_truncs if nc not in trained_truncs]
    print(f"mode={args.mode}  configs={len(configs)}  seeds={args.seeds}  "
          f"protocol={args.protocol}  maxiter={args.maxiter}")
    print(f"trained truncs {trained_truncs}  |  held-out {held}")

    if args.merge:
        merge_sweep(
            args.gate, configs, args.seeds, args.maxiter, args.eval_truncs,
            out_csv, manifest, protocol=args.protocol,
        )
        return

    run_sweep(
        args.gate, configs, args.seeds, args.maxiter, args.eval_truncs,
        out_csv, manifest, n_jobs=args.n_jobs, verbose=args.verbose,
        force=args.force, shard=shard, protocol=args.protocol,
        phase=args.phase,
    )


if __name__ == "__main__":
    main()
