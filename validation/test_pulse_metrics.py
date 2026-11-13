#!/usr/bin/env python3
"""
validation/test_pulse_metrics.py

Acceptance tests for the spectral cost metrics in
`analysis.penalty_sweep.pulse_metrics`, and for the selection rule they feed in
`visualization.penalty_viz.recommend`.

What is actually being pinned here
----------------------------------
`bandwidth_MHz` -- the RMS frequency about f = 0 -- was the scored pulse cost.
By Koenig-Huygens it decomposes exactly as

    <f^2> = <f>^2 + sigma_f^2,

so it folds the spectral centroid <f> into a quantity reported as a "width".
The centroid is a **gauge** choice, not a physical property: the lab-frame field
is Re[eps(t) e^{i w_LO t}], and (eps(t) e^{-2 pi i f0 t}, w_LO + 2 pi f0)
describes the identical field. Retuning the LO -- routine, free -- moves <f>
and therefore moves the reported bandwidth, with no change to the waveform.

`test_gauge_invariance` is the anchor of this file: it applies exactly that
transformation and asserts that `sigma_f` and `occ99` do not move while
`bandwidth_MHz` does. Everything the metric change rests on is in that test.

Run from the repo root:

    python validation/test_pulse_metrics.py -v
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis.penalty_sweep import occupied_width, pulse_metrics  # noqa: E402
from core.grape_core import chi, two_pi  # noqa: E402

N = 550
DT = 0.002                  # us
BIN = 1.0 / (N * DT)        # 0.909 MHz FFT bin spacing
CACHE = os.path.join(REPO_ROOT, "results", "penalty_sweep_cache", "X")

# Cached sweep pulses used for the regression tests: the incumbent recipe and
# the lambda_deriv = 0 negative control.
INCUMBENT_HASH = "024ea7f9987444a0"
CONTROL_HASH = "651a62fe533443d4"


def _pulse_path(h):
    return os.path.join(CACHE, f"u_{h}.npy")


def _load_or_skip(test, h):
    path = _pulse_path(h)
    if not os.path.exists(path):
        test.skipTest(
            f"cached pulse {path} absent -- regenerate with "
            "`python analysis/penalty_sweep.py --gate X --mode ofat "
            "--seeds 42 --maxiter 1500`"
        )
    return np.load(path)


def _pulse_from_drives(eps_c, eps_t):
    """
    Pack two complex drives into the (N, 4) control array.

    Matches `pulse_metrics`' own convention: the transmon columns are read back
    as u[:,2] + 1j*u[:,3], so this is its exact inverse.
    """
    u = np.empty((len(eps_c), 4))
    u[:, 0], u[:, 1] = eps_c.real, eps_c.imag
    u[:, 2], u[:, 3] = eps_t.real, eps_t.imag
    return u


def _smooth_random_drive(rng, n=N, cutoff=12):
    """A band-limited random complex waveform: a few low-frequency Fourier bins."""
    spec = np.zeros(n, dtype=complex)
    idx = np.fft.fftfreq(n, d=DT)
    keep = np.abs(idx) <= cutoff
    spec[keep] = rng.normal(size=keep.sum()) + 1j * rng.normal(size=keep.sum())
    return np.fft.ifft(spec) * n


def _drive_with_power_spectrum(power):
    """
    A complex drive whose DFT power spectrum is exactly `power`.

    Setting the DFT to sqrt(power) and inverting means `pulse_metrics`' own
    forward transform recovers it exactly -- no windowing, no leakage. Needed
    wherever a test asserts on a *moment*: the first and second moments are
    tail-weighted, so a tone placed off the bin grid biases them by percent
    even though the waveform looks monochromatic.
    """
    return np.fft.ifft(np.sqrt(np.asarray(power, dtype=float))) * len(power)


def _gaussian_line(f_centre, width_MHz=2.5, n=N):
    """Power spectrum of a narrow line centred exactly at `f_centre`."""
    f = np.fft.fftfreq(n, d=DT)
    return np.exp(-0.5 * ((f - f_centre) / width_MHz) ** 2)


class KoenigHuygensTest(unittest.TestCase):
    """bandwidth_MHz^2 == centroid^2 + sigma_f^2, exactly, per drive."""

    def _check(self, u, tag):
        m = pulse_metrics(u, DT)
        for drive in ("cav", "tra"):
            rms = m[f"bandwidth_{drive}_MHz"]
            cen = m[f"centroid_{drive}_MHz"]
            sig = m[f"sigma_f_{drive}_MHz"]
            self.assertAlmostEqual(
                rms ** 2, cen ** 2 + sig ** 2, delta=1e-10 * max(rms ** 2, 1.0),
                msg=f"{tag}/{drive}: raw second moment != <f>^2 + sigma^2",
            )

    def test_random_pulses(self):
        rng = np.random.default_rng(0)
        for k in range(5):
            u = _pulse_from_drives(_smooth_random_drive(rng),
                                   _smooth_random_drive(rng))
            self._check(u, f"random[{k}]")

    def test_real_pulses(self):
        for tag, h in (("incumbent", INCUMBENT_HASH), ("control", CONTROL_HASH)):
            self._check(_load_or_skip(self, h), tag)


class GaugeInvarianceTest(unittest.TestCase):
    """
    THE ANCHOR TEST.

    eps(t) -> eps(t) * exp(-2j*pi*f0*t) is a retuning of the drive LO by f0: the
    same physical field, described in a frame shifted by f0. A width must not
    notice it.
    """

    # Exact multiples of the FFT bin. On a length-N DFT a retune by k bins is a
    # cyclic rotation of the spectrum, so the invariance is exact and can be
    # asserted as such. `test_off_grid_retune` covers the general case.
    BIN_SHIFTS = (-13, -1, 3, 8)

    def _shift(self, eps, f0):
        t = np.arange(len(eps)) * DT
        return eps * np.exp(-2j * np.pi * f0 * t)

    def _retune_transmon(self, u, f0):
        """
        Retune the TRANSMON LO only.

        The two drives reach the device through separate synthesis chains with
        their own LOs, so a common shift of both is not a gauge transformation
        of anything -- and, because the headline metrics are a max over drives,
        it also silently swaps which drive the headline reports.
        """
        v = u.copy()
        eps = self._shift(u[:, 2] + 1j * u[:, 3], f0)
        v[:, 2], v[:, 3] = eps.real, eps.imag
        return v

    def test_shift_moves_centroid_and_bandwidth_but_not_width(self):
        rng = np.random.default_rng(11)
        u0 = _pulse_from_drives(_smooth_random_drive(rng),
                                _smooth_random_drive(rng))
        m0 = pulse_metrics(u0, DT)

        for k in self.BIN_SHIFTS:
            f0 = k * BIN
            m = pulse_metrics(self._retune_transmon(u0, f0), DT)

            # sigma_f and occ99: invariant to machine precision.
            self.assertAlmostEqual(
                m["sigma_f_tra_MHz"], m0["sigma_f_tra_MHz"], places=8,
                msg=f"k={k} bins: sigma_f moved under an LO retune",
            )
            self.assertAlmostEqual(
                m["occ99_tra_MHz"], m0["occ99_tra_MHz"], places=8,
                msg=f"k={k} bins: occ99 moved under an LO retune",
            )
            # centroid: shifts by exactly f0. `_shift` multiplies by
            # exp(-2j pi f0 t), which translates the spectrum DOWN by f0.
            self.assertAlmostEqual(
                m["centroid_tra_MHz"], m0["centroid_tra_MHz"] - f0, places=8,
                msg=f"k={k} bins: centroid did not track the retune",
            )
            # The cavity drive was not touched, so nothing about it moved.
            self.assertAlmostEqual(m["centroid_cav_MHz"], m0["centroid_cav_MHz"],
                                   places=10)

            # And the point of the exercise: the OLD metric does move.
            self.assertGreater(
                abs(m["bandwidth_tra_MHz"] - m0["bandwidth_tra_MHz"]), 0.1,
                msg=f"k={k} bins: bandwidth_MHz was expected to move under an "
                    "LO retune -- if it no longer does, this test is not "
                    "testing what it claims",
            )

    def test_real_pulse_is_gauge_invariant_too(self):
        """Same statement on a trained pulse rather than a synthetic one."""
        u0 = _load_or_skip(self, CONTROL_HASH)
        m0 = pulse_metrics(u0, DT)

        for k in self.BIN_SHIFTS:
            f0 = k * BIN
            m = pulse_metrics(self._retune_transmon(u0, f0), DT)
            self.assertAlmostEqual(m["sigma_f_tra_MHz"], m0["sigma_f_tra_MHz"],
                                   places=8)
            self.assertAlmostEqual(m["occ99_tra_MHz"], m0["occ99_tra_MHz"],
                                   places=8)
            self.assertAlmostEqual(m["centroid_tra_MHz"],
                                   m0["centroid_tra_MHz"] - f0, places=8)
            self.assertGreater(
                abs(m["bandwidth_tra_MHz"] - m0["bandwidth_tra_MHz"]), 0.1)

    def test_off_grid_retune(self):
        """
        The physically-quoted case: a 12 MHz retune, which is 13.2 FFT bins and
        so not a rotation of the DFT grid.

        Off-grid, the product waveform is no longer periodic on the sample
        window, and the resulting leakage has 1/f tails. `sigma_f` is an
        f^2-weighted statistic and `occ99` is not, so this bounds them
        differently -- but both stay put while `bandwidth_MHz` walks by 1.3 MHz.
        """
        u0 = _load_or_skip(self, CONTROL_HASH)
        m0 = pulse_metrics(u0, DT)
        m = pulse_metrics(self._retune_transmon(u0, -12.0), DT)

        self.assertAlmostEqual(m["sigma_f_tra_MHz"], m0["sigma_f_tra_MHz"],
                               delta=1e-3)
        self.assertLessEqual(abs(m["occ99_tra_MHz"] - m0["occ99_tra_MHz"]),
                             2 * BIN + 1e-9)
        # The reported RMS width falls from 16.90 to 15.59 MHz for a waveform
        # that is unchanged in every physical respect.
        self.assertAlmostEqual(m0["bandwidth_tra_MHz"], 16.904, places=2)
        self.assertLess(m["bandwidth_tra_MHz"], 16.0)


class DegenerateSpectrumTest(unittest.TestCase):
    """A pure tone has zero width by either construction."""

    def test_pure_tone(self):
        t = np.arange(N) * DT
        for f_tone in (0.0, 3 * BIN, -5 * BIN):
            # An exact bin frequency, so the DFT is a single non-zero sample.
            eps = 4.0 * np.exp(2j * np.pi * f_tone * t)
            m = pulse_metrics(_pulse_from_drives(eps, eps), DT)
            self.assertAlmostEqual(m["sigma_f_MHz"], 0.0, places=8)
            self.assertLessEqual(m["occ99_MHz"], BIN + 1e-9)
            self.assertAlmostEqual(m["centroid_tra_MHz"], f_tone, places=8)

    def test_zero_pulse_does_not_divide_by_zero(self):
        m = pulse_metrics(np.zeros((N, 4)), DT)
        for k in ("occ99_MHz", "sigma_f_MHz", "bandwidth_MHz", "centroid_tra_MHz",
                  # The constraint_report block divides an endpoint max by
                  # `peak_amp` and by the mid-third RMS, both zero here.
                  "endpoint_rel_to_peak", "endpoint_rel_to_mid",
                  "out_of_band_cav", "out_of_band_tra"):
            self.assertEqual(m[k], 0.0)


class MaskContainmentTest(unittest.TestCase):
    """
    A band-limited pulse cannot occupy more than its mask.

    This is the sanity bound that makes a budget stated as a *fraction of the
    mask* meaningful: the statistic is capped by the mask width by construction.
    """

    def test_occ99_within_mask(self):
        from core.fourier_cutoff import project_bandlimit

        rng = np.random.default_rng(3)
        u = rng.normal(size=(N, 4))
        u = project_bandlimit(u, DT, (-27.0, 27.0), (-33.0, 33.0))
        m = pulse_metrics(u, DT)
        self.assertLessEqual(m["occ99_cav_MHz"], 54.0 + 1e-9)
        self.assertLessEqual(m["occ99_tra_MHz"], 66.0 + 1e-9)

    def test_trained_pulses_within_mask(self):
        for h in (INCUMBENT_HASH, CONTROL_HASH):
            m = pulse_metrics(_load_or_skip(self, h), DT)
            self.assertLessEqual(m["occ99_cav_MHz"], 54.0 + 1e-9)
            self.assertLessEqual(m["occ99_tra_MHz"], 66.0 + 1e-9)


class SignConventionTest(unittest.TestCase):
    """
    Pins the sign-convention comment in `pulse_metrics`.

    Two sign flips cancel there -- the stored array is eps.conj(), and chi < 0
    puts photon-number peak n > 0 at negative code-frequency -- so
    `n_bar_drive = <f>_code / (chi/2pi)` is correct as written. "Fixing" either
    flip alone silently inverts the sign of every reported n_bar_drive, which
    this test catches.
    """

    def test_sign_of_n_bar_drive(self):
        chi_MHz = chi / two_pi                       # -2.194 MHz
        self.assertLess(chi_MHz, 0.0)                # the premise of the comment
        quiet = np.zeros(N, dtype=complex)

        for n in (1, 2, 3):
            # Power concentrated at f = +|chi| * n in CODE coordinates, i.e. in
            # the coordinates of the array pulse_metrics builds its FFT from.
            eps = _drive_with_power_spectrum(_gaussian_line(+abs(chi_MHz) * n))
            m = pulse_metrics(_pulse_from_drives(quiet, eps), DT)
            self.assertLess(
                m["n_bar_drive"], 0.0,
                msg=f"n={n}: +{abs(chi_MHz)*n:.3f} MHz in code coordinates must "
                    "give a NEGATIVE n_bar_drive",
            )
            self.assertAlmostEqual(m["n_bar_drive"], -n, places=6)

            eps = _drive_with_power_spectrum(_gaussian_line(-abs(chi_MHz) * n))
            m = pulse_metrics(_pulse_from_drives(quiet, eps), DT)
            self.assertGreater(
                m["n_bar_drive"], 0.0,
                msg=f"n={n}: -{abs(chi_MHz)*n:.3f} MHz in code coordinates must "
                    "give a POSITIVE n_bar_drive",
            )
            self.assertAlmostEqual(m["n_bar_drive"], +n, places=6)


class OccupiedWidthTest(unittest.TestCase):
    """The window really is the narrowest one carrying 99% of the power."""

    def test_matches_brute_force_on_a_trained_pulse(self):
        u = _load_or_skip(self, INCUMBENT_HASH)
        freqs = np.fft.fftfreq(len(u), d=DT)
        power = np.abs(np.fft.fft(u[:, 2] + 1j * u[:, 3])) ** 2
        p = power / power.sum()

        order = np.argsort(freqs)
        fs, ps = freqs[order], p[order]
        best = np.inf
        for a in range(len(fs)):            # O(N^2), fine at N = 550
            s = 0.0
            for b in range(a, len(fs)):
                s += ps[b]
                if s >= 0.99:
                    best = min(best, fs[b] - fs[a])
                    break

        width, lo, hi = occupied_width(power, freqs)
        self.assertAlmostEqual(width, best, places=9)
        self.assertAlmostEqual(hi - lo, width, places=12)
        # And the window it returns actually holds 99%.
        inside = (fs >= lo - 1e-9) & (fs <= hi + 1e-9)
        self.assertGreaterEqual(ps[inside].sum(), 0.99)


class CachedPulseRegressionTest(unittest.TestCase):
    """
    Numeric regression against the two cached pulses at the centre of the study.

    Transmon drive unless the key says `cav`. Values are as measured; the
    occupied widths are the true minimum-width windows (verified against brute
    force in `OccupiedWidthTest`).
    """

    EXPECTED = {
        INCUMBENT_HASH: {
            "bandwidth_tra_MHz": 8.7555, "centroid_tra_MHz": -4.4669,
            "sigma_f_tra_MHz": 7.5303, "occ99_tra_MHz": 33.6364,
            "tra_window": (-25.4545, 8.1818), "tra_mode": -2.7273,
            "bandwidth_cav_MHz": 5.1290, "centroid_cav_MHz": -0.2747,
            "sigma_f_cav_MHz": 5.1216, "occ99_cav_MHz": 25.4545,
        },
        CONTROL_HASH: {
            "bandwidth_tra_MHz": 16.9040, "centroid_tra_MHz": -12.0530,
            "sigma_f_tra_MHz": 11.8521, "occ99_tra_MHz": 61.8182,
            "tra_window": (-32.7273, 29.0909), "tra_mode": 0.9091,
            "bandwidth_cav_MHz": 9.8575, "centroid_cav_MHz": 0.0820,
            "sigma_f_cav_MHz": 9.8572, "occ99_cav_MHz": 42.7273,
        },
    }

    def test_regression(self):
        for h, exp in self.EXPECTED.items():
            u = _load_or_skip(self, h)
            m = pulse_metrics(u, DT)
            for key, want in exp.items():
                if key in ("tra_window", "tra_mode"):
                    continue
                self.assertAlmostEqual(
                    m[key], want, delta=abs(want) * 1e-3 + 1e-9,
                    msg=f"{h}: {key} = {m[key]:.4f}, expected {want:.4f}",
                )

            freqs = np.fft.fftfreq(len(u), d=DT)
            power = np.abs(np.fft.fft(u[:, 2] + 1j * u[:, 3])) ** 2
            _, lo, hi = occupied_width(power, freqs)
            self.assertAlmostEqual(lo, exp["tra_window"][0], places=3)
            self.assertAlmostEqual(hi, exp["tra_window"][1], places=3)
            self.assertAlmostEqual(freqs[np.argmax(power)], exp["tra_mode"],
                                   places=3)

    def test_headline_is_the_wider_drive(self):
        for h in self.EXPECTED:
            m = pulse_metrics(_load_or_skip(self, h), DT)
            for stem in ("occ99", "sigma_f", "bandwidth"):
                self.assertEqual(
                    m[f"{stem}_MHz"],
                    max(m[f"{stem}_cav_MHz"], m[f"{stem}_tra_MHz"]),
                )


class PerDriveAllowanceTest(unittest.TestCase):
    """
    The two drives are budgeted separately, against their own band masks.

    `max(occ99_cav_MHz, occ99_tra_MHz) <= tau` forces one threshold onto two
    constraints whose allowances differ (54 vs 66 MHz) and whose hardware chains
    are separate, so nothing physically experiences that max. Normalising each
    width by its own allowance restores the correct conjunction; that equivalence
    is `test_conjunction_equivalence`, and it is the whole reason a max is still
    the right aggregator.
    """

    def test_key_parity_between_zero_and_real_pulse(self):
        """
        `_METRIC_KEYS` is probed with an all-zero pulse, which takes the
        zero-power short-circuit inside `spectral_stats`. If that branch returned
        a reduced key set, the new columns would be missing from the probe and
        `run_one`'s cache backfill would never fire for them -- silently.
        """
        from analysis.penalty_sweep import _METRIC_KEYS

        zero = pulse_metrics(np.zeros((8, 4)), DT)
        real = pulse_metrics(_load_or_skip(self, INCUMBENT_HASH), DT)
        self.assertEqual(set(zero), set(real))
        self.assertEqual(set(_METRIC_KEYS), set(real))

    def test_fraction_definition(self):
        for h in (INCUMBENT_HASH, CONTROL_HASH):
            m = pulse_metrics(_load_or_skip(self, h), DT)
            self.assertAlmostEqual(m["occ99_frac_cav"] * 54.0,
                                   m["occ99_cav_MHz"], places=12)
            self.assertAlmostEqual(m["occ99_frac_tra"] * 66.0,
                                   m["occ99_tra_MHz"], places=12)
            self.assertEqual(m["occ99_frac"],
                             max(m["occ99_frac_cav"], m["occ99_frac_tra"]))
            self.assertEqual(m["band_limit_cav_MHz"], 54.0)
            self.assertEqual(m["band_limit_tra_MHz"], 66.0)
            expect = "cav" if m["occ99_frac_cav"] > m["occ99_frac_tra"] else "tra"
            self.assertEqual(m["binding_drive"], expect)

    def test_conjunction_equivalence(self):
        """max(x/a, y/b) <= t  <=>  (x <= t*a) AND (y <= t*b)."""
        rng = np.random.default_rng(5)
        for _ in range(200):
            x, y = rng.uniform(0, 54.0), rng.uniform(0, 66.0)
            frac = max(x / 54.0, y / 66.0)
            for t in (0.3, 0.5, 0.7, 0.9, frac):
                self.assertEqual(frac <= t,
                                 (x <= t * 54.0 + 1e-12) and (y <= t * 66.0 + 1e-12))

    def test_edges_bracket_the_width_and_lie_inside_the_mask(self):
        for h in (INCUMBENT_HASH, CONTROL_HASH):
            m = pulse_metrics(_load_or_skip(self, h), DT)
            for drive, half in (("cav", 27.0), ("tra", 33.0)):
                lo = m[f"occ99_{drive}_lo_MHz"]
                hi = m[f"occ99_{drive}_hi_MHz"]
                self.assertAlmostEqual(hi - lo, m[f"occ99_{drive}_MHz"], places=12)
                self.assertGreaterEqual(lo, -half - 1e-9)
                self.assertLessEqual(hi, half + 1e-9)

    def test_custom_allowances_scale_only_the_fractions(self):
        u = _load_or_skip(self, CONTROL_HASH)
        base = pulse_metrics(u, DT)
        wide = pulse_metrics(u, DT, band_limits={"cav": 108.0, "tra": 132.0})

        for drive in ("cav", "tra"):
            self.assertAlmostEqual(wide[f"occ99_frac_{drive}"],
                                   base[f"occ99_frac_{drive}"] / 2, places=12)
            # Every MHz column is untouched by the allowance.
            for stem in ("occ99_{}_MHz", "occ99_{}_lo_MHz", "occ99_{}_hi_MHz",
                         "bandwidth_{}_MHz", "sigma_f_{}_MHz"):
                key = stem.format(drive)
                self.assertEqual(wide[key], base[key])
        self.assertAlmostEqual(wide["occ99_frac"], base["occ99_frac"] / 2,
                               places=12)
        self.assertEqual(wide["band_limit_cav_MHz"], 108.0)

    def test_fractions_are_gauge_invariant(self):
        """
        A constant times an invariant quantity is invariant. Cheap to assert and
        it pins the fractions to the same guarantee the widths carry.
        """
        u0 = _load_or_skip(self, CONTROL_HASH)
        m0 = pulse_metrics(u0, DT)
        t = np.arange(len(u0)) * DT
        for k in (-13, 3, 8):
            v = u0.copy()
            eps = (u0[:, 2] + 1j * u0[:, 3]) * np.exp(-2j * np.pi * (k * BIN) * t)
            v[:, 2], v[:, 3] = eps.real, eps.imag
            m = pulse_metrics(v, DT)
            self.assertAlmostEqual(m["occ99_frac_tra"], m0["occ99_frac_tra"],
                                   places=10)
            self.assertAlmostEqual(m["occ99_frac"], m0["occ99_frac"], places=10)


class SelectionTest(unittest.TestCase):
    """
    Anti-regression on the selection rule itself.

    A budget ported to a new statistic without rescaling goes vacuous, and a
    vacuous constraint hands the recommendation to the highest-fidelity pulse in
    the study -- which is a lambda_deriv = 0 negative control, the exact
    configuration this sweep exists to reject. That failure is silent in every
    figure, so it is asserted here.
    """

    TABLES = ("tables/penalty_sweep_X_ofat.csv",
              "tables/penalty_sweep_X_grid_deriv_boundary.csv")

    # The fixtures below describe the FROZEN PRE-RAMP study, whose recommendation
    # was derived at a 70% budget with no fidelity floor. The module default has
    # since tightened to `pv.BUDGET_FRAC = 0.50` plus `pv.MIN_FIDELITY = 0.995`,
    # under which this pre-ramp table re-selects the incumbent itself. Pin the
    # historical rule explicitly rather than re-pinning the fixtures: what these
    # tests guard is the vacuous-budget trap, not the value of the threshold.
    # `test_current_default_rule_is_live` below covers the current default.
    PINNED_BUDGET = 0.70
    NO_FLOOR = 0.0

    def _agg(self):
        import pandas as pd
        import visualization.penalty_viz as pv

        frames = []
        for rel in self.TABLES:
            path = os.path.join(REPO_ROOT, rel)
            if not os.path.exists(path):
                self.skipTest(f"{rel} absent -- run analysis/penalty_sweep.py")
            frames.append(pd.read_csv(path))
        df = pd.concat(frames, ignore_index=True)
        if "occ99_frac" not in df.columns:
            self.skipTest("sweep CSVs predate occ99_frac -- re-run the sweep "
                          "(cached rows backfill without retraining)")
        return pv, pv.aggregate_seeds(df)

    def test_recommendation_is_not_a_zero_deriv_control(self):
        pv, agg = self._agg()
        rec = pv.recommend(agg, max_cost=self.PINNED_BUDGET,
                           min_fidelity=self.NO_FLOOR)
        self.assertNotEqual(
            float(rec["lambda_deriv"]), 0.0,
            msg=f"recommended {rec['label']!r} has lambda_deriv = 0 -- the "
                "budget has gone vacuous; check max_cost against the scale of "
                f"{rec['cost_metric']}",
        )

    RECOMMENDED = "deriv=1e-06_boundary=2e-05_amp=8e-05_disc=0.5"
    INCUMBENT = "deriv=1e-05_boundary=2e-05_amp=8e-05_disc=0.5"

    def test_recommendation_is_stable_at_the_pinned_budget(self):
        pv, agg = self._agg()
        rec = pv.recommend(agg, max_cost=self.PINNED_BUDGET,
                           min_fidelity=self.NO_FLOOR)
        self.assertEqual(rec["label"], self.RECOMMENDED)
        self.assertEqual(rec["cost_metric"], "occ99_frac")
        self.assertEqual(rec["cost"], rec["bandwidth"])   # legacy alias

        # The recipe is the claim; the fidelity is a seed-average that moves as
        # seeds are added (0.998716 at n=1, 0.999236 at n=3). Assert what is
        # actually being asserted -- that the recommendation beats the incumbent
        # -- rather than re-pinning a number that will drift again.
        incumbent = agg[agg["label"] == self.INCUMBENT]
        self.assertFalse(incumbent.empty, "incumbent row missing from the sweep")
        self.assertGreater(rec["F"],
                           float(incumbent.iloc[0]["F_ped_heldout_mean"]))

    def test_all_statistics_agree_at_equal_strictness(self):
        """
        The metric work does not change the answer -- it makes the answer
        defensible. Calibrated to equal strictness, all four candidate cost
        statistics select the same recipe.

        `equal strictness` is enforced here by calibrating each statistic to the
        SAME number of feasible configurations, via a quantile of that
        statistic's own distribution. The earlier version of this test hardcoded
        hand-tuned absolute thresholds (bandwidth_MHz <= 16.0 and friends) that
        happened to admit equal counts on the single-seed data. They stopped
        being equal once seeds were replicated: at 16.0, `bandwidth_MHz` admitted
        one config more than the others, that config was a lambda_deriv = 0
        negative control, and it took the recommendation -- the exact failure
        this class exists to catch, arriving through the calibration rather than
        through the budget. A quantile cannot drift out of calibration that way.
        """
        pv, agg = self._agg()
        n_ref = int((agg["occ99_frac"] <= self.PINNED_BUDGET).sum())
        self.assertGreater(n_ref, 0)
        self.assertLess(n_ref, len(agg), "budget is vacuous -- nothing rejected")
        q = n_ref / len(agg)

        for cost in ("bandwidth_MHz", "sigma_f_MHz", "occ99_MHz", "occ99_frac"):
            budget = float(agg[cost].quantile(q, interpolation="lower"))
            self.assertEqual(int((agg[cost] <= budget).sum()), n_ref,
                             msg=f"{cost} not calibrated to {n_ref} feasible")
            rec = pv.recommend(agg, max_cost=budget, cost=cost,
                               min_fidelity=self.NO_FLOOR)
            self.assertEqual(rec["label"], self.RECOMMENDED,
                             msg=f"{cost} <= {budget}")
            self.assertNotEqual(float(rec["lambda_deriv"]), 0.0,
                                msg=f"{cost} <= {budget} selected a "
                                    "lambda_deriv = 0 control")

    def test_binding_drive_is_not_constant(self):
        """
        The per-drive path must stay live.

        If every configuration binds on the transmon, normalising per drive has
        no observable consequence and a regression that collapsed the two back
        into one allowance would pass every other test in this file. Two grid
        configurations bind on the cavity; `deriv=0_boundary=6e-06` is the
        instructive one -- its cavity drive occupies 87.5% of the cavity mask
        while its transmon drive sits at 66.1%, so a transmon-only rule would
        have admitted a lambda_deriv = 0 negative control.
        """
        pv, agg = self._agg()
        counts = agg["binding_drive"].value_counts().to_dict()
        self.assertGreaterEqual(counts.get("cav", 0), 1, msg=f"counts={counts}")

        cav_bound = agg[agg["binding_drive"] == "cav"]
        self.assertTrue(
            (cav_bound["occ99_frac_cav"] > cav_bound["occ99_frac_tra"]).all())

        row = agg[agg["label"] == "deriv=0_boundary=6e-06_amp=8e-05_disc=0.5"]
        if not row.empty:
            row = row.iloc[0]
            self.assertEqual(row["binding_drive"], "cav")
            # Against the PINNED (pre-ramp) budget: at the current 50% default
            # the transmon share is over budget too, so the "a transmon-only rule
            # would have admitted this control" reading holds only at 70%.
            self.assertGreater(row["occ99_frac_cav"], self.PINNED_BUDGET)
            self.assertLess(row["occ99_frac_tra"], self.PINNED_BUDGET)
            self.assertEqual(float(row["lambda_deriv"]), 0.0)

    def test_current_default_rule_is_live(self):
        """
        The rule as it ships today -- `pv.DEFAULT_MAX_COST` plus
        `pv.DEFAULT_MIN_FIDELITY` -- must still bite and must still reject the
        negative control.

        Two ways this goes wrong silently: a threshold so loose that nothing is
        rejected (the vacuous-budget trap the rest of this class guards against
        at the pinned budget), or a floor so loose it is decorative. Assert both
        constraints are non-vacuous, and that neither has been tightened to the
        point where nothing survives.
        """
        pv, agg = self._agg()
        n_budget = int((agg["occ99_frac"] <= pv.DEFAULT_MAX_COST).sum())
        n_floor = int((agg["F_ped_heldout_mean"] >= pv.DEFAULT_MIN_FIDELITY).sum())

        self.assertGreater(n_budget, 0, "budget rejects everything")
        self.assertLess(n_budget, len(agg), "budget is vacuous -- nothing rejected")
        self.assertGreater(n_floor, 0, "fidelity floor rejects everything")
        self.assertLess(n_floor, len(agg),
                        "fidelity floor is vacuous -- nothing rejected")

        rec = pv.recommend(agg)
        self.assertNotEqual(
            float(rec["lambda_deriv"]), 0.0,
            msg=f"recommended {rec['label']!r} has lambda_deriv = 0 under the "
                "shipped default rule",
        )
        self.assertLessEqual(rec["cost"], pv.DEFAULT_MAX_COST)
        self.assertGreaterEqual(rec["F"], pv.DEFAULT_MIN_FIDELITY)

    def test_floor_is_applied_not_just_drawn(self):
        """
        The floor must be a filter, not decoration.

        Constructed rather than sampled: rank the budget-feasible configs, then
        put the floor between the top two and above the top one. A floor that is
        only drawn would return the same winner in both cases.
        """
        pv, agg = self._agg()
        feasible = agg[agg["occ99_frac"] <= self.PINNED_BUDGET]
        ranked = feasible.sort_values("F_ped_heldout_mean", ascending=False)
        best, second = ranked.iloc[0], ranked.iloc[1]

        # Between the top two: the winner is unchanged, but the runner-up is now
        # out of the feasible set.
        mid = 0.5 * (float(best["F_ped_heldout_mean"])
                     + float(second["F_ped_heldout_mean"]))
        rec = pv.recommend(agg, max_cost=self.PINNED_BUDGET, min_fidelity=mid)
        self.assertEqual(rec["label"], best["label"])

        # Above the top one: nothing survives, and the error says so.
        with self.assertRaises(ValueError) as ctx:
            pv.recommend(agg, max_cost=self.PINNED_BUDGET,
                         min_fidelity=float(best["F_ped_heldout_mean"]) + 1e-6)
        self.assertIn("F_ped_heldout_mean", str(ctx.exception))

    def test_naive_port_of_the_old_budget_is_vacuous(self):
        """
        Documents the trap the fraction-of-mask form exists to prevent: 16.0
        carried over to sigma_f admits everything and picks a control.
        """
        pv, agg = self._agg()
        self.assertEqual(int((agg["sigma_f_MHz"] <= 16.0).sum()), len(agg))
        rec = pv.recommend(agg, max_cost=16.0, cost="sigma_f_MHz",
                           min_fidelity=self.NO_FLOOR)
        self.assertEqual(float(rec["lambda_deriv"]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
