#!/usr/bin/env python3
"""
visualization/penalty_viz.py

Figures and selection logic for the penalty-weight Pareto sweep produced by
`analysis/penalty_sweep.py`.

The sweep is multi-objective: reported (held-out) gate fidelity, F_held,
against pulse hardware cost. These helpers aggregate over seeds, draw the
trade-off, and apply a **spectral-width budget** to turn the Pareto front into
a single recommended recipe.

Selection rule (`recommend`): among configurations whose mean cost statistic is
within `max_cost`, take the highest mean F_held. Ties break toward lower cost.

The cost statistic is `occ99_frac` and the budget is a **percentage**: each drive
may occupy at most `BUDGET_FRAC` of its **own** band mask. Both choices are
deliberate and load-bearing:

  * The **statistic**: `bandwidth_MHz`, the original default, is an RMS about
    f = 0 and so charges the drive's carrier offset as width, but that offset is
    a gauge choice (retuning the LO changes it and nothing physical). The 99%
    occupied width does not. See `analysis.penalty_sweep.pulse_metrics`.
  * The **per-drive normalisation**: `max(occ99_cav_MHz, occ99_tra_MHz) <= tau`
    is a conjunction of two constraints forced to share one threshold, but the
    drives do not share an allowance -- the cavity mask is 54 MHz wide and the
    transmon's 66 -- and they are separate hardware chains, so nothing
    physically experiences the max of the two. Normalising each by its own
    allowance restores the correct conjunction,
    `max(x/a, y/b) <= t  <=>  (x <= t*a) AND (y <= t*b)`.
  * The **dimensionlessness**: a bare threshold in MHz is not portable across
    statistics. Carrying the old `max_bandwidth=16.0` over to a centroid-referred
    width makes the constraint vacuous (45/45 feasible) and hands the
    recommendation to a lambda_deriv = 0 negative control -- the exact
    configuration the study exists to reject. A percentage cannot be mis-ported
    that way.

Color: the categorical hues are the Okabe-Ito colorblind-safe set, assigned to
penalty names in a FIXED order (`SWEPT_COLORS`) so a penalty keeps its hue
across every figure. The heatmap uses a single-hue sequential ramp (magnitude,
not polarity). Note: the dataviz skill's palette validator is a Node script and
Node is unavailable in this environment, so the palette was not machine-checked;
Okabe-Ito is used precisely because it is a published CVD-safe set.

Usage
-----
    import pandas as pd, visualization.penalty_viz as pv
    df = pd.read_csv("tables/penalty_sweep_X_ofat.csv")
    pv.plot_ofat(df, save_path="figures/penalty_ofat")
    fig, rec = pv.plot_pareto(df, save_path="figures/penalty_pareto",
                              max_cost=pv.DEFAULT_MAX_COST)
    pv.summary_table(df, out_csv="tables/penalty_sweep_summary.csv",
                     max_cost=pv.DEFAULT_MAX_COST)
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PENALTY_NAMES = ("deriv", "boundary", "amp", "disc")

# --- The cost statistic and its budget --------------------------------------
#
# The cost statistic is a FRACTION of each drive's own band mask, so it is
# dimensionless and the budget is a percentage. Per-drive allowances live in
# analysis.penalty_sweep.BAND_LIMITS_MHz; this module reads only the already-
# normalised column from the CSV, which keeps it decoupled from the sweep code.
BUDGET_FRAC = 0.70
DEFAULT_COST = "occ99_frac"
DEFAULT_MAX_COST = BUDGET_FRAC          # dimensionless, NOT MHz
LEGACY_MASK_WIDTH_MHz = 66.0            # back-compat fallback only


def _fmt_cost(cost, value):
    """Format a cost value with the right unit for its column."""
    if cost.endswith("_frac"):
        return f"{value:.0%} of own mask"
    return f"{value:.2f} MHz"


def _resolve_cost(max_cost, max_bandwidth):
    """
    Resolve the budget from the current `max_cost` and the deprecated
    `max_bandwidth` alias.

    The alias is kept only so a half-edited notebook still runs. It is NOT a
    rename: the two arguments are thresholds on different statistics and on
    different scales -- `max_cost` is now a dimensionless fraction in [0, 1],
    where `max_bandwidth` was MHz -- so a value ported across without rescaling
    will either be vacuous or reject everything.
    """
    if max_bandwidth is not None:
        if max_cost is not None:
            raise TypeError("pass max_cost or max_bandwidth, not both")
        warnings.warn(
            "max_bandwidth= is deprecated; use max_cost= (and note that the "
            f"default cost statistic is now {DEFAULT_COST!r}, a dimensionless "
            "fraction of each drive's own band mask -- rescale the threshold, "
            "do not port it)",
            DeprecationWarning, stacklevel=3,
        )
        return float(max_bandwidth)
    return DEFAULT_MAX_COST if max_cost is None else float(max_cost)


def _fallback_cost(agg, cost, max_cost):
    """
    Degrade gracefully on a CSV written before the per-drive fractions existed.

    Returns a possibly-substituted `(cost, max_cost)`. An old table has
    `occ99_MHz` but no `occ99_frac`, so the budget is re-expressed against the
    transmon mask -- the closest thing the old single-threshold rule meant.
    `max_cost` is expected already resolved by `_resolve_cost`, i.e. a fraction.
    """
    if cost != DEFAULT_COST or cost in agg.columns or "occ99_MHz" not in agg.columns:
        return cost, max_cost
    substitute = max_cost * LEGACY_MASK_WIDTH_MHz
    warnings.warn(
        f"{DEFAULT_COST!r} not in this table (it predates per-drive "
        f"allowances); falling back to cost='occ99_MHz' with max_cost="
        f"{substitute:.2f} MHz = {max_cost:.0%} x the "
        f"{LEGACY_MASK_WIDTH_MHz:.0f} MHz transmon mask. Re-run "
        "analysis/penalty_sweep.py to backfill the fractions without retraining.",
        RuntimeWarning, stacklevel=3,
    )
    return "occ99_MHz", substitute

# Sweep axes RECOGNIZED when reading a table. `amp_max` is a threshold, not a
# weight, so its dataframe column is named `amp_max` rather than
# `lambda_amp_max` -- use `axis_column()`.
#
# Deliberately a SUPERSET of analysis.penalty_sweep.AXIS_NAMES, which lists only
# the axes still runnable. `disc` was retired as an axis (provably inert -- see
# the note above penalty_sweep.PENALTY_NAMES) but its five rows are retained in
# the historical CSVs as this pipeline's basin-noise estimate, and `amp` appears
# in the older smoke tables. Both must stay here:
#
#   * plot_ofat draws only the axes actually present in the CSV (see the
#     `axis_list` filter below), so a trimmed table quietly omits their panels;
#   * plot_pareto uses `~swept.isin(AXIS_NAMES)` to identify GRID rows, so
#     dropping a retired name here would recolor those historical rows as grid
#     points rather than as the OFAT rows they are.
#
# Order fixes panel order in plot_ofat. Do not "sync" this with the sweep's list.
AXIS_NAMES = ("deriv", "boundary", "disc", "amp_max")


def axis_column(name):
    """Dataframe column holding the value of sweep axis `name`."""
    return "amp_max" if name == "amp_max" else f"lambda_{name}"


# Okabe-Ito, fixed assignment -- an axis keeps its hue in every figure.
SWEPT_COLORS = {
    "deriv": "#0072B2",     # blue
    "boundary": "#D55E00",  # vermillion
    "amp_max": "#009E73",   # green
    "disc": "#CC79A7",      # purple
    "amp": "#999999",       # legacy inert axis, if present in old CSVs
}
BASELINE_COLOR = "#000000"
SEQUENTIAL_CMAP = "Blues"   # single hue, light -> dark (magnitude)

PENALTY_LABEL = {
    "deriv": r"$\lambda_{\mathrm{deriv}}$",
    "boundary": r"$\lambda_{\mathrm{boundary}}$",
    "amp": r"$\lambda_{\mathrm{amp}}$",
    "disc": r"$\lambda_{\mathrm{disc}}$",
    "amp_max": r"$\epsilon_{\max}$ (rad/$\mu$s)",
}

METRIC_LABEL = {
    "F_ped_heldout_mean": "Held-out $F$",
    "occ99_frac": r"$W_{99}$ occupancy (% of own band mask)",
    "occ99_frac_cav": "Cavity 99% width (frac.)",
    "occ99_frac_tra": "Transmon 99% width (frac.)",
    "occ99_MHz": "99% occupied width (MHz)",
    "sigma_f_MHz": r"Spectral width $\sigma_f$ (MHz)",
    "centroid_tra_MHz": r"Transmon centroid $\langle f\rangle$ (MHz)",
    "n_bar_drive": r"$\bar n_{\mathrm{drive}}$",
    "bandwidth_MHz": "RMS bandwidth (MHz)",
    "robustness_spread": "Robustness spread",
    "peak_amp": "Peak amplitude (rad/$\\mu$s)",
    "roughness": "Roughness (rad/$\\mu$s per step)",
    "F_coh_train": "Training $F_{\\mathrm{coh}}$",
}

# Anything not listed here is dropped by `aggregate_seeds`.
_AGG_METRICS = [
    "F_ped_heldout_mean", "F_ped_heldout_min", "robustness_spread",
    "leakage_L1_mean", "peak_amp", "roughness",
    "occ99_frac", "occ99_frac_cav", "occ99_frac_tra",
    "occ99_MHz", "occ99_cav_MHz", "occ99_tra_MHz",
    "occ99_cav_lo_MHz", "occ99_cav_hi_MHz",
    "occ99_tra_lo_MHz", "occ99_tra_hi_MHz",
    "sigma_f_MHz", "sigma_f_cav_MHz", "sigma_f_tra_MHz",
    "centroid_cav_MHz", "centroid_tra_MHz", "n_bar_drive",
    "bandwidth_MHz", "bandwidth_cav_MHz", "bandwidth_tra_MHz",
    "F_coh_train", "overfit_gap", "iterations",
]


def _style_axis(ax):
    """Recessive grid and spines; the data should be the darkest thing."""
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#666666")


def _fmt_weight(v):
    """Compact tick label for a penalty weight: '0', '1e-5', '0.5'."""
    if v == 0:
        return "0"
    if 1e-3 <= abs(v) < 1e4:
        return f"{v:g}"
    mant, exp = f"{v:.0e}".split("e")
    return f"{mant}e{int(exp)}"


def _flatten_if_degenerate(ax, y, rel_tol=1e-7):
    """
    Keep a numerically-flat response looking flat.

    Some sweep axes are inert -- `lambda_disc` moves F_coh by ~1e-11 across a
    0 -> 5 span, because at these training truncations the discrepancy term is
    ~1e-8 against an O(1) fidelity term. Left alone, matplotlib autoscales that
    noise to fill the panel and prints an offset like `1e-11+6.3584e-1`, which
    reads as a dramatic curve. That is plotting noise as signal.

    When the response is below tolerance, this pins a readable symmetric range
    around the mean and says so on the panel.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return False

    rng = float(y.max() - y.min())
    scale = max(abs(float(y.mean())), 1e-30)
    if rng > rel_tol * scale:
        return False

    mid = float(y.mean())
    pad = max(abs(mid) * 1e-3, 1e-9)
    ax.set_ylim(mid - pad, mid + pad)
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.text(0.5, 0.92, f"no measurable response\n(range {rng:.1e})",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            color="#777777", style="italic")
    return True


def _save(fig, save_path):
    """Save both PNG (notebook) and PDF (LaTeX) next to each other."""
    if not save_path:
        return
    base, ext = os.path.splitext(save_path)
    if ext.lower() in (".png", ".pdf"):
        save_path = base
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(f"{save_path}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{save_path}.pdf", bbox_inches="tight")
    print(f"Saved {save_path}.png / .pdf")


# ============================================================
# Aggregation
# ============================================================

def aggregate_seeds(df):
    """
    Collapse the per-seed rows to one row per penalty configuration, carrying
    mean and std of every metric.

    The optimizer is non-convex, so the std across seeds is a real part of the
    result -- a config whose fidelity swings by more than the gap to its
    neighbours has not actually been shown to beat them.
    """
    # A config can legitimately appear in more than one CSV -- the OFAT and grid
    # sweeps overlap, and the second run is a cache hit, so the rows are
    # byte-identical. `config_hash` uniquely identifies a trained pulse, so
    # (config_hash, seed) uniquely identifies a RUN. Without this dedup those
    # repeats are counted as extra seeds: n_seeds reads 2 and the *_std columns
    # read 0, claiming a reproducibility measurement that was never made.
    if {"config_hash", "seed"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["config_hash", "seed"])

    keys = ["gate", "label", "swept", "multiplier", "is_baseline"] + \
           [f"lambda_{p}" for p in PENALTY_NAMES] + ["amp_max"]
    keys = [k for k in keys if k in df.columns]

    present = [m for m in _AGG_METRICS if m in df.columns]
    g = df.groupby(keys, dropna=False)

    out = g[present].mean().add_suffix("")
    std = g[present].std(ddof=0).add_suffix("_std")
    n = g.size().rename("n_seeds")

    agg = pd.concat([out, std, n], axis=1).reset_index()

    # `binding_drive` is a string, so it cannot be averaged. Recompute it from
    # the averaged fractions instead of trying to aggregate the raw column.
    if {"occ99_frac_cav", "occ99_frac_tra"}.issubset(agg.columns):
        agg["binding_drive"] = np.where(
            agg["occ99_frac_cav"] > agg["occ99_frac_tra"], "cav", "tra")
    return agg


# ============================================================
# Pareto machinery
# ============================================================

def pareto_front(agg, x=DEFAULT_COST, y="F_ped_heldout_mean"):
    """
    Indices of non-dominated points: minimize x (cost), maximize y (fidelity).

    A point is dominated if another is at least as good on both axes and
    strictly better on one.
    """
    xs = agg[x].to_numpy()
    ys = agg[y].to_numpy()
    keep = []
    for i in range(len(agg)):
        dominated = np.any(
            (xs <= xs[i]) & (ys >= ys[i]) &
            ((xs < xs[i]) | (ys > ys[i]))
        )
        if not dominated:
            keep.append(i)
    order = np.argsort(xs[keep])
    return [keep[j] for j in order]


def recommend(agg_or_df, max_cost=None, metric="F_ped_heldout_mean",
              cost=DEFAULT_COST, max_bandwidth=None):
    """
    Best-fidelity configuration subject to a spectral-width budget.

    `agg_or_df` may be either a raw per-seed frame or an already-aggregated one;
    it is aggregated if it still has a `seed` column.

    `cost` names the column the budget applies to (default `occ99_frac`) and
    `max_cost` the threshold (default `BUDGET_FRAC`, a dimensionless fraction of
    each drive's own band mask -- NOT MHz). `max_bandwidth` is a deprecated alias
    for `max_cost` -- see `_resolve_cost`.

    Returns a pandas Series (the winning row) with `F`, `cost` and `cost_metric`
    convenience fields added, plus `bandwidth` as a legacy alias for `cost`.
    Raises if nothing meets the budget.
    """
    max_cost = _resolve_cost(max_cost, max_bandwidth)
    agg = aggregate_seeds(agg_or_df) if "seed" in agg_or_df.columns else agg_or_df.copy()
    cost, max_cost = _fallback_cost(agg, cost, max_cost)

    if cost not in agg.columns:
        raise KeyError(
            f"cost column {cost!r} not in this table -- it predates the metric. "
            f"Re-run analysis/penalty_sweep.py (cached rows backfill without "
            f"retraining), or pass cost='bandwidth_MHz' with a rescaled budget."
        )

    feasible = agg[agg[cost] <= max_cost]
    if feasible.empty:
        raise ValueError(
            f"No configuration has {cost} <= {_fmt_cost(cost, max_cost)} "
            f"(minimum observed: {_fmt_cost(cost, agg[cost].min())})"
        )

    # Highest fidelity; ties -> lower cost.
    feasible = feasible.sort_values([metric, cost], ascending=[False, True])
    rec = feasible.iloc[0].copy()
    rec["F"] = rec[metric]
    rec["cost"] = rec[cost]
    rec["cost_metric"] = cost
    rec["bandwidth"] = rec[cost]   # legacy alias
    return rec


# ============================================================
# Figures
# ============================================================

def plot_ofat(df, save_path=None, metrics=("F_ped_heldout_mean", DEFAULT_COST)):
    """
    One-factor-at-a-time panels: one column per penalty, one row per metric.

    Deliberately NOT a dual-axis plot -- fidelity and spectral width live on
    separate rows sharing an x-axis, so neither scale is implied to be
    commensurate with the other.

    Zero-weight points (the negative controls) are drawn on a symlog x-axis so
    they appear in place rather than being dropped by a log scale.
    """
    agg = aggregate_seeds(df) if "seed" in df.columns else df.copy()
    ofat = agg[agg["swept"].isin(AXIS_NAMES)]
    if ofat.empty:
        raise ValueError("No OFAT rows found (column 'swept' has no axis names)")

    # Degrade gracefully on CSVs written before a metric existed.
    metrics = [m for m in metrics if m in agg.columns]
    if not metrics:
        raise ValueError("none of the requested metrics are in this table")

    # Only draw axes actually present in this CSV.
    axis_list = [a for a in AXIS_NAMES if (ofat["swept"] == a).any()]

    n_row, n_col = len(metrics), len(axis_list)
    fig, axes = plt.subplots(n_row, n_col, figsize=(3.4 * n_col, 3.0 * n_row),
                             squeeze=False)

    baseline_row = agg[agg["is_baseline"]]

    for j, pname in enumerate(axis_list):
        col = axis_column(pname)
        # The incumbent config is the shared rung of EVERY ladder, but config
        # enumeration deduplicates it into whichever ladder was built first.
        # Re-attach it here, or all but one panel silently lose their
        # incumbent point (and its dotted marker line).
        sub = ofat[ofat["swept"] == pname]
        if not baseline_row.empty and not sub["is_baseline"].any():
            sub = pd.concat([sub, baseline_row], ignore_index=True)
        sub = sub.drop_duplicates(subset=[col]).sort_values(col)
        if sub.empty:
            for i in range(n_row):
                axes[i][j].set_visible(False)
            continue
        x = sub[col].to_numpy()
        nonzero = x[x > 0]
        # Put the linear/log crossover exactly at the smallest swept value, so
        # no log decade tick lands inside the linear region and collides with
        # the 0 tick.
        linthresh = float(nonzero.min()) if len(nonzero) else 1e-12

        for i, metric in enumerate(metrics):
            ax = axes[i][j]
            y = sub[metric].to_numpy()
            err = sub.get(f"{metric}_std")
            err = err.to_numpy() if err is not None else None
            if err is not None and np.all(np.isnan(err)):
                err = None

            ax.errorbar(x, y, yerr=err, marker="o", markersize=6, linewidth=2,
                        color=SWEPT_COLORS.get(pname, "#777777"), capsize=3,
                        markeredgecolor="white", markeredgewidth=0.8)

            # Mark the incumbent value.
            base = sub[sub["is_baseline"]]
            if not base.empty:
                ax.axvline(float(base[col].iloc[0]),
                           color=BASELINE_COLOR, linestyle=":", linewidth=1.2,
                           alpha=0.7)

            # amp_max is a linear physical threshold, not a log-spanning weight.
            if pname == "amp_max":
                ax.set_xscale("linear")
            else:
                ax.set_xscale("symlog", linthresh=linthresh)
                # Weights are non-negative; without this, symlog renders the
                # negative decades too and their tick labels collide with 0.
                ax.set_xlim(left=0.0)
                # Tick exactly at the swept values. symlog's default locator
                # also emits decade ticks inside the linear region, which
                # overlap the 0 label.
                ax.set_xticks(x)
                ax.set_xticklabels([_fmt_weight(v) for v in x], rotation=45,
                                   ha="right", fontsize=8)
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

            _flatten_if_degenerate(ax, y)
            _style_axis(ax)
            if metric.endswith("_frac"):
                ax.yaxis.set_major_formatter(
                    matplotlib.ticker.PercentFormatter(xmax=1))
            if i == n_row - 1:
                ax.set_xlabel(PENALTY_LABEL[pname])
            if j == 0:
                ax.set_ylabel(METRIC_LABEL.get(metric, metric))
            if i == 0:
                ax.set_title(PENALTY_LABEL[pname], fontsize=11)

    fig.suptitle("Penalty one-factor-at-a-time sweep "
                 "(dotted line = incumbent; error bars = std over seeds)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, save_path)
    return fig


def plot_pareto(df, save_path=None, max_cost=None,
                x=DEFAULT_COST, y="F_ped_heldout_mean", max_bandwidth=None):
    """
    Fidelity-vs-cost scatter with the non-dominated front and the budgeted
    recommendation starred.

    `x` is the cost statistic and doubles as the column the budget applies to.
    `max_bandwidth` is a deprecated alias for `max_cost`.

    Returns (fig, recommended_row).
    """
    max_cost = _resolve_cost(max_cost, max_bandwidth)
    agg = aggregate_seeds(df) if "seed" in df.columns else df.copy()
    x, max_cost = _fallback_cost(agg, x, max_cost)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for pname in AXIS_NAMES:
        sub = agg[agg["swept"] == pname]
        if sub.empty:
            continue
        ax.scatter(sub[x], sub[y], s=55, color=SWEPT_COLORS.get(pname, "#777777"),
                   edgecolor="white", linewidth=0.9, label=f"swept {pname}",
                   zorder=3)

    # Grid-mode rows (swept == "derivxboundary") get a neutral mark.
    other = agg[~agg["swept"].isin(AXIS_NAMES)]
    if not other.empty:
        ax.scatter(other[x], other[y], s=45, color="#777777", alpha=0.75,
                   edgecolor="white", linewidth=0.8, label="grid", zorder=2)

    front = pareto_front(agg, x=x, y=y)
    if front:
        ax.plot(agg.iloc[front][x], agg.iloc[front][y], color="#333333",
                linewidth=1.4, alpha=0.75, zorder=2, label="Pareto front")

    base = agg[agg["is_baseline"]]
    if not base.empty:
        ax.scatter(base[x], base[y], s=150, marker="D", facecolor="none",
                   edgecolor=BASELINE_COLOR, linewidth=1.6, zorder=4,
                   label="incumbent")

    rec = None
    try:
        rec = recommend(agg, max_cost=max_cost, metric=y, cost=x)
        ax.scatter([rec[x]], [rec[y]], s=340, marker="*", color="#E69F00",
                   edgecolor="#333333", linewidth=1.0, zorder=5,
                   label="recommended")
    except ValueError as exc:
        print(f"[plot_pareto] {exc}")

    # Budget region drawn LAST, so the data has already set the y-limits. The
    # label uses a blended transform (data x, axes-fraction y) -- placing it at
    # a data-space y would drag the y-axis out to that value.
    x_hi = max(float(agg[x].max()) * 1.05, max_cost * 1.05)
    ax.axvspan(max_cost, x_hi, color="#999999", alpha=0.12, zorder=0)
    ax.axvline(max_cost, color="#555555", linestyle="--", linewidth=1.3,
               zorder=1)
    ax.text(max_cost, 0.98, f"  budget {_fmt_cost(x, max_cost)}",
            transform=ax.get_xaxis_transform(), rotation=90, va="top",
            ha="left", fontsize=9, color="#555555")
    ax.set_xlim(right=x_hi)

    if x.endswith("_frac"):
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_xlabel(METRIC_LABEL.get(x, x))
    ax.set_ylabel(METRIC_LABEL.get(y, y))
    ax.set_title("Penalty Pareto front: reported fidelity vs "
                 f"{METRIC_LABEL.get(x, x).split(' (')[0].lower()}")
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig, rec


def _dedup_trials(df):
    """Drop exact (config_hash, seed) duplicates -- the same rule `aggregate_seeds`
    applies before grouping, but WITHOUT the grouping/averaging itself. A config can
    legitimately appear in more than one source CSV (e.g. the OFAT and grid sweeps
    overlap on shared recipes, or a 3x3-corner table overlaps its parent 5x5 grid);
    without this, those repeats would be plotted twice."""
    if {"config_hash", "seed"}.issubset(df.columns):
        return df.drop_duplicates(subset=["config_hash", "seed"])
    return df.copy()


def find_best_trial(df, max_cost=None, metric="F_ped_heldout_mean",
                    cost=DEFAULT_COST, max_bandwidth=None):
    """
    Single best INDIVIDUAL trial (one specific (config, seed) run), not a
    seed-averaged config -- the raw-data analogue of `recommend`.

    Same feasible-filter-then-sort rule as `recommend` (cost <= max_cost, then
    highest `metric`, ties toward lower cost) but applied to `df` UNAGGREGATED,
    so a config that only wins on the strength of one lucky seed can surface
    here even though it would not win `recommend`'s seed-mean comparison. That
    is the point of this function and also its caveat: picking the best of many
    individual trials is optimistic by construction (selection bias), not
    evidence that this particular seed+config combination is truly better than
    its neighbours -- see `recommend`/§7.4's seed-mean statistics for that
    question.
    """
    max_cost = _resolve_cost(max_cost, max_bandwidth)
    trials = _dedup_trials(df)
    cost, max_cost = _fallback_cost(trials, cost, max_cost)

    if cost not in trials.columns:
        raise KeyError(f"cost column {cost!r} not in this table")

    feasible = trials[trials[cost] <= max_cost]
    if feasible.empty:
        raise ValueError(
            f"No trial has {cost} <= {_fmt_cost(cost, max_cost)} "
            f"(minimum observed: {_fmt_cost(cost, trials[cost].min())})"
        )

    feasible = feasible.sort_values([metric, cost], ascending=[False, True])
    best = feasible.iloc[0].copy()
    best["F"] = best[metric]
    best["cost"] = best[cost]
    best["cost_metric"] = cost
    return best


def plot_all_trials(df, save_path=None, max_cost=None,
                    x=DEFAULT_COST, y="F_ped_heldout_mean", max_bandwidth=None):
    """
    Every individual trial -- one marker per (config, seed) run, NO seed
    averaging -- on the same fidelity-vs-cost axes as `plot_pareto`.

    Visually distinguishes the training protocol (circle = single_phase,
    triangle = two_phase) on top of the usual swept-axis color coding, since a
    single nominal recipe can now have trials from both. The single best
    feasible trial (per `find_best_trial`) is starred, mirroring `plot_pareto`'s
    "recommended" marker -- but this is the best individual RUN, not the best
    seed-mean config, and is subject to the same selection-bias caveat spelled
    out in `find_best_trial`'s docstring.

    Returns (fig, best_trial).
    """
    max_cost = _resolve_cost(max_cost, max_bandwidth)
    trials = _dedup_trials(df)
    x, max_cost = _fallback_cost(trials, x, max_cost)

    protocol = (trials["protocol"] if "protocol" in trials.columns
                else pd.Series("single_phase", index=trials.index)).fillna("single_phase")
    marker_for = {"single_phase": "o", "two_phase": "^"}

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for pname in AXIS_NAMES:
        sub_axis = trials[trials["swept"] == pname]
        if sub_axis.empty:
            continue
        for proto, marker in marker_for.items():
            sub = sub_axis[protocol.loc[sub_axis.index] == proto]
            if sub.empty:
                continue
            ax.scatter(sub[x], sub[y], s=40, marker=marker,
                       color=SWEPT_COLORS.get(pname, "#777777"),
                       edgecolor="white", linewidth=0.7, alpha=0.85,
                       label=f"swept {pname} ({proto})", zorder=3)

    other_axis = trials[~trials["swept"].isin(AXIS_NAMES)]
    for proto, marker in marker_for.items():
        sub = other_axis[protocol.loc[other_axis.index] == proto]
        if sub.empty:
            continue
        ax.scatter(sub[x], sub[y], s=32, marker=marker, color="#777777",
                   alpha=0.6, edgecolor="white", linewidth=0.6,
                   label=f"grid ({proto})", zorder=2)

    best = None
    try:
        best = find_best_trial(trials, max_cost=max_cost, metric=y, cost=x)
        ax.scatter([best[x]], [best[y]], s=340, marker="*", color="#E69F00",
                   edgecolor="#333333", linewidth=1.0, zorder=5,
                   label="best trial")
    except ValueError as exc:
        print(f"[plot_all_trials] {exc}")

    x_hi = max(float(trials[x].max()) * 1.05, max_cost * 1.05)
    ax.axvspan(max_cost, x_hi, color="#999999", alpha=0.12, zorder=0)
    ax.axvline(max_cost, color="#555555", linestyle="--", linewidth=1.3, zorder=1)
    ax.text(max_cost, 0.98, f"  budget {_fmt_cost(x, max_cost)}",
            transform=ax.get_xaxis_transform(), rotation=90, va="top",
            ha="left", fontsize=9, color="#555555")
    ax.set_xlim(right=x_hi)

    if x.endswith("_frac"):
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_xlabel(METRIC_LABEL.get(x, x))
    ax.set_ylabel(METRIC_LABEL.get(y, y))
    ax.set_title(f"Every trial ({len(trials)} runs, unaggregated): "
                 f"reported fidelity vs {METRIC_LABEL.get(x, x).split(' (')[0].lower()}")
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, best


def plot_grid_heatmap(df, x_name, y_name, save_path=None,
                      metric="F_ped_heldout_mean", annot_metric=DEFAULT_COST):
    """
    2-D penalty grid as a single-hue sequential heatmap (magnitude -> darker).

    Cells are annotated with the secondary metric (the spectral cost by default)
    so the trade-off is visible in one panel without a second color scale.
    """
    agg = aggregate_seeds(df) if "seed" in df.columns else df.copy()
    if annot_metric not in agg.columns:      # older CSV
        annot_metric = "occ99_MHz" if "occ99_MHz" in agg.columns else "bandwidth_MHz"
    xcol, ycol = axis_column(x_name), axis_column(y_name)

    piv = agg.pivot_table(index=ycol, columns=xcol, values=metric, aggfunc="mean")
    ann = agg.pivot_table(index=ycol, columns=xcol, values=annot_metric,
                          aggfunc="mean")

    fig, ax = plt.subplots(figsize=(1.25 * len(piv.columns) + 3.2,
                                    1.0 * len(piv.index) + 2.6))
    im = ax.imshow(piv.to_numpy(), origin="lower", aspect="auto",
                   cmap=SEQUENTIAL_CMAP)

    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{v:.3g}" for v in piv.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{v:.3g}" for v in piv.index])
    ax.set_xlabel(PENALTY_LABEL.get(x_name, x_name))
    ax.set_ylabel(PENALTY_LABEL.get(y_name, y_name))

    vals = piv.to_numpy()
    vmid = np.nanmin(vals) + 0.55 * (np.nanmax(vals) - np.nanmin(vals))
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                continue
            a = ann.to_numpy()[i, j]
            # Ink color flips on the dark end of the ramp so text stays legible.
            color = "white" if v > vmid else "#222222"
            annot = f"{a:.0%}" if annot_metric.endswith("_frac") else f"{a:.1f} MHz"
            # Each line is explicitly labeled -- the cell color is fidelity ONLY;
            # the second line is spectral bandwidth occupancy (W99), not a
            # population/probability of any kind, and is not part of the color scale.
            ax.text(j, i, f"F = {v:.4f}\n$W_{{99}}$ = {annot}", ha="center",
                    va="center", fontsize=8, color=color)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(f"cell color → {METRIC_LABEL.get(metric, metric)}")
    cb.outline.set_visible(False)
    ax.set_title(
        f"Cell color = {METRIC_LABEL.get(metric, metric)} only\n"
        f"Cell text = fidelity value / {METRIC_LABEL.get(annot_metric, annot_metric)}\n"
        f"over {PENALTY_LABEL.get(x_name, x_name)} × {PENALTY_LABEL.get(y_name, y_name)}",
        fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ============================================================
# Tables
# ============================================================

def summary_table(df, out_csv=None, max_cost=None, top_k=12,
                  cost_column=DEFAULT_COST, max_bandwidth=None):
    """
    Ranked summary of the sweep, plus a LaTeX version written next to the CSV.

    Rows are ordered by held-out fidelity among budget-feasible configs first,
    then the infeasible ones. `within_budget` makes the constraint explicit so
    the table stands alone in the report.

    `cost_column` is the statistic the budget applies to (default `occ99_frac`);
    `max_bandwidth` is a deprecated alias for `max_cost`.
    """
    max_cost = _resolve_cost(max_cost, max_bandwidth)
    agg = aggregate_seeds(df) if "seed" in df.columns else df.copy()
    agg = agg.copy()
    cost_column, max_cost = _fallback_cost(agg, cost_column, max_cost)
    if cost_column not in agg.columns:      # much older CSV
        cost_column = "bandwidth_MHz"
    agg["within_budget"] = agg[cost_column] <= max_cost

    agg = agg.sort_values(
        ["within_budget", "F_ped_heldout_mean"], ascending=[False, False]
    )

    cols = ["label", "swept"] + [f"lambda_{p}" for p in PENALTY_NAMES] + [
        "amp_max",
        "F_ped_heldout_mean", "F_ped_heldout_std", "F_ped_heldout_min",
        "robustness_spread", cost_column,
        # Per-drive breakdown of the scored cost, and which drive it binds on.
        "occ99_frac_cav", "occ99_frac_tra", "binding_drive",
        "occ99_cav_MHz", "occ99_tra_MHz",
        # Reported, never scored: the gauge-invariant second moment, the gauge
        # quantity it was separated from, and that centroid read as a cavity
        # photon number.
        "sigma_f_MHz", "centroid_tra_MHz", "n_bar_drive", "bandwidth_MHz",
        "peak_amp", "roughness",
        "F_coh_train", "overfit_gap", "within_budget", "n_seeds",
    ]
    # dict.fromkeys dedups while preserving order -- `cost_column` may itself be
    # one of the columns listed after it.
    cols = [c for c in dict.fromkeys(cols) if c in agg.columns]
    table = agg[cols]

    print(f"budget: {cost_column} <= {_fmt_cost(cost_column, max_cost)}  "
          f"({int(table['within_budget'].sum())}/{len(table)} feasible)")

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        table.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")

        tex_path = os.path.splitext(out_csv)[0] + ".tex"
        _write_latex(table, tex_path, max_cost, top_k, cost_column)

    return table


def _write_latex(table, tex_path, max_cost, top_k, cost_column=DEFAULT_COST):
    """
    Compact booktabs table of the top-k rows, for \\input into the report.

    Also defines \\penaltybudget, so the report's caption states the statistic
    and threshold actually used rather than a hand-copied number that can drift
    out of date.
    """
    sub = table.head(top_k)
    cost_tex = cost_column.replace("_", r"\_")
    # The scored cost is the max of the two per-drive fractions, both of which
    # are printed, so the aggregate column is dropped from the printed table
    # rather than repeating a number the reader can take the max of. `spread`
    # goes too -- it is redundant with $F_{\min}$ here and the CSV keeps it.
    per_drive = cost_column.endswith("_frac") and \
        {"occ99_frac_cav", "occ99_frac_tra"}.issubset(sub.columns)
    cost_head = {
        "occ99_MHz": r"$W_{99}$ (MHz)",
        "bandwidth_MHz": r"BW (MHz)",
        "occ99_frac": r"$W_{99}$",
    }.get(cost_column, cost_tex)
    cols = [
        ("lambda_deriv", r"$\lambda_{\mathrm{d}}$", "{:.2g}"),
        ("lambda_boundary", r"$\lambda_{\mathrm{b}}$", "{:.2g}"),
        ("lambda_disc", r"$\lambda_{\mathrm{disc}}$", "{:.2g}"),
        ("amp_max", r"$\epsilon_{\max}$", "{:.0f}"),
        ("F_ped_heldout_mean", r"$F_{\mathrm{held}}$", "{:.5f}"),
        ("F_ped_heldout_min", r"$F_{\min}$", "{:.5f}"),
        (None if per_drive else "robustness_spread", r"spread", "{:.2e}"),
        (None if per_drive else cost_column, cost_head, "{:.2f}"),
        ("occ99_frac_cav" if per_drive else None, r"$W_{99}^{C}$", "{:.0%}"),
        ("occ99_frac_tra" if per_drive else None, r"$W_{99}^{T}$", "{:.0%}"),
        ("binding_drive" if per_drive else None, r"binding", "{}"),
        ("sigma_f_MHz", r"$\sigma_f$", "{:.2f}"),
        ("centroid_tra_MHz", r"$\langle f\rangle_T$", "{:+.2f}"),
        ("n_bar_drive", r"$\bar n_{\mathrm{drv}}$", "{:.2f}"),
        ("peak_amp", r"peak", "{:.1f}"),
        ("within_budget", r"in budget", "{}"),
    ]
    cols = [c for c in dict.fromkeys(cols) if c[0] is not None and c[0] in sub.columns]

    if cost_column.endswith("_frac"):
        budget_desc = (rf"$W_{{99}} \leq {max_cost:.0%}$".replace("%", r"\%") +
                       r" of each drive's own band mask")
    elif cost_column == "occ99_MHz":
        budget_desc = rf"$W_{{99}} \leq \SI{{{max_cost:.1f}}}{{\mega\hertz}}$"
    else:
        budget_desc = rf"\texttt{{{cost_tex}}} $\leq {max_cost:.1f}$ MHz"
    lines = [
        r"% Auto-generated by visualization/penalty_viz.summary_table -- do not hand-edit.",
        rf"% cost statistic: {cost_column}   budget: "
        rf"{_fmt_cost(cost_column, max_cost)}",
        r"\providecommand{\penaltybudget}{}%",
        r"\renewcommand{\penaltybudget}{" + budget_desc + r"}%",
        # 13 columns of numbers do not fit at the default column separation.
        # Scoped to the surrounding table environment in the report.
        r"\setlength{\tabcolsep}{3.5pt}%",
        r"\begin{tabular}{" + "l" * len(cols) + r"}",
        r"\toprule",
        " & ".join(h for _, h, _ in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in sub.iterrows():
        cells = []
        for key, _, fmt in cols:
            v = row[key]
            if key == "within_budget":
                cells.append(r"\checkmark" if bool(v) else "--")
            elif pd.isna(v):
                cells.append("--")
            else:
                # A bare % from a "{:.0%}" format starts a LaTeX comment and
                # would silently swallow the rest of the row.
                cells.append(fmt.format(v).replace("%", r"\%"))
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {tex_path}")
