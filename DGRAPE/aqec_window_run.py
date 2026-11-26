#!/usr/bin/env python3
"""
Studies of the realistic readout window (DGRAPE/aqec_window.py) for notebooks/AQEC_compare.ipynb §6.3. From the repo
root, one subcommand per stage (outputs to --out-dir, default ~/fwm_runs/aqec_window; copy them to tables/ and
pulses/dgrape/ afterwards):

    python3 DGRAPE/aqec_window_run.py dry-run                     # one timed grid cell at n_c 24
    python3 DGRAPE/aqec_window_run.py calib --n-jobs 9            # stage 0: pi pulse, residual photons, readout alone
        -> aqec_window_pi.csv, aqec_window_residual.csv, aqec_window_readout.csv
    python3 DGRAPE/aqec_window_run.py grid --n-jobs 9             # stage 1 -> aqec_window_grid.csv (parts resumable)
    python3 DGRAPE/aqec_window_run.py optimise --n-jobs 9         # stage 2 -> aqec_window_opt.csv,
        windows/aqec_window_<tag>.json, aqec_window_breakdown.csv, aqec_window_residual_check.csv
    python3 DGRAPE/aqec_window_run.py chains --windows rw_e30_l3 ... --n-jobs 9    # stage 3 -> aqec_window_B1/B2.csv
    python3 DGRAPE/aqec_window_run.py score-open --open aqec_open_T15_rw_e30_l3 ... # stage 4 scoring
        -> aqec_window_open_single.csv, aqec_window_open_B2.csv
    python3 DGRAPE/aqec_window_run.py check --windows rw_e30_l3 ...               # bins 5 vs 10 ns, n_c 24 vs 28
        -> aqec_window_check.csv

Fixed: Shirol's channel set, the EsT X gate (aqec_ideal.EST_PULSE), step 0 "forgive", the ideal map, n_c 24.

THE OBJECTIVE J = F_pro(gate + ideal step 1 + window, one gate) - s * n_res: n_res = the photons left in the readout
when the window ends (the next gate runs at n_r = 1 and would not see them), s = the cost per residual photon of the
next gate, from `calib` (aqec_window_residual.csv, the slope over n_res <= 0.1). The threshold bias is optimised for
every window and efficiency (free: aqec_window.Terms).

GRID (stage 1). Drive shapes: square, kick (2x amplitude for the first 0.1 us), CLEAR (two 0.1 us depletion segments,
integrated through). t_int 0.1-2.0 us in 0.1 steps (= the drive length for square and kick; drive + depletion for
CLEAR). Ring-down: passive until n_res <= 0.1 / 0.03 / 0.01, or CLEAR then n_res <= 0.01. nbar 2, 5, 10, 20.
Latency 0.1, 0.3, 0.5 us. pi pulse sigma --pi-sigma (default: the best of calib's table if present, else 4 ns).
Efficiencies 0.05, 0.15, 0.3, 0.6 re-scored per cell.

OPTIMISE (stage 2). For each (eta, latency) at n_cap 10, and at the headline (0.3, 0.3) also n_cap 5 and 20 and
passive ring-down only, and chi_ar = 0 (a sensitivity check: chi_ar is known only from Shirol's text): Nelder-Mead from the best grid cell (by J, n_peak <= n_cap) over nbar, t_drive, kick, t_kick,
detuning, extra integration time, depletion lengths (CLEAR), the ring-down target and the pi sigma. At the headline
(and eta 0.05) also a free 8-segment drive shape from that optimum. Each optimum gets: a window json, a one-channel-
at-a-time breakdown, and the exact residual-photon check (the next gate at n_r 6 with the real n_res).
"""

import argparse
import glob
import json
import os
import sys
import time

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from joblib import Parallel, delayed                  # noqa: E402
from scipy.optimize import minimize                   # noqa: E402

np.seterr(divide="ignore", over="ignore", invalid="ignore")

from DGRAPE import aqec_ideal as aq                   # noqa: E402
from DGRAPE import aqec_window as aw                  # noqa: E402
from DGRAPE import fwm                                # noqa: E402
from DGRAPE import fwm_repeat as fr                   # noqa: E402
from DGRAPE import fwm_tail as ft                     # noqa: E402
from DGRAPE import readout as ro                      # noqa: E402

N_C = 24
N_C_CHECK = 28
CHANNELS = "shirol"
ETAS = (0.05, 0.15, 0.3, 0.6)
LATENCIES = (0.1, 0.3, 0.5)
NBARS = (2.0, 5.0, 10.0, 20.0)
T_INTS = tuple(np.round(np.arange(0.1, 2.01, 0.1), 3))
RINGDOWNS = {"passive_0.1": 0.1, "passive_0.03": 0.03, "passive_0.01": 0.01}
HEADLINE = (0.3, 0.3)
N_CAP = 10.0
KICK, T_KICK, T_DEP = 2.0, 0.1, 0.1
SIGMAS = (4, 6, 8, 10, 12)
RESIDUAL_NBARS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
DEFAULT_OUT = os.path.expanduser("~/fwm_runs/aqec_window")


def _out(a, name):
    os.makedirs(a.out_dir, exist_ok=True)
    return os.path.join(a.out_dir, name)


def _table(a, name):
    """A stage's earlier output: --out-dir first, then tables/."""
    for p in (_out(a, name), os.path.join(REPO_ROOT, "tables", name)):
        if os.path.exists(p):
            return p
    return None


def residual_slope(a):
    if a.slope is not None:
        return a.slope
    p = _table(a, "aqec_window_residual.csv")
    if p is None:
        print("no aqec_window_residual.csv: s = 0 (run calib first)", flush=True)
        return 0.0
    df = pd.read_csv(p)
    return float(df["slope"].iloc[0])


def default_sigma(a):
    if a.pi_sigma is not None:
        return a.pi_sigma
    p = _table(a, "aqec_window_pi.csv")
    if p is None:
        return 4.0
    df = pd.read_csv(p)
    return float(df.loc[df["reset_error"].idxmin(), "sigma_ns"])


# ---------------------------------------------------------------------------
# One worker's fixed inputs
# ---------------------------------------------------------------------------

_CTX = {}


def context(n_c=N_C, channels=CHANNELS):
    """Model, sector, the state after the ideal map (before the pre-twist), targets, reference scores."""
    key = (n_c, channels)
    if key not in _CTX:
        m = aq.model(n_c, 1, channels)
        K = aq.kraus(n_c, 1)
        Rg = fr.gate_step(m, np.load(os.path.join(REPO_ROOT, aq.EST_PULSE)), fr.inputs(n_c, 1))
        R1 = aq.rotate(aq.idle(m, aq.forgive(Rg, n_c, 1), aq.TAU1), aq.phase(m, aq.TAU1))
        R2 = aq.apply_map(R1, K)
        T, w = ft.tail_targets(n_c, 1, "X", "traced")
        score = lambda R: fr.score_state(R, n_c, 1, "X")["F_pro"]  # noqa: E731
        _CTX[key] = dict(m=m, sec=aw.Sector(m), K=K, Rg=Rg, R2=R2, T=T, w=w, n_c=n_c, channels=channels,
                         F_gate=score(Rg), F_ideal=score(aq.aqec_step(m, Rg, K, "forgive")),
                         F_no_window=score(aq.aqec_step(m, Rg, K, "forgive", aq.TAU1, 0.0)))
    return _CTX[key]


def evaluate(spec, etas, slope, n_c=N_C, channels=CHANNELS, free_bias=True):
    """Rows {window settings, eta, bias, F_pro, J, metrics} for one window, re-scored for every eta."""
    c = context(n_c, channels)
    W = aw.Window(spec, c["m"], c["sec"])
    v = W.pretwist()
    tm = W.terms(aq.rotate(c["R2"], v), c["T"], c["w"])
    base = {k: val for k, val in W.metrics().items() if not k.startswith("P_")}
    base.update(shape=spec.drive.shape, nbar=spec.drive.nbar, t_drive=spec.drive.t_drive, kick=spec.drive.kick,
                t_kick=spec.drive.t_kick, t_dep1=spec.drive.t_dep1, t_dep2=spec.drive.t_dep2,
                detuning_mhz=spec.drive.detuning_mhz, latency=spec.latency, pi_sigma=spec.pi.sigma_ns,
                nres_target=spec.nres_target, n_c=n_c, channels=channels, remainder_trace=tm.remainder_trace,
                F_gate=c["F_gate"], F_ideal=c["F_ideal"], F_no_window=c["F_no_window"], slope=slope)
    rows = []
    for eta in etas:
        bias, F = tm.best_bias(eta) if free_bias else (spec.bias, tm.fidelity(eta, spec.bias))
        rows.append({**base, "eta": eta, "bias": bias, "F_pro": F, "J": F - slope * base["n_res"],
                     "snr_eta": float(ro.snr(W.record.s1_level[0], W.record.s1_level[1], eta)) if W.n_int else 0.0,
                     **W.confusion(eta, bias)})
    return rows


# ---------------------------------------------------------------------------
# Window specs
# ---------------------------------------------------------------------------

def grid_spec(shape, ringdown, nbar, t_int, latency, sigma):
    if shape == "clear":
        t_drive = round(t_int - 2 * T_DEP, 3)
        if t_drive < 0.05:
            return None
        drive = ro.DriveSpec("clear", nbar, t_drive, t_dep1=T_DEP, t_dep2=T_DEP)
        target = 0.01
    else:
        drive = ro.DriveSpec(shape, nbar, t_int, kick=KICK if shape == "kick" else 1.0,
                             t_kick=T_KICK if shape == "kick" else 0.0)
        target = RINGDOWNS[ringdown]
    return aw.WindowSpec(drive=drive, latency=latency, pi=aw.PiSpec("gauss", sigma), nres_target=target)


def grid_cells(sigma):
    cells = []
    for latency in LATENCIES:
        for nbar in NBARS:
            for t_int in T_INTS:
                for shape, rds in (("square", RINGDOWNS), ("kick", RINGDOWNS), ("clear", ("clear",))):
                    for rd in rds:
                        spec = grid_spec(shape, rd, nbar, t_int, latency, sigma)
                        if spec is not None:
                            cells.append((f"{shape}|{rd}", spec))
    return cells


PARAMS = {  # name: (getter, setter, lower, upper, initial simplex step)
    "nbar": (lambda s: s.drive.nbar, lambda s, v: setattr(s.drive, "nbar", v), 0.2, 40.0, 2.0),
    "t_drive": (lambda s: s.drive.t_drive, lambda s, v: setattr(s.drive, "t_drive", v), 0.03, 3.0, 0.1),
    "kick": (lambda s: s.drive.kick, lambda s, v: setattr(s.drive, "kick", v), 0.0, 5.0, 0.5),
    "t_kick": (lambda s: s.drive.t_kick, lambda s, v: setattr(s.drive, "t_kick", v), 0.0, 0.5, 0.05),
    "t_dep1": (lambda s: s.drive.t_dep1, lambda s, v: setattr(s.drive, "t_dep1", v), 0.01, 0.5, 0.04),
    "t_dep2": (lambda s: s.drive.t_dep2, lambda s, v: setattr(s.drive, "t_dep2", v), 0.01, 0.5, 0.04),
    "detuning_mhz": (lambda s: s.drive.detuning_mhz, lambda s, v: setattr(s.drive, "detuning_mhz", v), -2, 2, 0.2),
    "extra_int": (lambda s: (s.t_int or 0.0) and s.t_int - _drive_end(s),
                  lambda s, v: setattr(s, "t_int", None if v <= 0.5e-3 else _drive_end(s) + v), 0.0, 1.0, 0.05),
    "log_nres": (lambda s: np.log10(s.nres_target), lambda s, v: setattr(s, "nres_target", 10 ** v), -3, 0, 0.3),
    "pi_sigma": (lambda s: s.pi.sigma_ns, lambda s, v: setattr(s.pi, "sigma_ns", float(round(v))), 2, 16, 2.0),
}


def _drive_end(s):
    return ro.build_drive(s.drive).size * ro.DT


def param_names(shape):
    """Free parameters per shape; extra_int last, because it is measured from the drive end."""
    names = ["nbar", "t_drive", "kick", "t_kick", "detuning_mhz", "pi_sigma"]
    names += ["t_dep1", "t_dep2"] if shape in ("clear", "free") else ["log_nres"]
    return names + ["extra_int"]


def apply_params(base, names, x, n_seg=0):
    s = base.replace()
    for nm, v in zip(names, x[:len(names)]):
        lo, hi = PARAMS[nm][2], PARAMS[nm][3]
        PARAMS[nm][1](s, float(np.clip(v, lo, hi)))
    s.drive.t_kick = min(s.drive.t_kick, s.drive.t_drive)
    if n_seg:
        z = np.asarray(x[len(names):], float)
        s.drive.segments = tuple(complex(z[2 * i], z[2 * i + 1]) for i in range(n_seg))
    return s


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def _pi_row(sigma, n_c, channels):
    m = aq.model(n_c, 1, channels)
    params, infid = aw.calibrate_pi(m, sigma)
    u = aw.pi_waveform(params, sigma)
    out = {"sigma_ns": sigma, "length_ns": u.shape[0], "A": params[0], "drag": params[1],
           "delta_mhz": params[2] / (2 * np.pi), "closed_infidelity": infid}
    errs = []
    for n in (1, 3, 5):
        psi = np.zeros(m.d)
        psi[n * 3 + 1] = 1.0
        R = lnp_propagate(m, u, np.outer(psi, psi)[None])
        Pg = float(np.real(sum(R[0, k * 3, k * 3] for k in range(n_c))))
        out[f"P_g_from_{n}e"] = Pg
        errs.append(1 - Pg)
    out["reset_error"] = float(np.mean(errs))
    return out


def lnp_propagate(m, u, R):
    from DGRAPE import lindblad_np as lnp
    return lnp.propagate(m, u, R, ro.DT)


def _residual_row(nbar, n_c, channels):
    gate = np.load(os.path.join(REPO_ROOT, aq.EST_PULSE))
    r = aw.gate_with_residual_photons(gate, nbar, n_c, 6, channels)
    return {"n_res": nbar, "n_c": n_c, "n_r": 6, "channels": channels, **r}


def _readout_rows(nbar, n_c, channels):
    m = aq.model(10, 1, channels)
    sec = aw.Sector(m)
    rows = []
    for t in np.round(np.arange(0.05, 2.001, 0.05), 3):
        spec = aw.WindowSpec(drive=ro.DriveSpec("square", nbar, t), latency=0.0, pi=aw.PiSpec("ideal"), t_w=t)
        W = aw.Window(spec, m, sec)
        for eta in ETAS:
            rows.append({"nbar": nbar, "t_int": t, "eta": eta, "n_peak": W.metrics()["n_peak"],
                         "snr": float(ro.snr(W.record.s1_level[0], W.record.s1_level[1], eta)),
                         **W.confusion(eta, 0.0)})
    return rows


def cmd_calib(a):
    t0 = time.time()
    rows = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(_pi_row)(s, a.n_c, a.channels) for s in SIGMAS)
    pd.DataFrame(rows).to_csv(_out(a, "aqec_window_pi.csv"), index=False)
    print(pd.DataFrame(rows)[["sigma_ns", "closed_infidelity", "reset_error"]], flush=True)
    rr = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(_readout_rows)(nb, a.n_c, a.channels) for nb in NBARS)
    pd.DataFrame([r for x in rr for r in x]).to_csv(_out(a, "aqec_window_readout.csv"), index=False)
    res = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(_residual_row)(nb, a.n_c, a.channels)
                                                    for nb in RESIDUAL_NBARS)
    df = pd.DataFrame(res).sort_values("n_res")
    F0 = float(df.loc[df.n_res == 0, "F_pro"].iloc[0])
    small = df[(df.n_res > 0) & (df.n_res <= 0.1)]
    slope = float(-np.sum(small.n_res * (small.F_pro - F0)) / np.sum(small.n_res ** 2))
    df["slope"] = slope
    df.to_csv(_out(a, "aqec_window_residual.csv"), index=False)
    print(df[["n_res", "F_pro", "P_e_end"]], f"\nslope s = {slope:.4g} per photon  ({time.time() - t0:.0f} s)",
          flush=True)


def _grid_chunk(cells, slope, n_c, channels, path):
    rows = []
    for label, spec in cells:
        t0 = time.time()
        for r in evaluate(spec, ETAS, slope, n_c, channels):
            rows.append({"cell": label, "seconds": time.time() - t0, **r})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path, len(cells)


def cmd_grid(a):
    sigma, slope = default_sigma(a), residual_slope(a)
    cells = grid_cells(sigma)
    parts = _out(a, "grid_parts")
    os.makedirs(parts, exist_ok=True)
    chunks = [cells[i:i + a.chunk] for i in range(0, len(cells), a.chunk)]
    todo = [(i, ch) for i, ch in enumerate(chunks) if not os.path.exists(os.path.join(parts, f"chunk_{i:04d}.csv"))]
    print(f"{len(cells)} cells in {len(chunks)} chunks, {len(todo)} to run; sigma {sigma} ns, s {slope:.4g}",
          flush=True)
    t0 = time.time()
    for path, n in Parallel(n_jobs=a.n_jobs, backend="loky", return_as="generator_unordered")(
            delayed(_grid_chunk)(ch, slope, a.n_c, a.channels, os.path.join(parts, f"chunk_{i:04d}.csv"))
            for i, ch in todo):
        print(f"{os.path.basename(path)} ({n} cells)  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(parts, "chunk_*.csv")))], ignore_index=True)
    df.to_csv(_out(a, "aqec_window_grid.csv"), index=False)
    print(f"merged {len(df)} rows -> {_out(a, 'aqec_window_grid.csv')}")


def spec_from_row(r):
    drive = ro.DriveSpec(r["shape"], r["nbar"], r["t_drive"], kick=r["kick"], t_kick=r["t_kick"],
                         t_dep1=r["t_dep1"], t_dep2=r["t_dep2"], detuning_mhz=r["detuning_mhz"])
    t_int = None if abs(r["t_int_us"] - r["t_drive_us"]) < 5e-4 else float(r["t_int_us"])
    return aw.WindowSpec(drive=drive, t_int=t_int, latency=r["latency"], pi=aw.PiSpec("gauss", r["pi_sigma"]),
                         nres_target=r["nres_target"], eta=r["eta"], bias=r["bias"])


def tag_of(eta, latency, cap=N_CAP, kind=""):
    t = f"rw_e{int(round(eta * 100)):02d}_l{int(round(latency * 10)):d}"
    if cap != N_CAP:
        t += f"_c{int(cap)}"
    return t + (f"_{kind}" if kind else "")


def optimise_one(job, slope, n_c, channels, out_dir, maxfev):
    eta, latency, cap, kind, start = job["eta"], job["latency"], job["cap"], job["kind"], job["start"]
    tag = tag_of(eta, latency, cap, kind)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    logf = open(os.path.join(out_dir, "logs", f"opt_{tag}.log"), "a")
    t0 = time.time()
    base = start
    n_seg = 8 if kind == "free" else 0
    if kind == "free":
        base = start.replace()
        base.drive.shape = "free"
        segs = [base.drive.kick if (i + 0.5) / n_seg * base.drive.t_drive < base.drive.t_kick else 1.0
                for i in range(n_seg)]
        base.drive.segments = tuple(complex(v) for v in segs)
    names = param_names(base.drive.shape)
    x0 = [PARAMS[nm][0](base) for nm in names] + ([v for c in base.drive.segments for v in (c.real, c.imag)]
                                                   if n_seg else [])
    steps = [PARAMS[nm][4] for nm in names] + [0.2] * (2 * n_seg)
    best = {"J": -np.inf}

    def cost(x):
        s = apply_params(base, names, x, n_seg)
        try:
            r = evaluate(s, (eta,), slope, n_c, channels)[0]
        except (ValueError, np.linalg.LinAlgError) as e:
            logf.write(f"invalid: {e}\n")
            return 1.0
        J = r["J"] - (10.0 * (r["n_peak"] - cap) / cap if r["n_peak"] > cap else 0.0)
        if J > best["J"]:
            best.update(J=J, row=r, spec=s)
            logf.write(f"[{time.time() - t0:7.0f} s] J {J:.6f} F {r['F_pro']:.6f} t_w {r['t_w_us']:.3f} "
                       f"n_peak {r['n_peak']:.2f}\n")
            logf.flush()
        return -J

    simplex = np.array([x0] + [np.array(x0) + np.eye(len(x0))[i] * steps[i] for i in range(len(x0))])
    minimize(cost, x0, method="Nelder-Mead", options={"maxfev": maxfev, "initial_simplex": simplex,
                                                      "xatol": 1e-4, "fatol": 1e-7})
    spec = best["spec"].replace(eta=eta, bias=best["row"]["bias"])
    path = os.path.join(out_dir, "windows", f"aqec_window_{tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    aw.save_spec(spec, path, tag=tag, row=best["row"])
    logf.write("done\n")
    return {"tag": tag, "kind": kind or "best", "cap": cap, "seconds": time.time() - t0, **best["row"]}


def opt_jobs(grid, sigma):
    jobs = []

    def start(eta, latency, cap, shapes=None):
        g = grid[(grid.eta == eta) & (grid.latency == latency) & (grid.n_peak <= cap)]
        if shapes is not None:
            g = g[g["shape"].isin(shapes)]
        r = g.loc[g.J.idxmax()]
        return spec_from_row(r)

    for eta in ETAS:
        for lat in LATENCIES:
            jobs.append(dict(eta=eta, latency=lat, cap=N_CAP, kind="", start=start(eta, lat, N_CAP)))
    e, lat = HEADLINE
    for cap in (5.0, 20.0):
        jobs.append(dict(eta=e, latency=lat, cap=cap, kind="", start=start(e, lat, cap)))
    jobs.append(dict(eta=e, latency=lat, cap=N_CAP, kind="passive", start=start(e, lat, N_CAP, ("square", "kick"))))
    # sensitivity: the storage-readout cross-Kerr chi_ar (9.3 kHz, from Shirol's text only) switched off
    jobs.append(dict(eta=e, latency=lat, cap=N_CAP, kind="nochiar", start=start(e, lat, N_CAP).replace(chi_ar=0.0)))
    return jobs


def breakdown_rows(tag, spec, n_c=N_C):
    """One channel group at a time (and closed) for one window; the decision at the window's eta and bias."""
    groups = {"closed": frozenset(), "photon loss": frozenset({"a"}), "transmon decay": frozenset({"s_ge", "s_ef"}),
              "heating": frozenset({"up_ge", "up_ef", "up_a"}), "dephasing": frozenset({"phi_e", "phi_f", "phi_a"}),
              "all (Shirol)": None}
    rows = []
    for gname, keep in groups.items():
        m = aq.model(n_c, 1, "shirol") if keep is None else fwm.model_with_channels(n_c, 1, keep, True,
                                                                                    noise="shirol")
        K = aq.kraus(n_c, 1)
        Rg = fr.gate_step(m, np.load(os.path.join(REPO_ROOT, aq.EST_PULSE)), fr.inputs(n_c, 1))
        for variant, s in (("real window", spec), ("noiseless decision", spec.replace(decision="ideal")),
                           ("chi_ar off", spec.replace(chi_ar=0.0))):
            W = aw.Window(s, m)
            out = aw.real_step(W, Rg, K, "forgive")
            rows.append({"tag": tag, "channels": gname, "variant": variant,
                         **fr.score_state(out, n_c, 1, "X"), "coh_dropped": W.coh_dropped})
        rows.append({"tag": tag, "channels": gname, "variant": "ideal AQEC",
                     **fr.score_state(aq.aqec_step(m, Rg, K, "forgive"), n_c, 1, "X")})
        rows.append({"tag": tag, "channels": gname, "variant": "gate alone", **fr.score_state(Rg, n_c, 1, "X")})
    return rows


def cmd_optimise(a):
    slope = residual_slope(a)
    p = _table(a, "aqec_window_grid.csv")
    if p is None:
        raise SystemExit("run grid first")
    grid = pd.read_csv(p)
    grid = grid[grid.n_c == a.n_c]
    jobs = opt_jobs(grid, default_sigma(a))
    if a.only:
        jobs = [j for j in jobs if tag_of(j["eta"], j["latency"], j["cap"], j["kind"]) in a.only]
    t0 = time.time()
    rows = list(Parallel(n_jobs=a.n_jobs, backend="loky")(
        delayed(optimise_one)(j, slope, a.n_c, a.channels, a.out_dir, a.maxfev) for j in jobs))
    # free 8-segment shapes from the headline optimum and the eta 0.05 one
    free = []
    for eta in (HEADLINE[0], 0.05):
        tag = tag_of(eta, HEADLINE[1])
        r = next((x for x in rows if x["tag"] == tag), None)
        if r is not None and (not a.only or tag_of(eta, HEADLINE[1], N_CAP, "free") in a.only or tag in a.only):
            spec = aw.load_spec(os.path.join(a.out_dir, "windows", f"aqec_window_{tag}.json"))
            free.append(dict(eta=eta, latency=HEADLINE[1], cap=N_CAP, kind="free", start=spec))
    rows += list(Parallel(n_jobs=a.n_jobs, backend="loky")(
        delayed(optimise_one)(j, slope, a.n_c, a.channels, a.out_dir, 3 * a.maxfev) for j in free))
    df = pd.DataFrame(rows)
    df.to_csv(_out(a, "aqec_window_opt.csv"), index=False)
    print(df[["tag", "F_pro", "J", "t_w_us", "t_int_us", "nbar", "shape", "bias", "n_res"]], flush=True)
    specs = {r["tag"]: aw.load_spec(os.path.join(a.out_dir, "windows", f"aqec_window_{r['tag']}.json"))
             for r in rows}
    bd = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(breakdown_rows)(t, s, a.n_c) for t, s in specs.items())
    pd.DataFrame([r for x in bd for r in x]).to_csv(_out(a, "aqec_window_breakdown.csv"), index=False)
    chk = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(_residual_row)(r["n_res"], a.n_c, a.channels)
                                                    for r in rows)
    pd.DataFrame([{**c, "tag": r["tag"], "J_linear": r["J"], "F_pro_window": r["F_pro"]}
                  for c, r in zip(chk, rows)]).to_csv(_out(a, "aqec_window_residual_check.csv"), index=False)
    print(f"optimise done ({(time.time() - t0) / 60:.1f} min)", flush=True)


def _window_path(a, tag):
    for p in (os.path.join(a.out_dir, "windows", f"aqec_window_{tag}.json"),
              os.path.join(REPO_ROOT, "pulses", "dgrape", f"aqec_window_{tag}.json")):
        if os.path.exists(p):
            return p
    raise SystemExit(f"no window json for {tag}")


def _chain_job(kind, tag, path, n_c, n_max, channels, open_pulse=None):
    from DGRAPE import aqec_sequences as asq
    t0 = time.time()
    if kind == "b1":
        rows = asq.b1("X", ("ideal AQEC", "ideal map + real window"), n_max, n_c, 1, channels, window=path)
    elif kind == "b2":
        rows = asq.b2("X", "ideal map + real window", n_max, n_c, 1, channels, window=path)
    else:
        rows = asq.b2("X", "open step + real window", n_max, n_c, 1, channels, window=path, open_pulse=open_pulse)
    return pd.DataFrame(rows).assign(window=tag, open_pulse=open_pulse, seconds=time.time() - t0)


def cmd_chains(a):
    jobs = []
    for tag in a.windows:
        p = _window_path(a, tag)
        jobs += [("b1", tag, p, a.n_c, a.n_max), ("b2", tag, p, a.n_c, a.n_max), ("b2", tag, p, N_C_CHECK, a.n_max)]
    dfs = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(_chain_job)(k, t, p, n, nm, a.channels)
                                                    for k, t, p, n, nm in jobs)
    df = pd.concat(dfs, ignore_index=True)
    for prot in ("B1", "B2"):
        df[df.protocol == prot].to_csv(_out(a, f"aqec_window_{prot}.csv"), index=False)
    print(f"-> {_out(a, 'aqec_window_B1.csv')}, aqec_window_B2.csv", flush=True)


def _open_single(tag, wpath, n_c, channels, pulse_dir):
    from DGRAPE import aqec_open as ao
    u = np.load(os.path.join(pulse_dir, f"u_{tag}.npy"))
    spec = aw.load_spec(wpath)
    return {"open_pulse": tag, "window": os.path.basename(wpath), "n_c": n_c, "channels": channels,
            "F_pro": ao.open_score(u, n_c, channels, spec)}


def cmd_score_open(a):
    jobs, chains = [], []
    for tag in a.open:
        info_path = next((p for p in (os.path.join(a.pulse_dir, f"{tag}.json"),) if os.path.exists(p)), None)
        if info_path is None:
            raise SystemExit(f"no {tag}.json in {a.pulse_dir}")
        with open(info_path) as f:
            info = json.load(f)
        wpath = info.get("window_path")
        if wpath is None or not os.path.exists(wpath):
            wtag = tag.split("aqec_open_T15_")[-1]
            wpath = _window_path(a, wtag)
        for n in (a.n_c, N_C_CHECK):
            jobs.append((tag, wpath, n))
        chains.append(("open", tag, wpath, a.n_c, a.n_max, os.path.join(a.pulse_dir, f"u_{tag}.npy")))
    single = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(_open_single)(t, w, n, a.channels, a.pulse_dir)
                                                       for t, w, n in jobs)
    pd.DataFrame(single).to_csv(_out(a, "aqec_window_open_single.csv"), index=False)
    dfs = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(_chain_job)(k, t, w, n, nm, a.channels, up)
                                                    for k, t, w, n, nm, up in chains)
    pd.concat(dfs, ignore_index=True).to_csv(_out(a, "aqec_window_open_B2.csv"), index=False)
    print(pd.DataFrame(single), flush=True)


def _check_row(tag, path, variant, n_c, bin_ns, channels, slope):
    spec = aw.load_spec(path).replace(bin_ns=bin_ns)
    r = evaluate(spec, (spec.eta,), slope, n_c, channels, free_bias=False)[0]
    return {"tag": tag, "variant": variant, **r}


def cmd_check(a):
    slope = residual_slope(a)
    jobs = []
    for tag in a.windows:
        p = _window_path(a, tag)
        jobs += [(tag, p, "reference", a.n_c, 10), (tag, p, "bins 5 ns", a.n_c, 5),
                 (tag, p, f"n_c {N_C_CHECK}", N_C_CHECK, 10)]
    rows = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(_check_row)(*j, a.channels, slope) for j in jobs)
    df = pd.DataFrame(rows)
    ref = df[df.variant == "reference"].set_index("tag").F_pro
    df["dF_vs_reference"] = df.F_pro - df.tag.map(ref)
    df.to_csv(_out(a, "aqec_window_check.csv"), index=False)
    print(df[["tag", "variant", "F_pro", "dF_vs_reference"]], flush=True)
    bad = df[(df.variant == "bins 5 ns") & (df.dF_vs_reference.abs() >= 1e-5)].tag.tolist() + \
        df[(df.variant == f"n_c {N_C_CHECK}") & (df.dF_vs_reference.abs() >= 3e-4)].tag.tolist()
    print("CONVERGED" if not bad else f"NOT CONVERGED: {sorted(set(bad))}", flush=True)


def cmd_dry_run(a):
    sigma = default_sigma(a)
    t0 = time.time()
    c = context(a.n_c, a.channels)
    t1 = time.time()
    spec = grid_spec("square", "passive_0.01", 5.0, 0.8, 0.3, sigma)
    rows = evaluate(spec, ETAS, 0.0, a.n_c, a.channels)
    t2 = time.time()
    rows = evaluate(spec, ETAS, 0.0, a.n_c, a.channels)
    t3 = time.time()
    print(json.dumps({"n_c": a.n_c, "context_s": t1 - t0, "first_cell_s": t2 - t1, "cell_s": t3 - t2,
                      "F_gate": c["F_gate"], "F_ideal_aqec": c["F_ideal"],
                      "cells_in_grid": len(grid_cells(sigma)),
                      "rows": [{k: r[k] for k in ("eta", "bias", "F_pro", "t_w_us", "n_res", "P_g_given_e",
                                                  "P_e_given_g")} for r in rows]}, indent=1, default=float))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["dry-run", "calib", "grid", "optimise", "chains", "score-open", "check"])
    p.add_argument("--n-c", type=int, default=N_C)
    p.add_argument("--channels", default=CHANNELS, choices=["shirol", "decay", "all"])
    p.add_argument("--n-jobs", type=int, default=9)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--pi-sigma", type=float, default=None)
    p.add_argument("--slope", type=float, default=None, help="cost per residual photon (default: calib's table)")
    p.add_argument("--chunk", type=int, default=12, help="grid cells per part file")
    p.add_argument("--maxfev", type=int, default=300, help="Nelder-Mead evaluations per optimisation")
    p.add_argument("--only", nargs="*", default=None, help="optimise: only these tags")
    p.add_argument("--windows", nargs="*", default=[])
    p.add_argument("--open", nargs="*", default=[])
    p.add_argument("--pulse-dir", default=os.path.join(REPO_ROOT, "pulses", "dgrape"))
    p.add_argument("--n-max", type=int, default=20)
    a = p.parse_args()
    a.out_dir = os.path.expanduser(a.out_dir)
    {"dry-run": cmd_dry_run, "calib": cmd_calib, "grid": cmd_grid, "optimise": cmd_optimise, "chains": cmd_chains,
     "score-open": cmd_score_open, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
