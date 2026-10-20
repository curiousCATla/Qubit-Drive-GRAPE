#!/usr/bin/env python3
"""
Stage 2 landscape scan and loss-timing diagnostic (fwm_grape.ipynb §1.3 and §6.2). Forward only.

1. LOSS TIMING. With the d2 gate and no tones, switch storage photon loss on only for
   [t0, T_g] (two-segment propagation, n_c = 16, n_r = 1). cost(t0) = F(no loss) - F(loss from t0)
   is how much of the photon-loss cost comes from losses after t0.

2. LANDSCAPE. beta = 0 is a local maximum of F (a repair needs comb 1 AND comb 2, so its gain is
   4th order in beta, while crosstalk costs at 2nd order). L-BFGS-B must therefore start inside a
   basin where a repair is possible. Each candidate start is a pre-image x (N, 2K) that goes
   through the training chain (band, 48 ns ramp, /S_BETA), and is scored at the training
   settings (n_c = 16, CF4 S = 2, readout-traced target, decay set), with the pump-mixing term off
   and on (mixing = 1).

   --paths picks the tone set (default 1 3 5: the six Stage 2 tones, band +-2 MHz;
   --paths 1 2 3 4 5 6 7 8 9: the full comb of fwm_grape.ipynb §7, 18 tones, band +-1 MHz). For a
   set with odd-start teeth (n even, repair n-1 -> n from an odd n-1) the scan also tries the
   sequential starts with only the even-start teeth on ("_evenstart"): the odd-start teeth pump
   the odd code words themselves out of the code.

Outputs go to --outdir (default: a local folder outside iCloud, because iCloud renames files
rewritten while it syncs): scan.csv, loss_timing.csv, start_<name>.npy. Copy them into the repo
when the run is done.

    python DGRAPE/fwm_stage2_scan.py --n-jobs 14
    python DGRAPE/fwm_stage2_scan.py --paths 1 2 3 4 5 6 7 8 9 --outdir ~/fwm_runs/full --no-loss-timing
"""

import argparse
import os
import sys

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from joblib import Parallel, delayed                  # noqa: E402

from DGRAPE import fwm                                # noqa: E402
from DGRAPE import lindblad_np as lnp                 # noqa: E402
from DGRAPE.code import process_targets               # noqa: E402

DT = 0.002
U_D2 = os.path.join(REPO_ROOT, "pulses", "dgrape", "u_X_open_d2.npy")
DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "fwm_runs")


def swap_betas(length_us, factor=1.0, tones=fwm.TONES):
    """Per-tone |beta| that gives a full swap (rate * length = pi/2) in a window of length_us,
    times factor, clipped at the cap."""
    b = np.array([np.pi / 2 / (fwm.tone_rate(k, 1.0, tones=tones) * length_us)
                  for k in range(len(tones))])
    return np.minimum(factor * b, fwm.BETA_CAP)


def start_library(N, paths=fwm.dev.CORRECTED_FOCK, band=None):
    """{name: pre-image x (N, 2K)} of the candidate starts for the tone set `paths`. `band` does
    not enter x (the training chain applies it); it is accepted so callers pass the set's whole
    recipe in one place."""
    T = N * DT
    tones = fwm.tone_list(paths)
    t2 = fwm.beta_table2(paths).astype(complex)
    lib = {"off": np.zeros((N, 2 * len(tones)))}
    for f in (1, 3, 9):
        lib[f"tableII_x{f}_full"] = fwm.constant_preimage(N, np.minimum(f * t2, fwm.BETA_CAP), dt=DT)
    for f in (3, 9):
        lib[f"tableII_x{f}_last0.5"] = fwm.constant_preimage(
            N, np.minimum(f * t2, fwm.BETA_CAP), window=(T - 0.5, T), dt=DT)
    # Sequential repair: comb 1 swaps in window A, then comb 2 empties the transmon in window B.
    # "_evenstart": only the teeth whose repair starts from an even n-1 (n odd), if the set has others.
    even_start = np.array([1.0 if n % 2 == 1 else 0.0 for _, n in tones])
    variants = [("", np.ones(len(tones)))]
    if not even_start.all():
        variants.append(("_evenstart", even_start))
    for label, A, B in (("short", (T - 0.5, T - 0.25), (T - 0.25, T)),
                        ("long", (T - 0.7, T - 0.35), (T - 0.35, T))):
        for f in (0.5, 1.0):
            for suffix, sel in variants:
                b1 = swap_betas(A[1] - A[0], f, tones) * fwm.comb_mask(1, tones) * sel
                b2 = swap_betas(B[1] - B[0], f, tones) * fwm.comb_mask(2, tones) * sel
                x = (fwm.constant_preimage(N, b1, window=A, dt=DT)
                     + fwm.constant_preimage(N, b2, window=B, dt=DT))
                lib[f"seq_{label}_f{f}{suffix}"] = x
    A, B = (T - 0.7, T - 0.35), (T - 0.35, T)
    lib["seq_long_comb1_only"] = fwm.constant_preimage(
        N, swap_betas(0.35, 1.0, tones) * fwm.comb_mask(1, tones), window=A, dt=DT)
    lib["seq_long_comb2_only"] = fwm.constant_preimage(
        N, swap_betas(0.35, 1.0, tones) * fwm.comb_mask(2, tones), window=B, dt=DT)
    return lib


def score_start(name, x, u, n_c, S, paths=fwm.dev.CORRECTED_FOCK, band=None):
    band = fwm.default_band(paths) if band is None else band
    chain = fwm.BetaChain(u.shape[0], DT, band, n_tones=2 * len(paths))
    b12 = chain.to_physical(x)
    beta = fwm.beta_from_real(b12)
    row = {"start": name}
    for mixing in (0.0, 1.0):
        sc = fwm.score(u, beta, DT, n_c=n_c, S=S, mixing=mixing, paths=paths)
        key = "" if mixing == 0 else "_mix1"
        row["F_pro" + key] = sc["F_pro"]
        if mixing == 0:
            row["n_r_end"], row["P_f_end"] = sc["n_r_end"], sc["P_f_end"]
    row.update({k: v for k, v in fwm.pulse_stats(b12, fwm._cached_model(n_c, 2, True, 0.0, paths)[1], DT, S).items()
                if k != "max_beta_per_tone"})
    return row


def loss_timing(u, n_c=16, t0_grid=None):
    """[(t0, F)] with storage loss on only for [t0, T]; other decay channels always on."""
    N = u.shape[0]
    t0_grid = np.round(np.arange(0, N * DT + 1e-9, 0.1), 3) if t0_grid is None else t0_grid
    m_no = fwm.model_with_channels(n_c, 1, {"r", "s_ge", "s_ef"})
    m_all = fwm.model_with_channels(n_c, 1, None)
    E, T, w = process_targets("X", n_c, 1)
    spec = max(lnp.spec_for_pulse(m_no, u, DT), lnp.spec_for_pulse(m_all, u, DT), key=lambda p: p[0] * p[1])
    rows = []
    for t0 in t0_grid:
        k0 = int(round(t0 / DT))
        R = lnp.propagate(m_no, u[:k0], E, DT, spec) if k0 > 0 else E.astype(complex)
        R = lnp.propagate(m_all, u[k0:], R, DT, spec) if k0 < N else R
        rows.append({"t0_us": float(t0), "F_pro": float(np.sum(w * np.einsum("bij,bij->b", T.conj(), R).real) / 4)})
    df = pd.DataFrame(rows)
    F_none = df.F_pro.iloc[-1]
    df["loss_cost_after_t0"] = F_none - df.F_pro
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--n-c", type=int, default=16)
    p.add_argument("--S", type=int, default=fwm.S_TRAIN)
    p.add_argument("--n-jobs", type=int, default=14)
    p.add_argument("--paths", type=int, nargs="+", default=list(fwm.dev.CORRECTED_FOCK))
    p.add_argument("--band", type=float, nargs=2, default=None,
                   help="envelope band in MHz (default +-2 for paths 1 3 5, else +-1)")
    p.add_argument("--no-loss-timing", action="store_true", help="skip the loss-timing diagnostic")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    paths = tuple(a.paths)
    band = tuple(a.band) if a.band is not None else fwm.default_band(paths)
    u = np.load(U_D2)
    lib = start_library(u.shape[0], paths, band)
    for name, x in lib.items():
        np.save(os.path.join(a.outdir, f"start_{name}.npy"), x)
    jobs = [delayed(score_start)(name, x, u, a.n_c, a.S, paths, band) for name, x in lib.items()]
    if not a.no_loss_timing:
        jobs.append(delayed(loss_timing)(u, a.n_c))
    out = Parallel(n_jobs=a.n_jobs, backend="loky", verbose=5)(jobs)
    scan = pd.DataFrame(out[:len(lib)])
    F_off = float(scan.loc[scan.start == "off", "F_pro"].iloc[0])
    scan["change_vs_off"] = scan.F_pro - F_off
    scan.to_csv(os.path.join(a.outdir, "scan.csv"), index=False)
    pd.set_option("display.width", 200)
    print(scan.to_string(index=False))
    if not a.no_loss_timing:
        out[-1].to_csv(os.path.join(a.outdir, "loss_timing.csv"), index=False)
        print(out[-1].to_string(index=False))


if __name__ == "__main__":
    main()
