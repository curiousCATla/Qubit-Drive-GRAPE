#!/usr/bin/env python3
"""
FWM tails after the EsT H gate (notebooks/FWM_tails.ipynb §7). Forward only; DGRAPE/fwm_tail.py is used unchanged.

The H gate is seed 17 of the C_err search at 2 us (notebooks/est_gate_selection.ipynb §7,
pulses/dgrape/u_H_est_err_closed_T2_s17.npy); the X gate is seed 24 at 1.1 us (u_X_est_err_closed_s24.npy) with
its best 0.8 us tail X_tail_estg_T08_t2. The H tail is trained by DGRAPE/train_fwm_tail.py with --target H
(tags H_tail_esth_T08_{t2,z}). Everything here follows the §5 recipe: the gate runs at n_c 24 and its output is
cut to n_c 16 for the tail, two readout levels, decay set, readout traced out.

Subcommands, each writing one cached CSV:

    baselines   each gate alone, + idle 0.8 us, + QEC teeth 0.8 us, + the X tail, + the H tail (if trained),
                each scored against that gate's own target; the ideal instant repair at T_g (n_c 24, n_r 1);
                the mid-gate loss profile (fraction of the gate's photon losses an ideal repair can undo)
                -> tables/fwm_tail_H_baselines.csv
    profiles    one photon lost at time t of the tail, gate and tail otherwise closed: the conditional F_pro at
                the end of the tail (notebook §6.2's test), for every gate x tail pair
                -> tables/fwm_tail_H_profiles.csv
    code        the tails with NO gate in front, closed: on the ideal code inputs and on clean one-loss error
                words a E a' (scored against I), and the logical Z rotation that fits best
                -> tables/fwm_tail_H_code.csv
    cross-json  pseudo-run jsons + eta copies in --indir, so DGRAPE/fwm_tail_score.py can score the swapped
                cases: H_xtail_T08 (the X tail after H, target H) and X_htail_T08 (the H tail after X, target X)

    python DGRAPE/fwm_tail_hgate.py baselines --n-jobs 8
"""

import argparse
import json
import os
import shutil
import sys

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from joblib import Parallel, delayed                  # noqa: E402

from DGRAPE import fwm                                # noqa: E402
from DGRAPE import fwm_tail as ft                     # noqa: E402
from DGRAPE.code import process_targets  # noqa: E402
from DGRAPE.model import operators                    # noqa: E402

PULSE_DIR = os.path.join(REPO_ROOT, "pulses", "dgrape")
TABLE_DIR = os.path.join(REPO_ROOT, "tables")
H_PULSE = "pulses/dgrape/u_H_est_err_closed_T2_s17.npy"
X_PULSE = "pulses/dgrape/u_X_est_err_closed_s24.npy"
GATES = {"EsT-X": (X_PULSE, "X"), "EsT-H": (H_PULSE, "H")}       # label -> (pulse, target)
X_TAIL = "X_tail_estg_T08_t2"                                      # the best X 0.8 us tail (notebook Table 6)
H_TAILS = ("H_tail_esth_T08_t2", "H_tail_esth_T08_z")
TAIL_US = 0.8
N_C, GATE_N_C, N_R = 16, 24, 2


def load_pulse(path):
    return np.load(os.path.join(REPO_ROOT, path))


def load_tail(tag, indir=PULSE_DIR):
    with open(os.path.join(indir, f"{tag}.json")) as f:
        info = json.load(f)
    return np.load(os.path.join(indir, f"eta_{tag}.npy")), info


def best_h_tail(indir=PULSE_DIR):
    """(tag, eta, info) of the H tail with the higher training F_pro, or None if none is trained yet."""
    found = []
    for tag in H_TAILS:
        if os.path.exists(os.path.join(indir, f"{tag}.json")):
            eta, info = load_tail(tag, indir)
            found.append((info["F_pro"], tag, eta, info))
    if not found:
        return None
    _, tag, eta, info = max(found, key=lambda r: r[0])
    return tag, eta, info


def tails(indir=PULSE_DIR):
    """{label: (tag, eta)}: idle, QEC teeth (bare carriers, as §6), the X tail and the best H tail."""
    N = ft.n_steps(TAIL_US)
    out = {"idle": ("idle", np.zeros((N, 2), dtype=complex)),
           "QEC teeth": ("teeth", ft.eta_from_teeth(fwm.beta_table2(ft.REPAIR_PATHS), N, ft.DT, N_C)),
           "X tail": (X_TAIL, load_tail(X_TAIL)[0])}
    h = best_h_tail(indir)
    if h is not None:
        out["H tail"] = (h[0], h[1])
    return out


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def _score(gate_label, tail_label, eta):
    pulse, target = GATES[gate_label]
    r = ft.score(eta, n_c=N_C, n_r=N_R, gate=target, pulse=load_pulse(pulse), gate_n_c=GATE_N_C)
    return {"gate": gate_label, "target": target, "tail": tail_label, "F_pro": r["F_pro"], "F_empty": r["F_empty"],
            "n_r_end": r["n_r_end"], "P_f_end": r["P_f_end"]}


def ceilings(gate_label, n_c=GATE_N_C):
    """Ideal instant repair at T_g and the mid-gate loss profile (n_r 1), as est_gate_selection §3/§7."""
    pulse, target = GATES[gate_label]
    u = load_pulse(pulse)
    R = ft.gate_output(u, n_c, 1)
    _, T, w = process_targets(target, n_c, 1)
    F0 = ft.fidelity_from_outputs(R, T, w)
    rep = ft.ideal_repair_fidelity(R, T, w, n_c, 1)
    prof = ft.loss_time_profile(rep["phases"], n_c=n_c, gate=target, pulse=u)
    p = prof["p"]
    return {"gate": gate_label, "target": target, "n_c": n_c, "T_gate_us": u.shape[0] * ft.DT, "F_alone": F0,
            "F_ideal_repair": rep["F"], "ideal_repair_gain": rep["F"] - F0,
            "loss_prob": float(p.sum()),
            "mid_gate_repaired": float(np.sum(p * prof["f_repaired"]) / p.sum()),
            "mid_gate_unrepaired": float(np.sum(p * prof["f_unrepaired"]) / p.sum())}


def run_baselines(n_jobs, out):
    tl = tails()
    jobs = [(g, "none", np.zeros((0, 2), dtype=complex)) for g in GATES]
    jobs += [(g, lab, eta) for g in GATES for lab, (_, eta) in tl.items()]
    rows = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_score)(*j) for j in jobs)
    rows += Parallel(n_jobs=min(n_jobs, 2), backend="loky")(delayed(ceilings)(g) for g in GATES)
    df = pd.DataFrame(rows)
    alone = df[df["tail"] == "none"].set_index("gate").F_pro
    m = df["tail"].notna() & (df["tail"] != "none")
    df.loc[m, "gain_vs_alone"] = df[m].F_pro.values - alone.reindex(df[m].gate).values
    df.to_csv(out, index=False)
    print(df.to_string(float_format="%.6f"))
    return df


# ---------------------------------------------------------------------------
# profiles: one photon lost at time t of the tail (notebook §6.2, cell "Why the shot sits at the end")
# ---------------------------------------------------------------------------

def loss_branch_profile(eta, pulse, target, stride=10):
    """Rows {t_ns, f_branch}: the gate (n_c 24, cut to 16) and the tail run closed; at t_k one photon is
    removed by hand and the branch runs to the end of the tail; f_branch = its F_pro against `target` divided
    by the branch weight (per kappa_a dt). Same as the notebook's loss_branch_profile with the gate and target
    made parameters."""
    tm = ft.tail_model(N_C, N_R, keep=set())
    Tt, Wt = ft.tail_targets(N_C, N_R, target)
    R = ft.tail_input(load_pulse(pulse), N_C, N_R, GATE_N_C, keep=set())
    a = operators(N_C, N_R)["a"]
    rows = []
    for k in range(0, len(eta) + 1, stride):
        Rb = np.einsum("ij,bjk,lk->bil", a, R, a.conj())
        p = np.mean([np.real(np.trace(Rb[b])) for b in (0, 1)])
        Rend = ft.propagate(tm, eta[k:], Rb)[0] if k < len(eta) else Rb
        rows.append({"t_ns": k * ft.DT * 1e3, "f_branch": ft.fidelity_from_outputs(Rend, Tt, Wt) / p})
        if k < len(eta):
            R = ft.propagate(tm, eta[k:k + stride], R)[0]
    return rows


def run_profiles(n_jobs, out, stride=10):
    tl = tails()
    pairs = [(g, lab) for g in GATES for lab in ("X tail", "H tail", "idle") if lab in tl]
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(loss_branch_profile)(tl[lab][1], GATES[g][0], GATES[g][1], stride) for g, lab in pairs)
    rows = [{"gate": g, "tail": lab, "tag": tl[lab][0], **r} for (g, lab), rr in zip(pairs, res) for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.groupby(["gate", "tail"]).f_branch.agg(["first", "max", "last"]).to_string(float_format="%.4f"))
    return df


# ---------------------------------------------------------------------------
# code: the tails with no gate in front
# ---------------------------------------------------------------------------

def tail_on_code(label, tag, eta):
    """Closed tail with ideal inputs: the code process inputs E_b, and the clean one-loss error words a E_b a'
    (normalised by their mean weight). F against I, and against the best logical Z rotation Rz(theta).
    Now fwm_tail.code_word_scores (closed by default)."""
    return [{"tail": label, "tag": tag, **r} for r in ft.code_word_scores(eta, N_C, N_R)]


def run_code(n_jobs, out):
    tl = {"none": ("none", np.zeros((0, 2), dtype=complex)), **tails()}
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(tail_on_code)(lab, tag, eta)
                                                  for lab, (tag, eta) in tl.items())
    df = pd.DataFrame([r for rr in res for r in rr])
    df["kitten_limit"] = ft.kitten_repair_limit()
    df.to_csv(out, index=False)
    print(df.to_string(float_format="%.5f"))
    return df


# ---------------------------------------------------------------------------
# cross-json: swapped cases for DGRAPE/fwm_tail_score.py
# ---------------------------------------------------------------------------

def write_cross_runs(indir, h_dir=PULSE_DIR):
    """Pseudo-runs (json + eta) for the scorer: the X tail after H (target H), the best H tail after X (target X)."""
    os.makedirs(indir, exist_ok=True)
    h = best_h_tail(h_dir)
    if h is None:
        raise RuntimeError("no trained H tail yet")
    cases = {"H_xtail_T08": (X_TAIL, PULSE_DIR, H_PULSE, "H"), "X_htail_T08": (h[0], h_dir, X_PULSE, "X")}
    for new, (src, sdir, gate_pulse, target) in cases.items():
        eta, info = load_tail(src, sdir)
        np.save(os.path.join(indir, f"eta_{new}.npy"), eta)
        with open(os.path.join(indir, f"{new}.json"), "w") as f:
            json.dump({"tag": new, "start": f"tail {src}", "gate_pulse": gate_pulse, "gate_n_c": GATE_N_C,
                       "tail_us": info["tail_us"], "target": target, "F_pro": float("nan")}, f, indent=1)
    for tag in H_TAILS:                                # the trained H tails next to them, for one scorer call
        for fn in (f"{tag}.json", f"eta_{tag}.npy"):
            src = os.path.join(h_dir, fn)
            if os.path.exists(src) and os.path.abspath(h_dir) != os.path.abspath(indir):
                shutil.copy(src, os.path.join(indir, fn))
    print(f"wrote {sorted(cases)} to {indir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("baselines", "profiles", "code", "cross-json"))
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--out", default=None)
    p.add_argument("--indir", default=os.path.join(os.path.expanduser("~"), "fwm_runs", "tails_H", "score"))
    p.add_argument("--h-dir", default=PULSE_DIR, help="where the trained H tails are (default pulses/dgrape)")
    a = p.parse_args()
    default = {"baselines": "fwm_tail_H_baselines.csv", "profiles": "fwm_tail_H_profiles.csv",
               "code": "fwm_tail_H_code.csv"}
    if a.what == "cross-json":
        write_cross_runs(a.indir, a.h_dir)
        return
    out = a.out or os.path.join(TABLE_DIR, default[a.what])
    {"baselines": run_baselines, "profiles": run_profiles, "code": run_code}[a.what](a.n_jobs, out)


if __name__ == "__main__":
    main()
