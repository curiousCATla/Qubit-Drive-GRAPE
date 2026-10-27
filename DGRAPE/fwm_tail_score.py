#!/usr/bin/env python3
"""
Held-out scoring of trained FWM tails (FWM_tails.ipynb §7 step 6 and §8). Forward only.

Each tail (eta_<tag>.npy, with <tag>.json for its gate, gate_n_c and length) is scored with
fwm_tail.score under the training condition and under conditions it was not trained on:

    train        n_c 16, the gate run at the json's gate_n_c and cut (the training fidelity)
    nc_a, nc_b   truncation, gate and tail both run there, no cut: n_c 24 and 28 after a closed-trained
                 gate, 18 and 20 after d2
    nr3          a second readout photon         all_ch       dephasing + heating, gate and tail
    g/sqrt2      coupling convention             eta_x0.9, eta_x1.1   pump miscalibration
    stark_off    reliance on the Stark model     mix1, mix2   pump-mixing a'r' term on
    closed, loss_only, no_loss                   channel split: no jumps / photon loss only / all
                                                 but photon loss (in the gate and the tail)

Every gain is against references scored in the SAME condition: "alone" (the gate, no tail) and
"idle" (the gate + a zero tail of the same length); under "train" also "teeth" (Shirol's Table II
teeth held on over the tail, raw, as notebook Table 8). Rows carry F_pro (readout traced), F_empty,
<n_r>, P_e, P_f at the end and max|eta_c|; gain_vs_alone and gain_vs_idle are added at the end.

    python DGRAPE/fwm_tail_score.py --tags X_tail_est_T1_t2 ... --indir ~/fwm_runs/tails \\
        --out tables/fwm_tail_heldout.csv
"""

import argparse
import json
import os
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

N_C_TRAIN = 16
HELD_OUT_NC = {"d2": (18, 20), "closed": (24, 28)}    # closed-trained gates reach Fock 15-17
DECAY = {"a", "r", "s_ge", "s_ef"}                    # the decay set (DGRAPE/model.collapse_ops)
CONDITIONS = {
    "train": {},
    "nc_a": None,                                     # filled per gate (HELD_OUT_NC)
    "nc_b": None,
    "nr3": {"n_r": 3},
    "all_ch": {"dephasing": True, "heating": True},
    "g/sqrt2": {"g_scale": 1 / np.sqrt(2)},
    "eta_x0.9": {"eta_scale": 0.9},
    "eta_x1.1": {"eta_scale": 1.1},
    "stark_off": {"stark": False},
    "mix1": {"mixing": 1.0},
    "mix2": {"mixing": 2.0},
    "closed": {"keep": set()},
    "loss_only": {"keep": {"a"}},
    "no_loss": {"keep": DECAY - {"a"}},
}


def gate_kind(gate_pulse):
    return "d2" if gate_pulse == "d2" else "closed"


def gate_name(gate_pulse):
    if gate_pulse == "d2":
        return "d2"
    base = os.path.basename(gate_pulse)
    return {"u_X_est_closed.npy": "EsT", "u_X_ord_closed.npy": "Ord"}.get(base, base)


def condition_kwargs(cond, gate_pulse, gate_n_c):
    """ft.score keyword arguments (minus eta) for one condition."""
    kw = {"n_c": N_C_TRAIN, "n_r": 2, "gate_n_c": gate_n_c,
          "pulse": None if gate_pulse == "d2" else np.load(os.path.join(REPO_ROOT, gate_pulse))}
    if cond in ("nc_a", "nc_b"):
        kw["n_c"] = HELD_OUT_NC[gate_kind(gate_pulse)][0 if cond == "nc_a" else 1]
        kw["gate_n_c"] = None
    else:
        kw.update(CONDITIONS[cond])
    return kw


def run(row, eta, cond):
    kw = condition_kwargs(cond, row["gate_pulse"], row["gate_n_c"])
    res = ft.score(eta, occupations=(cond == "train" and len(eta) > 0), **kw)
    return {**row, "condition": cond, "n_c_scored": kw["n_c"], **res}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--indir", default=os.path.join(os.path.expanduser("~"), "fwm_runs", "tails"))
    p.add_argument("--out", required=True)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--n-jobs", type=int, default=12)
    a = p.parse_args()

    jobs, refs = [], {}
    for tag in a.tags:
        with open(os.path.join(a.indir, f"{tag}.json")) as f:
            info = json.load(f)
        eta = np.load(os.path.join(a.indir, f"eta_{tag}.npy"))
        gp = info["gate_pulse"]
        base = {"gate": gate_name(gp), "gate_pulse": gp, "gate_n_c": info.get("gate_n_c", N_C_TRAIN),
                "tail_us": info["tail_us"]}
        for cond in a.conditions:
            jobs.append(({**base, "tag": tag, "kind": "trained", "start": info["start"]}, eta, cond))
            N = ft.n_steps(info["tail_us"])
            ref_etas = [("alone", np.zeros((0, 2), dtype=complex)), ("idle", np.zeros((N, 2), dtype=complex))]
            if cond == "train":                       # Shirol's Table II teeth, raw (notebook Table 8)
                ref_etas.append(("teeth", ft.eta_from_teeth(fwm.beta_table2(ft.REPAIR_PATHS), N, ft.DT,
                                                            N_C_TRAIN)))
            for kind, eta_ref in ref_etas:
                key = (gp, base["gate_n_c"], 0.0 if kind == "alone" else info["tail_us"], kind, cond)
                if key not in refs:
                    refs[key] = ({**base, "tail_us": key[2], "tag": kind, "kind": kind, "start": ""},
                                 eta_ref, cond)
    jobs += list(refs.values())
    rows = Parallel(n_jobs=a.n_jobs, backend="loky", verbose=5)(delayed(run)(*j) for j in jobs)
    df = pd.DataFrame(rows)

    # gains against the references of the same gate and condition
    def ref_map(kind, with_tail):
        r = df[df.kind == kind]
        cols = ["gate", "condition"] + (["tail_us"] if with_tail else [])
        return r.set_index(cols)["F_pro"]
    alone, idle = ref_map("alone", False), ref_map("idle", True)
    tr = df.kind == "trained"
    df.loc[tr, "gain_vs_alone"] = df[tr].F_pro.values - alone.reindex(
        pd.MultiIndex.from_frame(df.loc[tr, ["gate", "condition"]])).values
    df.loc[tr, "gain_vs_idle"] = df[tr].F_pro.values - idle.reindex(
        pd.MultiIndex.from_frame(df.loc[tr, ["gate", "condition", "tail_us"]])).values
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 240)
    print(df[tr].pivot(index="condition", columns="tag", values="gain_vs_alone").to_string(float_format="%+.5f"))


if __name__ == "__main__":
    main()
