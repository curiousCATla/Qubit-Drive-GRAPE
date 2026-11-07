#!/usr/bin/env python3
"""
Held-out scoring of trained Gertler-style tails (FWM_tails_ge.ipynb §5-§6). Forward only.

Each tail (b4_<tag>.npy, with <tag>.json for its gate, length and cap) is scored with fwm_tail_ge.score
under the training condition and under conditions it was not trained on:

    train         n_c 16, transmon {g, e}, gate run at n_c 24 and cut (the training fidelity)
    nc28          gate and tail both at n_c 28, no cut (the go / no-go number)
    three_level   the same tail on the THREE-level transmon (f, alpha, chi_ef, T1_ef back in):
                  tests dropping |f> (leakage of the transmon comb, a' r' q on |f>)
    nr3           a second readout photon        all_ch       dephasing + heating, gate and tail
    g/sqrt2       coupling convention            pump_x0.9, pump_x1.1   both combs miscalibrated
    stark_off     reliance on the Stark model
    closed, loss_only, no_loss                   channel split: no jumps / photon loss only / all but
                                                 photon loss (in the gate and the tail)

Every gain is against references scored in the SAME condition: "alone" (the gate, no tail) and "idle"
(the gate + a zero tail of the same length); under "train" also "gertler" (Gertler's teeth at the run's
cap, raw). Rows carry F_pro (readout traced), F_empty, F_tracked (after the best free cavity-frame
rotation, fwm_tail_ge.frame_tracked_fidelity; reported, never trained), <n_r>, P_e, P_f at the end and the
comb maxima; gains come in both versions (gain_vs_alone, gain_vs_alone_tracked: tracked against tracked).

    python DGRAPE/fwm_tail_ge_score.py --tags X_tailge_est_T2_c20_g ... --indir ~/fwm_runs/tailge \\
        --out tables/fwm_tail_ge_heldout.csv

Chips (FWM_tails_ge_chips.ipynb): a tail trained on another chip carries "chip" (and "gate_chip") in its
json; it is then scored, with its references, on that chip (gate where it ran), and the extra condition
chi_ar_off drops the storage-readout cross-Kerr from the tail's chip. Every row also carries F_decoded
(fwm_tail_ge.decoded_fidelity: free theta n + phi n(n-1)/2) and gain_vs_alone_decoded.
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

from DGRAPE import fwm_tail_ge as ge                  # noqa: E402
from DGRAPE import ge_chip as gc                      # noqa: E402

DECAY = {"a", "r", "s_ge", "s_ef"}
CONDITIONS = {
    "train": {},
    "nc28": {"n_c": 28, "gate_n_c": None},
    "three_level": {"levels": 3},
    "nr3": {"n_r": 3},
    "all_ch": {"dephasing": True, "heating": True},
    "g/sqrt2": {"g_scale": 1 / np.sqrt(2)},
    "pump_x0.9": {"pump_scale": 0.9},
    "pump_x1.1": {"pump_scale": 1.1},
    "stark_off": {"stark": False},
    "closed": {"keep": set()},
    "loss_only": {"keep": {"a"}},
    "no_loss": {"keep": DECAY - {"a"}},
    "chi_ar_off": {},          # chip runs only: the tail's chip without chi_ar (handled in run)
}


def _chip(d):
    """A ge_chip.Chip from a json table_row (None stays None)."""
    return None if d is None else gc.derived(d["chi_ge_MHz"], d["chi_qr_MHz"])


def run(row, b4, cond, chip=None, gate_chip=None):
    import dataclasses
    kw = {"n_c": ge.N_C, "n_r": 2, "gate_n_c": row["gate_n_c"], "target": row["target"], **CONDITIONS[cond]}
    if cond == "chi_ar_off" and chip is not None:
        chip = dataclasses.replace(chip, chi_ar=0.0)
    res = ge.score(b4, row["gate"], chip=chip, gate_chip=gate_chip, decode=True, **kw)
    return {**row, "condition": cond, "n_c_scored": kw["n_c"], **res}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--indir", default=os.path.join(os.path.expanduser("~"), "fwm_runs", "tailge"))
    p.add_argument("--out", required=True)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--n-jobs", type=int, default=12)
    a = p.parse_args()

    jobs, refs = [], {}
    for tag in a.tags:
        with open(os.path.join(a.indir, f"{tag}.json")) as f:
            info = json.load(f)
        b4 = np.load(os.path.join(a.indir, f"b4_{tag}.npy"))
        T, cap = info["tail_us"], info["cap_M"]
        chip, gate_chip = _chip(info.get("chip")), _chip(info.get("gate_chip"))
        base = {"gate": info["gate"], "gate_n_c": info.get("gate_n_c", ge.GATE_N_C), "tail_us": T,
                "cap_M": cap, "target": info.get("target", "X"),
                "chip": gc.resolve(chip).tag if chip is not None else "shirol",
                "gate_chip": gate_chip.tag if gate_chip is not None else "shirol"}
        N = ge.n_steps(T)
        for cond in a.conditions:
            if cond == "chi_ar_off" and chip is None:
                continue
            jobs.append(({**base, "tag": tag, "kind": "trained", "start": info["start"]}, b4, cond, chip,
                         gate_chip))
            ref_b4 = [("alone", 0.0, 0.0, np.zeros((0, 4))), ("idle", T, 0.0, np.zeros((N, 4)))]
            if cond == "train":
                ref_b4.append(("gertler", T, cap, ge.start_library(T, cap, chip=chip)["gertler"]))
            for kind, T_ref, cap_ref, b in ref_b4:
                key = (base["gate"], base["chip"], base["gate_chip"], T_ref, cap_ref, kind, cond)
                if key not in refs:
                    refs[key] = ({**base, "tail_us": T_ref, "cap_M": cap_ref, "tag": kind, "kind": kind,
                                  "start": ""}, b, cond, chip, gate_chip)
    jobs += list(refs.values())
    rows = Parallel(n_jobs=a.n_jobs, backend="loky", verbose=5)(delayed(run)(*j) for j in jobs)
    df = pd.DataFrame(rows)

    key0 = ["gate", "chip", "gate_chip", "condition"]

    def ref_map(kind, with_tail, col):
        r = df[df.kind == kind]
        return r.set_index(key0 + (["tail_us"] if with_tail else []))[col]
    tr = df.kind == "trained"
    for col, suffix in (("F_pro", ""), ("F_tracked", "_tracked"), ("F_decoded", "_decoded")):
        alone, idle = ref_map("alone", False, col), ref_map("idle", True, col)
        df.loc[tr, "gain_vs_alone" + suffix] = df[tr][col].values - alone.reindex(
            pd.MultiIndex.from_frame(df.loc[tr, key0])).values
        df.loc[tr, "gain_vs_idle" + suffix] = df[tr][col].values - idle.reindex(
            pd.MultiIndex.from_frame(df.loc[tr, key0 + ["tail_us"]])).values
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 240)
    print(df[tr].pivot(index="condition", columns="tag", values="gain_vs_alone").to_string(float_format="%+.5f"))


if __name__ == "__main__":
    main()
