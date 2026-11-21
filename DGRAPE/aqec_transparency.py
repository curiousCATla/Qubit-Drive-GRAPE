"""
Is the open-trained AQEC step error transparent to one photon loss? notebooks/AQEC_compare.ipynb §6.2, the AQEC version
of notebooks/FWM_tails.ipynb §9. Forward only.

WHAT RUNS. The EsT X gate (aqec_ideal.EST_PULSE), then one AQEC step: step 1 = the 1.5 us open-trained pulse
(pulses/dgrape/u_aqec_open_T15_w.npy, 1 ns steps), step 2 = the 1.2 us readout window (no drive), then the ideal
transmon reset (aqec_ideal.forgive). This is aqec_pulse.open_step, mode "as is", split so that a jump can be put in.

NOISE. Only the readout decays (keep = {"r"}), as in FWM_tails §9. The AQEC chain runs with one readout level (exact:
nothing fills the readout), where the readout's decay operator is zero, so this is the CLOSED system plus one photon
loss put in by hand. The no-jump run therefore equals notebook Table 13's "closed system (gate and step)" F_pro.

ONE BRANCH PER LOSS TIME t (every STRIDE_NS over the whole 2.7 us step). The no-jump state at t is taken from one saved
run; one photon is removed by hand, rho -> a rho a' / p (p = mean trace over |0_L>, |1_L> = <n>); the rest of the step
runs unchanged (it does not know a photon was lost); the branch is scored after the reset against X:
    as is            fwm_repeat.score_state
    + ideal repair   the ideal AQEC map of aqec_ideal (code word -> itself; error word -> code word, transmon flagged),
                     then a reset, with two free phases (below) fitted once on the weight-summed branch of the whole
                     step, as FWM_tails §9.1 fitted its photon-add repair. Ceiling 1: the map undoes a loss exactly.

THE REPAIR WITH FREE PHASES. A loss at time t leaves the error word with phases the fixed map does not expect (Kerr,
chi, the pre-twist). repair_kraus(phases=(phi1, phi2)) uses the error words
    |0_E(phi1)> = (|0> + sqrt5 e^{i phi1} |4>)/sqrt6,   |1_E> = |2>,
    K_e = (|0_L><0_E(phi1)| + e^{i phi2} |1_L><1_E|) (x) |e><g|,   K_g = (1 - P_E(phi1)) (x) |g><g|;
at phases (0, 0) it is aqec_ideal.kraus.

NEXT PAIR (FWM_tails §9.2). A loss at time t of the first step, then gate 2 and a second AQEC step; scored after pair 1
(vs X), after gate 2 (vs I), after pair 2 (vs I), the first two also + the ideal repair (one phase set per checkpoint,
fitted on all loss times together, weighted).

Run from the repo root:
    python3 DGRAPE/aqec_transparency.py one   --n-jobs 8   -> tables/aqec_transparency_{jumps,summary}.csv
    python3 DGRAPE/aqec_transparency.py pair2 --n-jobs 8   -> tables/aqec_transparency_pair2.csv
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AQEC import device as dev                                      # noqa: E402
from AQEC.odd_kitten import error_basis, logical_basis              # noqa: E402
from DGRAPE import aqec_ideal as aq                                 # noqa: E402
from DGRAPE import aqec_pulse as ap                                 # noqa: E402
from DGRAPE import fwm                                              # noqa: E402
from DGRAPE import fwm_repeat as fr                                 # noqa: E402
from DGRAPE import fwm_tail as ft                                   # noqa: E402
from DGRAPE import lindblad_np as lnp                               # noqa: E402
from DGRAPE.model import _QUIET, N_Q, E, G, operators              # noqa: E402

N_C = 24
STEP_TAG = "aqec_open_T15_w"
STEP_PULSE = f"pulses/dgrape/u_{STEP_TAG}.npy"
DT_STEP = ap.DT             # 1 ns, the step pulse's grid (the gate runs on fr.DT = 2 ns)
TAU2 = ap.TAU2              # 1.2 us readout window
KEEP = frozenset({"r"})     # only the readout decays (zero operator at n_r = 1)
STRIDE_NS = 10              # loss times, one step
PAIR2_STRIDE_NS = 20        # loss times, next pair
TABLES = {"jumps": "tables/aqec_transparency_jumps.csv", "summary": "tables/aqec_transparency_summary.csv",
          "pair2": "tables/aqec_transparency_pair2.csv"}
# windows of the step for the tables (ns from the start of the step; the pulse ends at 1500, the window at 2700)
WINDOWS = {"pulse 0–500 ns": (0, 500), "pulse 500–1000 ns": (500, 1000), "pulse 1000–1500 ns": (1000, 1500),
           "readout window 1500–2700 ns": (1500, 2701), "whole step": (0, 2701)}

_MODELS = {}


def model(n_c=N_C, n_r=1):
    """DGRAPE model with only the readout decaying (cached per process)."""
    key = (n_c, n_r)
    if key not in _MODELS:
        _MODELS[key] = fwm.model_with_channels(n_c, n_r, KEEP, True)
    return _MODELS[key]


def step_pulse():
    return np.load(STEP_PULSE)


def gate_pulse():
    return np.load(aq.EST_PULSE)


def pulse_ns(u=None):
    """Length of the step pulse in ns (1500)."""
    u = step_pulse() if u is None else u
    return int(round(u.shape[0] * DT_STEP * 1e3))


# ---------------------------------------------------------------------------
# The ideal AQEC repair with two free phases
# ---------------------------------------------------------------------------

def repair_kraus(n_c=N_C, n_r=1, phases=(0.0, 0.0)):
    """(K_g, K_e) of the ideal AQEC map with free phases (module docstring); phases (0, 0) = aqec_ideal.kraus."""
    phi1, phi2 = phases
    B = logical_basis(n_c).astype(complex)
    Be = error_basis(n_c).astype(complex)
    Be[4, 0] *= np.exp(1j * phi1)
    PE = Be @ Be.conj().T
    Pq = np.zeros((N_Q, N_Q)); Pq[G, G] = 1
    Sq = np.zeros((N_Q, N_Q)); Sq[E, G] = 1
    L = np.outer(B[:, 0], Be[:, 0].conj()) + np.exp(1j * phi2) * np.outer(B[:, 1], Be[:, 1].conj())
    return aq._lift(np.eye(n_c) - PE, Pq, n_r).astype(complex), aq._lift(L, Sq, n_r).astype(complex)


def repair(R, n_c=N_C, n_r=1, phases=(0.0, 0.0)):
    """The ideal AQEC map, then the ideal reset."""
    return aq.forgive(aq.apply_map(R, repair_kraus(n_c, n_r, phases)), n_c, n_r)


def score(R, target, n_c=N_C, n_r=1):
    return fr.score_state(R, n_c, n_r, target)["F_pro"]


def repair_fidelity(R, target, n_c=N_C, n_r=1, n_starts=6, seed=0):
    """F_pro after the ideal repair, best phases (BFGS from n_starts random starts). {"F", "phases", "F_phase0"}."""
    T, w = fr._targets(n_c, n_r, target, "traced")

    def F_of(ph):
        return ft.fidelity_from_outputs(repair(R, n_c, n_r, ph), T, w)

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        r = minimize(lambda p: -F_of(p), rng.uniform(0, 2 * np.pi, 2), method="BFGS")
        if best is None or r.fun < best.fun:
            best = r
    return {"F": float(-best.fun), "phases": np.mod(best.x, 2 * np.pi), "F_phase0": F_of((0.0, 0.0))}


# ---------------------------------------------------------------------------
# The no-jump run and one branch
# ---------------------------------------------------------------------------

def mean_trace(R):
    return float(np.mean([np.real(np.trace(R[b])) for b in (0, 1)]))


def loss(R, n_c=N_C, n_r=1):
    """One photon removed by hand: (p, a R a' / p), p = mean trace over |0_L>, |1_L>."""
    a = operators(n_c, n_r)["a"]
    with np.errstate(**_QUIET):
        Rb = np.einsum("ij,bjk,lk->bil", a, R, a.conj())
    p = mean_trace(Rb)
    return p, (Rb / p if p > 1e-14 else None)


def no_jump_states(R_in, stride_ns=STRIDE_NS, n_c=N_C, u=None):
    """States of the no-jump step at every stride_ns: {t_ns: R} for t = 0 .. pulse end .. pulse end + 1200 (the last
    is the state just before the reset). R_in = the step's input (the gate's output)."""
    m = model(n_c)
    u = step_pulse() if u is None else u
    T_p = pulse_ns(u)
    s = int(round(stride_ns * 1e-3 / DT_STEP))
    with np.errstate(**_QUIET):
        R_p, hist = lnp.propagate(m, u, R_in, DT_STEP, spec=lnp.spec_for_pulse(m, u, DT_STEP), store=True)
    hist = list(hist) + [R_p]
    out = {k: hist[k] for k in range(0, len(hist), s)}
    R, t = R_p, T_p
    T_end = T_p + int(round(TAU2 * 1e3))
    while t < T_end:
        R = aq.idle(m, R, stride_ns * 1e-3)
        t += stride_ns
        out[t] = R
    return out


def rest_of_step(R, t_ns, n_c=N_C, u=None):
    """The rest of the step from t_ns (pulse remainder, window remainder), then the reset."""
    m = model(n_c)
    u = step_pulse() if u is None else u
    T_p = pulse_ns(u)
    k = int(round(t_ns * 1e-3 / DT_STEP))
    with np.errstate(**_QUIET):
        if k < u.shape[0]:
            R = lnp.propagate(m, u[k:], R, DT_STEP, spec=lnp.spec_for_pulse(m, u, DT_STEP))
        rest = (T_p + int(round(TAU2 * 1e3)) - max(t_ns, T_p)) * 1e-3
        if rest > 1e-12:
            R = aq.idle(m, R, rest)
        return aq.forgive(R, n_c, 1)


def full_step(R, n_c=N_C, u=None):
    """The whole AQEC step with no jump (= aqec_pulse.open_step on this model)."""
    return rest_of_step(R, 0, n_c, u)


def branch(R_t, t_ns, n_c=N_C):
    """One photon loss on the no-jump state at t_ns, then the rest of the step. Returns (p, end state / p or None)."""
    p, Rb = loss(R_t, n_c)
    if Rb is None:
        return p, None
    return p, rest_of_step(Rb, t_ns, n_c)


def first_input(n_c=N_C):
    """The gate's output (readout decay only = closed)."""
    with np.errstate(**_QUIET):
        return fr.gate_step(model(n_c), gate_pulse(), fr.inputs(n_c, 1))


def _populations(R, n_c=N_C):
    return aq.populations(R, model(n_c))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def window_of(t_ns, lo, hi):
    return lo <= t_ns < hi


def one_step(n_jobs=8, n_c=N_C, stride_ns=STRIDE_NS, log=print):
    """FWM_tails §9.1 for the AQEC step. Returns (jumps rows, summary rows)."""
    from joblib import Parallel, delayed
    t0 = time.time()
    Rg = first_input(n_c)
    states = no_jump_states(Rg, stride_ns, n_c)
    T_p = pulse_ns()
    R_end0 = aq.forgive(states[max(states)], n_c, 1)
    F_nojump = score(R_end0, "X", n_c)
    log(f"no jump: F_pro {F_nojump:.6f} ({time.time() - t0:.0f} s); {len(states)} branches")
    ts = sorted(states)
    out = Parallel(n_jobs=n_jobs)(delayed(branch)(states[t], t, n_c) for t in ts)
    ends = dict(zip(ts, out))
    log(f"branches done ({time.time() - t0:.0f} s)")
    ok = [t for t in ts if ends[t][1] is not None]
    R_all = sum(ends[t][0] * ends[t][1] for t in ok)
    fit = repair_fidelity(R_all / mean_trace(R_all), "X", n_c)
    ph = fit["phases"]
    log(f"repair phases {ph} (whole step summed: F {fit['F']:.4f}, at phases 0 {fit['F_phase0']:.4f})")
    rows = []
    for t in ts:
        p, R = ends[t]
        r = {"t_ns": t, "step": "pulse" if t < T_p else "window", "pop": p, "rate_per_us": dev.kappa_a,
             "stride_ns": stride_ns, "F_nojump": F_nojump, "phi1": ph[0], "phi2": ph[1]}
        if R is not None:
            pops = _populations(R, n_c)
            r.update(f_as_is=score(R, "X", n_c), f_repair=score(repair(R, n_c, 1, ph), "X", n_c),
                     f_repair_phase0=score(repair(R, n_c), "X", n_c), P_code=pops["P_code"], P_error=pops["P_error"])
        rows.append(r)
    srows = []
    for win, (lo, hi) in WINDOWS.items():
        sel = [t for t in ok if window_of(t, lo, hi)]
        R = sum(ends[t][0] * ends[t][1] for t in sel)
        R = R / mean_trace(R)
        f = repair_fidelity(R, "X", n_c)
        srows.append({"window": win, "n_branches": len(sel), "F_as_is": score(R, "X", n_c),
                      "F_repair_refit": f["F"], "F_repair_phase0": f["F_phase0"]})
    log(f"done ({time.time() - t0:.0f} s)")
    return rows, srows


def _pair2_branch(R_t, t_ns, n_c=N_C):
    """Loss at t_ns of the first step: rest of step 1, gate 2, step 2. (p, R after pair 1, after gate 2, after pair 2)."""
    p, R1 = branch(R_t, t_ns, n_c)
    if R1 is None:
        return p, None, None, None
    with np.errstate(**_QUIET):
        G2 = fr.gate_step(model(n_c), gate_pulse(), R1)
    return p, R1, G2, full_step(G2, n_c)


def next_pair(n_jobs=8, n_c=N_C, stride_ns=PAIR2_STRIDE_NS, log=print):
    """FWM_tails §9.2 for the AQEC step. Returns rows."""
    from joblib import Parallel, delayed
    t0 = time.time()
    Rg = first_input(n_c)
    states = no_jump_states(Rg, stride_ns, n_c)
    T_p = pulse_ns()
    P1 = aq.forgive(states[max(states)], n_c, 1)
    with np.errstate(**_QUIET):
        G2_0 = fr.gate_step(model(n_c), gate_pulse(), P1)
    ref = {"after pair 1": score(P1, "X", n_c), "after gate 2": score(G2_0, "I", n_c),
           "after pair 2": score(full_step(G2_0, n_c), "I", n_c)}
    log("no loss: " + ", ".join(f"{k} {v:.5f}" for k, v in ref.items()) + f" ({time.time() - t0:.0f} s)")
    ts = sorted(states)
    out = Parallel(n_jobs=n_jobs)(delayed(_pair2_branch)(states[t], t, n_c) for t in ts)
    log(f"branches done ({time.time() - t0:.0f} s)")
    good = [(t, o) for t, o in zip(ts, out) if o[1] is not None]
    wsum = sum(o[0] for _, o in good)
    tot = lambda j: sum(o[0] * o[j] for _, o in good) / wsum          # noqa: E731
    ph1 = repair_fidelity(tot(1), "X", n_c)["phases"]
    ph2 = repair_fidelity(tot(2), "I", n_c)["phases"]
    rows = []
    for t, (p, R1, G2, P2) in good:
        rows.append({"t_ns": t, "step": "pulse" if t < T_p else "window", "pop": p,
                     "after pair 1": score(R1, "X", n_c),
                     "after pair 1 + ideal repair": score(repair(R1, n_c, 1, ph1), "X", n_c),
                     "after gate 2": score(G2, "I", n_c),
                     "after gate 2 + ideal repair": score(repair(G2, n_c, 1, ph2), "I", n_c),
                     "after pair 2": score(P2, "I", n_c),
                     **{f"no loss, {k}": v for k, v in ref.items()}})
    log(f"done ({time.time() - t0:.0f} s)")
    return rows


def main(argv=None):
    import pandas as pd
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap_.add_argument("what", choices=("one", "pair2"))
    ap_.add_argument("--n-jobs", type=int, default=8)
    ap_.add_argument("--n-c", type=int, default=N_C)
    ap_.add_argument("--stride-ns", type=int, default=None)
    ap_.add_argument("--out-dir", default="tables")
    a = ap_.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    sfx = "" if a.n_c == N_C else f"_nc{a.n_c}"
    path = lambda k: os.path.join(a.out_dir, os.path.basename(TABLES[k]).replace(".csv", f"{sfx}.csv"))  # noqa: E731
    if a.what == "one":
        rows, srows = one_step(a.n_jobs, a.n_c, a.stride_ns or STRIDE_NS)
        pd.DataFrame(rows).to_csv(path("jumps"), index=False)
        pd.DataFrame(srows).to_csv(path("summary"), index=False)
        print(f"wrote {path('jumps')}, {path('summary')}")
    else:
        rows = next_pair(a.n_jobs, a.n_c, a.stride_ns or PAIR2_STRIDE_NS)
        pd.DataFrame(rows).to_csv(path("pair2"), index=False)
        print(f"wrote {path('pair2')}")


if __name__ == "__main__":
    np.seterr(**_QUIET)
    main()
