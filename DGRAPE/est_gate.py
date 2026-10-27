"""
An error-semitransparent (EsT) X gate on the Shirol chip (FWM_tails.ipynb §3): the EsT cost of EST/
(Roy, Wetherbee & Fatemi, arXiv:2603.15356) on the odd kitten code and the chip of the d2 gate,
trained in the CLOSED system and scored in the open system like d2.

WHY. A photon lost during d2 is scrambled by the rest of the pulse, so the FWM tail can repair only the
losses near the end of the gate (FWM_tails.ipynb §2.2). An EsT gate is also trained to move the error
words the way it moves the code words, so a loss at any time should end as a clean, repairable error.

SAME SYSTEM AS d2. Drift and controls are DGRAPE/model.py's at n_r = 1 (the readout is inert during a
gate), higher-order terms on; the code is DGRAPE/code.py's odd kitten code with the transmon in |g>.
Duration 1.1 us, dt 2 ns (N = 550), and d2's drive limits (DGRAPE/recipe.py): bands +-27 / +-33 MHz,
48 ns ramp, box |x| <= 25 on the pre-image, cap |eps| <= 25 rad/us per drive, and d2's seed-42 smooth
start. Only the cost differs from d2.

THE COST is EST/grape_eigh.cost_and_grad, unchanged (eigh propagator, exact adjoint):
    C1  1 - mean over the six cardinals of |<target|U(T)|psi>|^2          (the ordinary gate error)
    C2  1 - mean_{t,j} |<psi_Ej(t)| a |psi_Cj(t)>|^2 / ||a psi_Cj(t)||^2  (error transparency)
    C3  Var_t(v) / mean_t(v)^2 of the code trajectory's speed             (stops "parking")
    C4  mean_t sum_drives (|eps| - 25)_+^2                                (drive cap)
with EST's two-stage weights (EST/train_est.py SCHEDULES; validation/test_est_gate.py pins the copy):
    est   (1, 0.7, 7, 1)  then  (1, 0.1, 0, 1)
    ord   (1, 0, 0, 1)    twice               -- the control: same code path with C2 = C3 = 0
The error cardinals are a|cardinal> renormalized. They equal the cardinals built on the error words
(|0> + sqrt5 |4>)/sqrt6, |2> because a|0_L> and a|1_L> have the same norm sqrt3 (asserted in system()).

THE CHAIN is core.ramp.make_constraint_chain (d2's band -> ramp); dcost/dx is its adjoint of dcost/du.

METRICS. Eqs. 6-8 of the paper, via EST/diagnostics (delta_qec, leakage_Ej, eta_mismatch) unchanged, on
closed trajectories from diagnostics.propagate_states (core.grape_core.step_data, not the trainer's
propagator, so re-scoring is an independent check). Each is 0 for a perfectly transparent gate:
    Delta_QEC(t)  how far the evolved code words are from the Knill-Laflamme conditions for {I, a}
    L(t)          how much of the evolved error state has left the instantaneous error space
    eta(t)        distance between the code state's and the projected error state's Bloch vectors

Pulses go to pulses/dgrape/ (u_X_est_closed.npy, ...), never pulses/ or pulses/est/: the scripts there
assume other chips and codes.
"""

import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

from AQEC.odd_kitten import error_basis
from core.optimizer import make_smooth_warm_start
from core.ramp import make_constraint_chain
from DGRAPE.code import IDEAL_LOGICAL_U, embedded_basis
from DGRAPE.model import _QUIET, G, N_Q, control_hamiltonians, drift_hamiltonian, operators
from DGRAPE.recipe import RECIPE
from EST import diagnostics as dg
from EST.grape_eigh import cost_and_grad
from EST.kitten_code import _coeff_matrix, parked_fidelity

GATE = "X"
DT = 0.002                         # us, d2's grid
DURATION_US = 1.1                  # d2's duration
N_STEPS = int(round(DURATION_US / DT))
N_C_TRAIN = 20                     # one training truncation, as EST/; checked on others afterwards
SEED = 42                          # d2's seed
CAV_BAND = tuple(RECIPE["cav_band"])
TRA_BAND = tuple(RECIPE["tra_band"])
RAMP_NS = RECIPE["ramp_ns"]
U_BOX = RECIPE["hard_amp_limit"]   # L-BFGS-B box on the pre-image
EPS_MAX = RECIPE["penalties"]["amp_max"]
W_AMP = 1.0                        # C4 weight, as EST/
SCHEDULES = {                      # (w1, w2, w3) per stage, copied from EST/train_est.py
    "est": [(1.0, 0.7, 7.0), (1.0, 0.1, 0.0)],
    "ord": [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
}
OPTIONS = {"ftol": 1e-12, "gtol": 1e-8}    # EST/train_est.py
CLOSED_F1 = 0.999                  # EST's bar for "the gate closed"


def stage_weights(variant):
    """[(w1, w2, w3, w4)] per stage."""
    if variant not in SCHEDULES:
        raise KeyError(f"unknown variant {variant!r}; have {sorted(SCHEDULES)}")
    return [(w1, w2, w3, W_AMP) for w1, w2, w3 in SCHEDULES[variant]]


def tier(F1, gate=GATE):
    """EST's three tiers: 'parked' (progress (F1 - parked)/(1 - parked) <= 0.5: the pulse barely drove),
    'open' (drove, F1 < 0.999) or 'closed'."""
    p0 = parked_fidelity(gate)
    if (F1 - p0) / (1.0 - p0) <= 0.5:
        return "parked"
    return "open" if F1 < CLOSED_F1 else "closed"


def system(n_c, include_higher_order=True, gate=GATE):
    """Closed system at n_r = 1: H0 (d, d), Hc (4, d, d), A (storage a), code words (d, 2), the six code
    cardinals C0 and error cardinals E0 (d, 6), columns in EST.kitten_code.CARDINAL_ORDER, and the gate
    targets (d, 6)."""
    H0 = drift_hamiltonian(n_c, 1, include_higher_order)
    Hc = control_hamiltonians(n_c, 1)
    A = operators(n_c, 1)["a"]
    words = embedded_basis(n_c, 1)
    coeff = _coeff_matrix()
    C0 = words @ coeff
    with np.errstate(**_QUIET):
        raw = A @ C0
    E0 = raw / np.linalg.norm(raw, axis=0, keepdims=True)
    gq = np.zeros(N_Q)
    gq[G] = 1.0
    eb = error_basis(n_c)
    err_words = np.stack([np.kron(eb[:, k], gq) for k in range(2)], axis=1)
    if not np.allclose(E0, err_words @ coeff, atol=1e-12):
        raise AssertionError("a|cardinal> renormalized is not the error-word cardinal: the equal-norm "
                             "property of the code no longer holds")
    target = words @ (IDEAL_LOGICAL_U[gate] @ coeff)
    return SimpleNamespace(n_c=n_c, H0=H0, Hc=Hc, A=A, words=words, err_words=err_words, C0=C0, E0=E0,
                           target=target)


def start_preimage(seed=SEED, N=N_STEPS):
    """d2's cold start: core.optimizer.make_smooth_warm_start at the recipe's amplitude and cutoff."""
    return make_smooth_warm_start(N, amp_max=RECIPE["warm_start_amp"],
                                  cutoff_frac=RECIPE["warm_start_cutoff_frac"], seed=seed)


class EstObjective:
    """cost(x), grad(x) on the pre-image x (N, 4) for one weight tuple, sharing one evaluation per x.

        u = ramp(bandlimit(x)),   cost = sum_i w_i C_i(u)   (EST.grape_eigh.cost_and_grad)
    """

    def __init__(self, weights, n_c=N_C_TRAIN, N=N_STEPS, dt=DT, gate=GATE, cav_band=CAV_BAND,
                 tra_band=TRA_BAND, ramp_ns=RAMP_NS, eps_max=EPS_MAX, include_higher_order=True):
        self.weights = tuple(float(w) for w in weights)
        self.N, self.dt, self.gate, self.eps_max = N, dt, gate, eps_max
        self.sys = system(n_c, include_higher_order, gate)
        self.to_physical, self.to_preimage_grad = make_constraint_chain(N, dt, cav_band, tra_band, ramp_ns)
        self.n_evals = 0
        self._cache_x, self._cache = None, None

    def _terms(self, u, want_grad):
        s = self.sys
        with np.errstate(**_QUIET):
            cost, g, terms = cost_and_grad(u, s.H0, s.Hc, s.A, s.C0, s.E0, s.target, self.dt, self.weights,
                                           eps_max=self.eps_max, want_grad=want_grad)
        terms.update({"F1": 1.0 - terms["c1"], "F_ET": 1.0 - terms["c2"]})
        return cost, g, terms

    def evaluate(self, x):
        x = np.asarray(x, dtype=float).reshape(self.N, 4)
        if self._cache_x is not None and np.array_equal(x, self._cache_x):
            return self._cache
        cost, g_u, terms = self._terms(self.to_physical(x), True)
        self.n_evals += 1
        self._cache_x = x.copy()
        self._cache = (cost, self.to_preimage_grad(g_u).ravel(), terms)
        return self._cache

    def cost(self, x):
        return self.evaluate(x)[0]

    def grad(self, x):
        return self.evaluate(x)[1]

    def report(self, x):
        """Terms of the pre-image x without a gradient (plus the cost)."""
        x = np.asarray(x, dtype=float).reshape(self.N, 4)
        cost, _, terms = self._terms(self.to_physical(x), False)
        return {"cost": cost, **terms}


def train(variant="est", maxiter=2000, seed=SEED, n_c=N_C_TRAIN, init_x=None, gate=GATE, verbose=True,
          stage_hook=None, print_every=100):
    """Run the two-stage schedule from d2's start (or init_x) and return (u, x, info).

    Each stage is L-BFGS-B on the pre-image in the box |x| <= U_BOX, with EST's options, warm-started
    from the previous stage's x. An init_x outside the box is refused rather than clipped (as EST/).
    stage_hook(i, u, x, terms) runs after each stage (the CLI saves there)."""
    N = N_STEPS
    x = start_preimage(seed, N) if init_x is None else np.asarray(init_x, dtype=float).reshape(N, 4)
    if np.abs(x).max() > U_BOX:
        raise ValueError(f"start pre-image reaches {np.abs(x).max():.2f} > box {U_BOX}")
    stages = []
    t_run = time.time()
    obj = None
    for i, weights in enumerate(stage_weights(variant), start=1):
        obj = EstObjective(weights, n_c, N, DT, gate)
        start = obj.report(x)
        if verbose:
            print(f"=== {gate} / {variant} / stage {i}  weights={weights}  n_c={n_c}  N={N}", flush=True)
            print("    start  " + _fmt(start), flush=True)
        count = [0]
        t0 = time.time()

        def callback(xk, obj=obj, t0=t0, count=count):
            count[0] += 1
            if verbose and print_every and count[0] % print_every == 0:
                _, _, terms = obj.evaluate(xk)
                print(f"    it {count[0]:5d}  " + _fmt(terms) + f"  [{(time.time() - t0) / 60:.1f} min]",
                      flush=True)

        res = minimize(obj.cost, x.ravel(), jac=obj.grad, method="L-BFGS-B",
                       bounds=[(-U_BOX, U_BOX)] * (N * 4), options={"maxiter": maxiter, **OPTIONS},
                       callback=callback)
        x = res.x.reshape(N, 4)
        end = obj.report(x)
        if verbose:
            print("    end    " + _fmt(end) + f"   {res.message}  nit={res.nit}  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        stages.append({"stage": i, "weights": list(weights), "start": start, "end": end, "nit": int(res.nit),
                       "nfev": int(res.nfev), "message": str(res.message), "seconds": time.time() - t0,
                       "max_abs_preimage": float(np.abs(x).max()),
                       "frac_preimage_pinned": float(np.mean(np.abs(x) >= U_BOX * (1 - 1e-9)))})
        if stage_hook is not None:
            stage_hook(i, obj.to_physical(x), x.copy(), end)
    u = obj.to_physical(x)
    F1 = stages[-1]["end"]["F1"]
    eps_c = np.hypot(u[:, 0], u[:, 1]).max()
    eps_t = np.hypot(u[:, 2], u[:, 3]).max()
    info = {"gate": gate, "variant": variant, "seed": seed, "n_c": n_c, "N": N, "dt": DT,
            "duration_us": DURATION_US, "maxiter": maxiter, "init": "d2 smooth start" if init_x is None
            else "init_x", "cav_band": list(CAV_BAND), "tra_band": list(TRA_BAND), "ramp_ns": RAMP_NS,
            "u_box": U_BOX, "eps_max": EPS_MAX, "options": OPTIONS, "stages": stages, "F1": F1,
            "F_ET": stages[-1]["end"]["F_ET"], "tier": tier(F1, gate),
            "max_drive_modulus": {"cav": float(eps_c), "tra": float(eps_t)},
            "max_abs_preimage": float(np.abs(x).max()),
            "frac_preimage_pinned": float(np.mean(np.abs(x) >= U_BOX * (1 - 1e-9))),
            "wall_s": time.time() - t_run}
    return u, x, info


def _fmt(terms):
    return (f"F1 {terms['F1']:.6f}  F_ET {terms['F_ET']:.4f}  c3 {terms['c3']:.4f}  c4 {terms['c4']:.2e}"
            + (f"  cost {terms['cost']:.6e}" if "cost" in terms else ""))


# ---------------------------------------------------------------------------
# Scoring in the closed system
# ---------------------------------------------------------------------------

def metrics(u, n_c=N_C_TRAIN, gate=GATE, include_higher_order=True, dt=DT, fock_thresh=0.01):
    """Eqs. 6-8 over the gate, closed system. Returns t (us), delta_qec (T,), leakage (T,) and eta_avg (T,)
    (means over the six cardinals, i.e. the maximally mixed error state), eta_0L (|0_L> alone, the
    paper's Fig. 1f state), F1 (closed six-cardinal fidelity), max_active_fock (highest storage level
    above fock_thresh on any cardinal at any time) and fock_peak (n_c,)."""
    s = system(n_c, include_higher_order, gate)
    Psi0 = np.concatenate([s.words, s.C0, s.E0], axis=1)
    with np.errstate(**_QUIET):
        traj = dg.propagate_states(np.asarray(u, dtype=float), s.H0, list(s.Hc), Psi0, dt)
        words, code, err = traj[:, :, :2], traj[:, :, 2:8], traj[:, :, 8:]
        eta = dg.eta_mismatch(words, code, err, s.A)
        leak = dg.leakage_Ej(words, err, s.A)
        dq = dg.delta_qec(words, s.A)
    F1 = float(np.mean(np.abs(np.sum(s.target.conj() * code[-1], axis=0)) ** 2))
    pop = (np.abs(code) ** 2).reshape(traj.shape[0], n_c, N_Q, 6).sum(axis=2)    # (T, n_c, 6)
    peak = pop.max(axis=(0, 2))
    return {"t": np.arange(traj.shape[0]) * dt, "delta_qec": dq, "leakage": leak.mean(axis=1),
            "eta_avg": eta.mean(axis=1), "eta_0L": eta[:, 0], "F1": F1,
            "max_active_fock": int(np.nonzero(peak > fock_thresh)[0].max()), "fock_peak": peak}


def summary(m):
    """Time means of the three metrics and the end-of-gate leakage, from metrics()."""
    return {"F1": m["F1"], "delta_qec_mean": float(m["delta_qec"].mean()),
            "leakage_mean": float(m["leakage"].mean()), "leakage_T": float(m["leakage"][-1]),
            "eta_mean": float(m["eta_avg"].mean()), "max_active_fock": m["max_active_fock"]}


def truncation_scan(u, trunc=(16, 18, 20, 24, 28), gate=GATE, include_higher_order=True, dt=DT):
    """{n_c: closed F1}: a pulse that does not use the truncation wall reads the same at every n_c."""
    out = {}
    for n_c in trunc:
        s = system(n_c, include_higher_order, gate)
        with np.errstate(**_QUIET):
            _, _, terms = cost_and_grad(np.asarray(u, dtype=float), s.H0, s.Hc, s.A, s.C0, s.E0, s.target, dt,
                                        (1.0, 0.0, 0.0, 0.0), want_grad=False)
        out[n_c] = 1.0 - terms["c1"]
    return out
