"""
Open-system training of the real AQEC step (Cornell step 1 + readout window + reset) on Shirol's chip, after the EsT
X gate. The closed-trained pulses of DGRAPE/aqec_pulse.py are the warm starts. Notebook: notebooks/AQEC_compare.ipynb §6.

WHAT IS TRAINED. The cost is the score of the whole step, exactly as DGRAPE/aqec_pulse.open_step runs it ("as is"):

    rho_gate -> pulse u (lossy, Lindblad, dt = 1 ns) -> tau2 = 1.2 us lossy idle (the readout window) -> ideal reset
    F = F_pro(final state vs X), transmon in g, readout traced (fwm_repeat.score_state)

rho_gate is the EsT gate's output (E_00, E_11, E_01 through the seed-24 gate at gate_n_c = 24, cut to the training
n_c, as fwm_tail.tail_input does for the FWM tails). So this is the FWM-tail training problem with the tail replaced
by a cavity + transmon drive followed by the fixed window and reset.

WHY THE OLD GRADIENT STILL APPLIES. The window and the reset do not depend on u. F is linear in the final state:
F = sum_b w_b Re Tr[T_b' Reset(Idle(rho_b(T)))] / 4 = sum_b w_b Re Tr[Tw_b' rho_b(T)] / 4 with the folded target

    Tw_b = Idle'(Reset'(T_b)),   Reset'(Y) = <g|Y|g> (x) 1_q,   Idle' = exp(L' tau2) (one Taylor-split adjoint step).

Tw is computed once; lindblad_np.fidelity_and_grad with T -> Tw is then the exact gradient of the whole chain.

THE COST (minimised on the pre-image x, u = ramp(bandlimit(x)), the closed pulse's chain):

    1 - F  +  C4 (EST's drive cap, weight 1, as the closed training)  +  lambda_forbid * C_forbid

C_forbid = sum over the pulse's steps of Tr[P_F rho_b] for b = E_00, E_11, P_F = Fock >= 16 (DGRAPE's forbidden-state
penalty, a truncation guard). lambda_forbid = 5e-4: DGRAPE's 1e-3 per 2 ns step, halved for the 1 ns grid (same weight
per unit time). The guard sits at 16, not DGRAPE's 10, because the training n_c is 20, not 16, and the closed pulses use
Fock 10-15 heavily (2 us pulse: 2.9 % mean population at Fock >= 10, which at 10 would cost 0.057, more than half its
infidelity). Photon loss already charges photons physically; the guard only stops a lossy cost from being gamed at the
truncation wall (DGRAPE's first X run). Trained pulses are re-scored at n_c 24 and 28.

Channels: the decay set (photon loss, readout decay, transmon T1), as every DGRAPE and FWM-tail training. Pulses go to
~/fwm_runs/aqec_open first, then pulses/dgrape/u_aqec_open_*.npy.
"""

import time

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize

from core.ramp import make_constraint_chain
from DGRAPE import aqec_ideal as aq
from DGRAPE import aqec_pulse as ap
from DGRAPE import est_gate as eg
from DGRAPE import fwm_repeat as fr
from DGRAPE import fwm_tail as ft
from DGRAPE import lindblad_np as lnp
from DGRAPE.model import _QUIET, G, N_Q
from EST.grape_eigh import amplitude_cost

DT = ap.DT                 # 1 ns
TAU2 = ap.TAU2             # 1.2 us readout window
N_C_TRAIN = 20             # the closed pulses' training truncation
GATE_N_C = 24              # the EsT gate needs n_c 24
LAMBDA_FORBID = 5e-4
FORBID_FOCK_MIN = 16       # four guard levels below the training n_c 20 (DGRAPE: 10 at n_c 16)
OPTIONS = {"gtol": 1e-12, "ftol": 1e-15}      # DGRAPE/train.py: lossy gradients are small
CLOSED_START = {1.0: "aqec_T10_s42", 1.5: "aqec_T15_s43", 2.0: "aqec_T20_s42"}   # best closed pulse per length


def reset_adjoint(T, n_c, n_r=1):
    """Reset'(T_b) = <g|T_b|g> (x) 1_q for a stack (b, d, d): the Hilbert-Schmidt adjoint of aqec_ideal.forgive."""
    b = T.shape[0]
    S = T.reshape(b, n_c, N_Q, n_r, n_c, N_Q, n_r)
    Yg = S[:, :, G, :, :, G, :]                                     # (b, n, r, n', r')
    return np.einsum("bnrmq,jk->bnjrmkq", Yg, np.eye(N_Q)).reshape(T.shape).astype(complex)


def window_targets(m, T, tau2=TAU2):
    """Tw = Idle'(Reset'(T)) (module docstring)."""
    X = reset_adjoint(T, m.n_c, m.n_r)
    if tau2 == 0:
        return X
    u0 = np.zeros((1, 4))
    spec = lnp.spec_for_pulse(m, u0, tau2)
    with np.errstate(**_QUIET):
        return lnp.step_adjoint(m, u0[0], X, tau2, spec)


def gate_input(n_c, gate_n_c=GATE_N_C, channels="decay", gate_pulse=None):
    """(3, d, d) EsT-gate output at gate_n_c, cut to n_c (n_r = 1)."""
    gate = np.load(aq.EST_PULSE) if gate_pulse is None else np.asarray(gate_pulse, float)
    ng = max(n_c, gate_n_c)
    R = fr.gate_step(aq.model(ng, 1, channels), gate, fr.inputs(ng, 1))
    return ft.cut_storage(R, n_c, 1) if ng > n_c else R


_MODELS = {}


def _model(n_c, channels):
    key = (n_c, channels)
    if key not in _MODELS:
        _MODELS[key] = aq.model(n_c, 1, channels)
    return _MODELS[key]


def _piece(n_c, channels, R0, Tw, w, u, dt, forbid, want_grad):
    """One process input; module level so loky can pickle it (models cached per worker)."""
    m = _model(n_c, channels)
    with np.errstate(**_QUIET):
        if not want_grad:
            if forbid:
                R_N, hist = lnp.propagate(m, u, R0, dt, store=True)
                C = lnp._forbidden_sum(hist[1:] + [R_N], lnp.forbidden_mask(m, FORBID_FOCK_MIN), [0])
            else:
                R_N, C = lnp.propagate(m, u, R0, dt), 0.0
            return ft.fidelity_from_outputs(R_N, Tw, w), None, C, None
        if forbid:
            return lnp.fidelity_and_grad(m, u, R0, Tw, w, dt, forbid_mask=lnp.forbidden_mask(m, FORBID_FOCK_MIN),
                                         forbid_idx=(0,))
        F, g = lnp.fidelity_and_grad(m, u, R0, Tw, w, dt)
        return F, g, 0.0, None


class OpenAqecObjective:
    """cost(x), grad(x) on the pre-image x (N, 4) (module docstring). The three process inputs run as joblib jobs."""

    def __init__(self, N, n_c=N_C_TRAIN, gate_n_c=GATE_N_C, channels="decay", dt=DT, tau2=TAU2,
                 lambda_forbid=LAMBDA_FORBID, n_jobs=3, target="X", gate_pulse=None):
        self.N, self.n_c, self.channels, self.dt, self.tau2 = N, n_c, channels, dt, tau2
        self.lambda_forbid, self.n_jobs = lambda_forbid, n_jobs
        self.m = _model(n_c, channels)
        self.R0 = gate_input(n_c, gate_n_c, channels, gate_pulse)
        T, self.w = ft.tail_targets(n_c, 1, target, "traced")
        self.Tw = window_targets(self.m, T, tau2)
        self.to_physical, self.to_preimage_grad = make_constraint_chain(N, dt, eg.CAV_BAND, eg.TRA_BAND, eg.RAMP_NS)
        self._pool = None
        self._cache_x, self._cache = None, None
        self.n_evals = 0

    def _map(self, tasks):
        if self.n_jobs == 1:
            return [_piece(*t) for t in tasks]
        if self._pool is None:
            self._pool = Parallel(n_jobs=self.n_jobs, backend="loky")
            self._pool.__enter__()
        return self._pool(delayed(_piece)(*t) for t in tasks)

    def close(self):
        if self._pool is not None:
            self._pool.__exit__(None, None, None)
            self._pool = None

    def evaluate_physical(self, u, want_grad=True):
        """(F, dF/du, C_forbid, dC/du) of a physical pulse u (N, 4)."""
        u = np.asarray(u, float)
        forbid = self.lambda_forbid > 0 and self.n_c > FORBID_FOCK_MIN
        tasks = [(self.n_c, self.channels, self.R0[b:b + 1], self.Tw[b:b + 1], self.w[b:b + 1], u, self.dt,
                  forbid and b in lnp.FORBID_IDX, want_grad) for b in range(self.R0.shape[0])]
        out = self._map(tasks)
        F = sum(o[0] for o in out)
        C = float(sum(o[2] for o in out))
        if not want_grad:
            return F, None, C, None
        gF = sum(o[1] for o in out)
        gCs = [o[3] for o in out if o[3] is not None]
        return F, gF, C, (sum(gCs) if gCs else np.zeros_like(u))

    def evaluate(self, x):
        x = np.asarray(x, dtype=float).reshape(self.N, 4)
        if self._cache_x is not None and np.array_equal(x, self._cache_x):
            return self._cache
        u = self.to_physical(x)
        F, gF, C, gC = self.evaluate_physical(u)
        c4, g4 = amplitude_cost(u, eg.EPS_MAX)
        cost = 1.0 - F + eg.W_AMP * c4 + self.lambda_forbid * C
        g_u = -gF + eg.W_AMP * g4 + self.lambda_forbid * gC
        terms = {"F_pro": F, "c4": float(c4), "C_forbid": C, "forbid": self.lambda_forbid * C}
        self.n_evals += 1
        self._cache_x, self._cache = x.copy(), (float(cost), self.to_preimage_grad(g_u).ravel(), terms)
        return self._cache

    def cost(self, x):
        return self.evaluate(x)[0]

    def grad(self, x):
        return self.evaluate(x)[1]


def train(duration_us, init_x=None, seed=eg.SEED, maxiter=2000, n_c=N_C_TRAIN, channels="decay",
          lambda_forbid=LAMBDA_FORBID, n_jobs=3, verbose=True, print_every=25, checkpoint=None):
    """L-BFGS-B on the open cost from init_x (a pre-image of the same chain, e.g. a closed AQEC pulse's x_) or, with
    init_x None, d2's smooth seeded start. Keeps the best-seen F_pro iterate. Returns (u, x, info)."""
    N = ap.n_steps(duration_us)
    x = eg.start_preimage(seed, N) if init_x is None else np.asarray(init_x, dtype=float).reshape(N, 4)
    if np.abs(x).max() > eg.U_BOX * (1 + 1e-12):
        raise ValueError(f"start pre-image reaches {np.abs(x).max():.2f} > box {eg.U_BOX}")
    x = np.clip(x, -eg.U_BOX, eg.U_BOX)
    obj = OpenAqecObjective(N, n_c, channels=channels, lambda_forbid=lambda_forbid, n_jobs=n_jobs)
    t0 = time.time()
    count = [0]
    _, _, t_start = obj.evaluate(x)
    best = {"F": t_start["F_pro"], "x": x.copy(), "it": 0}
    history = [{"it": 0, **t_start, "minutes": 0.0}]

    def callback(xk):
        count[0] += 1
        cost, _, terms = obj.evaluate(xk)
        if terms["F_pro"] > best["F"]:
            best.update(F=terms["F_pro"], x=np.array(xk).reshape(N, 4), it=count[0])
        if print_every and count[0] % print_every == 0:
            history.append({"it": count[0], "cost": cost, **terms, "minutes": (time.time() - t0) / 60})
            if verbose:
                print(f"    it {count[0]:5d}  F {terms['F_pro']:.6f}  c4 {terms['c4']:.2e}  "
                      f"C_forbid {terms['C_forbid']:.3g}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
            if checkpoint is not None:
                checkpoint(obj.to_physical(best["x"]), best["x"], {"it": count[0], "best_it": best["it"],
                                                                   "best_F": best["F"], **terms})

    if verbose:
        print(f"=== open AQEC  T={duration_us} us  N={N}  n_c={n_c}  channels={channels}  "
              f"start {'init_x' if init_x is not None else f'seed {seed}'}  F {t_start['F_pro']:.6f}  "
              f"C_forbid {t_start['C_forbid']:.3g}", flush=True)
    try:
        res = minimize(obj.cost, x.ravel(), jac=obj.grad, method="L-BFGS-B",
                       bounds=[(-eg.U_BOX, eg.U_BOX)] * (N * 4), options={"maxiter": maxiter, **OPTIONS},
                       callback=callback)
        x_end = res.x.reshape(N, 4)
        F_end = obj.evaluate(x_end)[2]["F_pro"]
    finally:
        obj.close()
    x_use = x_end if F_end >= best["F"] else best["x"]
    u = obj.to_physical(x_use)
    info = {"duration_us": duration_us, "N": N, "dt": DT, "n_c": n_c, "gate_n_c": GATE_N_C, "channels": channels,
            "tau2": TAU2, "seed": seed if init_x is None else None, "lambda_forbid": lambda_forbid,
            "maxiter": maxiter, "nit": int(res.nit), "nfev": int(res.nfev), "message": str(res.message),
            "seconds": time.time() - t0, "F_start": t_start["F_pro"], "F_train": max(F_end, best["F"]),
            "best_it": best["it"] if F_end < best["F"] else int(res.nit), "history": history,
            "cav_band": list(eg.CAV_BAND), "tra_band": list(eg.TRA_BAND), "ramp_ns": eg.RAMP_NS,
            "u_box": eg.U_BOX, "eps_max": eg.EPS_MAX,
            "max_drive_modulus": {"cav": float(np.hypot(u[:, 0], u[:, 1]).max()),
                                  "tra": float(np.hypot(u[:, 2], u[:, 3]).max())},
            "max_abs_preimage": float(np.abs(x_use).max()),
            "frac_preimage_pinned": float(np.mean(np.abs(x_use) >= eg.U_BOX * (1 - 1e-9)))}
    if verbose:
        print(f"    end  F {info['F_train']:.6f}  {res.message}  nit={res.nit}  ({info['seconds'] / 60:.1f} min)",
              flush=True)
    return u, x_use, info


def open_score(u, n_c=24, channels="decay"):
    """F_pro of EsT gate + the step "as is" at n_c (gate at max(n_c, 24)), the forward check of a trained pulse."""
    m = aq.model(n_c, 1, channels)
    R = gate_input(n_c, GATE_N_C, channels)
    with np.errstate(**_QUIET):
        return fr.score_state(ap.open_step(m, R, u), n_c, 1, "X")["F_pro"]


def truncation_scan(u, trunc=(20, 24, 28)):
    return {n: {"F_open": open_score(u, n), **ap.block_fidelities(u, n)} for n in trunc}
