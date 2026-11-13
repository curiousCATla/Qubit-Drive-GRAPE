"""
A real AQEC pulse on Shirol's chip: the Cornell step 1 (Roy et al., arXiv:2603.15356, App. C 2, Eqs. C9-C10) trained
by closed-system GRAPE for the odd kitten code, the way the Ord gate was trained (DGRAPE/est_gate.py, variant "ord").
Notebook: notebooks/AQEC_compare.ipynb §5.

THE TARGET. One unitary U of length T on storage x transmon (n_r = 1, the readout is not driven) with

    U |c, g> = D |c, g>            for the six code cardinals c          (a code word is left alone)
    U |c_E, g> = D |c, e>          for the six error cardinals c_E        (an error word is fixed, the transmon flagged)

c_E is the cardinal built on the error words (|0> + sqrt5 |4>)/sqrt6, |2> (= a|c> renormalised). D = exp(+i H_d tau2)
is the pre-twist: over the readout window tau2 = 1.2 us that follows, the idle drift exp(-i H_d tau2) undoes it, so
both branches end right (Kerr on the g branch, Kerr + chi_ge + chi'_q on the e branch). The 12 conditions are scored
as EST's C1 (EST.grape_eigh.fidelity_cost): 1 - mean |<target|U|input>|^2. That leaves each input's global phase free,
so the relative phase of the g and e blocks is free too (the readout and reset make the two blocks incoherent), while
the six cardinals per block pin the logical phases inside each block.

THE COST is EST.grape_eigh.cost_and_grad with weights (1, 0, 0, 1) = Ord's: C1 on the 12 conditions plus the drive cap
C4. Passing the 12 inputs as the "code" stack and the same stack as the unused "error" stack keeps the code path
unchanged (C2 and C3 have weight 0).

SETTINGS. d2's / the EsT gate's drive limits (DGRAPE/recipe.py: bands +-27 / +-33 MHz, 48 ns ramp, box 25 on the
pre-image, cap 25 rad/us ~ 4 MHz per drive, the paper's own cap), dt = 1 ns (user, 2026-11-12), one training truncation
n_c 20 (checked at 24 and 28 afterwards), d2's smooth seeded start, L-BFGS-B with EST's options.

Pulses go to ~/fwm_runs first, then pulses/dgrape/ (u_aqec_T10_s42.npy, ...), never pulses/ or pulses/est/.
"""

import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

from AQEC.odd_kitten import error_basis, logical_basis
from core.ramp import make_constraint_chain
from DGRAPE import est_gate as eg
from DGRAPE.model import _QUIET, E, G, N_Q, control_hamiltonians, drift_hamiltonian, operators
from EST.grape_eigh import cost_and_grad
from EST.kitten_code import _coeff_matrix

DT = 0.001                 # us (1 ns)
TAU2 = 1.2                 # us, the readout window the pre-twist is for
N_C_TRAIN = 20
SEED = eg.SEED
WEIGHTS = (1.0, 0.0, 0.0, eg.W_AMP)      # Ord's
CLOSED_F = 0.999           # EST's bar for "closed"


def n_steps(duration_us, dt=DT):
    return int(round(duration_us / dt))


def _embed(vc, q):
    vq = np.zeros(N_Q)
    vq[q] = 1.0
    return np.kron(vc, vq)


def system(n_c, tau2=TAU2, include_higher_order=True):
    """H0, Hc, A, the 12 inputs (d, 12) = [code cardinals with g | error cardinals with g] and the 12 targets
    D [code cardinals with g | code cardinals with e], D = exp(+i H_d tau2)."""
    H0 = drift_hamiltonian(n_c, 1, include_higher_order)
    Hc = control_hamiltonians(n_c, 1)
    A = operators(n_c, 1)["a"]
    coeff = _coeff_matrix()                                   # (2, 6): cardinals in the logical basis
    Bl, Be = logical_basis(n_c) @ coeff, error_basis(n_c) @ coeff
    code_g = np.stack([_embed(Bl[:, k], G) for k in range(6)], axis=1)
    err_g = np.stack([_embed(Be[:, k], G) for k in range(6)], axis=1)
    code_e = np.stack([_embed(Bl[:, k], E) for k in range(6)], axis=1)
    D = np.exp(1j * np.real(np.diag(H0)) * tau2)
    inputs = np.concatenate([code_g, err_g], axis=1).astype(complex)
    targets = D[:, None] * np.concatenate([code_g, code_e], axis=1)
    return SimpleNamespace(n_c=n_c, tau2=tau2, H0=H0, Hc=Hc, A=A, inputs=inputs, targets=targets)


def block_fidelities(u, n_c, dt=DT, tau2=TAU2):
    """Closed-system scores of a physical pulse u (N, 4): F (all 12), F_code (code block), F_err (error block)."""
    from EST.grape_eigh import propagate_trajectory
    s = system(n_c, tau2)
    with np.errstate(**_QUIET):
        traj = propagate_trajectory(np.asarray(u, float), s.H0, s.Hc, s.inputs, dt)
    v = np.abs(np.sum(np.conj(s.targets) * traj[-1], axis=0)) ** 2
    return {"F": float(v.mean()), "F_code": float(v[:6].mean()), "F_err": float(v[6:].mean())}


class AqecObjective:
    """cost(x), grad(x) on the pre-image x (N, 4); u = ramp(bandlimit(x)) (core.ramp.make_constraint_chain)."""

    def __init__(self, n_c=N_C_TRAIN, N=1000, dt=DT, tau2=TAU2, weights=WEIGHTS):
        self.N, self.dt, self.weights = N, dt, tuple(weights)
        self.sys = system(n_c, tau2)
        self.to_physical, self.to_preimage_grad = make_constraint_chain(N, dt, eg.CAV_BAND, eg.TRA_BAND, eg.RAMP_NS)
        self.n_evals = 0
        self._cache_x, self._cache = None, None

    def _terms(self, u, want_grad):
        s = self.sys
        with np.errstate(**_QUIET):
            cost, g, terms = cost_and_grad(u, s.H0, s.Hc, s.A, s.inputs, s.inputs, s.targets, self.dt, self.weights,
                                           eps_max=eg.EPS_MAX, want_grad=want_grad)
        return cost, g, {"F": 1.0 - terms["c1"], "c4": terms["c4"]}

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


def train(duration_us, seed=SEED, maxiter=4000, n_c=N_C_TRAIN, dt=DT, tau2=TAU2, init_x=None, verbose=True,
          print_every=100, checkpoint=None):
    """L-BFGS-B from d2's smooth start (or init_x). Returns (u, x, info). checkpoint(u, x, terms) is called every
    print_every iterations."""
    N = n_steps(duration_us, dt)
    x = eg.start_preimage(seed, N) if init_x is None else np.asarray(init_x, dtype=float).reshape(N, 4)
    if np.abs(x).max() > eg.U_BOX:
        raise ValueError(f"start pre-image reaches {np.abs(x).max():.2f} > box {eg.U_BOX}")
    obj = AqecObjective(n_c, N, dt, tau2)
    t0 = time.time()
    count = [0]
    best = {"F": -1.0}

    def callback(xk):
        count[0] += 1
        _, _, terms = obj.evaluate(xk)
        if terms["F"] > best["F"]:
            best.update(F=terms["F"], x=np.array(xk).reshape(N, 4), it=count[0])
        if print_every and count[0] % print_every == 0:
            if verbose:
                print(f"    it {count[0]:5d}  F {terms['F']:.6f}  c4 {terms['c4']:.2e}  "
                      f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
            if checkpoint is not None:
                checkpoint(obj.to_physical(best["x"]), best["x"], {"it": count[0], **terms})

    if verbose:
        print(f"=== AQEC pulse  T={duration_us} us  N={N}  dt={dt * 1e3:.0f} ns  n_c={n_c}  seed={seed}  "
              f"start F {obj._terms(obj.to_physical(x), False)[2]['F']:.6f}", flush=True)
    res = minimize(obj.cost, x.ravel(), jac=obj.grad, method="L-BFGS-B",
                   bounds=[(-eg.U_BOX, eg.U_BOX)] * (N * 4), options={"maxiter": maxiter, **eg.OPTIONS},
                   callback=callback)
    x_end = res.x.reshape(N, 4)
    F_end = obj._terms(obj.to_physical(x_end), False)[2]["F"]
    x_use = x_end if F_end >= best["F"] else best["x"]
    u = obj.to_physical(x_use)
    info = {"duration_us": duration_us, "N": N, "dt": dt, "n_c": n_c, "seed": seed, "tau2": tau2,
            "weights": list(WEIGHTS), "maxiter": maxiter, "nit": int(res.nit), "nfev": int(res.nfev),
            "message": str(res.message), "seconds": time.time() - t0, "F_train": max(F_end, best["F"]),
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


def truncation_scan(u, trunc=(20, 24, 28), dt=DT, tau2=TAU2):
    return {n: block_fidelities(u, n, dt, tau2) for n in trunc}


# ---------------------------------------------------------------------------
# Open-system scoring: the trained pulse inside the AQEC step (DGRAPE model, decay set unless stated)
# ---------------------------------------------------------------------------

def open_step(m, R, pulse, dt=DT, tau2=TAU2, mode="as is"):
    """One real AQEC step on the stack R (n_r = 1 model m): [mode "forgive": reset the gate's leftover transmon
    excitation first, as the ideal step does] -> the trained pulse (lossy, its pre-twist built in) -> tau2 lossy idle
    (the readout window) -> ideal reset. mode "as is" (default) feeds the gate's output to the pulse unchanged."""
    from DGRAPE import aqec_ideal as aq
    from DGRAPE import lindblad_np as lnp
    R0 = aq.forgive(R, m.n_c, m.n_r) if mode == "forgive" else R
    R1 = lnp.propagate(m, np.asarray(pulse, float), R0, dt)
    return aq.forgive(aq.idle(m, R1, tau2), m.n_c, m.n_r)


def single_rows(gate_pulse, pulses, n_c=24, channels="decay"):
    """N = 1 after the gate, per trained pulse {label: u}: the real step (as is, forgive), the real step in the closed
    system (coherent error only, loss-free gate too), the real step without the window (pulse + instant reset), and
    the ideal step with tau1 = the pulse length. Rows {label, duration_us, protocol, F_pro}."""
    from DGRAPE import aqec_ideal as aq
    from DGRAPE import fwm, fwm_repeat as fr
    m = aq.model(n_c, 1, channels)
    mc = fwm.model_with_channels(n_c, 1, frozenset(), True)
    K = aq.kraus(n_c, 1)
    E_in = fr.inputs(n_c, 1)
    Rg = fr.gate_step(m, gate_pulse, E_in)
    Rg_c = fr.gate_step(mc, gate_pulse, E_in)
    sc = lambda R, mm=m: fr.score_state(R, mm.n_c, 1, "X")["F_pro"]        # noqa: E731
    rows = []
    for label, u in pulses.items():
        T = u.shape[0] * DT
        from DGRAPE import lindblad_np as lnp
        add = lambda proto, F: rows.append({"label": label, "duration_us": T, "channels": channels,  # noqa: E731
                                            "protocol": proto, "F_pro": F})
        add("real AQEC (as is)", sc(open_step(m, Rg, u)))
        add("real AQEC (leftover transmon reset first)", sc(open_step(m, Rg, u, mode="forgive")))
        # no window: the pulse, then its pre-twist undone for free (exp(-i H_d tau2), what the window's drift would do
        # without losses), then the reset
        add("real AQEC, no readout window", sc(aq.forgive(aq.rotate(lnp.propagate(m, u, Rg, DT), aq.phase(m, -TAU2)),
                                                          n_c, 1)))
        add("real AQEC, closed system (gate and step)", sc(open_step(mc, Rg_c, u), mc))
        add("ideal AQEC, same length", sc(aq.aqec_step(m, Rg, K, "forgive", T, TAU2)))
        # same-time memory: wait the step's length (pulse + window), Kerr phase undone for free, no reset. A step
        # below this repairs nothing on balance.
        add("idle, same length, Kerr undone", sc(aq.memory_idle(m, Rg, T + TAU2)))
    return rows


def b1_rows(gate_pulse, pulses, N_max, n_c=24, log=None):
    """N gates, then one real AQEC step per trained pulse. Rows {N, label, F_pro, time_us}."""
    from DGRAPE import aqec_ideal as aq
    from DGRAPE import fwm_repeat as fr
    m = aq.model(n_c, 1)
    T_g = gate_pulse.shape[0] * fr.DT            # the gate runs on the 2 ns grid
    R, rows = fr.inputs(n_c, 1), []
    for N in range(1, N_max + 1):
        R = fr.gate_step(m, gate_pulse, R)
        for label, u in pulses.items():
            rows.append({"N": N, "label": label, "time_us": N * T_g + u.shape[0] * DT + TAU2,
                         "F_pro": fr.score_state(open_step(m, R, u), n_c, 1, fr.target_of(N))["F_pro"]})
        if log:
            log(f"B1 N={N}")
    return rows


def b2_rows(gate_pulse, pulses, N_max, n_c=24, log=None):
    """N (gate + real AQEC step) pairs, one chain per trained pulse. Rows {N, label, F_pro, time_us}."""
    from DGRAPE import aqec_ideal as aq
    from DGRAPE import fwm_repeat as fr
    m = aq.model(n_c, 1)
    T_g = gate_pulse.shape[0] * fr.DT
    labels = list(pulses)
    R = np.concatenate([fr.inputs(n_c, 1)] * len(labels))
    rows = []
    for N in range(1, N_max + 1):
        R = fr.gate_step(m, gate_pulse, R)
        for i, label in enumerate(labels):
            u = pulses[label]
            blk = slice(3 * i, 3 * i + 3)
            R[blk] = open_step(m, R[blk], u)
            rows.append({"N": N, "label": label, "time_us": N * (T_g + u.shape[0] * DT + TAU2),
                         "F_pro": fr.score_state(R[blk], n_c, 1, fr.target_of(N))["F_pro"]})
        if log:
            log(f"B2 N={N}")
    return rows


CHANNEL_SETS = (("closed", frozenset()), ("photon loss", frozenset({"a"})),
                ("transmon decay", frozenset({"s_ge", "s_ef"})), ("decay set", None))


def channel_rows(pulses, n_c=24):
    """The AQEC step alone, on the ideal code words (no gate), one decay channel at a time, real pulse vs the ideal
    step of the same length, scored against I; plus the pulse's time-averaged photon number and transmon excitation
    (closed trajectories, averaged over the 6 code and the 6 error cardinals) and the photon losses expected during it,
    kappa_a <n> T."""
    from AQEC import device as dev
    from DGRAPE import aqec_ideal as aq
    from DGRAPE import fwm, fwm_repeat as fr
    from EST.grape_eigh import propagate_trajectory
    s = system(n_c)
    ops = operators(n_c, 1)
    n = np.real(np.diag(ops["n_a"]))
    pexc = np.real(np.diag(ops["Pe"]) + np.diag(ops["Pf"]))
    E_in = fr.inputs(n_c, 1)
    K = aq.kraus(n_c, 1)
    rows = []
    for label, u in pulses.items():
        T = u.shape[0] * DT
        with np.errstate(**_QUIET):
            P = np.abs(propagate_trajectory(np.asarray(u, float), s.H0, s.Hc, s.inputs, DT)) ** 2
        nbar, pe = P.transpose(0, 2, 1) @ n, P.transpose(0, 2, 1) @ pexc          # (N+1, 12)
        stats = {"nbar_code": float(nbar[:, :6].mean()), "nbar_err": float(nbar[:, 6:].mean()),
                 "nbar_max": float(nbar.max()), "P_exc_code": float(pe[:, :6].mean()),
                 "P_exc_err": float(pe[:, 6:].mean()), "losses_in_pulse": float(dev.kappa_a * nbar[:, :6].mean() * T)}
        for name, keep in CHANNEL_SETS:
            m = fwm.model_with_channels(n_c, 1, keep, True)
            rows.append({"label": label, "duration_us": T, "channels": name,
                         "F_real": fr.score_state(open_step(m, E_in, u), n_c, 1, "I")["F_pro"],
                         "F_ideal": fr.score_state(aq.aqec_step(m, E_in, K, "forgive", T, TAU2), n_c, 1, "I")["F_pro"],
                         "F_idle": fr.score_state(aq.memory_idle(m, E_in, T + TAU2), n_c, 1, "I")["F_pro"],
                         **stats})
    return rows
