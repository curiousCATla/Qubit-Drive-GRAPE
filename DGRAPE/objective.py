"""
Scipy-ready open-system GRAPE objective on the raw optimizer variable x (N, 4), with two
interchangeable fidelity-gradient backends:

    backend="np"   DGRAPE/lindblad_np.py  -- exact hand-derived adjoint (W form)
    backend="jax"  DGRAPE/lindblad_jax.py -- reverse-mode autodiff of the same polynomial

The cost mirrors core/optimizer.optimize_multi_state_pulse (experiments.ipynb §1, "Cost
function construction"), with the closed-system F_coh replaced by the Lindblad process
fidelity F_pro:

    cost(x) = 1 - (1/M) sum_i F_pro(u; n_c_i)                         (Heeres Eq. 23)
              + disc  * sum_{i != j} (F_i - F_j)^2                    (Heeres Eq. 24)
              + deriv * sum_k |u_k - u_{k-1}|^2
              + amp   * sum_k sum_d (|eps_d(k)| - amp_max)_+^2        (Supp. Eq. 19, 'modulus')
              + forbid * (1/M) sum_i C_i,  C_i = sum_{k=1}^{N} sum_{b=0_L,1_L} Tr[P_F rho_b(k)]
    u = R P(x)          P = band-limit projection (Supp. Eq. 22), R = Gaussian ramp
    dcost/dx = P(env * dcost/du)                                      (core.ramp chain)

C_i is Heeres' forbidden-state cost, P_F the projector on storage Fock >= forbid_fock_min
(DGRAPE/recipe.py), with its exact gradient from the same backward pass as F_i
(lindblad_np.fidelity_and_grad). On a truncation with n_c <= forbid_fock_min, P_F = 0 and C_i is
exactly 0, so it is skipped there (train.py refuses such a training list).

Every default comes from DGRAPE/recipe.py (the experiments.ipynb / main.py production
recipe; n_c=None means its trunc_list [12, 14, 16], which is NOT the Heeres list). Pass None to
disable a band pair or the ramp (ramp_ns=0 also disables), and `penalties={...}` to override
weights. Only the fidelity and C_i are backend-specific: the penalties, Eq. 24 and the chain
rule act on the physical-pulse gradient in numpy for both backends.

PARALLEL TRUNCATIONS. With n_jobs > 1 the per-truncation (F_i, C_i, gradients) run as joblib
jobs, one pool kept for the objective's lifetime (as core/optimizer.py). With the np backend
n_jobs may exceed the number of truncations (up to 3 per truncation): the largest truncations
are then split by process input (`_split_plan`), since one job is single-threaded (Accelerate
does not thread d ~ 100 products). Backend 'loky'
(processes) for "np", whose Taylor loop is Python-level and holds the GIL; "jax" always uses
'threading' (jitted functions do not pickle; XLA releases the GIL). n_jobs=1 is a plain loop.
Results do not depend on n_jobs (tested).
"""

import numpy as np
from joblib import Parallel, delayed

from core.optimizer import _AMP_PENALTIES
from core.grape_core import derivative_penalty
from core.ramp import make_constraint_chain
from DGRAPE import lindblad_np as lnp
from DGRAPE.code import average_gate_fidelity, process_targets
from DGRAPE.model import Model
from DGRAPE.recipe import RECIPE, penalties_with_defaults

_RECIPE = object()      # sentinel: "use DGRAPE/recipe.py", distinct from None = "disabled"


def _pick(value, key):
    return RECIPE[key] if value is _RECIPE else value


def _eval_trunc(model, E, T, w, u, dt, want_grad, mask, forbid_idx=lnp.FORBID_IDX):
    """(F, dF/du, C, dC/du) of one truncation -- or of a subset of its process inputs, whose
    pieces simply add up -- numpy backend; module level so loky can pickle it. Gradients are
    None without want_grad, C (and dC/du) 0 / None without a mask or forbid_idx."""
    if mask is not None and not forbid_idx:
        mask = None
    spec = lnp.spec_for_pulse(model, u, dt)
    if want_grad:
        if mask is None:
            F, g = lnp.fidelity_and_grad(model, u, E, T, w, dt, spec)
            return F, g, 0.0, None
        return lnp.fidelity_and_grad(model, u, E, T, w, dt, spec, forbid_mask=mask,
                                     forbid_idx=forbid_idx)
    if mask is None:
        return lnp.fidelity(model, u, E, T, w, dt, spec), None, 0.0, None
    R_N, hist = lnp.propagate(model, u, E, dt, spec, store=True)
    F = float(np.sum(w * np.einsum("bij,bij->b", T.conj(), R_N).real) / 4.0)
    return F, None, lnp._forbidden_sum(hist[1:] + [R_N], mask, forbid_idx), None


def _sum_pieces(pieces):
    """Add the (F, gF, C, gC) of one truncation's input subsets."""
    def add(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals[1:], vals[0].copy()) if vals else None
    return (sum(p[0] for p in pieces), add([p[1] for p in pieces]),
            sum(p[2] for p in pieces), add([p[3] for p in pieces]))


class OpenGrapeObjective:
    """Callable pair `cost(x)`, `grad(x)` sharing one evaluation per x.

    Also tracks the best mean F_pro seen over every evaluation, as an (x, u, F) pair,
    for train.py's best-vs-final rule (core.optimizer does the same).
    """

    def __init__(self, gate, N, dt, n_c=None, backend="np", n_r=2, cav_band=_RECIPE,
                 tra_band=_RECIPE, ramp_ns=_RECIPE, penalties=None, amp_norm=_RECIPE,
                 decay=True, dephasing=False, heating=False, include_higher_order=False,
                 rate_scale=1.0, forbid_fock_min=_RECIPE, n_jobs=_RECIPE,
                 parallel_backend="loky"):
        if backend not in ("np", "jax"):
            raise ValueError(f"backend must be 'np' or 'jax', not {backend!r}")
        self.gate, self.N, self.dt, self.backend = gate, N, dt, backend
        if n_c is None:
            n_c = RECIPE["trunc_list"]
        self.trunc = [n_c] if np.isscalar(n_c) else list(n_c)
        self.cav_band = _pick(cav_band, "cav_band")
        self.tra_band = _pick(tra_band, "tra_band")
        self.ramp_ns = _pick(ramp_ns, "ramp_ns") or None
        self.penalties = penalties_with_defaults(penalties)
        self.amp_norm = _pick(amp_norm, "amp_norm")
        if self.amp_norm not in _AMP_PENALTIES:
            raise ValueError(f"amp_norm={self.amp_norm!r}; expected one of {sorted(_AMP_PENALTIES)}")
        self._amp_fn = _AMP_PENALTIES[self.amp_norm]
        self.forbid_fock_min = _pick(forbid_fock_min, "forbid_fock_min")
        # np: up to one job per (truncation, process input); jax: one per truncation.
        n_inputs = 1 if backend == "jax" else 3
        self.n_jobs = max(1, min(int(_pick(n_jobs, "n_jobs")), n_inputs * len(self.trunc)))
        self.parallel_backend = "threading" if backend == "jax" else parallel_backend
        self._pool = None

        self.models, self.targets = [], []
        for nc in self.trunc:
            self.models.append(Model(nc, n_r, include_higher_order, decay, dephasing,
                                     heating, rate_scale))
            self.targets.append(process_targets(gate, nc, n_r))
        # P_F per truncation; None where it is empty (n_c <= fock_min) or the term is off.
        self.masks = [lnp.forbidden_mask(m, self.forbid_fock_min)
                      if self.penalties["forbid"] > 0 and m.n_c > self.forbid_fock_min else None
                      for m in self.models]
        self.to_physical, self.to_preimage_grad = make_constraint_chain(
            N, dt, self.cav_band, self.tra_band, self.ramp_ns)
        self.split = self._split_plan()
        if backend == "jax":
            from DGRAPE import lindblad_jax as ljx
            self._jax = [ljx.make_fidelity_fns(m, *tg, dt, forbid_mask=mk)
                         for m, tg, mk in zip(self.models, self.targets, self.masks)]
        self._cache_x = None
        self._cache = None
        self.n_evals = 0
        self.best = {"F": -np.inf, "x": None, "u": None}

    def config(self):
        """The settings this objective actually applies (for info dicts / notebooks)."""
        return {"trunc_list": self.trunc, "cav_band": self.cav_band, "tra_band": self.tra_band,
                "ramp_ns": self.ramp_ns, "penalties": dict(self.penalties),
                "amp_norm": self.amp_norm, "backend": self.backend,
                "forbid_fock_min": self.forbid_fock_min, "n_jobs": self.n_jobs,
                "c_ops": [name for name, _ in self.models[0].c_ops]}

    def _split_plan(self):
        """Truncations whose three process inputs run as separate jobs (np backend only).

        A truncation's inputs E_00, E_11, E_01 never interact, so its F, C and gradients are
        sums over inputs. When n_jobs exceeds the number of truncations, the LARGEST n_c are
        split first (each split turns one job into three) until there are >= n_jobs jobs:
        for [12, 14, 16], n_jobs 3 -> none, 5 -> {16}, 7 -> {14, 16}, 9 -> all.
        """
        jobs, split = len(self.trunc), set()
        for i in sorted(range(len(self.trunc)), key=lambda i: -self.trunc[i]):
            if jobs >= self.n_jobs:
                break
            split.add(i)
            jobs += 2
        return split

    # -- core evaluation ---------------------------------------------------
    def _map(self, fn, args):
        if self.n_jobs == 1:
            return [fn(*a) for a in args]
        if self._pool is None:
            self._pool = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)
            self._pool.__enter__()
        return self._pool(delayed(fn)(*a) for a in args)

    def close(self):
        """Shut the joblib pool down (it is also released when the object is collected)."""
        if self._pool is not None:
            self._pool.__exit__(None, None, None)
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def evaluate_truncations(self, u, want_grad=True):
        """Per-truncation [(F_i, dF_i/du, C_i, dC_i/du)] of the PHYSICAL pulse u, run as
        n_jobs parallel jobs. C_i = 0 (gradient None) where the forbidden term is off."""
        u = np.asarray(u, dtype=float)
        if self.backend == "jax" and want_grad:
            def run(i):
                spec = lnp.spec_for_pulse(self.models[i], u, self.dt)
                out = self._jax[i][1](u, *spec)
                if self.masks[i] is None:
                    return float(out[0]), np.asarray(out[1]), 0.0, None
                return (float(out[0]), np.asarray(out[1]), float(out[2]), np.asarray(out[3]))
            return self._map(run, [(i,) for i in range(len(self.models))])
        tasks, owner = [], []
        for i, (m, (E, T, w), mk) in enumerate(zip(self.models, self.targets, self.masks)):
            if i in self.split:
                for b in range(E.shape[0]):
                    fi = (0,) if b in lnp.FORBID_IDX else ()
                    tasks.append((m, E[b:b + 1], T[b:b + 1], w[b:b + 1], u, self.dt,
                                  want_grad, mk, fi))
                    owner.append(i)
            else:
                tasks.append((m, E, T, w, u, self.dt, want_grad, mk))
                owner.append(i)
        out = self._map(_eval_trunc, tasks)
        return [_sum_pieces([r for r, o in zip(out, owner) if o == i])
                for i in range(len(self.models))]

    def fidelities(self, u, want_grad=True):
        """Per-truncation F_pro (and dF/du) of the PHYSICAL pulse u."""
        res = self.evaluate_truncations(u, want_grad)
        Fs = [r[0] for r in res]
        return (Fs, [r[1] for r in res]) if want_grad else Fs

    def evaluate(self, x):
        x = np.asarray(x, dtype=float).reshape(self.N, 4)
        if self._cache_x is not None and np.array_equal(x, self._cache_x):
            return self._cache
        u = self.to_physical(x)
        res = self.evaluate_truncations(u)
        Fs, gs = [r[0] for r in res], [r[1] for r in res]
        M = len(Fs)
        F_mean = sum(Fs) / M
        if F_mean > self.best["F"]:
            self.best = {"F": F_mean, "x": x.copy(), "u": u.copy()}

        cost = 1.0 - F_mean
        g_u = -sum(gs) / M
        terms = {"infid": cost}
        pen = self.penalties
        if pen["disc"] > 0 and M > 1:
            # Eq. 24 over ordered pairs, as core/optimizer.py; reuses the Eq. 23 (F, g).
            d_cost, d_grad = 0.0, np.zeros_like(u)
            for i in range(M):
                for j in range(M):
                    if i != j:
                        delta = Fs[i] - Fs[j]
                        d_cost += delta ** 2
                        d_grad += 2.0 * delta * (gs[i] - gs[j])
            cost += pen["disc"] * d_cost
            g_u = g_u + pen["disc"] * d_grad
            terms["disc"] = pen["disc"] * d_cost
        if pen["deriv"] > 0:
            p, gp = derivative_penalty(u)
            cost += pen["deriv"] * p
            g_u = g_u + pen["deriv"] * gp
            terms["deriv"] = pen["deriv"] * p
        if pen["amp"] > 0:
            p, gp = self._amp_fn(u, amp_max=pen["amp_max"])
            cost += pen["amp"] * p
            g_u = g_u + pen["amp"] * gp
            terms["amp"] = pen["amp"] * p
        if pen["forbid"] > 0 and any(mk is not None for mk in self.masks):
            # Mean over ALL M truncations (C_i = 0 exactly where P_F is empty), as Eq. 23.
            p = sum(r[2] for r in res) / M
            cost += pen["forbid"] * p
            g_u = g_u + (pen["forbid"] / M) * sum(r[3] for r in res if r[3] is not None)
            terms["forbid"] = pen["forbid"] * p
            terms["C_forbid"] = [r[2] for r in res]
        g_x = self.to_preimage_grad(g_u)
        self.n_evals += 1
        self._cache_x = x.copy()
        self._cache = (float(cost), g_x.ravel(), {"F_pro": Fs, **terms})
        return self._cache

    def cost(self, x):
        return self.evaluate(x)[0]

    def grad(self, x):
        return self.evaluate(x)[1]

    # -- reporting -----------------------------------------------------------
    def report(self, u):
        """Per-truncation F_pro / F_avg of a PHYSICAL pulse (no gradient), plus the forbidden
        cost C (0 where the term is off or n_c <= forbid_fock_min)."""
        res = self.evaluate_truncations(np.asarray(u, dtype=float), want_grad=False)
        return [{"n_c": nc, "F_pro": F, "F_avg": average_gate_fidelity(F), "C_forbid": C}
                for nc, (F, _, C, _) in zip(self.trunc, res)]


def build_objective(gate, N, dt, n_c, backend="np", **kw):
    """(cost, grad, objective) for scipy.optimize.minimize(jac=grad)."""
    obj = OpenGrapeObjective(gate, N, dt, n_c, backend, **kw)
    return obj.cost, obj.grad, obj
