import os
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
from grape_core import make_hamiltonian, derivative_penalty, boundary_penalty, amplitude_penalty, fidelity_multi_state
from cat_code import validate_pulse_truncations
from fourier_cutoff import project_bandlimit
from optimizer import make_smooth_warm_start


def optimize_multi_state_pulse_max_trunc(
    get_state_pairs,           # factory function: get_state_pairs(n_c, n_t) -> list of (psi_i, psi_f) tuples
    trunc_list=[20, 24, 28],
    n_t=3,
    N=550,
    dt=0.002,
    penalties=None,
    refresh_every=10,
    warm_start=None,
    warm_start_amp=4.0,
    warm_start_cutoff_frac=0.04,
    warm_start_seed=42,
    save_path=None,
    n_jobs=3,
    maxiter=2000,
    cav_band=None,
    tra_band=None,
    hard_amp_limit=50.0,
    verbose=True
):
    """
    Max-truncation variant of optimize_multi_state_pulse.

    Unlike optimize_multi_state_pulse (which drives L-BFGS-B with the bare
    fidelity AVERAGED over trunc_list every objective call), this function's
    primary driving term is the bare fidelity at max(trunc_list) ONLY,
    evaluated with a fresh gradient on every call.

    Robustness against low truncations is instead enforced by a secondary
    "consistency" penalty, diff_cost = sum_k (F_max - F_k)^2 over the
    remaining (lower) truncations in trunc_list. This term requires
    propagating every lower-truncation Hamiltonian, so -- to keep the
    per-call cost down -- it is only RECOMPUTED every `refresh_every`
    L-BFGS-B iterations (via scipy's `callback`), not on every
    objective/gradient evaluation. Between refreshes its cost and gradient
    contributions are held frozen at the last-computed values (internally
    consistent for the line search, since a constant has zero slope).

    cav_band, tra_band, hard_amp_limit : see optimize_multi_state_pulse's
        docstring in optimizer.py -- identical semantics here.
    """
    if penalties is None:
        penalties = {'deriv': 0.00001, 'boundary': 0.00004, 'amp': 0.00012, 'amp_max': 40.0, 'disc': 0.5}
    else:
        penalties = penalties.copy()
        penalties.setdefault('disc', 0.5)

    bandlimit = cav_band is not None and tra_band is not None
    if (cav_band is None) != (tra_band is None):
        raise ValueError("cav_band and tra_band must both be given or both be None")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimizing pulse (max-trunc objective) | Truncations: {trunc_list}")
        print(f"Primary target: n_c={max(trunc_list)} | refresh_every={refresh_every} iters")
        print(f"{'='*60}")

    nc_max = max(trunc_list)
    lower_truncs = [nc for nc in trunc_list if nc != nc_max]

    # Build Hamiltonians once (main = the max-truncation target, lower = consistency term)
    H0_main, Hc_main = make_hamiltonian(n_t, nc_max)
    H0_lower, Hc_lower = [], []
    for nc in lower_truncs:
        H0_k, Hc_k = make_hamiltonian(n_t, nc)
        H0_lower.append(H0_k)
        Hc_lower.append(Hc_k)

    # Initial controls
    if isinstance(warm_start, str) and os.path.exists(warm_start):
        u0 = np.load(warm_start)
        if verbose: print(f"Loaded warm start from {warm_start}")
    elif isinstance(warm_start, np.ndarray):
        u0 = warm_start
    elif warm_start in ("zero", 0, "zeros"):
        u0 = np.zeros((N, 4))
        if verbose: print("Starting from zero controls (warm_start='zero')")
    else:
        u0 = make_smooth_warm_start(
            N, amp_max=warm_start_amp, cutoff_frac=warm_start_cutoff_frac, seed=warm_start_seed
        )
        if verbose:
            print(
                f"Starting from smooth random controls "
                f"(peak |u| = {np.max(np.abs(u0)):.4f} <= {warm_start_amp})"
            )

    x0 = u0.ravel()
    bounds = [(-hard_amp_limit, hard_amp_limit)] * (N * 4)

    # === Best bare-F_max tracking (mutable container for closure) ===
    best = {'u': None, 'F': -np.inf}

    # === Frozen consistency-term state, refreshed every `refresh_every` iterations ===
    diff_state = {'cost': 0.0, 'grad': np.zeros(N * 4)}
    iter_count = {'n': 0}

    def evaluate_trunc(u, H0_k, Hc_k, nc):
        state_pairs_k = get_state_pairs(n_c=nc, n_t=n_t)   # Rebuild states for this exact nc
        psi_i_list = [p[0] for p in state_pairs_k]
        psi_f_list = [p[1] for p in state_pairs_k]
        return fidelity_multi_state(u, H0_k, Hc_k, psi_i_list, psi_f_list, dt, want_grad=True)

    def project_to_x(g_u):
        return project_bandlimit(g_u, dt, cav_band, tra_band) if bandlimit else g_u

    def refresh_diff_term(u):
        """Expensive: propagates F_max and every lower truncation. Only called from callback()."""
        if not lower_truncs:
            diff_state['cost'] = 0.0
            diff_state['grad'] = np.zeros(N * 4)
            return

        F_max, grad_Fmax = evaluate_trunc(u, H0_main, Hc_main, nc_max)

        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_trunc)(u, H0_k, Hc_k, nc)
            for H0_k, Hc_k, nc in zip(H0_lower, Hc_lower, lower_truncs)
        )

        diff_cost = 0.0
        diff_grad_u = np.zeros_like(u)
        for F_k, grad_Fk in results:
            delta = F_max - F_k
            diff_cost += delta ** 2
            diff_grad_u += 2.0 * delta * (grad_Fmax - grad_Fk)

        diff_state['cost'] = penalties['disc'] * diff_cost
        diff_state['grad'] = penalties['disc'] * project_to_x(diff_grad_u).ravel()

    def objective(x):
        u_raw = x.reshape(N, 4)
        # Reparametrization (Heeres et al. 2017, Supp. Eq. 22): x is a free
        # pre-image; the physical pulse is its projection onto the
        # band-limited subspace. All fidelity/penalty terms below act on
        # the PHYSICAL pulse u, so the returned pulse is guaranteed
        # band-limited by construction.
        u = project_bandlimit(u_raw, dt, cav_band, tra_band) if bandlimit else u_raw

        F_max, grad_Fmax = evaluate_trunc(u, H0_main, Hc_main, nc_max)

        # Track best bare-F_max pulse seen so far
        if F_max > best['F']:
            best['F'] = F_max
            best['u'] = u.copy()

        cost = -F_max + diff_state['cost']
        g = -project_to_x(grad_Fmax).ravel() + diff_state['grad']

        # Penalties (added to cost and gradient)
        if penalties['deriv'] > 0:
            g_d, gr_d = derivative_penalty(u)
            cost += penalties['deriv'] * g_d
            g += penalties['deriv'] * project_to_x(gr_d).ravel()
        if penalties['boundary'] > 0:
            g_b, gr_b = boundary_penalty(u)
            cost += penalties['boundary'] * g_b
            g += penalties['boundary'] * project_to_x(gr_b).ravel()
        if penalties['amp'] > 0:
            g_a, gr_a = amplitude_penalty(u, amp_max=penalties['amp_max'])
            cost += penalties['amp'] * g_a
            g += penalties['amp'] * project_to_x(gr_a).ravel()

        return cost, g

    def callback(xk):
        iter_count['n'] += 1
        if iter_count['n'] == 1 or iter_count['n'] % refresh_every == 0:
            u_raw = xk.reshape(N, 4)
            u = project_bandlimit(u_raw, dt, cav_band, tra_band) if bandlimit else u_raw
            refresh_diff_term(u)
            if verbose and lower_truncs:
                print(f"  [iter {iter_count['n']:4d}] refreshed consistency term | "
                      f"diff_cost (unweighted) = {diff_state['cost'] / penalties['disc']:.3e}")

    # Run optimization
    res = minimize(objective, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                   callback=callback,
                   options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})

    # res.x is the raw pre-image; project to get the physical (band-limited) pulse.
    u_final = project_bandlimit(res.x.reshape(N, 4), dt, cav_band, tra_band) if bandlimit \
        else res.x.reshape(N, 4)

    # Decide which pulse to return: final from minimizer or best bare-F_max seen during run
    if best['u'] is not None and best['F'] > 0.5:   # only consider if reasonably good
        F_final_eval, _ = evaluate_trunc(u_final, H0_main, Hc_main, nc_max)
        if best['F'] > F_final_eval:
            u_opt = best['u'].copy()
            if verbose:
                print(f"Using best-seen pulse (bare F_max = {best['F']:.6f}) instead of final L-BFGS point (F_max = {F_final_eval:.6f})")
        else:
            u_opt = u_final
    else:
        u_opt = u_final

    if save_path:
        np.save(save_path, u_opt)
        if verbose: print(f"Saved optimized pulse to {save_path}")

    # === Post-optimization diagnostics: explicit per-training-truncation fidelities ===
    if verbose:
        print("\n" + "="*60)
        print("Post-optimization bare fidelity per training truncation")
        print("="*60)
        for nc in trunc_list:
            if nc == nc_max:
                H0_k, Hc_k = H0_main, Hc_main
            else:
                idx = lower_truncs.index(nc)
                H0_k, Hc_k = H0_lower[idx], Hc_lower[idx]
            state_pairs_k = get_state_pairs(n_c=nc, n_t=n_t)
            psi_i_list = [p[0] for p in state_pairs_k]
            psi_f_list = [p[1] for p in state_pairs_k]
            F_k, _ = fidelity_multi_state(u_opt, H0_k, Hc_k, psi_i_list, psi_f_list, dt, want_grad=False)
            marker = " (max, target)" if nc == nc_max else ""
            print(f"  n_c={nc:2d}: F = {F_k:.6f}{marker}")
        print("="*60)

    # Final evaluation at max truncation
    final_pairs = get_state_pairs(n_c=nc_max, n_t=n_t)
    psi_i_main = [p[0] for p in final_pairs]
    psi_f_main = [p[1] for p in final_pairs]
    F_final, _ = fidelity_multi_state(u_opt, H0_main, Hc_main, psi_i_main, psi_f_main, dt, want_grad=False)

    if verbose:
        print(f"\nFinished: {res.message}")
        print(f"Final reported Fidelity (at max trunc, n_c={nc_max}): {F_final:.6f}")
        if best['F'] > 0:
            print(f"Best bare F_max seen during optimization: {best['F']:.6f}")

    info = {
        'message': res.message,
        'success': res.success,
        'iterations': res.nit,
        'final_fidelity': F_final,
        'best_bare_F_during_opt': best['F'],
        'final_diff_cost': diff_state['cost'] / penalties['disc'] if penalties['disc'] > 0 else diff_state['cost'],
        'trunc_list': trunc_list
    }

    return u_opt, info


# ============================================================
# REFINEMENT FUNCTION
# ============================================================

def refine_pulse_max_trunc(
    get_state_pairs,
    initial_pulse,
    trunc_list=[22, 24, 26],
    n_t=3,
    extra_maxiter=2000,
    penalties=None,
    penalty_scale=1.0,
    refresh_every=10,
    widen_training=False,
    save_path=None,
    cav_band=None,
    tra_band=None,
    hard_amp_limit=40.0,
    verbose=True
):
    """
    Refine an already optimized pulse using warm start, driven by the
    max-truncation objective (see optimize_multi_state_pulse_max_trunc).

    Parameters mirror refine_pulse in optimizer.py; the only addition is
    `refresh_every`, forwarded to the max-trunc optimizer.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("REFINEMENT STARTED (max-trunc objective)")
        print("=" * 70)
        print(f"Training truncations : {trunc_list}")
        print(f"Extra maxiter        : {extra_maxiter}")
        print(f"Penalty scale        : {penalty_scale}")
        print(f"Refresh every         : {refresh_every} iterations")

    # --- Prepare base penalties ---
    if penalties is None:
        penalties = {
            'deriv': 0.00001,
            'boundary': 0.00002,
            'amp': 0.00008,
            'amp_max': 40.0,
            'disc': 0.5
        }

    penalties = penalties.copy()
    penalties.setdefault('disc', 0.5)

    # --- Apply penalty_scale (supports float or dict) ---
    if isinstance(penalty_scale, dict):
        for key, scale in penalty_scale.items():
            if key in penalties and key != 'amp_max':
                penalties[key] *= scale
                if verbose:
                    print(f"  Scaled '{key}' by {scale} → {penalties[key]:.2e}")
    elif penalty_scale != 1.0:
        for key in ['deriv', 'boundary', 'amp', 'disc']:
            if key in penalties:
                penalties[key] *= penalty_scale
        if verbose:
            print(f"  Applied uniform scale {penalty_scale} to regularization penalties")

    # --- Optional wider training advice (manual mode) ---
    if widen_training:
        recommended = sorted(set(trunc_list + [20, 28]))
        print(f"\n[INFO] widen_training=True")
        print(f"       Recommended wider set: {recommended}")
        print(f"       → You can re-run with trunc_list={recommended} if desired.\n")

    # --- Run optimization with warm start ---
    if verbose:
        print("\n--- Running refinement optimization ---\n")

    u_refined, info = optimize_multi_state_pulse_max_trunc(
        get_state_pairs=get_state_pairs,
        trunc_list=trunc_list,
        n_t=n_t,
        warm_start=initial_pulse,
        penalties=penalties,
        refresh_every=refresh_every,
        maxiter=extra_maxiter,
        save_path=save_path,
        cav_band=cav_band,
        tra_band=tra_band,
        hard_amp_limit=hard_amp_limit,
        verbose=verbose
    )

    # --- Post-refinement validation ---
    if verbose:
        print("\n--- Post-Refinement Validation ---")

    validate_pulse_truncations(
        u=u_refined,
        get_targets_func=get_state_pairs,
        n_t=n_t,
        title="Refined Pulse (max-trunc) - Full Truncation Validation"
    )

    if verbose:
        print("\n" + "=" * 70)
        print("REFINEMENT COMPLETE")
        print("=" * 70)
        if save_path:
            print(f"Saved refined pulse to: {save_path}")

    return u_refined, info
