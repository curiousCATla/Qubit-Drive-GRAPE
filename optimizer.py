import os
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
from grape_core import make_hamiltonian, smooth_initial_controls, derivative_penalty, boundary_penalty, amplitude_penalty, fidelity_multi_state
from cat_code import validate_pulse_truncations


def optimize_multi_state_pulse(
    get_state_pairs,           # factory function: get_state_pairs(n_c, n_t) -> list of (psi_i, psi_f) tuples
    trunc_list=[20, 24, 28],
    n_t=3,
    N=550,
    dt=0.002,
    penalties=None,
    warm_start=None,
    save_path=None,
    n_jobs=3,
    maxiter=2000,
    verbose=True
):
    if penalties is None:
        penalties = {'deriv': 0.000008, 'boundary': 0.00004, 'amp': 0.00012, 'amp_max': 40.0}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimizing pulse | Truncations: {trunc_list}")
        print(f"{'='*60}")

    # Build Hamiltonians once (for training truncations + main for final eval)
    H0_list, Hc_list = [], []
    for nc in trunc_list:
        H0_k, Hc_k = make_hamiltonian(n_t, nc)
        H0_list.append(H0_k)
        Hc_list.append(Hc_k)

    H0_main, Hc_main = make_hamiltonian(n_t, max(trunc_list))

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
        u0 = smooth_initial_controls(N, amp=10.0, cutoff_frac=0.04, seed=42)

    x0 = u0.ravel()
    bounds = [(-penalties['amp_max'], penalties['amp_max'])] * (N * 4)

    # === Best bare-F tracking (mutable container for closure) ===
    best = {'u': None, 'F': -np.inf}

    # === Parallel evaluation (unchanged core logic) ===
    def evaluate_trunc(u, H0_k, Hc_k, nc):
        state_pairs_k = get_state_pairs(n_c=nc, n_t=n_t)   # Rebuild states for this exact nc
        psi_i_list = [p[0] for p in state_pairs_k]
        psi_f_list = [p[1] for p in state_pairs_k]
        return fidelity_multi_state(u, H0_k, Hc_k, psi_i_list, psi_f_list, dt, want_grad=True)

    def objective(x):
        u = x.reshape(N, 4)

        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_trunc)(u, H0_k, Hc_k, nc)
            for H0_k, Hc_k, nc in zip(H0_list, Hc_list, trunc_list)
        )

        total_F = sum(F for F, _ in results)
        total_grad = sum(g for _, g in results if g is not None)

        M = len(trunc_list)
        F_avg = total_F / M          # bare average fidelity across training truncations
        grad_avg = total_grad / M

        # Track best bare-F pulse seen so far
        if F_avg > best['F']:
            best['F'] = F_avg
            best['u'] = u.copy()

        cost = -F_avg
        g = -grad_avg

        # Penalties (added to cost and gradient)
        if penalties['deriv'] > 0:
            g_d, gr_d = derivative_penalty(u)
            cost += penalties['deriv'] * g_d
            g += penalties['deriv'] * gr_d
        if penalties['boundary'] > 0:
            g_b, gr_b = boundary_penalty(u)
            cost += penalties['boundary'] * g_b
            g += penalties['boundary'] * gr_b
        if penalties['amp'] > 0:
            g_a, gr_a = amplitude_penalty(u, amp_max=penalties['amp_max'])
            cost += penalties['amp'] * g_a
            g += penalties['amp'] * gr_a

        return cost, g.ravel()

    # Run optimization
    res = minimize(objective, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                   options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})

    u_final = res.x.reshape(N, 4)

    # Decide which pulse to return: final from minimizer or best bare-F seen during run
    if best['u'] is not None and best['F'] > 0.5:   # only consider if reasonably good
        # Re-evaluate bare F of final point on training set for fair comparison
        def _bare_F(u):
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_trunc)(u, H0_k, Hc_k, nc)
                for H0_k, Hc_k, nc in zip(H0_list, Hc_list, trunc_list)
            )
            return sum(F for F, _ in results) / len(trunc_list)

        F_final_eval = _bare_F(u_final)
        if best['F'] > F_final_eval:
            u_opt = best['u'].copy()
            if verbose:
                print(f"Using best-seen pulse (bare F = {best['F']:.6f}) instead of final L-BFGS point (F = {F_final_eval:.6f})")
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
        for nc, H0_k, Hc_k in zip(trunc_list, H0_list, Hc_list):
            state_pairs_k = get_state_pairs(n_c=nc, n_t=n_t)
            psi_i_list = [p[0] for p in state_pairs_k]
            psi_f_list = [p[1] for p in state_pairs_k]
            F_k, _ = fidelity_multi_state(u_opt, H0_k, Hc_k, psi_i_list, psi_f_list, dt, want_grad=False)
            print(f"  n_c={nc:2d}: F = {F_k:.6f}")
        print("="*60)

    # Final evaluation at max truncation (for backward compatibility / info dict)
    final_pairs = get_state_pairs(n_c=max(trunc_list), n_t=n_t)
    psi_i_main = [p[0] for p in final_pairs]
    psi_f_main = [p[1] for p in final_pairs]
    F_final, _ = fidelity_multi_state(u_opt, H0_main, Hc_main, psi_i_main, psi_f_main, dt, want_grad=False)

    if verbose:
        print(f"\nFinished: {res.message}")
        print(f"Final reported Fidelity (at max trunc): {F_final:.6f}")
        if best['F'] > 0:
            print(f"Best bare multi-trunc F seen during optimization: {best['F']:.6f}")

    info = {
        'message': res.message,
        'success': res.success,
        'iterations': res.nit,
        'final_fidelity': F_final,
        'best_bare_F_during_opt': best['F'],
        'trunc_list': trunc_list
    }

    return u_opt, info


# ============================================================
# REFINEMENT FUNCTION
# ============================================================

def refine_pulse(
    get_state_pairs,
    initial_pulse,
    trunc_list=[22, 24, 26],
    n_t=3,
    extra_maxiter=2000,
    penalties=None,
    penalty_scale=1.0,
    widen_training=False,
    save_path=None,
    verbose=True
):
    """
    Refine an already optimized pulse using warm start.

    This function improves fidelity, robustness across truncations,
    and/or smoothness of an existing pulse.

    Parameters
    ----------
    get_state_pairs : callable
        Factory function returning state pairs for a given n_c
        (e.g. get_logical_X_state_pairs, get_logical_H_state_pairs, ...)
    initial_pulse : np.ndarray
        Previously optimized pulse to warm-start from (shape: (N, 4))
    trunc_list : list of int
        Cavity truncations used during refinement (manual control).
    extra_maxiter : int
        Additional L-BFGS-B iterations to perform.
    penalties : dict or None
        Base penalty dictionary. If None, uses sensible defaults.
    penalty_scale : float or dict
        Scaling factor(s) applied to penalties.
        - float: scales all regularization penalties (deriv, boundary, amp)
        - dict: allows per-penalty scaling, e.g. {'deriv': 0.4, 'boundary': 0.8}
    widen_training : bool
        If True, prints a recommendation for a wider training set.
        Actual widening is left manual (as requested).
    save_path : str or None
        Path to save the refined pulse. If None, does not save.
    verbose : bool
        Print progress and results.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("REFINEMENT STARTED")
        print("=" * 70)
        print(f"Training truncations : {trunc_list}")
        print(f"Extra maxiter        : {extra_maxiter}")
        print(f"Penalty scale        : {penalty_scale}")

    # --- Prepare base penalties ---
    if penalties is None:
        penalties = {
            'deriv': 0.00001,
            'boundary': 0.00002,
            'amp': 0.00008,
            'amp_max': 40.0
        }

    penalties = penalties.copy()

    # --- Apply penalty_scale (supports float or dict) ---
    if isinstance(penalty_scale, dict):
        for key, scale in penalty_scale.items():
            if key in penalties and key != 'amp_max':
                penalties[key] *= scale
                if verbose:
                    print(f"  Scaled '{key}' by {scale} → {penalties[key]:.2e}")
    elif penalty_scale != 1.0:
        for key in ['deriv', 'boundary', 'amp']:
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

    u_refined, info = optimize_multi_state_pulse(
        get_state_pairs=get_state_pairs,
        trunc_list=trunc_list,
        n_t=n_t,
        warm_start=initial_pulse,
        penalties=penalties,
        maxiter=extra_maxiter,
        save_path=save_path,
        verbose=verbose
    )

    # --- Post-refinement validation ---
    if verbose:
        print("\n--- Post-Refinement Validation ---")

    validate_pulse_truncations(
        u=u_refined,
        get_targets_func=get_state_pairs,
        n_t=n_t,
        title="Refined Pulse - Full Truncation Validation"
    )

    if verbose:
        print("\n" + "=" * 70)
        print("REFINEMENT COMPLETE")
        print("=" * 70)
        if save_path:
            print(f"Saved refined pulse to: {save_path}")

    return u_refined, info
