import os
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
from core.grape_core import make_hamiltonian, smooth_initial_controls, derivative_penalty, boundary_penalty, amplitude_penalty, fidelity_multi_state, refine_dt
from core.cat_code import validate_pulse_truncations
from core.fourier_cutoff import project_bandlimit


def make_smooth_warm_start(N, amp_max=4.0, cutoff_frac=0.04, seed=42):
    """Low-pass random controls with peak amplitude capped at amp_max (rad/μs)."""
    u0 = smooth_initial_controls(N, amp=amp_max, cutoff_frac=cutoff_frac, seed=seed)
    peak = np.max(np.abs(u0))
    if peak > amp_max:
        u0 *= amp_max / peak
    return u0


def optimize_multi_state_pulse(
    get_state_pairs,           # factory function: get_state_pairs(n_c, n_t) -> list of (psi_i, psi_f) tuples
    trunc_list=[20, 24, 28],
    n_t=3,
    N=550,
    dt=0.002,
    penalties=None,
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
    parallel_backend='loky',
    fidelity_fn=fidelity_multi_state,
    verbose=True
):
    """
    fidelity_fn : callable(u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad) -> (F, grad)
        Per-truncation fidelity metric used inside evaluate_trunc, before
        the cross-truncation averaging (Eq. 23) / discrepancy penalty
        (Eq. 24) logic below -- which is agnostic to this choice. Defaults
        to fidelity_multi_state (per-state average, blind to relative
        phase between state pairs). Pass coherent_fidelity_multi_state
        instead when training a gate whose correctness depends on the
        relative phase between its state pairs (e.g. T, H) -- see
        grape_core.coherent_fidelity_multi_state's docstring.
    cav_band, tra_band : (f_lo, f_hi) tuples in MHz, or None
        Hard frequency cutoffs on the cavity (eps_C = C_I + i*C_Q) and
        transmon (eps_T = T_I + i*T_Q) drives, mirroring Heeres et al. 2017
        Supplementary Eq. (22). When both are given, the raw L-BFGS-B
        variable x is treated as a pre-image and the PHYSICAL pulse used
        for fidelity, penalties, and the returned pulse is the orthogonal
        projection u = P(x) onto the band-limited subspace (see
        fourier_cutoff.project_bandlimit). Leave both None to disable
        (identical to previous behavior).
    hard_amp_limit : float
        L-BFGS-B box constraint on the raw variable x, in rad/us. This is
        the true hard amplitude bound, decoupled from penalties['amp_max']
        (which only sets the *soft* quadratic amplitude_penalty threshold).
        Note: when cav_band/tra_band are set, this bounds the raw
        pre-image x, not the projected physical pulse u -- band-edge
        ringing from project_bandlimit can push u's peak slightly outside
        [-hard_amp_limit, hard_amp_limit] even though x stays within bounds.
    parallel_backend : str
        joblib backend for the per-truncation Parallel evaluation ('loky'
        = separate processes, 'threading' = shared-memory threads). One
        Parallel pool is opened for the whole call (L-BFGS-B loop +
        best-vs-final re-evaluation + diagnostics) instead of being
        recreated on every objective() call. Benchmarked on this repo's
        workloads: 'loky' vs 'threading' and pool-reuse vs per-call
        recreation were all statistically indistinguishable here (joblib's
        loky backend already caches its executor globally across calls
        with matching parameters) -- kept configurable in case that stops
        holding on a different joblib version/machine.
    """
    if penalties is None:
        penalties = {'deriv': 0.00001, 'boundary': 0.00004, 'amp': 0.00012, 'amp_max': 40.0}
    else:
        penalties = penalties.copy()
    penalties.setdefault('disc', 0.0)

    bandlimit = cav_band is not None and tra_band is not None
    if (cav_band is None) != (tra_band is None):
        raise ValueError("cav_band and tra_band must both be given or both be None")

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

    # === Best bare-F tracking (mutable container for closure) ===
    best = {'u': None, 'F': -np.inf}

    # === Parallel evaluation (unchanged core logic) ===
    def evaluate_trunc(u, H0_k, Hc_k, nc):
        state_pairs_k = get_state_pairs(n_c=nc, n_t=n_t)   # Rebuild states for this exact nc
        psi_i_list = [p[0] for p in state_pairs_k]
        psi_f_list = [p[1] for p in state_pairs_k]
        return fidelity_fn(u, H0_k, Hc_k, psi_i_list, psi_f_list, dt, want_grad=True)

    # One Parallel pool for the entire call (L-BFGS-B loop + best-vs-final
    # re-evaluation + diagnostics below) instead of a fresh Parallel(...)
    # object on every objective() evaluation.
    with Parallel(n_jobs=n_jobs, backend=parallel_backend) as parallel:

        def objective(x):
            u_raw = x.reshape(N, 4)
            # Reparametrization (Heeres et al. 2017, Supp. Eq. 22): x is a free
            # pre-image; the physical pulse is its projection onto the
            # band-limited subspace. All fidelity/penalty terms below act on
            # the PHYSICAL pulse u, so the returned pulse is guaranteed
            # band-limited by construction.
            u = project_bandlimit(u_raw, dt, cav_band, tra_band) if bandlimit else u_raw

            results = parallel(
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

            # Heeres et al. 2017 Supp. Eq. 24: discrepancy penalty enforcing the
            # per-truncation fidelities agree with each other, not just each be
            # individually high. Reuses the (F_k, grad_k) pairs already computed
            # above for the Eq. 23 sum -- no extra propagation needed. Evaluated
            # fresh (no staleness) on every objective call, exactly like F_avg.
            if penalties['disc'] > 0:
                Fs = [F for F, _ in results]
                grads = [g_ for _, g_ in results]
                disc_cost = 0.0
                disc_grad = np.zeros_like(u)
                for i in range(M):
                    for j in range(M):
                        if i != j:
                            delta = Fs[i] - Fs[j]
                            disc_cost += delta ** 2
                            disc_grad += 2.0 * delta * (grads[i] - grads[j])
                cost += penalties['disc'] * disc_cost
                g += penalties['disc'] * disc_grad

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

            # Chain rule for the reparametrization: dCost/dx = P(dCost/du).
            # Valid because P is self-adjoint & idempotent (see fourier_cutoff.py).
            if bandlimit:
                g = project_bandlimit(g, dt, cav_band, tra_band)

            return cost, g.ravel()

        # Run optimization
        res = minimize(objective, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})

        # res.x is the raw pre-image; project to get the physical (band-limited) pulse.
        u_final = project_bandlimit(res.x.reshape(N, 4), dt, cav_band, tra_band) if bandlimit \
            else res.x.reshape(N, 4)

        # Decide which pulse to return: final from minimizer or best bare-F seen during run
        if best['u'] is not None and best['F'] > 0.5:   # only consider if reasonably good
            # Re-evaluate bare F of final point on training set for fair comparison
            def _bare_F(u):
                results = parallel(
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
        F_per_trunc = []
        for nc, H0_k, Hc_k in zip(trunc_list, H0_list, Hc_list):
            state_pairs_k = get_state_pairs(n_c=nc, n_t=n_t)
            psi_i_list = [p[0] for p in state_pairs_k]
            psi_f_list = [p[1] for p in state_pairs_k]
            F_k, _ = fidelity_multi_state(u_opt, H0_k, Hc_k, psi_i_list, psi_f_list, dt, want_grad=False)
            F_per_trunc.append(F_k)
            print(f"  n_c={nc:2d}: F = {F_k:.6f}")
        if len(F_per_trunc) > 1:
            max_disc = max(F_per_trunc) - min(F_per_trunc)
            print(f"  max pairwise |F_i - F_j| across training truncations: {max_disc:.3e}")
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
        'trunc_list': trunc_list,
        'disc_penalty_weight': penalties.get('disc', 0.0)
    }

    return u_opt, info


# ============================================================
# REFINEMENT FUNCTION
# ============================================================

def refine_pulse(
    get_state_pairs,
    initial_pulse,
    trunc_list=[20, 24, 28],
    n_t=3,
    extra_maxiter=2000,
    penalties=None,
    penalty_scale=1.0,
    widen_training=False,
    save_path=None,
    cav_band=None,
    tra_band=None,
    hard_amp_limit=40.0,
    parallel_backend='loky',
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
    cav_band, tra_band : (f_lo, f_hi) tuples in MHz, or None
        Forwarded to optimize_multi_state_pulse; hard frequency cutoff on
        the cavity/transmon drives (see its docstring).
    hard_amp_limit : float
        Forwarded to optimize_multi_state_pulse; true hard L-BFGS-B bound
        on the raw variable, decoupled from penalties['amp_max'] (which
        only sets the soft amplitude_penalty threshold).
    parallel_backend : str
        Forwarded to optimize_multi_state_pulse (joblib backend for the
        per-truncation Parallel evaluation).
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
        recommended = sorted(set(trunc_list + [18, 30]))
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
        cav_band=cav_band,
        tra_band=tra_band,
        hard_amp_limit=hard_amp_limit,
        parallel_backend=parallel_backend,
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


# ============================================================
# DT REFINEMENT FUNCTION
# ============================================================

def refine_pulse_dt(
    get_state_pairs,
    initial_pulse,
    s,
    dt=0.002,
    trunc_list=[20, 24, 28],
    n_t=3,
    extra_maxiter=2000,
    penalties=None,
    penalty_scale=1.0,
    widen_training=False,
    save_path=None,
    cav_band=None,
    tra_band=None,
    hard_amp_limit=40.0,
    parallel_backend='loky',
    verbose=True
):
    """
    Refine an already-optimized pulse onto a finer time grid, then
    re-optimize with warm start (mirrors refine_pulse; see its docstring
    for the shared parameters).

    Additional parameters
    ----------------------
    s : int
        Integer factor to shrink dt by. initial_pulse is upsampled via
        grape_core.refine_dt (zero-order hold: each row repeated s times),
        giving an (s*N, 4) warm start over the SAME total duration
        (N*dt == s*N*(dt/s)).
    dt : float
        The ORIGINAL step size (in us) of initial_pulse. new_dt = dt / s
        is computed here and passed through to optimize_multi_state_pulse
        as both N and dt overrides -- required because optimize_multi_state_pulse
        sizes its L-BFGS-B bounds from N, which must match the upsampled
        warm start's row count or scipy raises a shape mismatch.

    Penalty note: derivative_penalty's raw sum scales ~dt (so it shrinks by
    ~s at the finer grid) while amplitude_penalty's raw sum scales ~1/dt (so
    it grows by ~s); boundary_penalty is unaffected. This function does NOT
    rescale lambda_deriv/lambda_amp automatically -- penalties are passed
    through unchanged, exactly as given.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DT REFINEMENT STARTED")
        print("=" * 70)
        print(f"Original: N={initial_pulse.shape[0]}, dt={dt}")

    u0 = refine_dt(initial_pulse, s)
    new_dt = dt / s
    N_new = u0.shape[0]

    if verbose:
        print(f"Refined:  N={N_new}, dt={new_dt} (duration unchanged: "
              f"{initial_pulse.shape[0]*dt:.4f} us)")
        print(f"Training truncations : {trunc_list}")
        print(f"Extra maxiter        : {extra_maxiter}")
        print(f"Penalty scale        : {penalty_scale}")

    # --- Prepare base penalties (same defaults/scaling logic as refine_pulse) ---
    if penalties is None:
        penalties = {
            'deriv': 0.00001,
            'boundary': 0.00002,
            'amp': 0.00008,
            'amp_max': 40.0
        }

    penalties = penalties.copy()

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

    if widen_training:
        recommended = sorted(set(trunc_list + [20, 28]))
        print(f"\n[INFO] widen_training=True")
        print(f"       Recommended wider set: {recommended}")
        print(f"       → You can re-run with trunc_list={recommended} if desired.\n")

    if verbose:
        print("\n--- Running refinement optimization on finer grid ---\n")

    u_refined, info = optimize_multi_state_pulse(
        get_state_pairs=get_state_pairs,
        trunc_list=trunc_list,
        n_t=n_t,
        N=N_new,
        dt=new_dt,
        warm_start=u0,
        penalties=penalties,
        maxiter=extra_maxiter,
        save_path=save_path,
        cav_band=cav_band,
        tra_band=tra_band,
        hard_amp_limit=hard_amp_limit,
        parallel_backend=parallel_backend,
        verbose=verbose
    )

    if verbose:
        print("\n--- Post-Refinement Validation ---")

    validate_pulse_truncations(
        u=u_refined,
        get_targets_func=get_state_pairs,
        n_t=n_t,
        dt=new_dt,
        title="Refined Pulse (finer dt) - Full Truncation Validation"
    )

    info['dt'] = new_dt

    if verbose:
        print("\n" + "=" * 70)
        print("DT REFINEMENT COMPLETE")
        print("=" * 70)
        if save_path:
            print(f"Saved refined pulse to: {save_path}")

    return u_refined, info


# ============================================================
# LIGHTWEIGHT DT REFINEMENT (single-truncation training)
# ============================================================

def refine_pulse_dt_light(
    get_state_pairs,
    initial_pulse,
    s,
    dt=0.002,
    n_c=24,
    n_t=3,
    extra_maxiter=2000,
    penalties=None,
    penalty_scale=1.0,
    save_path=None,
    cav_band=None,
    tra_band=None,
    hard_amp_limit=40.0,
    validation_trunc_range=range(18, 31, 2),
    verbose=True
):
    """
    Lightweight dt refinement for a pulse whose fidelity already converges
    as n_c increases (e.g. the output of refine_pulse_dt / refine_pulse
    with a multi-truncation trunc_list). Upsamples the time grid exactly
    like refine_pulse_dt, but trains on a SINGLE fixed truncation n_c
    instead of a trunc_list -- no joblib Parallel pool, no per-iteration
    state-pair rebuild, no Eq. 24 discrepancy penalty, since the pulse is
    assumed to already be truncation-robust going in.

    Cross-truncation drift is instead only *checked*, not trained against:
    a sanity sweep over validation_trunc_range runs once at roughly the
    optimization's halfway point (via a scipy minimize callback, so the
    single L-BFGS-B run is never interrupted/restarted) and once more at
    the end.

    Parameters
    ----------
    get_state_pairs : callable
        Factory function returning state pairs for a given n_c (see
        refine_pulse_dt).
    initial_pulse : np.ndarray
        Previously optimized pulse to warm-start from (shape: (N, 4)).
    s : int
        Integer factor to shrink dt by (see refine_pulse_dt).
    dt : float
        Original step size (in us) of initial_pulse. new_dt = dt / s.
    n_c : int
        Single cavity truncation to train against.
    extra_maxiter : int
        L-BFGS-B iteration budget. The halfway sanity check fires at
        iteration ~extra_maxiter // 2.
    penalties : dict or None
        Base penalty dictionary (deriv/boundary/amp/amp_max). Same
        defaults as refine_pulse_dt.
    penalty_scale : float or dict
        Scaling factor(s) applied to penalties (see refine_pulse_dt).
    save_path : str or None
        Path to save the refined pulse. If None, does not save.
    cav_band, tra_band : (f_lo, f_hi) tuples in MHz, or None
        Hard frequency cutoff on the cavity/transmon drives (see
        optimize_multi_state_pulse's docstring). Both or neither.
    hard_amp_limit : float
        L-BFGS-B box constraint on the raw variable, in rad/us.
    validation_trunc_range : range or list
        Truncations swept by the mid-run and final sanity checks.
    verbose : bool
        Print progress and results.
    """
    if (cav_band is None) != (tra_band is None):
        raise ValueError("cav_band and tra_band must both be given or both be None")
    bandlimit = cav_band is not None and tra_band is not None

    if verbose:
        print("\n" + "=" * 70)
        print("LIGHT DT REFINEMENT STARTED")
        print("=" * 70)
        print(f"Original: N={initial_pulse.shape[0]}, dt={dt}")

    u0 = refine_dt(initial_pulse, s)
    new_dt = dt / s
    N_new = u0.shape[0]

    if verbose:
        print(f"Refined:  N={N_new}, dt={new_dt} (duration unchanged: "
              f"{initial_pulse.shape[0]*dt:.4f} us)")
        print(f"Training truncation  : n_c={n_c}")
        print(f"Extra maxiter        : {extra_maxiter}")
        print(f"Penalty scale        : {penalty_scale}")

    # --- Prepare base penalties (same defaults/scaling logic as refine_pulse_dt) ---
    if penalties is None:
        penalties = {
            'deriv': 0.00001,
            'boundary': 0.00002,
            'amp': 0.00008,
            'amp_max': 40.0
        }

    penalties = penalties.copy()

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

    # --- Build the single training Hamiltonian + state pairs once ---
    # (get_state_pairs depends only on n_c/n_t, not on u, so unlike
    # optimize_multi_state_pulse's per-call rebuild -- needed there to
    # support joblib multiprocess workers across several truncations --
    # a single in-process truncation only needs this built once.)
    H0, Hc = make_hamiltonian(n_t, n_c)
    state_pairs = get_state_pairs(n_c=n_c, n_t=n_t)
    psi_i_list = [p[0] for p in state_pairs]
    psi_f_list = [p[1] for p in state_pairs]

    def to_physical(x):
        u_raw = x.reshape(N_new, 4)
        return project_bandlimit(u_raw, new_dt, cav_band, tra_band) if bandlimit else u_raw

    def objective(x):
        u = to_physical(x)

        F, grad = fidelity_multi_state(u, H0, Hc, psi_i_list, psi_f_list, new_dt, want_grad=True)
        cost = -F
        g = -grad

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

        if bandlimit:
            g = project_bandlimit(g, new_dt, cav_band, tra_band)

        return cost, g.ravel()

    # --- Midpoint sanity check, fired once via callback (no restart) ---
    halfway_iter = extra_maxiter // 2
    check_state = {'count': 0, 'fired': False}

    def callback(xk):
        check_state['count'] += 1
        if not check_state['fired'] and check_state['count'] >= halfway_iter:
            check_state['fired'] = True
            u_mid = to_physical(xk)
            if verbose:
                print(f"\n[Halfway check @ iteration {check_state['count']}]")
            validate_pulse_truncations(
                u=u_mid,
                get_targets_func=get_state_pairs,
                trunc_range=validation_trunc_range,
                n_t=n_t,
                dt=new_dt,
                title="Halfway Cross-Truncation Sanity Check"
            )

    x0 = u0.ravel()
    bounds = [(-hard_amp_limit, hard_amp_limit)] * (N_new * 4)

    if verbose:
        print("\n--- Running light refinement optimization on finer grid ---\n")

    res = minimize(objective, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                   callback=callback,
                   options={'maxiter': extra_maxiter, 'ftol': 1e-12, 'gtol': 1e-8})

    u_refined = to_physical(res.x)

    F_final, _ = fidelity_multi_state(u_refined, H0, Hc, psi_i_list, psi_f_list, new_dt, want_grad=False)

    if save_path:
        np.save(save_path, u_refined)
        if verbose: print(f"Saved refined pulse to {save_path}")

    if verbose:
        print(f"\nFinished: {res.message}")
        print(f"Bare fidelity at training n_c={n_c}: {F_final:.6f}")
        print("\n--- Post-Refinement Validation ---")

    validate_pulse_truncations(
        u=u_refined,
        get_targets_func=get_state_pairs,
        trunc_range=validation_trunc_range,
        n_t=n_t,
        dt=new_dt,
        title="Final Cross-Truncation Sanity Check"
    )

    info = {
        'message': res.message,
        'success': res.success,
        'iterations': res.nit,
        'final_fidelity': F_final,
        'n_c': n_c,
        'dt': new_dt,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("LIGHT DT REFINEMENT COMPLETE")
        print("=" * 70)
        if save_path:
            print(f"Saved refined pulse to: {save_path}")

    return u_refined, info
