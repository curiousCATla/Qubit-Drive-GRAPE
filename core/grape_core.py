#!/usr/bin/env python3
import numpy as np
from numpy.linalg import eigh
from scipy.optimize import minimize
from core.fourier_cutoff import project_bandlimit

# Apple's Accelerate BLAS backend spuriously raises RuntimeWarnings (divide by
# zero / overflow / invalid value) on ordinary complex matmuls (verified: no
# actual NaN/Inf is produced, norms are preserved). Silence just these codes.
np.seterr(divide='ignore', over='ignore', invalid='ignore')

two_pi = 2 * np.pi
chi = two_pi * (-2.194) #in MHz (dispersive shift)
Kerr = two_pi * (-0.0037) #in MHz (Kerr correction: oscillator anharmonicity) 
chip = two_pi * (-0.019) #in MHz (second order dispersive shift)
alpha = two_pi * (-236.0) #in MHz (Transmon anharmonicity) 
dt = 0.002 #in μs

def make_ops(n_t, n_c):
    """
    Create the annihilation operators for the transmon and cavity system.

    Parameters
    n_t : int
        Number of transmon levels.
    n_c : int
        Number of cavity levels.
    """
    a = np.diag(np.sqrt(np.arange(1, n_c)), 1).astype(complex)  # Cavity annihilation matrix operator
    b = np.diag(np.sqrt(np.arange(1, n_t)), 1).astype(complex)  # Transmon annihilation matrix operator
    "lifting the operators to the joint Hilbert space"
    A=np.kron(np.eye(n_t), a)  # Cavity matrix operator in joint space. 
    B=np.kron(b, np.eye(n_c))  # Transmon matrix operator in joint space. 
    #kron makes: transmon index the slow (outer) one & cavity index the fast (inner) one. index(|t,c⟩) = t*n_c + c 
    return A, B

def make_hamiltonian(n_t, n_c):
    A, B = make_ops(n_t, n_c)
    Ad, Bd = A.conj().T, B.conj().T
    nA, nB = Ad @ A, Bd @ B
    H0 = chi * (nA @ nB) + (Kerr / 2) * (Ad @ Ad @ A @ A) + (chip/2) * (nB @ (Ad @ Ad @ A @ A)) 
    if n_t >= 3:
        H0 += (alpha / 2) * (Bd @ Bd @ B @ B) # 2 level transmon, the anharmonicity term is not needed since: B B|transmon> = 0 for the 2-level subspace.
    Hc = [A+Ad, 1j*(A-Ad), B+Bd, 1j*(B-Bd)] # 1j = i the imaginary unit
    return H0, Hc

def refine_dt(waveform, s):
    """Shrink dt by a factor of s via zero-order hold: repeat each of the N
    rows s times, giving an (s*N) x 4 array over the same total duration."""
    return np.repeat(waveform, s, axis=0)

def step_data(H0, Hc, u_k, dt):
    #Return propagator Uk and eig data (w,V) for one time step's controls u_k (length 4).
    Hk = H0 + u_k[0]*Hc[0] + u_k[1]*Hc[1] + u_k[2]*Hc[2] + u_k[3]*Hc[3]
    w, V = eigh(Hk) # w is the array of eigenvalues, V is the matrix of eigenvectors as columns
    Uk = V @ np.diag(np.exp(-1j * dt * w)) @ V.conj().T # Hk = V·diag(w)·V†, so we avoid computing the matrix exponential directly by using the eigendecomposition
    return Uk, w, V

def basis_state (n_t, n_c, t_level, c_level):
    #Return the basis state |t_level, c_level⟩ in the joint Hilbert space of the transmon and cavity.
    #A basis state is a vector of length n_t * n_c with a 1 at the index corresponding to the specified transmon and cavity levels, and 0s elsewhere.
    v = np.zeros(n_t * n_c, dtype=complex)
    v[t_level * n_c + c_level] = 1.0   # ordering: transmon (x) cavity
    return v

_EIGH_CHUNK = 256  # bound peak memory of the batched eigh call (see below)


def _fidelity_core(u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=True, return_raw=False):
    """
    Shared batched core for fidelity_grad/fidelity_multi_state.

    Uk and its eigendecomposition (w, V) depend only on (H0, Hc, u[k]) --
    NOT on which state is being transferred -- so instead of propagating
    each of the M state pairs through its own independent call to
    step_data (as fidelity_grad historically did once per state), every
    per-step quantity that's state-independent (eigh, and the gradient
    basis rotation V^dagger @ Hc[j] @ V) is computed exactly once per
    timestep and reused across all M states. The only per-state work left
    in the inner loop is cheap (n,M)-shaped matmuls/contractions.

    The eigh itself is additionally batched: the (N,n,n) stack of step
    Hamiltonians is diagonalized via np.linalg.eigh on chunks of
    _EIGH_CHUNK steps at a time (rather than one Python-level call per
    step), trading a bounded amount of extra peak memory for far fewer,
    larger LAPACK dispatches -- this is what makes refine_pulse_dt (which
    multiplies N by the dt-refinement factor s) scale better.

    return_raw : bool
        If True, additionally return the raw complex overlap v (M,) --
        v[m] = <psi_f_m|U|psi_i_m> before the |.|^2 -- and, when want_grad
        is also True, dv_all (N, 4, M) -- the per-step, per-channel,
        per-state contraction <lam_k|dUk|phi_{k-1}> *before* it is reduced
        against conj(v[m]) into the per-state gradient. Both are exactly
        what's needed to build fidelity metrics that combine the M states
        *coherently* (i.e. that are sensitive to relative phase between
        them) -- see coherent_fidelity_multi_state -- rather than the
        per-state-independent reduction used by fidelity_multi_state
        below. Default False keeps the existing (F, grad) return signature
        for all current callers.

    Returns
    -------
    F : (M,) real array, F[m] = |<psi_f_m| U |psi_i_m>|^2
    grad : (N, 4, M) real array, the PER-STATE (unaveraged, unsummed)
        gradient of F[m] w.r.t. u -- mirrors what M independent
        fidelity_grad calls would have returned, stacked along a new last
        axis -- or None if want_grad is False.
    v, dv_all : only returned when return_raw is True (see above).
    """
    N = u.shape[0]
    n = H0.shape[0]
    M = len(psi_i_list)

    Hc_stack = np.stack(Hc, axis=0)  # (4, n, n)
    Hk_stack = H0[None, :, :] + np.tensordot(u, Hc_stack, axes=([1], [0]))  # (N, n, n)

    Psi_i = np.stack(psi_i_list, axis=1)  # (n, M)
    Psi_f = np.stack(psi_f_list, axis=1)  # (n, M)

    if not want_grad:
        # Lean path: diagnostics/validation calls only need the final
        # overlap, so there's no need to keep the eigenvector/trajectory
        # history around once each step's contribution has been applied.
        psi = Psi_i.copy()
        for start in range(0, N, _EIGH_CHUNK):
            end = min(start + _EIGH_CHUNK, N)
            w_c, V_c = np.linalg.eigh(Hk_stack[start:end])
            for k in range(end - start):
                Vk = V_c[k]
                Uk = (Vk * np.exp(-1j * dt * w_c[k])[None, :]) @ Vk.conj().T
                psi = Uk @ psi
        v = np.sum(Psi_f.conj() * psi, axis=0)  # (M,)
        F = np.abs(v) ** 2
        if return_raw:
            return F, None, v, None
        return F, None

    w_stack = np.empty((N, n))
    V_stack = np.empty((N, n, n), dtype=complex)
    for start in range(0, N, _EIGH_CHUNK):
        end = min(start + _EIGH_CHUNK, N)
        w_stack[start:end], V_stack[start:end] = np.linalg.eigh(Hk_stack[start:end])
    ew_stack = np.exp(-1j * dt * w_stack)  # (N, n)

    # Forward pass: propagate all M states together, one matmul per step.
    phi = np.empty((N + 1, n, M), dtype=complex)
    phi[0] = Psi_i
    for k in range(N):
        Vk = V_stack[k]
        Uk = (Vk * ew_stack[k][None, :]) @ Vk.conj().T
        phi[k + 1] = Uk @ phi[k]

    v = np.sum(Psi_f.conj() * phi[N], axis=0)  # (M,) <f_m|U|i_m>
    F = np.abs(v) ** 2

    # Backward costates: lambda_k = U_{k+1}^dag ... U_N^dag |f>, batched.
    lam = np.empty((N + 1, n, M), dtype=complex)
    lam[N] = Psi_f
    for k in range(N - 1, -1, -1):
        Vk = V_stack[k]
        Uk_dag = (Vk * np.exp(1j * dt * w_stack[k])[None, :]) @ Vk.conj().T
        lam[k] = Uk_dag @ lam[k + 1]

    grad = np.zeros((N, 4, M))
    dv_all = np.empty((N, 4, M), dtype=complex) if return_raw else None
    for k in range(N):
        w, V = w_stack[k], V_stack[k]
        ew = ew_stack[k]

        p = V.conj().T @ phi[k]      # (n, M) project states onto Hk's eigenbasis
        q = V.conj().T @ lam[k + 1]  # (n, M)

        # ∂U_k/∂u_j = V · ( Φ ∘ (V† H_j V) ) · V†
        dw = w[:, None] - w[None, :]
        near = np.abs(dw) < 1e-10
        dw_safe = np.where(near, 1.0, dw)
        Phi = (ew[:, None] - ew[None, :]) / dw_safe
        Phi = np.where(near, (-1j * dt * ew)[:, None], Phi)

        # X_all[j] = V† Hc[j] V for all 4 control channels in one shot --
        # state-independent, computed once per step and shared below. Uses
        # broadcasted @ (BLAS gemm per slice), NOT np.einsum: plain
        # np.einsum without optimize=True falls back to a non-BLAS
        # contraction and is ~50x slower here for these small matrices.
        VH = V.conj().T
        X_all = VH[None, :, :] @ (Hc_stack @ V)  # (4, n, n)
        PhiX = Phi[None, :, :] * X_all  # (4, n, n)

        qc = q.conj()  # (n, M)
        tmp = PhiX @ p[None, :, :]  # <dUk|phi_{k-1}> per channel, (4, n, M)
        dv = np.sum(qc[None, :, :] * tmp, axis=1)  # <lam_k| dUk |phi_{k-1}>, (4, M)
        grad[k] = 2.0 * np.real(np.conj(v)[None, :] * dv)
        if return_raw:
            dv_all[k] = dv

    if return_raw:
        return F, grad, v, dv_all
    return F, grad


def fidelity_grad(u, H0, Hc, psi_i, psi_f, dt, want_grad=True):
    F, grad = _fidelity_core(u, H0, Hc, [psi_i], [psi_f], dt, want_grad=want_grad)
    return F[0], (grad[:, :, 0] if grad is not None else None)


def fidelity_multi_state(u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=True):
    """
    Average fidelity over multiple state transfers.
    Uses the simple average of |<f|U|i>|^2 (more stable for optimization).
    """
    M = len(psi_i_list)
    F, grad = _fidelity_core(u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=want_grad)
    F_avg = np.sum(F) / M
    grad_avg = np.sum(grad, axis=2) / M if want_grad else None
    return F_avg, grad_avg


def coherent_fidelity_multi_state(u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=True):
    """
    Coherent (process) fidelity over multiple state transfers:
        F = |sum_m <f_m|U|i_m>|^2 / M^2
    Unlike fidelity_multi_state's per-state average -- which only requires
    each output to match its own target up to an ARBITRARY, independent
    global phase -- this reduction sums the raw complex overlaps v_m before
    squaring, so it is only maximal when every branch matches its target
    with the SAME global phase. That's exactly the property
    fidelity_multi_state cannot see or enforce, and it's what a logical
    gate's *relative* phase between its |+Z_L>/|-Z_L> training pairs
    actually depends on (e.g. T's pi/4 relative phase, H's self-inverse
    H^2=I property). Use this in place of fidelity_multi_state when
    training gates whose correctness hinges on that relative phase.
    """
    M = len(psi_i_list)
    if want_grad:
        F, grad, v, dv_all = _fidelity_core(
            u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=True, return_raw=True
        )
        V = np.sum(v)
        F_coh = np.abs(V) ** 2 / M ** 2
        dv_sum = np.sum(dv_all, axis=2)  # (N, 4)
        grad_coh = (2.0 / M ** 2) * np.real(np.conj(V) * dv_sum)
        return F_coh, grad_coh
    else:
        F, _, v, _ = _fidelity_core(
            u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=False, return_raw=True
        )
        V = np.sum(v)
        F_coh = np.abs(V) ** 2 / M ** 2
        return F_coh, None


def leakage_grad(u, H0, Hc, psi_i, dt, n_c, n_t, N_cut, leak_tol=1e-5, want_grad=True):
    """
    Penalize cavity population above Fock level N_cut at every step of the
    trajectory (running cost), unlike fidelity_grad's terminal-only cost.

    Returns (cost, grad, max_leak):
    - cost: sum_k max(P(n>N_cut, t_k) - leak_tol, 0)^2 over the trajectory
    - grad: [N,4] gradient of cost w.r.t. u (None if want_grad=False)
    - max_leak: raw (unthresholded) worst-case P(n>N_cut, t) observed --
      diagnostic to check directly against the 1e-5 requirement.

    Same forward propagation and eigenbasis (Phi/X) machinery as
    fidelity_grad, but the backward costate is seeded with a source term at
    EVERY step (chi_k = 2*excess_k*mask*phi_k) instead of only at the final
    step, since the cost here is a sum over the whole trajectory rather than
    a single terminal overlap. Because each l_k is already real-valued, the
    final gradient contraction skips fidelity_grad's extra conj(v) factor
    (that factor arises specifically from the |v|^2 terminal-cost form).
    """
    N = u.shape[0]
    mask = (np.arange(n_t * n_c) % n_c) > N_cut  # True where cavity level > N_cut

    phi = [psi_i.copy()]
    Us, Ws, Vs = [], [], []
    psi = psi_i.copy()
    for k in range(N):
        Uk, w, V = step_data(H0, Hc, u[k], dt)
        psi = Uk @ psi
        phi.append(psi)
        Us.append(Uk); Ws.append(w); Vs.append(V)

    l = np.array([np.sum(np.abs(phi[k][mask]) ** 2) for k in range(1, N + 1)])  # P(n>N_cut, t_k)
    excess = np.maximum(l - leak_tol, 0.0)
    cost = np.sum(excess ** 2)
    max_leak = np.max(l)

    if not want_grad:
        return cost, None, max_leak

    # backward costates with a running source injected at every step
    mu = [None] * (N + 1)
    mu[N] = 2 * excess[N - 1] * mask * phi[N]
    for k in range(N - 1, 0, -1):
        mu[k] = 2 * excess[k - 1] * mask * phi[k] + Us[k].conj().T @ mu[k + 1]

    grad = np.zeros((N, 4))
    for k in range(N):
        w, V = Ws[k], Vs[k]

        p = V.conj().T @ phi[k]
        q = V.conj().T @ mu[k + 1]

        ew = np.exp(-1j * dt * w)
        dw = w[:, None] - w[None, :]
        near = np.abs(dw) < 1e-10
        dw_safe = np.where(near, 1.0, dw)
        Phi = (ew[:, None] - ew[None, :]) / dw_safe
        Phi = np.where(near, (-1j * dt * ew)[:, None], Phi)
        qc = q.conj()
        for j in range(4):
            X = V.conj().T @ Hc[j] @ V
            dv = qc @ ((Phi * X) @ p)
            grad[k, j] = 2.0 * np.real(dv)

    return cost, grad, max_leak

def leakage_multi_state(u, H0, Hc, psi_i_list, dt, n_c, n_t, N_cut, leak_tol=1e-5, want_grad=True):
    """Average leakage_grad over multiple initial states (mirrors fidelity_multi_state)."""
    M = len(psi_i_list)
    total_cost = 0.0
    total_grad = np.zeros_like(u)
    worst = 0.0

    for psi_i in psi_i_list:
        c, g, m = leakage_grad(u, H0, Hc, psi_i, dt, n_c, n_t, N_cut, leak_tol=leak_tol, want_grad=want_grad)
        total_cost += c
        worst = max(worst, m)
        if want_grad and g is not None:
            total_grad += g

    cost_avg = total_cost / M
    grad_avg = total_grad / M if want_grad else None
    return cost_avg, grad_avg, worst

def average_fidelity_two_transfers(u, H0, Hc, psi_i_list, psi_f_list, dt):
    """Compute average fidelity over two state transfers."""
    F_sum = 0.0
    for psi_i, psi_f in zip(psi_i_list, psi_f_list):
        F, _ = fidelity_grad(u, H0, Hc, psi_i, psi_f, dt, want_grad=False)
        F_sum += F
    return F_sum / len(psi_i_list)

def make_objective(H0, Hc, psi_i, psi_f, dt, N):
    #Return a function that computes the fidelity and its gradient for a given control sequence u.
    def objective(x):
        u = x.reshape(N, 4) # reshape the input array x into a 2D array with 4 columns (one for each control)
        F, grad = fidelity_grad(u, H0, Hc, psi_i, psi_f, dt)
        return -F, -grad.ravel() 
    return objective


def smooth_initial_controls(N, amp, cutoff_frac, seed):
    #Build the pulse in the frequency domain and keep only low frequencies
    rng = np.random.default_rng(seed)
    nf = N// 2 + 1  # number of real-FFT frequency bins
    kcut = max(1, int(cutoff_frac * nf))    # how many LOW bins to populate
    u = np.zeros((N, 4))
    for j in range(4):
        spec = np.zeros(nf, dtype=complex) #empty spectrum array of length nf, initialized to zero
        spec[:kcut] = rng.standard_normal(kcut) + 1j*rng.standard_normal(kcut)# populate the first kcut frequency bins with random complex numbers drawn from a standard normal distribution
        col = np.fft.irfft(spec, n=N)                 # -> length-N REAL series
        u[:, j] = amp * col / (np.std(col) + 1e-12)   # rescale to target rms
    return u

def derivative_penalty(u):
    #Copmute the smoothness penalty and its gradient for optimization
    
    diff = u[1:] - u[:-1]
    smooth_pen = np.sum(diff**2)

    grad = np.zeros_like(u)
    grad[1:] += 2*diff #condibution from u[k]
    grad[:-1] -=2*diff #contribution from u[k-1]

    return smooth_pen, grad

def boundary_penalty(u):
    """
    Penalty for control waveforms not starting and ending at zero.
    """
    g_boundary = np.sum(u[0]**2) + np.sum(u[-1]**2)
    grad = np.zeros_like(u)
    grad[0] += 2*u[0]
    grad[-1] += 2*u[-1]
    return g_boundary, grad

def amplitude_penalty(u, amp_max=40.0):
    # Compute excesss amplitude 
    excess = np.maximum(np.abs(u)-amp_max, 0)
    g_amp = np.sum(excess**2)
    grad = np.zeros_like(u)
    mask = np.abs(u) > amp_max
    grad[mask] = 2*excess[mask] * np.sign(u[mask])

    return g_amp, grad


def make_objective_with_pen(H0, Hc, psi_i, psi_f, dt, N, lambda_deriv=0.0, lambda_boundary=0.0, lambda_amp=0.0, amp_max=40.0, cav_band=None, tra_band=None):
    """
    Objective that includes fidelity + derivative + boundary + amplitude penalties.

    cav_band, tra_band : (f_lo, f_hi) tuples in MHz, or None
        Hard frequency cutoff (Heeres et al. 2017, Supp. Eq. 22) applied via
        orthogonal projection: x is a free pre-image, the physical pulse is
        u = P(x). Leave both None to disable.
    """
    bandlimit = cav_band is not None and tra_band is not None

    def objective(x):
        u_raw = x.reshape(N, 4)
        u = project_bandlimit(u_raw, dt, cav_band, tra_band) if bandlimit else u_raw

        # Fidelity
        F, grad_F = fidelity_grad(u, H0, Hc, psi_i, psi_f, dt)

        total_cost = -F
        total_grad = -grad_F

        # Derivative (smoothness) penalty
        if lambda_deriv > 0:
            g_deriv, grad_deriv = derivative_penalty(u)
            total_cost  += lambda_deriv * g_deriv
            total_grad  += lambda_deriv * grad_deriv

        # Boundary penalty (start and end at zero)
        if lambda_boundary > 0:
            g_bound, grad_bound = boundary_penalty(u)
            total_cost  += lambda_boundary * g_bound
            total_grad  += lambda_boundary * grad_bound

        # Amplitude penalty
        if lambda_amp > 0:
            g_amp, grad_amp = amplitude_penalty(u, amp_max=amp_max)
            total_cost  += lambda_amp * g_amp
            total_grad  += lambda_amp * grad_amp

        # Chain rule for the reparametrization: dCost/dx = P(dCost/du),
        # valid because P is self-adjoint & idempotent.
        if bandlimit:
            total_grad = project_bandlimit(total_grad, dt, cav_band, tra_band)

        return total_cost, total_grad.ravel()

    return objective


def make_objective_multi_trunc(H0_list, Hc_list, psi_i_list, psi_f_list, dt, N,lambda_deriv=0.0, lambda_boundary=0.0, lambda_amp=0.0, lambda_disc=0.0, amp_max=40.0, cav_band=None, tra_band=None):
    """
    Multi-truncation objective with discrepancy penalty + existing penalties.

    cav_band, tra_band : (f_lo, f_hi) tuples in MHz, or None
        Hard frequency cutoff (Heeres et al. 2017, Supp. Eq. 22) applied via
        orthogonal projection: x is a free pre-image, the physical pulse is
        u = P(x). Leave both None to disable.
    """
    n_trunc = len(H0_list)
    bandlimit = cav_band is not None and tra_band is not None
    def objective(x):
        u_raw = x.reshape(N, 4)
        u = project_bandlimit(u_raw, dt, cav_band, tra_band) if bandlimit else u_raw

        total_F = 0.0
        total_grad = np.zeros_like(u)
        F_values = []
        
        for H0, Hc, psi_i, psi_f in zip(H0_list, Hc_list, psi_i_list, psi_f_list):
            F, grad_F = fidelity_grad(u, H0, Hc, psi_i, psi_f, dt)
            total_F += F
            total_grad += grad_F
            F_values.append(F)
        
        # Average fidelity (more stable scaling)
        avg_F = total_F / n_trunc
        avg_grad = total_grad / n_trunc
        
        # We minimize -avg_F
        cost = -avg_F
        grad = -avg_grad
        
        # Discrepancy penalty (optional)
        if lambda_disc > 0 and len(F_values) > 1:
            g_disc = 0.0
            for i in range(len(F_values)):
                for j in range(i+1, len(F_values)):
                    g_disc += (F_values[i] - F_values[j])**2
            cost += lambda_disc * g_disc
            # (gradient of discrepancy left as 0 for simplicity)
        
        # Add other penalties
        if lambda_deriv > 0:
            g_d, gr_d = derivative_penalty(u)
            cost += lambda_deriv * g_d
            grad += lambda_deriv * gr_d
        
        if lambda_boundary > 0:
            g_b, gr_b = boundary_penalty(u)
            cost += lambda_boundary * g_b
            grad += lambda_boundary * gr_b
        
        if lambda_amp > 0:
            g_a, gr_a = amplitude_penalty(u, amp_max=amp_max)
            cost += lambda_amp * g_a
            grad += lambda_amp * gr_a

        # Chain rule for the reparametrization: dCost/dx = P(dCost/du),
        # valid because P is self-adjoint & idempotent.
        if bandlimit:
            grad = project_bandlimit(grad, dt, cav_band, tra_band)

        return cost, grad.ravel()

    return objective


def optimize_controls(H0, Hc, psi_i, psi_f, dt, N, u0, trunc_list=None,lambda_deriv = 0.0, lambda_boundary = 0.0, lambda_amp = 0.0, lambda_disc=0.0, amp_max= 40.0, cav_band=None, tra_band=None, hard_amp_limit=50.0):
    """
    cav_band, tra_band : (f_lo, f_hi) tuples in MHz, or None
        Hard frequency cutoff (Heeres et al. 2017, Supp. Eq. 22). When both
        are given, the returned pulse is exactly band-limited via
        orthogonal projection. Leave both None to disable.
    hard_amp_limit : float
        L-BFGS-B box constraint on the raw variable, in rad/us -- the true
        hard amplitude bound, decoupled from amp_max (which only sets the
        soft quadratic amplitude_penalty threshold).
    """
    bandlimit = cav_band is not None and tra_band is not None
    if trunc_list is not None and len(trunc_list) > 1:
        # === Multi-truncation mode ===
        print(f"Using multi-truncation mode with truncations: {trunc_list}")
        
        H0_list, Hc_list, psi_i_list, psi_f_list = [], [], [], []
        
        for nc in trunc_list:
            H0_k, Hc_k, psi_i_k, psi_f_k = build_system(nc)
            H0_list.append(H0_k)
            Hc_list.append(Hc_k)
            psi_i_list.append(psi_i_k)
            psi_f_list.append(psi_f_k)
        
        objective = make_objective_multi_trunc(
            H0_list, Hc_list, psi_i_list, psi_f_list, dt, N,
            lambda_deriv=lambda_deriv,
            lambda_boundary=lambda_boundary,
            lambda_amp=lambda_amp,
            lambda_disc=lambda_disc,
            amp_max=amp_max,
            cav_band=cav_band,
            tra_band=tra_band
        )

    else:
    # Single trunction, ptimize the control sequence u to maximize the fidelity between the initial state psi_i and the final state psi_f.
        objective = make_objective_with_pen(H0, Hc, psi_i, psi_f, dt, N,lambda_deriv=lambda_deriv,lambda_boundary=lambda_boundary,lambda_amp=lambda_amp,amp_max=amp_max,cav_band=cav_band,tra_band=tra_band)

    x0 = u0.ravel() # flatten the initial control array into a 1D array
    bounds = [(-hard_amp_limit, hard_amp_limit)] * (N * 4)
    res = minimize(objective, x0, method='L-BFGS-B', jac=True, bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8 })
    # res.x is the raw pre-image; project to get the physical (band-limited) pulse.
    u_opt = project_bandlimit(res.x.reshape(N, 4), dt, cav_band, tra_band) if bandlimit \
        else res.x.reshape(N, 4)
    if not res.success:
        print("Optimization failed:", res.message)
    # Final fidelity evaluation (always done at the largest truncation)
    if trunc_list is not None:
        n_c_final = max(trunc_list)
        H0_final, Hc_final, psi_i_final, psi_f_final = build_system(n_c_final)
    else:
        H0_final, Hc_final, psi_i_final, psi_f_final = H0, Hc, psi_i, psi_f
    
    F_final, _ = fidelity_grad(u_opt, H0_final, Hc_final, psi_i_final, psi_f_final, dt, want_grad=False)
    print(f"Final fidelity (at n_c={n_c_final if trunc_list else 'single'}): {F_final:.6f}")
    
    return u_opt, F_final, res

def build_system(n_c, n_t=2):
    """
    Build H0, Hc, psi_i, psi_f for a given cavity truncation n_c.
    """
    A, B = make_ops(n_t, n_c)
    Ad, Bd = A.conj().T, B.conj().T
    nA, nB = Ad @ A, Bd @ B

    H0 = (chi * (nA @ nB) +
          (Kerr / 2) * (Ad @ Ad @ A @ A) +
          (chip / 2) * (nB @ (Ad @ Ad @ A @ A)))

    if n_t >= 3:
        H0 += (alpha / 2) * (Bd @ Bd @ B @ B)

    Hc = [A + Ad, 1j * (A - Ad), B + Bd, 1j * (B - Bd)]

    psi_i = basis_state(n_t, n_c, 0, 0)   # |g,0⟩
    psi_f = basis_state(n_t, n_c, 0, 6)   # |g,6⟩

    return H0, Hc, psi_i, psi_f

