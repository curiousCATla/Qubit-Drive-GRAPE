#!/usr/bin/env python3
import numpy as np
from numpy.linalg import eigh
from scipy.optimize import minimize
from fourier_cutoff import project_bandlimit

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
        H0 += (alpha / 2) * (Bd @ Bd @ B @ B) # 2 level transmon, the anharmonicity term is not needed ~ 0
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

def fidelity_grad(u, H0, Hc, psi_i, psi_f, dt,want_grad=True):
    #u: [N,4] 2D array of real controls
    N = u.shape[0] # N is the number of rows in the array
    phi = [psi_i.copy()]
    Ws, Vs, Us = [], [], []
    psi= psi_i.copy() # psi is the current state of the system, initialized to the initial state psi_i
    for k in range(N):
        Uk, w, V = step_data(H0, Hc, u[k], dt) 
        psi = Uk @ psi # update the state of the system by applying the propagator Uk
        phi.append(psi) # store the state of the system
        Us.append(Uk); Ws.append(w); Vs.append(V) 
    v = np.vdot(psi_f, psi) # inner product <f|U|i>  (complex scalar)
    F = np.abs(v)**2 # fidelity is the squared magnitude of the inner product
    if not want_grad:
        return F, None
    # backward costates: lambda_k = U_{k+1}^dag ... U_N^dag |f>
    lam = [None]*(N+1) #creates a list containing N+1 None values. 
    lam[N] = psi_f.copy()
    for k in range(N-1, -1, -1):
        lam[k] = Us[k].conj().T @ lam[k+1]
    
    grad = np.zeros((N, 4))
    for k in range(N):
        w, V = Ws[k], Vs[k]
        
        p = V.conj().T @ phi[k] # project |phi_{k-1}> onto the eigenbasis of Hk
        q = V.conj().T @ lam[k+1] #  project <lambda_k| onto the eigenbasis of Hk

        #∂U_k/∂u_j = V · ( Φ ∘ (V† H_j V) ) · V†

        ew = np.exp(-1j*dt*w)
        dw = w[:,None] - w[None,:] # w_m - w_n for all pairs of eigenvalues, as an (n.n) matrix
        near =np.abs(dw) < 1e-10 # boolean array indicating where the differences are near zero
        #H0 is the drift in the rotating frame, not the lab frame; it subtracts exactly one ℏωC per photon
        #|g,0⟩ and |g,1⟩ are degenerate, so we need to handle the case where dw is near zero to avoid division by zero in the calculation of Phi.
        dw_safe = np.where(near, 1.0, dw) # replace near-zero differences with 1.0 to avoid division by zero
        Phi = (ew[:,None] - ew[None,:]) / dw_safe # off-diagonal difference quotients
        Phi = np.where(near, (-1j*dt*ew)[:,None], Phi) # fill the diagonal of Phi with the true derivative for near-zero differences
        qc = q.conj()
        for j in range(4):
            X = V.conj().T @ Hc[j] @ V # Hj in eigenbasis
            dv = qc @ ((Phi * X) @ p) # <lam_k| dUk |phi_{k-1}>
            grad[k, j] = 2.0 * np.real(np.conj(v)* dv) # gradient of fidelity with respect to control u_j at time step k
    return F, grad

def fidelity_multi_state(u, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=True):
    """
    Average fidelity over multiple state transfers.
    Uses the simple average of |<f|U|i>|^2 (more stable for optimization).
    """
    M = len(psi_i_list)
    total_F = 0.0
    total_grad = np.zeros_like(u)
    
    for psi_i, psi_f in zip(psi_i_list, psi_f_list):
        F_i, grad_i = fidelity_grad(u, H0, Hc, psi_i, psi_f, dt, want_grad=want_grad)
        total_F += F_i
        if want_grad and grad_i is not None:
            total_grad += grad_i
    
    F_avg = total_F / M
    grad_avg = total_grad / M if want_grad else None
    
    return F_avg, grad_avg

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

