import numpy as np
from numpy.linalg import eigh
from scipy.optimize import minimize


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
        with np.errstate(divide='ignore', invalid='ignore'):
            Phi = (ew[:,None] - ew[None,:]) / dw # off-diagonal difference quotients
        np.fill_diagonal(Phi, -1j*dt*ew) # fill the diagonal of Phi with the true derivative 
        qc = q.conj()
        for j in range(4):
            X = V.conj().T @ Hc[j] @ V # Hj in eigenbasis
            dv = qc @ ((Phi * X) @ p) # <lam_k| dUk |phi_{k-1}>
            grad[k, j] = 2.0 * np.real(np.conj(v)* dv) # gradient of fidelity with respect to control u_j at time step k
    return F, grad

def make_objective(H0, Hc, psi_i, psi_f, dt, N):
    #Return a function that computes the fidelity and its gradient for a given control sequence u.
    def objective(x):
        u = x.reshape(N, 4) # reshape the input array x into a 2D array with 4 columns (one for each control)
        F, grad = fidelity_grad(u, H0, Hc, psi_i, psi_f, dt)
        return -F, -grad.ravel() 
    return objective

def optimize_controls(H0, Hc, psi_i, psi_f, dt, N, u0):
    #Optimize the control sequence u to maximize the fidelity between the initial state psi_i and the final state psi_f.
    objective = make_objective(H0, Hc, psi_i, psi_f, dt, N)
    x0 = u0.ravel() # flatten the initial control array into a 1D array
    res = minimize(objective, x0, method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8 })
    u_opt = res.x.reshape(N, 4) # reshape the optimized control array back into a 2D array with 4 columns
    if not res.success:
        print("Optimization failed:", res.message)
    F_reported   = -res.fun
    F_recomputed, _ = fidelity_grad(u_opt, H0, Hc, psi_i, psi_f, dt, want_grad=False)
    print(F_reported, F_recomputed, abs(F_reported - F_recomputed))
    return u_opt, -res.fun, res

