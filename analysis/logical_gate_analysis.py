#!/usr/bin/env python3
import os
import sys

import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.cat_code import *
from core.grape_core import *

# ---------------- Parameters ----------------
n_c = 24          # main truncation for final reporting
n_t = 2
dt = 0.002
N = 550
alpha = np.sqrt(3.0)

lambda_deriv   = 0.000008
lambda_boundary = 0.00004
lambda_amp     = 0.00015
amp_max = 40.0

trunc_list = [20, 24, 28]

print(f"Optimizing U_enc with multi-truncation: {trunc_list}")

# ---------------- Build systems and targets for all truncations ----------------
H0_list, Hc_list = [], []
psi_i_list_all, psi_f_list_all = [], []

for nc in trunc_list:
    H0_k, Hc_k = make_hamiltonian(n_t, nc)
    H0_list.append(H0_k)
    Hc_list.append(Hc_k)
    
    psi_i_g_k, psi_f_plus_k, psi_i_e_k, psi_f_minus_k = get_encode_targets(n_c=nc, n_t=n_t, alpha=alpha)
    psi_i_list_all.append([psi_i_g_k, psi_i_e_k])
    psi_f_list_all.append([psi_f_plus_k, psi_f_minus_k])

# Main system (for reporting)
H0, Hc = make_hamiltonian(n_t, n_c)
psi_i_list = psi_i_list_all[trunc_list.index(n_c)]
psi_f_list = psi_f_list_all[trunc_list.index(n_c)]

# ---------------- Initial controls ----------------
# ---------------- Initial controls (Warm Start) ----------------
#  GET THE OPTIMIZED PULSE
USE_SAVED_PULSE = True          # ← Change to False if you want to re-optimize
PULSE_DIR = "pulses"
os.makedirs(PULSE_DIR, exist_ok=True)
enc_multi_path = os.path.join(PULSE_DIR, "u_enc_multi.npy")

if USE_SAVED_PULSE and os.path.exists(enc_multi_path):
    u0 = np.load(enc_multi_path)
    print(f"Using WARM START from {enc_multi_path}")
else:
    u0 = smooth_initial_controls(N, amp=12.0, cutoff_frac=0.04, seed=42)
    print("Starting from random smooth controls")

x0 = u0.ravel()

bounds = [(-amp_max, amp_max)] * (N * 4)

# ---------------- Multi-truncation Objective ----------------
def make_multi_trunc_encode_objective(H0_list, Hc_list, psi_i_list_all, psi_f_list_all, dt, N,
                                      lambda_deriv=0.0, lambda_boundary=0.0, lambda_amp=0.0, amp_max=40.0):
    
    def evaluate_truncation(u, H0_k, Hc_k, psi_i_k, psi_f_k):
        """Helper function for parallel execution"""
        return fidelity_multi_state(u, H0_k, Hc_k, psi_i_k, psi_f_k, dt, want_grad=True)
    
    def objective(x):
        u = x.reshape(N, 4)
        
        # === Parallel execution across truncations ===
        results = Parallel(n_jobs=3)(delayed(evaluate_truncation)(
            u, H0_k, Hc_k, psi_i_k, psi_f_k
        ) for H0_k, Hc_k, psi_i_k, psi_f_k in zip(H0_list, Hc_list, psi_i_list_all, psi_f_list_all))
        
        total_F = 0.0
        total_grad = np.zeros_like(u)
        
        for F_k, grad_k in results:
            total_F += F_k
            if grad_k is not None:
                total_grad += grad_k
        
        M = len(H0_list)
        F_avg = total_F / M
        grad_avg = total_grad / M
        
        cost = -F_avg
        g = -grad_avg
        
        # Penalties (same as before)
        if lambda_deriv > 0:
            g_d, gr_d = derivative_penalty(u)
            cost += lambda_deriv * g_d
            g += lambda_deriv * gr_d
        if lambda_boundary > 0:
            g_b, gr_b = boundary_penalty(u)
            cost += lambda_boundary * g_b
            g += lambda_boundary * gr_b
        if lambda_amp > 0:
            g_a, gr_a = amplitude_penalty(u, amp_max=amp_max)
            cost += lambda_amp * g_a
            g += lambda_amp * gr_a
        
        return cost, g.ravel()
    
    return objective

objective = make_multi_trunc_encode_objective(
    H0_list, Hc_list, psi_i_list_all, psi_f_list_all, dt, N,
    lambda_deriv=lambda_deriv,
    lambda_boundary=lambda_boundary,
    lambda_amp=lambda_amp,
    amp_max=amp_max
)

# ---------------- Run Optimization ----------------
print("Starting multi-truncation optimization...")
res = minimize(
    objective, x0,
    method='L-BFGS-B',
    jac=True,
    bounds=bounds,
    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': True}
)

u_enc = res.x.reshape(N, 4)
np.save(enc_multi_path, u_enc)

# ---------------- Final Evaluation ----------------
F_final, _ = fidelity_multi_state(u_enc, H0, Hc, psi_i_list, psi_f_list, dt, want_grad=False)
print(f"\nOptimization finished: {res.message}")
print(f"Final fidelity at n_c={n_c}: {F_final:.6f}")

# Full truncation validation
print("\nTruncation validation after multi-trunc training:")
for nc_test in [18, 20, 22, 24, 26, 28, 30]:
    H0_t, Hc_t = make_hamiltonian(n_t, nc_test)
    psi_i_g_t, psi_f_plus_t, psi_i_e_t, psi_f_minus_t = get_encode_targets(n_c=nc_test)
    F_t, _ = fidelity_multi_state(u_enc, H0_t, Hc_t,
                                   [psi_i_g_t, psi_i_e_t],
                                   [psi_f_plus_t, psi_f_minus_t], dt, want_grad=False)
    print(f"  n_c={nc_test:2d}: F = {F_t:.6f}")