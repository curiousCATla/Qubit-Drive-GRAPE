#!/usr/bin/env python3
"""
decoherence_sim.py

Post-optimization decoherence simulation for your GRAPE pulses.
Uses the Lindblad master equation with parameters from your
Supplementary Information (Heeres et al. 2017).

This simulates the realistic (decoherence-limited) evolution
after/bwhile applying your optimized control pulse.

Usage:
    from decoherence_sim import simulate_with_decoherence

    u = np.load("pulses/u_opt.npy")
    psi0 = basis_state(n_t=3, n_c=24, t_level=0, c_level=0)  # |g,0⟩
    rho_final = simulate_with_decoherence(u, psi0)
    # Then compute fidelity, Wigner function, etc.
"""

import os
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ============================================================
# IMPORTS FROM YOUR EXISTING PROJECT
# ============================================================
from core.grape_core import make_hamiltonian, make_ops, basis_state


def simulate_with_decoherence(u, psi0,
                              n_t=3, n_c=24, dt=0.002,
                              T1_C=2.7e-3, T1_T=170e-6, T_phi=43e-6,
                              verbose=True):
    """
    Evolve the density matrix through the GRAPE pulse while including
    decoherence using the Lindblad master equation.

    Parameters
    ----------
    u : np.ndarray
        Control pulse of shape (N, 4)
    psi0 : np.ndarray
        Initial state vector (joint transmon-cavity)
    n_t, n_c : int
        Number of transmon and cavity levels
    dt : float
        Time step in microseconds (same as your optimization)
    T1_C, T1_T, T_phi : float
        Coherence times in seconds (from your SI Table 1)
    verbose : bool
        Print progress and final diagnostics

    Returns
    -------
    rho_final : np.ndarray
        Final density matrix (shape: (d, d) where d = n_t * n_c)
    """
    d = n_t * n_c
    I = sp.identity(d, format='csr', dtype=complex)

    # === Rates from Supplementary Information ===
    # T1_C, T1_T, T_phi are given in seconds; dt (and the rest of this
    # simulation) is in microseconds, so convert rates to 1/µs to match.
    kappa     = 1e-6 / T1_C      # Cavity relaxation rate (1/µs)
    gamma     = 1e-6 / T1_T      # Transmon relaxation rate (1/µs)
    gamma_phi = 1e-6 / T_phi     # Transmon pure dephasing rate (1/µs)

    if verbose:
        print(f"\n=== Decoherence Simulation ===")
        print(f"T1_C  = {T1_C*1e3:.2f} ms   →  κ/2π ≈ {kappa*1e6/(2*np.pi):.1f} Hz")
        print(f"T1_T  = {T1_T*1e6:.1f} µs   →  γ/2π ≈ {gamma*1e6/(2*np.pi):.1f} kHz")
        print(f"T_φ   = {T_phi*1e6:.1f} µs  →  γ_φ/2π ≈ {gamma_phi*1e6/(2*np.pi):.1f} kHz")
        print(f"Total gate time ≈ {len(u)*dt*1000:.2f} ns\n")

    # Jump operators (sparse: A, B are single-band shift operators)
    A, B = make_ops(n_t, n_c)
    A, B = sp.csr_matrix(A), sp.csr_matrix(B)
    jump_ops = [A, B, B.conj().T @ B]           # a, b, b†b
    rates    = [kappa, gamma, gamma_phi]

    # Initial density matrix
    rho = np.outer(psi0, psi0.conj())
    rho_vec = rho.flatten(order='C')            # Vectorized density matrix (must match the
                                                 # row-major Kronecker convention used to build L below)

    H0, Hc = make_hamiltonian(n_t, n_c)

    for i, uk in enumerate(u):
        # Build instantaneous Hamiltonian
        H = (H0 +
             uk[0] * Hc[0] +
             uk[1] * Hc[1] +
             uk[2] * Hc[2] +
             uk[3] * Hc[3])
        H = sp.csr_matrix(H)

        # === Build Liouvillian superoperator (sparse) ===
        L = -1j * (sp.kron(H, I) - sp.kron(I, H.T))

        # Add dissipative terms
        for rate, L_op in zip(rates, jump_ops):
            L_diss = (sp.kron(L_op, L_op.conj()) -
                      0.5 * (sp.kron((L_op.conj().T @ L_op), I) +
                             sp.kron(I, (L_op.T @ L_op.conj()))))
            L = L + rate * L_diss

        # Evolve one time step: apply exp(L*dt) directly to the vector
        # (avoids ever forming the dense d^2 x d^2 matrix exponential)
        rho_vec = expm_multiply((L * dt).tocsc(), rho_vec)

        if verbose and (i + 1) % max(1, len(u)//10) == 0:
            print(f"  Step {i+1}/{len(u)} completed")

    # Reshape back to density matrix
    rho_final = rho_vec.reshape((d, d), order='C')

    if verbose:
        print("\nSimulation finished.")

    return rho_final


def compute_fidelity(rho, psi_target):
    """Compute state fidelity F = <ψ_target| ρ |ψ_target>"""
    return np.real(psi_target.conj().T @ rho @ psi_target)


# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    print("=== Decoherence Simulation Demo ===\n")

    # Example: Prepare |g,0⟩ then apply a pulse (you can replace with your real pulse)
    n_t, n_c = 3, 24
    psi0 = basis_state(n_t, n_c, t_level=0, c_level=0)   # |g, 0⟩

    # For demo, use a short zero pulse (you should load your real pulse)
    N_steps = 200
    u_demo = np.zeros((N_steps, 4))
    u_load = np.load("pulses/u_opt.npy")

    rho_final = simulate_with_decoherence(
        u=u_load,
        psi0=psi0,
        n_t=n_t,
        n_c=n_c,
        verbose=True
    )

    # Example target: |g,6⟩
    psi_target = basis_state(n_t, n_c, t_level=0, c_level=6)
    fid = compute_fidelity(rho_final, psi_target)

    print(f"\nFinal fidelity with |g,6⟩: {fid:.6f}")
    print("\nYou can now use rho_final to compute Wigner function, photon number distribution, etc.")