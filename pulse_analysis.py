#!/usr/bin/env python3
"""
pulse_analysis.py

Takes the optmized GRAPE pulse (u_opt) that prepares the state |g,6> from |g,0> 
and does the following analysis:
1. Simulate the full time-depedent cavity photon number distribution P(n.t)
2. Visualize the pulse and the photon number distribution
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0,'.')
from grape_core import *

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

two_pi = 2 * np.pi 
dt = 0.002 #in μs
n_t = 3 #number of transmon levels
n_c = 24 #number of cavity levels
N = 250 #number of time steps dt*N = total time of the pulse
amp_init = 6.0 #amplitude used for smooth random initialization
cutoff_frac = 0.03 #cutoff frequency fraction used for smooth random initialization
seed = 1 #random seed used for smooth random initialization

#Build the operators and Hamiltonian for the transmon-cavity system
A, B = make_ops(n_t, n_c)
H0, Hc = make_hamiltonian(n_t, n_c)
psi_i = basis_state(n_t, n_c, 0, 0) # initial state |g,0⟩
psi_f = basis_state(n_t, n_c, 0, 6) # target state |g,6⟩

def evaluate_fidelity_at_truncations(u, n_c_list, n_t=2, dt=0.002):
    """
    Evaluate the fidelity of a control pulse u at multiple cavity truncations.
    Returns a dictionary {n_c: fidelity}
    """
    results = {}
    for n_c in n_c_list:
        H0, Hc, psi_i, psi_f = build_system(n_c, n_t=n_t)
        F, _ = fidelity_grad(u, H0, Hc, psi_i, psi_f, dt, want_grad=False)
        results[n_c] = F
        print(f"n_c = {n_c:2d} → Fidelity = {F:.6f}")
    return results

#  GET THE OPTIMIZED PULSE
USE_SAVED_PULSE = False          # ← Change to False if you want to re-optimize

PULSE_DIR = "pulses"
os.makedirs(PULSE_DIR, exist_ok=True)

if USE_SAVED_PULSE:
    u_opt = np.load(os.path.join(PULSE_DIR, "u_opt.npy"))
    print(f"Loaded optimized pulse from {PULSE_DIR}/u_opt.npy")

else:
    #Create the smoothed initial controls u0 for the optimization
    u0 = smooth_initial_controls(N,amp_init, cutoff_frac, seed)
    #Run the optimizer
    #u_opt, F_opt, res = optimize_controls(H0, Hc, psi_i, psi_f, dt, N, u0, lambda_deriv= 0.0003, lambda_boundary= 0.0003, lambda_amp=0.001, amp_max=40.0)
    u_opt, F_opt, res = optimize_controls(H0, Hc, psi_i, psi_f, dt, N, u0,
                                          lambda_deriv= 0.00002,
                                          lambda_boundary= 0.0001,
                                          lambda_amp=0.00001
                                          , amp_max=40.0
                                          , cav_band=(-27.0, 27.0)
                                          , tra_band=(-33.0, 33.0))

    print(f"optimized fidelity: {F_opt:.6f}")
    print(f"Optimizer stopped because: {res.message}")

    #Save the optimized pulse
    np.save(os.path.join(PULSE_DIR, "u_opt.npy"), u_opt)
    print(f"Optimized pulse saved to {PULSE_DIR}/u_opt.npy")



def simulate_trajectory(u, H0, Hc, psi_i, dt, n_c, n_t):
    """
    Propagate the state forard in time and record: 
    - P(n,t): the cavity photon number distribution at each time step
    - <n>(t): the average cavity photon number vs time
    - transmon excited-state population
    """
    N_steps = u.shape[0] # numer of time steps, rows in the control array
    psi = psi_i.copy().astype(complex) # initialize the state vector

    times = np.arange(N_steps+1) * dt  #time points
    n_mean = np.zeros(N_steps+1) #average cavity photon number
    P = np.zeros((n_c, N_steps+1)) # P[n, time index]
    transmon_ex = np.zeros(N_steps + 1)

    #Record inital condition
    for nc in range(n_c):
        p = 0.0
        for nt in range(n_t):
            idx = nt * n_c + nc
            p += np.abs(psi[idx])**2
        P[nc, 0] = p
    #initialize the arrays
    n_mean[0] = np.sum(np.arange(n_c) * P[:, 0]) #average photon number at t=0
    prob_g = np.sum(np.abs(psi[0:n_c])**2) #probability of transmon in ground state
    transmon_ex[0] = 1 - prob_g #probability of transmon in excited state

    for k in range (N_steps):
        #apply one time step
        Uk, _, _ = step_data(H0, Hc, u[k], dt)
        psi = Uk @ psi

        #Calculate cavity photon populution
        for nc in range (n_c):
            p = 0.0
            for nt in range(n_t):
                idx = nt * n_c + nc
                p += np.abs(psi[idx])**2
            P[nc, k+1] = p
        
        n_mean[k+1] = np.sum(np.arange(n_c) * P[:, k+1])

        #track how much the tranmon is excited
        prob_g = np.sum(np.abs(psi[0:n_c])**2)
        transmon_ex[k+1] = 1.0 - prob_g
    
    return times, n_mean, P, transmon_ex

times, n_mean, P, transmon_ex = simulate_trajectory(u_opt, H0, Hc, psi_i, dt, n_c, n_t)
print(f"Maximum ⟨n⟩ reached during the pulse: {np.max(n_mean):.2f}")
print(f"Final ⟨n⟩ (should be close to 6):     {n_mean[-1]:.3f}")
print(f"Max transmon excited population:      {np.max(transmon_ex):.4f} "
      f"(this should be small — the transmon is used virtually)")
print(f"Final transmon excited population:    {transmon_ex[-1]:.6f}")

print("\nTruncation Validation")
n_c_list = [16, 18, 20, 22, 24, 26, 28, 30, 32]
fidelities = evaluate_fidelity_at_truncations(u_opt, n_c_list)

# Optional: print a nice summary
print("\nTruncation validation summary:")
for nc, f in fidelities.items():
    print(f"  n_c={nc:2d}: {f:.6f}")

# Plotting
print("\nMaking plots...")

t_ns = times * 1000 #convert to nanoseconds for x-axis
u_MHz = u_opt / two_pi # convert rad/µs → MHz

#Figure 1: Control waveforms 
fig1, ax1 = plt.subplots(2, 1, figsize = (10, 6), sharex=True)
ax1[0].plot(t_ns[:-1], u_MHz[:, 0], 'b-', label ='Cavity I (Re ε_C)' )
ax1[0].plot(t_ns[:-1], u_MHz[:, 1], 'b--', label='Cavity Q (Im ε_C)')
ax1[0].set_ylabel('Cavity drive (MHz)')
ax1[0].legend()
ax1[0].grid(True, alpha=0.3)

ax1[1].plot(t_ns[:-1], u_MHz[:, 2], 'r-', label='Transmon I (Re ε_T)')
ax1[1].plot(t_ns[:-1], u_MHz[:, 3], 'r--', label='Transmon Q (Im ε_T)')
ax1[1].set_ylabel('Transmon drive (MHz)')
ax1[1].set_xlabel('Time (ns)')
ax1[1].legend()
ax1[1].grid(True, alpha=0.3)

plt.suptitle('GRAPE Control Waveforms for |g,6⟩ Preparation')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'waveforms.png'), dpi=150)
print(f"Saved: {FIG_DIR}/waveforms.png")

#Figure 2: Photon number trajectory
fig2, ax2 = plt.subplots(figsize=(10, 5))
n_plot=17
im = ax2.pcolormesh(t_ns, np.arange(n_plot), P[:n_plot, :], cmap='Blues', shading='nearest')
ax2.set_xlabel('Time (ns)')
ax2.set_ylabel('Photon number n')
ax2.set_title(f'Photon Number Population Trajectory  |  max ⟨n⟩ = {np.max(n_mean):.1f}')

# Colorbar
cbar = plt.colorbar(im, ax=ax2, pad=0.02)
cbar.set_label('Population P(n,t)')

#<n>(t) average photon # graph 
ax2.plot(t_ns, n_mean, color='red', linewidth=2.8, alpha=0.75, label='<n>(t)')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'photon_trajectory.png'), dpi=150)
print(f"Saved: {FIG_DIR}/photon_trajectory.png")

#Figure 3: Selected Fock states
fig3, ax3 = plt.subplots(figsize=(10, 5))
for n in [0, 2, 4, 6, 8, 10, 12]:
    ax3.plot(t_ns, P[n, :], linewidth=1.8, label=f'n = {n}')
ax3.set_xlabel('Time (ns)')
ax3.set_ylabel('Population')
ax3.set_title('Evolution of Individual Fock State Populations')
ax3.legend(ncol=2)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fock_populations.png'), dpi=150)
print(f"Saved: {FIG_DIR}/fock_populations.png")

