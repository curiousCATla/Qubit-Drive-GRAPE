#!/usr/bin/env python3
"""
pulse_viz.py

Reusable visualization tools for GRAPE-optimized pulses
(U_enc, U_dec, logical gates, etc.).

Produces:
- Drive strength vs Time (I/Q waveforms)
- Pulse FFT (frequency spectrum)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from grape_core import two_pi, make_hamiltonian, step_data  # if you defined it there, otherwise use 2*np.pi
from cat_code import (
    get_logical_X_state_pairs,
    get_logical_Y_state_pairs,
    get_logical_Z_state_pairs,
    get_logical_H_state_pairs,
    get_logical_T_state_pairs,
    get_identity_state_pairs,
    get_encode_state_pairs,
    get_decode_state_pairs,
)
from compare_pulses import get_g6_state_pairs

def plot_pulse_waveforms(u, dt=0.002, title="Pulse Waveforms", save_path=None, show=True):
    """
    Plot I/Q drive amplitudes vs time for cavity and transmon.
    
    Parameters
    ----------
    u : np.ndarray
        Pulse of shape (N, 4)  [C_I, C_Q, T_I, T_Q]
    dt : float
        Time step in µs (default 0.002 = 2 ns)
    title : str
    save_path : str or None
        If provided, saves the figure
    show : bool
        Whether to call plt.show()
    """
    t_ns = np.arange(u.shape[0]) * dt * 1000  # convert to ns
    u_MHz = u / two_pi if 'two_pi' in globals() else u / (2 * np.pi)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Cavity drives
    axes[0].plot(t_ns, u_MHz[:, 0], 'r-', label='Cavity I (Re ε_C)')
    axes[0].plot(t_ns, u_MHz[:, 1], 'r--', label='Cavity Q (Im ε_C)')
    axes[0].set_ylabel('Cavity Drive (MHz)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Transmon drives
    axes[1].plot(t_ns, u_MHz[:, 2], 'b-', label='Transmon I (Re ε_T)')
    axes[1].plot(t_ns, u_MHz[:, 3], 'b--', label='Transmon Q (Im ε_T)')
    axes[1].set_ylabel('Transmon Drive (MHz)')
    axes[1].set_xlabel('Time (ns)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_pulse_spectrum(u, dt=0.002, title="Pulse Spectrum",
                        max_freq_mhz=50, save_path=None, show=True):
    """
    Plot FFT magnitude spectrum of the complex drive envelope ε(t) = I(t) + i·Q(t)
    for the cavity and transmon channels. FFTing I and Q separately (as real-valued
    signals) always yields a Hermitian-symmetric (mirrored) spectrum regardless of
    the pulse content; the physically meaningful spectrum is that of the complex
    envelope, which is generally asymmetric.
    Frequency axis is now correctly scaled in MHz.
    """
    N = u.shape[0]

    # Correct frequency axis in MHz (two-sided, so negative frequencies show too)
    dt_sec = dt * 1e-6                    # convert dt from µs → seconds
    freqs_mhz = np.fft.fftshift(np.fft.fftfreq(N, d=dt_sec)) / 1e6   # frequency in MHz

    u_MHz = u / (2 * np.pi)               # controls in MHz

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Cavity spectrum
    eps_C = u_MHz[:, 0] + 1j * u_MHz[:, 1]
    spec_C = np.abs(np.fft.fftshift(np.fft.fft(eps_C)))
    axes[0].plot(freqs_mhz, spec_C, '-', color='red', label=r'$\varepsilon_C$')
    axes[0].set_ylabel(r'Cavity Spectrum $|\varepsilon_C(f)|$ (a.u.)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Cavity Drive Spectrum')
    axes[0].set_xlim(-max_freq_mhz, max_freq_mhz)

    # Transmon spectrum
    eps_T = u_MHz[:, 2] + 1j * u_MHz[:, 3]
    spec_T = np.abs(np.fft.fftshift(np.fft.fft(eps_T)))
    axes[1].plot(freqs_mhz, spec_T, '-', color='blue', label=r'$\varepsilon_T$')
    axes[1].set_ylabel(r'Transmon Spectrum $|\varepsilon_T(f)|$ (a.u.)')
    axes[1].set_xlabel('Frequency (MHz)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Transmon Drive Spectrum')
    axes[1].set_xlim(-max_freq_mhz, max_freq_mhz)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def simulate_trajectory(u, H0, Hc, psi_i, dt, n_c, n_t):
    """
    Propagate the state forward in time and record:
    - P(n,t): the cavity photon number distribution at each time step
    - <n>(t): the average cavity photon number vs time
    - transmon excited-state population
    """
    N_steps = u.shape[0]  # number of time steps, rows in the control array
    psi = psi_i.copy().astype(complex)  # initialize the state vector

    times = np.arange(N_steps + 1) * dt  # time points
    n_mean = np.zeros(N_steps + 1)  # average cavity photon number
    P = np.zeros((n_c, N_steps + 1))  # P[n, time index]
    transmon_ex = np.zeros(N_steps + 1)

    # Record initial condition
    for nc in range(n_c):
        p = 0.0
        for nt in range(n_t):
            idx = nt * n_c + nc
            p += np.abs(psi[idx]) ** 2
        P[nc, 0] = p
    n_mean[0] = np.sum(np.arange(n_c) * P[:, 0])  # average photon number at t=0
    prob_g = np.sum(np.abs(psi[0:n_c]) ** 2)  # probability of transmon in ground state
    transmon_ex[0] = 1 - prob_g  # probability of transmon in excited state

    for k in range(N_steps):
        # apply one time step
        Uk, _, _ = step_data(H0, Hc, u[k], dt)
        psi = Uk @ psi

        # Calculate cavity photon population
        for nc in range(n_c):
            p = 0.0
            for nt in range(n_t):
                idx = nt * n_c + nc
                p += np.abs(psi[idx]) ** 2
            P[nc, k + 1] = p

        n_mean[k + 1] = np.sum(np.arange(n_c) * P[:, k + 1])

        # track how much the transmon is excited
        prob_g = np.sum(np.abs(psi[0:n_c]) ** 2)
        transmon_ex[k + 1] = 1.0 - prob_g

    return times, n_mean, P, transmon_ex


def plot_photon_trajectory(u, psi_i_list, labels=None, dt=0.002, n_c=24, n_t=3,
                            n_plot=None, title="Photon Number Trajectory",
                            save_path=None, show=True):
    """
    Plot the cavity photon-number population P(n,t) (heatmap) with <n>(t)
    overlaid, one panel per initial state in psi_i_list (e.g. the two
    logical basis states a gate is optimized against).
    """
    H0, Hc = make_hamiltonian(n_t, n_c)

    if labels is None:
        labels = [f"Input {i+1}" for i in range(len(psi_i_list))]
    n_plot = n_plot or min(n_c, 17)

    fig, axes = plt.subplots(len(psi_i_list), 1, figsize=(10, 4.5 * len(psi_i_list)),
                              sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, psi_i, label in zip(axes, psi_i_list, labels):
        times, n_mean, P, _ = simulate_trajectory(u, H0, Hc, psi_i, dt, n_c, n_t)
        t_ns = times * 1000  # convert to ns

        im = ax.pcolormesh(t_ns, np.arange(n_plot), P[:n_plot, :], cmap='Blues', shading='nearest')
        ax.plot(t_ns, n_mean, color='red', linewidth=2.2, alpha=0.8, label='⟨n⟩(t)')
        ax.set_ylabel('Photon number n')
        ax.set_title(f'{label}   |   max ⟨n⟩ = {np.max(n_mean):.2f}')
        ax.legend(loc='upper right')

        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('P(n,t)')

    axes[-1].set_xlabel('Time (ns)')
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def analyze_pulse(pulse_path, name="Pulse", dt=0.002, out_dir="figures", show=False):
    """
    Convenience function: load a pulse and generate both plots.
    """
    u = np.load(pulse_path)
    print(f"Loaded {pulse_path} | shape = {u.shape}")

    os.makedirs(out_dir, exist_ok=True)
    plot_pulse_waveforms(u, dt=dt, title=f"{name} - Waveforms",
                         save_path=os.path.join(out_dir, f"{name}_waveforms.png"), show=show)
    plot_pulse_spectrum(u, dt=dt, title=f"{name} - Spectrum",
                        save_path=os.path.join(out_dir, f"{name}_spectrum.png"), show=show)


# filename (in pulses/) -> (label, get_state_pairs factory, per-pair trajectory labels)
PULSE_GATE_MAP = {
    "u_opt_mt.npy": ("U_opt", get_g6_state_pairs, ["|g,0⟩ → |g,6⟩"]),
    "u_enc_mt.npy": ("U_enc", get_encode_state_pairs, ["|g,0⟩ → |+Z_L⟩", "|e,0⟩ → |-Z_L⟩"]),
    "u_dec_mt.npy": ("U_dec", get_decode_state_pairs, ["|+Z_L⟩ → |g,0⟩", "|-Z_L⟩ → |e,0⟩"]),
    "u_X_mt.npy":   ("U_X",   get_logical_X_state_pairs, ["|+Z_L⟩ → |-Z_L⟩", "|-Z_L⟩ → |+Z_L⟩"]),
    "u_Y_mt.npy":   ("U_Y",   get_logical_Y_state_pairs, ["|+Z_L⟩ → -i|-Z_L⟩", "|-Z_L⟩ → +i|+Z_L⟩"]),
    "u_Z_mt.npy":   ("U_Z",   get_logical_Z_state_pairs, ["|+Z_L⟩ → |+Z_L⟩", "|-Z_L⟩ → -|-Z_L⟩"]),
    "u_H_mt.npy":   ("U_H",   get_logical_H_state_pairs, ["|+Z_L⟩ → (|+Z_L⟩+|-Z_L⟩)/√2", "|-Z_L⟩ → (|+Z_L⟩-|-Z_L⟩)/√2"]),
    "u_T_mt.npy":   ("U_T",   get_logical_T_state_pairs, ["|+Z_L⟩ → |+Z_L⟩", "|-Z_L⟩ → e^{iπ/4}|-Z_L⟩"]),
    "u_I_mt.npy":   ("U_I",   get_identity_state_pairs, ["|+Z_L⟩ → |+Z_L⟩", "|-Z_L⟩ → |-Z_L⟩"]),
}

# Example usage
if __name__ == "__main__":
    # You can run this file directly with:
    # python pulse_viz.py
    os.makedirs("figures", exist_ok=True)

    for filename, (label, factory, pair_labels) in PULSE_GATE_MAP.items():
        pulse_path = os.path.join("pulses", filename)
        analyze_pulse(pulse_path, name=label)

        u = np.load(pulse_path)
        psi_i_list = [p[0] for p in factory(n_c=24, n_t=3)]
        plot_photon_trajectory(
            u, psi_i_list, labels=pair_labels, n_c=24, n_t=3,
            title=f"{label} — Photon Number Trajectory",
            save_path=os.path.join("figures", f"{label}_photon_trajectory.png"),
            show=False,
        )