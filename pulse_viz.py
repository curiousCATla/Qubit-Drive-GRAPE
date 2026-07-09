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
from grape_core import two_pi  # if you defined it there, otherwise use 2*np.pi

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


# Example usage
if __name__ == "__main__":
    # You can run this file directly with:
    # python pulse_viz.py
    analyze_pulse("pulses/u_enc_refined_t3v2.npy", name="U_enc")
    analyze_pulse("pulses/u_dec_refined_t3v2.npy", name="U_dec")
    analyze_pulse("pulses/u_X_refined_t3v2.npy", name="U_X")
    analyze_pulse("pulses/u_Z_refined_t3v2.npy", name="U_Z")
    analyze_pulse("pulses/u_H_refined_t3v2.npy", name="U_H")
    analyze_pulse("pulses/u_T_refined_t3v2.npy", name="U_T")
    analyze_pulse("pulses/u_I_refined_t3v2.npy", name="U_I")
    
    # analyze_pulse("pulses/u_X_logical_v1.npy", name="Logical_X")