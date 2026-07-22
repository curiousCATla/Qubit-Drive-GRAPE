#!/usr/bin/env python3
"""
wigner_viz.py

Wigner function visualization module for the cat-code GRAPE project.

Supports:
- Fock states
- Logical cat states (|±Z_L⟩)
- Final states after applying optimized GRAPE pulses

Plots are styled to resemble figures from Heeres et al. (2017).

Core workflow:
    plot_wigner_from_pulse(pulse, initial_state="cat_plus", ...)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre, gammaln

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.grape_core import make_hamiltonian, step_data, basis_state
from core.cat_code import get_logical_cat_states, embed_in_joint_space


def propagate_pulse(u, H0, Hc, psi0, dt=0.002):
    """Propagate initial joint state through GRAPE pulse u."""
    psi = psi0.copy().astype(complex)
    for uk in u:
        Uk, _, _ = step_data(H0, Hc, uk, dt)
        psi = Uk @ psi
    return psi


# ============================================================
# CORE: WIGNER FUNCTION COMPUTATION (Pure state, Fock basis)
# ============================================================
def compute_wigner(psi, xvec, yvec):
    """
    Compute the Wigner function of a pure state given in the Fock basis.

    Uses the full Cahill-Glauber matrix-element formula (diagonal +
    off-diagonal Fock coherences), which is required to reproduce the
    multi-ring nodal structure of Fock states |n⟩ (Heeres et al. Fig. 1C).
    A diagonal-only formula collapses to a single ring/blob because it
    drops the interference terms between different Fock components and
    forces W(0)=0 for n>0.

    W(α) = (2/π) exp(-2|α|²) [
             Σ_n |c_n|² (-1)^n L_n(4|α|²)
             + 2 Re Σ_{n<m} c_n c_m* (-1)^n √(n!/m!) (2α)^{m-n} L_n^{(m-n)}(4|α|²)
           ]

    eval_genlaguerre uses a numerically stable recurrence (unlike building
    genlaguerre's explicit polynomial coefficients), and factorial ratios
    are computed in log-space via gammaln to avoid overflow for large n_c.
    """
    X, Y = np.meshgrid(xvec, yvec)
    alpha = X + 1j * Y
    r2 = np.abs(alpha) ** 2
    arg = 4 * r2

    idx = np.where(np.abs(psi) > 1e-12)[0]

    W = np.zeros_like(alpha, dtype=float)
    for a_i, n in enumerate(idx):
        cn = psi[n]
        W += (np.abs(cn) ** 2) * ((-1) ** n) * eval_genlaguerre(n, 0, arg)
        for m in idx[a_i + 1:]:
            cm = psi[m]
            k = m - n
            log_ratio = 0.5 * (gammaln(n + 1) - gammaln(m + 1))
            prefactor = ((-1) ** n) * np.exp(log_ratio)
            term = cn * np.conj(cm) * prefactor * (2 * alpha) ** k * eval_genlaguerre(n, k, arg)
            W += 2 * np.real(term)

    W *= (2 / np.pi) * np.exp(-2 * r2)
    return W


# ============================================================
# PLOTTING FUNCTION (Paper-like styling)
# ============================================================
def plot_wigner(W, xvec, yvec, title=None, save_path=None, 
                cmap='RdBu_r', figsize=(7, 6), dpi=200, show=True):
    """
    Plot the Wigner function as a 2D color plot (styled similarly to Heeres et al. 2017).

    Parameters
    ----------
    W : 2D np.ndarray
        Wigner function values
    xvec, yvec : 1D arrays
        Phase space grid
    title : str or None
    save_path : str or None
    cmap : str
        Colormap (default 'RdBu_r' gives blue-negative / white-zero / red-positive)
    """
    fig, ax = plt.subplots(figsize=figsize)

    vmax = np.max(np.abs(W))
    im = ax.pcolormesh(xvec, yvec, W, cmap=cmap, shading='auto',
                       vmin=-vmax, vmax=vmax)

    ax.set_xlabel(r'Re($\alpha$)', fontsize=12)
    ax.set_ylabel(r'Im($\alpha$)', fontsize=12)
    ax.set_aspect('equal')

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'$W(\alpha)$', fontsize=11)

    if title:
        ax.set_title(title, fontsize=13, pad=10)

    ax.grid(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# STATE PREPARATION HELPERS
# ============================================================
def get_initial_joint_state(initial_state, n_c=24, n_t=3, alpha=np.sqrt(3.0), transmon='g'):
    """
    Prepare initial joint transmon-cavity state.
    Returns state vector of length n_t * n_c.

    transmon : 'g' or 'e'
        Which transmon level the cavity state is prepared on. Needed e.g. for
        encoding pulses whose output logical cat state depends on whether the
        transmon started in |g⟩ or |e⟩ (see cat_code.get_encode_state_pairs).
    """
    if transmon == 'g':
        t_level = 0
    elif transmon == 'e':
        t_level = 1
    else:
        raise ValueError(f"Unknown transmon level: {transmon}. Use 'g' or 'e'")

    if initial_state == "vacuum":
        psi = np.zeros(n_t * n_c, dtype=complex)
        psi[t_level * n_c] = 1.0  # |transmon, 0⟩
        return psi

    elif initial_state == "cat_plus":
        psi_cav, _ = get_logical_cat_states(alpha=alpha, n_c=n_c)
        return embed_in_joint_space(psi_cav, n_t=n_t, n_c=n_c, t_level=t_level)

    elif initial_state == "cat_minus":
        _, psi_cav = get_logical_cat_states(alpha=alpha, n_c=n_c)
        return embed_in_joint_space(psi_cav, n_t=n_t, n_c=n_c, t_level=t_level)

    elif initial_state.startswith("fock_"):
        try:
            n = int(initial_state.split("_")[1])
        except:
            raise ValueError(f"Invalid fock state format: {initial_state}. Use 'fock_6'")
        if n >= n_c:
            raise ValueError(f"Fock level {n} >= n_c={n_c}")
        psi = np.zeros(n_t * n_c, dtype=complex)
        psi[t_level * n_c + n] = 1.0  # |transmon, n⟩
        return psi

    else:
        raise ValueError(f"Unknown initial_state: {initial_state}")


def extract_cavity_state(psi_joint, n_c=24, n_t=3):
    """Extract cavity state vector assuming transmon is in |g⟩ (first n_c elements)."""
    return psi_joint[:n_c].copy()


# ============================================================
# HIGH-LEVEL CONVENIENCE FUNCTIONS
# ============================================================
def plot_wigner_fock(n, n_c=24, alpha_max=5.0, n_points=200, 
                     save_path=None, show=True, **kwargs):
    """Quick plot of Wigner function for Fock state |n⟩."""
    psi = np.zeros(n_c, dtype=complex)
    psi[n] = 1.0

    xvec = np.linspace(-alpha_max, alpha_max, n_points)
    W = compute_wigner(psi, xvec, xvec)

    title = f"Wigner function of Fock state |{n}⟩ (n_c={n_c})"
    plot_wigner(W, xvec, xvec, title=title, save_path=save_path, show=show, **kwargs)


def plot_wigner_cat(which='+', alpha=np.sqrt(3.0), n_c=24, 
                    alpha_max=5.0, n_points=200, save_path=None, show=True, **kwargs):
    """Quick plot of Wigner function for logical cat state."""
    psi_plus, psi_minus = get_logical_cat_states(alpha=alpha, n_c=n_c)
    psi = psi_plus if which == '+' else psi_minus

    xvec = np.linspace(-alpha_max, alpha_max, n_points)
    W = compute_wigner(psi, xvec, xvec)

    label = "+Z_L" if which == '+' else "-Z_L"
    title = f"Wigner function of |{label}⟩ (α={alpha:.2f}, n_c={n_c})"
    plot_wigner(W, xvec, xvec, title=title, save_path=save_path, show=show, **kwargs)


def plot_wigner_from_pulse(pulse, initial_state="vacuum", transmon='g', n_c=24, n_t=3,
                           alpha_max=5.0, n_points=200, title=None,
                           save_path=None, show=True, **kwargs):
    """
    Apply an optimized GRAPE pulse to an initial state and plot the Wigner function
    of the final cavity state.

    Parameters
    ----------
    pulse : np.ndarray or str
        Either the pulse array (N, 4) or path to .npy file
    initial_state : str
        One of: "vacuum", "cat_plus", "cat_minus", "fock_N"
    transmon : 'g' or 'e'
        Transmon level the initial_state is prepared on. Some pulses (e.g. an
        encoder) map |g⟩ and |e⟩ initial states to different output cat states.
    """
    # Load pulse if path is given
    if isinstance(pulse, str):
        if not os.path.exists(pulse):
            alt = os.path.join("pulses", pulse)
            if os.path.exists(alt):
                pulse = alt
            else:
                raise FileNotFoundError(f"Could not find pulse: {pulse}")
        u = np.load(pulse)
    else:
        u = pulse

    # Prepare initial state
    psi0 = get_initial_joint_state(initial_state, n_c=n_c, n_t=n_t, transmon=transmon)

    # Propagate
    H0, Hc = make_hamiltonian(n_t, n_c)
    psi_final = propagate_pulse(u, H0, Hc, psi0)

    # Extract cavity state (transmon in |g⟩)
    psi_cavity = extract_cavity_state(psi_final, n_c=n_c, n_t=n_t)

    # Compute Wigner
    xvec = np.linspace(-alpha_max, alpha_max, n_points)
    W = compute_wigner(psi_cavity, xvec, xvec)

    # Auto title
    if title is None:
        title = f"Wigner function after pulse on '{initial_state}' (transmon={transmon}, n_c={n_c})"

    plot_wigner(W, xvec, xvec, title=title, save_path=save_path, show=show, **kwargs)


# ============================================================
# GATE -> PULSE FILE MAP (cold-start Eq.23+24 pulses)
# ============================================================
# filename (in pulses/) -> (label, [(initial_state, transmon, description), ...])
# For U_opt the only relevant input is vacuum; for U_enc it's vacuum on each
# transmon level (which cat state you get depends on transmon level); for
# every other gate (dec + logical X/Y/Z/H/T/I) the inputs are the two cat
# basis states |+Z_L>/|-Z_L> with the transmon in |g> (see cat_code.py's
# get_*_state_pairs factories -- all embed at t_level=0 except get_encode).
WIGNER_GATE_MAP = {
    "u_opt_main.npy": ("U_opt", [
        ("vacuum", "g", "|g,0⟩ → |g,6⟩"),
    ]),
    "u_enc_main.npy": ("U_enc", [
        ("vacuum", "g", "+Z_L after encoding"),
        ("vacuum", "e", "-Z_L after encoding"),
    ]),
    "u_dec_main.npy": ("U_dec", [
        ("cat_plus", "g", "+Z_L → |g,0⟩ (decode)"),
        ("cat_minus", "g", "-Z_L → |e,0⟩ (decode)"),
    ]),
    "u_X_main.npy": ("U_X", [
        ("cat_plus", "g", "+Z_L after X"),
        ("cat_minus", "g", "-Z_L after X"),
    ]),
    "u_Y_main.npy": ("U_Y", [
        ("cat_plus", "g", "+Z_L after Y"),
        ("cat_minus", "g", "-Z_L after Y"),
    ]),
    "u_Z_main.npy": ("U_Z", [
        ("cat_plus", "g", "+Z_L after Z"),
        ("cat_minus", "g", "-Z_L after Z"),
    ]),
    "u_H_main.npy": ("U_H", [
        ("cat_plus", "g", "+Z_L after H"),
        ("cat_minus", "g", "-Z_L after H"),
    ]),
    "u_T_main.npy": ("U_T", [
        ("cat_plus", "g", "+Z_L after T"),
        ("cat_minus", "g", "-Z_L after T"),
    ]),
    "u_I_main.npy": ("U_I", [
        ("cat_plus", "g", "+Z_L after I"),
        ("cat_minus", "g", "-Z_L after I"),
    ]),
}

# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    print("=== Wigner Visualization Module ===")
    print("Rendering final-state Wigner functions for every cold-start pulse in WIGNER_GATE_MAP...")

    os.makedirs("wigner", exist_ok=True)

    for filename, (label, entries) in WIGNER_GATE_MAP.items():
        pulse_path = os.path.join("pulses", filename)
        for initial_state, transmon, desc in entries:
            safe_name = f"{label}_{initial_state}_{transmon}"
            plot_wigner_from_pulse(
                pulse=pulse_path,
                initial_state=initial_state,
                transmon=transmon,
                n_c=26,
                alpha_max=5.5,
                title=f"{label}: {desc}",
                save_path=os.path.join("wigner", f"{safe_name}.png"),
                show=False,
            )

    print("Done.")
