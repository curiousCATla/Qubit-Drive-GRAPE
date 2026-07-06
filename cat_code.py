#!/usr/bin/env python3
import numpy as np
from scipy.special import gammaln

def get_logical_cat_states(alpha=np.sqrt(3.0), n_c=24, dtype=complex):
    """
    Return normalized |+Z_L> and |-Z_L> (cavity only) using log-space computation
    for numerical stability. Matches eq. (2) and (3) in Heeres et al. 2017.
    """
    psi_plus = np.zeros(n_c, dtype=dtype)
    psi_minus = np.zeros(n_c, dtype=dtype)
    
    log_alpha = np.log(alpha) if alpha > 0 else 0.0
    
    # |+Z_L>: n = 0, 4, 8, 12, ...
    k = 0
    while True:
        n = 4 * k
        if n >= n_c:
            break
        # log(coeff) = n*log(α) - 0.5 * log(n!)
        log_coeff = n * log_alpha - 0.5 * gammaln(n + 1)
        psi_plus[n] = np.exp(log_coeff)
        k += 1
    
    # |-Z_L>: n = 2, 6, 10, 14, ...
    k = 0
    while True:
        n = 4 * k + 2
        if n >= n_c:
            break
        log_coeff = n * log_alpha - 0.5 * gammaln(n + 1)
        psi_minus[n] = np.exp(log_coeff)
        k += 1
    
    # Normalize
    norm_p = np.linalg.norm(psi_plus)
    norm_m = np.linalg.norm(psi_minus)
    if norm_p > 0:
        psi_plus /= norm_p
    if norm_m > 0:
        psi_minus /= norm_m
    
    return psi_plus, psi_minus

def embed_in_joint_space(psi_cavity, n_t = 2, n_c = 24, t_level = 0):
    """
    Embeds a cavity state psi (length n_c) into the joint cavity-transmon space of dimension n_c * n_t.
    Ordering: transmon is the slow index: index = n_c * transmon_level + cavity_level
    """
    psi_joint = np.zeros(n_c * n_t, dtype=complex)
    start = t_level * n_c
    psi_joint[start:start+n_c] = psi_cavity
    return psi_joint

def get_encode_targets(n_c=24, n_t=3, alpha=np.sqrt(3.0)):
    """
    Return the four joint states needed for U_enc optimization:
    psi_i_g, psi_f_plus, psi_i_e, psi_f_minus
    """
    psi_plus_cav, psi_minus_cav = get_logical_cat_states(alpha=alpha, n_c=n_c)
    # Initial states (cavity in vacuum)
    psi_i_g = np.zeros(n_t * n_c, dtype=complex)   # |g, 0>
    psi_i_g[0] = 1.0
    
    psi_i_e = np.zeros(n_t * n_c, dtype=complex)   # |e, 0>
    psi_i_e[n_c] = 1.0                             # transmon excited = index n_c

    # Target states (transmon back in |g>, cavity in logical state)
    psi_f_plus  = np.zeros(n_t * n_c, dtype=complex)
    psi_f_plus[0 : n_c] = psi_plus_cav             # |g> ⊗ |+Z_L>
    
    psi_f_minus = np.zeros(n_t * n_c, dtype=complex)
    psi_f_minus[0 : n_c] = psi_minus_cav           # |g> ⊗ |-Z_L>
    
    return psi_i_g, psi_f_plus, psi_i_e, psi_f_minus


def get_encode_state_pairs(n_c=24, n_t=3, alpha=np.sqrt(3.0)):
    """
    Factory for U_enc optimization.
    Maps transmon computational subspace states to logical cat states
    (cavity encoded, transmon reset to |g>).
    """
    psi_i_g, psi_f_plus, psi_i_e, psi_f_minus = get_encode_targets(n_c=n_c, n_t=n_t, alpha=alpha)
    return [(psi_i_g, psi_f_plus), (psi_i_e, psi_f_minus)]


def get_decode_state_pairs(n_c=24, n_t=3, alpha=np.sqrt(3.0)):
    """
    Factory for U_dec optimization (reverse of U_enc).
    Maps logical cat states back to transmon computational subspace.
    This makes U_dec the approximate inverse of U_enc on the logical subspace.
    """
    psi_i_g, psi_f_plus, psi_i_e, psi_f_minus = get_encode_targets(n_c=n_c, n_t=n_t, alpha=alpha)
    # Reverse the mapping: logical cat → original transmon states (cavity reset toward |0>)
    return [(psi_f_plus, psi_i_g), (psi_f_minus, psi_i_e)]


def get_logical_X_state_pairs(n_c=24, n_t=3, alpha=np.sqrt(3.0)):
    """
    Factory for logical X (bit-flip) gate on the even 4-component cat code.
    Swaps |+Z_L⟩ ↔ |-Z_L⟩ while keeping the transmon in |g⟩.
    Both initial and target states live in the even-parity logical subspace.
    """
    psi_plus_cav, psi_minus_cav = get_logical_cat_states(alpha=alpha, n_c=n_c)
    
    # Embed with transmon in ground state for both
    psi_g_plus  = embed_in_joint_space(psi_plus_cav,  n_t=n_t, n_c=n_c, t_level=0)
    psi_g_minus = embed_in_joint_space(psi_minus_cav, n_t=n_t, n_c=n_c, t_level=0)
    
    # Logical X: swap the two logical states
    return [(psi_g_plus, psi_g_minus), (psi_g_minus, psi_g_plus)]


def get_logical_Z_state_pairs(n_c=24, n_t=3, alpha=np.sqrt(3.0)):
    """
    Factory for Logical Z (phase-flip) gate on the even 4-component cat code.
    Applies |+Z_L⟩ → |+Z_L⟩ and |-Z_L⟩ → -|-Z_L⟩ (relative phase of π).
    """
    psi_plus_cav, psi_minus_cav = get_logical_cat_states(alpha=alpha, n_c=n_c)

    psi_g_plus  = embed_in_joint_space(psi_plus_cav,  n_t=n_t, n_c=n_c, t_level=0)
    psi_g_minus = embed_in_joint_space(psi_minus_cav, n_t=n_t, n_c=n_c, t_level=0)
    
    # Logical Z: relative phase of π on the |-Z_L⟩ state
    return [(psi_g_plus,  psi_g_plus), 
            (psi_g_minus, -psi_g_minus)]


def get_logical_H_state_pairs(n_c=24, n_t=3, alpha=np.sqrt(3.0)):
    """
    Factory for Logical Hadamard gate on the even 4-component cat code.
    |+Z_L⟩ → (| +Z_L⟩ + | -Z_L⟩)/√2
    |-Z_L⟩ → (| +Z_L⟩ - | -Z_L⟩)/√2
    """
    psi_plus_cav, psi_minus_cav = get_logical_cat_states(alpha=alpha, n_c=n_c)

    psi_g_plus  = embed_in_joint_space(psi_plus_cav,  n_t=n_t, n_c=n_c, t_level=0)
    psi_g_minus = embed_in_joint_space(psi_minus_cav, n_t=n_t, n_c=n_c, t_level=0)
    
    # Logical Hadamard
    psi_H_plus  = (psi_g_plus + psi_g_minus) / np.sqrt(2)
    psi_H_minus = (psi_g_plus - psi_g_minus) / np.sqrt(2)
    
    return [(psi_g_plus, psi_H_plus), (psi_g_minus, psi_H_minus)]


def get_logical_T_state_pairs(n_c=24, n_t=3, alpha=np.sqrt(3.0)):
    """
    Factory for Logical T (π/8 phase) gate on the even 4-component cat code.
    |+Z_L⟩ → |+Z_L⟩
    |-Z_L⟩ → e^{iπ/4} |-Z_L⟩
    """
    psi_plus_cav, psi_minus_cav = get_logical_cat_states(alpha=alpha, n_c=n_c)

    psi_g_plus  = embed_in_joint_space(psi_plus_cav,  n_t=n_t, n_c=n_c, t_level=0)
    psi_g_minus = embed_in_joint_space(psi_minus_cav, n_t=n_t, n_c=n_c, t_level=0)
    
    phase = np.exp(1j * np.pi / 4)
    return [(psi_g_plus,  psi_g_plus), 
            (psi_g_minus, phase * psi_g_minus)]


def get_logical_Y_state_pairs(n_c=24, n_t=3, alpha=np.sqrt(3.0)):
    """
    Logical Y gate on the even 4-component cat code.
    |+Z_L⟩ → -i |-Z_L⟩
    |-Z_L⟩ → +i |+Z_L⟩
    (Consistent with Y = iXZ up to global phase)
    """
    psi_plus_cav, psi_minus_cav = get_logical_cat_states(alpha=alpha, n_c=n_c)

    psi_g_plus  = embed_in_joint_space(psi_plus_cav,  n_t=n_t, n_c=n_c, t_level=0)
    psi_g_minus = embed_in_joint_space(psi_minus_cav, n_t=n_t, n_c=n_c, t_level=0)
    
    # Logical Y with standard phases
    return [
        (psi_g_plus,  -1j * psi_g_minus),
        (psi_g_minus,  1j * psi_g_plus)
    ]


def get_identity_state_pairs(n_c=24, n_t=3, alpha=np.sqrt(3.0)):
    """
    Identity operation on the logical cat qubit (reference / sanity check).
    Maps each logical state to itself.
    """
    psi_plus_cav, psi_minus_cav = get_logical_cat_states(alpha=alpha, n_c=n_c)

    psi_g_plus  = embed_in_joint_space(psi_plus_cav,  n_t=n_t, n_c=n_c, t_level=0)
    psi_g_minus = embed_in_joint_space(psi_minus_cav, n_t=n_t, n_c=n_c, t_level=0)
    
    return [
        (psi_g_plus,  psi_g_plus),
        (psi_g_minus, psi_g_minus)
    ]


def validate_pulse_truncations(
    u,
    get_targets_func,
    trunc_range=range(18, 31, 2),
    n_t=3,
    dt=0.002,
    title="Truncation Validation"
):
    """
    Function to evaluate a pulse across many cavity truncations.
    
    Parameters
    ----------
    u : np.ndarray
        Optimized control pulse of shape (N, 4)
    get_targets_func : callable
        A function that takes `n_c` and returns either:
        - list of (psi_i, psi_f) pairs   (preferred, matches optimizer factory)
        - or (psi_i_list, psi_f_list) tuple
        Example: get_state_pairs or get_encode_targets wrapped in factory
    trunc_range : range or list
        Which truncations to test (e.g. range(18, 31, 2))
    n_t, dt : int, float
        Transmon levels and time step
    title : str
        Title to print in the report
    """
    from grape_core import make_hamiltonian, fidelity_multi_state
    import numpy as np

    print(f"\n{'='*55}")
    print(f"{title}")
    print(f"{'='*55}")

    results = {}
    for nc in trunc_range:
        H0_t, Hc_t = make_hamiltonian(n_t, nc)
        targets = get_targets_func(n_c=nc, n_t=n_t)
        # Support both formats:
        # 1. list of (psi_i, psi_f) pairs  (used by optimizer factory & get_state_pairs)
        # 2. (psi_i_list, psi_f_list) tuple of lists (legacy / direct get_encode_targets)
        if isinstance(targets, (list, tuple)) and len(targets) > 0 and isinstance(targets[0], (list, tuple)) and len(targets[0]) == 2:
            # list of pairs format
            psi_i_list = [p[0] for p in targets]
            psi_f_list = [p[1] for p in targets]
        else:
            # assume already (psi_i_list, psi_f_list)
            psi_i_list, psi_f_list = targets

        F_t, _ = fidelity_multi_state(u, H0_t, Hc_t, psi_i_list, psi_f_list, dt, want_grad=False)
        results[nc] = F_t
        print(f"  n_c={nc:2d}: F = {F_t:.6f}")

    print(f"{'='*55}\n")
    return results


def verify_cat_states(psi_plus, psi_minus, alpha=np.sqrt(3.0), n_c=24, tol=1e-10):
    """
    Comprehensive verification of the logical cat states.
    Call this right after you generate psi_plus_cav and psi_minus_cav.
    """
    print("=" * 60)
    print("VERIFICATION OF LOGICAL CAT STATES (|α| = {:.3f}, n_c = {})".format(alpha, n_c))
    print("=" * 60)
    
    # 1. Norms
    norm_p = np.linalg.norm(psi_plus)
    norm_m = np.linalg.norm(psi_minus)
    print(f"\n1. Norms after normalization:")
    print(f"   ||+Z_L|| = {norm_p:.12f}   (should be 1.0)")
    print(f"   ||-Z_L|| = {norm_m:.12f}   (should be 1.0)")
    
    # 2. Orthogonality
    overlap = np.vdot(psi_plus, psi_minus)
    print(f"\n2. Overlap <+Z_L | -Z_L> = {overlap:.3e}   (should be ~0)")
    
    # 3. Photon number parity (should be even for both)
    n = np.arange(n_c)
    parity_plus  = np.sum(np.abs(psi_plus)**2  * (-1)**n)
    parity_minus = np.sum(np.abs(psi_minus)**2 * (-1)**n)
    print(f"\n3. Photon number parity expectation value:")
    print(f"   <P> for +Z_L = {parity_plus:.10f}   (should be +1)")
    print(f"   <P> for -Z_L = {parity_minus:.10f}  (should be +1)")
    
    # 4. Support on even photon numbers only
    odd_pop_plus  = np.sum(np.abs(psi_plus[1::2])**2)
    odd_pop_minus = np.sum(np.abs(psi_minus[1::2])**2)
    print(f"\n4. Total population on odd photon numbers:")
    print(f"   +Z_L odd population = {odd_pop_plus:.3e}   (should be ~0)")
    print(f"   -Z_L odd population = {odd_pop_minus:.3e}  (should be ~0)")
    
    # 5. Mod-4 structure (correct subspaces)
    pop_0mod4_plus  = np.sum(np.abs(psi_plus[0::4])**2)
    pop_2mod4_minus = np.sum(np.abs(psi_minus[2::4])**2)
    print(f"\n5. Population in correct mod-4 subspace:")
    print(f"   +Z_L in n≡0 (mod 4) = {pop_0mod4_plus:.6f}   (should be ~1)")
    print(f"   -Z_L in n≡2 (mod 4) = {pop_2mod4_minus:.6f}  (should be ~1)")
    
    # 6. Mean photon number
    n_mean_plus  = np.sum(n * np.abs(psi_plus)**2)
    n_mean_minus = np.sum(n * np.abs(psi_minus)**2)
    print(f"\n6. Mean photon number <n>:")
    print(f"   <n> for +Z_L ≈ {n_mean_plus:.3f}")
    print(f"   <n> for -Z_L ≈ {n_mean_minus:.3f}")
    
    # 7. Dominant Fock components (should be low n for α=√3)
    print(f"\n7. Largest Fock components (first 5 non-zero):")
    idx_p = np.argsort(np.abs(psi_plus))[::-1][:5]
    idx_m = np.argsort(np.abs(psi_minus))[::-1][:5]
    print("   +Z_L:", [(int(i), f"{np.abs(psi_plus[i]):.4f}") for i in idx_p if np.abs(psi_plus[i]) > 1e-8])
    print("   -Z_L:", [(int(i), f"{np.abs(psi_minus[i]):.4f}") for i in idx_m if np.abs(psi_minus[i]) > 1e-8])
    
    print("\n" + "=" * 60)
    print("Verification complete. All values should match expectations above.")
    print("=" * 60)
