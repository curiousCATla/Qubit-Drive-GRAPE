import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.cat_code import *
from core.optimizer import optimize_multi_state_pulse
import numpy as np
from core.grape_core import step_data, make_hamiltonian



def propagate_pulse(u, H0, Hc, psi0, dt):
    """Propagate an initial state through a full GRAPE pulse."""
    psi = psi0.copy().astype(complex)
    for uk in u:
        Uk, _, _ = step_data(H0, Hc, uk, dt)
        psi = Uk @ psi
    return psi

def check_composition(u_enc, u_dec, n_c_list=[20, 24, 28], dt=0.002):
    results = {}
    for nc in n_c_list:
        H0, Hc = make_hamiltonian(n_t=2, n_c=nc)

        # Get the encode initial states (transmon comp subspace)
        state_pairs = get_encode_state_pairs(nc)   # reuse the factory
        init_states = [p[0] for p in state_pairs]  # |g,0> and |e,0>

        fidelities = []
        for psi_init in init_states:
            # Encode
            psi_after_enc = propagate_pulse(u_enc, H0, Hc, psi_init, dt)
            # Decode
            psi_final = propagate_pulse(u_dec, H0, Hc, psi_after_enc, dt)
            # State fidelity back to original
            fid = np.abs(np.vdot(psi_init, psi_final))**2
            fidelities.append(fid)

        avg_fid = np.mean(fidelities)
        results[nc] = {'per_state': fidelities, 'avg': avg_fid}
        print(f"  n_c={nc:2d}:  |g,0>→...→|g,0>  F={fidelities[0]:.6f}   "
              f"|e,0>→...→|e,0>  F={fidelities[1]:.6f}   avg={avg_fid:.6f}")

    overall_avg = np.mean([r['avg'] for r in results.values()])
    print(f"\nOverall average round-trip state fidelity: {overall_avg:.6f}")
    return results


if __name__ == "__main__":



    # ============================================================
    # Identity operation (reference / sanity check)
    # ============================================================
    print("\n" + "=" * 70)

    print("=" * 70)

    from core.cat_code import get_identity_state_pairs

   

    u_H, info_H = optimize_multi_state_pulse(
        get_state_pairs=get_logical_H_state_pairs,
        trunc_list=[22, 24, 26],
        warm_start_amp=np.pi,
        save_path="pulses/u_H_logical_v2.npy",
        penalties={'deriv': 0.00001, 'boundary': 0.00002, 'amp': 0.00001, 'amp_max': 40.0},
        maxiter=1500,
        cav_band=(-27.0, 27.0),
        tra_band=(-33.0, 33.0),
        hard_amp_limit=50.0,
        verbose=True
    )

    print("\n--- Identity Validation ---")
    validate_pulse_truncations(
        u=u_H,
        get_targets_func=get_logical_H_state_pairs,
        title="Identity Full Truncation Validation"
    )

    print("\nIdentity optimization complete. Saved: u_H_logical_v2.npy")