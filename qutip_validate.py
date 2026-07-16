#!/usr/bin/env python3
"""
qutip_validate.py

Independent cross-check of a saved GRAPE pulse using QuTiP's own ODE-based
Schrodinger-equation solver (sesolve), instead of grape_core's hand-rolled
propagator (fidelity_grad, which exponentiates each step via eigh).

Why this is worth doing: grape_core.fidelity_grad is the function the
optimizer maximizes AND the function that reports "final fidelity". If it had
a bug (wrong operator ordering, a missing factor of 2, wrong dt convention),
the optimizer would happily converge and the fidelity report would look
great -- nothing inside grape_core.py can catch that. QuTiP is a separately
implemented, widely-validated propagator, so agreement between the two is
real evidence the physics is right, not just that the code is internally
consistent.

Start here: pulses/u_opt_mt.npy, the |g,0> -> |g,6> cavity Fock-state
preparation pulse (N=250 steps, 4 channels: cavity I/Q, transmon I/Q).

Usage:
    python qutip_validate.py
"""
import os
import numpy as np
import qutip as qt

from grape_core import chi, Kerr, chip, alpha, fidelity_multi_state, make_hamiltonian, basis_state
from compare_pulses import get_g6_state_pairs
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

PULSE_DIR = "pulses"
DT = 0.002              # microseconds, matches grape_core.dt
N_T = 3                 # transmon levels every pulse was trained with
TRUNC_LIST = [22, 24, 26, 28]   # 22/24/26 = trained on; 28 = generalization check

# filename -> (label, get_state_pairs factory) -- same factories the
# optimizer and compare_pulses.py use, so the *targets* are shared, but the
# *propagator* (qutip.sesolve below) is independent of grape_core's.
PULSE_MAP = {
    "u_opt_mt.npy": ("g0->g6 prep", get_g6_state_pairs),
    "u_X_mt.npy":   ("X",           get_logical_X_state_pairs),
    "u_Y_mt.npy":   ("Y",           get_logical_Y_state_pairs),
    "u_Z_mt.npy":   ("Z",           get_logical_Z_state_pairs),
    "u_H_mt.npy":   ("H",           get_logical_H_state_pairs),
    "u_T_mt.npy":   ("T",           get_logical_T_state_pairs),
    "u_I_mt.npy":   ("I",           get_identity_state_pairs),
    "u_enc_mt.npy": ("U_enc",       get_encode_state_pairs),
    "u_dec_mt.npy": ("U_dec",       get_decode_state_pairs),
}


def build_qutip_ops(n_t, n_c):
    """
    QuTiP-native rebuild of grape_core.make_ops -- written from scratch
    (not by calling grape_core) so it's a genuine second implementation.

    Same convention as grape_core.basis_state: transmon is the OUTER (slow)
    tensor factor, cavity is the INNER (fast) factor, i.e. index(|t,c>) =
    t*n_c + c. qt.tensor(X, Y) stacks Hilbert spaces in argument order, so
    qt.tensor(qeye(n_t), a) puts the transmon identity outermost -- matching
    np.kron(np.eye(n_t), a) in grape_core.make_ops.
    """
    a = qt.destroy(n_c)   # cavity lowering operator
    b = qt.destroy(n_t)   # transmon lowering operator
    A = qt.tensor(qt.qeye(n_t), a)   # cavity op, lifted to joint space
    B = qt.tensor(b, qt.qeye(n_c))   # transmon op, lifted to joint space
    return A, B


def build_qutip_hamiltonian(n_t, n_c):
    A, B = build_qutip_ops(n_t, n_c)
    Ad, Bd = A.dag(), B.dag()
    nA, nB = Ad * A, Bd * B

    H0 = chi * (nA * nB) + (Kerr / 2) * (Ad * Ad * A * A) + (chip / 2) * (nB * (Ad * Ad * A * A))
    if n_t >= 3:
        H0 += (alpha / 2) * (Bd * Bd * B * B)

    Hc = [A + Ad, 1j * (A - Ad), B + Bd, 1j * (B - Bd)]
    return H0, Hc


def qutip_final_state(u, n_t, n_c, psi0, dt=DT):
    """
    Propagate psi0 under the piecewise-constant control sequence u (shape
    [N,4]) using qutip.sesolve, and return the final state ket.

    The pulse is constant on each interval [k*dt, (k+1)*dt) -- exactly a
    step function -- so each control channel is built as a qutip
    Coefficient with order=0 (zero-order hold: sample value held constant until the next sample) interpolation,
    which reproduces that convention exactly rather than approximating it with a spline
    """
    N = u.shape[0]
    H0, Hc = build_qutip_hamiltonian(n_t, n_c)

    tlist = np.arange(N + 1) * dt     # N+1 interval edges: 0, dt, 2dt, ..., N*dt
    terms = [H0]
    for j in range(4):
        vals = np.append(u[:, j], u[-1, j])   # pad to len(tlist); last entry never affects the trajectory
        coeff = qt.coefficient(vals.astype(complex), tlist=tlist, order=0) 
        #this function turns the array of values into a callable function that returns the value at any time t, using zero-order hold interpolation coeff(t)
        terms.append([Hc[j], coeff])
    H = qt.QobjEvo(terms) #Combine the static and time-dependent parts into a single time-dependent Hamiltonian object

    psi0_qt = qt.Qobj(psi0.reshape(-1, 1), dims=[[n_t, n_c], [1, 1]]) #create a Quantum object for the initial state vector psi0, reshaped into a column vector [1,1] make it a single state
    result = qt.sesolve(H, psi0_qt, tlist) #solve the Schrodinger equation with the time-dependent Hamiltonian H, initial state psi0_qt, and time points tlist
    return result.states[-1]


def qutip_multi_state_fidelity(u, factory, n_t, n_c, dt=DT):
    """
    QuTiP-side counterpart of grape_core.fidelity_multi_state: some pulses
    (the logical gates, U_enc, U_dec) are optimized against *several*
    (psi_i, psi_f) pairs at once -- e.g. logical X must map |+Z_L>->|-Z_L>
    AND |-Z_L>->|+Z_L> simultaneously. Propagate each pair independently
    with sesolve and average |<f|psi(T)>|^2 across pairs, exactly matching
    grape_core's "simple average" convention.
    """
    pairs = factory(n_c=n_c, n_t=n_t)
    fids = []
    for psi_i, psi_f in pairs:
        psi_final = qutip_final_state(u, n_t, n_c, psi_i, dt=dt)
        psi_f_qt = qt.Qobj(psi_f.reshape(-1, 1), dims=[[n_t, n_c], [1, 1]])
        fids.append(np.abs(psi_f_qt.overlap(psi_final)) ** 2)
    return np.mean(fids)


def validate_pulse(filename, label, factory):
    path = os.path.join(PULSE_DIR, filename)
    u = np.load(path)
    print(f"\n{'='*60}")
    print(f"{label}  ({filename}, shape={u.shape})")
    print(f"{'='*60}")
    print(f"{'n_c':>4}  {'grape_core F':>14}  {'qutip F':>10}  {'|diff|':>10}")

    max_diff = 0.0
    for n_c in TRUNC_LIST:
        pairs = factory(n_c=n_c, n_t=N_T)
        psi_i_list = [p[0] for p in pairs]
        psi_f_list = [p[1] for p in pairs]

        # --- grape_core's own propagator ---
        H0_np, Hc_np = make_hamiltonian(N_T, n_c)
        F_gc, _ = fidelity_multi_state(u, H0_np, Hc_np, psi_i_list, psi_f_list, DT, want_grad=False)

        # --- QuTiP's independent ODE propagator ---
        F_qt = qutip_multi_state_fidelity(u, factory, N_T, n_c)

        diff = abs(F_gc - F_qt)
        max_diff = max(max_diff, diff)
        marker = " (trained)" if n_c in (22, 24, 26) else " (held out)"
        print(f"{n_c:>4}  {F_gc:>14.6f}  {F_qt:>10.6f}  {diff:>10.2e}{marker}")

    return max_diff


def main():
    print("QuTiP cross-check of every saved *_mt.npy pulse against grape_core's own propagator.")
    summary = []
    for filename, (label, factory) in PULSE_MAP.items():
        path = os.path.join(PULSE_DIR, filename)
        if not os.path.exists(path):
            print(f"\n[SKIP] {label}: {path} not found")
            continue
        max_diff = validate_pulse(filename, label, factory)
        summary.append((label, max_diff))

    print(f"\n{'='*60}")
    print("SUMMARY (max |grape_core F - qutip F| across truncations)")
    print(f"{'='*60}")
    for label, max_diff in summary:
        flag = "  <-- CHECK THIS" if max_diff > 1e-4 else ""
        print(f"  {label:14s} {max_diff:.2e}{flag}")


if __name__ == "__main__":
    main()
