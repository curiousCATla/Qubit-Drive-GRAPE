#!/usr/bin/env python3
"""
main.py

Standalone CLI entry point for the GRAPE logical-gate optimization campaign
(the non-validation half of experiments.ipynb): pick a gate, optimize its
control pulse, save it, and optionally produce diagnostic plots and a
decoherence-limited fidelity comparison.

Usage:
    python main.py --gate X
    python main.py --gate H --plot-waveform --plot-spectrum --plot-photon
    python main.py --gate enc --decoherence --t1-transmon 100e-6
    python main.py --help
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np

from core.grape_core import (
    make_hamiltonian,
    fidelity_multi_state,
    coherent_fidelity_multi_state,
)
from core.cat_code import (
    get_logical_X_state_pairs,
    get_logical_Y_state_pairs,
    get_logical_Z_state_pairs,
    get_logical_H_state_pairs,
    get_logical_T_state_pairs,
    get_identity_state_pairs,
    get_encode_state_pairs,
    get_decode_state_pairs,
    validate_pulse_truncations,
)
from core.optimizer import optimize_multi_state_pulse
from visualization.pulse_viz import (
    plot_pulse_waveforms,
    plot_pulse_spectrum,
    plot_photon_trajectory,
)
from analysis.decoherence import simulate_with_decoherence, compute_fidelity


GATE_FACTORIES = {
    "X": get_logical_X_state_pairs,
    "Y": get_logical_Y_state_pairs,
    "Z": get_logical_Z_state_pairs,
    "H": get_logical_H_state_pairs,
    "T": get_logical_T_state_pairs,
    "I": get_identity_state_pairs,
    "enc": get_encode_state_pairs,
    "dec": get_decode_state_pairs,
}

GATE_LABELS = {
    "X": "U_X", "Y": "U_Y", "Z": "U_Z", "H": "U_H", "T": "U_T", "I": "U_I",
    "enc": "U_enc", "dec": "U_dec",
}

GATE_HELP = (
    "Logical operation to optimize a control pulse for. "
    "X/Y/Z/H/T/I: single-qubit logical gates on the 4-component cat code. "
    "enc: |g,0>/|e,0> -> |+Z_L>/|-Z_L> (encode transmon state into cavity cat code). "
    "dec: inverse of enc (cat code -> transmon computational basis)."
)


def parse_band(values):
    """Parse a --cav-band/--tra-band argument: either ['none'] or two floats."""
    if len(values) == 1 and values[0].lower() == "none":
        return None
    if len(values) != 2:
        raise argparse.ArgumentTypeError(
            "expected either 'none' or two floats (f_lo f_hi)"
        )
    return (float(values[0]), float(values[1]))


def build_arg_parser():
    p = argparse.ArgumentParser(
        description=(
            "Optimize a GRAPE control pulse for one logical-gate operation, "
            "with optional diagnostic plots and a decoherence comparison. "
            "Excludes the validation suite in validation/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--gate", required=True, choices=sorted(GATE_FACTORIES),
                    help=GATE_HELP)
    p.add_argument("--fidelity-fn", choices=["auto", "average", "coherent"],
                    default="auto",
                    help="Fidelity metric used for training. 'auto' uses the "
                         "phase-sensitive coherent_fidelity_multi_state for H/T "
                         "and the plain average otherwise.")

    # --- Optimization hyperparameters ---
    opt = p.add_argument_group("optimization parameters")
    opt.add_argument("--trunc-list", type=int, nargs="+", default=[22, 24, 26],
                      help="Cavity truncations (n_c) trained on simultaneously.")
    opt.add_argument("--n-steps", type=int, default=550, help="Number of time steps N.")
    opt.add_argument("--dt", type=float, default=0.002, help="Time step in microseconds.")
    opt.add_argument("--n-t", type=int, default=3, help="Number of transmon levels.")
    opt.add_argument("--lambda-deriv", type=float, default=1e-5,
                      help="Weight of the pulse-smoothness (derivative) penalty.")
    opt.add_argument("--lambda-boundary", type=float, default=2e-5,
                      help="Weight of the boundary (start/end amplitude) penalty.")
    opt.add_argument("--lambda-amp", type=float, default=8e-5,
                      help="Weight of the soft amplitude penalty.")
    opt.add_argument("--lambda-disc", type=float, default=0.5,
                      help="Weight of the cross-truncation discrepancy penalty.")
    opt.add_argument("--amp-max", type=float, default=40.0,
                      help="Soft amplitude threshold (rad/us) used inside the amplitude penalty.")
    opt.add_argument("--hard-amp-limit", type=float, default=40.0,
                      help="Hard L-BFGS-B box constraint on pulse amplitude (rad/us).")
    opt.add_argument("--cav-band", type=str, nargs="+", default=["-27.0", "27.0"],
                      help="Cavity drive band limit in MHz: 'F_LO F_HI' or 'none'.")
    opt.add_argument("--tra-band", type=str, nargs="+", default=["-33.0", "33.0"],
                      help="Transmon drive band limit in MHz: 'F_LO F_HI' or 'none'.")
    opt.add_argument("--max-iter", type=int, default=1500,
                      help="Maximum L-BFGS-B iterations.")
    opt.add_argument("--n-jobs", type=int, default=3,
                      help="Parallel workers for per-truncation fidelity evaluation.")
    opt.add_argument("--warm-start", type=str, default=None,
                      help="Path to a .npy pulse to warm-start from, or 'zero'. "
                           "Default: smooth random warm start.")
    opt.add_argument("--seed", type=int, default=42,
                      help="Random seed (warm start + numpy global seed).")
    opt.add_argument("--save-path", type=str, default=None,
                      help="Where to save the optimized pulse. "
                           "Default: pulses/u_<gate>_opt.npy")

    # --- Plot toggles ---
    plot = p.add_argument_group("plots (all default off)")
    plot.add_argument("--plot-waveform", action="store_true",
                       help="Save a cavity/transmon I-Q waveform-vs-time plot.")
    plot.add_argument("--plot-photon", action="store_true",
                       help="Save a photon-number distribution + trajectory plot.")
    plot.add_argument("--plot-spectrum", action="store_true",
                       help="Save a drive-spectrum (FFT) plot.")
    plot.add_argument("--plot-fidelity-nc", action="store_true",
                       help="Save a fidelity-vs-cavity-truncation plot.")
    plot.add_argument("--fidelity-nc-min", type=int, default=20,
                       help="Lowest n_c swept for --plot-fidelity-nc.")
    plot.add_argument("--fidelity-nc-max", type=int, default=40,
                       help="Highest n_c swept for --plot-fidelity-nc.")
    plot.add_argument("--fidelity-nc-step", type=int, default=2,
                       help="Step size for the --plot-fidelity-nc sweep.")
    plot.add_argument("--fig-dir", type=str, default="figures",
                       help="Output directory for all plots.")

    # --- Decoherence ---
    deco = p.add_argument_group("decoherence (default off)")
    deco.add_argument("--decoherence", action="store_true",
                       help="Run a post-hoc closed-vs-open (Lindblad) fidelity "
                            "comparison on the optimized pulse. Does not affect "
                            "training, which is always unitary.")
    deco.add_argument("--t1-cavity", type=float, default=2.7e-3,
                       help="Cavity T1 in seconds.")
    deco.add_argument("--t1-transmon", type=float, default=170e-6,
                       help="Transmon T1 in seconds.")
    deco.add_argument("--t-phi", type=float, default=43e-6,
                       help="Transmon pure dephasing time T_phi in seconds.")
    deco.add_argument("--decoherence-nc", type=int, default=24,
                       help="Cavity truncation used only for the decoherence simulation.")
    deco.add_argument("--table-dir", type=str, default="tables",
                       help="Output directory for the decoherence comparison CSV.")

    return p


def resolve_band(values):
    try:
        return parse_band(values)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(f"error: {exc}")


def resolve_warm_start(warm_start):
    if warm_start is None:
        return None
    if warm_start == "zero":
        return "zero"
    return warm_start


def resolve_fidelity_fn(gate, choice):
    if choice == "coherent":
        return coherent_fidelity_multi_state
    if choice == "average":
        return fidelity_multi_state
    return coherent_fidelity_multi_state if gate in ("H", "T") else fidelity_multi_state


def run_plots(args, u_opt, factory, label):
    os.makedirs(args.fig_dir, exist_ok=True)

    if args.plot_waveform:
        plot_pulse_waveforms(
            u_opt, dt=args.dt, title=f"{label} - Waveforms",
            save_path=os.path.join(args.fig_dir, f"{label}_waveforms.png"),
            show=False,
        )

    if args.plot_spectrum:
        plot_pulse_spectrum(
            u_opt, dt=args.dt, title=f"{label} - Spectrum",
            save_path=os.path.join(args.fig_dir, f"{label}_spectrum.png"),
            show=False,
        )

    if args.plot_photon:
        n_c_plot = max(args.trunc_list)
        state_pairs = factory(n_c=n_c_plot, n_t=args.n_t)
        psi_i_list = [pair[0] for pair in state_pairs]
        plot_photon_trajectory(
            u_opt, psi_i_list, dt=args.dt, n_c=n_c_plot, n_t=args.n_t,
            title=f"{label} - Photon Number Trajectory",
            save_path=os.path.join(args.fig_dir, f"{label}_photon_trajectory.png"),
            show=False,
        )

    if args.plot_fidelity_nc:
        trunc_range = range(args.fidelity_nc_min, args.fidelity_nc_max + 1, args.fidelity_nc_step)
        results = validate_pulse_truncations(
            u=u_opt, get_targets_func=factory, trunc_range=trunc_range,
            n_t=args.n_t, dt=args.dt,
            title=f"{label} - Fidelity vs Cavity Truncation",
        )

        import matplotlib.pyplot as plt
        ncs = sorted(results)
        fs = [results[nc] for nc in ncs]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ncs, fs, marker="o")
        ax.set_xlabel("Cavity truncation n_c")
        ax.set_ylabel("Fidelity")
        ax.set_title(f"{label} - Fidelity vs n_c")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.fig_dir, f"{label}_fidelity_vs_nc.png"), dpi=150)
        plt.close(fig)


def run_decoherence(args, u_opt, factory, label):
    os.makedirs(args.table_dir, exist_ok=True)

    n_c = args.decoherence_nc
    H0, Hc = make_hamiltonian(args.n_t, n_c)
    state_pairs = factory(n_c=n_c, n_t=args.n_t)

    rows = []
    print(f"\n{'='*60}")
    print(f"{label} - Decoherence comparison (n_c={n_c})")
    print(f"{'='*60}")
    for idx, (psi_i, psi_f) in enumerate(state_pairs):
        f_closed, _ = fidelity_multi_state(u_opt, H0, Hc, [psi_i], [psi_f], args.dt, want_grad=False)

        rho_final = simulate_with_decoherence(
            u_opt, psi_i, n_t=args.n_t, n_c=n_c, dt=args.dt,
            T1_C=args.t1_cavity, T1_T=args.t1_transmon, T_phi=args.t_phi,
            verbose=False,
        )
        f_open = float(compute_fidelity(rho_final, psi_f))

        print(f"  pair {idx}: F_closed = {f_closed:.6f}   F_open = {f_open:.6f}")
        rows.append({"pair_index": idx, "F_closed": f_closed, "F_open": f_open})
    print(f"{'='*60}\n")

    csv_path = os.path.join(args.table_dir, f"{label}_decoherence.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_index", "F_closed", "F_open"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")


def main():
    args = build_arg_parser().parse_args()

    np.random.seed(args.seed)

    factory = GATE_FACTORIES[args.gate]
    label = GATE_LABELS[args.gate]
    fidelity_fn = resolve_fidelity_fn(args.gate, args.fidelity_fn)
    print(f"Using fidelity function: {fidelity_fn.__name__}")
    cav_band = resolve_band(args.cav_band)
    tra_band = resolve_band(args.tra_band)
    save_path = args.save_path or os.path.join("pulses", f"u_{args.gate}_opt.npy")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    penalties = {
        "deriv": args.lambda_deriv,
        "boundary": args.lambda_boundary,
        "amp": args.lambda_amp,
        "amp_max": args.amp_max,
        "disc": args.lambda_disc,
    }

    u_opt, info = optimize_multi_state_pulse(
        get_state_pairs=factory,
        trunc_list=args.trunc_list,
        n_t=args.n_t,
        N=args.n_steps,
        dt=args.dt,
        penalties=penalties,
        warm_start=resolve_warm_start(args.warm_start),
        warm_start_seed=args.seed,
        save_path=save_path,
        n_jobs=args.n_jobs,
        maxiter=args.max_iter,
        cav_band=cav_band,
        tra_band=tra_band,
        hard_amp_limit=args.hard_amp_limit,
        fidelity_fn=fidelity_fn,
        verbose=True,
    )

    print(f"\nGate: {label}")
    print(f"Optimization info: {info}")

    run_plots(args, u_opt, factory, label)

    if args.decoherence:
        run_decoherence(args, u_opt, factory, label)


if __name__ == "__main__":
    main()
