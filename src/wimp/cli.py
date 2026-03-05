"""Command-line interface for WIMP.

Provides ``wimp`` as a console entry point with subcommands for running
the processing pipeline, generating synthetic data, and comparing
protocol sensitivities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wimp",
        description="WIMP - Worm IMage Processor: NV magnetometry signal processing for C. elegans",
    )
    parser.add_argument(
        "--version", action="store_true", help="Print version and exit"
    )
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    run_p = sub.add_parser("run", help="Run the processing pipeline on a dataset")
    run_p.add_argument("data", help="Path to HDF5/npz/CSV data file or directory")
    run_p.add_argument(
        "-p", "--protocol", default="ramsey",
        choices=["ramsey", "echo", "hahn_echo", "t1", "dd"],
        help="Measurement protocol (default: ramsey)",
    )
    run_p.add_argument(
        "-o", "--output", default="wimp_output",
        help="Output directory (default: wimp_output)",
    )
    run_p.add_argument(
        "--lambda-reg", type=float, default=0.1,
        help="MNE regularisation parameter (default: 0.1)",
    )
    run_p.add_argument(
        "--no-source-loc", action="store_true",
        help="Skip source localisation step",
    )
    run_p.add_argument(
        "--save-intermediates", action="store_true",
        help="Save intermediate fit results to output directory",
    )
    run_p.add_argument(
        "--plot", action="store_true",
        help="Generate and save summary plots",
    )
    run_p.add_argument(
        "--deformable", action="store_true",
        help="Run per-frame deformable atlas registration",
    )
    run_p.add_argument(
        "--compute-crosstalk", action="store_true",
        help="Compute resolution matrix and crosstalk metrics",
    )

    # --- synthetic ---
    syn_p = sub.add_parser("synthetic", help="Generate synthetic experiment data")
    syn_p.add_argument(
        "-p", "--protocol", default="ramsey",
        choices=["ramsey", "echo", "hahn_echo", "t1", "dd"],
        help="Protocol (default: ramsey)",
    )
    syn_p.add_argument("--n-nds", type=int, default=10, help="Number of nanodiamonds")
    syn_p.add_argument("--n-neurons", type=int, default=5, help="Number of neurons")
    syn_p.add_argument("--n-timepoints", type=int, default=500, help="Time points")
    syn_p.add_argument("--snr", type=float, default=30.0, help="Signal-to-noise ratio")
    syn_p.add_argument("--seed", type=int, default=None, help="Random seed")
    syn_p.add_argument(
        "-o", "--output", default="synthetic_data.h5",
        help="Output file path (.h5 or directory for npz)",
    )

    # --- sensitivity ---
    sens_p = sub.add_parser("sensitivity", help="Compare protocol sensitivities")
    sens_p.add_argument("--contrast", type=float, default=0.03, help="Readout contrast")
    sens_p.add_argument("--t2star", type=float, default=1e-6, help="T2* (seconds)")
    sens_p.add_argument("--t2", type=float, default=100e-6, help="T2 (seconds)")
    sens_p.add_argument("--t1", type=float, default=5e-3, help="T1 (seconds)")
    sens_p.add_argument(
        "--readout-rate", type=float, default=1e5, help="Photon count rate (Hz)"
    )
    sens_p.add_argument("--n-nv", type=int, default=1, help="Number of NV centres")
    sens_p.add_argument(
        "-o", "--output", default=None,
        help="Save comparison table to CSV file",
    )

    # --- crosstalk ---
    ct_p = sub.add_parser("crosstalk", help="Analyse crosstalk for a given geometry")
    ct_p.add_argument("--n-nds", type=int, default=10, help="Number of nanodiamonds")
    ct_p.add_argument("--n-neurons", type=int, default=5, help="Number of neurons")
    ct_p.add_argument(
        "--lambda-reg", type=float, default=0.1,
        help="MNE regularisation parameter (default: 0.1)",
    )
    ct_p.add_argument(
        "--plot", action="store_true",
        help="Display resolution matrix plot",
    )
    ct_p.add_argument(
        "-o", "--output", default=None,
        help="Save resolution matrix plot to file",
    )

    # --- info ---
    info_p = sub.add_parser("info", help="Show dataset information")
    info_p.add_argument("data", help="Path to HDF5/npz data file or directory")

    return parser


def _cmd_run(args: argparse.Namespace) -> None:
    from wimp.pipeline import PipelineConfig, run_pipeline

    config = PipelineConfig(
        protocol=args.protocol,
        output_dir=args.output,
        lambda_reg=args.lambda_reg,
        source_localization=not args.no_source_loc,
        save_intermediates=args.save_intermediates,
        deformable=args.deformable,
        compute_crosstalk=args.compute_crosstalk,
    )

    results = run_pipeline(config, data_path=args.data)

    n_nds = results["n_nds"]
    print(f"Pipeline complete: {n_nds} nanodiamond(s) processed")
    print(f"Protocol: {args.protocol}")

    if results.get("field_timeseries") is not None:
        ft = results["field_timeseries"]
        print(f"Field timeseries shape: {ft.shape}")

    if results.get("source_estimate") is not None:
        print("Source localisation: completed")
    else:
        print("Source localisation: skipped")

    ct = results.get("crosstalk_metrics")
    if ct is not None:
        diag = ct["diagonal"]
        print(f"Crosstalk analysis: mean self-resolution = {float(diag.mean()):.3f}")
        print(f"  max crosstalk ratio = {float(ct['crosstalk_ratio'].max()):.3f}")

    if args.plot:
        _save_plots(results, args)

    if args.save_intermediates:
        print(f"Results saved to: {args.output}/")


def _save_plots(results: dict, args: argparse.Namespace) -> None:
    """Generate and save summary plots to the output directory."""
    import matplotlib
    matplotlib.use("Agg")
    from wimp import viz

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    fits = results.get("fits", [])
    if fits and args.protocol == "ramsey":
        fit0 = fits[0]
        if "frequency" in fit0:
            tau = fit0.get("tau_fit", np.linspace(0, 5e-6, 100))
            fig = viz.plot_ramsey_fringe(tau, np.zeros_like(tau), fit0)
            if fig is not None:
                fig.savefig(out / "ramsey_fit.png", dpi=150)
                print(f"Saved: {out / 'ramsey_fit.png'}")

    ft = results.get("field_timeseries")
    if ft is not None and ft.ndim == 2 and ft.shape[1] > 1:
        time = np.arange(ft.shape[1]) * 1e-3
        fig = viz.plot_field_timeseries(time, ft)
        if fig is not None:
            fig.savefig(out / "field_timeseries.png", dpi=150)
            print(f"Saved: {out / 'field_timeseries.png'}")


def _cmd_synthetic(args: argparse.Namespace) -> None:
    from wimp.synthetic import generate_full_experiment
    from wimp.io import WIMPDataset, save_hdf5, save_numpy

    print(f"Generating synthetic {args.protocol} experiment...")
    print(f"  NDs: {args.n_nds}, Neurons: {args.n_neurons}, "
          f"Timepoints: {args.n_timepoints}, SNR: {args.snr}")

    exp = generate_full_experiment(
        n_nds=args.n_nds,
        n_neurons=args.n_neurons,
        n_timepoints=args.n_timepoints,
        protocol=args.protocol,
        snr=args.snr,
        seed=args.seed,
    )

    ds = WIMPDataset(
        protocol=exp["protocol"],
        tau_array=exp["tau_array"],
        signal=exp["signal"],
        field_timeseries=exp.get("field_timeseries"),
        nd_positions=exp.get("nd_positions"),
        metadata=exp.get("metadata", {}),
    )

    out = Path(args.output)
    if out.suffix in (".h5", ".hdf5"):
        save_hdf5(ds, out)
        print(f"Saved HDF5: {out}")
    else:
        save_numpy(ds, out)
        print(f"Saved npz directory: {out}")


def _cmd_sensitivity(args: argparse.Namespace) -> None:
    from wimp.sensitivity import compare_protocols

    df = compare_protocols(
        contrast=args.contrast,
        t2star=args.t2star,
        t2=args.t2,
        t1=args.t1,
        readout_rate=args.readout_rate,
        n_nv=args.n_nv,
    )

    print("\nProtocol Sensitivity Comparison")
    print("=" * 60)
    print(df.to_string(index=False))
    print()

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved to: {args.output}")


def _cmd_info(args: argparse.Namespace) -> None:
    from wimp.io import load_hdf5, load_numpy

    p = Path(args.data)
    if p.is_dir():
        ds = load_numpy(p)
    else:
        ds = load_hdf5(p)

    print(f"\nDataset: {args.data}")
    print(f"  Protocol:        {ds.protocol}")
    print(f"  tau_array shape: {np.asarray(ds.tau_array).shape}")
    print(f"  signal shape:    {np.asarray(ds.signal).shape}")

    if ds.nd_positions is not None:
        print(f"  nd_positions:    {np.asarray(ds.nd_positions).shape}")
    else:
        print("  nd_positions:    None")

    if ds.field_timeseries is not None:
        print(f"  field_timeseries:{np.asarray(ds.field_timeseries).shape}")

    if ds.metadata:
        print(f"  metadata keys:   {list(ds.metadata.keys())}")
    print()


def _cmd_crosstalk(args: argparse.Namespace) -> None:
    from wimp.synthetic import generate_full_experiment
    from wimp.source import (
        lead_field_matrix, resolution_matrix, crosstalk_metrics,
    )

    exp = generate_full_experiment(
        n_nds=args.n_nds,
        n_neurons=args.n_neurons,
        protocol="ramsey",
        seed=0,
    )
    nd_pos = exp["nd_positions"]
    neuron_pos = exp["neuron_positions"]

    L = lead_field_matrix(neuron_pos, nd_pos)
    R = resolution_matrix(L, lambda_reg=args.lambda_reg)
    metrics = crosstalk_metrics(R)

    print(f"\nCrosstalk Analysis ({args.n_nds} NDs, {args.n_neurons} neurons)")
    print(f"Lambda regularisation: {args.lambda_reg}")
    print("=" * 60)
    for i, name in enumerate(metrics["neuron_names"]):
        print(f"  {name:>8s}: self-res={metrics['diagonal'][i]:.4f}  "
              f"crosstalk={metrics['crosstalk_ratio'][i]:.4f}  "
              f"spread={metrics['spread'][i]}")
    print(f"\n  Mean self-resolution: {metrics['diagonal'].mean():.4f}")
    print(f"  Max crosstalk ratio:  {metrics['crosstalk_ratio'].max():.4f}")
    print()

    if args.plot or args.output:
        import matplotlib
        matplotlib.use("Agg")
        from wimp.viz import plot_resolution_matrix

        fig = plot_resolution_matrix(R, metrics["neuron_names"])
        if fig is not None:
            if args.output:
                fig.savefig(args.output, dpi=150)
                print(f"Saved: {args.output}")
            else:
                fig.savefig("resolution_matrix.png", dpi=150)
                print("Saved: resolution_matrix.png")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``wimp`` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from wimp import __version__
        print(f"wimp {__version__}")
        return

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "run": _cmd_run,
        "synthetic": _cmd_synthetic,
        "sensitivity": _cmd_sensitivity,
        "crosstalk": _cmd_crosstalk,
        "info": _cmd_info,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return

    handler(args)


if __name__ == "__main__":
    main()
