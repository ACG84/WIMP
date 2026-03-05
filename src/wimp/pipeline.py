"""End-to-end pipeline orchestration for WIMP.

Provides configuration management, batch processing, and a high-level
:func:`run_pipeline` entry point that chains data loading, fitting,
source localisation, and visualisation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from wimp.constants import GAMMA_NV
from wimp.io import WIMPDataset, load_hdf5, load_numpy, save_hdf5, export_results
from wimp.relaxation import fit_ramsey, fit_t2_decay, fit_t1_recovery
from wimp.source import (
    lead_field_matrix, mne_inverse, resolution_matrix, crosstalk_metrics,
    time_varying_inverse,
)
from wimp.registration import deformable_register

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for a WIMP processing run."""

    protocol: str = "ramsey"
    """Measurement protocol (``ramsey``, ``echo``, ``t1``, ``dd``)."""

    tau_range: tuple[float, float] = (0.0, 5e-6)
    """Expected τ range (seconds) — used for sanity checks only."""

    n_averages: int = 1000
    """Number of measurement averages (informational)."""

    fitting_params: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments forwarded to the fitting routine."""

    source_localization: bool = True
    """Whether to run MNE source localisation."""

    lambda_reg: float = 0.1
    """Regularisation parameter for MNE."""

    registration_params: dict[str, Any] = field(default_factory=dict)
    """Parameters forwarded to atlas registration."""

    output_dir: str = "wimp_output"
    """Directory for results and intermediate files."""

    save_intermediates: bool = False
    """If True, save per-ND fit results and field timeseries."""

    deformable: bool = False
    """If True, run per-frame deformable atlas registration."""

    compute_crosstalk: bool = False
    """If True, compute resolution matrix and crosstalk metrics."""

    denoise_method: str | None = None
    """Denoising method applied before fitting (``None`` = disabled)."""

    denoise_params: dict[str, Any] = field(default_factory=dict)
    """Parameters forwarded to the denoising function."""


def default_config(protocol: str = "ramsey") -> PipelineConfig:
    """Return a sensible default configuration for a given protocol."""
    if protocol == "ramsey":
        return PipelineConfig(protocol="ramsey", tau_range=(0, 5e-6))
    if protocol in ("echo", "hahn_echo"):
        return PipelineConfig(protocol="echo", tau_range=(0, 500e-6))
    if protocol == "t1":
        return PipelineConfig(protocol="t1", tau_range=(0, 20e-3))
    if protocol == "dd":
        return PipelineConfig(protocol="dd", tau_range=(1e-7, 1e-4))
    raise ValueError(f"Unknown protocol: {protocol!r}")


def validate_config(config: PipelineConfig) -> None:
    """Raise ``ValueError`` if the configuration is inconsistent."""
    valid_protocols = {"ramsey", "echo", "hahn_echo", "t1", "dd"}
    if config.protocol not in valid_protocols:
        raise ValueError(
            f"protocol must be one of {valid_protocols}, got {config.protocol!r}"
        )
    if config.lambda_reg <= 0:
        raise ValueError("lambda_reg must be > 0")
    if config.tau_range[0] >= config.tau_range[1]:
        raise ValueError("tau_range[0] must be < tau_range[1]")


# ---------------------------------------------------------------------------
# Per-ND fitting
# ---------------------------------------------------------------------------


def _fit_single_nd(
    protocol: str,
    tau: NDArray,
    signal: NDArray,
    fitting_params: dict,
) -> dict[str, Any]:
    """Fit a single ND's decay/fringe curve."""
    if protocol == "ramsey":
        return fit_ramsey(tau, signal, **fitting_params)
    if protocol in ("echo", "hahn_echo"):
        return fit_t2_decay(tau, signal, **fitting_params)
    if protocol == "t1":
        return fit_t1_recovery(tau, signal, **fitting_params)
    if protocol == "dd":
        # DD sweep doesn't use standard relaxation fit; return raw
        return {"tau": tau, "signal": signal}
    raise ValueError(f"Unknown protocol for fitting: {protocol!r}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    config: PipelineConfig,
    dataset: WIMPDataset | None = None,
    data_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the full WIMP processing pipeline.

    Either *dataset* or *data_path* must be provided.

    Parameters
    ----------
    config : PipelineConfig
    dataset : WIMPDataset, optional
    data_path : str or Path, optional
        Path to an HDF5 file or npz directory.

    Returns
    -------
    results : dict
        Contains ``fits`` (list of per-ND fit dicts),
        ``field_timeseries`` (if Ramsey), ``source_estimate``
        (if localisation was run), and ``config``.
    """
    validate_config(config)

    # 1. Load data
    if dataset is None:
        if data_path is None:
            raise ValueError("Provide either dataset or data_path")
        p = Path(data_path)
        if p.is_dir():
            dataset = load_numpy(p)
        else:
            dataset = load_hdf5(p)

    tau = np.asarray(dataset.tau_array)
    signal = np.asarray(dataset.signal)

    # Ensure 2-D: (n_nds, n_tau)
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    n_nds = signal.shape[0]

    # 1b. Denoise (if requested)
    if config.denoise_method is not None:
        from wimp.denoising import denoise
        signal = denoise(signal, method=config.denoise_method, **config.denoise_params)

    # 2. Fit each ND
    fits = []
    for i in range(n_nds):
        fit = _fit_single_nd(config.protocol, tau, signal[i], config.fitting_params)
        fits.append(fit)

    # 3. Extract field timeseries (protocol-dependent)
    field_ts = None
    if config.protocol == "ramsey":
        # Each fit gives a single B value; if time-resolved data is
        # available (signal shape n_nds × n_timepoints × n_tau), fit
        # each time-point.
        if signal.ndim == 3:
            n_t = signal.shape[1]
            field_ts = np.zeros((n_nds, n_t))
            for i in range(n_nds):
                for t_idx in range(n_t):
                    r = fit_ramsey(tau, signal[i, t_idx], **config.fitting_params)
                    field_ts[i, t_idx] = r.get("b_field", 0.0)
        else:
            field_ts = np.array([f.get("b_field", 0.0) for f in fits]).reshape(n_nds, 1)

    # 4. Deformable registration (if requested)
    deformable_result = None
    if config.deformable and dataset.nd_positions is not None:
        # Build per-frame ND positions from static positions (single frame)
        nd_pos = np.asarray(dataset.nd_positions)
        frame_positions = nd_pos[np.newaxis, :, :]  # (1, n_nds, 3)
        deformable_result = deformable_register(frame_positions)

    # 5. Source localisation
    source_est = None
    res_mat = None
    ct_metrics = None

    if config.source_localization and dataset.nd_positions is not None and field_ts is not None:
        neuron_pos = config.registration_params.get("neuron_positions")

        if deformable_result is not None:
            # Use deformable neuron positions for time-varying inverse
            tv_result = time_varying_inverse(
                deformable_result.neuron_positions,
                np.asarray(dataset.nd_positions),
                field_ts,
                lambda_reg=config.lambda_reg,
                compute_resolution=config.compute_crosstalk,
            )
            source_est = tv_result["source_estimate"]
            if config.compute_crosstalk and tv_result.get("resolution_matrices"):
                res_mat = tv_result["resolution_matrices"][0]
                ct_metrics = crosstalk_metrics(
                    res_mat, deformable_result.neuron_names
                )
        elif neuron_pos is not None:
            neuron_pos = np.asarray(neuron_pos)
            L = lead_field_matrix(neuron_pos, dataset.nd_positions)
            source_est = mne_inverse(L, field_ts, lambda_reg=config.lambda_reg)

            if config.compute_crosstalk:
                res_mat = resolution_matrix(L, lambda_reg=config.lambda_reg)
                ct_metrics = crosstalk_metrics(res_mat)

    # 6. Package results
    results: dict[str, Any] = {
        "fits": fits,
        "field_timeseries": field_ts,
        "source_estimate": source_est,
        "deformable_registration": deformable_result,
        "resolution_matrix": res_mat,
        "crosstalk_metrics": ct_metrics,
        "config": asdict(config),
        "n_nds": n_nds,
    }

    # 7. Save intermediates
    if config.save_intermediates:
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        export_results(
            {k: v for k, v in results.items() if k != "config"},
            out / "results.json",
            fmt="json",
        )

    return results


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def batch_process(
    config: PipelineConfig,
    data_paths: list[str | Path],
) -> list[dict[str, Any]]:
    """Run the pipeline on multiple datasets.

    Parameters
    ----------
    config : PipelineConfig
    data_paths : list of paths

    Returns
    -------
    results : list of dicts
    """
    return [run_pipeline(config, data_path=p) for p in data_paths]


def _run_pipeline_for_path(args: tuple[PipelineConfig, str]) -> dict[str, Any]:
    """Module-level helper for :func:`batch_process_parallel` (picklable)."""
    config, path = args
    return run_pipeline(config, data_path=path)


def batch_process_parallel(
    config: PipelineConfig,
    data_paths: list[str | Path],
    *,
    n_workers: int | None = None,
    progress_callback: Callable[[int, int], Any] | None = None,
) -> list[dict[str, Any]]:
    """Run the pipeline on multiple datasets in parallel.

    Uses :class:`~concurrent.futures.ProcessPoolExecutor` for
    cross-dataset parallelism.  Falls back to sequential processing
    if multiprocessing is unavailable.

    Parameters
    ----------
    config : PipelineConfig
    data_paths : list of paths
    n_workers : int, optional
        Maximum worker processes.  ``None`` → CPU count.
    progress_callback : callable, optional
        Called as ``progress_callback(completed, total)`` after each
        dataset finishes.

    Returns
    -------
    results : list of dicts
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    total = len(data_paths)
    if total == 0:
        return []

    # Build per-dataset configs (unique output subdirs when saving)
    args_list: list[tuple[PipelineConfig, str]] = []
    for p in data_paths:
        if config.save_intermediates:
            stem = Path(p).stem
            cfg = replace(config, output_dir=str(Path(config.output_dir) / stem))
        else:
            cfg = config
        args_list.append((cfg, str(p)))

    try:
        results_map: dict[int, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(_run_pipeline_for_path, a): i
                for i, a in enumerate(args_list)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results_map[idx] = future.result()
                if progress_callback is not None:
                    progress_callback(len(results_map), total)

        return [results_map[i] for i in range(total)]
    except (OSError, RuntimeError):
        return batch_process(config, data_paths)
