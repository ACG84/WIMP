"""Data I/O and the :class:`WIMPDataset` container.

Supports HDF5, NumPy ``.npz``, and CSV round-trips for time-domain
NV magnetometry data.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------


@dataclass
class WIMPDataset:
    """Container for a single time-domain NV measurement session.

    Attributes
    ----------
    protocol : str
        Measurement protocol (``"ramsey"``, ``"echo"``, ``"t1"``,
        ``"dd"``).
    tau_array : ndarray
        Free-evolution times (seconds), shape ``(n_tau,)``.
    signal : ndarray
        Raw signal.  Shape is typically ``(n_nds, n_tau)`` for a single
        sweep or ``(n_nds, n_timepoints, n_tau)`` for time-resolved
        measurements.
    field_timeseries : ndarray | None
        Extracted magnetic-field time series ``(n_nds, n_timepoints)``
        (Tesla).  Populated after fitting.
    nd_positions : ndarray | None
        Nano-diamond positions ``(n_nds, 3)`` in metres.
    metadata : dict
        Arbitrary metadata (temperature, n_averages, …).
    frequency_axis : ndarray | None
        For DD noise spectroscopy: frequency axis (Hz).
    """

    protocol: str = "ramsey"
    tau_array: NDArray = field(default_factory=lambda: np.array([]))
    signal: NDArray = field(default_factory=lambda: np.array([]))
    field_timeseries: NDArray | None = None
    nd_positions: NDArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    frequency_axis: NDArray | None = None


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------


def save_hdf5(dataset: WIMPDataset, path: str | Path) -> None:
    """Save a :class:`WIMPDataset` to an HDF5 file."""
    import h5py

    path = Path(path)
    with h5py.File(path, "w") as f:
        f.attrs["protocol"] = dataset.protocol
        f.create_dataset("tau_array", data=np.asarray(dataset.tau_array))
        f.create_dataset("signal", data=np.asarray(dataset.signal))

        if dataset.field_timeseries is not None:
            f.create_dataset("field_timeseries", data=dataset.field_timeseries)
        if dataset.nd_positions is not None:
            f.create_dataset("nd_positions", data=dataset.nd_positions)
        if dataset.frequency_axis is not None:
            f.create_dataset("frequency_axis", data=dataset.frequency_axis)

        # Store metadata as JSON string attribute
        f.attrs["metadata"] = json.dumps(
            _make_serialisable(dataset.metadata)
        )


def load_hdf5(path: str | Path) -> WIMPDataset:
    """Load a :class:`WIMPDataset` from an HDF5 file."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        protocol = str(f.attrs["protocol"])
        tau_array = np.array(f["tau_array"])
        signal = np.array(f["signal"])

        field_ts = np.array(f["field_timeseries"]) if "field_timeseries" in f else None
        nd_pos = np.array(f["nd_positions"]) if "nd_positions" in f else None
        freq_ax = np.array(f["frequency_axis"]) if "frequency_axis" in f else None

        meta_str = f.attrs.get("metadata", "{}")
        metadata = json.loads(meta_str)

    return WIMPDataset(
        protocol=protocol,
        tau_array=tau_array,
        signal=signal,
        field_timeseries=field_ts,
        nd_positions=nd_pos,
        metadata=metadata,
        frequency_axis=freq_ax,
    )


# ---------------------------------------------------------------------------
# NumPy (.npz) I/O
# ---------------------------------------------------------------------------


def save_numpy(dataset: WIMPDataset, directory: str | Path) -> None:
    """Save a :class:`WIMPDataset` as ``.npz`` files in *directory*."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {
        "tau_array": np.asarray(dataset.tau_array),
        "signal": np.asarray(dataset.signal),
    }
    if dataset.field_timeseries is not None:
        arrays["field_timeseries"] = dataset.field_timeseries
    if dataset.nd_positions is not None:
        arrays["nd_positions"] = dataset.nd_positions
    if dataset.frequency_axis is not None:
        arrays["frequency_axis"] = dataset.frequency_axis

    np.savez(d / "data.npz", **arrays)

    meta = {"protocol": dataset.protocol}
    meta.update(_make_serialisable(dataset.metadata))
    with open(d / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)


def load_numpy(directory: str | Path) -> WIMPDataset:
    """Load a :class:`WIMPDataset` from a directory saved by
    :func:`save_numpy`."""
    d = Path(directory)
    data = np.load(d / "data.npz")

    with open(d / "metadata.json") as fh:
        meta = json.load(fh)

    protocol = meta.pop("protocol", "ramsey")

    return WIMPDataset(
        protocol=protocol,
        tau_array=data["tau_array"],
        signal=data["signal"],
        field_timeseries=data.get("field_timeseries"),
        nd_positions=data.get("nd_positions"),
        metadata=meta,
        frequency_axis=data.get("frequency_axis"),
    )


# ---------------------------------------------------------------------------
# Zarr I/O
# ---------------------------------------------------------------------------


def save_zarr(
    dataset: WIMPDataset,
    store: str | Path,
    *,
    chunk_size: int | None = None,
) -> None:
    """Save a :class:`WIMPDataset` to a Zarr store.

    Requires the optional ``zarr`` dependency.  Install with
    ``pip install 'wimp[zarr]'``.

    Parameters
    ----------
    dataset : WIMPDataset
        Dataset to save.
    store : str or Path
        Path to the zarr directory store.
    chunk_size : int | None
        Chunk size along the first axis.  If *None*, zarr picks
        default chunks.
    """
    try:
        import zarr
    except ImportError:
        raise ImportError(
            "Zarr I/O requires zarr. Install with: pip install 'wimp[zarr]'"
        ) from None

    store = Path(store)
    root = zarr.open(str(store), mode="w")

    root.attrs["protocol"] = dataset.protocol
    root.attrs["metadata"] = json.dumps(
        _make_serialisable(dataset.metadata)
    )

    chunks = (chunk_size,) if chunk_size else None
    tau = np.asarray(dataset.tau_array)
    root.create_dataset("tau_array", data=tau, chunks=chunks)

    sig = np.asarray(dataset.signal)
    sig_chunks = None
    if chunk_size and sig.ndim >= 1:
        sig_chunks = tuple(
            [chunk_size] + list(sig.shape[1:])
        ) if sig.ndim > 1 else (chunk_size,)
    root.create_dataset("signal", data=sig, chunks=sig_chunks)

    if dataset.field_timeseries is not None:
        root.create_dataset("field_timeseries", data=np.asarray(dataset.field_timeseries))
    if dataset.nd_positions is not None:
        root.create_dataset("nd_positions", data=np.asarray(dataset.nd_positions))
    if dataset.frequency_axis is not None:
        root.create_dataset("frequency_axis", data=np.asarray(dataset.frequency_axis))


def load_zarr(
    store: str | Path,
    *,
    lazy: bool = False,
) -> WIMPDataset:
    """Load a :class:`WIMPDataset` from a Zarr store.

    Parameters
    ----------
    store : str or Path
        Path to the zarr directory store.
    lazy : bool
        If *True*, keep arrays as zarr arrays (zero-copy, memory-mapped).
        If *False* (default), convert to NumPy arrays.

    Returns
    -------
    WIMPDataset
    """
    try:
        import zarr
    except ImportError:
        raise ImportError(
            "Zarr I/O requires zarr. Install with: pip install 'wimp[zarr]'"
        ) from None

    store = Path(store)
    root = zarr.open(str(store), mode="r")

    protocol = str(root.attrs["protocol"])
    meta_str = root.attrs.get("metadata", "{}")
    metadata = json.loads(meta_str)

    _conv = (lambda x: x) if lazy else np.asarray

    tau_array = _conv(root["tau_array"])
    signal = _conv(root["signal"])

    field_ts = _conv(root["field_timeseries"]) if "field_timeseries" in root else None
    nd_pos = _conv(root["nd_positions"]) if "nd_positions" in root else None
    freq_ax = _conv(root["frequency_axis"]) if "frequency_axis" in root else None

    return WIMPDataset(
        protocol=protocol,
        tau_array=tau_array,
        signal=signal,
        field_timeseries=field_ts,
        nd_positions=nd_pos,
        metadata=metadata,
        frequency_axis=freq_ax,
    )


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def load_csv_timeseries(
    path: str | Path,
    *,
    time_col: str = "time_s",
    field_cols: list[str] | None = None,
    delimiter: str = ",",
) -> tuple[NDArray, NDArray]:
    """Load a field time-series from a CSV file.

    Parameters
    ----------
    path : str or Path
        CSV file path.
    time_col : str
        Name of the column containing time stamps (seconds).
    field_cols : list[str] | None
        Column names for field data.  If *None*, all columns except
        *time_col* are used.
    delimiter : str
        Column delimiter.

    Returns
    -------
    time : ndarray, shape ``(n_timepoints,)``
    field : ndarray, shape ``(n_channels, n_timepoints)``
    """
    import pandas as pd

    df = pd.read_csv(path, delimiter=delimiter)
    time = df[time_col].values.astype(float)

    if field_cols is None:
        field_cols = [c for c in df.columns if c != time_col]

    field = df[field_cols].values.T.astype(float)
    return time, field


def export_results(
    results: dict[str, Any],
    path: str | Path,
    fmt: str = "json",
) -> None:
    """Export a results dictionary to disk.

    Parameters
    ----------
    results : dict
        Arbitrary results dictionary.
    path : str or Path
        Output file path.
    fmt : str
        ``"json"`` or ``"csv"``.
    """
    path = Path(path)

    if fmt == "json":
        with open(path, "w") as fh:
            json.dump(_make_serialisable(results), fh, indent=2)

    elif fmt == "csv":
        import pandas as pd
        # Flatten one-level nesting
        flat: dict[str, Any] = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                flat[k] = v.tolist()
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}.{kk}"] = vv
                    if isinstance(vv, np.ndarray):
                        flat[f"{k}.{kk}"] = vv.tolist()
            else:
                flat[k] = v
        pd.DataFrame([flat]).to_csv(path, index=False)

    else:
        raise ValueError(f"Unknown format: {fmt!r}")


def load_raw_measurement(
    path: str | Path,
    fmt: str = "auto",
) -> dict[str, NDArray]:
    """Load raw measurement arrays from a file.

    Supports ``.npz``, ``.npy``, ``.h5``/``.hdf5``, and ``.csv``.

    Returns
    -------
    arrays : dict[str, ndarray]
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if fmt != "auto":
        suffix = f".{fmt.lstrip('.')}"

    if suffix == ".npz":
        return dict(np.load(path))

    if suffix == ".npy":
        return {"data": np.load(path)}

    if suffix in (".h5", ".hdf5"):
        import h5py
        arrays = {}
        with h5py.File(path, "r") as f:
            for key in f:
                arrays[key] = np.array(f[key])
        return arrays

    if suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        return {col: df[col].values for col in df.columns}

    raise ValueError(f"Unsupported file format: {suffix}")


# ---------------------------------------------------------------------------
# NWB export
# ---------------------------------------------------------------------------


def export_nwb(
    dataset: WIMPDataset,
    results: dict[str, Any],
    path: str | Path,
    *,
    session_description: str = "WIMP NV magnetometry session",
    experimenter: str = "",
    lab: str = "",
    institution: str = "",
) -> None:
    """Export WIMP data and results to NWB format.

    Requires the optional ``pynwb`` dependency.  Install with
    ``pip install 'wimp[nwb]'``.

    Parameters
    ----------
    dataset : WIMPDataset
        Input dataset.
    results : dict
        Pipeline results (from :func:`wimp.pipeline.run_pipeline`).
    path : str or Path
        Output ``.nwb`` file path.
    session_description : str
        Free-text description of the recording session.
    experimenter, lab, institution : str
        Optional metadata fields.
    """
    try:
        from datetime import datetime
        from uuid import uuid4

        from pynwb import NWBFile, NWBHDF5IO, TimeSeries, ProcessingModule
    except ImportError:
        raise ImportError(
            "NWB export requires pynwb. "
            "Install with: pip install 'wimp[nwb]'"
        ) from None

    path = Path(path)

    nwbfile = NWBFile(
        session_description=session_description,
        identifier=str(uuid4()),
        session_start_time=datetime.now(),
        experimenter=experimenter or None,
        lab=lab or None,
        institution=institution or None,
    )

    # Raw signal — pynwb requires time on axis 0
    tau = np.asarray(dataset.tau_array)
    signal = np.asarray(dataset.signal)
    if signal.ndim > 1:
        signal = signal.T  # (n_nds, n_tau) -> (n_tau, n_nds)
    nwbfile.add_acquisition(
        TimeSeries(
            name="raw_signal",
            data=signal,
            unit="a.u.",
            timestamps=tau,
            description=f"Raw {dataset.protocol} signal",
        )
    )

    # Field timeseries (if available)
    if dataset.field_timeseries is not None:
        ft = np.asarray(dataset.field_timeseries)
        n_t = ft.shape[-1] if ft.ndim > 1 else len(ft)
        nwbfile.add_acquisition(
            TimeSeries(
                name="field_timeseries",
                data=ft,
                unit="T",
                rate=1.0,  # placeholder rate
                description="Extracted B-field time series",
            )
        )

    # Processing module for fit results
    proc_mod = ProcessingModule(
        name="wimp_processing",
        description="WIMP pipeline fit results",
    )

    # Store field values from results
    if results.get("field_timeseries") is not None:
        ft_result = np.asarray(results["field_timeseries"])
        proc_mod.add(
            TimeSeries(
                name="pipeline_field",
                data=ft_result,
                unit="T",
                rate=1.0,
                description="Pipeline-extracted B-field",
            )
        )

    # Source estimate
    if results.get("source_estimate") is not None:
        se = np.asarray(results["source_estimate"])
        proc_mod.add(
            TimeSeries(
                name="source_estimate",
                data=se,
                unit="A",
                rate=1.0,
                description="Source-localised neural current estimate",
            )
        )

    nwbfile.add_processing_module(proc_mod)

    # Write
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-friendly Python types."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
