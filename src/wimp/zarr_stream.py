"""Zarr-backed streaming I/O for real-time acquisition.

Provides :class:`ZarrStreamWriter` for appending frames during
acquisition and :class:`ZarrStreamReader` for reading back the
growing store.

Requires the optional ``zarr`` dependency.  Install with
``pip install 'wimp[zarr]'``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _import_zarr():
    """Lazy import with helpful error."""
    try:
        import zarr
        return zarr
    except ImportError:
        raise ImportError(
            "Zarr streaming requires zarr. "
            "Install with: pip install 'wimp[zarr]'"
        ) from None


# ---------------------------------------------------------------------------
# Stream writer
# ---------------------------------------------------------------------------


class ZarrStreamWriter:
    """Append measurement frames to a growing zarr store.

    Thread-safe: :meth:`append_frame` and :meth:`append_result` acquire
    an internal lock before writing.

    Parameters
    ----------
    store_path : str or Path
        Directory store path.
    tau_array : ndarray
        Free-evolution times shared by all frames.
    n_nds : int | None
        Number of nanodiamonds.  Inferred from the first frame if *None*.
    chunk_frames : int
        Chunk size along the frame axis for extensible datasets.
    """

    def __init__(
        self,
        store_path: str | Path,
        tau_array: NDArray,
        *,
        n_nds: int | None = None,
        chunk_frames: int = 100,
    ) -> None:
        zarr = _import_zarr()
        self._store_path = Path(store_path)
        self._root = zarr.open(str(self._store_path), mode="w")
        self._lock = threading.Lock()
        self._closed = False

        self._tau = np.asarray(tau_array)
        self._root.create_dataset("tau_array", data=self._tau)
        self._root.attrs["protocol"] = "realtime_stream"

        self._n_nds = n_nds
        self._chunk_frames = chunk_frames
        self._n_frames = 0

        # Lazily created on first append
        self._signals = None
        self._timestamps = None
        self._field_values = None

    # -- frame / result writers -----------------------------------------------

    def append_frame(self, frame: Any) -> None:
        """Append a :class:`~wimp.realtime.MeasurementFrame`.

        Parameters
        ----------
        frame : MeasurementFrame
            Must have ``signal``, ``timestamp`` attributes.
        """
        zarr = _import_zarr()
        sig = np.asarray(frame.signal)
        if sig.ndim == 1:
            sig = sig[np.newaxis, :]  # (1, n_tau)

        with self._lock:
            if self._closed:
                raise RuntimeError("Writer is closed")

            if self._signals is None:
                n_nds = self._n_nds or sig.shape[0]
                n_tau = sig.shape[1]
                self._signals = self._root.create_dataset(
                    "signals",
                    shape=(0, n_nds, n_tau),
                    chunks=(self._chunk_frames, n_nds, n_tau),
                    dtype=sig.dtype,
                )
                self._timestamps = self._root.create_dataset(
                    "timestamps",
                    shape=(0,),
                    chunks=(self._chunk_frames,),
                    dtype="f8",
                )

            idx = self._n_frames
            self._signals.resize(idx + 1, *self._signals.shape[1:])
            self._signals[idx] = sig

            self._timestamps.resize(idx + 1)
            self._timestamps[idx] = float(frame.timestamp)

            self._n_frames += 1

    def append_result(self, result: Any) -> None:
        """Append field values from a :class:`~wimp.realtime.RealtimeResult`.

        Parameters
        ----------
        result : RealtimeResult
            Must have ``field_values`` attribute (ndarray or None).
        """
        zarr = _import_zarr()
        fv = result.field_values
        if fv is None:
            return
        fv = np.asarray(fv)

        with self._lock:
            if self._closed:
                raise RuntimeError("Writer is closed")

            if self._field_values is None:
                n_nds = len(fv)
                self._field_values = self._root.create_dataset(
                    "field_values",
                    shape=(0, n_nds),
                    chunks=(self._chunk_frames, n_nds),
                    dtype=fv.dtype,
                )

            idx = self._field_values.shape[0]
            self._field_values.resize(idx + 1, self._field_values.shape[1])
            self._field_values[idx] = fv

    # -- properties -----------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Number of frames written so far."""
        return self._n_frames

    @property
    def closed(self) -> bool:
        return self._closed

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Finalise the store (mark as closed, write frame count)."""
        with self._lock:
            if self._closed:
                return
            self._root.attrs["n_frames"] = self._n_frames
            self._root.attrs["closed"] = True
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------------
# Stream reader
# ---------------------------------------------------------------------------


class ZarrStreamReader:
    """Read from a zarr store written by :class:`ZarrStreamWriter`.

    Supports reading from a store that is still being written to
    (the reader re-checks array sizes on each access).

    Parameters
    ----------
    store_path : str or Path
        Directory store path.
    """

    def __init__(self, store_path: str | Path) -> None:
        zarr = _import_zarr()
        self._store_path = Path(store_path)
        self._root = zarr.open(str(self._store_path), mode="r")

    @property
    def n_frames(self) -> int:
        """Number of signal frames in the store."""
        if "signals" not in self._root:
            return 0
        return self._root["signals"].shape[0]

    @property
    def timestamps(self) -> NDArray:
        """All timestamps as a numpy array."""
        if "timestamps" not in self._root:
            return np.array([])
        return np.asarray(self._root["timestamps"])

    def read_latest(self, n: int = 1) -> dict[str, Any]:
        """Read the last *n* frames.

        Returns
        -------
        dict
            ``signals`` : ndarray, shape ``(n, n_nds, n_tau)``
            ``timestamps`` : ndarray, shape ``(n,)``
        """
        total = self.n_frames
        if total == 0:
            return {"signals": np.array([]), "timestamps": np.array([])}
        n = min(n, total)
        return {
            "signals": np.asarray(self._root["signals"][-n:]),
            "timestamps": np.asarray(self._root["timestamps"][-n:]),
        }

    def to_dataset(self) -> "WIMPDataset":
        """Convert the store into a :class:`~wimp.io.WIMPDataset`.

        Signals are averaged across frames.  If ``field_values`` are
        present they become the ``field_timeseries``.

        Returns
        -------
        WIMPDataset
        """
        from wimp.io import WIMPDataset

        tau = np.asarray(self._root["tau_array"])

        if "signals" not in self._root or self.n_frames == 0:
            return WIMPDataset(tau_array=tau)

        # Average across the frame axis -> (n_nds, n_tau)
        signals = np.asarray(self._root["signals"])
        avg_signal = np.mean(signals, axis=0)

        field_ts = None
        if "field_values" in self._root:
            # (n_results, n_nds) -> transpose to (n_nds, n_results)
            field_ts = np.asarray(self._root["field_values"]).T

        return WIMPDataset(
            protocol="realtime_stream",
            tau_array=tau,
            signal=avg_signal,
            field_timeseries=field_ts,
            metadata={
                "n_frames": int(self.n_frames),
                "duration_s": float(
                    self.timestamps[-1] - self.timestamps[0]
                ) if len(self.timestamps) > 1 else 0.0,
            },
        )
