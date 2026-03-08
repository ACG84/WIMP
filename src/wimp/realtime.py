"""Real-time data acquisition and streaming for WIMP.

Provides a :class:`RealtimeProcessor` that ingests measurement frames
from a live data source (hardware callback, socket, or file watcher),
buffers them, performs incremental fitting, and emits results via
user-registered callbacks.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from wimp.io import WIMPDataset
from wimp.relaxation import fit_ramsey, fit_t2_decay, fit_t1_recovery, fit_odmr


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RealtimeConfig:
    """Configuration for real-time processing."""

    protocol: str = "ramsey"
    """Measurement protocol."""

    buffer_size: int = 100
    """Maximum number of frames kept in the rolling buffer."""

    fit_interval: float = 0.1
    """Minimum seconds between fit updates."""

    averaging: int = 1
    """Number of frames to average before fitting."""

    fitting_params: dict[str, Any] = field(default_factory=dict)
    """Extra keyword arguments forwarded to the fitting routine."""

    denoise_method: str | None = None
    """Optional denoising applied to each frame before fitting."""

    denoise_params: dict[str, Any] = field(default_factory=dict)
    """Parameters forwarded to the denoising function."""

    use_kalman: bool = False
    """If True, apply Kalman filtering to B-field estimates."""

    kalman_state_model: str = "position"
    """Kalman state model: ``"position"`` or ``"position_velocity"``."""

    kalman_process_noise: float = 1e-9
    """Process noise variance for the Kalman filter."""

    kalman_measurement_noise: float = 1e-8
    """Measurement noise variance for the Kalman filter."""


# ---------------------------------------------------------------------------
# Measurement frame
# ---------------------------------------------------------------------------


@dataclass
class MeasurementFrame:
    """A single time-domain measurement snapshot.

    Attributes
    ----------
    tau_array : ndarray, shape ``(n_tau,)``
        Free-evolution times.
    signal : ndarray, shape ``(n_nds, n_tau)`` or ``(n_tau,)``
        Signal for one or more nanodiamonds.
    timestamp : float
        Acquisition timestamp (seconds since epoch or arbitrary).
    metadata : dict
        Per-frame metadata (e.g. temperature, trigger info).
    """

    tau_array: NDArray
    signal: NDArray
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RealtimeResult:
    """Result emitted after each incremental fit update.

    Attributes
    ----------
    fits : list[dict]
        Per-ND fit results from the latest averaged data.
    field_values : ndarray | None
        Extracted B-field values (one per ND) if applicable.
    timestamp : float
        Timestamp of the most recent frame used.
    frame_count : int
        Total frames ingested so far.
    """

    fits: list[dict[str, Any]]
    field_values: NDArray | None
    timestamp: float
    frame_count: int
    kalman_states: list[NDArray] | None = None
    """Per-ND Kalman state vectors (for diagnostics)."""


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------


class KalmanFilter:
    """1-D Kalman filter for tracking a scalar quantity (e.g. B-field).

    Supports two state models:

    ``"position"``
        State = ``[B]``.  Constant model.

    ``"position_velocity"``
        State = ``[B, dB/dt]``.  Constant-velocity model — tracks drift.

    Parameters
    ----------
    state_model : str
    process_noise : float
    measurement_noise : float
    dt : float
        Default time step between predict/update cycles.
    """

    def __init__(
        self,
        state_model: str = "position",
        process_noise: float = 1e-9,
        measurement_noise: float = 1e-8,
        dt: float = 1.0,
    ) -> None:
        self._model = state_model
        self._dt = dt
        self._initialized = False

        if state_model == "position":
            self._dim = 1
            self._x = np.zeros(1)
            self._P = np.eye(1)
            self._H = np.array([[1.0]])
            self._Q = np.array([[process_noise]])
            self._R = np.array([[measurement_noise]])
        elif state_model == "position_velocity":
            self._dim = 2
            self._x = np.zeros(2)
            self._P = np.eye(2)
            self._H = np.array([[1.0, 0.0]])
            self._Q = np.diag([process_noise, process_noise * 10.0])
            self._R = np.array([[measurement_noise]])
        else:
            raise ValueError(
                f"state_model must be 'position' or 'position_velocity', "
                f"got {state_model!r}"
            )

    def _transition(self, dt: float) -> NDArray:
        if self._model == "position":
            return np.array([[1.0]])
        return np.array([[1.0, dt], [0.0, 1.0]])

    def predict(self, dt: float | None = None) -> None:
        """Propagate state and covariance forward."""
        if not self._initialized:
            return
        dt = dt if dt is not None else self._dt
        F = self._transition(dt)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._Q

    def update(self, measurement: float) -> float:
        """Incorporate a new measurement; return filtered value."""
        z = np.array([measurement])
        if not self._initialized:
            self._x[0] = measurement
            self._P = np.eye(self._dim) * self._R[0, 0]
            self._initialized = True
            return measurement

        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + (K @ y).ravel()
        self._P = (np.eye(self._dim) - K @ self._H) @ self._P
        return float(self._x[0])

    def reset(self) -> None:
        """Reset to uninitialised state."""
        self._initialized = False
        self._x = np.zeros(self._dim)
        self._P = np.eye(self._dim)

    @property
    def state(self) -> NDArray:
        """Current state vector."""
        return self._x.copy()


# ---------------------------------------------------------------------------
# Core processor
# ---------------------------------------------------------------------------


class RealtimeProcessor:
    """Streaming real-time processor for NV magnetometry data.

    Usage
    -----
    >>> proc = RealtimeProcessor(RealtimeConfig(protocol="ramsey"))
    >>> proc.on_result(my_callback)     # register a listener
    >>> proc.start()                     # start background processing
    >>> proc.push(frame)                 # feed frames from your DAQ
    >>> proc.stop()                      # graceful shutdown

    The processor runs a background thread that periodically fits the
    latest averaged data and invokes registered callbacks with a
    :class:`RealtimeResult`.
    """

    def __init__(self, config: RealtimeConfig | None = None) -> None:
        self.config = config or RealtimeConfig()
        self._buffer: deque[MeasurementFrame] = deque(
            maxlen=self.config.buffer_size
        )
        self._callbacks: list[Callable[[RealtimeResult], Any]] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_count = 0
        self._last_fit_time = 0.0
        self._kalman_filters: dict[int, KalmanFilter] = {}

    # -- public API ----------------------------------------------------------

    def on_result(self, callback: Callable[[RealtimeResult], Any]) -> None:
        """Register a callback invoked with each new :class:`RealtimeResult`."""
        self._callbacks.append(callback)

    def push(self, frame: MeasurementFrame) -> None:
        """Ingest a new measurement frame.

        Thread-safe: may be called from a hardware callback or any thread.
        """
        with self._lock:
            self._buffer.append(frame)
            self._frame_count += 1

    def push_arrays(
        self,
        tau_array: NDArray,
        signal: NDArray,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Convenience method to push raw numpy arrays as a frame."""
        ts = timestamp if timestamp is not None else time.time()
        frame = MeasurementFrame(
            tau_array=np.asarray(tau_array),
            signal=np.asarray(signal),
            timestamp=ts,
            metadata=metadata or {},
        )
        self.push(frame)

    def start(self) -> None:
        """Start the background processing thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="wimp-realtime"
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background thread and wait for it to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def process_once(self) -> RealtimeResult | None:
        """Perform a single fit on the current buffer (synchronous).

        Returns ``None`` if the buffer has fewer frames than
        ``config.averaging``.
        """
        return self._fit_latest()

    def clear(self) -> None:
        """Clear the frame buffer."""
        with self._lock:
            self._buffer.clear()
            self._frame_count = 0
        self._kalman_filters = {}

    @property
    def frame_count(self) -> int:
        """Total number of frames pushed so far."""
        return self._frame_count

    @property
    def buffer_len(self) -> int:
        """Current number of frames in the rolling buffer."""
        return len(self._buffer)

    @property
    def is_running(self) -> bool:
        """Whether the background thread is active."""
        return self._running

    # -- internals -----------------------------------------------------------

    def _run_loop(self) -> None:
        """Background loop that periodically fits and emits results."""
        while self._running:
            now = time.monotonic()
            if now - self._last_fit_time >= self.config.fit_interval:
                result = self._fit_latest()
                if result is not None:
                    self._last_fit_time = now
                    for cb in self._callbacks:
                        try:
                            cb(result)
                        except Exception:
                            pass  # don't let a bad callback kill the loop
            time.sleep(min(self.config.fit_interval / 4, 0.025))

    def _fit_latest(self) -> RealtimeResult | None:
        """Average the most recent frames and fit."""
        with self._lock:
            n_avg = self.config.averaging
            if len(self._buffer) < n_avg:
                return None
            frames = list(self._buffer)[-n_avg:]
            count = self._frame_count

        # Average signals
        signals = np.stack([np.atleast_2d(f.signal) for f in frames], axis=0)
        avg_signal = np.mean(signals, axis=0)  # (n_nds, n_tau)
        tau = np.asarray(frames[-1].tau_array)
        latest_ts = frames[-1].timestamp

        # Denoise (if requested)
        if self.config.denoise_method is not None:
            from wimp.denoising import denoise
            avg_signal = denoise(
                avg_signal,
                method=self.config.denoise_method,
                **self.config.denoise_params,
            )

        # Fit each ND
        n_nds = avg_signal.shape[0]
        fits: list[dict[str, Any]] = []
        field_vals: list[float] = []

        for i in range(n_nds):
            fit = self._fit_single(tau, avg_signal[i])
            fits.append(fit)
            field_vals.append(fit.get("b_field", float("nan")))

        field_arr = np.array(field_vals) if field_vals else None

        # Kalman filtering
        kalman_states: list[NDArray] | None = None
        if self.config.use_kalman and field_arr is not None:
            kalman_states = []
            for i in range(n_nds):
                if i not in self._kalman_filters:
                    self._kalman_filters[i] = KalmanFilter(
                        state_model=self.config.kalman_state_model,
                        process_noise=self.config.kalman_process_noise,
                        measurement_noise=self.config.kalman_measurement_noise,
                        dt=self.config.fit_interval,
                    )
                kf = self._kalman_filters[i]
                kf.predict()
                field_arr[i] = kf.update(float(field_arr[i]))
                kalman_states.append(kf.state)

        return RealtimeResult(
            fits=fits,
            field_values=field_arr,
            timestamp=latest_ts,
            frame_count=count,
            kalman_states=kalman_states,
        )

    def _fit_single(self, tau: NDArray, signal: NDArray) -> dict[str, Any]:
        """Fit a single ND curve according to the configured protocol."""
        protocol = self.config.protocol
        kw = self.config.fitting_params

        if protocol == "ramsey":
            return fit_ramsey(tau, signal, **kw)
        if protocol in ("echo", "hahn_echo"):
            return fit_t2_decay(tau, signal, **kw)
        if protocol == "t1":
            return fit_t1_recovery(tau, signal, **kw)
        if protocol == "cw_odmr":
            return fit_odmr(tau, signal, **kw)
        # Fallback: return raw data
        return {"tau": tau, "signal": signal}


# ---------------------------------------------------------------------------
# File-watcher source
# ---------------------------------------------------------------------------


class FileWatcherSource:
    """Watch a directory for new measurement files and feed a processor.

    Monitors a directory for new ``.npz`` or ``.npy`` files and pushes
    each as a :class:`MeasurementFrame` to a :class:`RealtimeProcessor`.

    Parameters
    ----------
    watch_dir : str or Path
        Directory to monitor.
    processor : RealtimeProcessor
        Target processor.
    poll_interval : float
        Seconds between directory polls (default 0.5).
    tau_key : str
        Key/filename for the tau array in npz files (default ``"tau_array"``).
    signal_key : str
        Key for the signal array (default ``"signal"``).
    """

    def __init__(
        self,
        watch_dir: str,
        processor: RealtimeProcessor,
        poll_interval: float = 0.5,
        tau_key: str = "tau_array",
        signal_key: str = "signal",
    ) -> None:
        from pathlib import Path

        self.watch_dir = Path(watch_dir)
        self.processor = processor
        self.poll_interval = poll_interval
        self.tau_key = tau_key
        self.signal_key = signal_key
        self._seen: set[str] = set()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start watching in a background thread."""
        if self._running:
            return
        # Snapshot existing files so we only process new ones
        self._seen = {
            p.name for p in self.watch_dir.iterdir()
            if p.suffix in (".npz", ".npy")
        } if self.watch_dir.exists() else set()
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="wimp-filewatcher"
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop watching."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _poll_loop(self) -> None:
        while self._running:
            if self.watch_dir.exists():
                for p in sorted(self.watch_dir.iterdir()):
                    if p.name in self._seen:
                        continue
                    if p.suffix == ".npz":
                        self._ingest_npz(p)
                        self._seen.add(p.name)
                    elif p.suffix == ".npy":
                        self._ingest_npy(p)
                        self._seen.add(p.name)
            time.sleep(self.poll_interval)

    def _ingest_npz(self, path) -> None:
        data = np.load(path)
        tau = data.get(self.tau_key)
        sig = data.get(self.signal_key)
        if tau is not None and sig is not None:
            self.processor.push_arrays(
                tau, sig, timestamp=time.time(),
                metadata={"source_file": str(path)},
            )

    def _ingest_npy(self, path) -> None:
        arr = np.load(path)
        # Assume rows = [tau, signal(s)]
        if arr.ndim == 2 and arr.shape[0] >= 2:
            tau = arr[0]
            sig = arr[1:]
            self.processor.push_arrays(
                tau, sig, timestamp=time.time(),
                metadata={"source_file": str(path)},
            )


# ---------------------------------------------------------------------------
# Socket source (TCP)
# ---------------------------------------------------------------------------


class TCPSource:
    """Receive measurement frames over a TCP socket.

    Expects each message to be a length-prefixed numpy ``.npz`` buffer::

        [4 bytes big-endian uint32 length][npz bytes]

    The npz must contain ``tau_array`` and ``signal`` arrays.

    Parameters
    ----------
    host : str
        Bind address (default ``"localhost"``).
    port : int
        Bind port (default ``5555``).
    processor : RealtimeProcessor
        Target processor.
    """

    def __init__(
        self,
        processor: RealtimeProcessor,
        host: str = "localhost",
        port: int = 5555,
    ) -> None:
        self.host = host
        self.port = port
        self.processor = processor
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the TCP listener in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._listen, daemon=True, name="wimp-tcp"
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _listen(self) -> None:
        import io
        import socket
        import struct

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(1.0)
        srv.bind((self.host, self.port))
        srv.listen(1)

        try:
            while self._running:
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                conn.settimeout(5.0)
                try:
                    self._handle_conn(conn)
                finally:
                    conn.close()
        finally:
            srv.close()

    def _handle_conn(self, conn) -> None:
        import io
        import struct

        while self._running:
            # Read 4-byte length header
            header = self._recv_exact(conn, 4)
            if header is None:
                break
            length = struct.unpack(">I", header)[0]
            if length == 0 or length > 100_000_000:
                break
            payload = self._recv_exact(conn, length)
            if payload is None:
                break

            data = np.load(io.BytesIO(payload))
            tau = data.get("tau_array")
            sig = data.get("signal")
            if tau is not None and sig is not None:
                self.processor.push_arrays(
                    tau, sig, timestamp=time.time(),
                )

    @staticmethod
    def _recv_exact(conn, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            try:
                chunk = conn.recv(n - len(buf))
            except (ConnectionError, OSError):
                return None
            if not chunk:
                return None
            buf += chunk
        return buf
