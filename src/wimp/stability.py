"""Signal stability analysis for time-domain NV magnetometry.

Provides drift correction, Allan variance / deviation computation, and
compressed-sensing reconstruction for undersampled tau sweeps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Drift correction
# ---------------------------------------------------------------------------


def drift_correct(
    timeseries: NDArray,
    *,
    method: str = "polynomial",
    degree: int = 2,
    cutoff: float | None = None,
    fs: float = 1.0,
) -> dict[str, Any]:
    """Remove slow drift from a time series.

    Parameters
    ----------
    timeseries : ndarray
        Shape ``(n_timepoints,)`` or ``(n_channels, n_timepoints)``.
    method : str
        ``"polynomial"`` (default), ``"highpass"``, or ``"linear"``.
    degree : int
        Polynomial degree (used when *method* is ``"polynomial"``).
    cutoff : float
        Highpass cutoff frequency in Hz (required when *method* is
        ``"highpass"``).
    fs : float
        Sampling rate in Hz (used by ``"highpass"``).

    Returns
    -------
    result : dict
        ``corrected`` (same shape as input), ``drift`` (estimated drift),
        ``method``.
    """
    squeeze = False
    ts = np.asarray(timeseries, dtype=float)
    if ts.ndim == 1:
        ts = ts[np.newaxis, :]
        squeeze = True

    n_ch, n_t = ts.shape

    if method == "polynomial":
        x = np.arange(n_t, dtype=float)
        drift = np.empty_like(ts)
        for i in range(n_ch):
            coeffs = np.polyfit(x, ts[i], degree)
            drift[i] = np.polyval(coeffs, x)
        corrected = ts - drift

    elif method == "linear":
        from scipy.signal import detrend

        corrected = np.empty_like(ts)
        drift = np.empty_like(ts)
        for i in range(n_ch):
            corrected[i] = detrend(ts[i], type="linear")
            drift[i] = ts[i] - corrected[i]

    elif method == "highpass":
        from scipy.signal import butter, sosfiltfilt

        if cutoff is None:
            raise ValueError("cutoff is required for highpass drift correction")
        nyq = fs / 2.0
        if cutoff >= nyq:
            # Cutoff above Nyquist → nothing to filter, return copy
            corrected = ts.copy()
            drift = np.zeros_like(ts)
        else:
            sos = butter(4, cutoff / nyq, btype="high", output="sos")
            corrected = np.empty_like(ts)
            drift = np.empty_like(ts)
            for i in range(n_ch):
                corrected[i] = sosfiltfilt(sos, ts[i])
                drift[i] = ts[i] - corrected[i]
    else:
        raise ValueError(
            f"Unknown drift correction method: {method!r}. "
            "Choose from 'polynomial', 'linear', 'highpass'."
        )

    if squeeze:
        corrected = corrected[0]
        drift = drift[0]

    return {"corrected": corrected, "drift": drift, "method": method}


# ---------------------------------------------------------------------------
# Allan variance
# ---------------------------------------------------------------------------


def allan_variance(
    timeseries: NDArray,
    fs: float,
    *,
    taus: NDArray | None = None,
    overlap: bool = True,
) -> dict[str, NDArray]:
    r"""Compute Allan variance and Allan deviation.

    Parameters
    ----------
    timeseries : ndarray, shape ``(n_timepoints,)``
        Stationary time series (e.g. repeated B-field measurements).
    fs : float
        Sampling rate (Hz).
    taus : ndarray, optional
        Averaging times at which to evaluate.  If *None*, a
        logarithmically spaced set is generated automatically.
    overlap : bool
        If True (default), use overlapping Allan variance for better
        statistical efficiency.

    Returns
    -------
    result : dict
        ``taus`` (seconds), ``avar``, ``adev``, ``adev_err``.
    """
    y = np.asarray(timeseries, dtype=float).ravel()
    n = len(y)
    dt = 1.0 / fs

    if taus is None:
        max_m = max(n // 3, 1)
        exponents = np.linspace(0, np.log10(max_m), min(50, max_m))
        ms = np.unique(np.round(10 ** exponents).astype(int))
        ms = ms[ms >= 1]
        taus_out = ms * dt
    else:
        taus_out = np.asarray(taus, dtype=float)
        ms = np.unique(np.round(taus_out * fs).astype(int))
        ms = ms[ms >= 1]
        taus_out = ms * dt

    avar_list = []
    adev_list = []
    err_list = []

    for m in ms:
        if overlap:
            # Overlapping Allan variance
            # Phase integral (cumulative sum) approach
            cum = np.cumsum(np.insert(y, 0, 0))  # length n+1
            # Averaged samples: mean of m consecutive values
            # avg_k = (cum[k+m] - cum[k]) / m  for k = 0, ..., n-m
            if 2 * m > n:
                continue
            diff = (cum[2 * m :] - 2 * cum[m : n - m + 1] + cum[: n - 2 * m + 1])
            n_pairs = len(diff)
            if n_pairs == 0:
                continue
            av = np.sum(diff ** 2) / (2.0 * n_pairs * m ** 2)
        else:
            # Non-overlapping Allan variance
            n_blocks = n // m
            if n_blocks < 2:
                continue
            truncated = y[: n_blocks * m].reshape(n_blocks, m)
            block_means = truncated.mean(axis=1)
            diffs = np.diff(block_means)
            av = 0.5 * np.mean(diffs ** 2)

        ad = np.sqrt(max(av, 0.0))
        # Approximate error: adev / sqrt(N_eff)
        n_eff = (n / m) if overlap else (n // m - 1)
        n_eff = max(n_eff, 1)
        err = ad / np.sqrt(2.0 * n_eff)

        avar_list.append(av)
        adev_list.append(ad)
        err_list.append(err)

    # Trim taus to match computed values
    valid_ms = []
    idx = 0
    for m in ms:
        if overlap:
            if 2 * m <= n and (n - 2 * m + 1) > 0:
                valid_ms.append(m)
        else:
            if n // m >= 2:
                valid_ms.append(m)

    taus_final = np.array(valid_ms, dtype=float) * dt

    return {
        "taus": taus_final,
        "avar": np.array(avar_list),
        "adev": np.array(adev_list),
        "adev_err": np.array(err_list),
    }


# ---------------------------------------------------------------------------
# Compressed-sensing reconstruction
# ---------------------------------------------------------------------------


def compressed_sensing_reconstruct(
    signal: NDArray,
    tau_measured: NDArray,
    tau_full: NDArray,
    *,
    n_components: int | None = None,
    alpha: float = 0.01,
) -> dict[str, Any]:
    """Reconstruct a full tau sweep from undersampled measurements.

    Uses a Discrete Cosine Transform (DCT) basis with L1-regularised
    least-squares (basis pursuit / LASSO).

    Parameters
    ----------
    signal : ndarray, shape ``(n_measured,)``
        Measured signal values at the sampled tau points.
    tau_measured : ndarray, shape ``(n_measured,)``
        Tau values where measurements were taken (seconds).
    tau_full : ndarray, shape ``(n_full,)``
        Full set of tau values to reconstruct onto (seconds).
    n_components : int, optional
        Number of DCT basis functions.  Defaults to ``len(tau_full)``.
    alpha : float
        L1 regularisation weight.

    Returns
    -------
    result : dict
        ``reconstructed`` (n_full,), ``tau_full``, ``residual``,
        ``coefficients``.
    """
    from scipy.optimize import minimize

    sig = np.asarray(signal, dtype=float).ravel()
    tau_m = np.asarray(tau_measured, dtype=float).ravel()
    tau_f = np.asarray(tau_full, dtype=float).ravel()
    n_full = len(tau_f)
    n_meas = len(sig)

    if n_components is None:
        n_components = n_full

    n_components = min(n_components, n_full)

    # Build DCT-II basis on the full grid: Psi[i, k] = cos(pi*k*(2i+1) / (2*N))
    idx = np.arange(n_full)
    ks = np.arange(n_components)
    Psi = np.cos(np.pi * ks[np.newaxis, :] * (2 * idx[:, np.newaxis] + 1) / (2 * n_full))

    # Measurement operator: select rows corresponding to measured tau positions
    # Find nearest index in tau_full for each tau_measured
    meas_idx = np.array([np.argmin(np.abs(tau_f - t)) for t in tau_m])
    Phi = np.eye(n_full)[meas_idx]  # (n_meas, n_full)

    # Sensing matrix
    A = Phi @ Psi  # (n_meas, n_components)

    # Objective: 0.5 * ||A @ x - signal||^2 + alpha * ||x||_1
    # Use smooth L1 approximation for gradient-based optimisation
    eps = 1e-8

    def objective(x: NDArray) -> float:
        residual = A @ x - sig
        data_term = 0.5 * np.sum(residual ** 2)
        l1_term = alpha * np.sum(np.sqrt(x ** 2 + eps))
        return float(data_term + l1_term)

    def gradient(x: NDArray) -> NDArray:
        residual = A @ x - sig
        grad_data = A.T @ residual
        grad_l1 = alpha * x / np.sqrt(x ** 2 + eps)
        return grad_data + grad_l1

    x0 = np.zeros(n_components)
    result = minimize(objective, x0, jac=gradient, method="L-BFGS-B",
                      options={"maxiter": 500})

    coefficients = result.x
    reconstructed = Psi @ coefficients

    # Residual at measurement points
    residual = sig - (Phi @ reconstructed)

    return {
        "reconstructed": reconstructed,
        "tau_full": tau_f,
        "residual": residual,
        "coefficients": coefficients,
    }
