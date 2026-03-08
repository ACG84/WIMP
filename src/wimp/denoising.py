"""Signal denoising for time-domain NV magnetometry data.

Provides four denoising methods — Savitzky-Golay, Butterworth lowpass,
moving average, and wavelet thresholding — plus a unified dispatcher.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def denoise(
    signal: NDArray,
    method: str = "savgol",
    *,
    axis: int = -1,
    **kwargs: Any,
) -> NDArray:
    """Apply a denoising filter to a measurement signal.

    Parameters
    ----------
    signal : ndarray
        Input signal.  Typically ``(n_tau,)`` or ``(n_nds, n_tau)``.
    method : str
        One of ``"savgol"``, ``"butterworth"``, ``"moving_average"``,
        or ``"wavelet"``.
    axis : int
        Array axis along which to filter (default ``-1``).
    **kwargs
        Forwarded to the underlying method function.

    Returns
    -------
    denoised : ndarray
        Same shape as *signal*.
    """
    _methods = {
        "savgol": denoise_savgol,
        "butterworth": denoise_butterworth,
        "moving_average": denoise_moving_average,
        "wavelet": denoise_wavelet,
    }
    func = _methods.get(method)
    if func is None:
        raise ValueError(
            f"Unknown denoising method {method!r}. "
            f"Choose from {sorted(_methods)}"
        )
    return func(signal, axis=axis, **kwargs)


# ---------------------------------------------------------------------------
# Savitzky-Golay
# ---------------------------------------------------------------------------


def denoise_savgol(
    signal: NDArray,
    *,
    window_length: int = 11,
    polyorder: int = 3,
    axis: int = -1,
) -> NDArray:
    """Savitzky-Golay smoothing filter.

    Parameters
    ----------
    signal : ndarray
    window_length : int
        Length of the filter window (must be odd and > polyorder).
    polyorder : int
        Polynomial order for the local fit.
    axis : int
        Axis along which to apply the filter.

    Returns
    -------
    denoised : ndarray
    """
    from scipy.signal import savgol_filter

    signal = np.asarray(signal, dtype=float)
    n = signal.shape[axis]
    wl = min(window_length, n)
    if wl % 2 == 0:
        wl = max(wl - 1, 1)
    po = min(polyorder, wl - 1)
    return savgol_filter(signal, window_length=wl, polyorder=po, axis=axis)


# ---------------------------------------------------------------------------
# Butterworth lowpass
# ---------------------------------------------------------------------------


def denoise_butterworth(
    signal: NDArray,
    *,
    cutoff: float,
    fs: float,
    order: int = 4,
    axis: int = -1,
) -> NDArray:
    """Zero-phase Butterworth lowpass filter.

    Parameters
    ----------
    signal : ndarray
    cutoff : float
        Lowpass cutoff frequency (Hz).
    fs : float
        Sampling rate (Hz).
    order : int
        Filter order.
    axis : int
        Axis along which to apply the filter.

    Returns
    -------
    denoised : ndarray
    """
    from scipy.signal import butter, sosfiltfilt

    signal = np.asarray(signal, dtype=float)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    nyq = fs / 2.0
    if cutoff >= nyq:
        return signal.copy()
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, signal, axis=axis)


# ---------------------------------------------------------------------------
# Moving average
# ---------------------------------------------------------------------------


def denoise_moving_average(
    signal: NDArray,
    *,
    window_size: int = 5,
    axis: int = -1,
) -> NDArray:
    """Uniform moving-average filter.

    Parameters
    ----------
    signal : ndarray
    window_size : int
        Averaging window width.
    axis : int
        Axis along which to smooth.

    Returns
    -------
    denoised : ndarray
    """
    from scipy.ndimage import uniform_filter1d

    signal = np.asarray(signal, dtype=float)
    ws = max(1, min(window_size, signal.shape[axis]))
    return uniform_filter1d(signal, size=ws, axis=axis, mode="nearest")


# ---------------------------------------------------------------------------
# Wavelet thresholding
# ---------------------------------------------------------------------------


def denoise_wavelet(
    signal: NDArray,
    *,
    wavelet: str = "db4",
    level: int | None = None,
    threshold_mode: str = "soft",
    axis: int = -1,
) -> NDArray:
    """Discrete wavelet transform denoising with universal threshold.

    Requires the ``PyWavelets`` package (``pip install PyWavelets``).

    Parameters
    ----------
    signal : ndarray
    wavelet : str
        Wavelet name (e.g. ``"db4"``, ``"sym6"``, ``"coif3"``).
    level : int, optional
        Decomposition level.  ``None`` → automatic.
    threshold_mode : str
        ``"soft"`` or ``"hard"`` thresholding.
    axis : int
        Axis along which to denoise.

    Returns
    -------
    denoised : ndarray
    """
    try:
        import pywt
    except ImportError:
        raise ImportError(
            "Wavelet denoising requires PyWavelets. "
            "Install it with: pip install PyWavelets"
        ) from None

    signal = np.asarray(signal, dtype=float)

    def _denoise_1d(x: NDArray) -> NDArray:
        n = len(x)
        max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
        lev = level if level is not None else max(1, max_level)
        lev = min(lev, max_level) if max_level > 0 else 1

        coeffs = pywt.wavedec(x, wavelet, level=lev)
        # Universal threshold: sigma * sqrt(2 * log(n))
        # Estimate sigma from finest detail coefficients
        detail = coeffs[-1]
        sigma = float(np.median(np.abs(detail)) / 0.6745) if len(detail) > 0 else 1.0
        threshold = sigma * np.sqrt(2.0 * np.log(max(n, 2)))

        denoised_coeffs = [coeffs[0]]  # keep approximation untouched
        for c in coeffs[1:]:
            denoised_coeffs.append(pywt.threshold(c, threshold, mode=threshold_mode))

        rec = pywt.waverec(denoised_coeffs, wavelet)
        return rec[:n]  # trim to original length

    return np.apply_along_axis(_denoise_1d, axis, signal)
