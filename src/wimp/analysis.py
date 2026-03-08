"""Advanced signal analysis for time-domain NV magnetometry.

Includes noise spectroscopy via dynamical decoupling, power spectral
density estimation, SNR computation, stimulus-triggered averaging,
and conduction-velocity measurement along the C. elegans nerve cord.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch, correlate, coherence as _coherence

from wimp.constants import GAMMA_NV

# ---------------------------------------------------------------------------
# Noise spectroscopy
# ---------------------------------------------------------------------------


def noise_spectroscopy(
    coherence_vs_tau: NDArray,
    tau_array: NDArray,
    n_pulses: int,
    *,
    amplitude: float = 1.0,
) -> dict[str, NDArray]:
    """Extract the environmental noise spectrum S(f) from a DD sweep.

    Uses the relation:

    .. math::
        \\chi(\\tau) = 8 N \\tau (2\\pi \\gamma)^2 S(f_0)

    where *f₀* = 1/(2τ) and coherence = exp(−χ).

    Parameters
    ----------
    coherence_vs_tau : ndarray
        Measured coherence (normalised 0–1) at each τ.
    tau_array : ndarray
        Free-evolution half-spacings (seconds).
    n_pulses : int
        Number of π pulses.
    amplitude : float
        If the raw signal rather than coherence is passed, divide by
        this to normalise.

    Returns
    -------
    result : dict
        ``freq`` (Hz), ``spectral_density`` (T²/Hz).
    """
    if n_pulses < 1:
        raise ValueError("n_pulses must be >= 1")

    tau = np.asarray(tau_array, dtype=float)
    coh = np.asarray(coherence_vs_tau, dtype=float) / amplitude

    # Filter out tau <= 0 entries to avoid division by zero
    valid = tau > 0
    tau = tau[valid]
    coh = coh[valid]

    # Clip to avoid log(0)
    coh = np.clip(coh, 1e-10, 1.0)
    chi = -np.log(coh)

    freq = 1.0 / (2.0 * tau)
    gamma_sq = (2.0 * np.pi * GAMMA_NV) ** 2
    S = chi / (8.0 * n_pulses * tau * gamma_sq)

    # Sort by ascending frequency
    order = np.argsort(freq)
    return {
        "freq": freq[order],
        "spectral_density": S[order],
    }


# ---------------------------------------------------------------------------
# PSD and SNR
# ---------------------------------------------------------------------------


def psd_welch(
    timeseries: NDArray,
    fs: float,
    *,
    nperseg: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Power spectral density via Welch's method.

    Parameters
    ----------
    timeseries : ndarray
        1-D time series.
    fs : float
        Sampling rate (Hz).
    nperseg : int, optional
        Segment length.  Defaults to min(256, len(timeseries)).

    Returns
    -------
    freq : ndarray (Hz)
    psd : ndarray (unit² / Hz)
    """
    ts = np.asarray(timeseries, dtype=float).ravel()
    if nperseg is None:
        nperseg = min(256, len(ts))
    freq, psd = welch(ts, fs=fs, nperseg=nperseg)
    return freq, psd


def snr_estimate(
    timeseries: NDArray,
    fs: float,
    signal_band: tuple[float, float],
    noise_band: tuple[float, float],
) -> float:
    """Estimate SNR from signal and noise frequency bands.

    Parameters
    ----------
    timeseries : ndarray
    fs : float
        Sampling rate (Hz).
    signal_band : (f_low, f_high)
        Frequency band containing the signal (Hz).
    noise_band : (f_low, f_high)
        Frequency band for noise estimation (Hz).

    Returns
    -------
    snr : float
        Linear (not dB) signal-to-noise ratio.
    """
    freq, psd = psd_welch(timeseries, fs)

    sig_mask = (freq >= signal_band[0]) & (freq <= signal_band[1])
    noise_mask = (freq >= noise_band[0]) & (freq <= noise_band[1])

    sig_power = np.mean(psd[sig_mask]) if sig_mask.any() else 0.0
    noise_power = np.mean(psd[noise_mask]) if noise_mask.any() else 1e-30

    return float(sig_power / noise_power)


# ---------------------------------------------------------------------------
# Stimulus-triggered averaging
# ---------------------------------------------------------------------------


def stimulus_triggered_average(
    field_timeseries: NDArray,
    trigger_times: NDArray,
    fs: float,
    window: tuple[float, float] = (-0.05, 0.2),
    *,
    baseline: tuple[float, float] | None = (-0.05, 0.0),
) -> dict[str, NDArray]:
    """Compute stimulus-triggered average of a field time series.

    Parameters
    ----------
    field_timeseries : ndarray, shape (n_channels, n_timepoints) or (n_timepoints,)
    trigger_times : ndarray
        Stimulus onset times (seconds).
    fs : float
        Sampling rate (Hz).
    window : (pre, post) in seconds
        Epoch window relative to trigger.
    baseline : (start, end) in seconds, optional
        Baseline correction window relative to trigger.

    Returns
    -------
    result : dict
        ``time`` (seconds relative to trigger), ``average`` (same
        units as input), ``std``, ``n_trials``.
    """
    ts = np.atleast_2d(np.asarray(field_timeseries, dtype=float))
    n_ch, n_t = ts.shape
    triggers = np.asarray(trigger_times, dtype=float)

    pre_samp = int(round(-window[0] * fs))
    post_samp = int(round(window[1] * fs))
    epoch_len = pre_samp + post_samp

    epochs = []
    for trig in triggers:
        idx = int(round(trig * fs))
        start = idx - pre_samp
        end = idx + post_samp
        if start < 0 or end > n_t:
            continue
        epoch = ts[:, start:end]

        if baseline is not None:
            bl_start = int(round((baseline[0] - window[0]) * fs))
            bl_end = int(round((baseline[1] - window[0]) * fs))
            bl_start = max(0, bl_start)
            bl_end = min(epoch_len, bl_end)
            bl_mean = np.mean(epoch[:, bl_start:bl_end], axis=1, keepdims=True)
            epoch = epoch - bl_mean

        epochs.append(epoch)

    if not epochs:
        time_axis = np.linspace(window[0], window[1], epoch_len)
        return {
            "time": time_axis,
            "average": np.zeros((n_ch, epoch_len)),
            "std": np.zeros((n_ch, epoch_len)),
            "n_trials": 0,
        }

    epochs_arr = np.stack(epochs, axis=0)  # (n_trials, n_ch, epoch_len)
    avg = np.mean(epochs_arr, axis=0)
    std = np.std(epochs_arr, axis=0)
    time_axis = np.linspace(window[0], window[1], epoch_len)

    return {
        "time": time_axis,
        "average": avg,
        "std": std,
        "n_trials": len(epochs),
    }


# ---------------------------------------------------------------------------
# Conduction velocity
# ---------------------------------------------------------------------------


def conduction_velocity(
    field_timeseries: NDArray,
    nd_positions: NDArray,
    fs: float,
    *,
    arc_lengths: NDArray | None = None,
) -> dict[str, Any]:
    """Estimate conduction velocity from time delays between NDs.

    Uses cross-correlation to find the lag between pairs of adjacent
    nanodiamonds.

    Parameters
    ----------
    field_timeseries : (n_nds, n_timepoints) array
    nd_positions : (n_nds, 3) array (metres)
    fs : float
        Sampling rate (Hz).
    arc_lengths : (n_nds,) array, optional
        Arc-length positions along the nerve cord.  If None, uses
        Euclidean distance between consecutive NDs.

    Returns
    -------
    result : dict
        ``velocity`` (m/s, median), ``pair_velocities``,
        ``pair_delays`` (seconds), ``pair_distances`` (metres).
    """
    ts = np.asarray(field_timeseries, dtype=float)
    n_nds = ts.shape[0]

    if n_nds < 2:
        return {"velocity": 0.0, "pair_velocities": np.array([]),
                "pair_delays": np.array([]), "pair_distances": np.array([])}

    if arc_lengths is not None:
        arc = np.asarray(arc_lengths, dtype=float)
    else:
        arc = np.concatenate([[0], np.cumsum(
            np.linalg.norm(np.diff(nd_positions, axis=0), axis=1)
        )])

    delays = []
    distances = []

    for i in range(n_nds - 1):
        a = ts[i] - np.mean(ts[i])
        b = ts[i + 1] - np.mean(ts[i + 1])
        cc = correlate(b, a, mode="full")
        lags = np.arange(-len(a) + 1, len(a)) / fs
        peak = lags[np.argmax(cc)]
        delays.append(peak)
        distances.append(arc[i + 1] - arc[i])

    delays = np.array(delays)
    distances = np.array(distances)

    velocities = np.where(np.abs(delays) > 0, distances / delays, 0.0)

    return {
        "velocity": float(np.median(velocities[velocities != 0])) if np.any(velocities != 0) else 0.0,
        "pair_velocities": velocities,
        "pair_delays": delays,
        "pair_distances": distances,
    }


# ---------------------------------------------------------------------------
# Neural event detection
# ---------------------------------------------------------------------------


def detect_neural_events(
    field_timeseries: NDArray,
    fs: float,
    *,
    threshold_sigma: float = 3.0,
) -> dict[str, Any]:
    """Detect neural events as threshold crossings in the field trace.

    Parameters
    ----------
    field_timeseries : (n_timepoints,) or (n_channels, n_timepoints)
    fs : float
        Sampling rate (Hz).
    threshold_sigma : float
        Detection threshold in units of standard deviation.

    Returns
    -------
    result : dict
        ``event_times`` (seconds), ``event_amplitudes``,
        ``event_indices``, ``threshold`` (Tesla).
    """
    ts = np.atleast_2d(np.asarray(field_timeseries, dtype=float))
    # Pool across channels
    pooled = np.mean(ts, axis=0)

    mu = np.mean(pooled)
    sigma = np.std(pooled)
    threshold = mu + threshold_sigma * sigma

    above = pooled > threshold
    # Find rising edges
    edges = np.diff(above.astype(int))
    event_idx = np.where(edges == 1)[0] + 1

    event_times = event_idx / fs
    event_amplitudes = pooled[event_idx]

    return {
        "event_times": event_times,
        "event_amplitudes": event_amplitudes,
        "event_indices": event_idx,
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# Echo phase extraction
# ---------------------------------------------------------------------------


def phase_from_echo(
    signal_i: NDArray,
    signal_q: NDArray,
) -> NDArray:
    """Extract unwrapped phase from in-phase / quadrature echo data.

    Parameters
    ----------
    signal_i, signal_q : ndarray
        I and Q components of the echo readout.

    Returns
    -------
    phase : ndarray
        Unwrapped phase (radians).
    """
    raw_phase = np.arctan2(signal_q, signal_i)
    return np.unwrap(raw_phase)


# ---------------------------------------------------------------------------
# Spike sorting / waveform classification
# ---------------------------------------------------------------------------


@dataclass
class SortedSpike:
    """A classified neural event."""

    cluster_id: int
    waveform: NDArray
    timestamp: float
    amplitude: float
    channel_index: int = 0


def extract_event_waveforms(
    field_timeseries: NDArray,
    event_indices: NDArray,
    *,
    window_pre: int = 10,
    window_post: int = 20,
    fs: float = 1.0,
) -> dict[str, Any]:
    """Cut waveform windows around detected events.

    Parameters
    ----------
    field_timeseries : (n_channels, n_timepoints) or (n_timepoints,)
    event_indices : (n_events,) sample indices
    window_pre, window_post : int
        Samples before/after each event to capture.
    fs : float
        Sampling rate (used to compute timestamps).

    Returns
    -------
    result : dict
        ``waveforms`` (n_events, window_len), ``event_indices``,
        ``timestamps`` (seconds).
    """
    ts = np.atleast_2d(np.asarray(field_timeseries, dtype=float))
    pooled = np.mean(ts, axis=0)
    n_t = len(pooled)
    idx = np.asarray(event_indices, dtype=int)

    waveforms = []
    valid_idx = []
    for i in idx:
        start = i - window_pre
        end = i + window_post
        if start < 0 or end > n_t:
            continue
        waveforms.append(pooled[start:end])
        valid_idx.append(i)

    if not waveforms:
        wl = window_pre + window_post
        return {
            "waveforms": np.empty((0, wl)),
            "event_indices": np.array([], dtype=int),
            "timestamps": np.array([]),
        }

    return {
        "waveforms": np.array(waveforms),
        "event_indices": np.array(valid_idx, dtype=int),
        "timestamps": np.array(valid_idx, dtype=float) / fs,
    }


def classify_waveforms(
    waveforms: NDArray,
    *,
    n_clusters: int = 3,
    n_components: int = 3,
) -> dict[str, Any]:
    """Classify event waveforms via PCA + k-means.

    Parameters
    ----------
    waveforms : (n_events, window_len)
    n_clusters : int
    n_components : int
        PCA dimensions to keep.

    Returns
    -------
    result : dict
        ``labels``, ``centroids``, ``pca_scores``,
        ``explained_variance``, ``cluster_waveforms``.
    """
    from scipy.cluster.vq import kmeans2

    W = np.asarray(waveforms, dtype=float)
    n_events = W.shape[0]
    if n_events == 0:
        return {
            "labels": np.array([], dtype=int),
            "centroids": np.empty((0, n_components)),
            "pca_scores": np.empty((0, n_components)),
            "explained_variance": np.array([]),
            "cluster_waveforms": [],
        }

    # PCA via SVD
    mean_wf = W.mean(axis=0)
    centered = W - mean_wf
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    nc = min(n_components, U.shape[1])
    scores = U[:, :nc] * S[:nc]
    explained = (S[:nc] ** 2) / max(np.sum(S ** 2), 1e-30)

    k = min(n_clusters, n_events)
    centroids, labels = kmeans2(scores, k, minit="++")

    cluster_waveforms = []
    for c in range(k):
        mask = labels == c
        if mask.any():
            cluster_waveforms.append(W[mask].mean(axis=0))
        else:
            cluster_waveforms.append(np.zeros(W.shape[1]))

    return {
        "labels": labels,
        "centroids": centroids,
        "pca_scores": scores,
        "explained_variance": explained,
        "cluster_waveforms": cluster_waveforms,
    }


def sort_spikes(
    field_timeseries: NDArray,
    fs: float,
    *,
    threshold_sigma: float = 3.0,
    n_clusters: int = 3,
    window_pre: int = 10,
    window_post: int = 20,
    n_components: int = 3,
) -> list[SortedSpike]:
    """Detect, extract, and classify neural events end-to-end.

    Chains :func:`detect_neural_events` → :func:`extract_event_waveforms`
    → :func:`classify_waveforms`.

    Returns
    -------
    spikes : list of SortedSpike
    """
    det = detect_neural_events(field_timeseries, fs, threshold_sigma=threshold_sigma)
    wf = extract_event_waveforms(
        field_timeseries, det["event_indices"],
        window_pre=window_pre, window_post=window_post, fs=fs,
    )
    if wf["waveforms"].shape[0] == 0:
        return []

    cls = classify_waveforms(
        wf["waveforms"], n_clusters=n_clusters, n_components=n_components,
    )

    spikes = []
    for k in range(len(wf["event_indices"])):
        spikes.append(SortedSpike(
            cluster_id=int(cls["labels"][k]),
            waveform=wf["waveforms"][k],
            timestamp=wf["timestamps"][k],
            amplitude=float(wf["waveforms"][k].max()),
        ))
    return spikes


# ---------------------------------------------------------------------------
# Blind source separation (ICA / PCA)
# ---------------------------------------------------------------------------


def pca_decompose(
    data: NDArray,
    *,
    n_components: int | None = None,
) -> dict[str, Any]:
    """PCA decomposition of multi-channel data.

    Parameters
    ----------
    data : (n_channels, n_timepoints)
    n_components : int, optional
        Components to keep.  ``None`` → all.

    Returns
    -------
    result : dict
        ``components`` (n_comp, n_timepoints), ``mixing`` (n_channels, n_comp),
        ``explained_variance``, ``mean``.
    """
    X = np.asarray(data, dtype=float)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    mean = X.mean(axis=1, keepdims=True)
    centered = X - mean

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    nc = n_components if n_components is not None else len(S)
    nc = min(nc, len(S))

    components = Vt[:nc]  # (nc, n_t)
    mixing = U[:, :nc] * S[:nc]  # (n_ch, nc)
    explained = (S[:nc] ** 2) / max(np.sum(S ** 2), 1e-30)

    return {
        "components": components,
        "mixing": mixing,
        "explained_variance": explained,
        "mean": mean.ravel(),
    }


def ica_decompose(
    data: NDArray,
    *,
    n_components: int | None = None,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> dict[str, Any]:
    """FastICA-based blind source separation.

    Parameters
    ----------
    data : (n_channels, n_timepoints)
    n_components : int, optional
    max_iter : int
    tol : float

    Returns
    -------
    result : dict
        ``sources`` (n_comp, n_timepoints), ``mixing`` (n_channels, n_comp),
        ``unmixing`` (n_comp, n_channels).
    """
    X = np.asarray(data, dtype=float)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    n_ch, n_t = X.shape
    nc = n_components if n_components is not None else n_ch
    nc = min(nc, n_ch)

    # Centre and whiten via PCA
    mean = X.mean(axis=1, keepdims=True)
    centered = X - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    K = (U[:, :nc] / S[:nc]).T  # whitening: (nc, n_ch)
    Z = K @ centered  # (nc, n_t)

    # FastICA with logcosh nonlinearity
    rng = np.random.default_rng(42)
    W = rng.standard_normal((nc, nc))
    W, _ = np.linalg.qr(W)

    for _ in range(max_iter):
        WZ = W @ Z
        g = np.tanh(WZ)
        gp = 1.0 - g ** 2
        W_new = (g @ Z.T) / n_t - (gp.mean(axis=1, keepdims=True) * W)
        W_new, _ = np.linalg.qr(W_new)
        if np.max(np.abs(np.abs(np.sum(W_new * W, axis=1)) - 1.0)) < tol:
            W = W_new
            break
        W = W_new

    sources = W @ Z  # (nc, n_t)
    unmixing = W @ K  # (nc, n_ch)
    mixing = np.linalg.pinv(unmixing)  # (n_ch, nc)

    return {
        "sources": sources,
        "mixing": mixing,
        "unmixing": unmixing,
    }


# ---------------------------------------------------------------------------
# Functional connectivity
# ---------------------------------------------------------------------------


def cross_correlation_matrix(
    timeseries: NDArray,
    *,
    max_lag: int | None = None,
    normalized: bool = True,
) -> dict[str, Any]:
    """Pairwise cross-correlation matrix for multi-channel timeseries.

    Parameters
    ----------
    timeseries : (n_channels, n_timepoints)
    max_lag : int, optional
        Maximum lag in samples.  ``None`` → zero-lag only.
    normalized : bool
        If True, return Pearson correlation coefficients.

    Returns
    -------
    result : dict
        ``matrix`` (n_ch, n_ch), ``lags`` (n_ch, n_ch) optimal lag per pair.
    """
    ts = np.atleast_2d(np.asarray(timeseries, dtype=float))
    n_ch, n_t = ts.shape

    corr_matrix = np.zeros((n_ch, n_ch))
    lag_matrix = np.zeros((n_ch, n_ch), dtype=int)

    for i in range(n_ch):
        a = ts[i] - ts[i].mean()
        norm_a = np.sqrt(np.sum(a ** 2)) if normalized else 1.0
        for j in range(n_ch):
            b = ts[j] - ts[j].mean()
            norm_b = np.sqrt(np.sum(b ** 2)) if normalized else 1.0
            cc = correlate(a, b, mode="full")
            if normalized and norm_a > 0 and norm_b > 0:
                cc = cc / (norm_a * norm_b)
            lags = np.arange(-n_t + 1, n_t)

            if max_lag is not None:
                mask = np.abs(lags) <= max_lag
                cc = cc[mask]
                lags = lags[mask]

            peak_idx = np.argmax(np.abs(cc))
            corr_matrix[i, j] = cc[peak_idx]
            lag_matrix[i, j] = lags[peak_idx]

    return {"matrix": corr_matrix, "lags": lag_matrix}


def granger_causality(
    timeseries: NDArray,
    *,
    max_lag: int = 5,
) -> dict[str, Any]:
    """Pairwise Granger causality F-statistics.

    Uses a simple bivariate autoregressive model.

    Parameters
    ----------
    timeseries : (n_channels, n_timepoints)
    max_lag : int
        Model order (number of lags).

    Returns
    -------
    result : dict
        ``f_stat`` (n_ch, n_ch), ``matrix`` (n_ch, n_ch).
    """
    ts = np.atleast_2d(np.asarray(timeseries, dtype=float))
    n_ch, n_t = ts.shape

    f_matrix = np.zeros((n_ch, n_ch))

    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                continue
            y = ts[i, max_lag:]
            n = len(y)

            # Restricted: own lags only
            X_r = np.column_stack([
                ts[i, max_lag - k - 1: n_t - k - 1] for k in range(max_lag)
            ])
            # Unrestricted: own + other lags
            X_u = np.column_stack([
                X_r,
                *[ts[j, max_lag - k - 1: n_t - k - 1][:, np.newaxis]
                  for k in range(max_lag)],
            ])

            rss_r = np.sum((y - X_r @ np.linalg.lstsq(X_r, y, rcond=None)[0]) ** 2)
            rss_u = np.sum((y - X_u @ np.linalg.lstsq(X_u, y, rcond=None)[0]) ** 2)

            df_num = max_lag
            df_den = max(n - 2 * max_lag, 1)
            if rss_u > 0:
                f_matrix[i, j] = ((rss_r - rss_u) / df_num) / (rss_u / df_den)

    return {"f_stat": f_matrix, "matrix": f_matrix}


def mutual_information(
    timeseries: NDArray,
    *,
    n_bins: int = 32,
) -> NDArray:
    """Pairwise mutual information between channels.

    Parameters
    ----------
    timeseries : (n_channels, n_timepoints)
    n_bins : int

    Returns
    -------
    mi_matrix : (n_channels, n_channels)
    """
    ts = np.atleast_2d(np.asarray(timeseries, dtype=float))
    n_ch = ts.shape[0]
    mi = np.zeros((n_ch, n_ch))

    for i in range(n_ch):
        for j in range(i, n_ch):
            hist_2d, _, _ = np.histogram2d(ts[i], ts[j], bins=n_bins)
            pxy = hist_2d / hist_2d.sum()
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)

            nz = pxy > 0
            outer = px[:, None] * py[None, :]
            mi_val = np.sum(pxy[nz] * np.log(pxy[nz] / outer[nz]))
            mi[i, j] = mi_val
            mi[j, i] = mi_val

    return mi


# ---------------------------------------------------------------------------
# Transfer entropy
# ---------------------------------------------------------------------------


def transfer_entropy(
    source: NDArray,
    target: NDArray,
    *,
    lag: int = 1,
    n_bins: int = 32,
    normalize: bool = False,
) -> float:
    r"""Compute transfer entropy from *source* to *target*.

    A model-free measure of directed information flow.

    .. math::
        TE_{X \to Y} = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)

    Parameters
    ----------
    source, target : ndarray, shape ``(n_timepoints,)``
    lag : int
        Time lag in samples.
    n_bins : int
        Number of histogram bins for discretisation.
    normalize : bool
        If True, normalise to [0, 1] by dividing by
        ``H(target_future | target_past)``.

    Returns
    -------
    te : float
        Transfer entropy (nats).
    """
    src = np.asarray(source, dtype=float).ravel()
    tgt = np.asarray(target, dtype=float).ravel()
    n = min(len(src), len(tgt))

    tgt_future = tgt[lag:n]
    tgt_past = tgt[:n - lag]
    src_past = src[:n - lag]

    # Digitise into bins
    def _digitise(arr: NDArray) -> NDArray:
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return np.zeros(len(arr), dtype=int)
        edges = np.linspace(lo - 1e-10, hi + 1e-10, n_bins + 1)
        return np.digitize(arr, edges[1:-1])

    tf = _digitise(tgt_future)
    tp = _digitise(tgt_past)
    sp = _digitise(src_past)

    # Joint and marginal histograms
    # p(tf, tp, sp)
    joint_3d = np.zeros((n_bins, n_bins, n_bins))
    for i in range(len(tf)):
        joint_3d[tf[i], tp[i], sp[i]] += 1
    joint_3d /= max(joint_3d.sum(), 1e-30)

    # p(tf, tp)
    p_tf_tp = joint_3d.sum(axis=2)
    # p(tp, sp)
    p_tp_sp = joint_3d.sum(axis=0)
    # p(tp)
    p_tp = p_tf_tp.sum(axis=0)

    # TE = sum p(tf, tp, sp) * log( p(tf|tp,sp) / p(tf|tp) )
    #    = sum p(tf, tp, sp) * log( p(tf,tp,sp) * p(tp) / (p(tp,sp) * p(tf,tp)) )
    te = 0.0
    nz = joint_3d > 0
    for idx in zip(*np.where(nz)):
        i_tf, i_tp, i_sp = idx
        p3 = joint_3d[i_tf, i_tp, i_sp]
        p_tp_val = p_tp[i_tp]
        p_tp_sp_val = p_tp_sp[i_tp, i_sp]
        p_tf_tp_val = p_tf_tp[i_tf, i_tp]
        if p_tp_sp_val > 0 and p_tf_tp_val > 0 and p_tp_val > 0:
            te += p3 * np.log(p3 * p_tp_val / (p_tp_sp_val * p_tf_tp_val))

    if normalize:
        # H(tf | tp) = - sum p(tf, tp) * log(p(tf|tp))
        h_cond = 0.0
        nz2 = p_tf_tp > 0
        for idx2 in zip(*np.where(nz2)):
            i_tf2, i_tp2 = idx2
            p2 = p_tf_tp[i_tf2, i_tp2]
            ptp = p_tp[i_tp2]
            if ptp > 0:
                h_cond -= p2 * np.log(p2 / ptp)
        if h_cond > 0:
            te /= h_cond

    return float(max(te, 0.0))


def transfer_entropy_matrix(
    timeseries: NDArray,
    *,
    lag: int = 1,
    n_bins: int = 32,
    normalize: bool = False,
) -> NDArray:
    """Pairwise transfer entropy matrix.

    Parameters
    ----------
    timeseries : ndarray, shape ``(n_channels, n_timepoints)``
    lag, n_bins, normalize
        Forwarded to :func:`transfer_entropy`.

    Returns
    -------
    te_matrix : ndarray, shape ``(n_channels, n_channels)``
        ``te_matrix[i, j]`` is the TE from channel *j* to channel *i*.
    """
    ts = np.atleast_2d(np.asarray(timeseries, dtype=float))
    n_ch = ts.shape[0]
    mat = np.zeros((n_ch, n_ch))

    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                continue
            mat[i, j] = transfer_entropy(
                ts[j], ts[i], lag=lag, n_bins=n_bins, normalize=normalize,
            )

    return mat


# ---------------------------------------------------------------------------
# Spectral coherence
# ---------------------------------------------------------------------------


def spectral_coherence(
    x: NDArray,
    y: NDArray,
    fs: float,
    *,
    nperseg: int = 256,
) -> dict[str, NDArray]:
    """Magnitude-squared coherence between two signals.

    Thin wrapper around :func:`scipy.signal.coherence`.

    Parameters
    ----------
    x, y : ndarray, shape ``(n_timepoints,)``
    fs : float
        Sampling rate (Hz).
    nperseg : int
        Segment length for Welch's method.

    Returns
    -------
    result : dict
        ``freq`` (Hz), ``coherence`` (0–1).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    seg = min(nperseg, len(x), len(y))
    freq, coh = _coherence(x, y, fs=fs, nperseg=seg)
    return {"freq": freq, "coherence": coh}


def coherence_matrix(
    timeseries: NDArray,
    fs: float,
    *,
    nperseg: int = 256,
    freq_band: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Pairwise coherence matrix for multi-channel data.

    Parameters
    ----------
    timeseries : ndarray, shape ``(n_channels, n_timepoints)``
    fs : float
        Sampling rate (Hz).
    nperseg : int
        Segment length for Welch's method.
    freq_band : (f_low, f_high), optional
        If provided, average coherence within this band.  Otherwise
        average over all frequencies.

    Returns
    -------
    result : dict
        ``matrix`` (n_ch, n_ch), ``freq`` (Hz).
    """
    ts = np.atleast_2d(np.asarray(timeseries, dtype=float))
    n_ch = ts.shape[0]

    mat = np.ones((n_ch, n_ch))
    freq_out = None

    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            res = spectral_coherence(ts[i], ts[j], fs, nperseg=nperseg)
            freq = res["freq"]
            coh = res["coherence"]
            if freq_out is None:
                freq_out = freq

            if freq_band is not None:
                mask = (freq >= freq_band[0]) & (freq <= freq_band[1])
                val = float(np.mean(coh[mask])) if mask.any() else 0.0
            else:
                val = float(np.mean(coh))

            mat[i, j] = val
            mat[j, i] = val

    if freq_out is None:
        freq_out = np.array([])

    return {"matrix": mat, "freq": freq_out}


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------


def permutation_test(
    data_a: NDArray,
    data_b: NDArray,
    *,
    statistic: str = "mean_diff",
    n_permutations: int = 10000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Two-sample permutation test.

    Parameters
    ----------
    data_a, data_b : ndarray, shape ``(n_samples,)``
    statistic : str
        ``"mean_diff"`` or ``"t_stat"``.
    n_permutations : int
    seed : int, optional

    Returns
    -------
    result : dict
        ``statistic`` (name), ``observed``, ``p_value``,
        ``null_distribution``.
    """
    a = np.asarray(data_a, dtype=float).ravel()
    b = np.asarray(data_b, dtype=float).ravel()
    rng = np.random.default_rng(seed)

    na = len(a)
    pooled = np.concatenate([a, b])

    def _compute_stat(x: NDArray, y: NDArray) -> float:
        if statistic == "mean_diff":
            return float(np.mean(x) - np.mean(y))
        if statistic == "t_stat":
            mx, my = np.mean(x), np.mean(y)
            sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
            se = np.sqrt(sx ** 2 / len(x) + sy ** 2 / len(y))
            return float((mx - my) / max(se, 1e-30))
        raise ValueError(f"Unknown statistic: {statistic!r}")

    observed = _compute_stat(a, b)

    null = np.empty(n_permutations)
    for k in range(n_permutations):
        perm = rng.permutation(pooled)
        null[k] = _compute_stat(perm[:na], perm[na:])

    p_value = float(np.mean(np.abs(null) >= np.abs(observed)))

    return {
        "statistic": statistic,
        "observed": observed,
        "p_value": p_value,
        "null_distribution": null,
    }


def cluster_permutation_test(
    timeseries_a: NDArray,
    timeseries_b: NDArray,
    *,
    threshold: float = 2.0,
    n_permutations: int = 1000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Cluster-based permutation test (Maris & Oostenveld 2007).

    Compares two groups of time series and identifies time clusters
    with statistically significant differences, correcting for
    multiple comparisons.

    Parameters
    ----------
    timeseries_a : ndarray, shape ``(n_obs, n_timepoints)``
        Observations in group A.
    timeseries_b : ndarray, shape ``(n_obs, n_timepoints)``
        Observations in group B.
    threshold : float
        t-statistic threshold for forming clusters.
    n_permutations : int
    seed : int, optional

    Returns
    -------
    result : dict
        ``significant_clusters`` (list of (start, end) tuples),
        ``cluster_p_values``, ``t_values``.
    """
    a = np.asarray(timeseries_a, dtype=float)
    b = np.asarray(timeseries_b, dtype=float)
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if b.ndim == 1:
        b = b[np.newaxis, :]

    n_t = min(a.shape[1], b.shape[1])
    a = a[:, :n_t]
    b = b[:, :n_t]
    rng = np.random.default_rng(seed)

    def _t_values(x: NDArray, y: NDArray) -> NDArray:
        nx, ny = x.shape[0], y.shape[0]
        mx, my = x.mean(axis=0), y.mean(axis=0)
        sx = x.std(axis=0, ddof=1) if nx > 1 else np.zeros(n_t)
        sy = y.std(axis=0, ddof=1) if ny > 1 else np.zeros(n_t)
        se = np.sqrt(sx ** 2 / max(nx, 1) + sy ** 2 / max(ny, 1))
        se = np.where(se > 0, se, 1e-30)
        return (mx - my) / se

    def _find_clusters(t_vals: NDArray) -> list[tuple[int, int, float]]:
        above = np.abs(t_vals) > threshold
        clusters: list[tuple[int, int, float]] = []
        in_cluster = False
        start = 0
        for i in range(len(above)):
            if above[i] and not in_cluster:
                start = i
                in_cluster = True
            elif not above[i] and in_cluster:
                cluster_stat = float(np.sum(t_vals[start:i]))
                clusters.append((start, i, cluster_stat))
                in_cluster = False
        if in_cluster:
            cluster_stat = float(np.sum(t_vals[start:]))
            clusters.append((start, len(above), cluster_stat))
        return clusters

    t_obs = _t_values(a, b)
    obs_clusters = _find_clusters(t_obs)

    if not obs_clusters:
        return {
            "significant_clusters": [],
            "cluster_p_values": np.array([]),
            "t_values": t_obs,
        }

    obs_stats = np.array([abs(c[2]) for c in obs_clusters])

    # Build null distribution of max cluster statistic
    pooled = np.concatenate([a, b], axis=0)
    na = a.shape[0]
    null_max = np.zeros(n_permutations)

    for k in range(n_permutations):
        perm_idx = rng.permutation(pooled.shape[0])
        pa = pooled[perm_idx[:na]]
        pb = pooled[perm_idx[na:]]
        t_perm = _t_values(pa, pb)
        perm_clusters = _find_clusters(t_perm)
        if perm_clusters:
            null_max[k] = max(abs(c[2]) for c in perm_clusters)

    # p-value per observed cluster
    p_vals = np.array([float(np.mean(null_max >= s)) for s in obs_stats])

    sig_clusters = [
        (c[0], c[1]) for c, p in zip(obs_clusters, p_vals) if p < 0.05
    ]

    return {
        "significant_clusters": sig_clusters,
        "cluster_p_values": p_vals,
        "t_values": t_obs,
    }


def fdr_correct(
    p_values: NDArray,
    *,
    alpha: float = 0.05,
    method: str = "bh",
) -> dict[str, Any]:
    """Benjamini-Hochberg (or Benjamini-Yekutieli) FDR correction.

    Parameters
    ----------
    p_values : ndarray
        Uncorrected p-values.
    alpha : float
        Desired FDR level.
    method : str
        ``"bh"`` (Benjamini-Hochberg) or ``"by"`` (Benjamini-Yekutieli).

    Returns
    -------
    result : dict
        ``rejected`` (bool array), ``corrected_p``, ``threshold``.
    """
    pv = np.asarray(p_values, dtype=float).ravel()
    m = len(pv)
    if m == 0:
        return {"rejected": np.array([], dtype=bool),
                "corrected_p": np.array([]),
                "threshold": 0.0}

    sorted_idx = np.argsort(pv)
    sorted_pv = pv[sorted_idx]
    ranks = np.arange(1, m + 1)

    if method == "bh":
        correction = 1.0
    elif method == "by":
        correction = np.sum(1.0 / ranks)
    else:
        raise ValueError(f"Unknown FDR method: {method!r}")

    # Adjusted p-values (step-up)
    adjusted = np.minimum(1.0, sorted_pv * m * correction / ranks)
    # Enforce monotonicity (from end to start)
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Map back to original order
    corrected = np.empty(m)
    corrected[sorted_idx] = adjusted

    # Threshold: largest p-value that passes
    passing = sorted_pv <= alpha * ranks / (m * correction)
    if passing.any():
        thresh = float(alpha * np.max(ranks[passing]) / (m * correction))
    else:
        thresh = 0.0

    return {
        "rejected": corrected <= alpha,
        "corrected_p": corrected,
        "threshold": thresh,
    }
