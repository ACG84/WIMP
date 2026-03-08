"""Fitting of time-domain NV relaxation and precession curves.

Provides model functions and non-linear least-squares fits for:
- Ramsey fringes (damped cosine → B-field extraction)
- T₂ echo decays (stretched exponential)
- T₁ spin-lattice recovery
- Bi-exponential decays
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from wimp.constants import GAMMA_NV, D0

# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------


def ramsey_model(
    tau: NDArray,
    amplitude: float,
    frequency: float,
    t2star: float,
    phase: float,
    offset: float,
    stretch: float = 1.0,
) -> NDArray:
    r"""Damped cosine for a Ramsey free-induction decay.

    .. math::
        S(\tau) = A \cos(2\pi f \tau + \varphi)
                  \exp\!\bigl[-(\tau / T_2^*)^n\bigr] + B
    """
    envelope = np.exp(-np.abs(tau / t2star) ** stretch)
    return amplitude * np.cos(2.0 * np.pi * frequency * tau + phase) * envelope + offset


def t2_decay_model(
    tau: NDArray,
    amplitude: float,
    t2: float,
    offset: float,
    stretch: float = 1.0,
) -> NDArray:
    r"""Stretched-exponential echo decay.

    .. math::
        S(\tau) = A \exp\!\bigl[-(2\tau / T_2)^n\bigr] + B
    """
    return amplitude * np.exp(-np.abs(2.0 * tau / t2) ** stretch) + offset


def t1_recovery_model(
    tau: NDArray,
    amplitude: float,
    t1: float,
    contrast: float,
    offset: float,
) -> NDArray:
    r"""Spin-lattice recovery curve.

    .. math::
        S(\tau) = A \bigl(1 - C \exp(-\tau / T_1)\bigr) + B
    """
    return amplitude * (1.0 - contrast * np.exp(-tau / t1)) + offset


def biexponential_decay(
    tau: NDArray,
    a1: float,
    t_fast: float,
    a2: float,
    t_slow: float,
    offset: float,
) -> NDArray:
    r"""Bi-exponential decay with two time constants.

    .. math::
        S(\tau) = a_1 e^{-\tau/T_{\rm fast}}
                + a_2 e^{-\tau/T_{\rm slow}} + B
    """
    return a1 * np.exp(-tau / t_fast) + a2 * np.exp(-tau / t_slow) + offset


# ---------------------------------------------------------------------------
# CW ODMR models
# ---------------------------------------------------------------------------


def lorentzian_dip(
    freq: NDArray,
    center: float,
    amplitude: float,
    linewidth: float,
) -> NDArray:
    r"""Single Lorentzian absorption dip.

    .. math::
        L(f) = -A \frac{\Gamma^2}{(f - f_0)^2 + \Gamma^2}

    Parameters
    ----------
    freq : ndarray
        Microwave frequencies (Hz).
    center : float
        Resonance frequency *f₀* (Hz).
    amplitude : float
        Peak depth (positive value; returned as a negative dip).
    linewidth : float
        Half-width at half-maximum *Γ* (Hz).
    """
    return -amplitude * linewidth**2 / ((freq - center)**2 + linewidth**2)


def odmr_model(
    freq: NDArray,
    baseline: float,
    f_minus: float,
    f_plus: float,
    a_minus: float,
    a_plus: float,
    gamma_minus: float,
    gamma_plus: float,
) -> NDArray:
    r"""Double-Lorentzian ODMR spectrum.

    .. math::
        S(f) = B + L(f; f^-, A^-, \Gamma^-) + L(f; f^+, A^+, \Gamma^+)

    The two dips correspond to the :math:`m_s = \pm 1` transitions at
    :math:`f^\pm = D_0 \pm \gamma_{\rm NV} B`.
    """
    return (
        baseline
        + lorentzian_dip(freq, f_minus, a_minus, gamma_minus)
        + lorentzian_dip(freq, f_plus, a_plus, gamma_plus)
    )


def odmr_single_dip_model(
    freq: NDArray,
    baseline: float,
    center: float,
    amplitude: float,
    linewidth: float,
) -> NDArray:
    r"""Single-dip ODMR spectrum (zero / low-field case).

    .. math::
        S(f) = B + L(f; f_0, A, \Gamma)
    """
    return baseline + lorentzian_dip(freq, center, amplitude, linewidth)


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------


def _initial_frequency_guess(tau: NDArray, signal: NDArray) -> float:
    """Estimate dominant oscillation frequency via FFT."""
    if len(tau) < 2:
        return 1e6  # fallback for single-element or empty arrays
    dt = np.median(np.diff(tau))
    if dt <= 0:
        return 1e6  # fallback
    sig_detrend = signal - np.mean(signal)
    n = len(sig_detrend)
    fft_mag = np.abs(np.fft.rfft(sig_detrend))
    freqs = np.fft.rfftfreq(n, d=dt)
    # Ignore DC bin
    fft_mag[0] = 0.0
    idx = int(np.argmax(fft_mag))
    return float(freqs[idx]) if freqs[idx] > 0 else 1e6


def goodness_of_fit(
    data: NDArray,
    model: NDArray,
    n_params: int,
) -> dict[str, float]:
    """Compute R², χ², AIC, and BIC for a fit."""
    residuals = data - model
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((data - np.mean(data)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = len(data)
    chi_squared = ss_res / max(n - n_params, 1)
    # AIC / BIC (assuming Gaussian errors)
    log_lik = -0.5 * n * np.log(2.0 * np.pi * ss_res / n) - n / 2.0 if ss_res > 0 else 0.0
    aic = 2.0 * n_params - 2.0 * log_lik
    bic = n_params * np.log(n) - 2.0 * log_lik if n > 0 else aic
    return {
        "r_squared": r_squared,
        "chi_squared": chi_squared,
        "aic": float(aic),
        "bic": float(bic),
    }


# ---------------------------------------------------------------------------
# High-level fitting routines
# ---------------------------------------------------------------------------


def fit_ramsey(
    tau: NDArray,
    signal: NDArray,
    *,
    p0: dict[str, float] | None = None,
    fix_stretch: bool = True,
) -> dict[str, Any]:
    """Fit a Ramsey fringe and extract the DC magnetic field.

    Parameters
    ----------
    tau : ndarray
        Free-evolution times (seconds).
    signal : ndarray
        Measured contrast signal (dimensionless).
    p0 : dict, optional
        Initial guesses keyed by parameter name.
    fix_stretch : bool
        If True, fix the stretch exponent to 1 (simple exponential).

    Returns
    -------
    result : dict
        Fitted parameters, covariance, goodness-of-fit, and
        ``b_field`` in Tesla.
    """
    tau = np.asarray(tau, dtype=float)
    signal = np.asarray(signal, dtype=float)

    # Defaults
    amp0 = (np.max(signal) - np.min(signal)) / 2.0
    off0 = np.mean(signal)
    freq0 = _initial_frequency_guess(tau, signal)
    t2s0 = float(tau[-1]) / 2.0 if len(tau) > 1 else 1e-6

    defaults: dict[str, float] = {
        "amplitude": amp0,
        "frequency": freq0,
        "t2star": t2s0,
        "phase": 0.0,
        "offset": off0,
    }
    if p0:
        defaults.update(p0)

    if fix_stretch:
        def _model(t, amp, freq, t2s, phi, off):
            return ramsey_model(t, amp, freq, t2s, phi, off, stretch=1.0)

        guess = [defaults["amplitude"], defaults["frequency"],
                 defaults["t2star"], defaults["phase"], defaults["offset"]]
        lower = [0, 0, 1e-12, -2 * np.pi, -np.inf]
        upper = [np.inf, np.inf, np.inf, 2 * np.pi, np.inf]
        popt, pcov = curve_fit(_model, tau, signal, p0=guess,
                               bounds=(lower, upper), maxfev=20000)
        names = ["amplitude", "frequency", "t2star", "phase", "offset"]
        fitted = dict(zip(names, popt))
        fitted["stretch"] = 1.0
    else:
        def _model_s(t, amp, freq, t2s, phi, off, n):
            return ramsey_model(t, amp, freq, t2s, phi, off, stretch=n)

        guess = [defaults["amplitude"], defaults["frequency"],
                 defaults["t2star"], defaults["phase"], defaults["offset"], 1.0]
        lower = [0, 0, 1e-12, -2 * np.pi, -np.inf, 0.5]
        upper = [np.inf, np.inf, np.inf, 2 * np.pi, np.inf, 3.0]
        popt, pcov = curve_fit(_model_s, tau, signal, p0=guess,
                               bounds=(lower, upper), maxfev=20000)
        names = ["amplitude", "frequency", "t2star", "phase", "offset", "stretch"]
        fitted = dict(zip(names, popt))
        _model = lambda t, *a: _model_s(t, *a)  # noqa: E731

    model_vals = ramsey_model(tau, **fitted)
    gof = goodness_of_fit(signal, model_vals, len(popt))

    fitted["b_field"] = fitted["frequency"] / GAMMA_NV
    fitted["cov"] = pcov
    fitted.update(gof)
    return fitted


def fit_t2_decay(
    tau: NDArray,
    signal: NDArray,
    *,
    p0: dict[str, float] | None = None,
    fix_stretch: bool = True,
) -> dict[str, Any]:
    """Fit a Hahn-echo T₂ decay curve.

    Returns
    -------
    result : dict
        Fitted parameters including ``t2`` (seconds).
    """
    tau = np.asarray(tau, dtype=float)
    signal = np.asarray(signal, dtype=float)

    amp0 = float(np.max(signal) - np.min(signal))
    off0 = float(np.min(signal))
    t2_0 = float(tau[-1]) / 2.0 if len(tau) > 1 else 1e-6

    defaults: dict[str, float] = {"amplitude": amp0, "t2": t2_0, "offset": off0}
    if p0:
        defaults.update(p0)

    if fix_stretch:
        def _model(t, amp, t2, off):
            return t2_decay_model(t, amp, t2, off, stretch=1.0)

        guess = [defaults["amplitude"], defaults["t2"], defaults["offset"]]
        lower = [0, 1e-12, -np.inf]
        upper = [np.inf, np.inf, np.inf]
        popt, pcov = curve_fit(_model, tau, signal, p0=guess,
                               bounds=(lower, upper), maxfev=20000)
        names = ["amplitude", "t2", "offset"]
        fitted = dict(zip(names, popt))
        fitted["stretch"] = 1.0
    else:
        def _model_s(t, amp, t2, off, n):
            return t2_decay_model(t, amp, t2, off, stretch=n)

        guess = [defaults["amplitude"], defaults["t2"], defaults["offset"], 1.0]
        lower = [0, 1e-12, -np.inf, 0.5]
        upper = [np.inf, np.inf, np.inf, 3.0]
        popt, pcov = curve_fit(_model_s, tau, signal, p0=guess,
                               bounds=(lower, upper), maxfev=20000)
        names = ["amplitude", "t2", "offset", "stretch"]
        fitted = dict(zip(names, popt))

    model_vals = t2_decay_model(tau, fitted["amplitude"], fitted["t2"],
                                fitted["offset"], fitted["stretch"])
    gof = goodness_of_fit(signal, model_vals, len(popt))

    fitted["cov"] = pcov
    fitted.update(gof)
    return fitted


def fit_t1_recovery(
    tau: NDArray,
    signal: NDArray,
    *,
    p0: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fit a T₁ spin-lattice recovery curve.

    Returns
    -------
    result : dict
        Fitted parameters including ``t1`` (seconds).
    """
    tau = np.asarray(tau, dtype=float)
    signal = np.asarray(signal, dtype=float)

    amp0 = float(np.max(signal))
    t1_0 = float(tau[-1]) / 3.0 if len(tau) > 1 else 1e-3
    c0 = 1.0
    off0 = float(np.min(signal))

    defaults: dict[str, float] = {
        "amplitude": amp0, "t1": t1_0, "contrast": c0, "offset": off0,
    }
    if p0:
        defaults.update(p0)

    guess = [defaults["amplitude"], defaults["t1"],
             defaults["contrast"], defaults["offset"]]
    lower = [0, 1e-12, 0, -np.inf]
    upper = [np.inf, np.inf, 2.0, np.inf]

    popt, pcov = curve_fit(t1_recovery_model, tau, signal, p0=guess,
                           bounds=(lower, upper), maxfev=20000)
    names = ["amplitude", "t1", "contrast", "offset"]
    fitted = dict(zip(names, popt))

    model_vals = t1_recovery_model(tau, *popt)
    gof = goodness_of_fit(signal, model_vals, len(popt))

    fitted["cov"] = pcov
    fitted.update(gof)
    return fitted


def fit_biexponential(
    tau: NDArray,
    signal: NDArray,
    *,
    p0: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fit a bi-exponential decay.

    Returns
    -------
    result : dict
        Fitted parameters: ``a1``, ``t_fast``, ``a2``, ``t_slow``, ``offset``.
    """
    tau = np.asarray(tau, dtype=float)
    signal = np.asarray(signal, dtype=float)

    amp = float(np.max(signal) - np.min(signal))
    t_range = float(tau[-1]) if len(tau) > 1 else 1e-3

    defaults: dict[str, float] = {
        "a1": amp * 0.6, "t_fast": t_range * 0.1,
        "a2": amp * 0.4, "t_slow": t_range * 0.5,
        "offset": float(np.min(signal)),
    }
    if p0:
        defaults.update(p0)

    guess = [defaults["a1"], defaults["t_fast"],
             defaults["a2"], defaults["t_slow"], defaults["offset"]]
    lower = [0, 1e-12, 0, 1e-12, -np.inf]
    upper = [np.inf, np.inf, np.inf, np.inf, np.inf]

    popt, pcov = curve_fit(biexponential_decay, tau, signal, p0=guess,
                           bounds=(lower, upper), maxfev=20000)
    names = ["a1", "t_fast", "a2", "t_slow", "offset"]
    fitted = dict(zip(names, popt))

    model_vals = biexponential_decay(tau, *popt)
    gof = goodness_of_fit(signal, model_vals, len(popt))

    fitted["cov"] = pcov
    fitted.update(gof)
    return fitted


def fit_odmr(
    freq: NDArray,
    signal: NDArray,
    *,
    n_dips: int | None = None,
    p0: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fit a CW ODMR spectrum and extract the DC magnetic field.

    Parameters
    ----------
    freq : ndarray
        Microwave frequency sweep (Hz).
    signal : ndarray
        Measured PL signal (a.u.).
    n_dips : int | None
        Number of dips to fit: ``1`` (single) or ``2`` (double).
        If *None*, auto-detect from the spectrum.
    p0 : dict, optional
        Initial guesses keyed by parameter name.

    Returns
    -------
    result : dict
        Fitted parameters, ``b_field`` (T), ``contrast``,
        ``t2star_estimated`` (s), ``n_dips``, covariance, and
        goodness-of-fit metrics.
    """
    from scipy.signal import find_peaks

    freq = np.asarray(freq, dtype=float)
    signal = np.asarray(signal, dtype=float)

    # --- Auto-detect number of dips ---
    if n_dips is None:
        # Smooth with a simple moving-average kernel
        kernel_size = max(3, len(signal) // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(signal, kernel, mode="same")

        neg_signal = -smoothed
        peaks, props = find_peaks(neg_signal, prominence=0)
        if len(peaks) >= 2:
            # Sort by prominence (deepest first)
            order = np.argsort(props["prominences"])[::-1]
            top2 = peaks[order[:2]]
            top2_sorted = np.sort(top2)
            sep = abs(freq[top2_sorted[1]] - freq[top2_sorted[0]])
            depth_ratio = props["prominences"][order[1]] / props["prominences"][order[0]]
            if sep > 5e6 and depth_ratio > 0.3:
                n_dips = 2
            else:
                n_dips = 1
        else:
            n_dips = 1

    baseline_est = float(np.max(signal))

    if n_dips == 2:
        return _fit_odmr_double(freq, signal, baseline_est, p0)
    return _fit_odmr_single(freq, signal, baseline_est, p0)


def _fit_odmr_single(
    freq: NDArray,
    signal: NDArray,
    baseline_est: float,
    p0: dict[str, float] | None,
) -> dict[str, Any]:
    """Fit a single-dip ODMR spectrum."""
    min_idx = int(np.argmin(signal))
    center_est = float(freq[min_idx])
    amp_est = float(baseline_est - signal[min_idx])
    lw_est = 5e6  # 5 MHz default

    defaults = {
        "baseline": baseline_est,
        "center": center_est,
        "amplitude": max(amp_est, 1e-6),
        "linewidth": lw_est,
    }
    if p0:
        defaults.update(p0)

    freq_lo = float(min(freq[0], freq[-1]))
    freq_hi = float(max(freq[0], freq[-1]))

    guess = [defaults["baseline"], defaults["center"],
             defaults["amplitude"], defaults["linewidth"]]
    lower = [min(0, defaults["baseline"] - abs(defaults["baseline"]) * 0.5),
             freq_lo, 0, 1e3]
    upper = [np.inf, freq_hi, np.inf, 100e6]

    popt, pcov = curve_fit(
        odmr_single_dip_model, freq, signal, p0=guess,
        bounds=(lower, upper), maxfev=20000,
    )
    names = ["baseline", "center", "amplitude", "linewidth"]
    fitted = dict(zip(names, popt))

    model_vals = odmr_single_dip_model(freq, *popt)
    gof = goodness_of_fit(signal, model_vals, len(popt))

    fitted["n_dips"] = 1
    fitted["b_field"] = abs(fitted["center"] - D0) / GAMMA_NV
    fitted["contrast"] = fitted["amplitude"] / fitted["baseline"] if fitted["baseline"] > 0 else 0.0
    fitted["t2star_estimated"] = 1.0 / (np.pi * fitted["linewidth"])
    fitted["cov"] = pcov
    fitted.update(gof)
    return fitted


def _fit_odmr_double(
    freq: NDArray,
    signal: NDArray,
    baseline_est: float,
    p0: dict[str, float] | None,
) -> dict[str, Any]:
    """Fit a double-dip ODMR spectrum."""
    from scipy.signal import find_peaks

    # Smooth before peak-finding to avoid noise peaks
    kernel_size = max(3, len(signal) // 50)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(signal, kernel, mode="same")

    neg_signal = -smoothed
    peaks, props = find_peaks(neg_signal, prominence=0)
    if len(peaks) >= 2:
        order = np.argsort(props["prominences"])[::-1]
        top2 = np.sort(peaks[order[:2]])
        f_minus_est = float(freq[top2[0]])
        f_plus_est = float(freq[top2[1]])
        a_minus_est = float(baseline_est - signal[top2[0]])
        a_plus_est = float(baseline_est - signal[top2[1]])
    else:
        # Fallback: symmetric dips around D0
        f_minus_est = D0 - 14e6
        f_plus_est = D0 + 14e6
        a_minus_est = float(baseline_est - np.min(signal))
        a_plus_est = a_minus_est

    lw_est = 5e6

    defaults = {
        "baseline": baseline_est,
        "f_minus": f_minus_est,
        "f_plus": f_plus_est,
        "a_minus": max(a_minus_est, 1e-6),
        "a_plus": max(a_plus_est, 1e-6),
        "gamma_minus": lw_est,
        "gamma_plus": lw_est,
    }
    if p0:
        defaults.update(p0)

    freq_lo = float(min(freq[0], freq[-1]))
    freq_hi = float(max(freq[0], freq[-1]))

    guess = [defaults["baseline"], defaults["f_minus"], defaults["f_plus"],
             defaults["a_minus"], defaults["a_plus"],
             defaults["gamma_minus"], defaults["gamma_plus"]]
    lower = [min(0, defaults["baseline"] - abs(defaults["baseline"]) * 0.5),
             freq_lo, freq_lo, 0, 0, 1e3, 1e3]
    upper = [np.inf, freq_hi, freq_hi, np.inf, np.inf, 100e6, 100e6]

    popt, pcov = curve_fit(
        odmr_model, freq, signal, p0=guess,
        bounds=(lower, upper), maxfev=20000,
    )
    names = ["baseline", "f_minus", "f_plus", "a_minus", "a_plus",
             "gamma_minus", "gamma_plus"]
    fitted = dict(zip(names, popt))

    model_vals = odmr_model(freq, *popt)
    gof = goodness_of_fit(signal, model_vals, len(popt))

    fitted["n_dips"] = 2
    fitted["splitting"] = abs(fitted["f_plus"] - fitted["f_minus"])
    fitted["b_field"] = fitted["splitting"] / (2.0 * GAMMA_NV)
    fitted["center_freq"] = (fitted["f_plus"] + fitted["f_minus"]) / 2.0
    avg_linewidth = (fitted["gamma_minus"] + fitted["gamma_plus"]) / 2.0
    avg_amplitude = (fitted["a_minus"] + fitted["a_plus"]) / 2.0
    fitted["contrast"] = avg_amplitude / fitted["baseline"] if fitted["baseline"] > 0 else 0.0
    fitted["t2star_estimated"] = 1.0 / (np.pi * avg_linewidth)
    fitted["cov"] = pcov
    fitted.update(gof)
    return fitted


# ---------------------------------------------------------------------------
# Field extraction utilities
# ---------------------------------------------------------------------------


def extract_field_ramsey(fit_result: dict[str, Any]) -> float:
    """Return the DC magnetic field (Tesla) from a Ramsey fit result."""
    return fit_result["frequency"] / GAMMA_NV


def extract_field_echo(phase: float, tau: float) -> float:
    """Return the AC magnetic-field amplitude (Tesla) from an echo
    phase measurement.

    Parameters
    ----------
    phase : float
        Accumulated phase (radians) from echo measurement.
    tau : float
        Free-evolution half-time (seconds).

    Returns
    -------
    b_ac : float
        AC field amplitude (Tesla).
    """
    # φ = 2π γ B_AC × (2τ / π)  for a sine-wave field at f=1/(2τ)
    # Simplified: B = φ / (2π γ 2τ /π) = φ π / (4π γ τ) = φ/(4 γ τ)
    return abs(phase) / (4.0 * GAMMA_NV * tau)


def extract_field_odmr(fit_result: dict[str, Any]) -> float:
    """Return the DC magnetic field (Tesla) from an ODMR fit result.

    For a double-dip fit, uses the splitting.  For a single-dip fit,
    uses the offset of the dip centre from *D₀*.
    """
    if fit_result.get("n_dips", 1) == 2:
        return fit_result["splitting"] / (2.0 * GAMMA_NV)
    return abs(fit_result["center"] - D0) / GAMMA_NV


# ---------------------------------------------------------------------------
# Bayesian fitting (MCMC)
# ---------------------------------------------------------------------------


def fit_ramsey_bayesian(
    tau: NDArray,
    signal: NDArray,
    *,
    n_samples: int = 2000,
    n_burn: int = 500,
    step_size: float = 0.01,
    seed: int | None = None,
) -> dict[str, Any]:
    """Metropolis-Hastings MCMC fit of a Ramsey fringe.

    Provides full posterior distributions for parameter uncertainty
    quantification, rather than the Gaussian approximation from ``curve_fit``.

    Parameters
    ----------
    tau : ndarray
    signal : ndarray
    n_samples : int
        Number of post-burn-in samples to collect.
    n_burn : int
        Burn-in samples to discard.
    step_size : float
        Relative proposal step size.
    seed : int, optional

    Returns
    -------
    result : dict
        ``samples`` (n_samples, n_params) — posterior samples,
        ``param_names``, ``median`` dict of median values,
        ``credible_intervals`` (2.5%, 97.5%) per param,
        ``b_field``, ``acceptance_rate``.
    """
    tau = np.asarray(tau, dtype=float)
    signal = np.asarray(signal, dtype=float)
    rng = np.random.default_rng(seed)

    # Get MLE starting point from standard fit
    mle = fit_ramsey(tau, signal)
    param_names = ["amplitude", "frequency", "t2star", "phase", "offset"]
    current = np.array([mle[n] for n in param_names])
    n_p = len(current)

    # Log-likelihood (Gaussian errors)
    sigma_est = np.std(signal - ramsey_model(tau, *current, stretch=1.0))
    sigma_est = max(sigma_est, 1e-10)

    def log_posterior(params):
        amp, freq, t2s, phi, off = params
        if amp < 0 or freq < 0 or t2s < 1e-15:
            return -np.inf
        model = ramsey_model(tau, amp, freq, t2s, phi, off, stretch=1.0)
        resid = signal - model
        return -0.5 * np.sum(resid ** 2) / sigma_est ** 2

    # Run MCMC
    total = n_samples + n_burn
    samples = np.zeros((total, n_p))
    current_lp = log_posterior(current)
    accepted = 0

    for i in range(total):
        proposal = current + rng.normal(0, step_size * np.abs(current) + 1e-15, size=n_p)
        prop_lp = log_posterior(proposal)
        if np.log(rng.uniform()) < prop_lp - current_lp:
            current = proposal
            current_lp = prop_lp
            accepted += 1
        samples[i] = current

    post_samples = samples[n_burn:]
    medians = {n: float(np.median(post_samples[:, k])) for k, n in enumerate(param_names)}
    medians["b_field"] = medians["frequency"] / GAMMA_NV

    ci = {}
    for k, n in enumerate(param_names):
        ci[n] = (float(np.percentile(post_samples[:, k], 2.5)),
                 float(np.percentile(post_samples[:, k], 97.5)))

    return {
        "samples": post_samples,
        "param_names": param_names,
        "median": medians,
        "credible_intervals": ci,
        "b_field": medians["b_field"],
        "acceptance_rate": accepted / total,
    }


# ---------------------------------------------------------------------------
# Multi-NV joint fitting
# ---------------------------------------------------------------------------


def fit_ramsey_joint(
    tau: NDArray,
    signals: NDArray,
    *,
    shared_params: list[str] | None = None,
) -> dict[str, Any]:
    """Jointly fit Ramsey fringes across multiple NDs with shared parameters.

    Parameters
    ----------
    tau : ndarray
        Free-evolution times (shared).
    signals : (n_nds, n_tau) array
    shared_params : list of str, optional
        Parameter names shared across all NDs.  Default: ``["frequency"]``
        (common B-field).

    Returns
    -------
    result : dict
        ``shared`` — shared parameter values, ``per_nd`` — list of per-ND
        fit dicts, ``b_field`` — from the shared frequency.
    """
    tau = np.asarray(tau, dtype=float)
    signals = np.atleast_2d(np.asarray(signals, dtype=float))
    n_nds = signals.shape[0]

    if shared_params is None:
        shared_params = ["frequency"]

    # First pass: independent fits for initial guesses
    ind_fits = [fit_ramsey(tau, signals[i]) for i in range(n_nds)]

    all_param_names = ["amplitude", "frequency", "t2star", "phase", "offset"]
    shared_set = set(shared_params)
    per_nd_names = [n for n in all_param_names if n not in shared_set]
    shared_names = [n for n in all_param_names if n in shared_set]

    # Build joint parameter vector: [shared..., nd0_per..., nd1_per..., ...]
    n_shared = len(shared_names)
    n_per = len(per_nd_names)
    n_total = n_shared + n_per * n_nds

    p0 = np.zeros(n_total)
    # Shared: average across NDs
    for k, name in enumerate(shared_names):
        p0[k] = np.mean([f[name] for f in ind_fits])
    # Per-ND
    for i in range(n_nds):
        offset = n_shared + i * n_per
        for k, name in enumerate(per_nd_names):
            p0[offset + k] = ind_fits[i][name]

    def _joint_model(tau_tiled, *params):
        shared_vals = {n: params[k] for k, n in enumerate(shared_names)}
        result = []
        for i in range(n_nds):
            offset = n_shared + i * n_per
            per_vals = {n: params[offset + k] for k, n in enumerate(per_nd_names)}
            all_vals = {**shared_vals, **per_vals}
            model = ramsey_model(
                tau, all_vals["amplitude"], all_vals["frequency"],
                all_vals["t2star"], all_vals["phase"], all_vals["offset"],
                stretch=1.0,
            )
            result.append(model)
        return np.concatenate(result)

    # Stack data for joint fit
    y_data = signals.ravel()
    tau_tiled = np.tile(tau, n_nds)

    # Bounds
    lower = []
    upper = []
    for name in shared_names:
        if name in ("amplitude", "frequency", "t2star"):
            lower.append(0)
            upper.append(np.inf)
        elif name == "phase":
            lower.append(-2 * np.pi)
            upper.append(2 * np.pi)
        else:
            lower.append(-np.inf)
            upper.append(np.inf)
    for _ in range(n_nds):
        for name in per_nd_names:
            if name in ("amplitude", "frequency", "t2star"):
                lower.append(0)
                upper.append(np.inf)
            elif name == "phase":
                lower.append(-2 * np.pi)
                upper.append(2 * np.pi)
            else:
                lower.append(-np.inf)
                upper.append(np.inf)

    popt, pcov = curve_fit(
        _joint_model, tau_tiled, y_data, p0=p0,
        bounds=(lower, upper), maxfev=50000,
    )

    # Unpack results
    shared_result = {n: float(popt[k]) for k, n in enumerate(shared_names)}
    per_nd_results = []
    for i in range(n_nds):
        offset = n_shared + i * n_per
        nd_result = {n: float(popt[offset + k]) for k, n in enumerate(per_nd_names)}
        nd_result.update(shared_result)
        nd_result["b_field"] = nd_result["frequency"] / GAMMA_NV
        per_nd_results.append(nd_result)

    b_field = shared_result.get("frequency", per_nd_results[0]["frequency"]) / GAMMA_NV

    return {
        "shared": shared_result,
        "per_nd": per_nd_results,
        "b_field": b_field,
        "cov": pcov,
        "n_nds": n_nds,
    }
