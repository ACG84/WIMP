"""NV-centre characterisation and system calibration.

Provides routines for measuring T₂*, T₂, T₁ from reference data,
cross-validating with a Hall probe, and correcting for temperature
drift via the zero-field splitting *D₀*.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import linregress

from wimp.constants import D0, D0_TEMP_COEFF, GAMMA_NV
from wimp.relaxation import fit_ramsey, fit_t2_decay, fit_t1_recovery, fit_odmr

# ---------------------------------------------------------------------------
# Full NV characterisation
# ---------------------------------------------------------------------------


def characterize_nv(
    ramsey_tau: NDArray,
    ramsey_signal: NDArray,
    echo_tau: NDArray,
    echo_signal: NDArray,
    t1_tau: NDArray,
    t1_signal: NDArray,
    *,
    contrast: float = 0.03,
    readout_rate: float = 1e5,
) -> dict[str, Any]:
    """Run a full NV characterisation from three calibration sweeps.

    Parameters
    ----------
    ramsey_tau, ramsey_signal : ndarray
        Ramsey fringe data.
    echo_tau, echo_signal : ndarray
        Hahn-echo decay data.
    t1_tau, t1_signal : ndarray
        T₁ recovery data.
    contrast : float
        Readout contrast (for sensitivity estimate).
    readout_rate : float
        Photon detection rate (Hz).

    Returns
    -------
    params : dict
        ``t2star``, ``t2``, ``t1`` (seconds),
        ``ramsey_fit``, ``echo_fit``, ``t1_fit`` (full fit dicts),
        ``dc_sensitivity``, ``ac_sensitivity`` (T/√Hz).
    """
    from wimp.sensitivity import dc_sensitivity, ac_sensitivity

    ram_fit = fit_ramsey(ramsey_tau, ramsey_signal)
    echo_fit = fit_t2_decay(echo_tau, echo_signal)
    t1_fit = fit_t1_recovery(t1_tau, t1_signal)

    t2star = ram_fit["t2star"]
    t2 = echo_fit["t2"]
    t1 = t1_fit["t1"]

    eta_dc = dc_sensitivity(contrast, t2star, readout_rate)
    eta_ac = ac_sensitivity(contrast, t2, readout_rate)

    return {
        "t2star": t2star,
        "t2": t2,
        "t1": t1,
        "ramsey_fit": ram_fit,
        "echo_fit": echo_fit,
        "t1_fit": t1_fit,
        "dc_sensitivity": eta_dc,
        "ac_sensitivity": eta_ac,
    }


def characterize_nv_cw(
    freq: NDArray,
    signal: NDArray,
    *,
    readout_rate: float = 1e5,
) -> dict[str, Any]:
    """Characterise an NV centre from a CW ODMR spectrum.

    Fits the ODMR spectrum, extracts linewidth, contrast, estimates
    T₂*, derives CW sensitivity, and estimates temperature shift.

    Parameters
    ----------
    freq : ndarray
        Frequency sweep (Hz).
    signal : ndarray
        ODMR signal.
    readout_rate : float
        Photon detection rate (Hz).

    Returns
    -------
    result : dict
        ``fit``, ``linewidth``, ``contrast``, ``t2star_estimated``,
        ``b_field``, ``cw_sensitivity``, ``center_freq``,
        ``temperature_shift``.
    """
    import math
    from wimp.sensitivity import cw_sensitivity
    from wimp.relaxation import extract_field_odmr

    fit = fit_odmr(freq, signal)

    # Extract linewidth and contrast from fit
    n_dips = fit.get("n_dips", 1)
    if n_dips == 2:
        linewidth = (fit["gamma_minus"] + fit["gamma_plus"]) / 2.0
        contrast = (fit["a_minus"] + fit["a_plus"]) / (2.0 * fit["baseline"])
        center_freq = (fit["f_minus"] + fit["f_plus"]) / 2.0
    else:
        linewidth = fit["linewidth"]
        contrast = fit["amplitude"] / fit["baseline"]
        center_freq = fit["center"]

    # Estimate T2* from linewidth: T2* ≈ 1 / (π Δf)
    t2star_est = 1.0 / (math.pi * linewidth) if linewidth > 0 else float("inf")

    # B-field
    b_field = extract_field_odmr(fit)

    # CW sensitivity
    eta = cw_sensitivity(contrast, linewidth, readout_rate)

    # Temperature shift from center frequency vs D0
    temp_shift = temperature_correction(center_freq)

    return {
        "fit": fit,
        "linewidth": linewidth,
        "contrast": contrast,
        "t2star_estimated": t2star_est,
        "b_field": b_field,
        "cw_sensitivity": eta,
        "center_freq": center_freq,
        "temperature_shift": temp_shift,
    }


# ---------------------------------------------------------------------------
# Known-field calibration
# ---------------------------------------------------------------------------


def calibrate_with_known_field(
    measured_field: NDArray,
    applied_field: NDArray,
) -> dict[str, float]:
    """Linear calibration from a known applied field.

    Parameters
    ----------
    measured_field : ndarray
        NV-measured B values (Tesla).
    applied_field : ndarray
        True applied B values (Tesla).

    Returns
    -------
    cal : dict
        ``scale``, ``offset``, ``r_squared``.
    """
    measured = np.asarray(measured_field, dtype=float).ravel()
    applied = np.asarray(applied_field, dtype=float).ravel()

    res = linregress(applied, measured)
    return {
        "scale": float(res.slope),
        "offset": float(res.intercept),
        "r_squared": float(res.rvalue ** 2),
    }


# ---------------------------------------------------------------------------
# Temperature correction
# ---------------------------------------------------------------------------


def temperature_correction(
    d0_measured: float,
    d0_ref: float = D0,
) -> float:
    """Estimate temperature shift from a measured zero-field splitting.

    .. math::
        \\Delta T = (D_{0,\\text{meas}} - D_{0,\\text{ref}})
                    / (dD_0/dT)

    Parameters
    ----------
    d0_measured : float
        Measured D₀ (Hz).
    d0_ref : float
        Reference D₀ (Hz) at known temperature.

    Returns
    -------
    delta_T : float
        Temperature change (Kelvin).  Positive means warmer.
    """
    return (d0_measured - d0_ref) / D0_TEMP_COEFF


def correct_field_for_temperature(
    field: NDArray,
    d0_measured: float,
    d0_ref: float = D0,
) -> NDArray:
    """Apply a first-order temperature correction to measured fields.

    The D₀ shift introduces a systematic frequency offset that looks
    like a spurious field; this function subtracts it.

    Returns
    -------
    corrected : ndarray
        Temperature-corrected magnetic field (Tesla).
    """
    delta_d0 = d0_measured - d0_ref
    spurious_b = delta_d0 / GAMMA_NV
    return np.asarray(field, dtype=float) - spurious_b


# ---------------------------------------------------------------------------
# Hall-probe cross-validation
# ---------------------------------------------------------------------------


def cross_validate_hall(
    nv_field: NDArray,
    hall_field: NDArray,
) -> dict[str, Any]:
    """Cross-validate NV measurements against a Hall-probe reference.

    Parameters
    ----------
    nv_field : ndarray
        NV-measured field values (Tesla).
    hall_field : ndarray
        Hall-probe reference values (Tesla).

    Returns
    -------
    result : dict
        ``slope``, ``offset``, ``r_squared``, ``residuals``,
        ``rms_residual``.
    """
    nv = np.asarray(nv_field, dtype=float).ravel()
    hall = np.asarray(hall_field, dtype=float).ravel()

    res = linregress(hall, nv)
    predicted = res.slope * hall + res.intercept
    residuals = nv - predicted

    return {
        "slope": float(res.slope),
        "offset": float(res.intercept),
        "r_squared": float(res.rvalue ** 2),
        "residuals": residuals,
        "rms_residual": float(np.sqrt(np.mean(residuals ** 2))),
    }


# ---------------------------------------------------------------------------
# Noise-floor estimate
# ---------------------------------------------------------------------------


def noise_floor_estimate(
    background: NDArray,
    fs: float,
) -> dict[str, Any]:
    """Estimate the noise floor from background (zero-field) measurements.

    Parameters
    ----------
    background : ndarray
        Field-equivalent background time series (Tesla).
    fs : float
        Sampling rate (Hz).

    Returns
    -------
    result : dict
        ``noise_rms`` (T), ``noise_psd`` (T²/Hz), ``freq`` (Hz).
    """
    from scipy.signal import welch

    bg = np.asarray(background, dtype=float).ravel()
    freq, psd = welch(bg, fs=fs, nperseg=min(256, len(bg)))

    return {
        "noise_rms": float(np.std(bg)),
        "noise_psd": psd,
        "freq": freq,
    }


# ---------------------------------------------------------------------------
# Linearity check
# ---------------------------------------------------------------------------


def linearity_check(
    field_steps: NDArray,
    measured_responses: NDArray,
) -> dict[str, Any]:
    """Check sensor linearity across a range of applied fields.

    Parameters
    ----------
    field_steps : ndarray
        Applied field values (Tesla).
    measured_responses : ndarray
        NV-measured responses (same unit / proportional to field).

    Returns
    -------
    result : dict
        ``slope``, ``r_squared``, ``max_deviation``, ``linear_range``.
    """
    fs = np.asarray(field_steps, dtype=float).ravel()
    mr = np.asarray(measured_responses, dtype=float).ravel()

    res = linregress(fs, mr)
    predicted = res.slope * fs + res.intercept
    deviation = np.abs(mr - predicted)

    return {
        "slope": float(res.slope),
        "r_squared": float(res.rvalue ** 2),
        "max_deviation": float(np.max(deviation)),
        "linear_range": (float(fs.min()), float(fs.max())),
    }


# ---------------------------------------------------------------------------
# Time-varying temperature correction
# ---------------------------------------------------------------------------


def temperature_series_correction(
    field_timeseries: NDArray,
    d0_timeseries: NDArray,
    *,
    d0_ref: float = D0,
) -> dict[str, Any]:
    """Apply time-varying temperature correction to a field series.

    When D₀ drifts with temperature, the measured Ramsey / ODMR
    frequency shifts, introducing a spurious apparent field.  This
    function subtracts that artefact at each time step.

    Parameters
    ----------
    field_timeseries : ndarray, shape ``(n_timepoints,)``
        Measured magnetic-field values (Tesla).
    d0_timeseries : ndarray, shape ``(n_timepoints,)``
        Zero-field splitting measured at each time step (Hz).
    d0_ref : float
        Reference D₀ (Hz) at known temperature.

    Returns
    -------
    result : dict
        ``corrected_field``, ``temperature_shifts`` (K),
        ``spurious_field`` (T).
    """
    field = np.asarray(field_timeseries, dtype=float).ravel()
    d0 = np.asarray(d0_timeseries, dtype=float).ravel()

    delta_d0 = d0 - d0_ref
    spurious_b = delta_d0 / GAMMA_NV
    corrected = field - spurious_b
    temp_shifts = delta_d0 / D0_TEMP_COEFF

    return {
        "corrected_field": corrected,
        "temperature_shifts": temp_shifts,
        "spurious_field": spurious_b,
    }


def estimate_temperature_from_ramsey(
    frequency: float,
    *,
    b_known: float = 0.0,
) -> float:
    """Infer temperature shift from a Ramsey frequency measurement.

    When the external field is known, the remaining frequency offset
    reflects a shift in D₀ caused by temperature change.

    Parameters
    ----------
    frequency : float
        Measured Ramsey oscillation frequency (Hz).
    b_known : float
        Known external magnetic field (Tesla).

    Returns
    -------
    delta_T : float
        Temperature change from reference (Kelvin).
    """
    # The Ramsey *oscillation* frequency is the detuning from the
    # microwave drive (set near D0), so it equals:
    #     f_osc ≈ |dD0/dT * dT ± gamma * B|
    # Rearrange: dD0 = f_osc - gamma * |B|, then dT = dD0 / (dD0/dT)
    delta_d0 = frequency - GAMMA_NV * abs(b_known)
    return delta_d0 / D0_TEMP_COEFF
