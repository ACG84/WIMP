"""Magnetic-field sensitivity estimates for pulsed NV protocols.

Computes the minimum detectable field (T/√Hz) for Ramsey, Hahn echo,
CPMG, XY-8, and T₁-based sensing modalities.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from wimp.constants import GAMMA_NV, HBAR, MU_B, G_NV

# ---------------------------------------------------------------------------
# Per-protocol sensitivity
# ---------------------------------------------------------------------------


def dc_sensitivity(
    contrast: float,
    t2star: float,
    readout_rate: float,
    n_nv: int = 1,
) -> float:
    r"""DC magnetic-field sensitivity (Ramsey protocol).

    .. math::
        \eta_{\rm DC} = \frac{\pi \hbar}{2\, g\, \mu_B\, C
                        \sqrt{R\, T_2^* \, N_{\rm NV}}}

    Parameters
    ----------
    contrast : float
        ODMR / readout contrast (0, 1].
    t2star : float
        Dephasing time T₂* (seconds).
    readout_rate : float
        Photon detection rate *R* (counts / second).
    n_nv : int
        Number of NV centres contributing.

    Returns
    -------
    eta : float
        Sensitivity in T / √Hz  (lower is better).
    """
    return (math.pi * HBAR) / (
        2.0 * G_NV * MU_B * contrast * math.sqrt(readout_rate * t2star * n_nv)
    )


def ac_sensitivity(
    contrast: float,
    t2: float,
    readout_rate: float,
    n_pulses: int = 1,
    n_nv: int = 1,
) -> float:
    r"""AC magnetic-field sensitivity (echo / DD protocols).

    For CPMG-*N* or XY-8 with *N* π-pulses the sensitivity improves
    as :math:`\propto 1/\sqrt{N}` relative to a single Hahn echo,
    provided *T*\ :sub:`2` allows the longer sequence.

    Parameters
    ----------
    contrast : float
        Readout contrast.
    t2 : float
        Coherence time T₂ (seconds).
    readout_rate : float
        Photon count rate (Hz).
    n_pulses : int
        Number of π refocusing pulses.
    n_nv : int
        Number of NV centres.

    Returns
    -------
    eta : float
        Sensitivity in T / √Hz.
    """
    eta_single = (math.pi * HBAR) / (
        2.0 * G_NV * MU_B * contrast * math.sqrt(readout_rate * t2 * n_nv)
    )
    # DD improvement: √N for N coherent inversions
    return eta_single / math.sqrt(max(n_pulses, 1))


def t1_sensitivity(
    contrast: float,
    t1: float,
    readout_rate: float,
    n_nv: int = 1,
) -> float:
    r"""Minimum detectable T₁ change (relaxometry contrast).

    Returns the smallest fractional change δT₁/T₁ detectable per
    √Hz of averaging.

    Parameters
    ----------
    contrast : float
        T₁ readout contrast.
    t1 : float
        Spin-lattice relaxation time (seconds).
    readout_rate : float
        Photon count rate (Hz).
    n_nv : int
        Number of NV centres.

    Returns
    -------
    delta_t1_frac : float
        Fractional δT₁/T₁ per √Hz.
    """
    snr_per_shot = contrast * math.sqrt(readout_rate * n_nv)
    # Optimum measurement time at τ ≈ T₁; sensitivity ∝ 1/(C √(R T₁ N))
    return 1.0 / (snr_per_shot * math.sqrt(t1))


def cw_sensitivity(
    contrast: float,
    linewidth: float,
    readout_rate: float,
    n_nv: int = 1,
) -> float:
    r"""CW ODMR magnetic-field sensitivity.

    .. math::
        \eta_{\rm CW} = \frac{4}{3\sqrt{3}}
                         \frac{\Delta f}{C \, \gamma_{\rm NV}
                         \sqrt{R \, N_{\rm NV}}}

    Parameters
    ----------
    contrast : float
        ODMR dip contrast (0, 1].
    linewidth : float
        ODMR linewidth (Hz, full width at half maximum).
    readout_rate : float
        Photon detection rate *R* (counts / second).
    n_nv : int
        Number of NV centres contributing.

    Returns
    -------
    eta : float
        Sensitivity in T / √Hz (lower is better).
    """
    return (4.0 / (3.0 * math.sqrt(3.0))) * linewidth / (
        contrast * GAMMA_NV * math.sqrt(readout_rate * n_nv)
    )


def shot_noise_limit(
    n_photons: float,
    contrast: float,
) -> float:
    """Minimum detectable phase shift (radians) per shot.

    Parameters
    ----------
    n_photons : float
        Detected photons per measurement shot.
    contrast : float
        Readout contrast.

    Returns
    -------
    delta_phi : float
        Phase noise floor (radians).
    """
    return 1.0 / (contrast * math.sqrt(max(n_photons, 1)))


# ---------------------------------------------------------------------------
# Protocol comparison
# ---------------------------------------------------------------------------


def compare_protocols(
    contrast: float = 0.03,
    t2star: float = 1e-6,
    t2: float = 100e-6,
    t1: float = 5e-3,
    readout_rate: float = 1e5,
    linewidth: float = 5e6,
    n_nv: int = 1,
    dd_pulses: int = 64,
) -> pd.DataFrame:
    """Compare sensitivities across all supported protocols.

    Returns a DataFrame with one row per protocol.
    """
    rows = [
        {
            "protocol": "Ramsey (DC)",
            "sensitivity_T_sqrtHz": dc_sensitivity(contrast, t2star, readout_rate, n_nv),
            "coherence_time_s": t2star,
            "notes": f"T2*={t2star*1e6:.1f} us",
        },
        {
            "protocol": "Hahn echo (AC)",
            "sensitivity_T_sqrtHz": ac_sensitivity(contrast, t2, readout_rate, 1, n_nv),
            "coherence_time_s": t2,
            "notes": f"T2={t2*1e6:.1f} us",
        },
        {
            "protocol": f"CPMG-{dd_pulses} (AC)",
            "sensitivity_T_sqrtHz": ac_sensitivity(contrast, t2, readout_rate, dd_pulses, n_nv),
            "coherence_time_s": t2,
            "notes": f"N={dd_pulses}, T2={t2*1e6:.1f} us",
        },
        {
            "protocol": "T1 relaxometry",
            "sensitivity_T_sqrtHz": float("nan"),  # not directly a B-field sensitivity
            "coherence_time_s": t1,
            "notes": f"dT1/T1={t1_sensitivity(contrast, t1, readout_rate, n_nv):.2e} /sqrtHz",
        },
        {
            "protocol": "CW ODMR",
            "sensitivity_T_sqrtHz": cw_sensitivity(contrast, linewidth, readout_rate, n_nv),
            "coherence_time_s": float("nan"),
            "notes": f"linewidth={linewidth*1e-6:.1f} MHz",
        },
    ]
    return pd.DataFrame(rows)


def optimal_protocol(
    target: str,
    *,
    characterization: dict[str, Any] | None = None,
    contrast: float = 0.03,
    t2star: float = 1e-6,
    t2: float = 100e-6,
    t1: float = 5e-3,
    readout_rate: float = 1e5,
    n_nv: int = 1,
) -> dict:
    """Recommend the best protocol for a given target field type.

    Parameters
    ----------
    target : str
        ``"dc"``, ``"ac"``, or ``"relaxometry"``.
    characterization : dict, optional
        If provided (e.g. from :func:`wimp.calibration.characterize_nv`),
        extract ``t2star``, ``t2``, ``t1`` from it.

    Returns
    -------
    recommendation : dict
        Keys: ``protocol``, ``sensitivity``, ``optimal_tau``.
    """
    if characterization is not None:
        t2star = characterization.get("t2star", t2star)
        t2 = characterization.get("t2", t2)
        t1 = characterization.get("t1", t1)
        contrast = characterization.get("contrast", contrast)
        readout_rate = characterization.get("readout_rate", readout_rate)
        n_nv = characterization.get("n_nv", n_nv)

    if target.lower() == "dc":
        eta = dc_sensitivity(contrast, t2star, readout_rate, n_nv)
        return {"protocol": "ramsey", "sensitivity_T_sqrtHz": eta,
                "optimal_tau_s": t2star}

    if target.lower() == "ac":
        # Compare echo vs DD
        eta_echo = ac_sensitivity(contrast, t2, readout_rate, 1, n_nv)
        eta_dd = ac_sensitivity(contrast, t2, readout_rate, 64, n_nv)
        if eta_dd < eta_echo:
            return {"protocol": "cpmg-64", "sensitivity_T_sqrtHz": eta_dd,
                    "optimal_tau_s": t2 / 2.0}
        return {"protocol": "hahn_echo", "sensitivity_T_sqrtHz": eta_echo,
                "optimal_tau_s": t2 / 2.0}

    if target.lower() in ("relaxometry", "t1"):
        delta = t1_sensitivity(contrast, t1, readout_rate, n_nv)
        return {"protocol": "t1_relaxometry",
                "sensitivity_dT1_frac_sqrtHz": delta,
                "optimal_tau_s": t1}

    if target.lower() == "cw_odmr":
        eta = cw_sensitivity(contrast, 5e6, readout_rate, n_nv)
        return {"protocol": "cw_odmr", "sensitivity_T_sqrtHz": eta,
                "optimal_tau_s": float("nan")}

    raise ValueError(f"Unknown target: {target!r}")


# ---------------------------------------------------------------------------
# Adaptive protocol recommendation
# ---------------------------------------------------------------------------


def adaptive_protocol_recommendation(
    characterization: dict[str, Any],
    *,
    target: str = "dc",
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recommend the best protocol from measured NV parameters.

    Takes the output of :func:`wimp.calibration.characterize_nv` and
    recommends an optimal protocol, tau, and expected sensitivity.

    Parameters
    ----------
    characterization : dict
        Must contain ``t2star``, ``t2``, ``t1`` (seconds).
        Optionally ``contrast`` and ``readout_rate``.
    target : str
        ``"dc"``, ``"ac"``, or ``"relaxometry"``.
    constraints : dict, optional
        Optional limits: ``max_sequence_duration`` (s),
        ``max_n_pulses``.

    Returns
    -------
    result : dict
        ``recommended_protocol``, ``recommended_tau``,
        ``predicted_sensitivity``, ``reasoning``, ``parameters``.
    """
    t2star = characterization["t2star"]
    t2 = characterization["t2"]
    t1 = characterization["t1"]
    contrast = characterization.get("contrast", 0.03)
    readout_rate = characterization.get("readout_rate", 1e5)
    n_nv = characterization.get("n_nv", 1)

    cons = constraints or {}
    max_dur = cons.get("max_sequence_duration", float("inf"))
    max_pulses = cons.get("max_n_pulses", 256)

    if target.lower() == "dc":
        optimal_tau = t2star / math.sqrt(2)
        optimal_tau = min(optimal_tau, max_dur)
        eta = dc_sensitivity(contrast, t2star, readout_rate, n_nv)
        return {
            "recommended_protocol": "ramsey",
            "recommended_tau": optimal_tau,
            "predicted_sensitivity": eta,
            "reasoning": (
                f"Ramsey is optimal for DC fields. "
                f"Optimal tau = T2*/sqrt(2) = {optimal_tau*1e6:.2f} us. "
                f"Predicted sensitivity: {eta:.2e} T/sqrt(Hz)."
            ),
            "parameters": {"t2star": t2star, "contrast": contrast},
        }

    if target.lower() == "ac":
        best_eta = ac_sensitivity(contrast, t2, readout_rate, 1, n_nv)
        best_proto = "hahn_echo"
        best_n = 1
        for n in [1, 4, 8, 16, 32, 64, 128, 256]:
            if n > max_pulses:
                break
            if t2 > max_dur:
                break
            eta = ac_sensitivity(contrast, t2, readout_rate, n, n_nv)
            if eta < best_eta:
                best_eta = eta
                best_n = n
                best_proto = f"cpmg-{n}" if n > 1 else "hahn_echo"

        optimal_tau = t2 / (2.0 * max(best_n, 1))
        return {
            "recommended_protocol": best_proto,
            "recommended_tau": optimal_tau,
            "predicted_sensitivity": best_eta,
            "reasoning": (
                f"AC sensing with {best_proto} ({best_n} pulses). "
                f"Sensitivity improves as 1/sqrt(N). "
                f"Predicted: {best_eta:.2e} T/sqrt(Hz)."
            ),
            "parameters": {"t2": t2, "n_pulses": best_n, "contrast": contrast},
        }

    if target.lower() in ("relaxometry", "t1"):
        delta = t1_sensitivity(contrast, t1, readout_rate, n_nv)
        return {
            "recommended_protocol": "t1_relaxometry",
            "recommended_tau": t1,
            "predicted_sensitivity": delta,
            "reasoning": (
                f"T1 relaxometry at tau = T1 = {t1*1e3:.2f} ms. "
                f"Fractional sensitivity: {delta:.2e} /sqrt(Hz)."
            ),
            "parameters": {"t1": t1, "contrast": contrast},
        }

    raise ValueError(f"Unknown target: {target!r}")


# ---------------------------------------------------------------------------
# Sensitivity sweeps
# ---------------------------------------------------------------------------


def sensitivity_vs_tau(
    protocol: str,
    tau_array: NDArray,
    *,
    contrast: float = 0.03,
    readout_rate: float = 1e5,
    n_nv: int = 1,
    n_pulses: int = 1,
    t2star: float | None = None,
    t2: float | None = None,
    t1: float | None = None,
) -> dict[str, NDArray]:
    r"""Sensitivity as a function of free-evolution time.

    Combines the shot-noise-limited sensitivity with the coherence
    decay envelope to show the optimal operating point.

    Parameters
    ----------
    protocol : str
        ``"ramsey"``, ``"echo"``, ``"t1"``.
    tau_array : ndarray
        Free-evolution times (seconds).
    contrast, readout_rate, n_nv, n_pulses
        NV parameters.
    t2star, t2, t1 : float, optional
        Coherence / relaxation times (required for the chosen protocol).

    Returns
    -------
    result : dict
        ``tau`` (seconds), ``sensitivity`` (T/√Hz).
    """
    tau = np.asarray(tau_array, dtype=float).ravel()

    if protocol == "ramsey":
        if t2star is None:
            raise ValueError("t2star required for Ramsey sensitivity")
        base = (math.pi * HBAR) / (
            2.0 * G_NV * MU_B * contrast
            * np.sqrt(readout_rate * np.maximum(tau, 1e-15) * n_nv)
        )
        decay_penalty = np.exp(tau / t2star)
        sens = base * decay_penalty

    elif protocol in ("echo", "hahn_echo"):
        if t2 is None:
            raise ValueError("t2 required for echo sensitivity")
        base = (math.pi * HBAR) / (
            2.0 * G_NV * MU_B * contrast
            * np.sqrt(readout_rate * np.maximum(tau, 1e-15) * n_nv)
        )
        decay_penalty = np.exp(tau / t2)
        dd_factor = 1.0 / math.sqrt(max(n_pulses, 1))
        sens = base * decay_penalty * dd_factor

    elif protocol == "t1":
        if t1 is None:
            raise ValueError("t1 required for T1 sensitivity")
        snr_per_shot = contrast * math.sqrt(readout_rate * n_nv)
        sens = 1.0 / (snr_per_shot * np.sqrt(np.maximum(tau, 1e-15)))
        decay_penalty = np.exp(tau / t1)
        sens = sens * decay_penalty

    elif protocol == "cw_odmr":
        raise ValueError(
            "CW ODMR sensitivity does not depend on tau. "
            "Use cw_sensitivity() instead."
        )

    else:
        raise ValueError(f"Unknown protocol: {protocol!r}")

    return {"tau": tau, "sensitivity": sens}


def measurement_time_for_target(
    target_sensitivity: float,
    protocol: str,
    *,
    contrast: float = 0.03,
    readout_rate: float = 1e5,
    n_nv: int = 1,
    n_pulses: int = 1,
    t2star: float = 1e-6,
    t2: float = 100e-6,
    t1: float = 5e-3,
) -> float:
    r"""Estimate measurement time to reach a target sensitivity.

    Sensitivity improves as :math:`1/\sqrt{T}` with averaging time *T*.

    Parameters
    ----------
    target_sensitivity : float
        Desired sensitivity (T/√Hz).
    protocol : str
        ``"ramsey"``, ``"echo"``, ``"t1"``.
    contrast, readout_rate, n_nv, n_pulses
        NV parameters.
    t2star, t2, t1 : float
        Coherence / relaxation times.

    Returns
    -------
    t_meas : float
        Required measurement time (seconds).
    """
    if protocol == "ramsey":
        eta_0 = dc_sensitivity(contrast, t2star, readout_rate, n_nv)
    elif protocol in ("echo", "hahn_echo"):
        eta_0 = ac_sensitivity(contrast, t2, readout_rate, n_pulses, n_nv)
    elif protocol == "t1":
        eta_0 = t1_sensitivity(contrast, t1, readout_rate, n_nv)
    elif protocol == "cw_odmr":
        eta_0 = cw_sensitivity(contrast, 5e6, readout_rate, n_nv)
    else:
        raise ValueError(f"Unknown protocol: {protocol!r}")

    return (eta_0 / target_sensitivity) ** 2
