"""Pulse sequence definitions and filter-function computation.

Supports Ramsey, Hahn echo, CPMG, and XY-8 dynamical-decoupling
protocols used in time-domain NV-centre magnetometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from wimp.constants import GAMMA_NV

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PulseSequence:
    """Description of a pulsed NV measurement protocol.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"ramsey"``, ``"hahn_echo"``).
    tau : float | NDArray
        Free-evolution time(s) in seconds.
    n_pi_pulses : int
        Number of π refocusing pulses in a single repetition.
    pulse_phases : list[float]
        Phase (radians) of each π pulse.
    total_sensing_time : float | NDArray
        Net phase-accumulation time (seconds).  For a Hahn echo this
        equals ``2 * tau``; for CPMG-N it equals ``2 * N * tau``.
    n_blocks : int
        Repeat count for composite sequences (XY-8 blocks, etc.).
    """

    name: str
    tau: float | NDArray
    n_pi_pulses: int = 0
    pulse_phases: list[float] = field(default_factory=list)
    total_sensing_time: float | NDArray = 0.0
    n_blocks: int = 1


# ---------------------------------------------------------------------------
# Sequence factories
# ---------------------------------------------------------------------------


def ramsey_sequence(tau: float | NDArray) -> PulseSequence:
    """Create a Ramsey (free-induction-decay) sequence.

    π/2 — τ — π/2 — readout
    """
    tau = np.asarray(tau, dtype=float)
    return PulseSequence(
        name="ramsey",
        tau=tau,
        n_pi_pulses=0,
        pulse_phases=[],
        total_sensing_time=tau,
        n_blocks=1,
    )


def hahn_echo_sequence(tau: float | NDArray) -> PulseSequence:
    """Create a Hahn-echo sequence.

    π/2 — τ — π — τ — π/2 — readout
    """
    tau = np.asarray(tau, dtype=float)
    return PulseSequence(
        name="hahn_echo",
        tau=tau,
        n_pi_pulses=1,
        pulse_phases=[0.0],
        total_sensing_time=2.0 * tau,
        n_blocks=1,
    )


def cpmg_sequence(tau: float | NDArray, n_pulses: int) -> PulseSequence:
    """Create a CPMG dynamical-decoupling sequence.

    π/2 — [τ — π — τ]×N — π/2 — readout

    Parameters
    ----------
    tau : float or array
        Half-spacing between π pulses (seconds).
    n_pulses : int
        Number of π pulses (*N*).
    """
    if n_pulses < 1:
        raise ValueError("n_pulses must be >= 1")
    tau = np.asarray(tau, dtype=float)
    return PulseSequence(
        name="cpmg",
        tau=tau,
        n_pi_pulses=n_pulses,
        pulse_phases=[0.0] * n_pulses,
        total_sensing_time=2.0 * n_pulses * tau,
        n_blocks=1,
    )


def xy8_sequence(tau: float | NDArray, n_blocks: int = 1) -> PulseSequence:
    """Create an XY-8 dynamical-decoupling sequence.

    One XY-8 block consists of 8 π pulses with phases
    X-Y-X-Y-Y-X-Y-X.  ``n_blocks`` repeats of that block are
    concatenated.

    Parameters
    ----------
    tau : float or array
        Half-spacing between π pulses (seconds).
    n_blocks : int
        Number of XY-8 blocks.
    """
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")
    tau = np.asarray(tau, dtype=float)
    single_block = [0.0, math.pi / 2, 0.0, math.pi / 2,
                    math.pi / 2, 0.0, math.pi / 2, 0.0]
    n_pi = 8 * n_blocks
    return PulseSequence(
        name="xy8",
        tau=tau,
        n_pi_pulses=n_pi,
        pulse_phases=single_block * n_blocks,
        total_sensing_time=2.0 * n_pi * tau,
        n_blocks=n_blocks,
    )


# ---------------------------------------------------------------------------
# Filter (modulation) functions
# ---------------------------------------------------------------------------


def filter_function(
    sequence: PulseSequence,
    freq: NDArray,
) -> NDArray:
    """Compute the frequency-domain filter function *W(f)* of a pulse
    sequence.

    The filter function describes the spectral sensitivity of the
    sequence: a slowly-varying (DC) field is picked up by Ramsey, a
    field oscillating at *f* = 1/(2τ) is picked up by echo/DD, etc.

    Parameters
    ----------
    sequence : PulseSequence
        Pulse sequence descriptor (uses scalar ``tau``).
    freq : ndarray
        Frequency array (Hz).  Values ≤ 0 are silently clipped to a
        tiny positive number so that divisions are safe.

    Returns
    -------
    W : ndarray, same shape as *freq*
        Dimensionless filter function (normalised so that peak ≈ 1).
    """
    freq = np.asarray(freq, dtype=float)
    # Use scalar tau for filter function computation.
    tau = float(np.asarray(sequence.tau).flat[0]) if np.ndim(sequence.tau) else float(sequence.tau)

    eps = 1e-30  # avoid division by zero
    omega = 2.0 * np.pi * np.maximum(freq, eps)

    if sequence.name == "ramsey":
        # W(f) = sinc^2(pi f tau)
        x = np.pi * freq * tau
        W = np.sinc(freq * tau) ** 2  # np.sinc(x) = sin(pi*x)/(pi*x)

    elif sequence.name == "hahn_echo":
        # W(f) = 8 sin^4(pi f tau) / (omega tau)^2   (normalised)
        s = np.sin(np.pi * freq * tau)
        W = s ** 4  # simplified & normalised to peak=1

    elif sequence.name == "cpmg":
        N = sequence.n_pi_pulses
        s = np.sin(np.pi * freq * tau)
        # |sin(N pi f tau) / sin(pi f tau)|^2  *  sin^4(pi f tau)
        denom = np.where(np.abs(s) > 1e-15, s, 1e-15)
        ratio = np.sin(N * np.pi * freq * tau) / denom
        W = (ratio * s ** 2) ** 2
        # Normalise
        peak = np.max(W)
        if peak > 0:
            W = W / peak

    elif sequence.name == "xy8":
        # XY-8 has same magnitude filter as CPMG-8N with phase cycling.
        N = sequence.n_pi_pulses  # 8 * n_blocks
        s = np.sin(np.pi * freq * tau)
        denom = np.where(np.abs(s) > 1e-15, s, 1e-15)
        ratio = np.sin(N * np.pi * freq * tau) / denom
        W = (ratio * s ** 2) ** 2
        peak = np.max(W)
        if peak > 0:
            W = W / peak

    else:
        raise ValueError(f"Unknown sequence name: {sequence.name!r}")

    return W


def filter_peak_frequency(sequence: PulseSequence) -> float:
    """Return the centre frequency (Hz) of the filter passband.

    For echo/DD sequences this is 1/(2τ); for Ramsey it is 0 (DC).
    """
    tau = float(np.asarray(sequence.tau).flat[0]) if np.ndim(sequence.tau) else float(sequence.tau)
    if sequence.name == "ramsey":
        return 0.0
    return 1.0 / (2.0 * tau)


# ---------------------------------------------------------------------------
# Phase accumulation
# ---------------------------------------------------------------------------


def accumulated_phase(
    sequence: PulseSequence,
    field_func: Callable[[NDArray], NDArray],
    tau: float | None = None,
    dt: float = 1e-9,
) -> float:
    """Numerically compute the accumulated NV-spin phase for a
    time-varying field *B(t)*.

    Parameters
    ----------
    sequence : PulseSequence
        Pulse protocol definition.
    field_func : callable
        ``field_func(t)`` returns *B* in Tesla at times *t* (array).
    tau : float, optional
        Override the free-evolution time (seconds).  Defaults to the
        scalar value stored in *sequence.tau*.
    dt : float
        Integration time step (seconds).

    Returns
    -------
    phi : float
        Accumulated phase (radians).
    """
    if tau is None:
        tau = float(np.asarray(sequence.tau).flat[0]) if np.ndim(sequence.tau) else float(sequence.tau)

    T = float(np.asarray(sequence.total_sensing_time).flat[0]) if np.ndim(sequence.total_sensing_time) else float(sequence.total_sensing_time)
    if tau != 0:
        # re-derive T for the given tau
        if sequence.name == "ramsey":
            T = tau
        elif sequence.name == "hahn_echo":
            T = 2.0 * tau
        else:
            T = 2.0 * sequence.n_pi_pulses * tau

    t = np.arange(0, T, dt)
    B = field_func(t)

    # Build modulation function: +1/−1 switching at each π pulse
    mod = np.ones_like(t)
    if sequence.n_pi_pulses > 0:
        for k in range(1, sequence.n_pi_pulses + 1):
            # π pulses at times (2k−1)*tau
            t_flip = (2 * k - 1) * tau
            mod[t >= t_flip] *= -1

    phi = 2.0 * np.pi * GAMMA_NV * np.trapezoid(mod * B, t)
    return float(phi)


# ---------------------------------------------------------------------------
# Optimal τ helper
# ---------------------------------------------------------------------------


def optimal_tau(
    protocol: str,
    *,
    t2star: float | None = None,
    t2: float | None = None,
    target_freq: float | None = None,
) -> float:
    """Suggest an optimal free-evolution time.

    Parameters
    ----------
    protocol : str
        ``"ramsey"``, ``"hahn_echo"``, ``"cpmg"``, or ``"xy8"``.
    t2star, t2 : float, optional
        Coherence times (seconds).  Required for protocols that use
        them.
    target_freq : float, optional
        Target AC frequency (Hz) for echo/DD protocols.

    Returns
    -------
    tau : float
        Recommended τ in seconds.
    """
    if protocol == "ramsey":
        if t2star is None:
            raise ValueError("t2star required for Ramsey optimal tau")
        # Maximum sensitivity at τ ≈ T2*
        return t2star

    if protocol in ("hahn_echo", "cpmg", "xy8"):
        if target_freq is not None and target_freq > 0:
            # Centre the filter on the target frequency
            return 1.0 / (2.0 * target_freq)
        if t2 is not None:
            # Maximise signal × decay ⇒ τ ≈ T2 / 2 (for echo)
            return t2 / 2.0
        raise ValueError("Provide target_freq or t2 for echo/DD optimal tau")

    raise ValueError(f"Unknown protocol: {protocol!r}")
