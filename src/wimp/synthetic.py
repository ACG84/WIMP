"""Synthetic data generation for time-domain NV magnetometry.

Creates realistic simulated measurement data for testing and
validation of the WIMP processing pipeline.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from wimp.constants import GAMMA_NV, MU0, CANONICAL_BODY_LENGTH
from wimp.pulses import PulseSequence, ramsey_sequence, hahn_echo_sequence, cpmg_sequence
from wimp.relaxation import ramsey_model, t2_decay_model, t1_recovery_model

# ---------------------------------------------------------------------------
# Single-curve generators
# ---------------------------------------------------------------------------


def generate_ramsey_data(
    tau_array: NDArray,
    b_field: float,
    t2star: float,
    snr: float = 50.0,
    *,
    amplitude: float = 1.0,
    phase: float = 0.0,
    offset: float = 0.0,
    stretch: float = 1.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a synthetic Ramsey fringe.

    Parameters
    ----------
    tau_array : ndarray
        Free-evolution times (seconds).
    b_field : float
        Applied DC magnetic field (Tesla).
    t2star : float
        Dephasing time (seconds).
    snr : float
        Signal-to-noise ratio (amplitude / σ_noise).

    Returns
    -------
    result : dict
        ``tau``, ``signal``, ``clean_signal``, and ``ground_truth``
        sub-dict.
    """
    rng = np.random.default_rng(seed)
    tau = np.asarray(tau_array, dtype=float)
    freq = b_field * GAMMA_NV

    clean = ramsey_model(tau, amplitude, freq, t2star, phase, offset, stretch)
    noise_std = amplitude / snr if snr > 0 else 0.0
    noisy = clean + rng.normal(0, noise_std, size=clean.shape)

    return {
        "tau": tau,
        "signal": noisy,
        "clean_signal": clean,
        "ground_truth": {
            "b_field": b_field,
            "frequency": freq,
            "t2star": t2star,
            "amplitude": amplitude,
            "phase": phase,
            "offset": offset,
            "stretch": stretch,
        },
    }


def generate_echo_data(
    tau_array: NDArray,
    t2: float,
    snr: float = 50.0,
    *,
    amplitude: float = 1.0,
    offset: float = 0.0,
    stretch: float = 1.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a synthetic Hahn-echo T₂ decay curve."""
    rng = np.random.default_rng(seed)
    tau = np.asarray(tau_array, dtype=float)

    clean = t2_decay_model(tau, amplitude, t2, offset, stretch)
    noise_std = amplitude / snr if snr > 0 else 0.0
    noisy = clean + rng.normal(0, noise_std, size=clean.shape)

    return {
        "tau": tau,
        "signal": noisy,
        "clean_signal": clean,
        "ground_truth": {
            "t2": t2,
            "amplitude": amplitude,
            "offset": offset,
            "stretch": stretch,
        },
    }


def generate_t1_data(
    tau_array: NDArray,
    t1: float,
    snr: float = 50.0,
    *,
    amplitude: float = 1.0,
    contrast: float = 1.0,
    offset: float = 0.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a synthetic T₁ recovery curve."""
    rng = np.random.default_rng(seed)
    tau = np.asarray(tau_array, dtype=float)

    clean = t1_recovery_model(tau, amplitude, t1, contrast, offset)
    noise_std = amplitude / snr if snr > 0 else 0.0
    noisy = clean + rng.normal(0, noise_std, size=clean.shape)

    return {
        "tau": tau,
        "signal": noisy,
        "clean_signal": clean,
        "ground_truth": {
            "t1": t1,
            "amplitude": amplitude,
            "contrast": contrast,
            "offset": offset,
        },
    }


def generate_dd_sweep(
    tau_array: NDArray,
    n_pulses: int,
    noise_spectrum_func: Callable[[NDArray], NDArray],
    snr: float = 50.0,
    *,
    amplitude: float = 1.0,
    t2: float = 200e-6,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a dynamical-decoupling sweep for noise spectroscopy.

    The decay at each τ is determined by the overlap of the DD filter
    function with the environmental noise spectrum *S(f)*.

    Parameters
    ----------
    tau_array : ndarray
        Sweep of half-spacings (seconds).
    n_pulses : int
        Number of π pulses (CPMG-N).
    noise_spectrum_func : callable
        ``S(f)`` in T²/Hz.  Called with an array of frequencies.
    snr : float
        Signal-to-noise ratio.

    Returns
    -------
    result : dict
        Includes ``tau``, ``signal``, ``coherence``, and the
        corresponding ``freq_axis`` (1/(2τ)).
    """
    rng = np.random.default_rng(seed)
    tau = np.asarray(tau_array, dtype=float)

    coherence = np.zeros_like(tau)
    freq_axis = 1.0 / (2.0 * tau)

    for i, t in enumerate(tau):
        # Approximate: decay ∝ exp(-χ) where χ ∝ ∫ S(f) W(f) df
        # For CPMG-N peaked at f0 = 1/(2τ), χ ≈ 8 N τ S(f0)
        f0 = 1.0 / (2.0 * t) if t > 0 else 1e30
        S_f0 = float(noise_spectrum_func(np.array([f0]))[0])
        chi = 8.0 * n_pulses * t * (2.0 * np.pi * GAMMA_NV) ** 2 * S_f0
        coherence[i] = np.exp(-chi)

    clean = amplitude * coherence
    noise_std = amplitude / snr if snr > 0 else 0.0
    noisy = clean + rng.normal(0, noise_std, size=clean.shape)

    return {
        "tau": tau,
        "signal": noisy,
        "clean_signal": clean,
        "coherence": coherence,
        "freq_axis": freq_axis,
        "ground_truth": {
            "n_pulses": n_pulses,
            "amplitude": amplitude,
            "t2": t2,
        },
    }


# ---------------------------------------------------------------------------
# Neural magnetic field forward model
# ---------------------------------------------------------------------------


def _dipole_field_z(
    source_pos: NDArray,
    moment: NDArray,
    field_pos: NDArray,
) -> float:
    """Z-component of the magnetic dipole field at *field_pos*.

    Parameters
    ----------
    source_pos : (3,) array — dipole position (m).
    moment : (3,) array — magnetic dipole moment (A m²).
    field_pos : (3,) array — observation point (m).

    Returns
    -------
    Bz : float — z-component of B (T).
    """
    r_vec = field_pos - source_pos
    r = np.linalg.norm(r_vec)
    if r < 1e-15:
        return 0.0
    r_hat = r_vec / r
    m_dot_r = np.dot(moment, r_hat)
    Bz = (MU0 / (4.0 * np.pi)) * (3.0 * m_dot_r * r_hat[2] - moment[2]) / r ** 3
    return float(Bz)


def generate_neural_field(
    neuron_positions: NDArray,
    nd_positions: NDArray,
    current_waveforms: NDArray,
    dt: float,
    *,
    dipole_orientation: NDArray | None = None,
    moment_scale: float = 1e-18,
) -> dict[str, Any]:
    """Compute B(t) at each nanodiamond from neural current sources.

    Parameters
    ----------
    neuron_positions : (n_neurons, 3) array
        Neuron positions (metres).
    nd_positions : (n_nds, 3) array
        Nanodiamond positions (metres).
    current_waveforms : (n_neurons, n_timepoints) array
        Normalised neural current time courses.
    dt : float
        Sample interval (seconds).
    dipole_orientation : (n_neurons, 3) array, optional
        Unit dipole direction per neuron.  Defaults to z-hat.
    moment_scale : float
        Scale factor to convert normalised current to magnetic dipole
        moment (A·m²).

    Returns
    -------
    result : dict
        ``field`` : (n_nds, n_timepoints) in Tesla.
        ``time`` : (n_timepoints,) in seconds.
        ``lead_field`` : (n_nds, n_neurons) static coupling matrix.
    """
    n_neurons = neuron_positions.shape[0]
    n_nds = nd_positions.shape[0]
    n_t = current_waveforms.shape[1]

    if dipole_orientation is None:
        dipole_orientation = np.zeros((n_neurons, 3))
        dipole_orientation[:, 2] = 1.0  # z-oriented

    # Build static lead-field (Bz at each ND per unit moment)
    lead = np.zeros((n_nds, n_neurons))
    for j in range(n_neurons):
        m_hat = dipole_orientation[j]
        for i in range(n_nds):
            lead[i, j] = _dipole_field_z(
                neuron_positions[j], m_hat, nd_positions[i]
            )

    # Scale currents → moments → fields
    field = moment_scale * lead @ current_waveforms  # (n_nds, n_t)
    time = np.arange(n_t) * dt

    return {
        "field": field,
        "time": time,
        "lead_field": lead,
    }


# ---------------------------------------------------------------------------
# Full synthetic experiment
# ---------------------------------------------------------------------------


def generate_full_experiment(
    n_nds: int = 10,
    n_neurons: int = 5,
    n_timepoints: int = 500,
    dt: float = 1e-3,
    protocol: str = "ramsey",
    t2star: float = 1e-6,
    t2: float = 100e-6,
    t1: float = 5e-3,
    snr: float = 30.0,
    seed: int | None = None,
    *,
    body_length: float = CANONICAL_BODY_LENGTH,
) -> dict[str, Any]:
    """Generate a complete synthetic WIMP experiment.

    Creates nanodiamond and neuron positions along a worm body,
    simulates neural activity, computes magnetic fields, and generates
    per-ND time-domain measurement data.

    Returns
    -------
    experiment : dict
        Contains ``dataset`` (:class:`~wimp.io.WIMPDataset`-compatible
        fields), ``neuron_positions``, ``nd_positions``,
        ``current_waveforms``, ``field``, and per-ND ``fit_data``.
    """
    rng = np.random.default_rng(seed)

    # Distribute NDs and neurons along the worm body (x-axis)
    nd_x = np.linspace(0.1 * body_length, 0.9 * body_length, n_nds)
    nd_positions = np.column_stack([
        nd_x,
        rng.normal(0, 3e-6, n_nds),  # slight y scatter
        np.full(n_nds, 5e-6),  # NDs sit ~5 µm above nerve cord
    ])

    neuron_x = np.linspace(0.15 * body_length, 0.85 * body_length, n_neurons)
    neuron_positions = np.column_stack([
        neuron_x,
        np.zeros(n_neurons),
        np.zeros(n_neurons),
    ])

    # Simple synthetic neural activity: sparse spikes convolved with
    # an exponential kernel
    current_waveforms = np.zeros((n_neurons, n_timepoints))
    for j in range(n_neurons):
        n_spikes = rng.poisson(5)
        spike_times = rng.integers(0, n_timepoints, size=n_spikes)
        current_waveforms[j, spike_times] = rng.uniform(0.5, 1.5, n_spikes)
        # Convolve with an exponential decay (τ = 20 ms)
        kernel_len = min(int(0.02 / dt), n_timepoints)
        kernel = np.exp(-np.arange(kernel_len) * dt / 0.02)
        current_waveforms[j] = np.convolve(
            current_waveforms[j], kernel, mode="same"
        )

    # Forward model → B(t) at each ND
    fwd = generate_neural_field(
        neuron_positions, nd_positions, current_waveforms, dt
    )
    field_ts = fwd["field"]  # (n_nds, n_timepoints)

    # Generate per-ND measurement data for the chosen protocol
    n_tau = 100
    fit_data_list = []

    if protocol == "ramsey":
        tau_arr = np.linspace(0, 5 * t2star, n_tau)
        for i in range(n_nds):
            b_mean = float(np.mean(np.abs(field_ts[i])))
            fd = generate_ramsey_data(
                tau_arr, b_mean, t2star, snr, seed=None if seed is None else seed + i
            )
            fit_data_list.append(fd)

    elif protocol in ("echo", "hahn_echo"):
        tau_arr = np.linspace(0, 3 * t2, n_tau)
        for i in range(n_nds):
            fd = generate_echo_data(
                tau_arr, t2, snr, seed=None if seed is None else seed + i
            )
            fit_data_list.append(fd)

    elif protocol == "t1":
        tau_arr = np.linspace(0, 5 * t1, n_tau)
        for i in range(n_nds):
            fd = generate_t1_data(
                tau_arr, t1, snr, seed=None if seed is None else seed + i
            )
            fit_data_list.append(fd)

    elif protocol == "dd":
        tau_arr = np.logspace(-7, -4, n_tau)  # 100 ns → 100 µs
        noise_func = lambda f: 1e-20 * np.ones_like(f)  # flat noise floor
        for i in range(n_nds):
            fd = generate_dd_sweep(
                tau_arr, 16, noise_func, snr,
                seed=None if seed is None else seed + i,
            )
            fit_data_list.append(fd)

    else:
        raise ValueError(f"Unknown protocol: {protocol!r}")

    # Build signal array: (n_nds, n_tau)
    signal_arr = np.stack([fd["signal"] for fd in fit_data_list])

    return {
        "protocol": protocol,
        "tau_array": tau_arr,
        "signal": signal_arr,
        "neuron_positions": neuron_positions,
        "nd_positions": nd_positions,
        "current_waveforms": current_waveforms,
        "field_timeseries": field_ts,
        "time": fwd["time"],
        "lead_field": fwd["lead_field"],
        "fit_data": fit_data_list,
        "dt": dt,
        "metadata": {
            "n_nds": n_nds,
            "n_neurons": n_neurons,
            "n_timepoints": n_timepoints,
            "dt": dt,
            "protocol": protocol,
            "t2star": t2star,
            "t2": t2,
            "t1": t1,
            "snr": snr,
            "seed": seed,
        },
    }


# ---------------------------------------------------------------------------
# Deformable synthetic experiment
# ---------------------------------------------------------------------------


def generate_deformable_experiment(
    n_nds: int = 10,
    n_neurons: int = 5,
    n_frames: int = 20,
    dt: float = 0.1,
    amplitude: float = 5e-6,
    wavelength: float = 500e-6,
    freq: float = 1.0,
    snr: float = 30.0,
    seed: int | None = None,
    *,
    body_length: float = CANONICAL_BODY_LENGTH,
) -> dict[str, Any]:
    """Generate a synthetic experiment with a deformable worm body.

    Simulates locomotion as a travelling sinusoidal wave that
    deforms ND and neuron positions over time.

    Parameters
    ----------
    n_nds : int
        Number of nanodiamonds.
    n_neurons : int
        Number of neurons.
    n_frames : int
        Number of time frames.
    dt : float
        Frame interval (seconds).
    amplitude : float
        Peak lateral displacement of the body wave (metres).
    wavelength : float
        Spatial wavelength of the body wave (metres).
    freq : float
        Temporal frequency of the body wave (Hz).
    snr : float
        Signal-to-noise ratio for field measurements.
    seed : int, optional

    Returns
    -------
    experiment : dict
        ``frame_nd_positions`` : (n_frames, n_nds, 3)
        ``frame_neuron_positions`` : (n_frames, n_neurons, 3)
        ``current_waveforms`` : (n_neurons, n_frames)
        ``field_timeseries`` : (n_nds, n_frames)
        ``atlas`` : dict
        ``time`` : (n_frames,)
    """
    from wimp.source import lead_field_matrix as _lfm

    rng = np.random.default_rng(seed)
    time = np.arange(n_frames) * dt

    # Baseline positions along body axis
    nd_x = np.linspace(0.1 * body_length, 0.9 * body_length, n_nds)
    neuron_x = np.linspace(0.15 * body_length, 0.85 * body_length, n_neurons)

    nd_y0 = rng.normal(0, 2e-6, n_nds)
    nd_z0 = np.full(n_nds, 5e-6)

    neuron_y0 = np.zeros(n_neurons)
    neuron_z0 = np.zeros(n_neurons)

    # Per-frame deformation: travelling sine wave y(x, t)
    frame_nd_pos = np.zeros((n_frames, n_nds, 3))
    frame_neuron_pos = np.zeros((n_frames, n_neurons, 3))

    for t_idx, t in enumerate(time):
        phase = 2.0 * np.pi * (nd_x / wavelength - freq * t)
        dy_nd = amplitude * np.sin(phase)
        frame_nd_pos[t_idx, :, 0] = nd_x
        frame_nd_pos[t_idx, :, 1] = nd_y0 + dy_nd
        frame_nd_pos[t_idx, :, 2] = nd_z0

        phase_n = 2.0 * np.pi * (neuron_x / wavelength - freq * t)
        dy_neuron = amplitude * np.sin(phase_n)
        frame_neuron_pos[t_idx, :, 0] = neuron_x
        frame_neuron_pos[t_idx, :, 1] = neuron_y0 + dy_neuron
        frame_neuron_pos[t_idx, :, 2] = neuron_z0

    # Synthetic neural currents
    current_waveforms = np.zeros((n_neurons, n_frames))
    for j in range(n_neurons):
        n_spikes = rng.poisson(3)
        spike_frames = rng.integers(0, n_frames, size=max(n_spikes, 1))
        current_waveforms[j, spike_frames] = rng.uniform(0.5, 1.5, len(spike_frames))
        # Smooth with exponential kernel
        kernel_len = min(5, n_frames)
        kernel = np.exp(-np.arange(kernel_len) * dt / max(dt * 3, 0.01))
        current_waveforms[j] = np.convolve(current_waveforms[j], kernel, mode="same")

    # Per-frame forward model -> field timeseries
    field_ts = np.zeros((n_nds, n_frames))
    moment_scale = 1e-18

    for t_idx in range(n_frames):
        L_t = _lfm(frame_neuron_pos[t_idx], frame_nd_pos[t_idx])
        field_ts[:, t_idx] = moment_scale * L_t @ current_waveforms[:, t_idx]

    # Add noise
    if snr > 0:
        signal_rms = np.sqrt(np.mean(field_ts ** 2)) if np.any(field_ts) else 1e-12
        noise_std = signal_rms / snr
        field_ts += rng.normal(0, noise_std, field_ts.shape)

    # Build a minimal atlas for this experiment
    atlas = {
        "names": [f"N{i}" for i in range(n_neurons)],
        "fractional_positions": np.column_stack([
            neuron_x / body_length,
            np.zeros(n_neurons),
            np.zeros(n_neurons),
        ]),
        "types": ["excitatory"] * n_neurons,
    }

    return {
        "frame_nd_positions": frame_nd_pos,
        "frame_neuron_positions": frame_neuron_pos,
        "current_waveforms": current_waveforms,
        "field_timeseries": field_ts,
        "atlas": atlas,
        "time": time,
        "n_nds": n_nds,
        "n_neurons": n_neurons,
        "n_frames": n_frames,
        "body_length": body_length,
    }
