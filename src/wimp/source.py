"""Neural source localisation from NV magnetic-field measurements.

Implements lead-field (forward model) computation and minimum-norm
estimation (MNE) for recovering neural current distributions from
nanodiamond magnetometry data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from wimp.constants import MU0

# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------


def lead_field_matrix(
    neuron_positions: NDArray,
    nd_positions: NDArray,
    *,
    orientation: NDArray | None = None,
) -> NDArray:
    """Compute the lead-field matrix mapping dipole sources to B_z.

    Parameters
    ----------
    neuron_positions : (n_neurons, 3) array
        Source (neuron) positions in metres.
    nd_positions : (n_nds, 3) array
        Sensor (nanodiamond) positions in metres.
    orientation : (n_neurons, 3) array, optional
        Unit dipole direction per source.  Defaults to z-hat.

    Returns
    -------
    L : (n_nds, n_neurons) array
        Lead-field matrix (T per A·m² of dipole moment).
    """
    n_neurons = neuron_positions.shape[0]
    n_nds = nd_positions.shape[0]

    if orientation is None:
        orientation = np.zeros((n_neurons, 3))
        orientation[:, 2] = 1.0

    L = np.zeros((n_nds, n_neurons))
    prefactor = MU0 / (4.0 * np.pi)

    for j in range(n_neurons):
        m = orientation[j]
        for i in range(n_nds):
            r_vec = nd_positions[i] - neuron_positions[j]
            r = np.linalg.norm(r_vec)
            if r < 1e-15:
                continue
            r_hat = r_vec / r
            m_dot_r = np.dot(m, r_hat)
            # z-component of dipole field
            L[i, j] = prefactor * (3.0 * m_dot_r * r_hat[2] - m[2]) / r ** 3

    return L


def forward_model(
    lead_field: NDArray,
    neural_currents: NDArray,
    *,
    moment_scale: float = 1.0,
) -> NDArray:
    """Predict magnetic field at sensors from neural currents.

    Parameters
    ----------
    lead_field : (n_nds, n_neurons) array
    neural_currents : (n_neurons,) or (n_neurons, n_timepoints) array
        Current amplitudes (arbitrary units scaled by *moment_scale*).
    moment_scale : float
        Scaling from current units to A·m².

    Returns
    -------
    B : (n_nds,) or (n_nds, n_timepoints) array
        Predicted Bz (Tesla).
    """
    return moment_scale * lead_field @ neural_currents


# ---------------------------------------------------------------------------
# Inverse solvers
# ---------------------------------------------------------------------------


def mne_inverse_kernel(
    lead_field: NDArray,
    *,
    lambda_reg: float = 0.1,
) -> NDArray:
    r"""Compute the MNE inverse kernel.

    .. math::
        K = L^T (L L^T + \lambda I)^{-1}

    Parameters
    ----------
    lead_field : (n_nds, n_neurons) array
    lambda_reg : float
        Tikhonov regularisation parameter.

    Returns
    -------
    K : (n_neurons, n_nds) array
        Inverse kernel so that ``J = K @ B``.
    """
    L = lead_field  # (M, N)
    gram = L @ L.T  # (M, M)
    reg = lambda_reg * np.trace(gram) / gram.shape[0] * np.eye(gram.shape[0])
    inv_gram = np.linalg.solve(gram + reg, np.eye(gram.shape[0]))
    return L.T @ inv_gram  # (N, M)


def mne_inverse(
    lead_field: NDArray,
    b_timeseries: NDArray,
    *,
    lambda_reg: float = 0.1,
    moment_scale: float = 1.0,
) -> NDArray:
    """Minimum-norm estimation of neural currents.

    .. math::
        \\hat{J} = L^T (L L^T + \\lambda I)^{-1} B

    Parameters
    ----------
    lead_field : (n_nds, n_neurons) array
    b_timeseries : (n_nds,) or (n_nds, n_timepoints) array
        Measured Bz (Tesla).
    lambda_reg : float
        Tikhonov regularisation parameter.
    moment_scale : float
        If the lead field was computed with unit moments, set this to
        match the forward-model scaling.

    Returns
    -------
    J : (n_neurons,) or (n_neurons, n_timepoints) array
        Estimated source currents.
    """
    K = mne_inverse_kernel(lead_field, lambda_reg=lambda_reg)
    return K @ (b_timeseries / moment_scale)


def lcmv_beamformer(
    lead_field: NDArray,
    b_timeseries: NDArray,
    target_idx: int,
) -> NDArray:
    """Linearly constrained minimum-variance beamformer.

    Focuses on a single source index and returns its estimated time
    course while suppressing other sources.

    Parameters
    ----------
    lead_field : (n_nds, n_neurons) array
    b_timeseries : (n_nds, n_timepoints) array
    target_idx : int
        Index of the source neuron to focus on.

    Returns
    -------
    s : (n_timepoints,) array
        Estimated source time course.
    """
    B = b_timeseries  # (M, T)
    l = lead_field[:, target_idx]  # (M,)

    # Data covariance
    C = np.cov(B)  # (M, M)
    if C.ndim == 0:
        C = C.reshape(1, 1)

    # Regularise
    C += 1e-10 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])

    C_inv_l = np.linalg.solve(C, l)
    w = C_inv_l / np.dot(l, C_inv_l)

    return w @ B


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def source_snr(
    true_sources: NDArray,
    estimated_sources: NDArray,
) -> dict[str, float]:
    """Evaluate reconstruction quality.

    Returns
    -------
    metrics : dict
        ``correlation`` : Pearson-r across flattened arrays.
        ``relative_error`` : ‖est − true‖ / ‖true‖.
        ``peak_localisation_error`` : index distance between peaks.
    """
    true_flat = true_sources.ravel()
    est_flat = estimated_sources.ravel()

    # Correlation
    if np.std(true_flat) == 0 or np.std(est_flat) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(true_flat, est_flat)[0, 1])

    # Relative error
    norm_true = np.linalg.norm(true_flat)
    rel_err = float(np.linalg.norm(est_flat - true_flat) / norm_true) if norm_true > 0 else float("inf")

    # Peak localisation (per-source, if 2-D)
    if true_sources.ndim == 2:
        true_peak = int(np.argmax(np.max(np.abs(true_sources), axis=1)))
        est_peak = int(np.argmax(np.max(np.abs(estimated_sources), axis=1)))
        peak_err = abs(true_peak - est_peak)
    else:
        peak_err = abs(int(np.argmax(np.abs(true_flat))) - int(np.argmax(np.abs(est_flat))))

    return {
        "correlation": corr,
        "relative_error": rel_err,
        "peak_localisation_error": peak_err,
    }


# ---------------------------------------------------------------------------
# Resolution and crosstalk analysis
# ---------------------------------------------------------------------------


def resolution_matrix(
    lead_field: NDArray,
    *,
    lambda_reg: float = 0.1,
) -> NDArray:
    r"""Compute the MNE resolution matrix.

    .. math::
        R = K L = L^T (L L^T + \lambda I)^{-1} L

    ``R[i, j]`` quantifies how much true source *j* leaks into the
    estimate at source *i*.  A perfect inversion gives ``R = I``.

    Parameters
    ----------
    lead_field : (n_nds, n_neurons) array
    lambda_reg : float

    Returns
    -------
    R : (n_neurons, n_neurons) array
    """
    K = mne_inverse_kernel(lead_field, lambda_reg=lambda_reg)
    return K @ lead_field


def crosstalk_metrics(
    R: NDArray,
    neuron_names: list[str] | None = None,
) -> dict[str, Any]:
    """Summarise a resolution matrix into per-source crosstalk statistics.

    Parameters
    ----------
    R : (n_neurons, n_neurons) array
        Resolution matrix from :func:`resolution_matrix`.
    neuron_names : list of str, optional

    Returns
    -------
    metrics : dict
        ``diagonal`` — self-resolution per source.
        ``off_diagonal_max`` — worst-case leakage per source.
        ``crosstalk_ratio`` — off_diagonal_max / diagonal per source.
        ``spread`` — point-spread width (# sources with |R| > 0.1 * diag).
        ``total_leakage`` — sum of |off-diagonal| per row.
        ``neuron_names`` — echoed back for labelling.
    """
    n = R.shape[0]
    diag = np.abs(np.diag(R))
    off_diag_max = np.zeros(n)
    spread = np.zeros(n, dtype=int)
    total_leakage = np.zeros(n)

    for i in range(n):
        row = np.abs(R[i].copy())
        row[i] = 0.0
        off_diag_max[i] = float(np.max(row)) if n > 1 else 0.0
        total_leakage[i] = float(np.sum(row))
        threshold = 0.1 * diag[i] if diag[i] > 0 else 0.0
        spread[i] = int(np.sum(np.abs(R[i]) > threshold))

    crosstalk_ratio = np.where(diag > 0, off_diag_max / diag, 0.0)

    names = neuron_names if neuron_names is not None else [f"S{i}" for i in range(n)]

    return {
        "diagonal": diag,
        "off_diagonal_max": off_diag_max,
        "crosstalk_ratio": crosstalk_ratio,
        "spread": spread.tolist(),
        "total_leakage": total_leakage,
        "neuron_names": names,
    }


# ---------------------------------------------------------------------------
# Time-varying inverse
# ---------------------------------------------------------------------------


def time_varying_inverse(
    neuron_pos_per_frame: NDArray,
    nd_pos_per_frame: NDArray,
    b_timeseries: NDArray,
    *,
    lambda_reg: float = 0.1,
    compute_resolution: bool = False,
    orientation: NDArray | None = None,
) -> dict[str, Any]:
    """MNE inverse with per-frame geometry for deformable bodies.

    Parameters
    ----------
    neuron_pos_per_frame : (n_frames, n_neurons, 3) array
    nd_pos_per_frame : (n_nds, 3) or (n_frames, n_nds, 3) array
        If 2-D, the same ND positions are used for every frame.
    b_timeseries : (n_nds, n_frames) array
        Measured field at each ND for each frame.
    lambda_reg : float
    compute_resolution : bool
        If True, also return per-frame resolution matrices.
    orientation : (n_neurons, 3) array, optional

    Returns
    -------
    result : dict
        ``source_estimate`` : (n_neurons, n_frames) array.
        ``lead_fields`` : list of (n_nds, n_neurons) arrays.
        ``resolution_matrices`` : list of (n_neurons, n_neurons) arrays
        (only when *compute_resolution* is True).
    """
    n_frames = neuron_pos_per_frame.shape[0]
    n_neurons = neuron_pos_per_frame.shape[1]
    n_nds = b_timeseries.shape[0]

    # Broadcast static ND positions
    nd_static = nd_pos_per_frame.ndim == 2
    if nd_static:
        nd_all = np.broadcast_to(
            nd_pos_per_frame[np.newaxis, :, :],
            (n_frames, n_nds, 3),
        )
    else:
        nd_all = nd_pos_per_frame

    source_est = np.zeros((n_neurons, n_frames))
    lead_fields: list[NDArray] = []
    res_mats: list[NDArray] = []

    for t in range(n_frames):
        L_t = lead_field_matrix(
            neuron_pos_per_frame[t],
            nd_all[t],
            orientation=orientation,
        )
        lead_fields.append(L_t)

        K_t = mne_inverse_kernel(L_t, lambda_reg=lambda_reg)
        source_est[:, t] = K_t @ b_timeseries[:, t]

        if compute_resolution:
            res_mats.append(K_t @ L_t)

    result: dict[str, Any] = {
        "source_estimate": source_est,
        "lead_fields": lead_fields,
    }
    if compute_resolution:
        result["resolution_matrices"] = res_mats

    return result
