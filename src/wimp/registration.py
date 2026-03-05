"""C. elegans body-pose estimation and atlas registration.

Maps nanodiamond positions onto the canonical worm neuroanatomical
atlas by fitting a centreline spline and transforming atlas coordinates
into the image frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import UnivariateSpline, splprep, splev
from scipy.spatial import cKDTree

from wimp.constants import (
    CANONICAL_BODY_LENGTH,
    LONGITUDINAL_VARIABILITY,
    TRANSVERSE_VARIABILITY,
)

# ---------------------------------------------------------------------------
# Neuron types along the ventral nerve cord
# ---------------------------------------------------------------------------

NEURON_CLASSES: dict[str, str] = {
    "DB": "excitatory_dorsal",
    "VB": "excitatory_ventral",
    "DD": "inhibitory_dorsal",
    "VD": "inhibitory_ventral",
    "DA": "excitatory_dorsal",
    "VA": "excitatory_ventral",
    "AS": "excitatory",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CenterlineResult:
    """Result of centreline fitting."""

    control_points: NDArray
    """(n_ctrl, 3) spline control points (metres)."""

    arc_length: NDArray
    """(n_pts,) cumulative arc length along the fitted curve (metres)."""

    curve: NDArray
    """(n_pts, 3) densely sampled points on the fitted curve."""

    body_length: float
    """Total arc length of the centreline (metres)."""

    tck: Any = None
    """Spline representation for ``splev``."""


@dataclass
class RegistrationResult:
    """Result of atlas → image registration."""

    neuron_names: list[str]
    neuron_positions_image: NDArray
    """(n_neurons, 3) registered positions in image coordinates."""

    covariance: NDArray
    """(n_neurons, 3, 3) positional uncertainty per neuron."""

    centerline: CenterlineResult | None = None


@dataclass
class DeformableRegistrationResult:
    """Result of per-frame deformable atlas registration."""

    neuron_names: list[str]
    neuron_positions: NDArray
    """(n_frames, n_neurons, 3) registered neuron positions per frame."""

    covariance: NDArray
    """(n_frames, n_neurons, 3, 3) positional uncertainty per neuron per frame."""

    centerlines: list[CenterlineResult]
    """Per-frame centerline fits."""

    frame_indices: NDArray
    """(n_frames,) integer indices of the processed frames."""


# ---------------------------------------------------------------------------
# Atlas loading
# ---------------------------------------------------------------------------


def load_atlas(
    path: str | None = None,
) -> dict[str, Any]:
    """Load a C. elegans neuron atlas.

    If *path* is ``None``, returns a minimal built-in atlas of ventral
    nerve cord motor neurons with canonical fractional positions along
    the body axis.

    Returns
    -------
    atlas : dict
        Keys: ``names`` (list[str]), ``fractional_positions`` (n, 3),
        ``types`` (list[str]).
    """
    if path is not None:
        import pandas as pd
        df = pd.read_csv(path)
        return {
            "names": df["name"].tolist(),
            "fractional_positions": df[["x_frac", "y_frac", "z_frac"]].values,
            "types": df.get("type", ["unknown"] * len(df)).tolist(),
        }

    # Built-in minimal VNC motor neuron atlas
    # Fractional positions along body axis (x ∈ [0, 1])
    names = []
    positions = []
    types = []

    motor_types = ["DB", "VB", "DD", "VD"]
    # Approximate positions of DB1-7, VB1-11, DD1-6, VD1-13
    for cls in motor_types:
        if cls == "DB":
            count = 7
        elif cls == "VB":
            count = 11
        elif cls == "DD":
            count = 6
        elif cls == "VD":
            count = 13
        else:
            count = 5

        fracs = np.linspace(0.2, 0.85, count)
        for k, f in enumerate(fracs, 1):
            names.append(f"{cls}{k}")
            y_off = 2e-6 if cls.startswith("D") else -2e-6  # dorsal/ventral
            positions.append([f, y_off / CANONICAL_BODY_LENGTH, 0.0])
            types.append(NEURON_CLASSES.get(cls, "unknown"))

    return {
        "names": names,
        "fractional_positions": np.array(positions),
        "types": types,
    }


# ---------------------------------------------------------------------------
# Centreline fitting
# ---------------------------------------------------------------------------


def fit_centerline(
    point_cloud: NDArray,
    n_control_points: int = 20,
    smoothing: float | None = None,
) -> CenterlineResult:
    """Fit a smooth spline centreline through a 3-D point cloud.

    Parameters
    ----------
    point_cloud : (n_points, 3) array
        Unordered 3-D coordinates (metres) of landmarks / NDs along
        the worm body.
    n_control_points : int
        Desired number of equally-spaced output points.
    smoothing : float, optional
        ``splprep`` smoothing parameter.  ``None`` → automatic.

    Returns
    -------
    result : CenterlineResult
    """
    pts = np.asarray(point_cloud, dtype=float)

    # Sort roughly by x (anterior → posterior)
    order = np.argsort(pts[:, 0])
    pts = pts[order]

    # Fit parametric 3-D spline
    s = smoothing if smoothing is not None else len(pts)
    tck, u = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=s, k=min(3, len(pts) - 1))

    # Evaluate at many points to compute arc length
    n_dense = max(1000, 10 * n_control_points)
    u_dense = np.linspace(0, 1, n_dense)
    x, y, z = splev(u_dense, tck)
    curve = np.column_stack([x, y, z])

    # Cumulative arc length
    diffs = np.diff(curve, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0], np.cumsum(seg_lengths)])
    body_length = float(arc[-1])

    # Re-sample at equally-spaced arc-length positions
    u_ctrl = np.linspace(0, 1, n_control_points)
    xc, yc, zc = splev(u_ctrl, tck)
    ctrl = np.column_stack([xc, yc, zc])

    return CenterlineResult(
        control_points=ctrl,
        arc_length=arc,
        curve=curve,
        body_length=body_length,
        tck=tck,
    )


def arc_length_parameterize(
    centerline: CenterlineResult,
    n_points: int = 500,
) -> tuple[NDArray, NDArray]:
    """Re-sample the centreline at equal arc-length intervals.

    Returns
    -------
    s : (n_points,) — arc-length coordinates (metres).
    pts : (n_points, 3) — 3-D positions.
    """
    L = centerline.body_length
    s_target = np.linspace(0, L, n_points)

    # Interpolate from existing dense arc-length → parameter u
    arc = centerline.arc_length
    u_dense = np.linspace(0, 1, len(arc))
    u_interp = np.interp(s_target, arc, u_dense)

    x, y, z = splev(u_interp, centerline.tck)
    pts = np.column_stack([x, y, z])
    return s_target, pts


# ---------------------------------------------------------------------------
# Atlas registration
# ---------------------------------------------------------------------------


def register_atlas_to_image(
    atlas: dict[str, Any],
    centerline: CenterlineResult,
    body_length: float | None = None,
) -> RegistrationResult:
    """Map canonical atlas positions onto a fitted centreline.

    Parameters
    ----------
    atlas : dict
        From :func:`load_atlas`.
    centerline : CenterlineResult
    body_length : float, optional
        Override for the measured body length (metres).

    Returns
    -------
    result : RegistrationResult
    """
    L = body_length if body_length is not None else centerline.body_length
    frac_pos = atlas["fractional_positions"]
    names = atlas["names"]
    n_neurons = len(names)

    # For each neuron, sample the centreline at its fractional position
    positions = np.zeros((n_neurons, 3))
    covariances = np.zeros((n_neurons, 3, 3))

    for k in range(n_neurons):
        s_frac = frac_pos[k, 0]  # fractional along body
        u = np.clip(s_frac, 0, 1)
        x, y, z = splev(u, centerline.tck)
        base = np.array([float(x), float(y), float(z)])

        # Add lateral offset from atlas
        base[1] += frac_pos[k, 1] * CANONICAL_BODY_LENGTH
        base[2] += frac_pos[k, 2] * CANONICAL_BODY_LENGTH

        positions[k] = base

        # Uncertainty model
        sigma_long = LONGITUDINAL_VARIABILITY * L
        sigma_trans = TRANSVERSE_VARIABILITY
        covariances[k] = np.diag([sigma_long ** 2, sigma_trans ** 2, sigma_trans ** 2])

    return RegistrationResult(
        neuron_names=names,
        neuron_positions_image=positions,
        covariance=covariances,
        centerline=centerline,
    )


# ---------------------------------------------------------------------------
# Body-pose estimation
# ---------------------------------------------------------------------------


def estimate_body_pose(
    nd_positions: NDArray,
    *,
    n_control_points: int = 10,
    smoothing: float | None = None,
) -> CenterlineResult:
    """Estimate the worm centreline from nanodiamond positions alone.

    This is a convenience wrapper around :func:`fit_centerline` that
    uses ND positions as the point cloud.
    """
    return fit_centerline(
        nd_positions,
        n_control_points=n_control_points,
        smoothing=smoothing,
    )


def uncertainty_ellipses(
    registration: RegistrationResult,
) -> NDArray:
    """Return 3×3 covariance matrices for each neuron.

    Convenience accessor; identical to ``registration.covariance``.
    """
    return registration.covariance


# ---------------------------------------------------------------------------
# Nearest-ND mapping
# ---------------------------------------------------------------------------


def assign_nds_to_neurons(
    nd_positions: NDArray,
    neuron_positions: NDArray,
    max_distance: float | None = None,
) -> list[list[int]]:
    """Assign each neuron its nearest nanodiamond(s).

    Parameters
    ----------
    nd_positions : (n_nds, 3) array
    neuron_positions : (n_neurons, 3) array
    max_distance : float, optional
        Maximum assignment distance (metres).

    Returns
    -------
    assignments : list of lists
        ``assignments[j]`` contains the ND indices closest to neuron *j*.
    """
    tree = cKDTree(nd_positions)
    assignments: list[list[int]] = []
    for j in range(neuron_positions.shape[0]):
        dist, idx = tree.query(neuron_positions[j], k=1)
        if max_distance is not None and dist > max_distance:
            assignments.append([])
        else:
            assignments.append([int(idx)])
    return assignments


# ---------------------------------------------------------------------------
# Deformable registration
# ---------------------------------------------------------------------------


def tracks_to_frame_positions(
    tracks: Any,
    *,
    pixel_scale: float = 1.0,
    z_offset: float = 0.0,
) -> tuple[NDArray, NDArray]:
    """Convert a trackpy-style DataFrame to per-frame 3-D positions.

    Parameters
    ----------
    tracks : DataFrame
        Must have columns ``frame``, ``particle``, ``x``, and ``y``.
    pixel_scale : float
        Conversion from pixel to metres for x/y.
    z_offset : float
        Fixed z-coordinate (metres) for all particles.

    Returns
    -------
    frame_positions : (n_frames, n_particles, 3) array
        NaN where a particle is missing in a given frame.
    frame_indices : (n_frames,) int array
        Sorted unique frame numbers.
    """
    import pandas as pd

    df = tracks if isinstance(tracks, pd.DataFrame) else pd.DataFrame(tracks)

    frame_indices = np.sort(df["frame"].unique())
    particles = np.sort(df["particle"].unique())
    n_frames = len(frame_indices)
    n_particles = len(particles)

    frame_map = {f: i for i, f in enumerate(frame_indices)}
    part_map = {p: j for j, p in enumerate(particles)}

    positions = np.full((n_frames, n_particles, 3), np.nan)

    for _, row in df.iterrows():
        fi = frame_map[row["frame"]]
        pi = part_map[row["particle"]]
        positions[fi, pi, 0] = row["x"] * pixel_scale
        positions[fi, pi, 1] = row["y"] * pixel_scale
        positions[fi, pi, 2] = z_offset

    return positions, np.asarray(frame_indices, dtype=int)


def deformable_register(
    frame_positions: NDArray,
    atlas: dict[str, Any] | None = None,
    *,
    min_nds_per_frame: int = 4,
    smoothing: float | None = None,
) -> DeformableRegistrationResult:
    """Per-frame atlas registration using tracked ND positions.

    Parameters
    ----------
    frame_positions : (n_frames, n_particles, 3) array
        ND positions per frame (NaN for missing).
    atlas : dict, optional
        From :func:`load_atlas`.  ``None`` → built-in atlas.
    min_nds_per_frame : int
        Minimum valid NDs to attempt spline fitting; frames with fewer
        are interpolated from neighbours.
    smoothing : float, optional
        ``splprep`` smoothing factor.

    Returns
    -------
    result : DeformableRegistrationResult
    """
    if atlas is None:
        atlas = load_atlas()

    n_frames = frame_positions.shape[0]
    n_neurons = len(atlas["names"])

    all_positions = np.full((n_frames, n_neurons, 3), np.nan)
    all_cov = np.zeros((n_frames, n_neurons, 3, 3))
    centerlines: list[CenterlineResult | None] = [None] * n_frames
    valid_frames: list[int] = []

    for t in range(n_frames):
        pts = frame_positions[t]
        valid_mask = ~np.any(np.isnan(pts), axis=1)
        n_valid = int(np.sum(valid_mask))

        if n_valid < min_nds_per_frame:
            continue

        valid_pts = pts[valid_mask]
        cl = fit_centerline(valid_pts, smoothing=smoothing)
        reg = register_atlas_to_image(atlas, cl)

        all_positions[t] = reg.neuron_positions_image
        all_cov[t] = reg.covariance
        centerlines[t] = cl
        valid_frames.append(t)

    # Interpolate missing frames from nearest valid neighbours
    if valid_frames:
        valid_set = set(valid_frames)
        for t in range(n_frames):
            if t in valid_set:
                continue
            # Find nearest valid frame
            dists = [abs(t - v) for v in valid_frames]
            nearest = valid_frames[int(np.argmin(dists))]
            all_positions[t] = all_positions[nearest]
            all_cov[t] = all_cov[nearest]
            centerlines[t] = centerlines[nearest]

    # Filter out None centerlines (shouldn't happen after interpolation)
    cl_list = [c for c in centerlines if c is not None]

    return DeformableRegistrationResult(
        neuron_names=atlas["names"],
        neuron_positions=all_positions,
        covariance=all_cov,
        centerlines=cl_list,
        frame_indices=np.arange(n_frames, dtype=int),
    )


def smooth_neuron_trajectories(
    neuron_positions: NDArray,
    window: int = 5,
) -> NDArray:
    """Temporal smoothing of neuron position trajectories.

    Parameters
    ----------
    neuron_positions : (n_frames, n_neurons, 3) array
    window : int
        Moving-average window size along the frame axis.

    Returns
    -------
    smoothed : (n_frames, n_neurons, 3) array
    """
    from scipy.ndimage import uniform_filter1d

    if window <= 1:
        return neuron_positions.copy()

    return uniform_filter1d(neuron_positions, size=window, axis=0, mode="nearest")
