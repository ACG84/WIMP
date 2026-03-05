"""Nanodiamond localisation and tracking in fluorescence images.

Wraps `trackpy <https://soft-matter.github.io/trackpy/>`_ for
particle detection and frame-to-frame linking.  This module is
optional: it gracefully degrades if *trackpy* is not installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

_TRACKPY_AVAILABLE = False
try:
    import trackpy as tp
    _TRACKPY_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Return True if the trackpy backend is importable."""
    return _TRACKPY_AVAILABLE


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_nds(
    image: NDArray,
    diameter: int = 11,
    *,
    minmass: float | None = None,
    threshold: float | None = None,
) -> Any:
    """Locate nanodiamond candidates in a single fluorescence image.

    Parameters
    ----------
    image : 2-D ndarray
        Grey-scale fluorescence frame.
    diameter : int
        Expected particle diameter in pixels (must be odd).
    minmass : float, optional
        Minimum integrated brightness.
    threshold : float, optional
        Clip pixels below this value before detection.

    Returns
    -------
    features : pandas.DataFrame
        Columns include ``x``, ``y``, ``mass``, ``size``, ``ecc``.

    Raises
    ------
    RuntimeError
        If trackpy is not installed.
    """
    if not _TRACKPY_AVAILABLE:
        raise RuntimeError("trackpy is required for ND localisation")

    kwargs: dict[str, Any] = {"diameter": diameter}
    if minmass is not None:
        kwargs["minmass"] = minmass
    if threshold is not None:
        kwargs["threshold"] = threshold

    return tp.locate(image, **kwargs)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


def track_nds(
    image_stack: NDArray,
    diameter: int = 11,
    *,
    search_range: float = 5.0,
    memory: int = 3,
    minmass: float | None = None,
) -> Any:
    """Detect and link nanodiamonds across frames.

    Parameters
    ----------
    image_stack : (n_frames, height, width) ndarray
    diameter : int
        Particle diameter (pixels).
    search_range : float
        Maximum displacement between frames (pixels).
    memory : int
        Number of frames a particle can vanish and reappear.
    minmass : float, optional
        Minimum integrated brightness.

    Returns
    -------
    tracks : pandas.DataFrame
        Columns: ``x``, ``y``, ``frame``, ``particle``, ``mass``, …
    """
    if not _TRACKPY_AVAILABLE:
        raise RuntimeError("trackpy is required for ND tracking")

    import pandas as pd

    frames = []
    for i, img in enumerate(image_stack):
        kw: dict[str, Any] = {"diameter": diameter}
        if minmass is not None:
            kw["minmass"] = minmass
        features = tp.locate(img, **kw)
        features["frame"] = i
        frames.append(features)

    all_features = pd.concat(frames, ignore_index=True)
    tracks = tp.link(all_features, search_range=search_range, memory=memory)
    return tracks


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def extract_nd_signals(
    image_stack: NDArray,
    tracks: Any,
    radius: int = 5,
) -> dict[str, Any]:
    """Extract per-ND intensity time series from tracked positions.

    Parameters
    ----------
    image_stack : (n_frames, H, W) ndarray
    tracks : DataFrame
        Output from :func:`track_nds`.
    radius : int
        Aperture radius (pixels) for integration.

    Returns
    -------
    result : dict
        ``signals`` : (n_particles, n_frames) ndarray.
        ``particle_ids`` : list of particle labels.
        ``positions`` : (n_particles, 2) mean (x, y) pixel positions.
    """
    import pandas as pd

    particle_ids = sorted(tracks["particle"].unique())
    n_frames = image_stack.shape[0]
    n_particles = len(particle_ids)

    signals = np.zeros((n_particles, n_frames))
    positions = np.zeros((n_particles, 2))

    for pi, pid in enumerate(particle_ids):
        sub = tracks[tracks["particle"] == pid]
        positions[pi, 0] = sub["x"].mean()
        positions[pi, 1] = sub["y"].mean()

        for _, row in sub.iterrows():
            frame = int(row["frame"])
            cx, cy = int(round(row["x"])), int(round(row["y"]))
            img = image_stack[frame]
            h, w = img.shape

            y0 = max(0, cy - radius)
            y1 = min(h, cy + radius + 1)
            x0 = max(0, cx - radius)
            x1 = min(w, cx + radius + 1)

            signals[pi, frame] = np.sum(img[y0:y1, x0:x1])

    return {
        "signals": signals,
        "particle_ids": particle_ids,
        "positions": positions,
    }
