"""Publication-quality visualisation for time-domain NV magnetometry.

All plotting routines accept an optional *ax* argument.  If ``None``,
a new figure/axes pair is created and the ``Figure`` is returned.
When *ax* is provided, the plot is drawn onto it and ``None`` is
returned, allowing composition of multi-panel figures.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for headless use
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from wimp.constants import seconds_to_us, tesla_to_mt, hz_to_mhz

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_ax(ax: plt.Axes | None, **fig_kw) -> tuple[Figure | None, plt.Axes]:
    if ax is not None:
        return None, ax
    fig, ax = plt.subplots(**fig_kw)
    return fig, ax


# ---------------------------------------------------------------------------
# Time-domain curves
# ---------------------------------------------------------------------------


def plot_ramsey_fringe(
    tau: NDArray,
    signal: NDArray,
    fit_result: dict | None = None,
    *,
    ax: plt.Axes | None = None,
    tau_units: str = "us",
) -> Figure | None:
    """Plot a Ramsey fringe with optional fit overlay."""
    fig, ax = _get_ax(ax)
    t = np.asarray(tau)
    scale = 1e6 if tau_units == "us" else 1.0
    label_unit = "us" if tau_units == "us" else "s"

    ax.plot(t * scale, signal, "o", ms=3, alpha=0.6, label="Data")

    if fit_result is not None:
        from wimp.relaxation import ramsey_model
        t_fit = np.linspace(t.min(), t.max(), 500)
        s_fit = ramsey_model(
            t_fit,
            fit_result["amplitude"],
            fit_result["frequency"],
            fit_result["t2star"],
            fit_result.get("phase", 0),
            fit_result.get("offset", 0),
            fit_result.get("stretch", 1),
        )
        ax.plot(t_fit * scale, s_fit, "-", lw=1.5, label="Fit")
        b_ut = tesla_to_mt(fit_result.get("b_field", 0)) * 1e3  # to uT
        ax.set_title(f"Ramsey  |  B = {b_ut:.2f} uT  |  T2* = {fit_result['t2star']*1e6:.1f} us")

    ax.set_xlabel(f"Free evolution time ({label_unit})")
    ax.set_ylabel("Signal (a.u.)")
    ax.legend(fontsize=8)
    if fig is not None:
        fig.tight_layout()
    return fig


def plot_decay_curve(
    tau: NDArray,
    signal: NDArray,
    fit_result: dict | None = None,
    *,
    decay_type: str = "t2",
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plot a T₂ or generic exponential decay curve."""
    fig, ax = _get_ax(ax)
    t = np.asarray(tau)

    ax.plot(t * 1e6, signal, "o", ms=3, alpha=0.6, label="Data")

    if fit_result is not None:
        from wimp.relaxation import t2_decay_model
        t_fit = np.linspace(t.min(), t.max(), 500)
        s_fit = t2_decay_model(
            t_fit,
            fit_result["amplitude"],
            fit_result["t2"],
            fit_result.get("offset", 0),
            fit_result.get("stretch", 1),
        )
        ax.plot(t_fit * 1e6, s_fit, "-", lw=1.5, label="Fit")
        ax.set_title(f"T2 decay  |  T2 = {fit_result['t2']*1e6:.1f} us")

    ax.set_xlabel("2*tau (us)")
    ax.set_ylabel("Signal (a.u.)")
    ax.legend(fontsize=8)
    if fig is not None:
        fig.tight_layout()
    return fig


def plot_t1_recovery(
    tau: NDArray,
    signal: NDArray,
    fit_result: dict | None = None,
    *,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plot a T₁ spin-lattice recovery curve."""
    fig, ax = _get_ax(ax)
    t = np.asarray(tau)

    ax.plot(t * 1e3, signal, "o", ms=3, alpha=0.6, label="Data")

    if fit_result is not None:
        from wimp.relaxation import t1_recovery_model
        t_fit = np.linspace(t.min(), t.max(), 500)
        s_fit = t1_recovery_model(
            t_fit,
            fit_result["amplitude"],
            fit_result["t1"],
            fit_result.get("contrast", 1),
            fit_result.get("offset", 0),
        )
        ax.plot(t_fit * 1e3, s_fit, "-", lw=1.5, label="Fit")
        ax.set_title(f"T1 recovery  |  T1 = {fit_result['t1']*1e3:.2f} ms")

    ax.set_xlabel("Wait time (ms)")
    ax.set_ylabel("Signal (a.u.)")
    ax.legend(fontsize=8)
    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Filter functions
# ---------------------------------------------------------------------------


def plot_filter_function(
    freq: NDArray,
    W: NDArray,
    *,
    sequence_name: str = "",
    ax: plt.Axes | None = None,
    log_x: bool = True,
) -> Figure | None:
    """Plot a pulse-sequence filter function W(f)."""
    fig, ax = _get_ax(ax)

    ax.plot(np.asarray(freq) * 1e-6, W, lw=1.5)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("W(f) (a.u.)")
    ax.set_title(f"Filter function{' — ' + sequence_name if sequence_name else ''}")
    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Noise spectrum
# ---------------------------------------------------------------------------


def plot_noise_spectrum(
    freq: NDArray,
    spectral_density: NDArray,
    *,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plot an environmental noise spectrum S(f)."""
    fig, ax = _get_ax(ax)

    ax.loglog(np.asarray(freq) * 1e-6, spectral_density, lw=1.5)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("S(f) (T$^2$ / Hz)")
    ax.set_title("Noise spectrum")
    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Field time series
# ---------------------------------------------------------------------------


def plot_field_timeseries(
    time: NDArray,
    field: NDArray,
    *,
    nd_labels: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plot magnetic-field time series for one or more NDs."""
    fig, ax = _get_ax(ax)
    t = np.asarray(time) * 1e3  # ms
    f = np.atleast_2d(np.asarray(field))

    for i in range(f.shape[0]):
        label = nd_labels[i] if nd_labels else f"ND {i}"
        ax.plot(t, tesla_to_mt(f[i]) * 1e3, lw=0.8, label=label)  # uT

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("B (uT)")
    ax.set_title("Field time series")
    if f.shape[0] <= 10:
        ax.legend(fontsize=7)
    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sensitivity comparison
# ---------------------------------------------------------------------------


def plot_sensitivity_comparison(
    protocols_df: Any,
    *,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Bar chart comparing sensitivity across protocols."""
    fig, ax = _get_ax(ax)
    import pandas as pd

    df = protocols_df
    valid = df.dropna(subset=["sensitivity_T_sqrtHz"])
    ax.barh(valid["protocol"], valid["sensitivity_T_sqrtHz"] * 1e9)
    ax.set_xlabel("Sensitivity (nT / sqrt(Hz))")
    ax.set_title("Protocol sensitivity comparison")
    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Source map
# ---------------------------------------------------------------------------


def plot_source_map(
    neuron_positions: NDArray,
    currents: NDArray,
    *,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Scatter plot of estimated neural source amplitudes."""
    fig, ax = _get_ax(ax)
    pos = np.asarray(neuron_positions)
    amp = np.asarray(currents).ravel()

    if pos.shape[1] >= 2:
        sc = ax.scatter(
            pos[:, 0] * 1e6, pos[:, 1] * 1e6,
            c=amp, cmap="RdBu_r", s=60, edgecolors="k", linewidths=0.5,
        )
        plt.colorbar(sc, ax=ax, label="Current (a.u.)")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Source map")
    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2-D field map
# ---------------------------------------------------------------------------


def plot_field_map_2d(
    nd_positions: NDArray,
    field_values: NDArray,
    *,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Scatter plot of field values at ND positions."""
    fig, ax = _get_ax(ax)
    pos = np.asarray(nd_positions)
    fv = np.asarray(field_values).ravel()

    sc = ax.scatter(
        pos[:, 0] * 1e6, pos[:, 1] * 1e6,
        c=tesla_to_mt(fv) * 1e3, cmap="viridis", s=60, edgecolors="k",
    )
    plt.colorbar(sc, ax=ax, label="B (uT)")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Field map")
    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Image overlay
# ---------------------------------------------------------------------------


def overlay_on_image(
    image: NDArray,
    nd_positions_px: NDArray,
    field_values: NDArray,
    *,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Overlay field-mapped ND markers on a fluorescence image."""
    fig, ax = _get_ax(ax)
    ax.imshow(image, cmap="gray", origin="lower")
    pos = np.asarray(nd_positions_px)
    fv = np.asarray(field_values).ravel()

    sc = ax.scatter(pos[:, 0], pos[:, 1], c=fv, cmap="hot", s=40, edgecolors="w", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="Field (a.u.)")
    ax.set_title("ND field overlay")
    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def plot_registration(
    centerline_pts: NDArray,
    atlas_positions: NDArray,
    nd_positions: NDArray,
    *,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plot worm centreline with atlas and ND positions."""
    fig, ax = _get_ax(ax)

    cl = np.asarray(centerline_pts)
    ax.plot(cl[:, 0] * 1e6, cl[:, 1] * 1e6, "k-", lw=1, label="Centreline")

    at = np.asarray(atlas_positions)
    ax.plot(at[:, 0] * 1e6, at[:, 1] * 1e6, "rs", ms=4, label="Atlas neurons")

    nd = np.asarray(nd_positions)
    ax.plot(nd[:, 0] * 1e6, nd[:, 1] * 1e6, "go", ms=5, label="NDs")

    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.legend(fontsize=8)
    ax.set_title("Atlas registration")
    ax.set_aspect("equal")
    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Resolution / crosstalk
# ---------------------------------------------------------------------------


def plot_resolution_matrix(
    R: NDArray,
    neuron_names: list[str] | None = None,
    *,
    ax: plt.Axes | None = None,
    cmap: str = "RdBu_r",
    vmax: float | None = None,
) -> Figure | None:
    """Plot the MNE resolution matrix as a heatmap."""
    fig, ax = _get_ax(ax)
    n = R.shape[0]
    vm = vmax if vmax is not None else float(np.max(np.abs(R)))
    im = ax.imshow(R, cmap=cmap, vmin=-vm, vmax=vm, origin="upper", aspect="equal")
    plt.colorbar(im, ax=ax, label="Resolution weight")

    if neuron_names is not None and n <= 30:
        ax.set_xticks(range(n))
        ax.set_xticklabels(neuron_names, rotation=90, fontsize=6)
        ax.set_yticks(range(n))
        ax.set_yticklabels(neuron_names, fontsize=6)

    ax.set_xlabel("True source")
    ax.set_ylabel("Estimated source")
    ax.set_title("Resolution matrix")
    if fig is not None:
        fig.tight_layout()
    return fig


def plot_crosstalk_summary(
    metrics: dict,
    *,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Grouped bar chart of self-resolution and crosstalk ratio per neuron."""
    fig, ax = _get_ax(ax)
    names = metrics["neuron_names"]
    n = len(names)
    x = np.arange(n)
    width = 0.35

    ax.bar(x - width / 2, metrics["diagonal"], width, label="Self-resolution")
    ax.bar(x + width / 2, metrics["crosstalk_ratio"], width, label="Crosstalk ratio")

    if n <= 20:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Value")
    ax.set_title("Self-resolution vs. crosstalk")
    ax.legend(fontsize=8)
    if fig is not None:
        fig.tight_layout()
    return fig


def plot_deformable_registration(
    dereg_result: Any,
    frame_idx: int,
    nd_positions: NDArray,
    *,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plot a single frame from a deformable registration result."""
    neuron_pos = dereg_result.neuron_positions[frame_idx]

    cl = None
    if frame_idx < len(dereg_result.centerlines):
        cl = dereg_result.centerlines[frame_idx]

    centerline_pts = cl.curve if cl is not None else neuron_pos

    return plot_registration(centerline_pts, neuron_pos, nd_positions, ax=ax)


def plot_neuron_trajectories(
    neuron_positions: NDArray,
    neuron_names: list[str] | None = None,
    *,
    coordinate: int = 1,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Line plot of neuron position along one axis over frames.

    Parameters
    ----------
    neuron_positions : (n_frames, n_neurons, 3) array
    neuron_names : list of str, optional
    coordinate : int
        Axis index (0=x, 1=y, 2=z).
    """
    fig, ax = _get_ax(ax)
    n_frames, n_neurons, _ = neuron_positions.shape
    coord_labels = {0: "x", 1: "y", 2: "z"}

    for j in range(n_neurons):
        label = neuron_names[j] if neuron_names else f"Neuron {j}"
        ax.plot(range(n_frames), neuron_positions[:, j, coordinate] * 1e6, lw=0.8, label=label)

    ax.set_xlabel("Frame")
    ax.set_ylabel(f"{coord_labels.get(coordinate, '?')} position (um)")
    ax.set_title("Neuron trajectories")
    if n_neurons <= 10:
        ax.legend(fontsize=7)
    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Neural activity graph ("audio visualiser")
# ---------------------------------------------------------------------------


def _compute_layout(
    neuron_positions: NDArray,
    layout: str,
    n_neurons: int,
) -> NDArray:
    """Return 2-D layout positions for the activity graph."""
    if layout == "circular":
        angles = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)
        radius = 1.0
        return np.column_stack([np.cos(angles), np.sin(angles)]) * radius
    # "body" or "custom": use neuron_positions[:, :2]
    pos = np.asarray(neuron_positions)
    if pos.ndim == 1:
        return pos[:2][np.newaxis, :]
    return pos[:, :2] * 1e6  # convert metres → um


def plot_neural_activity_graph(
    source_estimate: NDArray,
    neuron_positions: NDArray,
    *,
    neuron_names: list[str] | None = None,
    layout: str = "body",
    time_axis: NDArray | None = None,
    connections: NDArray | None = None,
    colormap: str = "coolwarm",
    line_width: float = 0.8,
    waveform_scale: float = 1.0,
    alpha: float = 0.8,
    show_labels: bool = True,
    show_connections: bool = False,
    highlight_active: bool = True,
    active_threshold: float = 0.5,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Audio-visualiser-style plot of neural activity at each neuron position.

    Each neuron is placed on the canvas according to *layout*, and its
    source-estimate waveform is drawn as a line emanating from that
    position, coloured by peak activity intensity.

    Parameters
    ----------
    source_estimate : (n_neurons, n_timepoints)
    neuron_positions : (n_neurons, 2 or 3) in metres
    layout : str
        ``"body"`` (anatomical x/y), ``"circular"`` (ring), or
        ``"custom"`` (use positions as-is).
    connections : (n_edges, 2) int array, optional
        Neuron index pairs for connectivity edges.
    waveform_scale : float
        Scaling factor for waveform amplitude display.
    """
    fig, ax = _get_ax(ax, figsize=(10, 6))

    src = np.atleast_2d(np.asarray(source_estimate, dtype=float))
    n_neurons, n_t = src.shape
    pos_2d = _compute_layout(neuron_positions, layout, n_neurons)
    cmap = plt.get_cmap(colormap)

    # Compute spacings for waveform scaling
    if n_neurons > 1:
        from scipy.spatial.distance import pdist
        dists = pdist(pos_2d)
        spacing = np.median(dists) if len(dists) > 0 else 1.0
    else:
        spacing = 1.0
    spacing = max(spacing, 1e-10)

    global_max = np.max(np.abs(src)) if np.max(np.abs(src)) > 0 else 1.0

    # Draw connections
    if show_connections and connections is not None:
        conns = np.asarray(connections)
        for edge in conns:
            i, j = int(edge[0]), int(edge[1])
            ax.plot(
                [pos_2d[i, 0], pos_2d[j, 0]],
                [pos_2d[i, 1], pos_2d[j, 1]],
                "k-", lw=0.3, alpha=0.3, zorder=0,
            )

    # Draw waveforms
    for j in range(n_neurons):
        trace = src[j]
        peak_activity = np.max(np.abs(trace)) / global_max
        colour = cmap(0.5 + 0.5 * peak_activity * np.sign(trace.mean()))

        x_base = pos_2d[j, 0]
        y_base = pos_2d[j, 1]

        # Time trace runs along x, amplitude along y
        t_offsets = np.linspace(-0.4, 0.4, n_t) * spacing * waveform_scale
        y_offsets = trace / global_max * spacing * 0.4 * waveform_scale

        ax.plot(
            x_base + t_offsets, y_base + y_offsets,
            lw=line_width, alpha=alpha, color=colour, zorder=2,
        )

        # Highlight active neurons
        if highlight_active and peak_activity >= active_threshold:
            ax.scatter(
                [x_base], [y_base], s=120, c=[peak_activity],
                cmap=colormap, vmin=0, vmax=1,
                alpha=0.3, edgecolors="none", zorder=1,
            )

        if show_labels:
            name = neuron_names[j] if neuron_names and j < len(neuron_names) else f"N{j}"
            ax.annotate(
                name, (x_base, y_base), fontsize=6, ha="center", va="bottom",
                xytext=(0, spacing * 0.25 * waveform_scale),
                textcoords="offset points",
            )

    if layout == "circular":
        ax.set_aspect("equal")
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")

    ax.set_title("Neural activity graph")
    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Animated deformable registration
# ---------------------------------------------------------------------------


def animate_deformable_registration(
    dereg_result: Any,
    nd_positions: NDArray,
    *,
    interval: int = 200,
    repeat: bool = True,
    save_path: str | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> Any:
    """Generate an animation of per-frame deformable registration.

    Parameters
    ----------
    dereg_result : DeformableRegistrationResult
    nd_positions : (n_nds, 3) or (n_frames, n_nds, 3) array
    interval : int
        Milliseconds between frames.
    save_path : str, optional
        If given, save the animation to this file (requires a writer).

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    from matplotlib.animation import FuncAnimation

    nd_pos = np.asarray(nd_positions)
    n_frames = dereg_result.neuron_positions.shape[0]

    fig_anim, ax_anim = plt.subplots(figsize=figsize)

    def _update(frame_idx):
        ax_anim.clear()
        # Neuron positions for this frame
        neuron_pos = dereg_result.neuron_positions[frame_idx]

        # ND positions (static or per-frame)
        if nd_pos.ndim == 3:
            nd = nd_pos[frame_idx]
        else:
            nd = nd_pos

        # Centerline
        if frame_idx < len(dereg_result.centerlines) and dereg_result.centerlines[frame_idx] is not None:
            cl = dereg_result.centerlines[frame_idx].curve
            ax_anim.plot(cl[:, 0] * 1e6, cl[:, 1] * 1e6, "k-", lw=1, label="Centreline")

        ax_anim.plot(neuron_pos[:, 0] * 1e6, neuron_pos[:, 1] * 1e6, "rs", ms=4, label="Neurons")
        ax_anim.plot(nd[:, 0] * 1e6, nd[:, 1] * 1e6, "go", ms=5, label="NDs")
        ax_anim.set_xlabel("x (um)")
        ax_anim.set_ylabel("y (um)")
        ax_anim.set_title(f"Frame {frame_idx}")
        ax_anim.legend(fontsize=7)
        ax_anim.set_aspect("equal")

    anim = FuncAnimation(fig_anim, _update, frames=n_frames, interval=interval, repeat=repeat)

    if save_path is not None:
        anim.save(save_path)

    return anim


# ---------------------------------------------------------------------------
# Interactive napari viewer
# ---------------------------------------------------------------------------


def napari_view_experiment(
    nd_positions: NDArray,
    field_values: NDArray | None = None,
    neuron_positions: NDArray | None = None,
    source_estimate: NDArray | None = None,
    neuron_names: list[str] | None = None,
    *,
    image: NDArray | None = None,
) -> Any:
    """Launch a napari viewer with experiment data layers.

    Requires ``napari`` (install with ``pip install 'wimp[interactive]'``).

    Parameters
    ----------
    nd_positions : (n_nds, 3) or (n_nds, 2)
    field_values : (n_nds,) optional
    neuron_positions : (n_neurons, 3) optional
    source_estimate : (n_neurons,) optional
    neuron_names : list of str, optional
    image : ndarray, optional
        Background fluorescence image.

    Returns
    -------
    viewer : napari.Viewer
    """
    try:
        import napari
    except ImportError:
        raise ImportError(
            "napari is required for interactive viewing. "
            "Install it with: pip install 'wimp[interactive]'"
        ) from None

    viewer = napari.Viewer()

    if image is not None:
        viewer.add_image(image, name="Fluorescence")

    nd = np.asarray(nd_positions)
    # Convert to um for display
    nd_um = nd * 1e6 if nd.max() < 1.0 else nd
    # napari expects (row, col) or (z, y, x) — use y, x
    if nd_um.shape[1] >= 3:
        nd_display = nd_um[:, [1, 0]]  # y, x
    else:
        nd_display = nd_um[:, :2]

    nd_kwargs: dict[str, Any] = {"name": "Nanodiamonds", "size": 5}
    if field_values is not None:
        fv = np.asarray(field_values).ravel()
        nd_kwargs["face_color"] = "field"
        nd_kwargs["properties"] = {"field": fv}
        nd_kwargs["face_colormap"] = "viridis"
    else:
        nd_kwargs["face_color"] = "green"

    viewer.add_points(nd_display, **nd_kwargs)

    if neuron_positions is not None:
        npos = np.asarray(neuron_positions)
        npos_um = npos * 1e6 if npos.max() < 1.0 else npos
        if npos_um.shape[1] >= 3:
            npos_display = npos_um[:, [1, 0]]
        else:
            npos_display = npos_um[:, :2]

        n_kwargs: dict[str, Any] = {"name": "Neurons", "size": 8, "symbol": "square"}
        if source_estimate is not None:
            se = np.asarray(source_estimate).ravel()
            n_kwargs["face_color"] = "activity"
            n_kwargs["properties"] = {"activity": se}
            if neuron_names is not None:
                n_kwargs["properties"]["name"] = neuron_names
            n_kwargs["face_colormap"] = "RdBu"
        else:
            n_kwargs["face_color"] = "red"
            if neuron_names is not None:
                n_kwargs["properties"] = {"name": neuron_names}

        viewer.add_points(npos_display, **n_kwargs)

    return viewer


# ---------------------------------------------------------------------------
# Connectome graph overlays
# ---------------------------------------------------------------------------


def plot_connectome_graph(
    connectome: dict[str, Any],
    *,
    activity: NDArray | None = None,
    propagation_step: int | None = None,
    layout: str = "circular",
    neuron_names: list[str] | None = None,
    show_weights: bool = False,
    edge_threshold: float = 0.0,
    colormap: str = "coolwarm",
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plot a connectome as a directed graph with optional activity overlay.

    Requires the optional ``networkx`` dependency.  Install with
    ``pip install 'wimp[graph]'``.

    Parameters
    ----------
    connectome : dict
        Connectome dict with keys ``"neuron_names"``/``"neurons"`` and
        ``"adjacency"``.  Accepted from :func:`wimp.connectome.load_connectome`
        or :func:`wimp.connectome.known_circuit`.
    activity : ndarray | None
        Per-node activity values for colouring.  If a 2-D trajectory
        ``(n_steps, n_nodes)`` is given, use *propagation_step* to
        select which row to display.
    propagation_step : int | None
        Row index into a 2-D *activity* array.
    layout : str
        ``"circular"`` or ``"spring"``.
    neuron_names : list[str] | None
        Override label list.  Defaults to connectome node names.
    show_weights : bool
        Annotate edges with synapse weights.
    edge_threshold : float
        Hide edges with absolute weight below this value.
    colormap : str
        Matplotlib colormap name for node activity colours.
    ax : Axes | None
        Existing axes to draw on.

    Returns
    -------
    Figure | None
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "Connectome graph plotting requires networkx. "
            "Install with: pip install 'wimp[graph]'"
        ) from None

    # Accept both full connectome and circuit dicts
    if "neuron_names" in connectome:
        names = connectome["neuron_names"]
    elif "neurons" in connectome:
        names = connectome["neurons"]
    else:
        raise ValueError("connectome must contain 'neuron_names' or 'neurons'")

    adj = np.asarray(connectome["adjacency"])
    n = len(names)

    if neuron_names is None:
        neuron_names = list(names)

    # Resolve activity values
    node_vals = None
    if activity is not None:
        activity = np.asarray(activity)
        if activity.ndim == 2:
            step = propagation_step if propagation_step is not None else activity.shape[0] - 1
            node_vals = activity[step]
        else:
            node_vals = activity

    # Build graph
    G = nx.DiGraph()
    for i, name in enumerate(neuron_names):
        G.add_node(name)
    for i in range(n):
        for j in range(n):
            w = adj[i, j]
            if abs(w) > edge_threshold:
                G.add_edge(neuron_names[i], neuron_names[j], weight=float(w))

    fig, ax = _get_ax(ax)

    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.circular_layout(G)

    # Node colours
    cmap = plt.get_cmap(colormap)
    if node_vals is not None:
        vmin, vmax = np.min(node_vals), np.max(node_vals)
        if vmax == vmin:
            vmax = vmin + 1.0
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        node_colors = [cmap(norm(node_vals[i])) for i in range(n)]
    else:
        node_colors = "#6baed6"

    # Edge widths proportional to weight
    edges = list(G.edges(data=True))
    if edges:
        weights = [abs(e[2].get("weight", 1.0)) for e in edges]
        max_w = max(weights) if weights else 1.0
        widths = [1.0 + 2.0 * w / max_w for w in weights]
    else:
        widths = []

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=400)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
    nx.draw_networkx_edges(
        G, pos, ax=ax, width=widths, alpha=0.6,
        arrows=True, arrowsize=10, connectionstyle="arc3,rad=0.1",
    )

    if show_weights and edges:
        edge_labels = {(e[0], e[1]): f"{e[2]['weight']:.1f}" for e in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=5)

    # Colorbar
    if node_vals is not None and fig is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="activity", shrink=0.8)

    ax.set_title("Connectome graph")
    ax.axis("off")
    return fig


def plot_connectome_propagation(
    trajectory: NDArray,
    connectome: dict[str, Any],
    *,
    steps: list[int] | None = None,
    layout: str = "circular",
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plot propagation across a connectome at multiple time steps.

    Parameters
    ----------
    trajectory : ndarray, shape ``(n_steps+1, n_nodes)``
        Activity trajectory from
        :func:`wimp.connectome.connectome_propagation`.
    connectome : dict
        Connectome dict.
    steps : list[int] | None
        Indices into *trajectory* to display.  Defaults to 4
        evenly spaced steps including start and end.
    layout : str
        Graph layout algorithm.
    ax : Axes | None
        If provided, only the last step is plotted in this axes.

    Returns
    -------
    Figure | None
    """
    trajectory = np.asarray(trajectory)
    n_total = trajectory.shape[0]

    if ax is not None:
        plot_connectome_graph(
            connectome, activity=trajectory, propagation_step=n_total - 1,
            layout=layout, ax=ax,
        )
        return None

    if steps is None:
        indices = np.linspace(0, n_total - 1, min(4, n_total), dtype=int)
        steps = list(indices)

    n_panels = len(steps)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    for panel_ax, step_idx in zip(axes, steps):
        plot_connectome_graph(
            connectome, activity=trajectory, propagation_step=step_idx,
            layout=layout, ax=panel_ax,
        )
        panel_ax.set_title(f"Step {step_idx}")

    fig.tight_layout()
    return fig


def plot_circuit_diagram(
    circuit_name: str,
    *,
    activity: NDArray | None = None,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plot a named C. elegans circuit as a graph.

    Convenience wrapper around :func:`plot_connectome_graph` that
    calls :func:`wimp.connectome.known_circuit` internally.

    Parameters
    ----------
    circuit_name : str
        e.g. ``"tap_withdrawal"``, ``"thermotaxis"``, ``"chemotaxis"``,
        ``"nose_touch"``.
    activity : ndarray | None
        Per-neuron activity values for colouring.
    ax : Axes | None
        Existing axes to draw on.

    Returns
    -------
    Figure | None
    """
    from wimp.connectome import known_circuit

    circ = known_circuit(circuit_name)
    return plot_connectome_graph(circ, activity=activity, ax=ax)
