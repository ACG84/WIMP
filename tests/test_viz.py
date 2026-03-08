"""Tests for wimp.viz."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from wimp.viz import (
    plot_ramsey_fringe,
    plot_decay_curve,
    plot_t1_recovery,
    plot_odmr_spectrum,
    plot_filter_function,
    plot_noise_spectrum,
    plot_field_timeseries,
    plot_sensitivity_comparison,
    plot_source_map,
    plot_field_map_2d,
    overlay_on_image,
    plot_registration,
    plot_resolution_matrix,
    plot_crosstalk_summary,
    plot_deformable_registration,
    plot_neuron_trajectories,
)


@pytest.fixture(autouse=True)
def _close_figs():
    """Close all figures after each test."""
    yield
    plt.close("all")


class TestPlotRamseyFringe:
    def test_returns_figure(self):
        tau = np.linspace(0, 5e-6, 100)
        signal = np.cos(2 * np.pi * 1e6 * tau)
        fig = plot_ramsey_fringe(tau, signal)
        assert isinstance(fig, Figure)

    def test_with_fit_result(self):
        tau = np.linspace(0, 5e-6, 100)
        signal = np.cos(2 * np.pi * 1e6 * tau)
        fit = {
            "amplitude": 1.0, "frequency": 1e6, "t2star": 2e-6,
            "phase": 0.0, "offset": 0.0, "stretch": 1.0, "b_field": 3.6e-5,
        }
        fig = plot_ramsey_fringe(tau, signal, fit)
        assert isinstance(fig, Figure)

    def test_with_existing_ax(self):
        tau = np.linspace(0, 5e-6, 100)
        signal = np.cos(2 * np.pi * 1e6 * tau)
        _, ax = plt.subplots()
        result = plot_ramsey_fringe(tau, signal, ax=ax)
        assert result is None  # draws onto existing ax


class TestPlotDecayCurve:
    def test_returns_figure(self):
        tau = np.linspace(0, 300e-6, 100)
        signal = np.exp(-tau / 100e-6)
        fig = plot_decay_curve(tau, signal)
        assert isinstance(fig, Figure)

    def test_with_fit(self):
        tau = np.linspace(0, 300e-6, 100)
        signal = np.exp(-tau / 100e-6)
        fit = {"amplitude": 1.0, "t2": 100e-6, "offset": 0.0, "stretch": 1.0}
        fig = plot_decay_curve(tau, signal, fit)
        assert isinstance(fig, Figure)


class TestPlotT1Recovery:
    def test_returns_figure(self):
        tau = np.linspace(0, 25e-3, 100)
        signal = 1 - np.exp(-tau / 5e-3)
        fig = plot_t1_recovery(tau, signal)
        assert isinstance(fig, Figure)


class TestPlotFilterFunction:
    def test_returns_figure(self):
        freq = np.linspace(1e3, 1e7, 1000)
        W = np.sin(freq / 1e6) ** 2
        fig = plot_filter_function(freq, W)
        assert isinstance(fig, Figure)


class TestPlotNoiseSpectrum:
    def test_returns_figure(self):
        freq = np.logspace(3, 7, 100)
        sd = 1e-20 * np.ones_like(freq)
        fig = plot_noise_spectrum(freq, sd)
        assert isinstance(fig, Figure)


class TestPlotFieldTimeseries:
    def test_returns_figure(self):
        time = np.arange(100) * 1e-3
        field = np.random.default_rng(0).normal(size=(3, 100)) * 1e-9
        fig = plot_field_timeseries(time, field)
        assert isinstance(fig, Figure)

    def test_with_labels(self):
        time = np.arange(100) * 1e-3
        field = np.random.default_rng(0).normal(size=(2, 100)) * 1e-9
        fig = plot_field_timeseries(time, field, nd_labels=["A", "B"])
        assert isinstance(fig, Figure)


class TestPlotSensitivityComparison:
    def test_returns_figure(self):
        import pandas as pd
        df = pd.DataFrame({
            "protocol": ["Ramsey", "Echo"],
            "sensitivity_T_sqrtHz": [1e-9, 5e-10],
        })
        fig = plot_sensitivity_comparison(df)
        assert isinstance(fig, Figure)


class TestPlotSourceMap:
    def test_returns_figure(self):
        pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]) * 1e-6
        currents = np.array([0.5, 1.0, 0.3])
        fig = plot_source_map(pos, currents)
        assert isinstance(fig, Figure)


class TestPlotFieldMap2D:
    def test_returns_figure(self):
        pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]) * 1e-6
        field = np.array([1e-9, 2e-9, 1.5e-9])
        fig = plot_field_map_2d(pos, field)
        assert isinstance(fig, Figure)


class TestOverlayOnImage:
    def test_returns_figure(self):
        img = np.random.default_rng(0).normal(size=(100, 100))
        pos = np.array([[50, 50], [30, 70]])
        fv = np.array([1.0, 2.0])
        fig = overlay_on_image(img, pos, fv)
        assert isinstance(fig, Figure)


class TestPlotRegistration:
    def test_returns_figure(self):
        cl = np.column_stack([np.linspace(0, 1e-3, 50), np.zeros(50), np.zeros(50)])
        atlas = np.column_stack([np.linspace(0.1e-3, 0.9e-3, 5), np.zeros(5), np.zeros(5)])
        nds = np.column_stack([np.linspace(0.1e-3, 0.9e-3, 8), np.zeros(8), np.zeros(8)])
        fig = plot_registration(cl, atlas, nds)
        assert isinstance(fig, Figure)


class TestPlotResolutionMatrix:
    def test_returns_figure(self):
        R = np.eye(5)
        fig = plot_resolution_matrix(R)
        assert isinstance(fig, Figure)

    def test_with_names(self):
        R = np.random.default_rng(0).normal(size=(4, 4))
        fig = plot_resolution_matrix(R, ["A", "B", "C", "D"])
        assert isinstance(fig, Figure)


class TestPlotCrosstalkSummary:
    def test_returns_figure(self):
        metrics = {
            "neuron_names": ["A", "B", "C"],
            "diagonal": np.array([0.8, 0.7, 0.9]),
            "crosstalk_ratio": np.array([0.1, 0.2, 0.05]),
        }
        fig = plot_crosstalk_summary(metrics)
        assert isinstance(fig, Figure)


class TestPlotDeformableRegistration:
    def test_returns_figure(self):
        from wimp.registration import (
            DeformableRegistrationResult,
            CenterlineResult,
        )

        n_neurons = 3
        curve = np.column_stack([
            np.linspace(0, 1e-3, 50), np.zeros(50), np.zeros(50)
        ])
        cl = CenterlineResult(
            control_points=curve[:10],
            arc_length=np.linspace(0, 1e-3, 50),
            curve=curve,
            body_length=1e-3,
            tck=None,
        )
        dereg = DeformableRegistrationResult(
            neuron_names=["A", "B", "C"],
            neuron_positions=np.zeros((2, n_neurons, 3)),
            covariance=np.zeros((2, n_neurons, 3, 3)),
            centerlines=[cl, cl],
            frame_indices=np.array([0, 1]),
        )
        nd_pos = np.column_stack([
            np.linspace(0.1e-3, 0.9e-3, 5), np.zeros(5), np.zeros(5)
        ])
        fig = plot_deformable_registration(dereg, 0, nd_pos)
        assert isinstance(fig, Figure)


class TestPlotNeuronTrajectories:
    def test_returns_figure(self):
        pos = np.random.default_rng(0).normal(size=(20, 3, 3)) * 1e-6
        fig = plot_neuron_trajectories(pos)
        assert isinstance(fig, Figure)

    def test_with_names(self):
        pos = np.random.default_rng(0).normal(size=(10, 2, 3)) * 1e-6
        fig = plot_neuron_trajectories(pos, ["N0", "N1"])
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Neural activity graph
# ---------------------------------------------------------------------------

from wimp.viz import plot_neural_activity_graph


class TestPlotNeuralActivityGraph:
    def test_returns_figure_body_layout(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(5, 50))
        pos = rng.normal(size=(5, 3)) * 1e-4
        fig = plot_neural_activity_graph(src, pos, layout="body")
        assert isinstance(fig, Figure)

    def test_returns_figure_circular_layout(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(5, 50))
        pos = rng.normal(size=(5, 3)) * 1e-4
        fig = plot_neural_activity_graph(src, pos, layout="circular")
        assert isinstance(fig, Figure)

    def test_with_connections(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(4, 30))
        pos = rng.normal(size=(4, 3)) * 1e-4
        connections = np.array([[0, 1], [1, 2], [2, 3]])
        fig = plot_neural_activity_graph(
            src, pos, connections=connections,
            show_connections=True,
        )
        assert isinstance(fig, Figure)

    def test_with_neuron_names(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(3, 20))
        pos = rng.normal(size=(3, 2)) * 1e-4
        fig = plot_neural_activity_graph(
            src, pos, neuron_names=["AVAL", "AVAR", "AVBL"],
        )
        assert isinstance(fig, Figure)

    def test_with_existing_ax(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(3, 20))
        pos = rng.normal(size=(3, 2)) * 1e-4
        _, ax = plt.subplots()
        result = plot_neural_activity_graph(src, pos, ax=ax)
        assert result is None

    def test_single_neuron(self):
        src = np.array([[1.0, 2.0, 1.5, 0.5]])
        pos = np.array([[0.0, 0.0, 0.0]])
        fig = plot_neural_activity_graph(src, pos)
        assert isinstance(fig, Figure)

    def test_highlight_and_labels(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(4, 30))
        pos = rng.normal(size=(4, 3)) * 1e-4
        fig = plot_neural_activity_graph(
            src, pos,
            highlight_active=True,
            active_threshold=0.3,
            show_labels=True,
        )
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Animated deformable registration
# ---------------------------------------------------------------------------

from wimp.viz import animate_deformable_registration


class TestAnimateDeformableRegistration:
    def test_returns_animation(self):
        from matplotlib.animation import FuncAnimation
        from wimp.registration import (
            DeformableRegistrationResult,
            CenterlineResult,
        )

        curve = np.column_stack([
            np.linspace(0, 1e-3, 50), np.zeros(50), np.zeros(50)
        ])
        cl = CenterlineResult(
            control_points=curve[:10],
            arc_length=np.linspace(0, 1e-3, 50),
            curve=curve,
            body_length=1e-3,
            tck=None,
        )
        n_frames = 3
        n_neurons = 3
        dereg = DeformableRegistrationResult(
            neuron_names=["A", "B", "C"],
            neuron_positions=np.random.default_rng(0).normal(size=(n_frames, n_neurons, 3)) * 1e-4,
            covariance=np.zeros((n_frames, n_neurons, 3, 3)),
            centerlines=[cl] * n_frames,
            frame_indices=np.arange(n_frames),
        )
        nd_pos = np.column_stack([
            np.linspace(0.1e-3, 0.9e-3, 5), np.zeros(5), np.zeros(5)
        ])
        anim = animate_deformable_registration(dereg, nd_pos)
        assert isinstance(anim, FuncAnimation)


# ---------------------------------------------------------------------------
# Connectome graph visualisations
# ---------------------------------------------------------------------------

nx = pytest.importorskip("networkx")

from wimp.viz import plot_connectome_graph, plot_connectome_propagation, plot_circuit_diagram
from wimp.connectome import load_connectome, known_circuit, connectome_propagation


class TestPlotConnectomeGraph:
    def test_returns_figure(self):
        conn = load_connectome()
        fig = plot_connectome_graph(conn)
        assert isinstance(fig, Figure)

    def test_with_activity(self):
        conn = load_connectome()
        n = len(conn["neuron_names"])
        activity = np.random.default_rng(0).normal(size=n)
        fig = plot_connectome_graph(conn, activity=activity)
        assert isinstance(fig, Figure)

    def test_with_existing_ax(self):
        conn = load_connectome()
        _, ax = plt.subplots()
        result = plot_connectome_graph(conn, ax=ax)
        assert result is None

    def test_circuit_dict(self):
        circ = known_circuit("tap_withdrawal")
        fig = plot_connectome_graph(circ)
        assert isinstance(fig, Figure)

    def test_spring_layout(self):
        circ = known_circuit("thermotaxis")
        fig = plot_connectome_graph(circ, layout="spring")
        assert isinstance(fig, Figure)

    def test_show_weights(self):
        circ = known_circuit("chemotaxis")
        fig = plot_connectome_graph(circ, show_weights=True)
        assert isinstance(fig, Figure)


class TestPlotConnectomePropagation:
    def test_returns_figure(self):
        conn = load_connectome()
        n = len(conn["neuron_names"])
        a0 = np.zeros(n)
        a0[0] = 1.0
        traj = connectome_propagation(a0, conn["adjacency"], n_steps=5)
        fig = plot_connectome_propagation(traj, conn)
        assert isinstance(fig, Figure)

    def test_with_ax(self):
        circ = known_circuit("tap_withdrawal")
        n = len(circ["neurons"])
        a0 = np.zeros(n)
        a0[0] = 1.0
        traj = connectome_propagation(a0, circ["adjacency"], n_steps=3)
        _, ax = plt.subplots()
        result = plot_connectome_propagation(traj, circ, ax=ax)
        assert result is None


class TestPlotCircuitDiagram:
    def test_returns_figure(self):
        fig = plot_circuit_diagram("tap_withdrawal")
        assert isinstance(fig, Figure)

    def test_with_activity(self):
        circ = known_circuit("thermotaxis")
        n = len(circ["neurons"])
        activity = np.ones(n) * 0.5
        fig = plot_circuit_diagram("thermotaxis", activity=activity)
        assert isinstance(fig, Figure)


class TestPlotODMRSpectrum:
    def test_returns_figure(self):
        from wimp.constants import D0
        freq = np.linspace(2.86e9, 2.88e9, 200)
        signal = np.ones_like(freq) - 0.03 * (5e6)**2 / ((freq - D0)**2 + (5e6)**2)
        fig = plot_odmr_spectrum(freq, signal)
        assert isinstance(fig, Figure)

    def test_with_fit_result(self):
        from wimp.constants import D0, GAMMA_NV
        from wimp.relaxation import fit_odmr
        b = 1e-3
        f_minus = D0 - GAMMA_NV * b
        f_plus = D0 + GAMMA_NV * b
        freq = np.linspace(2.78e9, 2.96e9, 500)
        from wimp.relaxation import odmr_model
        signal = odmr_model(freq, 1.0, f_minus, f_plus, 0.03, 0.03, 5e6, 5e6)
        rng = np.random.default_rng(44)
        signal += rng.normal(0, 0.001, signal.shape)
        fit = fit_odmr(freq, signal)
        fig = plot_odmr_spectrum(freq, signal, fit)
        assert isinstance(fig, Figure)

    def test_with_existing_ax(self):
        from wimp.constants import D0
        freq = np.linspace(2.86e9, 2.88e9, 200)
        signal = np.ones_like(freq) - 0.03 * (5e6)**2 / ((freq - D0)**2 + (5e6)**2)
        _, ax = plt.subplots()
        result = plot_odmr_spectrum(freq, signal, ax=ax)
        assert result is None  # returns None when ax is provided

