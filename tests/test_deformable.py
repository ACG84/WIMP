"""Tests for deformable registration and related functionality."""

import numpy as np
import pytest

from wimp.registration import (
    DeformableRegistrationResult,
    tracks_to_frame_positions,
    deformable_register,
    smooth_neuron_trajectories,
    load_atlas,
)
from wimp.synthetic import generate_deformable_experiment
from wimp.constants import CANONICAL_BODY_LENGTH


# ---------------------------------------------------------------------------
# tracks_to_frame_positions
# ---------------------------------------------------------------------------


class TestTracksToFramePositions:
    def test_basic_conversion(self):
        import pandas as pd

        tracks = pd.DataFrame({
            "frame": [0, 0, 1, 1],
            "particle": [0, 1, 0, 1],
            "x": [10.0, 20.0, 11.0, 21.0],
            "y": [5.0, 6.0, 5.5, 6.5],
        })
        pos, frames = tracks_to_frame_positions(tracks)
        assert pos.shape == (2, 2, 3)
        assert frames.shape == (2,)
        np.testing.assert_array_equal(frames, [0, 1])
        assert pos[0, 0, 0] == 10.0
        assert pos[1, 1, 1] == 6.5

    def test_nan_for_missing_particle(self):
        import pandas as pd

        tracks = pd.DataFrame({
            "frame": [0, 0, 1],
            "particle": [0, 1, 0],
            "x": [10.0, 20.0, 11.0],
            "y": [5.0, 6.0, 5.5],
        })
        pos, frames = tracks_to_frame_positions(tracks)
        assert pos.shape == (2, 2, 3)
        assert np.isnan(pos[1, 1, 0])  # particle 1 missing in frame 1

    def test_pixel_scale(self):
        import pandas as pd

        tracks = pd.DataFrame({
            "frame": [0], "particle": [0], "x": [100.0], "y": [200.0],
        })
        pos, _ = tracks_to_frame_positions(tracks, pixel_scale=0.5e-6)
        assert pos[0, 0, 0] == pytest.approx(50e-6)
        assert pos[0, 0, 1] == pytest.approx(100e-6)

    def test_z_offset(self):
        import pandas as pd

        tracks = pd.DataFrame({
            "frame": [0], "particle": [0], "x": [10.0], "y": [5.0],
        })
        pos, _ = tracks_to_frame_positions(tracks, z_offset=5e-6)
        assert pos[0, 0, 2] == pytest.approx(5e-6)


# ---------------------------------------------------------------------------
# deformable_register
# ---------------------------------------------------------------------------


class TestDeformableRegister:
    def _make_straight_worm(self, n_nds=10, n_frames=5):
        """Straight worm along x-axis."""
        x = np.linspace(0.1 * CANONICAL_BODY_LENGTH,
                         0.9 * CANONICAL_BODY_LENGTH, n_nds)
        pos = np.zeros((n_frames, n_nds, 3))
        for t in range(n_frames):
            pos[t, :, 0] = x
            pos[t, :, 1] = 0.0
            pos[t, :, 2] = 0.0
        return pos

    def test_output_shapes(self):
        frame_pos = self._make_straight_worm(n_nds=10, n_frames=5)
        result = deformable_register(frame_pos)
        atlas = load_atlas()
        n_neurons = len(atlas["names"])
        assert isinstance(result, DeformableRegistrationResult)
        assert result.neuron_positions.shape == (5, n_neurons, 3)
        assert result.covariance.shape == (5, n_neurons, 3, 3)
        assert len(result.centerlines) == 5
        assert result.frame_indices.shape == (5,)

    def test_consistent_across_identical_frames(self):
        frame_pos = self._make_straight_worm(n_nds=10, n_frames=3)
        result = deformable_register(frame_pos)
        # All frames identical → neuron positions should be the same
        np.testing.assert_allclose(
            result.neuron_positions[0],
            result.neuron_positions[1],
            atol=1e-10,
        )

    def test_custom_atlas(self):
        frame_pos = self._make_straight_worm(n_nds=8, n_frames=2)
        atlas = {
            "names": ["A", "B", "C"],
            "fractional_positions": np.array([
                [0.3, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.7, 0.0, 0.0],
            ]),
            "types": ["excitatory"] * 3,
        }
        result = deformable_register(frame_pos, atlas)
        assert result.neuron_positions.shape == (2, 3, 3)
        assert result.neuron_names == ["A", "B", "C"]

    def test_interpolates_missing_frames(self):
        """Frames with too few NDs should be interpolated."""
        frame_pos = self._make_straight_worm(n_nds=6, n_frames=5)
        # Make frame 2 have only 2 valid NDs (below min_nds_per_frame=4)
        frame_pos[2, 2:, :] = np.nan
        result = deformable_register(frame_pos, min_nds_per_frame=4)
        # Frame 2 should still have valid positions (interpolated)
        assert not np.any(np.isnan(result.neuron_positions[2]))

    def test_moving_worm(self):
        """Worm with lateral deformation produces different positions."""
        n_nds = 10
        n_frames = 3
        x = np.linspace(0.1 * CANONICAL_BODY_LENGTH,
                         0.9 * CANONICAL_BODY_LENGTH, n_nds)
        frame_pos = np.zeros((n_frames, n_nds, 3))
        for t in range(n_frames):
            frame_pos[t, :, 0] = x
            frame_pos[t, :, 1] = 5e-6 * np.sin(2 * np.pi * x / CANONICAL_BODY_LENGTH + t)
        result = deformable_register(frame_pos)
        # Positions differ between frames
        diff = np.linalg.norm(
            result.neuron_positions[0] - result.neuron_positions[1], axis=1
        )
        assert np.max(diff) > 0


# ---------------------------------------------------------------------------
# smooth_neuron_trajectories
# ---------------------------------------------------------------------------


class TestSmoothNeuronTrajectories:
    def test_constant_input(self):
        pos = np.ones((10, 5, 3)) * 42.0
        smoothed = smooth_neuron_trajectories(pos, window=3)
        np.testing.assert_allclose(smoothed, pos)

    def test_reduces_noise(self):
        rng = np.random.default_rng(0)
        n_frames, n_neurons = 50, 3
        # Noisy sine wave
        clean = np.sin(np.linspace(0, 4 * np.pi, n_frames))[:, None, None]
        clean = np.broadcast_to(clean, (n_frames, n_neurons, 3)).copy()
        noisy = clean + rng.normal(0, 0.3, clean.shape)
        smoothed = smooth_neuron_trajectories(noisy, window=5)
        # Smoothed should be closer to clean than noisy
        err_noisy = np.mean((noisy - clean) ** 2)
        err_smooth = np.mean((smoothed - clean) ** 2)
        assert err_smooth < err_noisy

    def test_window_1_identity(self):
        rng = np.random.default_rng(0)
        pos = rng.normal(size=(10, 5, 3))
        smoothed = smooth_neuron_trajectories(pos, window=1)
        np.testing.assert_allclose(smoothed, pos)


# ---------------------------------------------------------------------------
# generate_deformable_experiment
# ---------------------------------------------------------------------------


class TestGenerateDeformableExperiment:
    def test_output_shapes(self):
        exp = generate_deformable_experiment(
            n_nds=6, n_neurons=3, n_frames=10, seed=42,
        )
        assert exp["frame_nd_positions"].shape == (10, 6, 3)
        assert exp["frame_neuron_positions"].shape == (10, 3, 3)
        assert exp["current_waveforms"].shape == (3, 10)
        assert exp["field_timeseries"].shape == (6, 10)
        assert exp["time"].shape == (10,)
        assert len(exp["atlas"]["names"]) == 3

    def test_deformation_varies(self):
        exp = generate_deformable_experiment(
            n_nds=8, n_neurons=4, n_frames=10, amplitude=10e-6, seed=0,
        )
        pos = exp["frame_nd_positions"]
        y_range = pos[:, :, 1].max() - pos[:, :, 1].min()
        assert y_range > 1e-6  # should show deformation
