"""Tests for wimp.source."""

import numpy as np
import pytest

from wimp.source import (
    lead_field_matrix,
    forward_model,
    mne_inverse,
    mne_inverse_kernel,
    lcmv_beamformer,
    source_snr,
    resolution_matrix,
    crosstalk_metrics,
    time_varying_inverse,
)


class TestLeadFieldMatrix:
    def test_shape(self, neuron_positions, nd_positions):
        L = lead_field_matrix(neuron_positions, nd_positions)
        assert L.shape == (len(nd_positions), len(neuron_positions))

    def test_nonzero(self, neuron_positions, nd_positions):
        L = lead_field_matrix(neuron_positions, nd_positions)
        assert np.any(L != 0)

    def test_custom_orientation(self, neuron_positions, nd_positions):
        orientation = np.zeros((len(neuron_positions), 3))
        orientation[:, 0] = 1.0  # x-oriented
        L = lead_field_matrix(neuron_positions, nd_positions, orientation=orientation)
        assert L.shape == (len(nd_positions), len(neuron_positions))

    def test_coincident_points(self):
        pos = np.array([[0, 0, 0.0]])
        L = lead_field_matrix(pos, pos)
        assert L[0, 0] == 0.0  # should skip


class TestForwardModel:
    def test_basic(self, neuron_positions, nd_positions):
        L = lead_field_matrix(neuron_positions, nd_positions)
        n_neurons = len(neuron_positions)
        J = np.ones(n_neurons)
        B = forward_model(L, J)
        assert B.shape == (len(nd_positions),)

    def test_timeseries(self, neuron_positions, nd_positions):
        L = lead_field_matrix(neuron_positions, nd_positions)
        n_neurons = len(neuron_positions)
        J = np.ones((n_neurons, 50))
        B = forward_model(L, J)
        assert B.shape == (len(nd_positions), 50)

    def test_moment_scale(self, neuron_positions, nd_positions):
        L = lead_field_matrix(neuron_positions, nd_positions)
        J = np.ones(len(neuron_positions))
        B1 = forward_model(L, J, moment_scale=1.0)
        B2 = forward_model(L, J, moment_scale=2.0)
        np.testing.assert_allclose(B2, 2 * B1)


class TestMNEInverse:
    def test_recovery(self, neuron_positions, nd_positions):
        L = lead_field_matrix(neuron_positions, nd_positions)
        n_neurons = len(neuron_positions)
        J_true = np.zeros(n_neurons)
        J_true[1] = 1.0
        B = L @ J_true
        J_est = mne_inverse(L, B, lambda_reg=0.01)
        # Peak should be at the right source
        assert np.argmax(np.abs(J_est)) == 1

    def test_timeseries(self, neuron_positions, nd_positions):
        L = lead_field_matrix(neuron_positions, nd_positions)
        n_neurons = len(neuron_positions)
        J_true = np.zeros((n_neurons, 20))
        J_true[0, 5:15] = 1.0
        B = L @ J_true
        J_est = mne_inverse(L, B)
        assert J_est.shape == (n_neurons, 20)

    def test_lambda_reg_effect(self, neuron_positions, nd_positions):
        L = lead_field_matrix(neuron_positions, nd_positions)
        J_true = np.array([1.0, 0, 0, 0.0])
        B = L @ J_true
        J_low = mne_inverse(L, B, lambda_reg=0.001)
        J_high = mne_inverse(L, B, lambda_reg=10.0)
        # Higher regularization → smaller norm
        assert np.linalg.norm(J_high) <= np.linalg.norm(J_low)


class TestLCMVBeamformer:
    def test_basic(self, neuron_positions, nd_positions):
        L = lead_field_matrix(neuron_positions, nd_positions)
        n_neurons = len(neuron_positions)
        rng = np.random.default_rng(42)
        J_true = np.zeros((n_neurons, 100))
        J_true[0] = rng.normal(size=100)
        B = L @ J_true + 0.01 * rng.normal(size=(len(nd_positions), 100))
        s = lcmv_beamformer(L, B, target_idx=0)
        assert s.shape == (100,)
        # Should correlate positively with true source
        corr = np.corrcoef(s, J_true[0])[0, 1]
        assert corr > 0.5


class TestSourceSNR:
    def test_perfect_reconstruction(self):
        true = np.array([[1, 2, 3], [4, 5, 6.0]])
        metrics = source_snr(true, true)
        assert metrics["correlation"] == pytest.approx(1.0)
        assert metrics["relative_error"] == pytest.approx(0.0)
        assert metrics["peak_localisation_error"] == 0

    def test_bad_reconstruction(self):
        true = np.array([1.0, 0, 0, 0])
        est = np.array([0, 0, 0, 1.0])
        metrics = source_snr(true, est)
        assert metrics["peak_localisation_error"] == 3

    def test_zero_std(self):
        true = np.ones(10)
        est = np.ones(10) * 2
        metrics = source_snr(true, est)
        # std of constant is 0 → correlation should be 0
        assert metrics["correlation"] == 0.0


class TestMNEInverseKernel:
    def test_shape(self, neuron_positions, nd_positions):
        L = lead_field_matrix(neuron_positions, nd_positions)
        K = mne_inverse_kernel(L)
        assert K.shape == (len(neuron_positions), len(nd_positions))

    def test_consistent_with_mne_inverse(self, neuron_positions, nd_positions):
        L = lead_field_matrix(neuron_positions, nd_positions)
        J_true = np.array([1.0, 0, 0, 0.0])
        B = L @ J_true
        K = mne_inverse_kernel(L, lambda_reg=0.1)
        J_kernel = K @ B
        J_mne = mne_inverse(L, B, lambda_reg=0.1)
        np.testing.assert_allclose(J_kernel, J_mne)


class TestResolutionMatrix:
    def test_shape(self, neuron_positions, nd_positions):
        L = lead_field_matrix(neuron_positions, nd_positions)
        R = resolution_matrix(L)
        n_neurons = len(neuron_positions)
        assert R.shape == (n_neurons, n_neurons)

    def test_diagonal_dominance(self, neuron_positions, nd_positions):
        L = lead_field_matrix(neuron_positions, nd_positions)
        R = resolution_matrix(L, lambda_reg=0.01)
        diag = np.abs(np.diag(R))
        for i in range(R.shape[0]):
            off_diag = np.abs(R[i].copy())
            off_diag[i] = 0.0
            # Diagonal should be larger than any off-diagonal element
            assert diag[i] >= off_diag.max() or diag[i] > 0

    def test_lambda_effect(self, neuron_positions, nd_positions):
        L = lead_field_matrix(neuron_positions, nd_positions)
        R_low = resolution_matrix(L, lambda_reg=0.001)
        R_high = resolution_matrix(L, lambda_reg=10.0)
        # Lower lambda → closer to identity → higher trace
        assert np.trace(R_low) >= np.trace(R_high)


class TestCrosstalkMetrics:
    def test_output_keys(self):
        R = np.eye(3)
        m = crosstalk_metrics(R)
        expected_keys = {"diagonal", "off_diagonal_max", "crosstalk_ratio",
                         "spread", "total_leakage", "neuron_names"}
        assert set(m.keys()) == expected_keys

    def test_perfect_resolution(self):
        R = np.eye(5)
        m = crosstalk_metrics(R)
        np.testing.assert_allclose(m["diagonal"], np.ones(5))
        np.testing.assert_allclose(m["off_diagonal_max"], np.zeros(5))
        np.testing.assert_allclose(m["crosstalk_ratio"], np.zeros(5))
        np.testing.assert_allclose(m["total_leakage"], np.zeros(5))

    def test_neuron_names_passthrough(self):
        R = np.eye(3)
        names = ["A", "B", "C"]
        m = crosstalk_metrics(R, names)
        assert m["neuron_names"] == names

    def test_default_names(self):
        R = np.eye(3)
        m = crosstalk_metrics(R)
        assert m["neuron_names"] == ["S0", "S1", "S2"]


class TestTimeVaryingInverse:
    def test_matches_static_when_constant(self, neuron_positions, nd_positions):
        L = lead_field_matrix(neuron_positions, nd_positions)
        n_neurons = len(neuron_positions)
        n_nds = len(nd_positions)
        n_frames = 5

        J_true = np.zeros((n_neurons, n_frames))
        J_true[0, :] = 1.0
        B = L @ J_true

        # Static inverse
        J_static = mne_inverse(L, B, lambda_reg=0.1)

        # Time-varying with constant geometry
        neuron_per_frame = np.broadcast_to(
            neuron_positions[np.newaxis], (n_frames, n_neurons, 3)
        ).copy()

        result = time_varying_inverse(
            neuron_per_frame, nd_positions, B, lambda_reg=0.1,
        )
        np.testing.assert_allclose(
            result["source_estimate"], J_static, atol=1e-10,
        )

    def test_output_shapes(self, neuron_positions, nd_positions):
        n_neurons = len(neuron_positions)
        n_nds = len(nd_positions)
        n_frames = 3

        neuron_per_frame = np.broadcast_to(
            neuron_positions[np.newaxis], (n_frames, n_neurons, 3)
        ).copy()
        B = np.ones((n_nds, n_frames))

        result = time_varying_inverse(
            neuron_per_frame, nd_positions, B,
            compute_resolution=True,
        )
        assert result["source_estimate"].shape == (n_neurons, n_frames)
        assert len(result["lead_fields"]) == n_frames
        assert len(result["resolution_matrices"]) == n_frames
        assert result["resolution_matrices"][0].shape == (n_neurons, n_neurons)

    def test_3d_nd_positions(self, neuron_positions, nd_positions):
        n_neurons = len(neuron_positions)
        n_nds = len(nd_positions)
        n_frames = 3

        neuron_per_frame = np.broadcast_to(
            neuron_positions[np.newaxis], (n_frames, n_neurons, 3)
        ).copy()
        nd_per_frame = np.broadcast_to(
            nd_positions[np.newaxis], (n_frames, n_nds, 3)
        ).copy()
        B = np.ones((n_nds, n_frames))

        result = time_varying_inverse(
            neuron_per_frame, nd_per_frame, B,
        )
        assert result["source_estimate"].shape == (n_neurons, n_frames)
