"""Tests for wimp.stability."""

import numpy as np
import pytest

from wimp.stability import (
    drift_correct,
    allan_variance,
    compressed_sensing_reconstruct,
)


class TestDriftCorrect:
    def test_polynomial_removes_quadratic(self):
        t = np.arange(1000, dtype=float)
        drift = 0.001 * t ** 2
        signal = np.sin(0.1 * t) + drift
        result = drift_correct(signal, method="polynomial", degree=2)
        assert result["corrected"].shape == signal.shape
        assert result["method"] == "polynomial"
        # Corrected should be closer to pure sine than original
        pure = np.sin(0.1 * t)
        assert np.std(result["corrected"] - pure) < np.std(signal - pure)

    def test_linear_removes_ramp(self):
        t = np.arange(500, dtype=float)
        signal = 3.0 + 0.01 * t + np.random.default_rng(0).normal(0, 0.1, 500)
        result = drift_correct(signal, method="linear")
        assert result["corrected"].shape == signal.shape
        # Mean of corrected should be near zero
        assert abs(np.mean(result["corrected"])) < 0.5

    def test_highpass_method(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(size=500)
        result = drift_correct(signal, method="highpass", cutoff=1.0, fs=100.0)
        assert result["corrected"].shape == signal.shape

    def test_highpass_requires_cutoff(self):
        signal = np.ones(100)
        with pytest.raises(ValueError, match="cutoff"):
            drift_correct(signal, method="highpass")

    def test_unknown_method_raises(self):
        signal = np.ones(100)
        with pytest.raises(ValueError, match="Unknown"):
            drift_correct(signal, method="nonexistent")

    def test_2d_input(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(size=(3, 200))
        signal += np.linspace(0, 1, 200)[np.newaxis, :]  # add linear drift
        result = drift_correct(signal, method="polynomial", degree=1)
        assert result["corrected"].shape == (3, 200)
        assert result["drift"].shape == (3, 200)

    def test_1d_roundtrip_shape(self):
        signal = np.random.default_rng(0).normal(size=100)
        result = drift_correct(signal)
        assert result["corrected"].ndim == 1


class TestAllanVariance:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=1000)
        result = allan_variance(ts, fs=100.0)
        assert "taus" in result
        assert "avar" in result
        assert "adev" in result
        assert "adev_err" in result

    def test_white_noise_scaling(self):
        rng = np.random.default_rng(42)
        ts = rng.normal(0, 1.0, size=10000)
        result = allan_variance(ts, fs=1000.0)
        # For white noise, ADEV ~ sigma / sqrt(tau * fs)
        # Check that ADEV decreases with tau
        if len(result["adev"]) > 2:
            assert result["adev"][-1] < result["adev"][0]

    def test_constant_signal_zero_variance(self):
        ts = np.ones(500) * 5.0
        result = allan_variance(ts, fs=100.0)
        np.testing.assert_allclose(result["avar"], 0.0, atol=1e-15)

    def test_custom_taus(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=1000)
        taus = np.array([0.01, 0.05, 0.1])
        result = allan_variance(ts, fs=100.0, taus=taus)
        assert len(result["taus"]) <= len(taus)

    def test_non_overlapping(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=1000)
        result = allan_variance(ts, fs=100.0, overlap=False)
        assert len(result["adev"]) > 0


class TestCompressedSensingReconstruct:
    def test_output_keys(self):
        tau_full = np.linspace(0, 5e-6, 100)
        tau_meas = tau_full[::5]  # 20% sampling
        signal = np.cos(2 * np.pi * 1e6 * tau_meas)
        result = compressed_sensing_reconstruct(signal, tau_meas, tau_full)
        assert "reconstructed" in result
        assert "tau_full" in result
        assert "residual" in result
        assert "coefficients" in result

    def test_output_shape(self):
        tau_full = np.linspace(0, 5e-6, 50)
        tau_meas = tau_full[::3]
        signal = np.cos(2 * np.pi * 1e6 * tau_meas)
        result = compressed_sensing_reconstruct(signal, tau_meas, tau_full)
        assert result["reconstructed"].shape == (50,)
        assert result["tau_full"].shape == (50,)

    def test_overdetermined_low_residual(self):
        tau_full = np.linspace(0, 1, 30)
        # Use almost all points (26 out of 30)
        tau_meas = tau_full[::1][:26]
        signal = np.cos(2 * np.pi * 3 * tau_meas)
        result = compressed_sensing_reconstruct(
            signal, tau_meas, tau_full, n_components=15,
        )
        assert np.max(np.abs(result["residual"])) < 1.0
