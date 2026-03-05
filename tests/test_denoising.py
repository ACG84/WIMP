"""Tests for wimp.denoising."""

import numpy as np
import pytest

from wimp.denoising import (
    denoise,
    denoise_savgol,
    denoise_butterworth,
    denoise_moving_average,
    denoise_wavelet,
)


class TestDenoiseSavgol:
    def test_preserves_shape_1d(self):
        rng = np.random.default_rng(0)
        signal = np.sin(np.linspace(0, 4 * np.pi, 100)) + rng.normal(0, 0.1, 100)
        result = denoise_savgol(signal)
        assert result.shape == signal.shape

    def test_preserves_shape_2d(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(size=(3, 100))
        result = denoise_savgol(signal)
        assert result.shape == signal.shape

    def test_reduces_noise(self):
        rng = np.random.default_rng(42)
        clean = np.sin(np.linspace(0, 4 * np.pi, 200))
        noisy = clean + rng.normal(0, 0.3, 200)
        denoised = denoise_savgol(noisy, window_length=11, polyorder=3)
        # Denoised should be closer to the clean signal than the noisy one
        assert np.std(denoised - clean) < np.std(noisy - clean)

    def test_short_signal(self):
        signal = np.array([1.0, 2.0, 3.0])
        result = denoise_savgol(signal, window_length=11)
        assert result.shape == signal.shape

    def test_custom_polyorder(self):
        signal = np.random.default_rng(0).normal(size=50)
        result = denoise_savgol(signal, window_length=7, polyorder=2)
        assert result.shape == signal.shape


class TestDenoiseButterworth:
    def test_preserves_shape(self):
        signal = np.random.default_rng(0).normal(size=200)
        result = denoise_butterworth(signal, cutoff=50.0, fs=1000.0)
        assert result.shape == signal.shape

    def test_removes_high_freq(self):
        fs = 1000.0
        t = np.arange(0, 1, 1 / fs)
        low_freq = np.sin(2 * np.pi * 10 * t)
        high_freq = 0.5 * np.sin(2 * np.pi * 400 * t)
        signal = low_freq + high_freq
        filtered = denoise_butterworth(signal, cutoff=50.0, fs=fs)
        # Filtered should be close to the low-frequency component
        assert np.std(filtered - low_freq) < np.std(signal - low_freq)

    def test_cutoff_above_nyquist_returns_copy(self):
        signal = np.random.default_rng(0).normal(size=100)
        result = denoise_butterworth(signal, cutoff=600.0, fs=1000.0)
        np.testing.assert_array_equal(result, signal)

    def test_2d_input(self):
        signal = np.random.default_rng(0).normal(size=(3, 200))
        result = denoise_butterworth(signal, cutoff=50.0, fs=1000.0)
        assert result.shape == signal.shape


class TestDenoiseMovingAverage:
    def test_preserves_shape(self):
        signal = np.random.default_rng(0).normal(size=100)
        result = denoise_moving_average(signal, window_size=5)
        assert result.shape == signal.shape

    def test_reduces_noise(self):
        rng = np.random.default_rng(42)
        clean = np.sin(np.linspace(0, 4 * np.pi, 200))
        noisy = clean + rng.normal(0, 0.3, 200)
        denoised = denoise_moving_average(noisy, window_size=7)
        assert np.std(denoised - clean) < np.std(noisy - clean)

    def test_2d_input(self):
        signal = np.random.default_rng(0).normal(size=(2, 50))
        result = denoise_moving_average(signal, window_size=3)
        assert result.shape == signal.shape

    def test_window_size_clamped(self):
        signal = np.array([1.0, 2.0, 3.0])
        result = denoise_moving_average(signal, window_size=100)
        assert result.shape == signal.shape


class TestDenoiseWavelet:
    def test_preserves_shape(self):
        pywt = pytest.importorskip("pywt")
        signal = np.random.default_rng(0).normal(size=128)
        result = denoise_wavelet(signal)
        assert result.shape == signal.shape

    def test_reduces_noise(self):
        pywt = pytest.importorskip("pywt")
        rng = np.random.default_rng(42)
        clean = np.sin(np.linspace(0, 4 * np.pi, 256))
        noisy = clean + rng.normal(0, 0.5, 256)
        denoised = denoise_wavelet(noisy)
        assert np.std(denoised - clean) < np.std(noisy - clean)

    def test_2d_input(self):
        pywt = pytest.importorskip("pywt")
        signal = np.random.default_rng(0).normal(size=(3, 128))
        result = denoise_wavelet(signal)
        assert result.shape == signal.shape

    def test_hard_threshold(self):
        pywt = pytest.importorskip("pywt")
        signal = np.random.default_rng(0).normal(size=128)
        result = denoise_wavelet(signal, threshold_mode="hard")
        assert result.shape == signal.shape


class TestDenoiseDispatcher:
    def test_routes_savgol(self):
        signal = np.random.default_rng(0).normal(size=100)
        result = denoise(signal, method="savgol")
        assert result.shape == signal.shape

    def test_routes_butterworth(self):
        signal = np.random.default_rng(0).normal(size=100)
        result = denoise(signal, method="butterworth", cutoff=50.0, fs=1000.0)
        assert result.shape == signal.shape

    def test_routes_moving_average(self):
        signal = np.random.default_rng(0).normal(size=100)
        result = denoise(signal, method="moving_average")
        assert result.shape == signal.shape

    def test_routes_wavelet(self):
        pywt = pytest.importorskip("pywt")
        signal = np.random.default_rng(0).normal(size=128)
        result = denoise(signal, method="wavelet")
        assert result.shape == signal.shape

    def test_unknown_method_raises(self):
        signal = np.random.default_rng(0).normal(size=100)
        with pytest.raises(ValueError, match="Unknown denoising method"):
            denoise(signal, method="nonexistent")
