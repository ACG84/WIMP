"""Tests for wimp.localization."""

import numpy as np
import pytest

from wimp.localization import is_available


class TestIsAvailable:
    def test_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)


class TestWithoutTrackpy:
    """Tests that work regardless of whether trackpy is installed."""

    def test_detect_nds_requires_trackpy(self):
        if is_available():
            pytest.skip("trackpy is installed; cannot test missing-dep path")
        from wimp.localization import detect_nds
        with pytest.raises(RuntimeError, match="trackpy"):
            detect_nds(np.zeros((100, 100)))

    def test_track_nds_requires_trackpy(self):
        if is_available():
            pytest.skip("trackpy is installed; cannot test missing-dep path")
        from wimp.localization import track_nds
        with pytest.raises(RuntimeError, match="trackpy"):
            track_nds(np.zeros((10, 100, 100)))


class TestWithTrackpy:
    """Tests that only run if trackpy is available."""

    @pytest.fixture(autouse=True)
    def _require_trackpy(self):
        if not is_available():
            pytest.skip("trackpy not installed")

    def test_detect_nds_on_synthetic_image(self):
        from wimp.localization import detect_nds
        # Create synthetic image with a bright spot
        rng = np.random.default_rng(42)
        img = rng.normal(100, 10, size=(200, 200)).astype(float)
        # Add a bright Gaussian-like spot
        y, x = np.mgrid[0:200, 0:200]
        img += 500 * np.exp(-((x - 100) ** 2 + (y - 100) ** 2) / (2 * 5**2))

        features = detect_nds(img, diameter=11, minmass=1000)
        assert len(features) >= 1

    def test_track_nds_on_stack(self):
        from wimp.localization import track_nds
        rng = np.random.default_rng(42)
        n_frames = 5
        stack = rng.normal(100, 10, size=(n_frames, 200, 200)).astype(float)
        y, x = np.mgrid[0:200, 0:200]
        for i in range(n_frames):
            stack[i] += 500 * np.exp(-((x - 100) ** 2 + (y - 100 - i) ** 2) / (2 * 5**2))

        tracks = track_nds(stack, diameter=11, search_range=5, minmass=1000)
        assert len(tracks) >= n_frames

    def test_extract_nd_signals(self):
        from wimp.localization import track_nds, extract_nd_signals
        rng = np.random.default_rng(42)
        n_frames = 5
        stack = rng.normal(100, 10, size=(n_frames, 200, 200)).astype(float)
        y, x = np.mgrid[0:200, 0:200]
        for i in range(n_frames):
            stack[i] += 500 * np.exp(-((x - 100) ** 2 + (y - 100) ** 2) / (2 * 5**2))

        tracks = track_nds(stack, diameter=11, search_range=5, minmass=1000)
        result = extract_nd_signals(stack, tracks, radius=5)
        assert "signals" in result
        assert "particle_ids" in result
        assert "positions" in result
