"""Tests for wimp.registration."""

import numpy as np
import pytest

from wimp.registration import (
    load_atlas,
    fit_centerline,
    arc_length_parameterize,
    register_atlas_to_image,
    estimate_body_pose,
    uncertainty_ellipses,
    assign_nds_to_neurons,
    NEURON_CLASSES,
    CenterlineResult,
    RegistrationResult,
)
from wimp.constants import CANONICAL_BODY_LENGTH


@pytest.fixture
def worm_points():
    """Points along a slightly curved worm body."""
    n = 30
    x = np.linspace(0, 1e-3, n)
    y = 20e-6 * np.sin(2 * np.pi * x / 1e-3)
    z = np.zeros(n)
    return np.column_stack([x, y, z])


@pytest.fixture
def centerline(worm_points):
    return fit_centerline(worm_points, n_control_points=15)


class TestNeuronClasses:
    def test_known_classes(self):
        assert "DB" in NEURON_CLASSES
        assert "VB" in NEURON_CLASSES
        assert "DD" in NEURON_CLASSES
        assert "VD" in NEURON_CLASSES


class TestLoadAtlas:
    def test_builtin_atlas(self):
        atlas = load_atlas()
        assert "names" in atlas
        assert "fractional_positions" in atlas
        assert "types" in atlas
        assert len(atlas["names"]) > 0
        assert len(atlas["names"]) == len(atlas["fractional_positions"])

    def test_builtin_has_motor_neurons(self):
        atlas = load_atlas()
        names = atlas["names"]
        assert any(n.startswith("DB") for n in names)
        assert any(n.startswith("VB") for n in names)

    def test_fractional_positions_in_range(self):
        atlas = load_atlas()
        x_fracs = atlas["fractional_positions"][:, 0]
        assert np.all(x_fracs >= 0)
        assert np.all(x_fracs <= 1)


class TestFitCenterline:
    def test_returns_centerline_result(self, worm_points):
        result = fit_centerline(worm_points)
        assert isinstance(result, CenterlineResult)

    def test_body_length_reasonable(self, worm_points):
        result = fit_centerline(worm_points)
        # Should be close to 1mm (±20%)
        assert 0.8e-3 < result.body_length < 1.2e-3

    def test_control_points_count(self, worm_points):
        n_ctrl = 15
        result = fit_centerline(worm_points, n_control_points=n_ctrl)
        assert result.control_points.shape == (n_ctrl, 3)

    def test_curve_shape(self, worm_points):
        result = fit_centerline(worm_points)
        assert result.curve.ndim == 2
        assert result.curve.shape[1] == 3
        assert len(result.curve) >= 1000

    def test_arc_length_monotonic(self, worm_points):
        result = fit_centerline(worm_points)
        assert np.all(np.diff(result.arc_length) >= 0)

    def test_tck_stored(self, worm_points):
        result = fit_centerline(worm_points)
        assert result.tck is not None


class TestArcLengthParameterize:
    def test_output_shape(self, centerline):
        n_pts = 200
        s, pts = arc_length_parameterize(centerline, n_points=n_pts)
        assert s.shape == (n_pts,)
        assert pts.shape == (n_pts, 3)

    def test_s_starts_at_zero(self, centerline):
        s, _ = arc_length_parameterize(centerline)
        assert s[0] == pytest.approx(0.0)

    def test_s_ends_at_body_length(self, centerline):
        s, _ = arc_length_parameterize(centerline)
        assert s[-1] == pytest.approx(centerline.body_length, rel=0.01)


class TestRegisterAtlasToImage:
    def test_basic_registration(self, centerline):
        atlas = load_atlas()
        result = register_atlas_to_image(atlas, centerline)
        assert isinstance(result, RegistrationResult)
        n = len(atlas["names"])
        assert result.neuron_positions_image.shape == (n, 3)
        assert result.covariance.shape == (n, 3, 3)
        assert result.neuron_names == atlas["names"]

    def test_covariance_positive_diagonal(self, centerline):
        atlas = load_atlas()
        result = register_atlas_to_image(atlas, centerline)
        for k in range(len(atlas["names"])):
            diag = np.diag(result.covariance[k])
            assert np.all(diag > 0)


class TestEstimateBodyPose:
    def test_from_nd_positions(self, nd_positions):
        result = estimate_body_pose(nd_positions)
        assert isinstance(result, CenterlineResult)
        assert result.body_length > 0


class TestUncertaintyEllipses:
    def test_accessor(self, centerline):
        atlas = load_atlas()
        reg = register_atlas_to_image(atlas, centerline)
        cov = uncertainty_ellipses(reg)
        assert cov is reg.covariance


class TestAssignNDsToNeurons:
    def test_basic_assignment(self, nd_positions, neuron_positions):
        assignments = assign_nds_to_neurons(nd_positions, neuron_positions)
        assert len(assignments) == len(neuron_positions)
        for a in assignments:
            assert isinstance(a, list)
            if len(a) > 0:
                assert 0 <= a[0] < len(nd_positions)

    def test_max_distance_filter(self, nd_positions, neuron_positions):
        # Very small max_distance should exclude everything
        assignments = assign_nds_to_neurons(
            nd_positions, neuron_positions, max_distance=1e-12
        )
        for a in assignments:
            assert len(a) == 0
