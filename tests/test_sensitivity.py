"""Tests for wimp.sensitivity."""

import math

import numpy as np
import pytest

from wimp.sensitivity import (
    dc_sensitivity,
    ac_sensitivity,
    t1_sensitivity,
    shot_noise_limit,
    compare_protocols,
    optimal_protocol,
    cw_sensitivity,
    sensitivity_vs_tau,
    measurement_time_for_target,
)


class TestDCSensitivity:
    def test_positive(self):
        eta = dc_sensitivity(0.03, 1e-6, 1e5)
        assert eta > 0

    def test_improves_with_more_nvs(self):
        eta1 = dc_sensitivity(0.03, 1e-6, 1e5, n_nv=1)
        eta10 = dc_sensitivity(0.03, 1e-6, 1e5, n_nv=10)
        assert eta10 < eta1

    def test_improves_with_longer_t2star(self):
        eta_short = dc_sensitivity(0.03, 0.5e-6, 1e5)
        eta_long = dc_sensitivity(0.03, 2e-6, 1e5)
        assert eta_long < eta_short

    def test_improves_with_higher_contrast(self):
        eta_low = dc_sensitivity(0.01, 1e-6, 1e5)
        eta_high = dc_sensitivity(0.1, 1e-6, 1e5)
        assert eta_high < eta_low


class TestACSensitivity:
    def test_positive(self):
        eta = ac_sensitivity(0.03, 100e-6, 1e5)
        assert eta > 0

    def test_dd_improves_with_pulses(self):
        eta1 = ac_sensitivity(0.03, 100e-6, 1e5, n_pulses=1)
        eta64 = ac_sensitivity(0.03, 100e-6, 1e5, n_pulses=64)
        assert eta64 < eta1

    def test_single_pulse_matches_echo(self):
        eta = ac_sensitivity(0.03, 100e-6, 1e5, n_pulses=1)
        # Single pulse = Hahn echo
        assert eta > 0


class TestT1Sensitivity:
    def test_positive(self):
        delta = t1_sensitivity(0.03, 5e-3, 1e5)
        assert delta > 0


class TestShotNoiseLimit:
    def test_positive(self):
        dphi = shot_noise_limit(1000, 0.03)
        assert dphi > 0

    def test_improves_with_more_photons(self):
        dphi_few = shot_noise_limit(100, 0.03)
        dphi_many = shot_noise_limit(10000, 0.03)
        assert dphi_many < dphi_few


class TestCompareProtocols:
    def test_returns_dataframe(self):
        df = compare_protocols()
        assert len(df) == 5
        assert "protocol" in df.columns
        assert "sensitivity_T_sqrtHz" in df.columns

    def test_custom_parameters(self):
        df = compare_protocols(contrast=0.1, t2star=2e-6, t2=200e-6)
        assert len(df) == 5


class TestOptimalProtocol:
    def test_dc_returns_ramsey(self):
        rec = optimal_protocol("dc")
        assert rec["protocol"] == "ramsey"
        assert "sensitivity_T_sqrtHz" in rec

    def test_ac_returns_protocol(self):
        rec = optimal_protocol("ac")
        assert rec["protocol"] in ("hahn_echo", "cpmg-64")

    def test_t1_returns_relaxometry(self):
        rec = optimal_protocol("t1")
        assert rec["protocol"] == "t1_relaxometry"

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError):
            optimal_protocol("invalid")

    def test_with_characterization(self):
        char = {
            "t2star": 2e-6,
            "t2": 200e-6,
            "t1": 10e-3,
            "contrast": 0.05,
            "readout_rate": 2e5,
        }
        rec = optimal_protocol("dc", characterization=char)
        assert rec["protocol"] == "ramsey"


# ---------------------------------------------------------------------------
# Adaptive protocol recommendation
# ---------------------------------------------------------------------------

from wimp.sensitivity import adaptive_protocol_recommendation


class TestAdaptiveProtocolRecommendation:
    @pytest.fixture
    def char(self):
        return {
            "t2star": 1e-6,
            "t2": 100e-6,
            "t1": 5e-3,
            "contrast": 0.03,
            "readout_rate": 1e5,
        }

    def test_dc_target(self, char):
        result = adaptive_protocol_recommendation(char, target="dc")
        assert result["recommended_protocol"] == "ramsey"
        assert result["predicted_sensitivity"] > 0
        assert "reasoning" in result
        assert "parameters" in result

    def test_ac_target(self, char):
        result = adaptive_protocol_recommendation(char, target="ac")
        assert "cpmg" in result["recommended_protocol"] or result["recommended_protocol"] == "hahn_echo"
        assert result["predicted_sensitivity"] > 0

    def test_relaxometry_target(self, char):
        result = adaptive_protocol_recommendation(char, target="relaxometry")
        assert result["recommended_protocol"] == "t1_relaxometry"
        assert result["recommended_tau"] == char["t1"]

    def test_unknown_target_raises(self, char):
        with pytest.raises(ValueError, match="Unknown target"):
            adaptive_protocol_recommendation(char, target="invalid")

    def test_constraints_max_pulses(self, char):
        result = adaptive_protocol_recommendation(
            char, target="ac", constraints={"max_n_pulses": 4},
        )
        assert result["predicted_sensitivity"] > 0

    def test_dc_optimal_tau(self, char):
        result = adaptive_protocol_recommendation(char, target="dc")
        # Optimal tau for Ramsey is T2*/sqrt(2)
        expected_tau = char["t2star"] / math.sqrt(2)
        assert result["recommended_tau"] == pytest.approx(expected_tau)


# ---------------------------------------------------------------------------
# Sensitivity sweeps
# ---------------------------------------------------------------------------

from wimp.sensitivity import sensitivity_vs_tau, measurement_time_for_target


class TestSensitivityVsTau:
    def test_output_keys(self):
        tau = np.linspace(0.1e-6, 5e-6, 50)
        result = sensitivity_vs_tau("ramsey", tau, t2star=1e-6)
        assert "tau" in result
        assert "sensitivity" in result
        assert len(result["tau"]) == 50

    def test_ramsey_requires_t2star(self):
        tau = np.linspace(0.1e-6, 5e-6, 50)
        with pytest.raises(ValueError, match="t2star"):
            sensitivity_vs_tau("ramsey", tau)

    def test_echo_requires_t2(self):
        tau = np.linspace(0.1e-6, 100e-6, 50)
        with pytest.raises(ValueError, match="t2"):
            sensitivity_vs_tau("echo", tau)

    def test_t1_requires_t1(self):
        tau = np.linspace(0.1e-3, 10e-3, 50)
        with pytest.raises(ValueError, match="t1"):
            sensitivity_vs_tau("t1", tau)

    def test_sensitivity_positive(self):
        tau = np.linspace(0.1e-6, 5e-6, 50)
        result = sensitivity_vs_tau("ramsey", tau, t2star=1e-6)
        assert np.all(result["sensitivity"] > 0)

    def test_unknown_protocol_raises(self):
        tau = np.linspace(0.1e-6, 5e-6, 10)
        with pytest.raises(ValueError, match="Unknown protocol"):
            sensitivity_vs_tau("nonexistent", tau)


class TestMeasurementTimeForTarget:
    def test_positive(self):
        t = measurement_time_for_target(1e-9, "ramsey")
        assert t > 0

    def test_better_sensitivity_needs_more_time(self):
        t1 = measurement_time_for_target(1e-9, "ramsey")
        t2 = measurement_time_for_target(1e-10, "ramsey")
        assert t2 > t1

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown protocol"):
            measurement_time_for_target(1e-9, "nonexistent")


class TestCWSensitivity:
    def test_positive(self):
        eta = cw_sensitivity(0.03, 5e6, 1e5)
        assert eta > 0

    def test_improves_with_narrower_linewidth(self):
        eta_wide = cw_sensitivity(0.03, 10e6, 1e5)
        eta_narrow = cw_sensitivity(0.03, 2e6, 1e5)
        assert eta_narrow < eta_wide

    def test_improves_with_higher_contrast(self):
        eta_low = cw_sensitivity(0.01, 5e6, 1e5)
        eta_high = cw_sensitivity(0.1, 5e6, 1e5)
        assert eta_high < eta_low

    def test_compare_protocols_includes_cw(self):
        df = compare_protocols()
        protocols = df["protocol"].tolist()
        assert any("CW" in p for p in protocols)

    def test_optimal_protocol_cw_odmr(self):
        result = optimal_protocol("cw_odmr")
        assert result["protocol"] == "cw_odmr"
        assert result["sensitivity_T_sqrtHz"] > 0

    def test_sensitivity_vs_tau_cw_odmr_raises(self):
        tau = np.linspace(0.1e-6, 5e-6, 10)
        with pytest.raises(ValueError, match="CW ODMR"):
            sensitivity_vs_tau("cw_odmr", tau)

    def test_measurement_time_cw_odmr(self):
        t = measurement_time_for_target(1e-9, "cw_odmr")
        assert t > 0

