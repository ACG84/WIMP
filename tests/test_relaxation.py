"""Tests for wimp.relaxation."""

import numpy as np
import pytest

from wimp.constants import GAMMA_NV
from wimp.relaxation import (
    ramsey_model,
    t2_decay_model,
    t1_recovery_model,
    biexponential_decay,
    goodness_of_fit,
    fit_ramsey,
    fit_t2_decay,
    fit_t1_recovery,
    fit_biexponential,
    extract_field_ramsey,
    extract_field_echo,
    _initial_frequency_guess,
)


class TestModels:
    def test_ramsey_model_at_zero(self):
        val = ramsey_model(np.array([0.0]), 1.0, 1e6, 1e-6, 0.0, 0.0)
        assert val[0] == pytest.approx(1.0)

    def test_ramsey_model_offset(self):
        val = ramsey_model(np.array([0.0]), 1.0, 1e6, 1e-6, 0.0, 0.5)
        assert val[0] == pytest.approx(1.5)

    def test_ramsey_model_decay(self):
        tau = np.array([0.0, 10e-6])
        vals = ramsey_model(tau, 1.0, 1e6, 1e-6, 0.0, 0.0)
        # At 10*T2*, signal should be very decayed
        assert abs(vals[1]) < 0.01

    def test_t2_decay_at_zero(self):
        val = t2_decay_model(np.array([0.0]), 1.0, 100e-6, 0.0)
        assert val[0] == pytest.approx(1.0)

    def test_t2_decay_at_large_tau(self):
        val = t2_decay_model(np.array([1.0]), 1.0, 100e-6, 0.5)
        assert val[0] == pytest.approx(0.5, abs=0.01)

    def test_t1_recovery_at_zero(self):
        # S(0) = A(1 - C*exp(0)) + B = A(1 - C) + B
        val = t1_recovery_model(np.array([0.0]), 1.0, 5e-3, 1.0, 0.0)
        assert val[0] == pytest.approx(0.0)

    def test_t1_recovery_at_infinity(self):
        val = t1_recovery_model(np.array([1.0]), 1.0, 5e-3, 1.0, 0.0)
        assert val[0] == pytest.approx(1.0, abs=0.01)

    def test_biexponential_at_zero(self):
        val = biexponential_decay(np.array([0.0]), 0.6, 1e-4, 0.4, 1e-3, 0.1)
        assert val[0] == pytest.approx(0.6 + 0.4 + 0.1)

    def test_biexponential_at_large_t(self):
        val = biexponential_decay(np.array([10.0]), 0.6, 1e-4, 0.4, 1e-3, 0.1)
        assert val[0] == pytest.approx(0.1, abs=0.01)

    def test_ramsey_stretch_effect(self):
        tau = np.linspace(0, 5e-6, 100)
        s1 = ramsey_model(tau, 1.0, 1e6, 1e-6, 0.0, 0.0, stretch=1.0)
        s2 = ramsey_model(tau, 1.0, 1e6, 1e-6, 0.0, 0.0, stretch=2.0)
        # Different stretch should give different curves
        assert not np.allclose(s1, s2)


class TestGoodnessOfFit:
    def test_perfect_fit(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gof = goodness_of_fit(data, data, 2)
        assert gof["r_squared"] == pytest.approx(1.0)
        assert gof["chi_squared"] == pytest.approx(0.0)

    def test_bad_fit(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        gof = goodness_of_fit(data, model, 2)
        assert gof["r_squared"] < 0.5

    def test_returns_all_keys(self):
        data = np.array([1.0, 2.0, 3.0])
        gof = goodness_of_fit(data, data, 1)
        assert "r_squared" in gof
        assert "chi_squared" in gof
        assert "aic" in gof
        assert "bic" in gof


class TestFitRamsey:
    def test_fit_recovers_field(self):
        from wimp.synthetic import generate_ramsey_data
        tau = np.linspace(0, 5e-6, 200)
        b_true = 50e-6
        data = generate_ramsey_data(tau, b_true, 1e-6, snr=100, seed=42)
        fit = fit_ramsey(data["tau"], data["signal"])
        assert fit["b_field"] == pytest.approx(b_true, rel=0.05)

    def test_fit_recovers_t2star(self):
        from wimp.synthetic import generate_ramsey_data
        tau = np.linspace(0, 5e-6, 200)
        t2star_true = 1e-6
        data = generate_ramsey_data(tau, 50e-6, t2star_true, snr=100, seed=42)
        fit = fit_ramsey(data["tau"], data["signal"])
        assert fit["t2star"] == pytest.approx(t2star_true, rel=0.15)

    def test_fit_good_r_squared(self):
        from wimp.synthetic import generate_ramsey_data
        tau = np.linspace(0, 5e-6, 200)
        data = generate_ramsey_data(tau, 50e-6, 1e-6, snr=100, seed=42)
        fit = fit_ramsey(data["tau"], data["signal"])
        assert fit["r_squared"] > 0.9

    def test_fit_with_free_stretch(self):
        from wimp.synthetic import generate_ramsey_data
        tau = np.linspace(0, 5e-6, 200)
        data = generate_ramsey_data(tau, 50e-6, 1e-6, snr=100, seed=42)
        fit = fit_ramsey(data["tau"], data["signal"], fix_stretch=False)
        assert "stretch" in fit

    def test_fit_with_custom_p0(self):
        from wimp.synthetic import generate_ramsey_data
        tau = np.linspace(0, 5e-6, 200)
        data = generate_ramsey_data(tau, 50e-6, 1e-6, snr=100, seed=42)
        fit = fit_ramsey(data["tau"], data["signal"], p0={"phase": 0.1})
        assert "b_field" in fit

    def test_fit_returns_cov(self):
        from wimp.synthetic import generate_ramsey_data
        tau = np.linspace(0, 5e-6, 200)
        data = generate_ramsey_data(tau, 50e-6, 1e-6, snr=100, seed=42)
        fit = fit_ramsey(data["tau"], data["signal"])
        assert "cov" in fit
        assert fit["cov"].ndim == 2


class TestFitT2Decay:
    def test_fit_recovers_t2(self):
        from wimp.synthetic import generate_echo_data
        tau = np.linspace(0, 300e-6, 150)
        t2_true = 100e-6
        data = generate_echo_data(tau, t2_true, snr=100, seed=7)
        fit = fit_t2_decay(data["tau"], data["signal"])
        assert fit["t2"] == pytest.approx(t2_true, rel=0.15)

    def test_fit_good_r_squared(self):
        from wimp.synthetic import generate_echo_data
        tau = np.linspace(0, 300e-6, 150)
        data = generate_echo_data(tau, 100e-6, snr=100, seed=7)
        fit = fit_t2_decay(data["tau"], data["signal"])
        assert fit["r_squared"] > 0.9


class TestFitT1Recovery:
    def test_fit_recovers_t1(self):
        from wimp.synthetic import generate_t1_data
        tau = np.linspace(0, 25e-3, 150)
        t1_true = 5e-3
        data = generate_t1_data(tau, t1_true, snr=100, seed=12)
        fit = fit_t1_recovery(data["tau"], data["signal"])
        assert fit["t1"] == pytest.approx(t1_true, rel=0.15)


class TestFitBiexponential:
    def test_fit_recovers_two_timescales(self):
        tau = np.linspace(0, 1e-3, 200)
        t_fast, t_slow = 50e-6, 500e-6
        clean = biexponential_decay(tau, 0.6, t_fast, 0.4, t_slow, 0.1)
        rng = np.random.default_rng(0)
        noisy = clean + rng.normal(0, 0.01, size=clean.shape)
        fit = fit_biexponential(tau, noisy)
        # Ensure both time constants are recovered (order may swap)
        fitted_times = sorted([fit["t_fast"], fit["t_slow"]])
        true_times = sorted([t_fast, t_slow])
        assert fitted_times[0] == pytest.approx(true_times[0], rel=0.3)
        assert fitted_times[1] == pytest.approx(true_times[1], rel=0.3)


class TestFieldExtraction:
    def test_extract_field_ramsey(self):
        result = {"frequency": 1.4012e6}
        b = extract_field_ramsey(result)
        expected = 1.4012e6 / GAMMA_NV
        assert b == pytest.approx(expected)

    def test_extract_field_echo(self):
        # phi / (4 * gamma * tau)
        phi = 1.0
        tau = 10e-6
        b = extract_field_echo(phi, tau)
        expected = 1.0 / (4.0 * GAMMA_NV * tau)
        assert b == pytest.approx(expected)

    def test_extract_field_echo_negative_phase(self):
        b = extract_field_echo(-1.0, 10e-6)
        assert b > 0  # should take absolute value


class TestInitialFrequencyGuess:
    def test_detects_known_frequency(self):
        freq_true = 5e5
        # Need sufficient samples for FFT resolution: df = 1/(N*dt)
        # 2000 points over 10 us -> dt=5ns, df=50kHz, adequate for 500kHz
        tau = np.linspace(0, 10e-6, 2000)
        signal = np.cos(2 * np.pi * freq_true * tau)
        guess = _initial_frequency_guess(tau, signal)
        assert guess == pytest.approx(freq_true, rel=0.1)


# ---------------------------------------------------------------------------
# Bayesian fitting (MCMC)
# ---------------------------------------------------------------------------

from wimp.relaxation import fit_ramsey_bayesian


class TestFitRamseyBayesian:
    def test_output_keys(self):
        from wimp.synthetic import generate_ramsey_data
        tau = np.linspace(0, 5e-6, 200)
        data = generate_ramsey_data(tau, 50e-6, 1e-6, snr=100, seed=42)
        result = fit_ramsey_bayesian(data["tau"], data["signal"],
                                     n_samples=200, n_burn=50, seed=42)
        assert "samples" in result
        assert "param_names" in result
        assert "median" in result
        assert "credible_intervals" in result
        assert "b_field" in result
        assert "acceptance_rate" in result

    def test_samples_shape(self):
        from wimp.synthetic import generate_ramsey_data
        tau = np.linspace(0, 5e-6, 200)
        data = generate_ramsey_data(tau, 50e-6, 1e-6, snr=100, seed=42)
        n_samples = 300
        result = fit_ramsey_bayesian(data["tau"], data["signal"],
                                     n_samples=n_samples, n_burn=50, seed=42)
        assert result["samples"].shape == (n_samples, 5)

    def test_median_close_to_mle(self):
        from wimp.synthetic import generate_ramsey_data
        tau = np.linspace(0, 5e-6, 200)
        b_true = 50e-6
        data = generate_ramsey_data(tau, b_true, 1e-6, snr=100, seed=42)
        result = fit_ramsey_bayesian(data["tau"], data["signal"],
                                     n_samples=500, n_burn=100, seed=42)
        # Median B-field should be in the right ballpark
        assert result["b_field"] == pytest.approx(b_true, rel=0.2)

    def test_credible_intervals_contain_median(self):
        from wimp.synthetic import generate_ramsey_data
        tau = np.linspace(0, 5e-6, 200)
        data = generate_ramsey_data(tau, 50e-6, 1e-6, snr=100, seed=42)
        result = fit_ramsey_bayesian(data["tau"], data["signal"],
                                     n_samples=200, n_burn=50, seed=42)
        for name in result["param_names"]:
            lo, hi = result["credible_intervals"][name]
            median_val = result["median"][name]
            assert lo <= median_val <= hi

    def test_acceptance_rate_reasonable(self):
        from wimp.synthetic import generate_ramsey_data
        tau = np.linspace(0, 5e-6, 200)
        data = generate_ramsey_data(tau, 50e-6, 1e-6, snr=100, seed=42)
        result = fit_ramsey_bayesian(data["tau"], data["signal"],
                                     n_samples=200, n_burn=50, seed=42)
        assert 0.0 < result["acceptance_rate"] < 1.0


# ---------------------------------------------------------------------------
# Multi-NV joint fitting
# ---------------------------------------------------------------------------

from wimp.relaxation import fit_ramsey_joint


class TestFitRamseyJoint:
    def test_output_keys(self):
        from wimp.synthetic import generate_ramsey_data
        tau = np.linspace(0, 5e-6, 200)
        b_true = 50e-6
        signals = np.stack([
            generate_ramsey_data(tau, b_true, 1e-6, snr=100, seed=i)["signal"]
            for i in range(3)
        ])
        result = fit_ramsey_joint(tau, signals)
        assert "shared" in result
        assert "per_nd" in result
        assert "b_field" in result
        assert "cov" in result
        assert "n_nds" in result

    def test_shared_frequency(self):
        from wimp.synthetic import generate_ramsey_data
        tau = np.linspace(0, 5e-6, 200)
        b_true = 50e-6
        signals = np.stack([
            generate_ramsey_data(tau, b_true, 1e-6, snr=100, seed=i)["signal"]
            for i in range(3)
        ])
        result = fit_ramsey_joint(tau, signals, shared_params=["frequency"])
        # All per-ND results should have the same frequency
        freqs = [nd["frequency"] for nd in result["per_nd"]]
        assert all(f == pytest.approx(freqs[0]) for f in freqs)

    def test_recovers_field(self):
        from wimp.synthetic import generate_ramsey_data
        tau = np.linspace(0, 5e-6, 200)
        b_true = 50e-6
        signals = np.stack([
            generate_ramsey_data(tau, b_true, 1e-6, snr=100, seed=i)["signal"]
            for i in range(2)
        ])
        result = fit_ramsey_joint(tau, signals)
        assert result["b_field"] == pytest.approx(b_true, rel=0.1)

    def test_n_nds_correct(self):
        from wimp.synthetic import generate_ramsey_data
        tau = np.linspace(0, 5e-6, 200)
        n = 4
        signals = np.stack([
            generate_ramsey_data(tau, 50e-6, 1e-6, snr=50, seed=i)["signal"]
            for i in range(n)
        ])
        result = fit_ramsey_joint(tau, signals)
        assert result["n_nds"] == n
        assert len(result["per_nd"]) == n

    def test_single_nd(self):
        from wimp.synthetic import generate_ramsey_data
        tau = np.linspace(0, 5e-6, 200)
        data = generate_ramsey_data(tau, 50e-6, 1e-6, snr=100, seed=42)
        signals = data["signal"][np.newaxis, :]
        result = fit_ramsey_joint(tau, signals)
        assert result["n_nds"] == 1
