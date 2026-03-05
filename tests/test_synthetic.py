"""Tests for wimp.synthetic."""

import numpy as np
import pytest

from wimp.synthetic import (
    generate_ramsey_data,
    generate_echo_data,
    generate_t1_data,
    generate_dd_sweep,
    generate_neural_field,
    generate_full_experiment,
    _dipole_field_z,
)


class TestGenerateRamseyData:
    def test_output_keys(self):
        tau = np.linspace(0, 5e-6, 100)
        result = generate_ramsey_data(tau, 50e-6, 1e-6, snr=50, seed=0)
        assert "tau" in result
        assert "signal" in result
        assert "clean_signal" in result
        assert "ground_truth" in result

    def test_signal_shape(self):
        tau = np.linspace(0, 5e-6, 100)
        result = generate_ramsey_data(tau, 50e-6, 1e-6, snr=50, seed=0)
        assert result["signal"].shape == (100,)

    def test_deterministic_with_seed(self):
        tau = np.linspace(0, 5e-6, 100)
        r1 = generate_ramsey_data(tau, 50e-6, 1e-6, snr=50, seed=42)
        r2 = generate_ramsey_data(tau, 50e-6, 1e-6, snr=50, seed=42)
        np.testing.assert_array_equal(r1["signal"], r2["signal"])

    def test_ground_truth(self):
        tau = np.linspace(0, 5e-6, 100)
        result = generate_ramsey_data(tau, 50e-6, 1e-6, snr=50, seed=0)
        assert result["ground_truth"]["b_field"] == 50e-6

    def test_zero_snr_no_noise(self):
        tau = np.linspace(0, 5e-6, 100)
        result = generate_ramsey_data(tau, 50e-6, 1e-6, snr=0, seed=0)
        np.testing.assert_array_almost_equal(
            result["signal"], result["clean_signal"]
        )


class TestGenerateEchoData:
    def test_output_keys(self):
        tau = np.linspace(0, 300e-6, 100)
        result = generate_echo_data(tau, 100e-6, snr=50, seed=0)
        assert "tau" in result
        assert "signal" in result
        assert "ground_truth" in result

    def test_ground_truth_t2(self):
        tau = np.linspace(0, 300e-6, 100)
        result = generate_echo_data(tau, 100e-6, snr=50, seed=0)
        assert result["ground_truth"]["t2"] == 100e-6


class TestGenerateT1Data:
    def test_output_keys(self):
        tau = np.linspace(0, 25e-3, 100)
        result = generate_t1_data(tau, 5e-3, snr=50, seed=0)
        assert "tau" in result
        assert "signal" in result
        assert "ground_truth" in result


class TestGenerateDDSweep:
    def test_output_keys(self):
        tau = np.logspace(-7, -4, 100)
        noise_func = lambda f: 1e-20 * np.ones_like(f)
        result = generate_dd_sweep(tau, 16, noise_func, snr=50, seed=0)
        assert "tau" in result
        assert "coherence" in result
        assert "freq_axis" in result

    def test_freq_axis_matches_tau(self):
        tau = np.logspace(-7, -4, 50)
        noise_func = lambda f: 1e-20 * np.ones_like(f)
        result = generate_dd_sweep(tau, 16, noise_func, snr=50, seed=0)
        expected_freq = 1.0 / (2.0 * tau)
        np.testing.assert_array_almost_equal(result["freq_axis"], expected_freq)


class TestDipoleFieldZ:
    def test_field_at_distance(self):
        # Dipole at origin, z-oriented, field point along z
        source = np.array([0.0, 0.0, 0.0])
        moment = np.array([0.0, 0.0, 1.0])
        field_pt = np.array([0.0, 0.0, 1e-6])
        bz = _dipole_field_z(source, moment, field_pt)
        # Should be positive (field along z for z-dipole on axis)
        assert bz > 0

    def test_zero_distance(self):
        source = np.array([0.0, 0.0, 0.0])
        moment = np.array([0.0, 0.0, 1.0])
        bz = _dipole_field_z(source, moment, source)
        assert bz == 0.0

    def test_field_falls_off_with_distance(self):
        source = np.array([0.0, 0.0, 0.0])
        moment = np.array([0.0, 0.0, 1.0])
        bz_close = _dipole_field_z(source, moment, np.array([0, 0, 1e-6]))
        bz_far = _dipole_field_z(source, moment, np.array([0, 0, 2e-6]))
        assert abs(bz_close) > abs(bz_far)


class TestGenerateNeuralField:
    def test_output_shape(self):
        n_neurons, n_nds, n_t = 3, 5, 100
        neuron_pos = np.column_stack([
            np.linspace(0, 1e-3, n_neurons),
            np.zeros(n_neurons),
            np.zeros(n_neurons),
        ])
        nd_pos = np.column_stack([
            np.linspace(0, 1e-3, n_nds),
            np.zeros(n_nds),
            np.full(n_nds, 5e-6),
        ])
        currents = np.random.default_rng(0).normal(size=(n_neurons, n_t))
        result = generate_neural_field(neuron_pos, nd_pos, currents, dt=1e-3)
        assert result["field"].shape == (n_nds, n_t)
        assert result["time"].shape == (n_t,)
        assert result["lead_field"].shape == (n_nds, n_neurons)


class TestGenerateFullExperiment:
    @pytest.mark.parametrize("protocol", ["ramsey", "echo", "t1", "dd"])
    def test_protocols(self, protocol):
        exp = generate_full_experiment(
            n_nds=4, n_neurons=2, n_timepoints=50,
            protocol=protocol, seed=1,
        )
        assert exp["protocol"] == protocol
        assert exp["signal"].shape[0] == 4
        assert exp["nd_positions"].shape == (4, 3)
        assert exp["neuron_positions"].shape == (2, 3)

    def test_unknown_protocol(self):
        with pytest.raises(ValueError):
            generate_full_experiment(protocol="unknown")

    def test_deterministic_with_seed(self):
        e1 = generate_full_experiment(n_nds=3, seed=99)
        e2 = generate_full_experiment(n_nds=3, seed=99)
        np.testing.assert_array_equal(e1["signal"], e2["signal"])
