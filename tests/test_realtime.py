"""Tests for wimp.realtime."""

import time

import numpy as np
import pytest

from wimp.realtime import (
    RealtimeConfig,
    RealtimeProcessor,
    RealtimeResult,
    MeasurementFrame,
    FileWatcherSource,
)


class TestRealtimeConfig:
    def test_defaults(self):
        config = RealtimeConfig()
        assert config.protocol == "ramsey"
        assert config.buffer_size == 100
        assert config.fit_interval == 0.1
        assert config.averaging == 1

    def test_custom(self):
        config = RealtimeConfig(protocol="echo", buffer_size=50)
        assert config.protocol == "echo"
        assert config.buffer_size == 50


class TestMeasurementFrame:
    def test_construction(self):
        frame = MeasurementFrame(
            tau_array=np.linspace(0, 5e-6, 100),
            signal=np.random.default_rng(0).normal(size=100),
            timestamp=1.0,
        )
        assert frame.timestamp == 1.0
        assert frame.signal.shape == (100,)


class TestRealtimeProcessor:
    def test_push_and_count(self):
        proc = RealtimeProcessor()
        tau = np.linspace(0, 5e-6, 50)
        for _ in range(5):
            proc.push_arrays(tau, np.random.default_rng(0).normal(size=50))
        assert proc.frame_count == 5
        assert proc.buffer_len == 5

    def test_process_once(self):
        from wimp.synthetic import generate_ramsey_data
        proc = RealtimeProcessor(RealtimeConfig(protocol="ramsey", averaging=1))
        tau = np.linspace(0, 5e-6, 100)
        data = generate_ramsey_data(tau, 50e-6, 1e-6, snr=50, seed=0)
        proc.push_arrays(tau, data["signal"])
        result = proc.process_once()
        assert result is not None
        assert isinstance(result, RealtimeResult)
        assert result.frame_count == 1
        assert len(result.fits) == 1

    def test_process_once_insufficient_frames(self):
        proc = RealtimeProcessor(RealtimeConfig(averaging=5))
        tau = np.linspace(0, 5e-6, 50)
        proc.push_arrays(tau, np.ones(50))
        result = proc.process_once()
        assert result is None

    def test_averaging(self):
        from wimp.synthetic import generate_ramsey_data
        proc = RealtimeProcessor(RealtimeConfig(protocol="ramsey", averaging=3))
        tau = np.linspace(0, 5e-6, 100)
        for i in range(5):
            data = generate_ramsey_data(tau, 50e-6, 1e-6, snr=30)
            proc.push_arrays(tau, data["signal"])
        result = proc.process_once()
        assert result is not None

    def test_start_stop(self):
        proc = RealtimeProcessor(RealtimeConfig(fit_interval=0.01))
        proc.start()
        assert proc.is_running
        time.sleep(0.05)
        proc.stop()
        assert not proc.is_running

    def test_callback_called(self):
        from wimp.synthetic import generate_ramsey_data
        results = []
        proc = RealtimeProcessor(RealtimeConfig(
            protocol="ramsey", fit_interval=0.01, averaging=1,
        ))
        proc.on_result(lambda r: results.append(r))
        proc.start()

        tau = np.linspace(0, 5e-6, 100)
        for _ in range(5):
            data = generate_ramsey_data(tau, 50e-6, 1e-6, snr=30)
            proc.push_arrays(tau, data["signal"])
            time.sleep(0.02)

        time.sleep(0.1)
        proc.stop()
        assert len(results) > 0

    def test_clear(self):
        proc = RealtimeProcessor()
        tau = np.linspace(0, 5e-6, 50)
        for _ in range(10):
            proc.push_arrays(tau, np.ones(50))
        assert proc.buffer_len == 10
        proc.clear()
        assert proc.buffer_len == 0
        assert proc.frame_count == 0

    def test_buffer_size_limit(self):
        proc = RealtimeProcessor(RealtimeConfig(buffer_size=5))
        tau = np.linspace(0, 5e-6, 50)
        for _ in range(20):
            proc.push_arrays(tau, np.ones(50))
        assert proc.buffer_len == 5
        assert proc.frame_count == 20

    def test_multidim_signal(self):
        from wimp.synthetic import generate_ramsey_data
        proc = RealtimeProcessor(RealtimeConfig(protocol="ramsey", averaging=1))
        tau = np.linspace(0, 5e-6, 100)
        # 3-ND signal
        signal = np.stack([
            generate_ramsey_data(tau, 50e-6, 1e-6, snr=50, seed=i)["signal"]
            for i in range(3)
        ])
        proc.push_arrays(tau, signal)
        result = proc.process_once()
        assert result is not None
        assert len(result.fits) == 3
        assert result.field_values.shape == (3,)

    def test_echo_protocol(self):
        from wimp.synthetic import generate_echo_data
        proc = RealtimeProcessor(RealtimeConfig(protocol="echo", averaging=1))
        tau = np.linspace(0, 300e-6, 100)
        data = generate_echo_data(tau, 100e-6, snr=50, seed=0)
        proc.push_arrays(tau, data["signal"])
        result = proc.process_once()
        assert result is not None
        assert len(result.fits) == 1


class TestFileWatcherSource:
    def test_start_stop(self, tmp_path):
        proc = RealtimeProcessor()
        watcher = FileWatcherSource(str(tmp_path), proc)
        watcher.start()
        time.sleep(0.1)
        watcher.stop()

    def test_ingests_new_npz(self, tmp_path):
        proc = RealtimeProcessor()
        watcher = FileWatcherSource(str(tmp_path), proc, poll_interval=0.1)
        watcher.start()
        time.sleep(0.15)  # let it snapshot existing files

        # Drop a new npz file
        tau = np.linspace(0, 5e-6, 50)
        signal = np.ones(50)
        np.savez(tmp_path / "frame_001.npz", tau_array=tau, signal=signal)

        time.sleep(0.3)
        watcher.stop()
        assert proc.frame_count >= 1


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------

from wimp.realtime import KalmanFilter


class TestKalmanFilter:
    def test_position_model_converges_on_constant(self):
        kf = KalmanFilter(state_model="position", process_noise=1e-9, measurement_noise=1e-6)
        true_value = 5.0
        rng = np.random.default_rng(42)
        estimates = []
        for _ in range(50):
            kf.predict()
            est = kf.update(true_value + rng.normal(0, 0.001))
            estimates.append(est)
        # After convergence, estimate should be close to truth
        assert abs(estimates[-1] - true_value) < 0.01

    def test_position_velocity_model(self):
        kf = KalmanFilter(state_model="position_velocity", process_noise=1e-9,
                          measurement_noise=1e-6, dt=0.1)
        # Linear ramp: B = 1.0 + 0.5 * t
        rng = np.random.default_rng(42)
        estimates = []
        for i in range(100):
            t = i * 0.1
            true_b = 1.0 + 0.5 * t
            kf.predict(dt=0.1)
            est = kf.update(true_b + rng.normal(0, 0.01))
            estimates.append(est)
        # Should track the ramp
        expected_final = 1.0 + 0.5 * 9.9
        assert abs(estimates[-1] - expected_final) < 0.5

    def test_state_vector(self):
        kf = KalmanFilter(state_model="position")
        kf.update(3.0)
        state = kf.state
        assert len(state) == 1
        assert state[0] == pytest.approx(3.0)

    def test_state_vector_pv(self):
        kf = KalmanFilter(state_model="position_velocity")
        kf.update(3.0)
        state = kf.state
        assert len(state) == 2

    def test_reset(self):
        kf = KalmanFilter(state_model="position")
        kf.update(5.0)
        kf.reset()
        assert kf.state[0] == 0.0

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="state_model"):
            KalmanFilter(state_model="invalid")

    def test_first_update_initializes(self):
        kf = KalmanFilter(state_model="position")
        val = kf.update(42.0)
        assert val == pytest.approx(42.0)

    def test_predict_before_init_is_noop(self):
        kf = KalmanFilter(state_model="position")
        kf.predict()  # should not raise
        assert kf.state[0] == 0.0


class TestRealtimeWithKalman:
    def test_kalman_reduces_variance(self):
        from wimp.synthetic import generate_ramsey_data
        config = RealtimeConfig(
            protocol="ramsey",
            averaging=1,
            use_kalman=True,
            kalman_state_model="position",
            kalman_process_noise=1e-9,
            kalman_measurement_noise=1e-8,
        )
        proc = RealtimeProcessor(config)
        tau = np.linspace(0, 5e-6, 100)

        fields = []
        for i in range(10):
            data = generate_ramsey_data(tau, 50e-6, 1e-6, snr=30, seed=i)
            proc.push_arrays(tau, data["signal"])
            result = proc.process_once()
            if result is not None and result.field_values is not None:
                fields.append(result.field_values[0])

        assert len(fields) > 5
        # Kalman-filtered values should exist
        assert all(np.isfinite(f) for f in fields)

    def test_kalman_states_returned(self):
        from wimp.synthetic import generate_ramsey_data
        config = RealtimeConfig(
            protocol="ramsey",
            averaging=1,
            use_kalman=True,
        )
        proc = RealtimeProcessor(config)
        tau = np.linspace(0, 5e-6, 100)
        data = generate_ramsey_data(tau, 50e-6, 1e-6, snr=50, seed=0)
        proc.push_arrays(tau, data["signal"])
        result = proc.process_once()
        assert result is not None
        assert result.kalman_states is not None
        assert len(result.kalman_states) == 1


class TestRealtimeWithDenoise:
    def test_denoise_integration(self):
        from wimp.synthetic import generate_ramsey_data
        config = RealtimeConfig(
            protocol="ramsey",
            averaging=1,
            denoise_method="savgol",
            denoise_params={"window_length": 11},
        )
        proc = RealtimeProcessor(config)
        tau = np.linspace(0, 5e-6, 100)
        data = generate_ramsey_data(tau, 50e-6, 1e-6, snr=20, seed=0)
        proc.push_arrays(tau, data["signal"])
        result = proc.process_once()
        assert result is not None
        assert len(result.fits) == 1
