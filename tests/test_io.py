"""Tests for wimp.io."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from wimp.io import (
    WIMPDataset,
    save_hdf5,
    load_hdf5,
    save_numpy,
    load_numpy,
    load_csv_timeseries,
    export_results,
    load_raw_measurement,
    _make_serialisable,
)


class TestWIMPDataset:
    def test_default_construction(self):
        ds = WIMPDataset()
        assert ds.protocol == "ramsey"
        assert ds.signal.shape == (0,)
        assert ds.field_timeseries is None
        assert ds.nd_positions is None

    def test_construction_with_data(self, tau_ramsey):
        sig = np.random.default_rng(0).normal(size=(3, len(tau_ramsey)))
        ds = WIMPDataset(
            protocol="echo",
            tau_array=tau_ramsey,
            signal=sig,
            metadata={"key": "value"},
        )
        assert ds.protocol == "echo"
        assert ds.signal.shape == (3, len(tau_ramsey))
        assert ds.metadata["key"] == "value"


class TestHDF5IO:
    def test_roundtrip(self, sample_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.h5"
            save_hdf5(sample_dataset, path)
            loaded = load_hdf5(path)

            assert loaded.protocol == sample_dataset.protocol
            np.testing.assert_array_almost_equal(
                loaded.tau_array, sample_dataset.tau_array
            )
            np.testing.assert_array_almost_equal(
                loaded.signal, sample_dataset.signal
            )

    def test_roundtrip_with_optional_fields(self, tau_ramsey):
        ds = WIMPDataset(
            protocol="ramsey",
            tau_array=tau_ramsey,
            signal=np.ones((2, len(tau_ramsey))),
            field_timeseries=np.ones((2, 10)),
            nd_positions=np.ones((2, 3)),
            frequency_axis=np.linspace(1e3, 1e6, 50),
            metadata={"temp": 295.0, "n_averages": 1000},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "full.h5"
            save_hdf5(ds, path)
            loaded = load_hdf5(path)

            assert loaded.field_timeseries is not None
            np.testing.assert_array_almost_equal(
                loaded.field_timeseries, ds.field_timeseries
            )
            np.testing.assert_array_almost_equal(
                loaded.nd_positions, ds.nd_positions
            )
            np.testing.assert_array_almost_equal(
                loaded.frequency_axis, ds.frequency_axis
            )
            assert loaded.metadata["temp"] == 295.0


class TestNumpyIO:
    def test_roundtrip(self, sample_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "npz_out"
            save_numpy(sample_dataset, d)
            loaded = load_numpy(d)

            assert loaded.protocol == sample_dataset.protocol
            np.testing.assert_array_almost_equal(
                loaded.tau_array, sample_dataset.tau_array
            )
            np.testing.assert_array_almost_equal(
                loaded.signal, sample_dataset.signal
            )


class TestCSVTimeseries:
    def test_load_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ts.csv"
            path.write_text("time_s,ch0,ch1\n0.0,1.0,2.0\n0.001,1.1,2.1\n0.002,1.2,2.2\n")
            time, field = load_csv_timeseries(path)
            assert time.shape == (3,)
            assert field.shape == (2, 3)
            assert time[1] == pytest.approx(0.001)


class TestExportResults:
    def test_export_json(self):
        results = {"b_field": np.array([1e-6, 2e-6]), "r_squared": 0.99}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.json"
            export_results(results, path, fmt="json")
            data = json.loads(path.read_text())
            assert data["r_squared"] == 0.99

    def test_export_csv(self):
        results = {"metric": 42.0, "name": "test"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.csv"
            export_results(results, path, fmt="csv")
            text = path.read_text()
            assert "metric" in text

    def test_unknown_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unknown format"):
                export_results({}, Path(tmpdir) / "out.txt", fmt="xml")


class TestLoadRawMeasurement:
    def test_npz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.npz"
            np.savez(path, tau=np.array([1, 2, 3]), signal=np.array([4, 5, 6]))
            result = load_raw_measurement(path)
            assert "tau" in result
            assert "signal" in result

    def test_npy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.npy"
            np.save(path, np.array([1, 2, 3]))
            result = load_raw_measurement(path)
            assert "data" in result

    def test_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.csv"
            path.write_text("a,b\n1,2\n3,4\n")
            result = load_raw_measurement(path)
            assert "a" in result

    def test_unsupported_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.xyz"
            path.write_text("hello")
            with pytest.raises(ValueError, match="Unsupported"):
                load_raw_measurement(path)


class TestMakeSerializable:
    def test_numpy_array(self):
        result = _make_serialisable(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_numpy_scalar(self):
        result = _make_serialisable(np.float64(3.14))
        assert isinstance(result, float)

    def test_numpy_int(self):
        result = _make_serialisable(np.int64(42))
        assert isinstance(result, int)

    def test_numpy_bool(self):
        result = _make_serialisable(np.bool_(True))
        assert isinstance(result, bool)

    def test_nested_dict(self):
        obj = {"a": np.array([1]), "b": {"c": np.float64(2.0)}}
        result = _make_serialisable(obj)
        assert result == {"a": [1], "b": {"c": 2.0}}


# ---------------------------------------------------------------------------
# NWB export
# ---------------------------------------------------------------------------


class TestExportNWB:
    @pytest.fixture(autouse=True)
    def _skip_without_pynwb(self):
        pytest.importorskip("pynwb")

    def test_creates_file(self, sample_dataset, tmp_path):
        from wimp.io import export_nwb

        results = {"field_timeseries": np.ones(10), "fits": []}
        path = tmp_path / "test.nwb"
        export_nwb(sample_dataset, results, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_with_source_estimate(self, sample_dataset, tmp_path):
        from wimp.io import export_nwb

        results = {
            "field_timeseries": np.ones(10),
            "source_estimate": np.ones((3, 10)),
            "fits": [],
        }
        path = tmp_path / "test2.nwb"
        export_nwb(
            sample_dataset, results, path,
            session_description="test session",
            experimenter="test",
        )
        assert path.exists()

    def test_roundtrip_readable(self, sample_dataset, tmp_path):
        from wimp.io import export_nwb
        import pynwb

        results = {"field_timeseries": np.ones(10), "fits": []}
        path = tmp_path / "test3.nwb"
        export_nwb(sample_dataset, results, path)

        with pynwb.NWBHDF5IO(str(path), "r") as io:
            nwb = io.read()
            assert nwb.identifier is not None
