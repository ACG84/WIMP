"""Tests for zarr I/O and zarr streaming."""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from wimp.io import WIMPDataset, save_zarr, load_zarr
from wimp.zarr_stream import ZarrStreamWriter, ZarrStreamReader


# ---------------------------------------------------------------------------
# Zarr I/O (save_zarr / load_zarr)
# ---------------------------------------------------------------------------


class TestSaveLoadZarr:
    def test_roundtrip(self, sample_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.zarr"
            save_zarr(sample_dataset, path)
            loaded = load_zarr(path)

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
            path = Path(tmpdir) / "full.zarr"
            save_zarr(ds, path)
            loaded = load_zarr(path)

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

    def test_lazy_load(self, sample_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lazy.zarr"
            save_zarr(sample_dataset, path)
            loaded = load_zarr(path, lazy=True)

            # Zarr arrays should not be plain numpy
            assert not isinstance(loaded.signal, np.ndarray)
            # But should be convertible
            np.testing.assert_array_almost_equal(
                np.asarray(loaded.signal), np.asarray(sample_dataset.signal)
            )

    def test_chunk_size(self, sample_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chunked.zarr"
            save_zarr(sample_dataset, path, chunk_size=50)
            loaded = load_zarr(path)
            np.testing.assert_array_almost_equal(
                loaded.tau_array, sample_dataset.tau_array
            )

    def test_metadata_roundtrip(self, tau_ramsey):
        ds = WIMPDataset(
            tau_array=tau_ramsey,
            signal=np.ones(len(tau_ramsey)),
            metadata={"key": "value", "numeric": 42},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meta.zarr"
            save_zarr(ds, path)
            loaded = load_zarr(path)
            assert loaded.metadata["key"] == "value"
            assert loaded.metadata["numeric"] == 42


# ---------------------------------------------------------------------------
# Zarr stream writer / reader
# ---------------------------------------------------------------------------

@dataclass
class _FakeFrame:
    """Minimal stand-in for MeasurementFrame."""
    signal: np.ndarray
    timestamp: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class _FakeResult:
    """Minimal stand-in for RealtimeResult."""
    field_values: np.ndarray | None = None
    timestamp: float = 0.0


class TestZarrStreamWriter:
    def test_write_and_read_frames(self):
        tau = np.linspace(0, 5e-6, 50)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stream.zarr"
            with ZarrStreamWriter(path, tau) as writer:
                for i in range(5):
                    frame = _FakeFrame(
                        signal=np.ones((2, 50)) * i,
                        timestamp=float(i),
                    )
                    writer.append_frame(frame)
                assert writer.n_frames == 5

            reader = ZarrStreamReader(path)
            assert reader.n_frames == 5

    def test_n_frames_property(self):
        tau = np.linspace(0, 5e-6, 50)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stream2.zarr"
            writer = ZarrStreamWriter(path, tau)
            assert writer.n_frames == 0
            writer.append_frame(_FakeFrame(
                signal=np.ones(50), timestamp=0.0
            ))
            assert writer.n_frames == 1
            writer.close()

    def test_context_manager_closes(self):
        tau = np.linspace(0, 5e-6, 50)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stream3.zarr"
            with ZarrStreamWriter(path, tau) as writer:
                writer.append_frame(_FakeFrame(
                    signal=np.ones(50), timestamp=0.0
                ))
            assert writer.closed

    def test_append_result(self):
        tau = np.linspace(0, 5e-6, 50)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stream4.zarr"
            with ZarrStreamWriter(path, tau) as writer:
                writer.append_frame(_FakeFrame(
                    signal=np.ones((2, 50)), timestamp=0.0
                ))
                writer.append_result(_FakeResult(
                    field_values=np.array([1e-6, 2e-6])
                ))
                writer.append_result(_FakeResult(
                    field_values=np.array([3e-6, 4e-6])
                ))

            reader = ZarrStreamReader(path)
            ds = reader.to_dataset()
            assert ds.field_timeseries is not None
            assert ds.field_timeseries.shape == (2, 2)

    def test_write_after_close_raises(self):
        tau = np.linspace(0, 5e-6, 50)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "closed.zarr"
            writer = ZarrStreamWriter(path, tau)
            writer.close()
            with pytest.raises(RuntimeError, match="closed"):
                writer.append_frame(_FakeFrame(
                    signal=np.ones(50), timestamp=0.0
                ))


class TestZarrStreamReader:
    def test_read_latest(self):
        tau = np.linspace(0, 5e-6, 50)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stream5.zarr"
            with ZarrStreamWriter(path, tau) as writer:
                for i in range(10):
                    writer.append_frame(_FakeFrame(
                        signal=np.ones((1, 50)) * i, timestamp=float(i),
                    ))

            reader = ZarrStreamReader(path)
            latest = reader.read_latest(3)
            assert latest["signals"].shape[0] == 3
            assert latest["timestamps"].shape[0] == 3
            # Should be last 3 frames
            np.testing.assert_array_equal(latest["timestamps"], [7.0, 8.0, 9.0])

    def test_to_dataset(self):
        tau = np.linspace(0, 5e-6, 50)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stream6.zarr"
            with ZarrStreamWriter(path, tau) as writer:
                for i in range(4):
                    writer.append_frame(_FakeFrame(
                        signal=np.ones((2, 50)) * (i + 1), timestamp=float(i),
                    ))

            reader = ZarrStreamReader(path)
            ds = reader.to_dataset()
            assert ds.protocol == "realtime_stream"
            # Average of [1, 2, 3, 4] = 2.5
            np.testing.assert_allclose(ds.signal, 2.5)
            assert ds.metadata["n_frames"] == 4

    def test_timestamps_property(self):
        tau = np.linspace(0, 5e-6, 50)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stream7.zarr"
            with ZarrStreamWriter(path, tau) as writer:
                for i in range(3):
                    writer.append_frame(_FakeFrame(
                        signal=np.ones(50), timestamp=float(i) * 0.5,
                    ))

            reader = ZarrStreamReader(path)
            np.testing.assert_allclose(reader.timestamps, [0.0, 0.5, 1.0])

    def test_empty_store(self):
        tau = np.linspace(0, 5e-6, 50)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stream8.zarr"
            with ZarrStreamWriter(path, tau) as writer:
                pass  # no frames written

            reader = ZarrStreamReader(path)
            assert reader.n_frames == 0
            latest = reader.read_latest()
            assert latest["signals"].size == 0
