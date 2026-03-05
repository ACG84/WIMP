"""Tests for wimp.connectome."""

import numpy as np
import pytest

from wimp.connectome import (
    CONNECTOME_NEURONS,
    load_connectome,
    map_to_connectome,
    connectome_propagation,
    known_circuit,
)


class TestConnectomeNeurons:
    def test_has_expected_neurons(self):
        assert "AVAL" in CONNECTOME_NEURONS
        assert "ALML" in CONNECTOME_NEURONS
        assert "AFDL" in CONNECTOME_NEURONS
        assert "ASEL" in CONNECTOME_NEURONS

    def test_neuron_metadata(self):
        info = CONNECTOME_NEURONS["AVAL"]
        assert info["type"] == "interneuron"
        assert info["class"] == "command"

    def test_sensory_neurons(self):
        for name in ["ALML", "ALMR", "AVM", "PLML", "PLMR"]:
            assert CONNECTOME_NEURONS[name]["type"] == "sensory"


class TestLoadConnectome:
    def test_default_keys(self):
        conn = load_connectome()
        assert "neuron_names" in conn
        assert "adjacency" in conn
        assert "adjacency_chemical" in conn
        assert "adjacency_gap" in conn

    def test_adjacency_shape(self):
        conn = load_connectome()
        n = len(conn["neuron_names"])
        assert conn["adjacency"].shape == (n, n)
        assert conn["adjacency_chemical"].shape == (n, n)
        assert conn["adjacency_gap"].shape == (n, n)

    def test_gap_junctions_symmetric(self):
        conn = load_connectome()
        np.testing.assert_array_equal(
            conn["adjacency_gap"], conn["adjacency_gap"].T
        )

    def test_adjacency_is_sum(self):
        conn = load_connectome()
        np.testing.assert_array_almost_equal(
            conn["adjacency"],
            conn["adjacency_chemical"] + conn["adjacency_gap"],
        )

    def test_has_nonzero_connections(self):
        conn = load_connectome()
        assert np.sum(conn["adjacency"]) > 0

    def test_load_from_npz(self, tmp_path):
        conn = load_connectome()
        path = tmp_path / "conn.npz"
        np.savez(
            path,
            neuron_names=np.array(conn["neuron_names"]),
            adjacency_chemical=conn["adjacency_chemical"],
            adjacency_gap=conn["adjacency_gap"],
        )
        loaded = load_connectome(path)
        assert loaded["neuron_names"] == conn["neuron_names"]
        np.testing.assert_array_equal(loaded["adjacency"], conn["adjacency"])

    def test_load_from_csv(self, tmp_path):
        path = tmp_path / "conn.csv"
        path.write_text(
            "source,target,weight,type\n"
            "A,B,3,chemical\n"
            "B,C,2,gap\n"
        )
        conn = load_connectome(path)
        assert "A" in conn["neuron_names"]
        assert conn["adjacency"].shape[0] == 3

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "conn.xyz"
        path.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            load_connectome(path)


class TestMapToConnectome:
    def test_output_keys(self):
        result = map_to_connectome(["AVAL", "AVBL"], np.array([1.0, 2.0]))
        assert "mapped_activity" in result
        assert "connectome_nodes" in result
        assert "adjacency" in result
        assert "mapped_indices" in result
        assert "unmapped_neurons" in result

    def test_mapped_values(self):
        result = map_to_connectome(["AVAL"], np.array([5.0]))
        conn = load_connectome()
        idx = conn["neuron_names"].index("AVAL")
        assert result["mapped_activity"][idx] == 5.0

    def test_unmapped_neurons(self):
        result = map_to_connectome(
            ["AVAL", "FAKE_NEURON"], np.array([1.0, 2.0])
        )
        assert "FAKE_NEURON" in result["unmapped_neurons"]

    def test_2d_activity(self):
        activity = np.ones((2, 10))
        result = map_to_connectome(["AVAL", "AVBL"], activity)
        n_conn = len(result["connectome_nodes"])
        assert result["mapped_activity"].shape == (n_conn, 10)

    def test_nan_for_unmapped(self):
        result = map_to_connectome(["AVAL"], np.array([1.0]))
        conn = load_connectome()
        idx_avbl = conn["neuron_names"].index("AVBL")
        assert np.isnan(result["mapped_activity"][idx_avbl])


class TestConnectomePropagation:
    def test_output_shape(self):
        n = 10
        a0 = np.zeros(n)
        a0[0] = 1.0
        adj = np.eye(n, k=1)
        traj = connectome_propagation(a0, adj, n_steps=5)
        assert traj.shape == (6, n)

    def test_initial_state_preserved(self):
        n = 5
        a0 = np.array([1.0, 0, 0, 0, 0])
        adj = np.eye(n, k=1)
        traj = connectome_propagation(a0, adj, n_steps=3)
        np.testing.assert_array_equal(traj[0], a0)

    def test_activity_spreads(self):
        conn = load_connectome()
        n = len(conn["neuron_names"])
        a0 = np.zeros(n)
        idx_aval = conn["neuron_names"].index("AVAL")
        a0[idx_aval] = 1.0
        traj = connectome_propagation(a0, conn["adjacency"], n_steps=5)
        # After propagation, more neurons should have nonzero activity
        assert np.sum(traj[-1] > 1e-6) > np.sum(traj[0] > 1e-6)

    def test_decay_zero_preserves_initial(self):
        n = 5
        a0 = np.array([1.0, 0, 0, 0, 0])
        adj = np.ones((n, n))
        traj = connectome_propagation(a0, adj, n_steps=3, decay=0.0)
        # With decay=0, a[t+1] = a[0] always
        for t in range(4):
            np.testing.assert_array_equal(traj[t], a0)


class TestKnownCircuit:
    def test_tap_withdrawal(self):
        circ = known_circuit("tap_withdrawal")
        assert "neurons" in circ
        assert "adjacency" in circ
        assert "description" in circ
        assert "ALML" in circ["neurons"]
        assert "AVAL" in circ["neurons"]

    def test_thermotaxis(self):
        circ = known_circuit("thermotaxis")
        assert "AFDL" in circ["neurons"]
        assert "AIYL" in circ["neurons"]

    def test_chemotaxis(self):
        circ = known_circuit("chemotaxis")
        assert "ASEL" in circ["neurons"]
        assert len(circ["neurons"]) == 12

    def test_nose_touch(self):
        circ = known_circuit("nose_touch")
        assert "ASHL" in circ["neurons"]

    def test_adjacency_shape(self):
        circ = known_circuit("tap_withdrawal")
        n = len(circ["neurons"])
        assert circ["adjacency"].shape == (n, n)

    def test_unknown_circuit_raises(self):
        with pytest.raises(ValueError, match="Unknown circuit"):
            known_circuit("nonexistent")
