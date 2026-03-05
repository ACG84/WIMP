"""C. elegans connectome integration for WIMP.

Provides utilities to map NV magnetometry neural activity data onto the
canonical C. elegans connectome, explore known neural circuits, and model
signal propagation through the connectome graph.

References
----------
White et al. (1986) *Phil. Trans. R. Soc. Lond. B* 314, 1–340.
Cook et al. (2019) *Nature* 571, 63–71.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Built-in neuron catalogue (simplified, ~30 key neurons)
# ---------------------------------------------------------------------------

CONNECTOME_NEURONS: dict[str, dict[str, str]] = {
    # Command interneurons (locomotion)
    "AVAL": {"type": "interneuron", "class": "command"},
    "AVAR": {"type": "interneuron", "class": "command"},
    "AVBL": {"type": "interneuron", "class": "command"},
    "AVBR": {"type": "interneuron", "class": "command"},
    "AVDL": {"type": "interneuron", "class": "command"},
    "AVDR": {"type": "interneuron", "class": "command"},
    "AVEL": {"type": "interneuron", "class": "command"},
    "AVER": {"type": "interneuron", "class": "command"},
    "PVCL": {"type": "interneuron", "class": "command"},
    "PVCR": {"type": "interneuron", "class": "command"},
    # Mechanosensory neurons
    "ALML": {"type": "sensory", "class": "mechanosensory"},
    "ALMR": {"type": "sensory", "class": "mechanosensory"},
    "AVM":  {"type": "sensory", "class": "mechanosensory"},
    "PLML": {"type": "sensory", "class": "mechanosensory"},
    "PLMR": {"type": "sensory", "class": "mechanosensory"},
    # Nociceptive / polymodal
    "ASHL": {"type": "sensory", "class": "nociceptive"},
    "ASHR": {"type": "sensory", "class": "nociceptive"},
    # Thermosensory
    "AFDL": {"type": "sensory", "class": "thermosensory"},
    "AFDR": {"type": "sensory", "class": "thermosensory"},
    # Chemosensory
    "ASEL": {"type": "sensory", "class": "chemosensory"},
    "ASER": {"type": "sensory", "class": "chemosensory"},
    "AWCL": {"type": "sensory", "class": "chemosensory"},
    "AWCR": {"type": "sensory", "class": "chemosensory"},
    # Interneurons in processing circuits
    "AIYL": {"type": "interneuron", "class": "first_layer"},
    "AIYR": {"type": "interneuron", "class": "first_layer"},
    "AIZL": {"type": "interneuron", "class": "first_layer"},
    "AIZR": {"type": "interneuron", "class": "first_layer"},
    "RIAL": {"type": "interneuron", "class": "second_layer"},
    "RIAR": {"type": "interneuron", "class": "second_layer"},
    "RIBL": {"type": "interneuron", "class": "second_layer"},
    "RIBR": {"type": "interneuron", "class": "second_layer"},
}
"""Simplified catalogue of key C. elegans neurons."""


# ---------------------------------------------------------------------------
# Built-in simplified connectome
# ---------------------------------------------------------------------------


def _build_default_connectome() -> dict[str, Any]:
    """Build a simplified connectome adjacency from published data."""
    names = list(CONNECTOME_NEURONS.keys())
    n = len(names)
    name_to_idx = {name: i for i, name in enumerate(names)}

    # Chemical synapse edges (source -> target, weight)
    # Simplified from Cook et al. 2019 / WormAtlas
    chemical_edges = [
        # Touch circuit
        ("ALML", "AVDL", 5), ("ALMR", "AVDR", 5),
        ("AVM", "AVDL", 3), ("AVM", "AVDR", 3),
        ("AVM", "AVBL", 2), ("AVM", "AVBR", 2),
        ("PLML", "PVCL", 4), ("PLMR", "PVCR", 4),
        ("AVDL", "AVAL", 8), ("AVDR", "AVAR", 8),
        ("PVCL", "AVBL", 6), ("PVCR", "AVBR", 6),
        # Command neuron cross-connections
        ("AVAL", "AVAR", 3), ("AVBL", "AVBR", 3),
        ("AVEL", "AVER", 2),
        # Thermotaxis circuit
        ("AFDL", "AIYL", 7), ("AFDR", "AIYR", 7),
        ("AIYL", "RIAL", 4), ("AIYR", "RIAR", 4),
        # Chemotaxis circuit
        ("ASEL", "AIYL", 5), ("ASER", "AIYR", 5),
        ("ASEL", "AIZL", 3), ("ASER", "AIZR", 3),
        ("AIYL", "RIAL", 3), ("AIZL", "RIBL", 3),
        ("AIYR", "RIAR", 3), ("AIZR", "RIBR", 3),
        # Nociceptive
        ("ASHL", "AVDL", 4), ("ASHR", "AVDR", 4),
        ("ASHL", "AVAL", 2), ("ASHR", "AVAR", 2),
    ]

    # Gap junction edges (symmetric, weight)
    gap_edges = [
        ("AVAL", "AVAR", 10), ("AVBL", "AVBR", 10),
        ("ALML", "ALMR", 3), ("PLML", "PLMR", 3),
        ("PVCL", "PVCR", 6),
        ("AFDL", "AFDR", 2),
        ("AIYL", "AIYR", 2),
        ("RIAL", "RIAR", 3),
    ]

    adj_chem = np.zeros((n, n))
    adj_gap = np.zeros((n, n))

    for src, tgt, w in chemical_edges:
        if src in name_to_idx and tgt in name_to_idx:
            adj_chem[name_to_idx[src], name_to_idx[tgt]] = w

    for a, b, w in gap_edges:
        if a in name_to_idx and b in name_to_idx:
            adj_gap[name_to_idx[a], name_to_idx[b]] = w
            adj_gap[name_to_idx[b], name_to_idx[a]] = w

    return {
        "neuron_names": names,
        "adjacency_chemical": adj_chem,
        "adjacency_gap": adj_gap,
        "adjacency": adj_chem + adj_gap,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_connectome(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the C. elegans connectome.

    If *path* is ``None``, return the built-in simplified connectome
    (~30 key neurons with published synapse weights).  Otherwise load
    from a CSV or NPZ file.

    Parameters
    ----------
    path : str or Path, optional
        Path to a CSV (columns: ``source, target, weight, type``) or
        NPZ file.

    Returns
    -------
    connectome : dict
        ``neuron_names``, ``adjacency``, ``adjacency_chemical``,
        ``adjacency_gap``.
    """
    if path is None:
        return _build_default_connectome()

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {
            "neuron_names": list(data["neuron_names"]),
            "adjacency_chemical": data["adjacency_chemical"],
            "adjacency_gap": data["adjacency_gap"],
            "adjacency": data["adjacency_chemical"] + data["adjacency_gap"],
        }

    if suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        names = sorted(set(df["source"].tolist() + df["target"].tolist()))
        n = len(names)
        name_to_idx = {name: i for i, name in enumerate(names)}

        adj_chem = np.zeros((n, n))
        adj_gap = np.zeros((n, n))

        for _, row in df.iterrows():
            i = name_to_idx[row["source"]]
            j = name_to_idx[row["target"]]
            w = float(row.get("weight", 1))
            syn_type = str(row.get("type", "chemical")).lower()
            if syn_type == "gap":
                adj_gap[i, j] = w
                adj_gap[j, i] = w
            else:
                adj_chem[i, j] = w

        return {
            "neuron_names": names,
            "adjacency_chemical": adj_chem,
            "adjacency_gap": adj_gap,
            "adjacency": adj_chem + adj_gap,
        }

    raise ValueError(f"Unsupported connectome file format: {suffix}")


def map_to_connectome(
    neuron_names: list[str],
    activity: NDArray,
    connectome: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map measured neural activity onto connectome nodes.

    Parameters
    ----------
    neuron_names : list of str
        Names of neurons with measured activity.
    activity : ndarray, shape ``(n_neurons,)`` or ``(n_neurons, n_t)``
        Activity values for the measured neurons.
    connectome : dict, optional
        Pre-loaded connectome.  If *None*, load the default.

    Returns
    -------
    result : dict
        ``mapped_activity`` (n_connectome, ...) with NaN for unmapped,
        ``connectome_nodes``, ``adjacency``, ``mapped_indices``,
        ``unmapped_neurons``.
    """
    if connectome is None:
        connectome = load_connectome()

    conn_names = connectome["neuron_names"]
    n_conn = len(conn_names)
    name_to_idx = {name: i for i, name in enumerate(conn_names)}

    act = np.asarray(activity, dtype=float)
    if act.ndim == 1:
        mapped = np.full(n_conn, np.nan)
    else:
        mapped = np.full((n_conn,) + act.shape[1:], np.nan)

    mapped_idx = []
    unmapped = []

    for k, name in enumerate(neuron_names):
        if name in name_to_idx:
            i = name_to_idx[name]
            if act.ndim == 1:
                mapped[i] = act[k]
            else:
                mapped[i] = act[k]
            mapped_idx.append(i)
        else:
            unmapped.append(name)

    return {
        "mapped_activity": mapped,
        "connectome_nodes": conn_names,
        "adjacency": connectome["adjacency"],
        "mapped_indices": np.array(mapped_idx, dtype=int),
        "unmapped_neurons": unmapped,
    }


def connectome_propagation(
    initial_activity: NDArray,
    adjacency: NDArray,
    *,
    n_steps: int = 5,
    decay: float = 0.5,
) -> NDArray:
    """Model activity propagation on the connectome graph.

    Uses a simple diffusion model:

    .. math::
        a_{t+1} = \\alpha\\, \\hat{A}\\, a_t + (1 - \\alpha)\\, a_0

    where :math:`\\hat{A}` is the row-normalised adjacency and
    :math:`\\alpha` is the *decay* parameter.

    Parameters
    ----------
    initial_activity : ndarray, shape ``(n_neurons,)``
    adjacency : ndarray, shape ``(n_neurons, n_neurons)``
    n_steps : int
        Number of propagation steps.
    decay : float
        Mixing parameter in [0, 1].

    Returns
    -------
    trajectory : ndarray, shape ``(n_steps + 1, n_neurons)``
        Activity at each step, with step 0 being the initial state.
    """
    a0 = np.asarray(initial_activity, dtype=float).ravel()
    A = np.asarray(adjacency, dtype=float)
    n = len(a0)

    # Row-normalise (handle zero-degree nodes)
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    A_norm = A / row_sums

    trajectory = np.zeros((n_steps + 1, n))
    trajectory[0] = a0

    for t in range(n_steps):
        trajectory[t + 1] = decay * (A_norm @ trajectory[t]) + (1.0 - decay) * a0

    return trajectory


def known_circuit(name: str) -> dict[str, Any]:
    """Return a subgraph for a well-known C. elegans circuit.

    Parameters
    ----------
    name : str
        Circuit name: ``"tap_withdrawal"``, ``"thermotaxis"``,
        ``"chemotaxis"``, or ``"nose_touch"``.

    Returns
    -------
    circuit : dict
        ``neurons`` (list of names), ``adjacency`` (sub-matrix),
        ``description``.
    """
    circuits: dict[str, dict[str, Any]] = {
        "tap_withdrawal": {
            "neurons": [
                "ALML", "ALMR", "AVM", "AVDL", "AVDR",
                "AVAL", "AVAR", "AVBL", "AVBR",
            ],
            "description": (
                "Tap withdrawal circuit: anterior mechanosensory neurons "
                "(ALM, AVM) activate backward command (AVA via AVD), "
                "producing a reversal response."
            ),
        },
        "thermotaxis": {
            "neurons": [
                "AFDL", "AFDR", "AIYL", "AIYR", "RIAL", "RIAR",
            ],
            "description": (
                "Thermotaxis circuit: thermosensory AFD neurons drive "
                "AIY interneurons, which modulate RIA for navigational "
                "turns toward the cultivation temperature."
            ),
        },
        "chemotaxis": {
            "neurons": [
                "ASEL", "ASER", "AWCL", "AWCR",
                "AIYL", "AIYR", "AIZL", "AIZR",
                "RIAL", "RIAR", "RIBL", "RIBR",
            ],
            "description": (
                "Chemotaxis circuit: ASE and AWC chemosensory neurons "
                "activate AIY/AIZ interneurons, which drive RIA/RIB "
                "for biased random walk toward attractants."
            ),
        },
        "nose_touch": {
            "neurons": [
                "ASHL", "ASHR", "AVDL", "AVDR", "AVAL", "AVAR",
            ],
            "description": (
                "Nose-touch avoidance: nociceptive ASH neurons activate "
                "command interneurons AVD and AVA to initiate backward "
                "locomotion."
            ),
        },
    }

    if name not in circuits:
        available = ", ".join(sorted(circuits.keys()))
        raise ValueError(
            f"Unknown circuit: {name!r}. Available: {available}"
        )

    info = circuits[name]
    neuron_list = info["neurons"]

    # Build sub-adjacency from the default connectome
    conn = _build_default_connectome()
    name_to_idx = {n: i for i, n in enumerate(conn["neuron_names"])}

    n_sub = len(neuron_list)
    sub_adj = np.zeros((n_sub, n_sub))
    for i, ni in enumerate(neuron_list):
        for j, nj in enumerate(neuron_list):
            if ni in name_to_idx and nj in name_to_idx:
                sub_adj[i, j] = conn["adjacency"][name_to_idx[ni], name_to_idx[nj]]

    return {
        "neurons": neuron_list,
        "adjacency": sub_adj,
        "description": info["description"],
    }
