"""WIMP - Worm IMage Processor.

Time-domain NV-center magnetometry signal processing for C. elegans
neural imaging.  Supports Ramsey, Hahn echo, T1 relaxometry, and
dynamical-decoupling (CPMG / XY-8) pulse protocols.

Released under the Unlicense for iGEM 2026.
"""

__version__ = "0.1.0"

from wimp.constants import GAMMA_NV, D0, MU0
from wimp.pulses import (
    PulseSequence,
    ramsey_sequence,
    hahn_echo_sequence,
    cpmg_sequence,
    xy8_sequence,
    filter_function,
)
from wimp.relaxation import (
    fit_ramsey,
    fit_t2_decay,
    fit_t1_recovery,
    extract_field_ramsey,
    fit_ramsey_bayesian,
    fit_ramsey_joint,
)
from wimp.io import WIMPDataset, save_hdf5, load_hdf5, save_zarr, load_zarr
from wimp.zarr_stream import ZarrStreamWriter, ZarrStreamReader
from wimp.realtime import (
    RealtimeProcessor,
    RealtimeConfig,
    MeasurementFrame,
    KalmanFilter,
)
from wimp.registration import (
    DeformableRegistrationResult,
    deformable_register,
    tracks_to_frame_positions,
)
from wimp.source import resolution_matrix, crosstalk_metrics
from wimp.denoising import denoise
from wimp.analysis import (
    SortedSpike,
    sort_spikes,
    extract_event_waveforms,
    classify_waveforms,
    pca_decompose,
    ica_decompose,
    cross_correlation_matrix,
    granger_causality,
    mutual_information,
    transfer_entropy,
    transfer_entropy_matrix,
    spectral_coherence,
    coherence_matrix,
    permutation_test,
    cluster_permutation_test,
    fdr_correct,
)
from wimp.pipeline import batch_process_parallel
from wimp.stability import drift_correct, allan_variance, compressed_sensing_reconstruct
from wimp.connectome import (
    load_connectome,
    map_to_connectome,
    connectome_propagation,
    known_circuit,
    CONNECTOME_NEURONS,
)
