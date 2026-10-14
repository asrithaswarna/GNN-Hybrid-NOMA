"""
Unit tests for SCMA factor graph, codebook generation, and Message Passing Algorithm (MPA).
"""

import pytest
import numpy as np
from src.config import SCMAConfig
from src.noma_scma import SCMAEngine


def test_scma_factor_graph_regularity():
    cfg = SCMAConfig(num_layers=6, num_resources=4, codebook_size=4, mpa_iterations=4)
    engine = SCMAEngine(cfg)

    # Factor graph indicator matrix shape [K, J] -> [4, 6]
    assert engine.F.shape == (4, 6)

    # Column regularity: each user connects to exactly d_v = 2 resources
    col_sums = np.sum(engine.F, axis=0)
    for s in col_sums:
        assert s == 2

    # Row regularity: each resource connects to exactly d_f = 3 users
    row_sums = np.sum(engine.F, axis=1)
    for s in row_sums:
        assert s == 3


def test_scma_codebook_sparsity():
    cfg = SCMAConfig(num_layers=6, num_resources=4, codebook_size=4)
    engine = SCMAEngine(cfg)

    # Shape: [J, M, K] -> [6, 4, 4]
    assert engine.codebooks.shape == (6, 4, 4)

    # Check sparsity pattern matches factor matrix F
    for j in range(6):
        for m in range(4):
            for k in range(4):
                if engine.F[k, j] == 0:
                    assert np.abs(engine.codebooks[j, m, k]) == 0.0
                else:
                    assert np.abs(engine.codebooks[j, m, k]) > 0.0


def test_mpa_detector_output_distribution():
    cfg = SCMAConfig(num_layers=6, num_resources=4, codebook_size=4, mpa_iterations=3)
    engine = SCMAEngine(cfg)

    # Simulated received vector y [4] and channel matrix H [4, 6]
    y = np.array([0.5 + 0.2j, -0.3 + 0.1j, 0.4 - 0.4j, 0.1 + 0.5j], dtype=complex)
    H = np.ones((4, 6), dtype=complex)

    posteriors = engine.mpa_detector(y, H, noise_variance=0.1)

    # Posteriors must have shape [J, M]
    assert posteriors.shape == (6, 4)

    # Probabilities for each user must sum to 1.0
    for j in range(6):
        assert np.isclose(np.sum(posteriors[j, :]), 1.0, atol=1e-5)
        # All probabilities must be non-negative
        assert np.all(posteriors[j, :] >= 0.0)


def test_scma_simulation_runs():
    cfg = SCMAConfig(num_layers=6, num_resources=4, codebook_size=4, mpa_iterations=2)
    engine = SCMAEngine(cfg)
    gains = np.ones(6)

    res = engine.simulate_transmission(channel_gains=gains, snr_db=10.0, num_blocks=10)
    assert res.num_users == 6
    assert res.num_resources == 4
    assert res.overload_ratio == 1.5
    assert 0.0 <= res.bit_error_rate <= 1.0
    assert 0.0 <= res.symbol_error_rate <= 1.0
    assert res.sum_rate_bps > 0.0
