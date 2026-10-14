"""
Unit tests for PD-NOMA achievable rates, SIC decoding, and fairness index.
Includes analytical hand-calculated numerical verifications.
"""

import pytest
import numpy as np
from src.config import PDNOMAConfig
from src.channel_model import UserChannelState
from src.noma_pd import PDNOMAEngine, compute_jains_fairness_index


def test_hand_calculated_pd_noma_rates():
    """Verify PD-NOMA equations against exact manual analytical calculation."""
    cfg = PDNOMAConfig(alpha_weak=0.75, alpha_strong=0.25, sic_residual_factor=0.0)
    engine = PDNOMAEngine(cfg)

    # Dummy users with fixed channel gains
    u_weak = UserChannelState(
        user_id=0, x=300, y=0, distance=300, path_loss_db=0, path_loss_linear=1,
        fading_complex=1+0j, channel_complex=1+0j, channel_gain=1.0e-4, channel_gain_db=-40,
        snr_linear=100, snr_db=20, min_rate_req_bps=1.0e5
    )
    u_strong = UserChannelState(
        user_id=1, x=50, y=0, distance=50, path_loss_db=0, path_loss_linear=1,
        fading_complex=1+0j, channel_complex=1+0j, channel_gain=1.0e-2, channel_gain_db=-20,
        snr_linear=10000, snr_db=40, min_rate_req_bps=1.0e5
    )

    power_w = 1.0
    bandwidth_hz = 1.0e6
    noise_power_w = 1.0e-6

    res = engine.compute_pair_rates(u_weak, u_strong, power_w, bandwidth_hz, noise_power_w)

    # Hand calculation:
    # SINR_w = (0.75 * 1.0 * 1e-4) / (0.25 * 1.0 * 1e-4 + 1e-6) = 7.5e-5 / 2.6e-5 = 75 / 26
    expected_sinr_w = 75.0 / 26.0
    expected_rate_w = bandwidth_hz * np.log2(1.0 + expected_sinr_w)

    # SINR_s (perfect SIC) = (0.25 * 1.0 * 1e-2) / (0 + 1e-6) = 2.5e-3 / 1e-6 = 2500
    expected_sinr_s = 2500.0
    expected_rate_s = bandwidth_hz * np.log2(1.0 + expected_sinr_s)

    assert np.isclose(res.sinr_weak, expected_sinr_w, rtol=1e-5)
    assert np.isclose(res.rate_weak_bps, expected_rate_w, rtol=1e-5)
    assert np.isclose(res.sinr_strong, expected_sinr_s, rtol=1e-5)
    assert np.isclose(res.rate_strong_bps, expected_rate_s, rtol=1e-5)
    assert res.sic_successful is True


def test_imperfect_sic_degradation():
    """Verify that imperfect SIC reduces the achievable rate of the strong user."""
    cfg_perfect = PDNOMAConfig(alpha_weak=0.75, alpha_strong=0.25, sic_residual_factor=0.0)
    cfg_imperfect = PDNOMAConfig(alpha_weak=0.75, alpha_strong=0.25, sic_residual_factor=0.05)

    eng_perf = PDNOMAEngine(cfg_perfect)
    eng_imp = PDNOMAEngine(cfg_imperfect)

    u_weak = UserChannelState(
        user_id=0, x=300, y=0, distance=300, path_loss_db=0, path_loss_linear=1,
        fading_complex=1+0j, channel_complex=1+0j, channel_gain=1.0e-4, channel_gain_db=-40,
        snr_linear=100, snr_db=20, min_rate_req_bps=1.0e5
    )
    u_strong = UserChannelState(
        user_id=1, x=50, y=0, distance=50, path_loss_db=0, path_loss_linear=1,
        fading_complex=1+0j, channel_complex=1+0j, channel_gain=1.0e-2, channel_gain_db=-20,
        snr_linear=10000, snr_db=40, min_rate_req_bps=1.0e5
    )

    p, b, n0 = 1.0, 1.0e6, 1.0e-6
    res_perf = eng_perf.compute_pair_rates(u_weak, u_strong, p, b, n0)
    res_imp = eng_imp.compute_pair_rates(u_weak, u_strong, p, b, n0)

    # Weak user rate remains identical (SIC cancellation happens at strong user)
    assert np.isclose(res_perf.rate_weak_bps, res_imp.rate_weak_bps)
    # Strong user rate must strictly degrade under imperfect SIC
    assert res_imp.rate_strong_bps < res_perf.rate_strong_bps


def test_jains_fairness_properties():
    # Identical rates -> Fairness = 1.0
    assert np.isclose(compute_jains_fairness_index([10.0, 10.0, 10.0, 10.0]), 1.0)
    # Extreme inequality: 1 user gets everything, (N-1) get 0 -> Fairness = 1/N
    assert np.isclose(compute_jains_fairness_index([100.0, 0.0, 0.0, 0.0]), 0.25)
    # Empty list -> 0.0
    assert compute_jains_fairness_index([]) == 0.0
