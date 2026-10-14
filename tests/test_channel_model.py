"""
Unit tests for wireless channel model.
"""

import pytest
import numpy as np
from src.config import NetworkConfig, ChannelConfig
from src.channel_model import WirelessNetwork, UserChannelState


def test_channel_geometry_and_bounds():
    net_cfg = NetworkConfig(num_users=20, cell_radius_min=20.0, cell_radius_max=500.0)
    chan_cfg = ChannelConfig()
    network = WirelessNetwork(net_cfg, chan_cfg, seed=42)
    users = network.generate_users()

    assert len(users) == 20
    for u in users:
        # User distance must be within annular region [R_min, R_max]
        assert net_cfg.cell_radius_min - 1e-5 <= u.distance <= net_cfg.cell_radius_max + 1e-5
        # Coordinates must match distance
        assert np.isclose(np.sqrt(u.x**2 + u.y**2), u.distance)
        # Channel gain must be strictly positive
        assert u.channel_gain > 0.0
        assert u.path_loss_linear > 0.0
        assert u.snr_linear > 0.0


def test_path_loss_monotonicity():
    chan_cfg = ChannelConfig(path_loss_exponent=3.5, reference_distance=1.0, reference_path_loss_db=38.4)
    network = WirelessNetwork(NetworkConfig(), chan_cfg)

    distances = np.array([50.0, 100.0, 200.0, 400.0])
    pl_db = network.calculate_path_loss_db(distances)

    # Path loss in dB must strictly increase with distance
    for i in range(len(distances) - 1):
        assert pl_db[i] < pl_db[i + 1]


def test_reproducibility_with_seed():
    net_cfg = NetworkConfig(num_users=10)
    chan_cfg = ChannelConfig()

    net1 = WirelessNetwork(net_cfg, chan_cfg, seed=123)
    users1 = net1.generate_users()

    net2 = WirelessNetwork(net_cfg, chan_cfg, seed=123)
    users2 = net2.generate_users()

    for u1, u2 in zip(users1, users2):
        assert np.isclose(u1.x, u2.x)
        assert np.isclose(u1.y, u2.y)
        assert np.isclose(u1.channel_gain, u2.channel_gain)
