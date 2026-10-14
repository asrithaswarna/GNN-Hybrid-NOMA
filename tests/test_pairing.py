"""
Unit tests for user pairing algorithms.
Verifies no duplicate users, proper odd-user handling, and plan validity.
"""

import pytest
import numpy as np
from src.config import NetworkConfig, ChannelConfig, PDNOMAConfig
from src.channel_model import WirelessNetwork
from src.noma_pd import PDNOMAEngine
from src.pairing import UserPairingManager


@pytest.fixture
def sample_network():
    net = WirelessNetwork(NetworkConfig(num_users=10), ChannelConfig(), seed=42)
    users = net.generate_users()
    pd_engine = PDNOMAEngine(PDNOMAConfig())
    mgr = UserPairingManager(pd_engine)
    return users, mgr


def test_even_user_pairing(sample_network):
    users, mgr = sample_network
    assert len(users) == 10

    # Random pairing
    p_rand = mgr.random_pairing(users)
    assert len(p_rand.pairs) == 5
    assert len(p_rand.unpaired_users) == 0
    assert p_rand.validate(10) is True

    # Near-far pairing
    p_nf = mgr.channel_gain_pairing(users, strategy="near_far")
    assert len(p_nf.pairs) == 5
    assert len(p_nf.unpaired_users) == 0
    assert p_nf.validate(10) is True

    # Greedy pairing
    p_greedy = mgr.greedy_sum_rate_pairing(users, power_w=1.0, bandwidth_hz=1e6, noise_power_w=1e-6)
    assert len(p_greedy.pairs) == 5
    assert len(p_greedy.unpaired_users) == 0
    assert p_greedy.validate(10) is True


def test_odd_user_pairing():
    net = WirelessNetwork(NetworkConfig(num_users=9), ChannelConfig(), seed=42)
    users = net.generate_users()
    pd_engine = PDNOMAEngine(PDNOMAConfig())
    mgr = UserPairingManager(pd_engine)

    assert len(users) == 9

    # Random pairing with 9 users -> 4 pairs, 1 unpaired
    p_rand = mgr.random_pairing(users)
    assert len(p_rand.pairs) == 4
    assert len(p_rand.unpaired_users) == 1
    assert p_rand.validate(9) is True

    # Near-far with 9 users
    p_nf = mgr.channel_gain_pairing(users, strategy="near_far")
    assert len(p_nf.pairs) == 4
    assert len(p_nf.unpaired_users) == 1
    assert p_nf.validate(9) is True

    # Greedy with 9 users
    p_greedy = mgr.greedy_sum_rate_pairing(users, power_w=1.0, bandwidth_hz=1e6, noise_power_w=1e-6)
    assert len(p_greedy.pairs) == 4
    assert len(p_greedy.unpaired_users) == 1
    assert p_greedy.validate(9) is True


def test_gnn_affinity_pairing(sample_network):
    users, mgr = sample_network
    n = len(users)

    # Simulated affinity matrix
    aff = np.random.uniform(0.1, 0.9, size=(n, n))
    plan = mgr.gnn_affinity_pairing(aff, users)

    assert len(plan.pairs) == 5
    assert len(plan.unpaired_users) == 0
    assert plan.validate(10) is True
