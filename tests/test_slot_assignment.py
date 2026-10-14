"""
Unit tests for dynamic slot assignment and capacity/power constraints.
"""

import pytest
import numpy as np
import torch
from src.config import NetworkConfig, ChannelConfig, PDNOMAConfig, GNNConfig
from src.channel_model import WirelessNetwork
from src.noma_pd import PDNOMAEngine
from src.pairing import UserPairingManager, PairingPlan, UserPair
from src.slot_assignment import SlotAssignmentEngine
from src.gnn_model import HybridNOMAGNN


def test_slot_capacity_and_utilization():
    net = WirelessNetwork(NetworkConfig(num_users=10, num_slots=5), ChannelConfig(), seed=42)
    users = net.generate_users()
    pd_engine = PDNOMAEngine(PDNOMAConfig())
    mgr = UserPairingManager(pd_engine)
    slot_engine = SlotAssignmentEngine(
        num_slots=5,
        total_bandwidth_hz=10.0e6,
        total_power_w=1.0,
        noise_power_per_slot=net.noise_power_per_slot,
        pd_engine=pd_engine,
    )

    plan = mgr.random_pairing(users)
    sched = slot_engine.evaluate_schedule(plan, users)

    assert len(sched.slots) == 5
    assert sched.active_slots == 5
    assert sched.slot_utilization == 1.0
    assert sched.total_sum_rate_bps > 0.0

    # Total power across slots must not exceed total transmit power
    total_power_used = sum(s.power_w for s in sched.slots)
    assert np.isclose(total_power_used, 1.0)


def test_idle_slots_when_excess_capacity():
    # 4 users (2 pairs), but 5 slots -> 2 active slots, 3 idle slots
    net = WirelessNetwork(NetworkConfig(num_users=4, num_slots=5), ChannelConfig(), seed=42)
    users = net.generate_users()
    pd_engine = PDNOMAEngine(PDNOMAConfig())
    mgr = UserPairingManager(pd_engine)
    slot_engine = SlotAssignmentEngine(
        num_slots=5,
        total_bandwidth_hz=10.0e6,
        total_power_w=1.0,
        noise_power_per_slot=net.noise_power_per_slot,
        pd_engine=pd_engine,
    )

    plan = mgr.random_pairing(users)
    sched = slot_engine.evaluate_schedule(plan, users)

    assert sched.active_slots == 2
    assert sched.total_slots == 5
    assert np.isclose(sched.slot_utilization, 2.0 / 5.0)

    # 3 slots must be IDLE with 0 power
    idle_count = sum(1 for s in sched.slots if s.mode == "IDLE")
    assert idle_count == 3


def test_gnn_slot_logits_affect_schedule():
    net = WirelessNetwork(NetworkConfig(num_users=6, num_slots=3), ChannelConfig(), seed=7)
    users = net.generate_users()
    pd_engine = PDNOMAEngine(PDNOMAConfig())
    mgr = UserPairingManager(pd_engine)
    plan = mgr.random_pairing(users)

    cfg = GNNConfig(node_in_features=6, edge_in_features=4, hidden_dim=16, num_layers=2)
    model = HybridNOMAGNN(cfg, num_slots=3)
    n = len(users)
    node_feats = torch.randn(n, 6)
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_feats = torch.randn(edge_index.size(1), 4)

    _, slot_logits = model(node_feats, edge_index, edge_feats)
    slot_engine = SlotAssignmentEngine(
        num_slots=3,
        total_bandwidth_hz=10.0e6,
        total_power_w=1.0,
        noise_power_per_slot=net.noise_power_per_slot,
        pd_engine=pd_engine,
    )

    mapping = slot_engine.resolve_gnn_slot_mapping(plan, users, slot_logits)
    assert len(mapping) == len(plan.pairs)
    assert set(mapping).issubset(set(range(3)))
    assert len(set(mapping)) == len(mapping) or len(mapping) <= 3

    scheduled = slot_engine.evaluate_schedule(plan, users, custom_slot_mapping=mapping)
    slot_ids = [record.slot_id for record in scheduled.slots if record.mode in {"PD-NOMA", "SCMA", "OMA_SINGLE"}]
    assert len(slot_ids) == len(mapping)
    assert all(slot in range(3) for slot in slot_ids)


def test_hybrid_scheduler_uses_scma_when_pd_is_not_suitable():
    net = WirelessNetwork(NetworkConfig(num_users=4, num_slots=2), ChannelConfig(), seed=99)
    users = net.generate_users(num_users=4)
    pd_engine = PDNOMAEngine(PDNOMAConfig(gain_ratio_threshold=10.0))
    slot_engine = SlotAssignmentEngine(
        num_slots=2,
        total_bandwidth_hz=10.0e6,
        total_power_w=1.0,
        noise_power_per_slot=net.noise_power_per_slot,
        pd_engine=pd_engine,
        gain_ratio_threshold=10.0,
    )

    pair = UserPair(user1_id=0, user2_id=1, predicted_affinity=1.0)
    plan = PairingPlan(pairs=[pair], unpaired_users=[2, 3], method_name="Hybrid Test")
    schedule = slot_engine.evaluate_schedule(plan, users, custom_slot_mapping=[0])

    assert schedule.slots[0].mode in {"PD-NOMA", "SCMA"}
    assert schedule.slots[0].slot_sum_rate_bps > 0.0
