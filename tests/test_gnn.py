"""
Unit tests for Graph Neural Network architecture, tensor shapes, and loss backward pass.
"""

import pytest
import torch
from src.config import GNNConfig
from src.channel_model import WirelessNetwork, NetworkConfig, ChannelConfig
from src.graph_builder import GraphBuilder
from src.gnn_model import HybridNOMAGNN


def test_gnn_forward_shapes():
    cfg = GNNConfig(node_in_features=6, edge_in_features=4, hidden_dim=32, num_layers=2)
    model = HybridNOMAGNN(cfg, num_slots=5)

    n = 8
    node_feats = torch.randn(n, 6)
    # Complete graph edges (n * (n - 1))
    src = []
    dst = []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_feats = torch.randn(edge_index.size(1), 4)

    pair_logits, slot_logits = model(node_feats, edge_index, edge_feats)

    # Check output shapes
    assert pair_logits.shape == (n, n)
    assert slot_logits.shape == (n, 5)

    # Check symmetry of pair affinity logits: A[i, j] == A[j, i]
    assert torch.allclose(pair_logits, pair_logits.t(), atol=1e-5)


def test_gnn_pairing_probabilities():
    cfg = GNNConfig(node_in_features=6, edge_in_features=4, hidden_dim=32, num_layers=2)
    model = HybridNOMAGNN(cfg, num_slots=5)

    n = 6
    node_feats = torch.randn(n, 6)
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_feats = torch.randn(edge_index.size(1), 4)

    probs = model.predict_pairing_probabilities(node_feats, edge_index, edge_feats)

    # Probabilities must be in [0, 1]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    # Diagonal must be 0
    assert torch.equal(torch.diag(probs), torch.zeros(n))
    # Probabilities must be symmetric
    assert torch.allclose(probs, probs.t(), atol=1e-5)


def test_gnn_gradient_flow():
    cfg = GNNConfig(node_in_features=6, edge_in_features=4, hidden_dim=32, num_layers=2)
    model = HybridNOMAGNN(cfg, num_slots=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n = 6
    node_feats = torch.randn(n, 6)
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_feats = torch.randn(edge_index.size(1), 4)

    pair_logits, slot_logits = model(node_feats, edge_index, edge_feats)
    target_pair = torch.randint(0, 2, (n, n)).float()
    target_slot = torch.randint(0, 4, (n,)).long()

    loss = torch.nn.functional.binary_cross_entropy_with_logits(pair_logits, target_pair) + \
           torch.nn.functional.cross_entropy(slot_logits, target_slot)

    optimizer.zero_grad()
    loss.backward()

    # Verify gradients computed for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"

    optimizer.step()
