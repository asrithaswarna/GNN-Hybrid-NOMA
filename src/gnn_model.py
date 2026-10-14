"""
Edge-Aware Graph Neural Network (GNN) module for dynamic user pairing and slot assignment.
Implemented in native PyTorch for 100% portable execution on CPU/GPU without external wheel dependencies.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import GNNConfig


class EdgeConditionedConvLayer(nn.Module):
    """
    Message passing layer that conditions node updates on multidimensional edge attributes.
    m_{j -> i} = MLP_msg([h_j, e_{ij}])
    h_i^{next} = LayerNorm(h_i + MLP_upd([h_i, sum_j m_{j -> i}]))
    """

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, out_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.res_proj = nn.Linear(node_dim, out_dim) if node_dim != out_dim else nn.Identity()

    def forward(
        self,
        node_feats: torch.Tensor,     # [N, node_dim]
        edge_index: torch.Tensor,     # [2, E]
        edge_feats: torch.Tensor,     # [E, edge_dim]
    ) -> torch.Tensor:
        n = node_feats.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Form message input from source node and edge feature
        msg_input = torch.cat([node_feats[src], edge_feats], dim=-1) # [E, node_dim + edge_dim]
        messages = self.msg_mlp(msg_input)                           # [E, out_dim]

        # Aggregate messages at destination nodes (normalized sum)
        aggregated = torch.zeros((n, messages.size(-1)), device=node_feats.device, dtype=node_feats.dtype)
        aggregated.index_add_(0, dst, messages)

        # Degree normalization
        deg = torch.zeros((n, 1), device=node_feats.device, dtype=node_feats.dtype)
        ones = torch.ones((edge_index.size(1), 1), device=node_feats.device, dtype=node_feats.dtype)
        deg.index_add_(0, dst, ones)
        deg = torch.clamp(deg, min=1.0)
        aggregated = aggregated / deg

        # Update node features with residual connection
        upd_input = torch.cat([node_feats, aggregated], dim=-1)
        updated = self.update_mlp(upd_input)
        out = self.norm(self.res_proj(node_feats) + updated)
        return out


class HybridNOMAGNN(nn.Module):
    """
    Complete Graph Neural Network for Hybrid NOMA user pairing and slot assignment.
    Outputs:
    1. Symmetric pair affinity logits matrix [N, N].
    2. Per-user slot assignment logits [N, num_slots].
    """

    def __init__(self, cfg: GNNConfig, num_slots: int = 5):
        super().__init__()
        self.cfg = cfg
        self.num_slots = num_slots

        # Input projection
        self.node_embed = nn.Sequential(
            nn.Linear(cfg.node_in_features, cfg.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(cfg.hidden_dim),
        )

        # Stacked Edge-Conditioned Message Passing Layers
        self.conv_layers = nn.ModuleList([
            EdgeConditionedConvLayer(
                node_dim=cfg.hidden_dim,
                edge_dim=cfg.edge_in_features,
                out_dim=cfg.hidden_dim,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.num_layers)
        ])

        # Edge Affinity Scoring Head: Bilinear + MLP for pair affinity A[i, j]
        # Combines [h_i, h_j, |h_i - h_j|, h_i * h_j]
        pair_head_dim = cfg.hidden_dim * 4
        self.pair_head = nn.Sequential(
            nn.Linear(pair_head_dim, cfg.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

        # Slot Assignment Head: Classifies each user into slot preferences
        self.slot_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, num_slots),
        )

    def forward(
        self,
        node_feats: torch.Tensor,     # [N, node_in_dim]
        edge_index: torch.Tensor,     # [2, E]
        edge_feats: torch.Tensor,     # [E, edge_in_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            pair_affinity_logits: [N, N] symmetric matrix of pair affinity logits.
            slot_logits: [N, num_slots] slot assignment preference logits.
        """
        n = node_feats.size(0)

        # 1. Node embedding & graph message passing
        h = self.node_embed(node_feats)
        for conv in self.conv_layers:
            h = conv(h, edge_index, edge_feats)

        # 2. Pair affinity matrix computation
        # Broadcast pairs across all (i, j)
        h_i = h.unsqueeze(1).expand(-1, n, -1)   # [N, N, D]
        h_j = h.unsqueeze(0).expand(n, -1, -1)   # [N, N, D]
        diff = torch.abs(h_i - h_j)              # [N, N, D]
        prod = h_i * h_j                         # [N, N, D]

        pair_repr = torch.cat([h_i, h_j, diff, prod], dim=-1) # [N, N, 4*D]
        pair_logits = self.pair_head(pair_repr).squeeze(-1)    # [N, N]

        # Enforce symmetry: A_ij = A_ji, and zero self-loops
        sym_pair_logits = 0.5 * (pair_logits + pair_logits.t())
        sym_pair_logits = sym_pair_logits.masked_fill(torch.eye(n, dtype=torch.bool, device=h.device), -1e9)

        # 3. Slot classification logits
        slot_logits = self.slot_head(h)          # [N, num_slots]

        return sym_pair_logits, slot_logits

    def predict_pairing_probabilities(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Computes sigmoid pairing affinity probabilities in [0, 1]."""
        with torch.no_grad():
            pair_logits, _ = self.forward(node_feats, edge_index, edge_feats)
            probs = torch.sigmoid(pair_logits)
            probs.fill_diagonal_(0.0)
            return probs

    def predict_slot_logits(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Return the per-user slot-preference logits used by the scheduler."""
        with torch.no_grad():
            _, slot_logits = self.forward(node_feats, edge_index, edge_feats)
            return slot_logits
