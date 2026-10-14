"""
Graph construction module for wireless network topologies.
Transforms user physical channels and spatial coordinates into node features,
edge features, and PyTorch graph tensors for GNN processing.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch
from src.channel_model import UserChannelState


@dataclass
class NetworkGraph:
    """Graph representation of the wireless network topology."""
    num_nodes: int
    node_features: torch.Tensor       # [N, d_node]
    edge_index: torch.Tensor          # [2, num_edges]
    edge_features: torch.Tensor       # [num_edges, d_edge]
    adjacency_matrix: torch.Tensor    # [N, N]
    pair_labels: Optional[torch.Tensor] = None # [N, N] binary pairing targets (if supervised)
    slot_labels: Optional[torch.Tensor] = None # [N] slot assignment targets


class GraphBuilder:
    """
    Constructs edge-attributed complete/k-NN graphs from user wireless channel states.
    Zero target leakage: only environmental channel & spatial features are encoded.
    """

    def __init__(self, max_cell_radius: float = 500.0):
        self.r_max = max_cell_radius

    def build_graph(
        self,
        users: List[UserChannelState],
        target_pair_matrix: Optional[np.ndarray] = None,
        target_slot_array: Optional[np.ndarray] = None,
    ) -> NetworkGraph:
        """
        Convert list of UserChannelState instances to NetworkGraph with normalized features.
        """
        n = len(users)
        coords = np.zeros((n, 2), dtype=float)
        distances = np.zeros(n, dtype=float)
        gains_lin = np.zeros(n, dtype=float)
        gains_db = np.zeros(n, dtype=float)
        req_rates = np.zeros(n, dtype=float)
        h_complex = np.zeros(n, dtype=complex)

        for i, u in enumerate(users):
            coords[i] = [u.x, u.y]
            distances[i] = u.distance
            gains_lin[i] = u.channel_gain
            gains_db[i] = u.channel_gain_db
            req_rates[i] = u.min_rate_req_bps
            h_complex[i] = u.channel_complex

        mean_gain = max(float(np.mean(gains_lin)), 1e-18)
        mean_db = float(np.mean(gains_db))
        std_db = max(float(np.std(gains_db)), 1.0)

        # 1. Node Features [N, 6]
        # [x/R, y/R, dist/R, normalized_gain_db, gain/mean_gain, min_rate/1e6]
        node_feats = np.zeros((n, 6), dtype=np.float32)
        node_feats[:, 0] = coords[:, 0] / self.r_max
        node_feats[:, 1] = coords[:, 1] / self.r_max
        node_feats[:, 2] = distances / self.r_max
        node_feats[:, 3] = (gains_db - mean_db) / std_db
        node_feats[:, 4] = gains_lin / mean_gain
        node_feats[:, 5] = req_rates / 1.0e6

        # 2. Complete Graph Edges (undirected without self-loops)
        edge_src: List[int] = []
        edge_dst: List[int] = []
        edge_attr: List[List[float]] = []

        adj = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                edge_src.append(i)
                edge_dst.append(j)
                adj[i, j] = 1.0

                # Spatial distance between users
                d_ij = np.linalg.norm(coords[i] - coords[j]) / (2.0 * self.r_max)

                # Channel gain disparity in dB: crucial indicator for PD-NOMA pairing
                gain_disparity = abs(gains_db[i] - gains_db[j]) / 40.0

                # Channel correlation: inner product of normalized channel vectors
                norm_i = np.abs(h_complex[i])
                norm_j = np.abs(h_complex[j])
                if norm_i > 1e-12 and norm_j > 1e-12:
                    corr = float(np.abs(np.conj(h_complex[i]) * h_complex[j]) / (norm_i * norm_j))
                else:
                    corr = 0.0

                # Combined interference potential
                interf_pot = float((gains_lin[i] + gains_lin[j]) / (2.0 * mean_gain))

                edge_attr.append([float(d_ij), float(gain_disparity), float(corr), interf_pot])

        edge_index_tensor = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
        node_feats_tensor = torch.tensor(node_feats, dtype=torch.float32)
        adj_tensor = torch.tensor(adj, dtype=torch.float32)

        pair_labels_t = None
        if target_pair_matrix is not None:
            pair_labels_t = torch.tensor(target_pair_matrix, dtype=torch.float32)

        slot_labels_t = None
        if target_slot_array is not None:
            slot_labels_t = torch.tensor(target_slot_array, dtype=torch.long)

        return NetworkGraph(
            num_nodes=n,
            node_features=node_feats_tensor,
            edge_index=edge_index_tensor,
            edge_features=edge_attr_tensor,
            adjacency_matrix=adj_tensor,
            pair_labels=pair_labels_t,
            slot_labels=slot_labels_t,
        )
