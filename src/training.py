"""
Dataset generation and GNN training module.
Handles synthetic network topology generation, supervised label extraction,
mini-batch training, validation, early stopping, and loss history logging.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import ProjectConfig
from src.channel_model import WirelessNetwork, UserChannelState
from src.graph_builder import GraphBuilder, NetworkGraph
from src.noma_pd import PDNOMAEngine
from src.slot_assignment import SlotAssignmentEngine
from src.optimization import OptimizationOracle
from src.gnn_model import HybridNOMAGNN


@dataclass
class TrainingHistory:
    """Logs training metrics across epochs."""
    train_losses: List[float]
    val_losses: List[float]
    best_val_loss: float
    epochs_trained: int


class DatasetGenerator:
    """Generates synthetic channel graphs with oracle supervision labels."""

    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg
        self.network = WirelessNetwork(cfg.network, cfg.channel, seed=cfg.system.seed)
        self.pd_engine = PDNOMAEngine(cfg.noma_pd)
        self.slot_engine = SlotAssignmentEngine(
            num_slots=cfg.network.num_slots,
            total_bandwidth_hz=cfg.network.total_bandwidth,
            total_power_w=cfg.network.total_transmit_power_w,
            noise_power_per_slot=self.network.noise_power_per_slot,
            pd_engine=self.pd_engine,
            gain_ratio_threshold=cfg.noma_pd.gain_ratio_threshold,
        )
        self.oracle = OptimizationOracle(
            slot_engine=self.slot_engine,
            pd_engine=self.pd_engine,
            fairness_weight=cfg.optimization.fairness_weight,
        )
        self.graph_builder = GraphBuilder(max_cell_radius=cfg.network.cell_radius_max)

    def generate_split(self, num_samples: int, seed_offset: int = 0) -> List[NetworkGraph]:
        """Generate a list of network graph instances with supervised oracle labels."""
        graphs: List[NetworkGraph] = []
        base_seed = self.cfg.system.seed + seed_offset

        for i in range(num_samples):
            self.network.set_seed(base_seed + i)
            users = self.network.generate_users(
                num_users=self.cfg.network.num_users,
                min_rate_bps=self.cfg.optimization.min_rate_threshold_bps,
            )

            # Generate oracle labels
            pair_target, slot_target, _ = self.oracle.generate_labels(
                users, method=self.cfg.optimization.oracle_method
            )

            graph = self.graph_builder.build_graph(
                users=users,
                target_pair_matrix=pair_target,
                target_slot_array=slot_target,
            )
            graphs.append(graph)

        return graphs


class GNNTrainer:
    """Manages model optimization, loss tracking, and checkpoint saving."""

    def __init__(self, model: HybridNOMAGNN, cfg: ProjectConfig, device: str = "cpu"):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = Adam(
            self.model.parameters(),
            lr=cfg.gnn.learning_rate,
            weight_decay=cfg.gnn.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5)

    def compute_loss(self, graph: NetworkGraph) -> torch.Tensor:
        """
        Compute joint loss for pairing matrix and slot assignment:
        L = BCEWithLogits(Pair_logits, Pair_target) + 0.3 * CrossEntropy(Slot_logits, Slot_target)
        """
        node_feats = graph.node_features.to(self.device)
        edge_index = graph.edge_index.to(self.device)
        edge_feats = graph.edge_features.to(self.device)

        pair_logits, slot_logits = self.model(node_feats, edge_index, edge_feats)

        assert graph.pair_labels is not None
        assert graph.slot_labels is not None
        pair_targets = graph.pair_labels.to(self.device)
        slot_targets = graph.slot_labels.to(self.device)

        n = graph.num_nodes
        # Non-diagonal mask to avoid self-pair penalty
        mask = ~torch.eye(n, dtype=torch.bool, device=self.device)

        # Class imbalance weighting: positive pairs are sparse (N pairs vs N*(N-1) entries)
        pos_weight = torch.tensor([float(n - 2)], device=self.device)
        bce_loss = F.binary_cross_entropy_with_logits(
            pair_logits[mask],
            pair_targets[mask],
            pos_weight=pos_weight,
        )

        slot_loss = F.cross_entropy(slot_logits, slot_targets)

        total_loss = bce_loss + 0.3 * slot_loss
        return total_loss

    def train_epoch(self, train_graphs: List[NetworkGraph]) -> float:
        self.model.train()
        total_loss = 0.0

        # Shuffle graphs
        indices = np.random.permutation(len(train_graphs))
        for idx in indices:
            g = train_graphs[idx]
            self.optimizer.zero_grad()
            loss = self.compute_loss(g)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += float(loss.item())

        return total_loss / len(train_graphs) if train_graphs else 0.0

    def evaluate_loss(self, val_graphs: List[NetworkGraph]) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for g in val_graphs:
                loss = self.compute_loss(g)
                total_loss += float(loss.item())
        return total_loss / len(val_graphs) if val_graphs else 0.0

    def train(
        self,
        train_graphs: List[NetworkGraph],
        val_graphs: List[NetworkGraph],
        epochs: Optional[int] = None,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> TrainingHistory:
        """Run full training and validation loop."""
        num_epochs = epochs if epochs is not None else self.cfg.gnn.epochs
        best_val = float("inf")
        best_state = None

        train_losses: List[float] = []
        val_losses: List[float] = []

        for ep in range(1, num_epochs + 1):
            t_loss = self.train_epoch(train_graphs)
            v_loss = self.evaluate_loss(val_graphs)
            self.scheduler.step(v_loss)

            train_losses.append(t_loss)
            val_losses.append(v_loss)

            if v_loss < best_val:
                best_val = v_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if verbose and (ep % 5 == 0 or ep == num_epochs):
                print(f"Epoch {ep:03d}/{num_epochs:03d} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")

        # Restore best model checkpoint
        if best_state is not None:
            self.model.load_state_dict(best_state)
            if save_path is not None:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_path)

        return TrainingHistory(
            train_losses=train_losses,
            val_losses=val_losses,
            best_val_loss=best_val,
            epochs_trained=num_epochs,
        )
