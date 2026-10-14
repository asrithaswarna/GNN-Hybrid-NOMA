"""
Configuration dataclasses and YAML loader with strict validation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class SystemConfig:
    seed: int = 42
    device: str = "cpu"
    output_dir: str = "results"


@dataclass
class NetworkConfig:
    num_users: int = 10
    cell_radius_min: float = 20.0       # meters
    cell_radius_max: float = 500.0      # meters
    carrier_frequency: float = 2.4e9    # Hz
    total_bandwidth: float = 10.0e6     # Hz (10 MHz)
    num_slots: int = 5                  # Resource slots
    total_transmit_power_w: float = 1.0 # Watts (30 dBm)
    noise_spectral_density_dbm: float = -174.0 # dBm/Hz
    noise_figure_db: float = 5.0        # dB

    def validate(self) -> None:
        if self.num_users < 2:
            raise ValueError(f"num_users must be >= 2, got {self.num_users}")
        if self.cell_radius_min <= 0 or self.cell_radius_min >= self.cell_radius_max:
            raise ValueError(
                f"Invalid cell radius: min ({self.cell_radius_min}) must be > 0 and < max ({self.cell_radius_max})"
            )
        if self.total_bandwidth <= 0:
            raise ValueError(f"total_bandwidth must be > 0, got {self.total_bandwidth}")
        if self.num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {self.num_slots}")
        if self.total_transmit_power_w <= 0:
            raise ValueError(f"total_transmit_power_w must be > 0, got {self.total_transmit_power_w}")


@dataclass
class ChannelConfig:
    path_loss_exponent: float = 3.5
    reference_distance: float = 1.0     # meters
    reference_path_loss_db: float = 38.4 # dB at reference_distance
    fading_type: str = "rayleigh"
    fading_variance: float = 1.0

    def validate(self) -> None:
        if self.path_loss_exponent < 2.0 or self.path_loss_exponent > 6.0:
            raise ValueError(f"Path loss exponent {self.path_loss_exponent} outside standard range [2.0, 6.0]")
        if self.reference_distance <= 0:
            raise ValueError("reference_distance must be > 0")


@dataclass
class PDNOMAConfig:
    alpha_weak: float = 0.75
    alpha_strong: float = 0.25
    sic_residual_factor: float = 0.01   # Imperfect SIC residual factor beta
    gain_ratio_threshold: float = 1.5   # |h_strong|^2 / |h_weak|^2 threshold

    def validate(self) -> None:
        if self.alpha_weak <= 0 or self.alpha_strong <= 0:
            raise ValueError("Power allocation factors must be strictly positive")
        if abs((self.alpha_weak + self.alpha_strong) - 1.0) > 1e-5:
            raise ValueError(f"alpha_weak + alpha_strong must sum to 1.0, got {self.alpha_weak + self.alpha_strong}")
        if self.alpha_weak <= self.alpha_strong:
            raise ValueError("Weak user must receive more power than strong user (alpha_weak > alpha_strong)")
        if self.sic_residual_factor < 0.0 or self.sic_residual_factor > 1.0:
            raise ValueError(f"sic_residual_factor must be in [0, 1], got {self.sic_residual_factor}")


@dataclass
class SCMAConfig:
    num_layers: int = 6                # J
    num_resources: int = 4             # K
    codebook_size: int = 4             # M
    mpa_iterations: int = 4
    degrees_per_user: int = 2          # d_v
    degrees_per_resource: int = 3      # d_f

    def validate(self) -> None:
        if self.num_layers <= self.num_resources:
            raise ValueError(f"SCMA requires overloading: num_layers ({self.num_layers}) > num_resources ({self.num_resources})")
        if self.mpa_iterations < 1:
            raise ValueError("mpa_iterations must be at least 1")


@dataclass
class OptimizationConfig:
    oracle_method: str = "exhaustive"  # "exhaustive", "hungarian", or "greedy"
    fairness_weight: float = 0.2
    min_rate_threshold_bps: float = 1.0e5

    def validate(self) -> None:
        if self.oracle_method not in ["exhaustive", "hungarian", "greedy"]:
            raise ValueError(f"Unknown oracle_method: {self.oracle_method}")
        if self.fairness_weight < 0:
            raise ValueError("fairness_weight must be non-negative")


@dataclass
class GNNConfig:
    node_in_features: int = 6
    edge_in_features: int = 4
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 1.0e-4
    epochs: int = 50
    batch_size: int = 16

    def validate(self) -> None:
        if self.hidden_dim < 8:
            raise ValueError("hidden_dim must be at least 8")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class DatasetConfig:
    num_train_samples: int = 200
    num_val_samples: int = 40
    num_test_samples: int = 50

    def validate(self) -> None:
        if self.num_train_samples < 1 or self.num_val_samples < 1 or self.num_test_samples < 1:
            raise ValueError("Dataset sample counts must be >= 1")


@dataclass
class ProjectConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    noma_pd: PDNOMAConfig = field(default_factory=PDNOMAConfig)
    noma_scma: SCMAConfig = field(default_factory=SCMAConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def validate(self) -> None:
        self.network.validate()
        self.channel.validate()
        self.noma_pd.validate()
        self.noma_scma.validate()
        self.optimization.validate()
        self.gnn.validate()
        self.dataset.validate()


def load_config(config_path: Optional[str] = None) -> ProjectConfig:
    """
    Load YAML configuration and construct validated ProjectConfig.
    Falls back to default parameters if file does not exist.
    """
    if config_path is None:
        config_path = "config.yaml"

    p = Path(config_path)
    if not p.exists():
        cfg = ProjectConfig()
        cfg.validate()
        return cfg

    with open(p, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    def _float(val, default):
        return float(val) if val is not None else default

    def _int(val, default):
        return int(val) if val is not None else default

    net_d = data.get("network", {})
    chan_d = data.get("channel", {})
    pd_d = data.get("noma_pd", {})
    scma_d = data.get("noma_scma", {})
    opt_d = data.get("optimization", {})
    gnn_d = data.get("gnn", {})
    ds_d = data.get("dataset", {})

    cfg = ProjectConfig(
        system=SystemConfig(
            seed=_int(data.get("system", {}).get("seed"), 42),
            device=str(data.get("system", {}).get("device", "cpu")),
            output_dir=str(data.get("system", {}).get("output_dir", "results")),
        ),
        network=NetworkConfig(
            num_users=_int(net_d.get("num_users"), 10),
            cell_radius_min=_float(net_d.get("cell_radius_min"), 20.0),
            cell_radius_max=_float(net_d.get("cell_radius_max"), 500.0),
            carrier_frequency=_float(net_d.get("carrier_frequency"), 2.4e9),
            total_bandwidth=_float(net_d.get("total_bandwidth"), 10.0e6),
            num_slots=_int(net_d.get("num_slots"), 5),
            total_transmit_power_w=_float(net_d.get("total_transmit_power_w"), 1.0),
            noise_spectral_density_dbm=_float(net_d.get("noise_spectral_density_dbm"), -174.0),
            noise_figure_db=_float(net_d.get("noise_figure_db"), 5.0),
        ),
        channel=ChannelConfig(
            path_loss_exponent=_float(chan_d.get("path_loss_exponent"), 3.5),
            reference_distance=_float(chan_d.get("reference_distance"), 1.0),
            reference_path_loss_db=_float(chan_d.get("reference_path_loss_db"), 38.4),
            fading_type=str(chan_d.get("fading_type", "rayleigh")),
            fading_variance=_float(chan_d.get("fading_variance"), 1.0),
        ),
        noma_pd=PDNOMAConfig(
            alpha_weak=_float(pd_d.get("alpha_weak"), 0.75),
            alpha_strong=_float(pd_d.get("alpha_strong"), 0.25),
            sic_residual_factor=_float(pd_d.get("sic_residual_factor"), 0.01),
            gain_ratio_threshold=_float(pd_d.get("gain_ratio_threshold"), 1.5),
        ),
        noma_scma=SCMAConfig(
            num_layers=_int(scma_d.get("num_layers"), 6),
            num_resources=_int(scma_d.get("num_resources"), 4),
            codebook_size=_int(scma_d.get("codebook_size"), 4),
            mpa_iterations=_int(scma_d.get("mpa_iterations"), 4),
            degrees_per_user=_int(scma_d.get("degrees_per_user"), 2),
            degrees_per_resource=_int(scma_d.get("degrees_per_resource"), 3),
        ),
        optimization=OptimizationConfig(
            oracle_method=str(opt_d.get("oracle_method", "exhaustive")),
            fairness_weight=_float(opt_d.get("fairness_weight"), 0.2),
            min_rate_threshold_bps=_float(opt_d.get("min_rate_threshold_bps"), 1.0e5),
        ),
        gnn=GNNConfig(
            node_in_features=_int(gnn_d.get("node_in_features"), 6),
            edge_in_features=_int(gnn_d.get("edge_in_features"), 4),
            hidden_dim=_int(gnn_d.get("hidden_dim"), 64),
            num_layers=_int(gnn_d.get("num_layers"), 3),
            dropout=_float(gnn_d.get("dropout"), 0.1),
            learning_rate=_float(gnn_d.get("learning_rate"), 0.001),
            weight_decay=_float(gnn_d.get("weight_decay"), 1.0e-4),
            epochs=_int(gnn_d.get("epochs"), 50),
            batch_size=_int(gnn_d.get("batch_size"), 16),
        ),
        dataset=DatasetConfig(
            num_train_samples=_int(ds_d.get("num_train_samples"), 200),
            num_val_samples=_int(ds_d.get("num_val_samples"), 40),
            num_test_samples=_int(ds_d.get("num_test_samples"), 50),
        ),
    )
    cfg.validate()
    return cfg
