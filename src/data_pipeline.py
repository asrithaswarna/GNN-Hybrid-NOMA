"""Reusable data loading, validation, feature engineering, and splitting."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.channel_model import UserChannelState, WirelessNetwork
from src.config import ProjectConfig
from src.graph_builder import GraphBuilder, NetworkGraph
from src.noma_pd import PDNOMAEngine
from src.optimization import OptimizationOracle
from src.slot_assignment import SlotAssignmentEngine

CSV_REQUIRED_COLUMNS = {
    "scenario_id", "user_id", "x", "y", "distance", "channel_gain", "min_rate_req_bps"
}
FEATURE_COLUMNS = [
    "x", "y", "distance", "channel_gain", "channel_gain_db", "snr_db", "min_rate_req_bps"
]
TARGET_COLUMNS = ["rate_target_bps", "outage_target"]


@dataclass
class ScenarioSplits:
    train: List[List[UserChannelState]]
    validation: List[List[UserChannelState]]
    test: List[List[UserChannelState]]
    source: str
    dataset_path: str = ""


def validate_csv_frame(frame: pd.DataFrame, expected_users: Optional[int] = None) -> None:
    missing = sorted(CSV_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("CSV must contain at least one user row")
    numeric_columns = sorted(CSV_REQUIRED_COLUMNS - {"scenario_id"})
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"CSV column '{column}' contains missing or non-numeric values")
    if (pd.to_numeric(frame["user_id"], errors="coerce") < 0).any():
        raise ValueError("user_id values must be non-negative")
    if (pd.to_numeric(frame["distance"], errors="coerce") <= 0).any():
        raise ValueError("distance values must be positive and measured in meters")
    if (pd.to_numeric(frame["channel_gain"], errors="coerce") < 0).any():
        raise ValueError("channel_gain values must be non-negative linear power gains")
    duplicate_mask = frame.duplicated(["scenario_id", "user_id"], keep=False)
    if duplicate_mask.any():
        raise ValueError("CSV contains duplicate user_id values within a scenario")
    counts = frame.groupby("scenario_id")["user_id"].nunique()
    if expected_users is not None and (counts != expected_users).any():
        raise ValueError(f"Each scenario must contain exactly {expected_users} users")
    if expected_users is not None:
        expected_ids = set(range(expected_users))
        for scenario_id, group in frame.groupby("scenario_id"):
            actual_ids = set(pd.to_numeric(group["user_id"], errors="raise").astype(int))
            if actual_ids != expected_ids:
                raise ValueError(
                    f"Scenario {scenario_id} user_id values must be exactly 0 through {expected_users - 1}"
                )


def _frame_to_scenarios(frame: pd.DataFrame, cfg: ProjectConfig) -> List[List[UserChannelState]]:
    scenarios: List[List[UserChannelState]] = []
    noise_power = 10.0 ** ((cfg.network.noise_spectral_density_dbm - 30.0) / 10.0)
    noise_power *= 10.0 ** (cfg.network.noise_figure_db / 10.0)
    noise_power *= cfg.network.total_bandwidth / cfg.network.num_slots
    slot_power = cfg.network.total_transmit_power_w / cfg.network.num_slots
    for _, group in frame.groupby("scenario_id", sort=True):
        users: List[UserChannelState] = []
        for row in group.sort_values("user_id").itertuples(index=False):
            gain = float(row.channel_gain)
            gain_db = 10.0 * np.log10(gain) if gain > 1e-30 else -300.0
            snr_linear = slot_power * gain / noise_power if noise_power > 0 else 0.0
            users.append(UserChannelState(
                user_id=int(row.user_id), x=float(row.x), y=float(row.y),
                distance=float(row.distance), path_loss_db=float(getattr(row, "path_loss_db", 0.0)),
                path_loss_linear=float(getattr(row, "path_loss_linear", 1.0)),
                fading_complex=complex(getattr(row, "fading_real", np.sqrt(gain)), getattr(row, "fading_imag", 0.0)),
                channel_complex=complex(np.sqrt(gain), 0.0), channel_gain=gain,
                channel_gain_db=gain_db, snr_linear=float(snr_linear),
                snr_db=float(10.0 * np.log10(snr_linear)) if snr_linear > 1e-30 else -300.0,
                min_rate_req_bps=float(row.min_rate_req_bps),
            ))
        scenarios.append(users)
    return scenarios


def load_csv_scenarios(path: str, cfg: ProjectConfig) -> List[List[UserChannelState]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV does not exist: {csv_path}")
    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"Could not read dataset CSV '{csv_path}': {exc}") from exc
    validate_csv_frame(frame, expected_users=cfg.network.num_users)
    return _frame_to_scenarios(frame, cfg)


def generate_synthetic_scenarios(cfg: ProjectConfig, num_samples: int, seed_offset: int = 2000) -> List[List[UserChannelState]]:
    network = WirelessNetwork(cfg.network, cfg.channel, seed=cfg.system.seed + seed_offset)
    scenarios = []
    for index in range(num_samples):
        network.set_seed(cfg.system.seed + seed_offset + index)
        scenarios.append(network.generate_users(
            num_users=cfg.network.num_users,
            min_rate_bps=cfg.optimization.min_rate_threshold_bps,
        ))
    return scenarios


def split_scenarios(scenarios: Sequence[List[UserChannelState]], seed: int = 42,
                    train_fraction: float = 0.6, validation_fraction: float = 0.2) -> Tuple[list, list, list]:
    if not scenarios:
        raise ValueError("At least one scenario is required")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must be positive and leave a non-empty test split")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(scenarios))
    train_end = max(1, int(len(indices) * train_fraction))
    validation_end = max(train_end + 1, int(len(indices) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(indices) - 1) if len(indices) > 2 else len(indices)
    return ([scenarios[i] for i in indices[:train_end]],
            [scenarios[i] for i in indices[train_end:validation_end]],
            [scenarios[i] for i in indices[validation_end:]])


def scenarios_to_frame(scenarios: Sequence[List[UserChannelState]], cfg: ProjectConfig) -> pd.DataFrame:
    """Create user-level features and simulation-derived rate/outage targets."""
    rows: List[Dict[str, Union[float, int]]] = []
    slot_power = cfg.network.total_transmit_power_w / cfg.network.num_slots
    bandwidth = cfg.network.total_bandwidth / cfg.network.num_slots
    for scenario_id, users in enumerate(scenarios):
        for user in users:
            snr = slot_power * user.channel_gain
            noise = 10.0 ** ((cfg.network.noise_spectral_density_dbm - 30.0) / 10.0)
            noise *= 10.0 ** (cfg.network.noise_figure_db / 10.0)
            noise *= bandwidth
            snr = snr / noise if noise > 0 else 0.0
            rate = bandwidth * np.log2(1.0 + snr)
            rows.append({
                "scenario_id": scenario_id, "user_id": user.user_id,
                "x": user.x, "y": user.y, "distance": user.distance,
                "channel_gain": user.channel_gain, "channel_gain_db": user.channel_gain_db,
                "snr_db": 10.0 * np.log10(snr) if snr > 1e-30 else -300.0,
                "min_rate_req_bps": user.min_rate_req_bps,
                "rate_target_bps": float(rate),
                "outage_target": int(rate < user.min_rate_req_bps),
            })
    return pd.DataFrame(rows)


def scenarios_to_graphs(scenarios: Sequence[List[UserChannelState]], cfg: ProjectConfig) -> List[NetworkGraph]:
    """Convert validated user scenarios into supervised GNN samples."""
    if not scenarios:
        raise ValueError("At least one scenario is required to build GNN graphs")
    network = WirelessNetwork(cfg.network, cfg.channel, seed=cfg.system.seed)
    pd_engine = PDNOMAEngine(cfg.noma_pd)
    slot_engine = SlotAssignmentEngine(
        num_slots=cfg.network.num_slots,
        total_bandwidth_hz=cfg.network.total_bandwidth,
        total_power_w=cfg.network.total_transmit_power_w,
        noise_power_per_slot=network.noise_power_per_slot,
        pd_engine=pd_engine,
        gain_ratio_threshold=cfg.noma_pd.gain_ratio_threshold,
    )
    oracle = OptimizationOracle(slot_engine, pd_engine, cfg.optimization.fairness_weight)
    builder = GraphBuilder(cfg.network.cell_radius_max)
    graphs = []
    for users in scenarios:
        pair_labels, slot_labels, _ = oracle.generate_labels(users, cfg.optimization.oracle_method)
        graphs.append(builder.build_graph(users, pair_labels, slot_labels))
    return graphs


def load_or_generate_scenarios(cfg: ProjectConfig, data_source: str = "synthetic", dataset: Optional[str] = None,
                               num_samples: int = 30) -> ScenarioSplits:
    if data_source == "synthetic":
        scenarios = generate_synthetic_scenarios(cfg, num_samples=num_samples)
        source_path = ""
    elif data_source == "csv":
        if not dataset:
            raise ValueError("--dataset is required when --data-source csv is selected")
        scenarios = load_csv_scenarios(dataset, cfg)
        source_path = str(Path(dataset))
    else:
        raise ValueError("data_source must be 'synthetic' or 'csv'")
    train, validation, test = split_scenarios(scenarios, seed=cfg.system.seed)
    return ScenarioSplits(train, validation, test, data_source, source_path)
