import pandas as pd
import pytest

from src.config import ProjectConfig
from src.data_pipeline import load_csv_scenarios, scenarios_to_frame, split_scenarios, validate_csv_frame


def _valid_frame():
    return pd.DataFrame([
        {"scenario_id": 0, "user_id": 0, "x": 10.0, "y": 0.0, "distance": 10.0, "channel_gain": 1e-4, "min_rate_req_bps": 1e5},
        {"scenario_id": 0, "user_id": 1, "x": 20.0, "y": 0.0, "distance": 20.0, "channel_gain": 2e-4, "min_rate_req_bps": 1e5},
        {"scenario_id": 1, "user_id": 0, "x": 30.0, "y": 0.0, "distance": 30.0, "channel_gain": 1e-4, "min_rate_req_bps": 1e5},
        {"scenario_id": 1, "user_id": 1, "x": 40.0, "y": 0.0, "distance": 40.0, "channel_gain": 2e-4, "min_rate_req_bps": 1e5},
        {"scenario_id": 2, "user_id": 0, "x": 50.0, "y": 0.0, "distance": 50.0, "channel_gain": 1e-4, "min_rate_req_bps": 1e5},
        {"scenario_id": 2, "user_id": 1, "x": 60.0, "y": 0.0, "distance": 60.0, "channel_gain": 2e-4, "min_rate_req_bps": 1e5},
    ])


def test_csv_loader_and_features(tmp_path):
    path = tmp_path / "users.csv"
    _valid_frame().to_csv(path, index=False)
    cfg = ProjectConfig()
    cfg.network.num_users = 2
    scenarios = load_csv_scenarios(str(path), cfg)
    assert len(scenarios) == 3
    features = scenarios_to_frame(scenarios, cfg)
    assert {"rate_target_bps", "outage_target", "channel_gain_db"}.issubset(features.columns)


def test_csv_validation_errors(tmp_path):
    frame = _valid_frame().drop(columns=["channel_gain"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_csv_frame(frame)
    frame = _valid_frame()
    frame["distance"] = frame["distance"].astype(object)
    frame.loc[0, "distance"] = "bad"
    with pytest.raises(ValueError, match="non-numeric"):
        validate_csv_frame(frame)
    frame = _valid_frame().copy()
    frame.loc[1, "user_id"] = 0
    with pytest.raises(ValueError, match="duplicate"):
        validate_csv_frame(frame)


def test_scenario_split_is_disjoint():
    scenarios = [[object()] for _ in range(10)]
    train, validation, test = split_scenarios(scenarios, seed=42)
    assert len(train) + len(validation) + len(test) == 10
    assert set(map(id, train)).isdisjoint(map(id, validation))
    assert set(map(id, train)).isdisjoint(map(id, test))
