import pandas as pd

from src.config import ProjectConfig
from src.data_pipeline import load_or_generate_scenarios, scenarios_to_graphs


def test_csv_scenarios_build_supervised_gnn_graphs(tmp_path):
    rows = []
    for scenario_id in range(3):
        for user_id in range(4):
            rows.append({
                "scenario_id": scenario_id,
                "user_id": user_id,
                "x": 20.0 + user_id,
                "y": float(user_id),
                "distance": 25.0 + user_id,
                "channel_gain": 1e-4 * (user_id + 1),
                "min_rate_req_bps": 1e5,
            })
    path = tmp_path / "scenario_users.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    cfg = ProjectConfig()
    cfg.network.num_users = 4
    splits = load_or_generate_scenarios(cfg, data_source="csv", dataset=str(path), num_samples=3)
    graphs = scenarios_to_graphs(splits.train, cfg)
    assert graphs
    assert graphs[0].pair_labels is not None
    assert graphs[0].slot_labels is not None
    assert graphs[0].num_nodes == 4
