from src.classical_models import (
    _build_cv_splits,
    compare_regressors,
    evaluate_classical_models,
    paired_bootstrap_mae_difference,
    predict_rate,
)
from src.config import ProjectConfig
from src.data_pipeline import generate_synthetic_scenarios, scenarios_to_frame
from src.data_pipeline import FEATURE_COLUMNS, TARGET_COLUMNS
from src.unsupervised import run_clustering


def test_classical_models_and_clustering(tmp_path):
    cfg = ProjectConfig()
    cfg.network.num_users = 4
    scenarios = generate_synthetic_scenarios(cfg, num_samples=12)
    frame = scenarios_to_frame(scenarios, cfg)
    train = frame[frame["scenario_id"] < 8]
    test = frame[frame["scenario_id"] >= 8]
    result = compare_regressors(train, test, str(tmp_path / "models"))
    assert set(result["model"]) == {"Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree", "Random Forest"}
    artifact = tmp_path / "models" / "random_forest.joblib"
    predictions = predict_rate(str(artifact), test)
    assert len(predictions) == len(test)
    clustering = run_clustering(frame, str(tmp_path / "clusters"))
    assert len(clustering["assignments"]) == len(frame)
    assert len(clustering["explained_variance"]) == 2


def test_scenario_grouped_cv_keeps_scenarios_out_of_both_sides():
    cfg = ProjectConfig()
    cfg.network.num_users = 4
    frame = scenarios_to_frame(generate_synthetic_scenarios(cfg, num_samples=6), cfg)
    x = frame[["x", "y", "distance", "channel_gain", "channel_gain_db", "snr_db", "min_rate_req_bps"]]
    y = frame["rate_target_bps"] / 1e6

    folds, strategy, group_count = _build_cv_splits(frame, x, y)

    assert strategy == "GroupKFold"
    assert group_count == 6
    for train_indices, validation_indices in folds:
        train_scenarios = set(frame.iloc[train_indices]["scenario_id"])
        validation_scenarios = set(frame.iloc[validation_indices]["scenario_id"])
        assert train_scenarios.isdisjoint(validation_scenarios)


def test_regression_targets_are_not_direct_input_features():
    assert set(FEATURE_COLUMNS).isdisjoint(TARGET_COLUMNS)
    assert "rate_target_bps" not in FEATURE_COLUMNS


def test_paired_bootstrap_resamples_whole_scenarios_reproducibly():
    y_true = [0.0, 0.0, 10.0, 10.0]
    baseline = [1.0, 2.0, 12.0, 13.0]
    selected = [0.0, 1.0, 10.0, 12.0]
    scenario_ids = [0, 0, 1, 1]

    first = paired_bootstrap_mae_difference(
        y_true, baseline, selected, scenario_ids, n_bootstrap=200, random_state=7
    )
    second = paired_bootstrap_mae_difference(
        y_true, baseline, selected, scenario_ids, n_bootstrap=200, random_state=7
    )

    assert first["observed_difference_mae_mbps"] == 1.25
    assert first["independent_test_units"] == 2
    assert first["resampling_unit"] == "scenario"
    assert first["ci_lower_mae_mbps"] == second["ci_lower_mae_mbps"]
    assert first["ci_upper_mae_mbps"] == second["ci_upper_mae_mbps"]


def test_module5_evaluation_writes_grouped_cv_tuning_and_test_outputs(tmp_path):
    cfg = ProjectConfig()
    cfg.network.num_users = 4
    train_frame = scenarios_to_frame(generate_synthetic_scenarios(cfg, 8, seed_offset=10), cfg)
    test_frame = scenarios_to_frame(generate_synthetic_scenarios(cfg, 3, seed_offset=100), cfg)

    result = evaluate_classical_models(
        train_frame, test_frame, str(tmp_path / "models"), seed=11, n_bootstrap=100
    )

    output = tmp_path / "models"
    assert result["status"]["status"] == "completed"
    assert result["status"]["cv_strategy"] == "GroupKFold"
    assert (output / "cross_validation_metrics.csv").exists()
    assert (output / "hyperparameter_search.csv").exists()
    assert (output / "hyperparameter_search_results.csv").exists()
    assert (output / "model_selection_test_metrics.csv").exists()
    assert (output / "paired_bootstrap_comparison.csv").exists()
    assert (output / "selected_model.joblib").exists()
    assert set(result["test_metrics"]["role"]) == {"bootstrap baseline", "selected model"}
    assert result["bootstrap"]["independent_test_units"] == 3
    assert len(predict_rate(str(output / "selected_model.joblib"), test_frame)) == len(test_frame)
