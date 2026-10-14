"""Small classical ML comparison on the simulated user-rate target."""

import json
import warnings
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.data_pipeline import FEATURE_COLUMNS


def _model_pipeline(model) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])


def _regression_models(seed: int = 42) -> Dict[str, Pipeline]:
    return {
        "Linear Regression": _model_pipeline(LinearRegression()),
        "Ridge Regression": _model_pipeline(Ridge(alpha=1.0)),
        "Lasso Regression": _model_pipeline(Lasso(alpha=0.01, max_iter=5000, random_state=seed)),
        "Decision Tree": _model_pipeline(DecisionTreeRegressor(max_depth=6, random_state=seed)),
        "Random Forest": _model_pipeline(
            RandomForestRegressor(n_estimators=50, max_depth=8, random_state=seed, n_jobs=1)
        ),
    }


def _build_cv_splits(
    frame: pd.DataFrame,
    x: pd.DataFrame,
    y: pd.Series,
    max_splits: int = 5,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str, int]:
    if len(frame) < 2:
        warnings.warn("At least two training rows are required for cross-validation.", RuntimeWarning)
        return [], "unavailable", 0

    if "scenario_id" in frame.columns:
        if frame["scenario_id"].isna().any():
            raise ValueError("scenario_id cannot contain missing values for scenario-aware evaluation")
        groups = frame["scenario_id"].to_numpy()
        group_count = int(pd.Series(groups).nunique())
        if group_count < 2:
            warnings.warn(
                "Scenario-aware cross-validation requires at least two training scenarios; "
                "cross-validation and hyperparameter search were skipped.",
                RuntimeWarning,
            )
            return [], "unavailable: fewer than two scenarios", group_count
        splitter = GroupKFold(n_splits=min(max_splits, group_count))
        return list(splitter.split(x, y, groups)), "GroupKFold", group_count

    warnings.warn(
        "scenario_id is unavailable; using row-wise KFold, which cannot prevent related "
        "observations from crossing folds.",
        RuntimeWarning,
    )
    splitter = KFold(n_splits=min(max_splits, len(frame)), shuffle=True, random_state=seed)
    return list(splitter.split(x, y)), "KFold (scenario IDs unavailable)", len(frame)


def paired_bootstrap_mae_difference(
    y_true,
    baseline_predictions,
    selected_predictions,
    groups=None,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> Dict[str, object]:
    """Bootstrap baseline MAE minus selected-model MAE by independent scenario."""
    y_true = np.asarray(y_true, dtype=float)
    baseline_predictions = np.asarray(baseline_predictions, dtype=float)
    selected_predictions = np.asarray(selected_predictions, dtype=float)
    if not (len(y_true) == len(baseline_predictions) == len(selected_predictions)):
        raise ValueError("Targets and paired prediction arrays must have the same length")
    if len(y_true) == 0:
        raise ValueError("At least one test observation is required for bootstrap comparison")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")

    if groups is None:
        warnings.warn(
            "scenario_id is unavailable for bootstrap; treating each test row as independent "
            "may underestimate uncertainty.",
            RuntimeWarning,
        )
        groups = np.arange(len(y_true))
        unit = "row (scenario IDs unavailable)"
    else:
        groups = np.asarray(groups)
        if len(groups) != len(y_true):
            raise ValueError("groups must have the same length as the target and predictions")
        if pd.isna(groups).any():
            raise ValueError("Bootstrap groups cannot contain missing values")
        unit = "scenario"

    group_codes, unique_groups = pd.factorize(groups, sort=False)
    group_count = len(unique_groups)
    baseline_errors = np.abs(y_true - baseline_predictions)
    selected_errors = np.abs(y_true - selected_predictions)
    baseline_sums = np.bincount(group_codes, weights=baseline_errors, minlength=group_count)
    selected_sums = np.bincount(group_codes, weights=selected_errors, minlength=group_count)
    group_sizes = np.bincount(group_codes, minlength=group_count)
    observed_difference = float(np.mean(baseline_errors) - np.mean(selected_errors))

    result: Dict[str, object] = {
        "difference_direction": "MAE_baseline - MAE_selected",
        "observed_difference_mae_mbps": observed_difference,
        "ci_lower_mae_mbps": float("nan"),
        "ci_upper_mae_mbps": float("nan"),
        "ci_includes_zero": None,
        "independent_test_units": group_count,
        "bootstrap_repetitions": n_bootstrap,
        "resampling_unit": unit,
        "limitation": "",
    }
    if group_count < 2:
        limitation = (
            "Only one independent test scenario is available; a scenario-level confidence "
            "interval cannot be estimated."
        )
        warnings.warn(limitation, RuntimeWarning)
        result["limitation"] = limitation
        return result
    if group_count < 10:
        limitation = (
            f"Only {group_count} independent test scenarios are available; the scenario-level "
            "bootstrap interval may be unstable."
        )
        warnings.warn(limitation, RuntimeWarning)
        result["limitation"] = limitation

    rng = np.random.default_rng(random_state)
    sampled_groups = rng.integers(0, group_count, size=(n_bootstrap, group_count))
    denominator = group_sizes[sampled_groups].sum(axis=1)
    bootstrap_differences = (
        baseline_sums[sampled_groups].sum(axis=1) - selected_sums[sampled_groups].sum(axis=1)
    ) / denominator
    lower, upper = np.percentile(bootstrap_differences, [2.5, 97.5])
    result["ci_lower_mae_mbps"] = float(lower)
    result["ci_upper_mae_mbps"] = float(upper)
    result["ci_includes_zero"] = bool(lower <= 0.0 <= upper)
    return result


def compare_regressors(train_frame: pd.DataFrame, test_frame: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    x_train, y_train = train_frame[FEATURE_COLUMNS], train_frame["rate_target_bps"] / 1e6
    x_test, y_test = test_frame[FEATURE_COLUMNS], test_frame["rate_target_bps"] / 1e6
    models = _regression_models()
    rows = []
    for name, model in models.items():
        start = perf_counter()
        model.fit(x_train, y_train)
        train_ms = (perf_counter() - start) * 1000.0
        start = perf_counter()
        predictions = model.predict(x_test)
        inference_ms = (perf_counter() - start) * 1000.0
        mse = mean_squared_error(y_test, predictions)
        rows.append({
            "model": name, "task": "user rate regression", "split": "scenario test",
            "mae_mbps": mean_absolute_error(y_test, predictions), "mse_mbps2": mse,
            "rmse_mbps": np.sqrt(mse), "r2": r2_score(y_test, predictions),
            "training_time_ms": train_ms, "inference_time_ms": inference_ms,
        })
        joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS, "target_unit": "Mbps"}, output / (name.lower().replace(" ", "_") + ".joblib"))
        if name in {"Decision Tree", "Random Forest"}:
            estimator = model.named_steps["model"]
            pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": estimator.feature_importances_}).to_csv(
                output / (name.lower().replace(" ", "_") + "_feature_importance.csv"), index=False
            )
    result = pd.DataFrame(rows)
    result.to_csv(output / "classical_model_comparison.csv", index=False)
    return result


def evaluate_classical_models(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    output_dir: str,
    seed: int = 42,
    n_bootstrap: int = 2000,
) -> Dict[str, object]:
    """Run scenario-aware CV, training-only tuning, then paired held-out test evaluation."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    x_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame["rate_target_bps"] / 1e6
    folds, cv_strategy, training_scenarios = _build_cv_splits(train_frame, x_train, y_train, seed=seed)

    if not folds:
        comparison = compare_regressors(train_frame, test_frame, str(output))
        status = {
            "status": "skipped",
            "reason": "At least two independent training scenarios are required for grouped CV and tuning.",
            "cv_strategy": cv_strategy,
            "training_scenarios": training_scenarios,
        }
        with open(output / "module5_evaluation_status.json", "w", encoding="utf-8") as file:
            json.dump(status, file, indent=2)
        return {"status": status, "comparison": comparison}

    models = _regression_models(seed)
    cv_rows = []
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    for name, model in models.items():
        scores = cross_validate(
            model, x_train, y_train, cv=folds, scoring=scoring, n_jobs=1, error_score="raise"
        )
        cv_rows.append({
            "model": name,
            "cv_strategy": cv_strategy,
            "folds": len(folds),
            "training_scenarios": training_scenarios,
            "mae_mean_mbps": float(np.mean(-scores["test_mae"])),
            "mae_std_mbps": float(np.std(-scores["test_mae"], ddof=1)) if len(folds) > 1 else 0.0,
            "rmse_mean_mbps": float(np.mean(-scores["test_rmse"])),
            "rmse_std_mbps": float(np.std(-scores["test_rmse"], ddof=1)) if len(folds) > 1 else 0.0,
            "r2_mean": float(np.mean(scores["test_r2"])),
            "r2_std": float(np.std(scores["test_r2"], ddof=1)) if len(folds) > 1 else 0.0,
        })
    cv_metrics = pd.DataFrame(cv_rows)
    cv_metrics.to_csv(output / "cross_validation_metrics.csv", index=False)

    parameter_distributions = {
        "Ridge Regression": {
            "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        },
        "Random Forest": {
            "model__n_estimators": [25, 50],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
    }
    search_models = {
        "Ridge Regression": _model_pipeline(Ridge()),
        "Random Forest": _model_pipeline(
            RandomForestRegressor(random_state=seed, n_jobs=1)
        ),
    }
    searches = {}
    search_summary_rows = []
    search_result_rows = []
    for name, estimator in search_models.items():
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=parameter_distributions[name],
            n_iter=4,
            scoring="neg_mean_absolute_error",
            cv=folds,
            refit=True,
            random_state=seed,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(x_train, y_train)
        searches[name] = search
        search_summary_rows.append({
            "model": name,
            "configurations_tested": len(search.cv_results_["params"]),
            "best_cv_mae_mbps": float(-search.best_score_),
            "best_cv_score_neg_mae": float(search.best_score_),
            "best_params": json.dumps(search.best_params_, sort_keys=True),
            "cv_strategy": cv_strategy,
            "folds": len(folds),
            "random_seed": seed,
        })
        for rank, (params, mean_score, std_score) in enumerate(zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
        ), start=1):
            search_result_rows.append({
                "model": name,
                "candidate": rank,
                "mean_cv_mae_mbps": float(-mean_score),
                "std_cv_mae_mbps": float(std_score),
                "params": json.dumps(params, sort_keys=True),
            })

    selected_name, selected_search = max(searches.items(), key=lambda item: item[1].best_score_)
    search_summary = pd.DataFrame(search_summary_rows)
    search_summary["selected_model"] = search_summary["model"].eq(selected_name)
    search_summary.to_csv(output / "hyperparameter_search.csv", index=False)
    pd.DataFrame(search_result_rows).to_csv(output / "hyperparameter_search_results.csv", index=False)

    # Only after CV and search selection are complete do we evaluate any estimator on test data.
    comparison = compare_regressors(train_frame, test_frame, str(output))
    selected_model = selected_search.best_estimator_
    baseline_model = clone(models["Linear Regression"]).fit(x_train, y_train)
    x_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame["rate_target_bps"] / 1e6
    baseline_predictions = np.asarray(baseline_model.predict(x_test), dtype=float)
    selected_predictions = np.asarray(selected_model.predict(x_test), dtype=float)
    test_rows = []
    for name, predictions, role in (
        ("Linear Regression", baseline_predictions, "bootstrap baseline"),
        (selected_name, selected_predictions, "selected model"),
    ):
        mse = mean_squared_error(y_test, predictions)
        test_rows.append({
            "model": name,
            "role": role,
            "mae_mbps": float(mean_absolute_error(y_test, predictions)),
            "mse_mbps2": float(mse),
            "rmse_mbps": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test, predictions)) if len(y_test) > 1 else float("nan"),
        })
    pd.DataFrame(test_rows).to_csv(output / "model_selection_test_metrics.csv", index=False)

    groups: Optional[np.ndarray] = None
    if "scenario_id" in test_frame.columns:
        groups = test_frame["scenario_id"].to_numpy()
    bootstrap_result = paired_bootstrap_mae_difference(
        y_test,
        baseline_predictions,
        selected_predictions,
        groups=groups,
        n_bootstrap=n_bootstrap,
        random_state=seed,
    )
    pd.DataFrame([bootstrap_result]).to_csv(output / "paired_bootstrap_comparison.csv", index=False)

    artifact = {
        "model": selected_model,
        "feature_columns": FEATURE_COLUMNS,
        "target_unit": "Mbps",
        "selected_model": selected_name,
        "best_params": selected_search.best_params_,
        "configurations_tested": len(selected_search.cv_results_["params"]),
        "best_cv_score_neg_mae": float(selected_search.best_score_),
        "best_cv_mae_mbps": float(-selected_search.best_score_),
    }
    joblib.dump(artifact, output / "selected_model.joblib")
    status = {
        "status": "completed",
        "cv_strategy": cv_strategy,
        "folds": len(folds),
        "training_scenarios": training_scenarios,
        "selected_model": selected_name,
        "best_params": selected_search.best_params_,
        "configurations_tested": len(selected_search.cv_results_["params"]),
        "best_cv_score_neg_mae": float(selected_search.best_score_),
        "best_cv_mae_mbps": float(-selected_search.best_score_),
        "test_metrics_file": "model_selection_test_metrics.csv",
        "bootstrap_file": "paired_bootstrap_comparison.csv",
    }
    with open(output / "module5_evaluation_status.json", "w", encoding="utf-8") as file:
        json.dump(status, file, indent=2)
    return {
        "status": status,
        "cv_metrics": cv_metrics,
        "hyperparameter_search": search_summary,
        "comparison": comparison,
        "test_metrics": pd.DataFrame(test_rows),
        "bootstrap": bootstrap_result,
    }


def predict_rate(model_artifact: str, frame: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(model_artifact)
    missing = sorted(set(bundle["feature_columns"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Inference data is missing features: {', '.join(missing)}")
    return np.asarray(bundle["model"].predict(frame[bundle["feature_columns"]]), dtype=float)
