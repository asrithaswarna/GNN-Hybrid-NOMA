"""

Supports command line arguments, executes full GNN training, baseline evaluation,
SCMA simulation, and generates publication plots and metric tables.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import load_config
from src.utilities import set_seed, get_device, save_json, save_dataframe_csv
from src.channel_model import WirelessNetwork
from src.gnn_model import HybridNOMAGNN
from src.training import DatasetGenerator, GNNTrainer
from src.evaluation import NetworkEvaluator
from src.visualization import Visualizer
from src.pairing import UserPairingManager
from src.data_pipeline import (
    load_or_generate_scenarios,
    scenarios_to_frame,
    scenarios_to_graphs,
    generate_synthetic_scenarios,
)
from src.classical_models import compare_regressors, evaluate_classical_models, predict_rate
from src.unsupervised import run_clustering
from src.model_io import save_model_metadata
from src.monitoring import PredictionMonitor


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Graph Neural Network-Based Dynamic User Pairing and Slot Assignment for Hybrid NOMA Networks"
    )
    parser.add_argument("--users", type=int, default=10, help="Number of users in the network (default: 10)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--slots", type=int, default=5, help="Number of orthogonal slots (default: 5)")
    parser.add_argument("--output", type=str, default="results", help="Directory to save results (default: results)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--presentation-mode", "--simple-output", dest="presentation_mode", action="store_true",
                        help="Show a simple, readable presentation summary (the default).")
    parser.add_argument("--detailed-output", action="store_true",
                        help="Show the full technical training, benchmark, and SCMA logs.")
    parser.add_argument("--show-plots", action="store_true",
                        help="Display generated Matplotlib figures in a desktop window after saving them.")
    parser.add_argument("--mode", choices=["train", "evaluate", "inference", "clustering", "compare", "retrain"],
                        default="train", help="Workflow to run (default: train).")
    parser.add_argument("--data-source", choices=["synthetic", "csv"], default="synthetic",
                        help="Input source for analysis workflows (default: synthetic).")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Per-user scenario CSV path for --data-source csv.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Classical model artifact for inference mode.")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000,
                        help="Paired scenario-bootstrap repetitions in compare mode (default: 2000).")
    parser.add_argument("--ber-blocks", type=int, default=100,
                        help="Uncoded BER transmission blocks per scenario and SNR (default: 100).")
    return parser.parse_args(args)


def _simple_method_name(method_name):
    names = {
        "Orthogonal Multiple Access (OMA)": "OMA",
        "Random Pairing PD-NOMA": "Random Pairing",
        "Near-Far Channel Gain NOMA": "Near-Far Pairing",
        "Greedy Sum-Rate NOMA": "Greedy Pairing",
        "Optimization Oracle": "Optimization Reference",
        "Proposed GNN-Based NOMA": "GNN Pairing",
    }
    return names.get(method_name, method_name)


def _build_pair_rows(pairing_plan, schedule, users):
    user_lookup = {u.user_id: u for u in users}
    pair_rows = []

    for idx, pair in enumerate(pairing_plan.pairs, start=1):
        slot_id = None
        pair_method = "Not scheduled"
        weak_user = None
        strong_user = None
        pair_rate_mbps = 0.0

        for rec in schedule.slots:
            if (
                rec.mode in {"PD-NOMA", "SCMA"}
                and pair.user1_id in rec.assigned_user_ids
                and pair.user2_id in rec.assigned_user_ids
            ):
                slot_id = rec.slot_id
                pair_method = rec.mode
                pair_rate_mbps = rec.slot_sum_rate_bps / 1e6
                if rec.mode == "PD-NOMA" and rec.pd_pair_details is not None:
                    pd_res = rec.pd_pair_details
                    weak_user = f"U{pd_res.weak_user_id}"
                    strong_user = f"U{pd_res.strong_user_id}"
                break

        u1 = user_lookup[pair.user1_id]
        u2 = user_lookup[pair.user2_id]
        if weak_user is None:
            weak_user = f"U{min(u1.user_id, u2.user_id)}"
        if strong_user is None:
            strong_user = f"U{max(u1.user_id, u2.user_id)}"

        pair_rows.append({
            "pair_number": idx,
            "users": f"U{u1.user_id} and U{u2.user_id}",
            "slot": f"S{slot_id + 1}" if slot_id is not None else "Not scheduled",
            "slot_number": slot_id + 1 if slot_id is not None else None,
            "method": pair_method,
            "weak_user": weak_user,
            "strong_user": strong_user,
            "rate_mbps": pair_rate_mbps,
        })

    for rec in schedule.slots:
        if rec.mode == "OMA_SINGLE" and len(rec.assigned_user_ids) == 1:
            user_id = rec.assigned_user_ids[0]
            pair_rows.append({
                "pair_number": len(pair_rows) + 1,
                "users": f"U{user_id}",
                "slot": f"S{rec.slot_id + 1}",
                "slot_number": rec.slot_id + 1,
                "method": "OMA_SINGLE",
                "weak_user": f"U{user_id}",
                "strong_user": "-",
                "rate_mbps": rec.slot_sum_rate_bps / 1e6,
            })

    return pair_rows


def format_simple_summary(num_users, num_slots, seed, epochs, train_samples, val_samples,
                          best_val_loss, total_rate_mbps, avg_rate_mbps, fairness,
                          outage_percent, pair_rows, benchmark_rows, scma_rows,
                          reference_method, training_rows=None, scma_integrated=False,
                          data_source="synthetic", test_samples=0, summary_sections=None,
                          pd_noma_average_pair_rate=0.0, classical_model_results=None,
                          module5_evaluation=None,
                          best_validation_epoch=None,
                          linear_regression_r2=None, random_forest_r2=None):
    if summary_sections is None:
        summary_sections = {
            "data_preparation": True,
            "classical_ml": True,
            "gnn_training": True,
            "user_pairing": True,
            "communication": True,
            "final_summary": True,
            "limitations": True,
        }

    train_test_text = f"{train_samples} / {val_samples} / {test_samples}"
    best_epoch = best_validation_epoch if best_validation_epoch is not None else epochs

    feature_names = ["x", "y", "distance", "channel_gain", "channel_gain_db", "snr_db", "min_rate_req_bps"]
    if classical_model_results is not None and not classical_model_results.empty:
        linear_row = classical_model_results[classical_model_results["model"] == "Linear Regression"]
        random_forest_row = classical_model_results[classical_model_results["model"] == "Random Forest"]
        if not linear_row.empty:
            linear_regression_r2 = float(linear_row.iloc[0]["r2"])
        if not random_forest_row.empty:
            random_forest_r2 = float(random_forest_row.iloc[0]["r2"])

    benchmark_rows = benchmark_rows or []
    scma_rows = scma_rows or []
    pair_rows = pair_rows or []

    model_lines = [
        "MODEL PERFORMANCE",
    ]
    if linear_regression_r2 is not None:
        model_lines.append(f"Linear Regression R2 = {linear_regression_r2:.4f}")
    else:
        model_lines.append("Linear Regression R2 = unavailable")
    if random_forest_r2 is not None:
        model_lines.append(f"Random Forest R2 = {random_forest_r2:.4f}")
    else:
        model_lines.append("Random Forest R2 = unavailable")
    if module5_evaluation and module5_evaluation["status"]["status"] == "completed":
        cv_metrics = module5_evaluation["cv_metrics"]
        selected_model = module5_evaluation["status"]["selected_model"]
        best_params = module5_evaluation["status"]["best_params"]
        printable_params = ", ".join(
            f"{key.removeprefix('model__')}={value}" for key, value in best_params.items()
        )
        bootstrap = module5_evaluation["bootstrap"]
        model_lines.extend([
            "",
            "MODULE 5 MODEL EVALUATION",
            f"Cross-validation : {module5_evaluation['status']['cv_strategy']} "
            f"({module5_evaluation['status']['folds']} folds; training scenarios only)",
            "CV metrics (mean +/- std): MAE and RMSE in Mbps; R2 unitless",
        ])
        for row in cv_metrics.itertuples(index=False):
            model_lines.append(
                f"  {row.model}: MAE={row.mae_mean_mbps:.4f} +/- {row.mae_std_mbps:.4f}; "
                f"RMSE={row.rmse_mean_mbps:.4f} +/- {row.rmse_std_mbps:.4f}; "
                f"R2={row.r2_mean:.4f} +/- {row.r2_std:.4f}"
            )
        model_lines.extend([
            f"Selected model : {selected_model}",
            f"Selected model best parameters : {printable_params}",
            f"Configurations tested : {module5_evaluation['status']['configurations_tested']}",
            f"Best CV score (negative MAE) : {module5_evaluation['status']['best_cv_score_neg_mae']:.4f}",
            f"Selected model best CV MAE : {module5_evaluation['status']['best_cv_mae_mbps']:.4f} Mbps",
            "Held-out test metrics:",
        ])
        for row in module5_evaluation["test_metrics"].itertuples(index=False):
            model_lines.append(
                f"  {row.model} ({row.role}): MAE={row.mae_mbps:.4f} Mbps, "
                f"RMSE={row.rmse_mbps:.4f} Mbps, R2={row.r2:.4f}"
            )
        model_lines.extend([
            f"Paired bootstrap (MAE_baseline - MAE_selected): "
            f"{bootstrap['observed_difference_mae_mbps']:.4f} Mbps",
            f"95% CI: [{bootstrap['ci_lower_mae_mbps']:.4f}, "
            f"{bootstrap['ci_upper_mae_mbps']:.4f}] Mbps; "
            f"includes zero: {bootstrap['ci_includes_zero']} "
            f"({bootstrap['bootstrap_repetitions']} repetitions, "
            f"{bootstrap['independent_test_units']} independent scenarios)",
        ])
        if bootstrap["limitation"]:
            model_lines.append(f"Bootstrap limitation: {bootstrap['limitation']}")
    elif module5_evaluation:
        model_lines.extend([
            "",
            "MODULE 5 MODEL EVALUATION",
            f"Skipped: {module5_evaluation['status']['reason']}",
        ])
    model_lines.append(f"GNN Rate = {total_rate_mbps:.2f} Mbps")

    lines = [
        "=" * 52,
        "GNN-BASED HYBRID NOMA NETWORK",
        "NETWORK SETUP",
        "",
        f"Users : {num_users}",
        f"Slots : {num_slots}",
        f"Data Source : {data_source}",
        f"Train / Val / Test : {train_test_text}",
        "",
        "",
        "1. DATA PREPARATION",
        "",
        "Features used : " + ", ".join(feature_names),
        "Preprocessing : Feature pipeline + StandardScaler",
        "Data preparation : Completed",
        "",
        "",
        "1. ML MODEL RESULTS",
        "",
        *model_lines,
        "",
        "GNN TRAINING",
        f"Epochs completed : {epochs}",
        "Stopping : configured epoch limit (no early stopping)",
        f"Best Validation Loss : {best_val_loss:.4f}",
        f"Best Validation Epoch : {best_epoch}",
        f"Test Performance : {total_rate_mbps:.2f} Mbps total rate | {avg_rate_mbps:.2f} Mbps average user rate",
        "",
        "",
        "1. GNN USER PAIRING",
        "",
        "PAIR USERS SLOT METHOD",
    ]

    if pair_rows:
        for row in pair_rows:
            slot_label = row.get("slot", f"Slot {row.get('slot_number', 1)}")
            method = row.get("method", "PD-NOMA")
            users_text = row.get("users", "U0 and U1")
            lines.append(f"{row.get('pair_number', 1)}. {users_text} -> {slot_label} -> {method}")
    else:
        lines.append("No valid pairing found.")
    slot_allocations = {slot: [] for slot in range(1, num_slots + 1)}
    for row in pair_rows:
        slot_number = row.get("slot_number")
        if slot_number in slot_allocations:
            slot_allocations[slot_number].append(row.get("users", "allocation"))
    occupancy = " | ".join(
        f"S{slot}: {', '.join(users) if users else 'IDLE'}"
        for slot, users in slot_allocations.items()
    )
    lines.append(f"Slot occupancy : {occupancy}")
    lines.append("Capacity rule : at most one allocation per slot; unused slots remain idle.")

    lines.extend([
        "",
        "Pairing source : GNN",
        "Slot assignment : GNN slot prediction with deterministic capacity-aware resolution",
        "",
        "",
        "1. COMMUNICATION PERFORMANCE",
        "",
        "PAIRING / RESOURCE ALLOCATION",
        "",
        "Method            Rate      Fairness  Outage",
    ])

    for row in benchmark_rows:
        method_label = _simple_method_name(row.get("method", ""))
        rate = float(row.get("total_rate_mbps", 0.0))
        fairness_val = float(row.get("fairness", 0.0))
        outage_val = float(row.get("outage_percent", row.get("outage_probability_mean", 0.0) * 100.0))
        lines.append(f"{method_label:<16} {rate:>7.2f} Mbps  {fairness_val:>7.3f}  {outage_val:>6.2f}%")

    lines.extend([
        "",
        "PD-NOMA",
        f"Average Pair Rate : {pd_noma_average_pair_rate:.2f} Mbps",
        "SIC : Enabled",
        "",
        "",
        "SCMA PERFORMANCE",
        "",
        "SNR     BER      SER      Data Rate",
    ])

    snr_values = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    for snr in snr_values:
        match = next((row for row in scma_rows if abs(float(row.get("snr_db", 0.0)) - snr) < 1e-9), None)
        if match is None:
            ber = 0.0
            ser = 0.0
            rate = 0.0
        else:
            ber = float(match.get("ber", 0.0))
            ser = float(match.get("ser", 0.0))
            rate = float(match.get("rate_mbps", 0.0))
        lines.append(f"{snr:>4.0f} dB  {ber:>7.4f}  {ser:>7.4f}  {rate:>7.2f} Mbps")

    lines.extend([
        "",
        "SCMA evaluation : Standalone SCMA experiment preserved; hybrid schedule may select SCMA per slot",
        "",
        "",
        "1. FINAL SUMMARY",
        "",
        f"GNN benchmark rate : {total_rate_mbps:.2f} Mbps",
        f"Reference rate : {float(reference_method.get('total_rate_mbps', 0.0)):.2f} Mbps",
        f"Average user rate : {avg_rate_mbps:.2f} Mbps",
        f"Fairness : {fairness:.3f}",
        f"Outage : {outage_percent:.2f}%",
        "",
        "Classical regression models evaluated (cross-validation) : Linear Regression, Ridge Regression, Lasso Regression, Decision Tree, Random Forest",
        "GNN model evaluated : Proposed GNN-Based NOMA",
        "Pairing methods : Random, Near-Far, Greedy, Optimization, GNN",
        "Communication methods : OMA, PD-NOMA, SCMA",
        "",
        "Result:",
        "GNN was trained to learn user-pairing patterns",
        "from wireless network data.",
        "",
        "Note:",
        "The GNN slot-preference head drives the final slot assignment through deterministic conflict resolution and slot-capacity checks.",
        "",
        "PD-NOMA and SCMA are integrated in the hybrid scheduler: each allocation chooses the feasible communication mode that best matches the slot conditions.",
        "",
        "Results saved successfully.",
        "=" * 52,
    ])
    return "\n".join(lines)


def _apply_cli_config(args):
    cfg = load_config(args.config)
    cfg.network.num_users = args.users
    cfg.network.num_slots = args.slots
    cfg.gnn.epochs = args.epochs
    cfg.system.seed = args.seed
    cfg.system.output_dir = args.output
    cfg.validate()
    return cfg


def _run_analysis_mode(args, cfg):
    output_dir = Path(cfg.system.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in {"compare", "clustering", "inference", "retrain"}:
        splits = load_or_generate_scenarios(
            cfg, data_source=args.data_source, dataset=args.dataset,
            num_samples=max(cfg.dataset.num_train_samples + cfg.dataset.num_val_samples + cfg.dataset.num_test_samples, 12),
        )
        train_frame = scenarios_to_frame(splits.train, cfg)
        test_frame = scenarios_to_frame(splits.test, cfg)

    if args.mode == "compare":
        classical_dir = output_dir / "classical_models"
        evaluation = evaluate_classical_models(
            train_frame,
            test_frame,
            str(classical_dir),
            seed=cfg.system.seed,
            n_bootstrap=args.bootstrap_repetitions,
        )
        result = evaluation["comparison"]
        save_model_metadata(str(output_dir / "classical_models" / "experiment.json"), {
            "data_source": splits.source, "dataset_path": splits.dataset_path,
            "task": "user rate regression", "train_scenarios": len(splits.train),
            "validation_scenarios": len(splits.validation), "test_scenarios": len(splits.test),
            "seed": cfg.system.seed,
            "module5_evaluation": evaluation["status"],
        })
        print(result.to_string(index=False))
        if evaluation["status"]["status"] == "completed":
            print("\nCross-validation (mean +/- std):")
            print(evaluation["cv_metrics"].to_string(index=False))
            print("\nHyperparameter search (scoring=negative MAE):")
            for row in evaluation["hyperparameter_search"].itertuples(index=False):
                params = row.best_params
                readable_params = ", ".join(
                    f"{key.removeprefix('model__')}={value}"
                    for key, value in json.loads(params).items()
                )
                selected = " [SELECTED]" if row.selected_model else ""
                print(
                    f"{row.model}{selected}: {row.configurations_tested} configurations; "
                    f"best CV score={row.best_cv_score_neg_mae:.4f} "
                    f"(MAE={row.best_cv_mae_mbps:.4f} Mbps); params: {readable_params}"
                )
            print("\nTuned-model held-out test metrics:")
            print(evaluation["test_metrics"].to_string(index=False))
            bootstrap = evaluation["bootstrap"]
            print(
                "Paired bootstrap: "
                f"MAE_baseline - MAE_selected = {bootstrap['observed_difference_mae_mbps']:.4f} Mbps; "
                f"95% CI [{bootstrap['ci_lower_mae_mbps']:.4f}, "
                f"{bootstrap['ci_upper_mae_mbps']:.4f}] Mbps; "
                f"includes zero: {bootstrap['ci_includes_zero']}."
            )
            if bootstrap["limitation"]:
                print(f"Bootstrap limitation: {bootstrap['limitation']}")
        else:
            print(f"Module 5 evaluation skipped: {evaluation['status']['reason']}")
        print(f"Classical model comparison saved to {classical_dir}")
        return

    if args.mode == "clustering":
        result = run_clustering(pd.concat([train_frame, test_frame], ignore_index=True), str(output_dir / "clustering"), cfg.system.seed)
        print(f"PCA explained variance: {result['explained_variance']}")
        print(f"Clustering outputs saved to {output_dir / 'clustering'}")
        return

    if args.mode == "inference":
        artifact = args.model_path or str(output_dir / "classical_models" / "random_forest.joblib")
        monitor = PredictionMonitor(str(output_dir / "monitoring" / "predictions.json"))
        predictions = predict_rate(artifact, test_frame)
        for value in predictions[:5]:
            monitor.record(float(value), 0.0)
        print(f"Loaded model: {artifact}")
        print(f"Predictions generated: {len(predictions)}")
        print(f"First predicted rate: {predictions[0]:.3f} Mbps")
        return

    if args.mode == "evaluate":
        import torch
        model_path = args.model_path or str(output_dir / "best_gnn_model.pt")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"GNN checkpoint does not exist: {model_path}. Run --mode train first.")
        device = get_device(cfg.system.device)
        model = HybridNOMAGNN(cfg=cfg.gnn, num_slots=cfg.network.num_slots).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        evaluator = NetworkEvaluator(cfg=cfg, gnn_model=model)
        metrics = evaluator.evaluate_test_set(
            num_test_instances=cfg.dataset.num_test_samples, seed_offset=1000
        )
        print(pd.DataFrame([value.__dict__ for value in metrics.values()]).to_string(index=False))
        figures_dir = output_dir / "figures"
        metrics_dir = output_dir / "metrics"
        figures_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        visualizer = Visualizer(output_dir=str(figures_dir), show_plots=args.show_plots)
        outage_path = visualizer.plot_outage_comparison(metrics)
        ber_results = evaluator.evaluate_ber_snr_sweep(
            num_test_instances=cfg.dataset.num_test_samples,
            seed_offset=1000,
            num_blocks=args.ber_blocks,
        )
        ber_csv_path = metrics_dir / "ber_comparison.csv"
        save_dataframe_csv(ber_results, str(ber_csv_path))
        ber_plot_path = visualizer.plot_ber_comparison(ber_results)
        print(f"Saved outage comparison graph to: {outage_path}")
        print(f"Saved BER comparison CSV to: {ber_csv_path}")
        print(f"Saved BER comparison graph to: {ber_plot_path}")
        return

    if args.mode == "retrain":
        candidate_dir = output_dir / "retrain_candidate"
        result = compare_regressors(train_frame, test_frame, str(candidate_dir))
        existing = output_dir / "classical_models" / "random_forest.joblib"
        print(result.to_string(index=False))
        print(f"Retraining candidate saved to {candidate_dir}")
        print(f"Existing model retained: {existing.exists()} (no automatic replacement performed)")
        return


def main():
    args = parse_args()
    cfg = _apply_cli_config(args)
    if args.mode != "train":
        _run_analysis_mode(args, cfg)
        return
    detailed = args.detailed_output
    if detailed:
        print("=" * 80)
        print("HYBRID NOMA DYNAMIC USER PAIRING & SLOT ASSIGNMENT VIA GRAPH NEURAL NETWORKS")
       
        print("=" * 80)
    else:
        print("Starting experiment...", flush=True)

    # 1. Load configuration and apply CLI overrides
    device = get_device(cfg.system.device)
    set_seed(cfg.system.seed)
    if detailed:
        print(f"[Config] Users: {cfg.network.num_users} | Slots: {cfg.network.num_slots} | Epochs: {cfg.gnn.epochs} | Seed: {cfg.system.seed} | Device: {device}")

    # Set up directory layout
    output_dir = Path(cfg.system.output_dir)
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    visualizer = Visualizer(output_dir=str(figures_dir), show_plots=args.show_plots)

    # 2. Dataset Generation
    if detailed:
        print("\n[Phase 1] Generating Synthetic Channel Topologies with Combinatorial Oracle Supervision...")
    else:
        print("Preparing network data...", flush=True)
    data_gen = DatasetGenerator(cfg)
    if args.data_source == "csv":
        splits = load_or_generate_scenarios(
            cfg, data_source="csv", dataset=args.dataset,
            num_samples=cfg.dataset.num_train_samples + cfg.dataset.num_val_samples + cfg.dataset.num_test_samples,
        )
        train_graphs = scenarios_to_graphs(splits.train, cfg)
        val_graphs = scenarios_to_graphs(splits.validation, cfg)
        csv_test_scenarios = splits.test
    else:
        train_graphs = data_gen.generate_split(num_samples=cfg.dataset.num_train_samples, seed_offset=100)
        val_graphs = data_gen.generate_split(num_samples=cfg.dataset.num_val_samples, seed_offset=500)
        csv_test_scenarios = None
    if detailed:
        print(f"Generated {len(train_graphs)} training graphs and {len(val_graphs)} validation graphs.")

    # 3. Model Initialization & Training
    if detailed:
        print("\n[Phase 2] Initializing and Training Edge-Aware Hybrid NOMA GNN...")
    else:
        print("Training the model...", flush=True)
    model = HybridNOMAGNN(cfg=cfg.gnn, num_slots=cfg.network.num_slots)
    trainer = GNNTrainer(model=model, cfg=cfg, device=device)
    ckpt_path = str(output_dir / "best_gnn_model.pt")
    history = trainer.train(train_graphs=train_graphs, val_graphs=val_graphs,
                            epochs=cfg.gnn.epochs, save_path=ckpt_path, verbose=detailed)
    if detailed:
        print(f"Training completed. Best validation loss: {history.best_val_loss:.4f}. Model saved to {ckpt_path}")

    # Plot training loss
    loss_plot_path = visualizer.plot_training_history(history, filename="training_loss_curves.png")
    if detailed:
        print(f"Saved loss curve to: {loss_plot_path}")

    # 4. Evaluation and Benchmarking
    if detailed:
        print("\n[Phase 3] Running Multi-Algorithm Benchmark Evaluation...")
    else:
        print("Evaluating the methods...", flush=True)
    evaluator = NetworkEvaluator(cfg=cfg, gnn_model=model)
    benchmark_metrics = evaluator.evaluate_test_set(
        num_test_instances=cfg.dataset.num_test_samples,
        seed_offset=1000,
        scenarios=csv_test_scenarios,
    )

    # Convert to DataFrame and print table
    df_results = evaluator.to_dataframe(benchmark_metrics)
    if detailed:
        print("\n" + "=" * 110)
        print("BENCHMARK EVALUATION RESULTS (Across Identical Test Topologies)")
        print("=" * 110)
    display_cols = [
        "method_name",
        "sum_rate_mean_mbps",
        "sum_rate_std_mbps",
        "user_rate_mean_mbps",
        "jains_fairness_mean",
        "outage_probability_mean",
        "slot_utilization_mean",
        "inference_time_ms",
    ]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    if detailed:
        print(df_results[display_cols].to_string(index=False))
        print("=" * 110)

    # Save benchmark table
    csv_path = str(metrics_dir / "benchmark_results.csv")
    save_dataframe_csv(df_results, csv_path)
    if detailed:
        print(f"Saved benchmark CSV to: {csv_path}")

    outage_plot_path = visualizer.plot_outage_comparison(
        benchmark_metrics, filename="outage_comparison.png"
    )
    print(f"Saved outage comparison graph to: {outage_plot_path}")

    # 5. SCMA MPA Detector Simulation
    if detailed:
        print("\n[Phase 4] Evaluating SCMA Message Passing Algorithm (MPA) Detector Across SNR...")
    else:
        print("Testing SCMA performance...", flush=True)
    scma_results = evaluator.evaluate_scma_snr_sweep(snr_db_range=[-5.0, 0.0, 5.0, 10.0, 15.0, 20.0], num_blocks=100)
    for r in scma_results:
        if detailed:
            print(f"SNR: {r.snr_db:5.1f} dB | BER: {r.bit_error_rate:8.5f} | SER: {r.symbol_error_rate:8.5f} | Sum-Rate: {r.sum_rate_bps/1e6:6.2f} Mbps")
    scma_plot_path = visualizer.plot_scma_ber_curve(scma_results, filename="scma_ber_curve.png")
    if detailed:
        print(f"Saved SCMA BER curve to: {scma_plot_path}")

    # 6. Bit-level comparison for pairing policies, independent of the SCMA MPA sweep.
    print("Evaluating uncoded BPSK BER for the benchmark pairing methods...", flush=True)
    ber_results = evaluator.evaluate_ber_snr_sweep(
        num_blocks=args.ber_blocks,
        num_test_instances=cfg.dataset.num_test_samples,
        seed_offset=1000,
        scenarios=csv_test_scenarios,
    )
    ber_csv_path = str(metrics_dir / "ber_comparison.csv")
    save_dataframe_csv(ber_results, ber_csv_path)
    ber_plot_path = visualizer.plot_ber_comparison(
        ber_results, filename="ber_comparison.png"
    )
    print(f"Saved BER comparison CSV to: {ber_csv_path}")
    print(f"Saved BER comparison graph to: {ber_plot_path}")

    # 7. Generate Topology & Benchmark Plots
    if detailed:
        print("\n[Phase 5] Generating Publication Figures...")
    else:
        print("Creating graphs and saving results...", flush=True)
    # Generate single topology for illustration
    net = WirelessNetwork(cfg.network, cfg.channel, seed=cfg.system.seed)
    demo_users = csv_test_scenarios[0] if csv_test_scenarios else net.generate_users(num_users=cfg.network.num_users)
    graph_b = data_gen.graph_builder.build_graph(demo_users)
    model.eval()
    import torch
    with torch.no_grad():
        aff_probs = model.predict_pairing_probabilities(
            graph_b.node_features.to(device),
            graph_b.edge_index.to(device),
            graph_b.edge_features.to(device)
        ).cpu().numpy()
        slot_logits = model.predict_slot_logits(
            graph_b.node_features.to(device),
            graph_b.edge_index.to(device),
            graph_b.edge_features.to(device)
        ).cpu()
    pairing_mgr = UserPairingManager(evaluator.pd_engine)
    demo_plan = pairing_mgr.gnn_affinity_pairing(aff_probs, demo_users)
    demo_slot_mapping = evaluator.slot_engine.resolve_gnn_slot_mapping(demo_plan, demo_users, slot_logits)
    demo_sched = evaluator.slot_engine.evaluate_schedule(demo_plan, demo_users, custom_slot_mapping=demo_slot_mapping)

    topo_plot_path = visualizer.plot_network_topology(
        users=demo_users,
        pairing_plan=demo_plan,
        cell_radius_max=cfg.network.cell_radius_max,
        filename="network_topology.png"
    )
    if detailed:
        print(f"Saved topology plot to: {topo_plot_path}")

    bench_plot_path = visualizer.plot_benchmark_comparison(benchmark_metrics, filename="sum_rate_benchmark.png")
    if detailed:
        print(f"Saved benchmark bar chart to: {bench_plot_path}")

    # Save summary JSON
    summary_dict = {
        "config": {
            "num_users": cfg.network.num_users,
            "num_slots": cfg.network.num_slots,
            "epochs": cfg.gnn.epochs,
            "seed": cfg.system.seed,
            "device": device,
            "data_source": args.data_source,
            "dataset_path": args.dataset or "",
        },
        "benchmark_summary": {k: v.__dict__ for k, v in benchmark_metrics.items()},
    }
    save_json(summary_dict, str(metrics_dir / "summary.json"))

    if not detailed:
        pair_rows = _build_pair_rows(demo_plan, demo_sched, demo_users)
        pd_noma_pair_rates = [
            float(record.slot_sum_rate_bps / 1e6)
            for record in demo_sched.slots
            if record.mode == "PD-NOMA" and record.slot_sum_rate_bps > 0
        ]
        benchmark_rows = [
            {
                "method": method_name,
                "total_rate_mbps": float(metrics.sum_rate_mean_mbps),
                "avg_rate_mbps": float(metrics.user_rate_mean_mbps),
                "fairness": float(metrics.jains_fairness_mean),
                "outage_percent": float(metrics.outage_probability_mean * 100.0),
            }
            for method_name, metrics in benchmark_metrics.items()
        ]
        scma_rows = [
            {
                "snr_db": float(result.snr_db),
                "ber": float(result.bit_error_rate),
                "ser": float(result.symbol_error_rate),
                "rate_mbps": float(result.sum_rate_bps / 1e6),
            }
            for result in scma_results
        ]
        gnn_metrics = benchmark_metrics["Proposed GNN-Based NOMA"]
        reference_metrics = benchmark_metrics["Optimization Oracle"]
        selected_epochs = {1, 10, 20, 30, 40, cfg.gnn.epochs}
        training_rows = [
            {
                "epoch": epoch,
                "train_loss": history.train_losses[epoch - 1],
                "val_loss": history.val_losses[epoch - 1],
            }
            for epoch in sorted(selected_epochs)
            if 1 <= epoch <= len(history.train_losses)
        ]
        classical_evaluation = evaluate_classical_models(
            scenarios_to_frame(generate_synthetic_scenarios(cfg, num_samples=max(cfg.dataset.num_train_samples, 12), seed_offset=100), cfg),
            scenarios_to_frame(generate_synthetic_scenarios(cfg, num_samples=max(cfg.dataset.num_test_samples, 12), seed_offset=500), cfg),
            str(output_dir / "classical_models"),
            seed=cfg.system.seed,
            n_bootstrap=args.bootstrap_repetitions,
        )
        classical_model_result = classical_evaluation["comparison"]
        simple_summary = format_simple_summary(
            num_users=cfg.network.num_users,
            num_slots=cfg.network.num_slots,
            seed=cfg.system.seed,
            epochs=cfg.gnn.epochs,
            train_samples=len(train_graphs),
            val_samples=len(val_graphs),
            test_samples=cfg.dataset.num_test_samples,
            best_val_loss=history.best_val_loss,
            total_rate_mbps=gnn_metrics.sum_rate_mean_mbps,
            avg_rate_mbps=gnn_metrics.user_rate_mean_mbps,
            fairness=gnn_metrics.jains_fairness_mean,
            outage_percent=gnn_metrics.outage_probability_mean * 100.0,
            pair_rows=pair_rows,
            benchmark_rows=benchmark_rows,
            scma_rows=scma_rows,
            reference_method={"total_rate_mbps": reference_metrics.sum_rate_mean_mbps},
            training_rows=training_rows,
            data_source=args.data_source,
            summary_sections={
                "data_preparation": True,
                "classical_ml": True,
                "unsupervised": True,
                "gnn_training": True,
                "gnn_evaluation": True,
                "user_pairing": True,
                "slot_assignment": True,
                "pd_noma": True,
                "wireless_performance": True,
                "scma": True,
                "deployment": True,
                "final_summary": True,
                "limitations": True,
            },
            pd_noma_average_pair_rate=float(np.mean(pd_noma_pair_rates)) if pd_noma_pair_rates else 0.0,
            classical_model_results=classical_model_result,
            module5_evaluation=classical_evaluation,
            best_validation_epoch=int(np.argmin(history.val_losses)) + 1,
        )
        print("\n" + simple_summary)
        presentation_path = output_dir / "presentation_summary.txt"
        presentation_path.write_text(simple_summary + "\n", encoding="utf-8")
        print("Presentation summary saved.")

    if detailed:
        print(f"\n[Complete] All results, figures, and metrics saved to '{output_dir}'.")
    else:
        print("Experiment completed.")


if __name__ == "__main__":
    main()
