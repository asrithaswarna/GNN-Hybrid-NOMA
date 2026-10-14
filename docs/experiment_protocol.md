# Experimental Evaluation Protocol & Benchmark Guide

## 0. Additional ML Workflows

The default `--mode train --data-source synthetic` command preserves the original GNN/NOMA workflow. Optional validated CSV scenarios can be used for GNN training/evaluation, classical regression, and clustering:

```bash
python main.py --mode compare --data-source synthetic --users 6 --output results/module5_compare
python main.py --mode compare --data-source csv --dataset data/example_users.csv --users 4 --output results/csv_analysis
python main.py --mode clustering --data-source synthetic --users 6 --output results/clustering
python main.py --mode inference --model-path results/module5_compare/classical_models/random_forest.joblib --output results/inference
python main.py --mode retrain --data-source synthetic --users 6 --output results/retrain
```

The classical task is user-rate regression using scenario-level train/validation/test separation. Reported metrics are MAE, MSE, RMSE, and R2. Wireless metrics such as sum rate, Jain fairness, outage, BER, and SER remain separate domain metrics from the GNN/NOMA evaluator. A candidate retraining run does not silently overwrite an existing model.

In `--mode compare`, classical-model selection additionally uses GroupKFold splits grouped by
`scenario_id` (up to five folds). Imputation and scaling are fitted inside each estimator
pipeline on each training fold. Linear Regression, Ridge, Lasso, Decision Tree, and Random Forest
receive comparable fold-wise MAE, RMSE, and R2 summaries. RandomizedSearchCV tunes Ridge and
Random Forest against negative MAE using those same training-only folds; the winner is selected
without consulting the held-out test set. The validation split is not used for this model
selection. After selection, final test metrics compare the selected estimator with Linear
Regression.

`paired_bootstrap_comparison.csv` resamples complete test scenarios (not individual users) and
reports the 95% percentile interval for `MAE_baseline - MAE_selected`. A positive difference
favors the selected estimator. An interval containing zero is not evidence of a statistically
reliable difference. Fewer than ten test scenarios triggers a limitation warning; with fewer than
two scenarios the interval is unavailable.

The `compare` mode saves `cross_validation_metrics.csv`, `hyperparameter_search.csv`,
`hyperparameter_search_results.csv`, `model_selection_test_metrics.csv`,
`paired_bootstrap_comparison.csv`, and `selected_model.joblib` under
`<output>/classical_models/`. `--bootstrap-repetitions` controls the bootstrap count (default
2,000).

## 1. Objective & Benchmark Principles
This protocol outlines the scientific procedure to reproduce and validate all experimental comparisons presented in the B.Tech project.

To maintain strict scientific validity:
1. **Identical Topologies**: All algorithms (OMA, Random NOMA, Near-Far NOMA, Greedy NOMA, Oracle, GNN) are evaluated on the exact same channel realizations for each trial.
2. **Zero Data Leakage**: The test set topologies use independent random seeds (`seed = 2000` to `2049`) not seen during GNN training or validation.
3. **Statistical Reporting**: All results are reported as **Mean $\pm$ Standard Deviation** over multiple independent random network instances.

---

## 2. Experimental Execution Commands

### Execution of Modular Project (Version A)
```bash
# Standard 10-user run with 25 epochs
python main.py --users 10 --epochs 25 --seed 42 --slots 5 --output results

# Fast verification run (e.g. quick test on 6 users), separate from presentation results
python main.py --users 6 --epochs 2 --seed 42 --slots 3 --output test-output/train-smoke
```

The former standalone-script examples were removed because
`hybrid_noma_gnn_complete.py` is not present in this project; `main.py` is the supported entry point.

### Automated Unit Testing Suite
```bash
# Run all tests with verbose output
python -m pytest tests/ -v
```

---

## 3. Evaluation Metrics

1. **Total Network Sum Rate ($\text{Mbps}$)**:
   $$R_{\text{total}} = \sum_{s=1}^S R_s \quad [\text{Mbps}]$$
2. **Average User Rate ($\text{Mbps}$)**:
   $$\bar{R} = \frac{1}{N} \sum_{i=1}^N R_i \quad [\text{Mbps}]$$
3. **Minimum User Rate ($\text{Mbps}$)**:
   $$R_{\min} = \min_{i \in \{1, \dots, N\}} R_i$$
   Reflects quality-of-service for disadvantaged cell-edge users.
4. **Jain's Fairness Index**:
   $$J = \frac{\left(\sum_{i=1}^N R_i\right)^2}{N \sum_{i=1}^N R_i^2} \in \left[\frac{1}{N}, 1.0\right]$$
5. **Outage Probability**:
   $$P_{\text{outage}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(R_i < R_{\text{threshold}})$$
6. **Slot Utilization Rate ($\%$)**:
   $$\eta_{\text{slot}} = \frac{S_{\text{active}}}{S} \times 100\%$$
7. **Inference Latency ($\text{ms}$)**:
   Runtime per network topology instance measured via high-resolution performance timers.
8. **Pairing Validity Rate ($\%$)**:
   Percentage of produced pairings satisfying all hard physical constraints (must equal 100.0%).

---

## 4. Expected Output Artifacts

Running the experiment produces the following structured outputs in `results/`:
- `results/figures/network_topology.png`: Spatial positions, Base Station, and pairing lines.
- `results/figures/training_loss_curves.png`: Training and validation loss convergence curves.
- `results/figures/sum_rate_benchmark.png`: Bar chart comparing Sum Rate and Jain's Fairness.
- `results/figures/scma_ber_curve.png`: Semilogy BER/SER curves for SCMA MPA detector across SNRs.
- `results/metrics/benchmark_results.csv`: Complete numerical metrics across all tested methods.
- `results/metrics/summary.json`: JSON metadata including hyperparameter records and performance aggregates.
- `results/best_gnn_model.pt`: Checkpoint of best GNN model weights.
- `results/classical_models/cross_validation_metrics.csv`: Scenario-grouped CV metrics for the five classical regressors.
- `results/classical_models/hyperparameter_search_results.csv`: Ridge and Random Forest randomized-search candidates.
- `results/classical_models/model_selection_test_metrics.csv` and `paired_bootstrap_comparison.csv`: selected-model held-out metrics and paired scenario-bootstrap interval.
