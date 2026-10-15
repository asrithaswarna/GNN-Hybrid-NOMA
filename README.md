# Graph Neural Network-Based Dynamic User Pairing and Slot Assignment for Hybrid NOMA Networks

This project simulates a wireless cell and trains an edge-conditioned Graph Neural Network (GNN) to predict user pairing affinities and resource-slot preferences. A constraint-aware scheduler turns those predictions into a valid hybrid multiple-access schedule using PD-NOMA, SCMA, and OMA fallback. Classical regressors predict per-user simulation-derived rates; they do not replace the GNN scheduler.

## Objectives

- Learn topology-aware user pairing and slot preferences from simulated wireless scenarios.
- Evaluate the GNN against established wireless scheduling baselines on common test scenarios.
- Compare classical rate-regression models and evaluate their generalization without mixing users from the same scenario across folds.

## Features

- Synthetic channel generation (log-distance path loss and Rayleigh fading); no downloaded dataset is required for normal runs.
- Edge-conditioned GNN, oracle-supervised pairing/slot targets, and deterministic schedule constraint handling.
- Hybrid PD-NOMA (with imperfect SIC), SCMA/MPA, and OMA fallback.
- Wireless evaluation against OMA, Random, Near-Far, Greedy Sum-Rate, Optimization Oracle, and the GNN.
- Outage probability comparison and a bit-level uncoded BPSK BER-versus-SNR comparison across the six pairing methods.
- Classical Linear, Ridge, Lasso, Decision Tree, and Random Forest regression for simulated per-user rate.
- Optional K-means/DBSCAN/PCA analysis and a local FastAPI prediction service.

##  Evaluation, Selection, and Calibration

The classical comparison workflow is `python main.py --mode compare`. It uses a train/validation/test split at the scenario level and:

1. **K-fold cross-validation:** up to five `GroupKFold` folds keep all user rows from one `scenario_id` together. Linear, Ridge, Lasso, Decision Tree, and Random Forest are compared by mean and standard deviation of MAE, RMSE, and R². Imputation and scaling live inside scikit-learn pipelines and are fit only on each fold's training rows.
2. **Randomized hyperparameter search:** `RandomizedSearchCV` tunes Ridge and Random Forest with negative MAE on those same training folds. It evaluates four candidates per model using a fixed seed. The model with the better cross-validated score is selected; validation data is not used to select it.
3. **Paired scenario bootstrap:** after selection, the selected estimator and Linear Regression baseline are compared on the held-out test set. The reported difference is `MAE_baseline - MAE_selected`; positive favors the selected model. The 95% percentile interval resamples complete scenarios, keeping paired predictions and correlated user rows together. An interval containing zero is not evidence of a reliable difference; few scenarios trigger a warning.

When there are fewer than two training scenarios, Module 5 CV/tuning is skipped and a status file explains why; the original model comparison is still produced. Missing scenario IDs force row-wise K-fold or bootstrap with explicit warnings because scenario independence cannot be guaranteed.
The default `train` workflow also runs this classical Module 5 comparison for the presentation summary; `--bootstrap-repetitions` controls its bootstrap count as well. The standalone `compare` workflow runs the same classical evaluation without GNN training.

## Project Structure(source code)

.
├── main.py                       # train/evaluate/compare/clustering/inference/retrain CLI
├── config.yaml                   # network, channel, NOMA, GNN, and dataset configuration
├── requirements.txt              # runtime, API, and test dependencies
├── RUN_COMMANDS.md               # Windows and Unix command reference
├── src/
│   ├── __init__.py
│   ├── channel_model.py          # wireless users, propagation, fading
│   ├── graph_builder.py          # graph features and supervised labels
│   ├── gnn_model.py              # hybrid NOMA GNN
│   ├── training.py               # synthetic graph generation and training
│   ├── evaluation.py             # wireless benchmark, SCMA sweep, and pairing-method BER sweep
│   ├── classical_models.py       # regression, CV, tuning, bootstrap, inference
│   ├── data_pipeline.py          # scenario CSV/synthetic data, splitting, features
│   ├── optimization.py           # pairing optimization/oracle
│   ├── pairing.py                # pairing algorithms
│   ├── slot_assignment.py        # schedule and slot evaluation
│   ├── noma_pd.py, noma_scma.py  # PD-NOMA and SCMA engines
│   ├── unsupervised.py           # clustering and PCA
│   ├── config.py, utilities.py   # configuration and common helpers
│   ├── model_io.py, monitoring.py
│   └── visualization.py
├── api/                           # optional FastAPI service and schemas
│   ├── main.py
│   └── schemas.py
├── data/
│   ├── README.md
│   └── example_users.csv
├── tests/                         # pytest suite: GNN, NOMA, pairing, slots, data, API, etc.
├── docs/                          # protocol, methodology, plans, limitations, presentation, viva
├── notebooks/
│   └── hybrid_noma_experiment.ipynb
├── archive/                       # legacy experiment results and logs
└── results*/                      # checked-in/prior results and local experiment outputs


## Prerequisites and Installation

Python **3.9 or newer** is the project's documented baseline. The comparison and GNN smoke runs for the current changes were executed with Python 3.14.3. A fresh dependency installation was not tested as part of that run.

From the project root (the directory containing `main.py`):

**Windows PowerShell**

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python 3.14 is not installed, use an installed Python 3.9+ version supported by the selected PyTorch/scikit-learn wheels.

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` includes PyTorch, NumPy/SciPy, pandas, scikit-learn, NetworkX, Matplotlib, PyYAML, joblib, pytest, and the optional API stack (FastAPI, Pydantic, Uvicorn). The API is optional at runtime; the package is listed so the complete project can be installed with one command.

## Data and Targets

The simulator creates data in code. GNN supervised labels are generated by the optimization oracle; the classical regression target is each user's Shannon-rate estimate (`rate_target_bps`), converted to Mbps for model evaluation. The feature columns are `x`, `y`, `distance`, `channel_gain`, `channel_gain_db`, `snr_db`, and `min_rate_req_bps`; neither target column is directly passed to the regressors. Note that the rate target is deterministically computed from simulated SNR, which is also included as an input feature. Therefore, this experiment evaluates reconstruction of the configured simulator's rate formula; high R² is not evidence of generalization to measured radio data or a target independent of SNR. Scenario identity and user identity are `scenario_id` and `user_id`.

The optional per-user CSV input must contain:

```text
scenario_id,user_id,x,y,distance,channel_gain,min_rate_req_bps
```

Each scenario must have exactly the configured number of users (set `--users` to match), with unique user IDs. `channel_gain` is linear power gain and `distance` is in meters. Derived SNR/rate features and targets are calculated by the pipeline; do not supply fabricated scenario identifiers for correlated users.

For the standard GNN training command, default sample counts are 200 train, 40 validation, and 50 test graphs (configurable in `config.yaml`). Analysis modes generate/load scenarios and split them by scenario; compare mode runs CV and hyperparameter selection on the train portion only. Its held-out test portion is reserved for final metrics and bootstrap.

## Run the Project

The exact copy-paste-ready commands for Windows PowerShell and Linux/macOS are in [RUN_COMMANDS.md](./RUN_COMMANDS.md). Main workflows:

```bash
# Train the GNN, run wireless benchmarks, and print Module 5 classical results
python main.py --users 10 --epochs 25 --slots 5 --seed 42 --output results

# Evaluate a saved GNN checkpoint against the wireless baselines
python main.py --mode evaluate --model-path results/best_gnn_model.pt --users 10 --slots 5 --seed 42 --output results

# Classical regression, grouped CV, randomized search, held-out test, paired bootstrap
python main.py --mode compare --data-source synthetic --users 6 --seed 42 --output results/module5_compare --bootstrap-repetitions 2000

# Optional: use validated scenario CSV for analysis
python main.py --mode compare --data-source csv --dataset data/example_users.csv --users 4 --output results/csv_analysis
```

Other available modes are `clustering`, `inference`, and `retrain`. CLI options and defaults can be checked with `python main.py --help`. Seed controls are `--seed` and `system.seed` in `config.yaml`; the bootstrap repetition count is set with `--bootstrap-repetitions`. The fold cap (five) and search candidates (four per tuned model) are constants in `src/classical_models.py`.

The standard `train` workflow and checkpoint-based `evaluate` workflow produce outage and BER comparison outputs. BER work per test scenario and SNR can be adjusted with `--ber-blocks` (default: 100).

The BER comparison uses uncoded BPSK over common path-loss-plus-Rayleigh test channel scenarios with AWGN and perfect receiver CSI. OMA uses coherent detection on orthogonal transmissions. Paired methods use two-user PD-NOMA superposition, marginal-ML detection for the weak user, and hard successive interference cancellation for the strong user. This compares pairing policies under a shared OMA/PD-NOMA physical layer; it is **not** a BER evaluation of the hybrid scheduler's SCMA mode. The independent SCMA QPSK/MPA BER curve remains available as `scma_ber_curve.png`. See [limitations](./docs/limitations.md#35-pairing-method-ber-comparison) for assumptions and scope.

## Outputs

The canonical full training run is stored in `results/`. Re-running a workflow may replace files
with the same names in its output directory; back up or move the current results first if they must
be retained. Standalone comparison and optional analysis outputs use named subdirectories of
`results/` so they do not replace the canonical GNN checkpoint.

`train` mode writes:

- `best_gnn_model.pt`, `presentation_summary.txt`
- `figures/network_topology.png`, `figures/training_loss_curves.png`, `figures/sum_rate_benchmark.png`
- `figures/outage_comparison.png`, `figures/ber_comparison.png`, `figures/scma_ber_curve.png`
- `metrics/benchmark_results.csv`, `metrics/ber_comparison.csv`, `metrics/summary.json`
- `classical_models/classical_model_comparison.csv`, Module 5 CV/tuning/test/bootstrap CSVs, and fitted model artifacts

`metrics/ber_comparison.csv` contains one row per SNR and supported method, including the observed bit errors, transmitted bit count, and computed BER. `outage_comparison.png` plots the benchmark evaluator's `outage_probability_mean` as a percentage; creating it does not rerun the benchmark.

`compare` mode writes under `<output>/classical_models/`. The documented standalone command uses
`results/module5_compare/`:

- `classical_model_comparison.csv` and existing model/feature-importance artifacts
- `cross_validation_metrics.csv`
- `hyperparameter_search.csv`, `hyperparameter_search_results.csv`
- `model_selection_test_metrics.csv`, `paired_bootstrap_comparison.csv`
- `selected_model.joblib`, `module5_evaluation_status.json`, and `experiment.json`

The preserved standalone comparison from 2026-10-14 is in
`results/module5_compare_final_20261014/`; the latest corrected full GNN run is at the root of
`results/`. An incomplete historical 8-user metrics/plot set is retained separately under
`results/archive/results_b_8users_4slots_5epochs/` and is not part of the canonical run.

MAE is the mean absolute prediction error; RMSE is the square root of mean squared error and penalizes large errors more strongly; R² measures explained target variance (1 is perfect, 0 is the constant-mean reference, and negative values can occur). Cross-validation summaries measure training-set fold performance, not final test performance. `hyperparameter_search.csv` records each tuned estimator's best negative-MAE CV score and parameters. The paired-bootstrap interval describes test-scenario uncertainty for the MAE difference, not wireless sum-rate uncertainty.

Wireless sum rate, Jain fairness, outage, SCMA BER/SER, and inference time are domain evaluation metrics and are not replaced by regression scores.

## Tests and Troubleshooting

```bash
python -m pytest tests -v
python -c "import main; import src.classical_models; import src.evaluation; print('imports OK')"
```

The current full test suite passed (41 tests). Figures are rendered with Matplotlib's non-interactive Agg backend on Windows and headless systems. If the scenario-aware comparison warns about too few groups, increase available independent scenarios in `config.yaml` or use a larger CSV dataset with distinct scenarios. The synthetic channel assumptions are not measured hardware results, and a small test-scenario bootstrap can have an unstable interval.

## Documentation

- [Experiment protocol](./docs/experiment_protocol.md)
- [Presentation guide](./docs/presentation_guide.md)
- [Mathematical model](./docs/mathematical_model.md)
- [Limitations](./docs/limitations.md)
