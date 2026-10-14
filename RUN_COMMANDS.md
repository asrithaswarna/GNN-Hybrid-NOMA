# Run Commands

Run every command from the project root: the directory containing `main.py` and `requirements.txt`.
The project documents Python 3.9+; the current code changes were run with Python 3.14.3. Dependency
installation in a clean environment has not been tested.

## 1. Create and activate an environment

### Windows PowerShell

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python 3.14 is not installed, use `py -3.12` (or another installed Python 3.9+ version with
compatible PyTorch and scikit-learn wheels) in the first command.

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No dataset download or preparation is required for synthetic experiments. To use CSV data, prepare
one row per user with the columns `scenario_id,user_id,x,y,distance,channel_gain,min_rate_req_bps`.
Scenario IDs must identify independent simulated/captured networks, and each scenario must contain
exactly `--users` rows with distinct user IDs.

## 2. Standard GNN training and wireless benchmark (normal run)

This trains the GNN, evaluates the wireless baselines and SCMA SNR sweep, runs the classical
Module 5 evaluation, prints its CV/selection/test/bootstrap summary, and writes figures and metrics
under `results/`:

```powershell
python main.py --users 10 --epochs 25 --slots 5 --seed 42 --output results
```

The same command works in Bash. Keep smoke outputs separate from presentation results:

```powershell
python main.py --users 6 --epochs 2 --slots 3 --seed 42 --output test-output/train-smoke
```

The smoke command completes two epochs because `--epochs 2` sets the requested training limit;
the trainer does not stop early. Use `--epochs 25` (or another chosen count) for a longer run.
The summary reports the actual lowest-validation-loss epoch separately from total epochs completed.
Use `--detailed-output` for technical console logs or `--show-plots` to request opening saved plots
on a desktop with a working GUI.

The default bootstrap count is 2,000. For a smaller smoke run, pass
`--bootstrap-repetitions 200`; this setting applies to both GNN `train` mode's classical evaluation
and standalone `compare` mode.

## 3. Evaluate a saved GNN checkpoint (optional)

Use the checkpoint produced by training:

```powershell
python main.py --mode evaluate --model-path results/best_gnn_model.pt --users 10 --slots 5 --seed 42 --output results
```

The evaluator prints wireless metrics for OMA, Random Pairing, Near-Far, Greedy Sum-Rate,
Optimization Oracle, and the GNN. This mode uses generated test scenarios and displays metrics in
the terminal; its current implementation does not write a separate evaluation CSV.

## 4. Classical regression and all Module 5 evaluation steps (optional)

The normal GNN training command already performs these classical steps. Run compare mode alone
when you want the classical comparison without training the GNN. It performs the existing
regression comparison, scenario-grouped CV, randomized Ridge/Random Forest search, final held-out
test evaluation, and paired scenario bootstrap:

```powershell
python main.py --mode compare --data-source synthetic --users 6 --seed 42 --output results/module5_compare --bootstrap-repetitions 2000
```

For the validated example CSV (four users per scenario):

```powershell
python main.py --mode compare --data-source csv --dataset data/example_users.csv --users 4 --seed 42 --output results/csv_analysis --bootstrap-repetitions 2000
```

There is no separate CLI command for individual CV, search, or bootstrap: they are run together by
`--mode compare`. CV is capped at five GroupKFold folds. Hyperparameter search tests four randomized
candidates each for Ridge and Random Forest using negative MAE, with fixed `--seed`. The bootstrap
defaults to 2,000 repetitions; lower it for a quick smoke check. More scenarios improve the
stability of grouped CV and bootstrap inference. Training/CV/search never use test metrics for
selection.

## 5. Other optional modes

```powershell
# K-means, DBSCAN, and PCA
python main.py --mode clustering --data-source synthetic --users 6 --seed 42 --output results/clustering

# Predict using a classical model artifact produced by compare/train
python main.py --mode inference --model-path results/classical_models/random_forest.joblib --users 10 --seed 42 --output results

# Create a retraining candidate without replacing the installed model
python main.py --mode retrain --data-source synthetic --users 6 --seed 42 --output results/retrain

# Optional local API; requires FastAPI/Uvicorn and a model at the API's configured path
uvicorn api.main:app --reload
```

The API reads `results/classical_models/random_forest.joblib`; run the normal train or compare
workflow with `--output results` before calling the API.

## 6. Tests and import check

```powershell
python -m pytest tests -v
python -c "import main; import src.classical_models; import src.evaluation; print('imports OK')"
```

The current full-suite run completed with 37 passed. Test-generated files use pytest temporary
directories and are not written into the presentation results.

## 7. Output locations

For `--output results`:

- GNN: `best_gnn_model.pt`, `presentation_summary.txt`
- Figures: `figures/network_topology.png`, `figures/training_loss_curves.png`,
  `figures/sum_rate_benchmark.png`, `figures/scma_ber_curve.png`
- Wireless metrics: `metrics/benchmark_results.csv`, `metrics/summary.json`
- Classical comparison and artifacts: `classical_models/`

For `--mode compare --output results/module5_compare`:

- `results/module5_compare/classical_models/classical_model_comparison.csv`
- `results/module5_compare/classical_models/cross_validation_metrics.csv`
- `results/module5_compare/classical_models/hyperparameter_search.csv`
- `results/module5_compare/classical_models/hyperparameter_search_results.csv`
- `results/module5_compare/classical_models/model_selection_test_metrics.csv`
- `results/module5_compare/classical_models/paired_bootstrap_comparison.csv`
- `results/module5_compare/classical_models/selected_model.joblib`
- `results/module5_compare/classical_models/module5_evaluation_status.json`

The preserved 2026-10-14 standalone comparison is under
`results/module5_compare_final_20261014/classical_models/`. The standard full training command
writes to `results/` and can replace same-named files there; move or back up that folder before
starting a new canonical run if you need to retain the previous results.

## 8. Reproducibility and configuration

Use the same `--seed` to reproduce a run. `config.yaml` stores system/network settings and default
dataset sample counts (`num_train_samples`, `num_val_samples`, `num_test_samples`). The compare
workflow groups by `scenario_id`, fits imputation/scaling within each fold, tunes on training
scenarios, and reserves the test scenarios for the final metrics and paired bootstrap. The bootstrap
difference is `MAE_baseline - MAE_selected`; positive values favor the selected estimator. Read a
small-sample warning as a limitation on inference rather than a claim of significance.
