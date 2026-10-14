# Version C Implementation Plan

## Existing architecture

Version C is a command-line Python project. `main.py` loads `config.yaml`, generates synthetic wireless network graphs with `DatasetGenerator.generate_split()`, trains `HybridNOMAGNN`, evaluates OMA/PD-NOMA/oracle/GNN methods with `NetworkEvaluator`, runs the separate SCMA detector experiment, and saves plots, CSV metrics, JSON metadata, and a PyTorch checkpoint.

The reusable core data structures are `UserChannelState`, `NetworkGraph`, `PairingPlan`, `NetworkScheduleResult`, and `MethodBenchmarkMetrics`. The current default data source is synthetic channel simulation. SCMA is currently evaluated separately from the PD-NOMA scheduler; this plan will preserve that honest scope.

## Planned modifications

### Stage 1: Data and lifecycle infrastructure
- Add `src/data_pipeline.py` for input validation, synthetic sample extraction, optional per-user CSV loading, feature engineering, and train/validation/test scenario splitting.
- Define and document a CSV schema based on `UserChannelState` fields. Validate file existence, required columns, numeric values, missing values, duplicate user IDs within scenarios, user counts, units, and ranges.
- Keep synthetic generation as the default and record data source metadata.
- Add reusable preprocessing fitted only on training data and reused for validation, test, and inference.

### Stage 2: Classical supervised models
- Add `src/classical_models.py` with a small, genuine regression task: predict simulated single-user achievable rate from validated channel/user features.
- Train and compare Ridge Regression, Decision Tree Regression, and Random Forest Regression on the same scenario-level split.
- Report MAE, MSE, RMSE, and R2; save model artifacts and feature importance for tree models.
- Keep this task separate from the GNN pairing objective and label the comparison accordingly.

### Stage 3: Unsupervised analysis
- Add `src/unsupervised.py` with standardized user/channel features, K-means, DBSCAN, and PCA.
- Save assignments, PCA explained variance, cluster summaries, and plots.
- Compare clusters descriptively with GNN/near-far pairing without claiming that clustering is optimal scheduling.

### Stage 4: Evaluation and CLI workflow
- Add consistent model-comparison reports containing model, task, split, source, seed, metrics, and timing.
- Extend the CLI with `--mode train|evaluate|inference|clustering|compare|retrain`, `--data-source synthetic|csv`, and `--dataset` while preserving existing commands.
- Keep presentation and detailed output modes. Ensure final summary metrics come from the same identified evaluation result.

### Stage 5: ML engineering
- Add model save/load and inference helpers for classical artifacts and the existing GNN checkpoint.
- Add lightweight JSON/CSV monitoring for prediction count, invalid inputs, latency, and feature/prediction summaries.
- Add a retraining workflow that evaluates a candidate against the existing artifact and reports acceptance rather than silently replacing a model.
- Add an optional FastAPI API with `/health`, `/model-info`, and `/predict`. The normal experiment command will not start the service.

### Stage 6: Documentation and tests
- Update README, data schema documentation, methodology, limitations, experiment protocol, B.Tech report guide, presentation guide, and viva answers.
- Add tests for CSV validation, preprocessing consistency, classical models, clustering, metric consistency, packaging, monitoring, and API behavior where dependencies are available.
- Run existing tests, new tests, compile checks, synthetic workflow, CSV workflow, compare, clustering, inference, and API smoke checks.

## M1-M6 mapping

| Module | Implemented concepts | Planned evidence |
|---|---|---|
| M1 | End-to-end lifecycle; training-serving boundary | `data_pipeline.py`, CLI modes, artifacts, monitoring, lifecycle docs |
| M2 | Ridge/linear regression; supervised preprocessing and regression metrics | `classical_models.py`, compare command, saved metrics/tests |
| M3 | Decision Tree; Random Forest and feature importance | `classical_models.py`, comparison CSV/JSON/tests |
| M4 | K-means; DBSCAN; PCA | `unsupervised.py`, cluster outputs/plots/tests |
| M5 | Scenario train/validation/test split; standard and wireless metric separation | data split/reporting modules, consistency tests, benchmark outputs |
| M6 | Model packaging/reproducibility; inference/API/monitoring/retraining | `src/model_io.py`, `src/monitoring.py`, `api/`, CLI modes, tests |

## New files

- `docs/implementation_plan.md`
- `src/data_pipeline.py`
- `src/classical_models.py`
- `src/unsupervised.py`
- `src/model_io.py`
- `src/monitoring.py`
- `api/main.py`
- `api/schemas.py`
- Focused tests for each new module
- Optional example CSV only if it is clearly marked as synthetic

## Modified files

- `main.py`
- `requirements.txt`
- `README.md`
- `data/README.md`
- Relevant methodology, limitations, experiment, report, presentation, and viva documentation
- `config.yaml` only where new settings are genuinely needed

## Expected commands

```text
python main.py --mode train --data-source synthetic --users 6 --epochs 2 --slots 3 --output test-output/train-smoke
python main.py --mode compare --data-source synthetic --users 6 --output results/module5_compare
python main.py --mode clustering --data-source synthetic --users 6 --output results/clustering
python main.py --mode inference --model-path results/best_gnn_model.pt --output results/inference
python main.py --mode retrain --data-source synthetic --output results/retrain
uvicorn api.main:app --reload
pytest -q
```

## Known limitations

- The current GNN evaluates PD-NOMA pair scheduling; SCMA remains a separate detector experiment and is not silently presented as integrated scheduling.
- Classical models predict a clearly defined simulated user-rate target and are not claimed to replace the GNN pairing model.
- CSV support will require a complete per-user scenario schema and does not turn synthetic results into real-world validation.
- FastAPI and scikit-learn are optional runtime dependencies; core GNN/NOMA execution should remain usable when optional analysis features are not installed.
- Monitoring is local lightweight logging, not production observability.
