# Dataset package

This directory contains a reconstructed, synthetic dataset for the project's default GNN scenario setup plus a reproducible EDA report. It has 290 scenario graphs, 2,900 user rows, and 26,100 directed edge rows.

**Important provenance note:** the project does not save raw channel scenarios. The project code uses NumPy `default_rng`; this standard-library generator uses `random.Random`. The files reproduce configured distributions, seed indices, graph schema, and oracle label logic, but are not the exact numerical draws from an earlier run.

Start with [EDA_Report.md](EDA_Report.md), then see [dataset_metadata.json](dataset_metadata.json) for machine-readable statistics. Regenerate using `python Dataset\generate_dataset_eda.py` from the repository root (Python 3.10+; standard library only).

| File | Contents |
|---|---|
| `train_users.csv` | 200 scenarios, 2,000 users |
| `validation_users.csv` | 40 scenarios, 400 users |
| `benchmark_users.csv` | 50 scenarios, 500 users |
| `all_splits_edges.csv` | Directed pair candidates and labels for all splits |
| `scenario_summary.csv` | Per-scenario oracle outcomes |
| `feature_correlations.csv` | Pooled user-feature correlations |
| `figures/` | SVG EDA visualizations |
| `generate_dataset_eda.py` | Standard-library dataset and report generator |

Do not merge rows randomly across train, validation, and benchmark. Keep all users/edges from a scenario together. Benchmark is a simulator evaluation set, not a classic ML test split for the optional regression workflow.
