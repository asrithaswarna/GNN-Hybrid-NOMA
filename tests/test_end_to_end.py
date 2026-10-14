"""
Integration and end-to-end tests for full pipeline execution and invalid configuration checks.
"""

import pytest
from src.config import ProjectConfig, NetworkConfig, PDNOMAConfig, SCMAConfig
from src.training import DatasetGenerator, GNNTrainer
from src.gnn_model import HybridNOMAGNN
from src.evaluation import NetworkEvaluator


def test_invalid_configurations_raise_exceptions():
    """Verify that erroneous parameter configurations are caught by strict validators."""
    # 1. num_users < 2
    with pytest.raises(ValueError, match="num_users must be >= 2"):
        NetworkConfig(num_users=1).validate()

    # 2. cell_radius_min >= cell_radius_max
    with pytest.raises(ValueError, match="Invalid cell radius"):
        NetworkConfig(cell_radius_min=600.0, cell_radius_max=500.0).validate()

    # 3. alpha_weak + alpha_strong != 1.0
    with pytest.raises(ValueError, match="alpha_weak \\+ alpha_strong must sum to 1.0"):
        PDNOMAConfig(alpha_weak=0.6, alpha_strong=0.6).validate()

    # 4. alpha_weak <= alpha_strong
    with pytest.raises(ValueError, match="Weak user must receive more power"):
        PDNOMAConfig(alpha_weak=0.4, alpha_strong=0.6).validate()

    # 5. sic_residual_factor outside [0, 1]
    with pytest.raises(ValueError, match="sic_residual_factor must be in"):
        PDNOMAConfig(sic_residual_factor=-0.1).validate()

    # 6. SCMA overloading violated (J <= K)
    with pytest.raises(ValueError, match="SCMA requires overloading"):
        SCMAConfig(num_layers=4, num_resources=4).validate()


def test_small_end_to_end_pipeline():
    """Verify that full pipeline (dataset -> train -> eval) executes without error on small settings."""
    cfg = ProjectConfig()
    cfg.network.num_users = 6
    cfg.network.num_slots = 3
    cfg.gnn.epochs = 2
    cfg.gnn.hidden_dim = 16
    cfg.dataset.num_train_samples = 4
    cfg.dataset.num_val_samples = 2
    cfg.dataset.num_test_samples = 3
    cfg.validate()

    # Generate data
    data_gen = DatasetGenerator(cfg)
    train_graphs = data_gen.generate_split(cfg.dataset.num_train_samples, seed_offset=10)
    val_graphs = data_gen.generate_split(cfg.dataset.num_val_samples, seed_offset=20)

    assert len(train_graphs) == 4
    assert len(val_graphs) == 2

    # Train model
    model = HybridNOMAGNN(cfg.gnn, num_slots=cfg.network.num_slots)
    trainer = GNNTrainer(model, cfg, device="cpu")
    history = trainer.train(train_graphs, val_graphs, epochs=2)

    assert history.epochs_trained == 2
    assert len(history.train_losses) == 2

    # Evaluate
    evaluator = NetworkEvaluator(cfg, gnn_model=model)
    benchmark = evaluator.evaluate_test_set(num_test_instances=3, seed_offset=30)

    assert "Proposed GNN-Based NOMA" in benchmark
    assert "Optimization Oracle" in benchmark
    assert "Orthogonal Multiple Access (OMA)" in benchmark

    for name, metric in benchmark.items():
        assert metric.sum_rate_mean_mbps > 0.0
        assert 0.0 <= metric.jains_fairness_mean <= 1.0
        assert metric.validity_rate == 100.0
