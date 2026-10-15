import numpy as np
import pytest

from src.config import ProjectConfig
from src.evaluation import BENCHMARK_METHOD_ORDER, NetworkEvaluator
from src.gnn_model import HybridNOMAGNN


def test_ber_sweep_computes_bits_for_all_benchmark_methods():
    cfg = ProjectConfig()
    cfg.network.num_users = 4
    cfg.network.num_slots = 2
    cfg.gnn.hidden_dim = 16
    cfg.validate()
    evaluator = NetworkEvaluator(
        cfg,
        gnn_model=HybridNOMAGNN(cfg.gnn, num_slots=cfg.network.num_slots),
    )
    evaluator.network.set_seed(123)
    scenarios = [evaluator.network.generate_users(num_users=cfg.network.num_users)]

    results = evaluator.evaluate_ber_snr_sweep(
        snr_db_range=[0.0, 10.0],
        num_blocks=8,
        scenarios=scenarios,
    )

    assert list(results.columns) == ["snr_db", "method", "bit_errors", "total_bits", "ber"]
    assert set(results["method"]) == set(BENCHMARK_METHOD_ORDER)
    assert len(results) == 2 * len(BENCHMARK_METHOD_ORDER)
    assert np.all(results["total_bits"] == cfg.network.num_users * 8)
    assert np.all((results["ber"] >= 0.0) & (results["ber"] <= 1.0))
    assert np.allclose(results["ber"], results["bit_errors"] / results["total_bits"])


def test_ber_sweep_rejects_zero_transmission_blocks():
    cfg = ProjectConfig()
    evaluator = NetworkEvaluator(cfg)

    with pytest.raises(ValueError, match="num_blocks must be >= 1"):
        evaluator.evaluate_ber_snr_sweep(num_blocks=0, num_test_instances=1)
