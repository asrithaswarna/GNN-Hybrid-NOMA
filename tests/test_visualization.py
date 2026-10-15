from src.training import TrainingHistory
from src.visualization import Visualizer
from src.evaluation import MethodBenchmarkMetrics


def test_visualizer_shows_plot_after_saving(tmp_path, monkeypatch):
    visualizer = Visualizer(output_dir=str(tmp_path))
    history = TrainingHistory(
        train_losses=[1.0, 0.5],
        val_losses=[1.1, 0.6],
        best_val_loss=0.6,
        epochs_trained=2,
    )

    calls = []
    monkeypatch.setattr(visualizer, "_show_if_possible", lambda fig, image_path=None: calls.append((fig, image_path)))

    output_path = visualizer.plot_training_history(history, filename="loss.png")

    assert output_path.endswith("loss.png")
    assert len(calls) == 1
    assert calls[0][1].endswith("loss.png")


def test_outage_comparison_uses_mean_probability_and_skips_invalid_values(tmp_path, capsys):
    visualizer = Visualizer(output_dir=str(tmp_path))
    metrics = {
        "Orthogonal Multiple Access (OMA)": MethodBenchmarkMetrics(
            method_name="OMA", sum_rate_mean_mbps=1.0, sum_rate_std_mbps=0.0,
            user_rate_mean_mbps=1.0, user_rate_std_mbps=0.0,
            min_user_rate_mean_mbps=1.0, jains_fairness_mean=1.0,
            jains_fairness_std=0.0, outage_probability_mean=0.25,
            slot_utilization_mean=100.0, inference_time_ms=1.0, validity_rate=100.0,
        ),
        "Random Pairing PD-NOMA": MethodBenchmarkMetrics(
            method_name="Random", sum_rate_mean_mbps=1.0, sum_rate_std_mbps=0.0,
            user_rate_mean_mbps=1.0, user_rate_std_mbps=0.0,
            min_user_rate_mean_mbps=1.0, jains_fairness_mean=1.0,
            jains_fairness_std=0.0, outage_probability_mean=float("nan"),
            slot_utilization_mean=100.0, inference_time_ms=1.0, validity_rate=100.0,
        ),
    }

    output_path = visualizer.plot_outage_comparison(metrics)

    assert output_path.endswith("outage_comparison.png")
    assert (tmp_path / "outage_comparison.png").is_file()
    assert "Skipping outage plot value" in capsys.readouterr().out


def test_ber_comparison_plot_creates_file_with_computed_results(tmp_path):
    import pandas as pd

    visualizer = Visualizer(output_dir=str(tmp_path))
    results = pd.DataFrame([
        {"snr_db": 0.0, "method": "OMA", "bit_errors": 2, "total_bits": 100, "ber": 0.02},
        {"snr_db": 10.0, "method": "OMA", "bit_errors": 0, "total_bits": 100, "ber": 0.0},
    ])

    output_path = visualizer.plot_ber_comparison(results)

    assert output_path.endswith("ber_comparison.png")
    assert (tmp_path / "ber_comparison.png").is_file()
