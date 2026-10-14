from src.training import TrainingHistory
from src.visualization import Visualizer


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
