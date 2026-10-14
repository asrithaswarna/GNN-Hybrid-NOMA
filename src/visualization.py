"""
Publication-quality visualization module using Matplotlib.
Generates network topology plots, loss curves, bar chart comparisons,
fairness benchmarks, and SCMA BER vs SNR curves.
"""

import os
import platform
import subprocess
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import matplotlib

# Prefer a GUI backend on desktop machines so figures can visibly open after running.
# Fall back to a non-interactive backend in headless environments such as servers or CI.
if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or platform.system() == "Windows":
    for backend in ("TkAgg", "QtAgg", "macosx", "WXAgg"):
        try:
            matplotlib.use(backend, force=True)
            break
        except Exception:
            continue
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.channel_model import UserChannelState
from src.pairing import PairingPlan
from src.training import TrainingHistory
from src.noma_scma import SCMAResult
from src.evaluation import MethodBenchmarkMetrics


class Visualizer:
    """Creates and saves academic figures for B.Tech project presentation and report."""

    def __init__(self, output_dir: str = "results/figures", show_plots: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.show_plots = show_plots
        # Professional styling
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
        })

    def _show_if_possible(self, fig, image_path: Optional[str] = None):
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return
        if not self.show_plots:
            return

        target_path = image_path or getattr(fig, "filename", None)

        try:
            if platform.system() == "Windows":
                if target_path and hasattr(os, "startfile"):
                    os.startfile(target_path)
                    return
            elif platform.system() == "Darwin":
                if target_path:
                    subprocess.run(["open", target_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
            elif target_path:
                subprocess.run(["xdg-open", target_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
        except Exception:
            pass

        if platform.system() == "Windows" or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            try:
                plt.show(block=False)
                plt.pause(0.001)
                return
            except Exception:
                pass

        if target_path:
            print(f"Plot saved and available at: {target_path}")

        try:
            plt.close(fig)
        except Exception:
            pass

    def plot_network_topology(
        self,
        users: List[UserChannelState],
        pairing_plan: PairingPlan,
        cell_radius_max: float = 500.0,
        filename: str = "network_topology.png",
    ) -> str:
        """Visualizes Base Station, users, cell boundary, and pairing connections."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw cell boundary
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(cell_radius_max * np.cos(theta), cell_radius_max * np.sin(theta), "k--", alpha=0.5, label="Cell Boundary (500m)")

        # Draw Base Station at origin
        ax.scatter([0], [0], color="red", marker="^", s=250, label="Base Station (BS)", zorder=5)

        # Plot users
        x_coords = [u.x for u in users]
        y_coords = [u.y for u in users]
        gains_db = [u.channel_gain_db for u in users]

        sc = ax.scatter(x_coords, y_coords, c=gains_db, cmap="viridis", s=100, edgecolors="black", label="Users (UEs)", zorder=4)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Channel Gain (dB)")

        # Annotate user IDs
        for u in users:
            ax.annotate(f"U{u.user_id}", (u.x + 8, u.y + 8), fontsize=9, fontweight="bold")

        # Draw pairing lines
        user_dict = {u.user_id: u for u in users}
        for pair in pairing_plan.pairs:
            u1 = user_dict[pair.user1_id]
            u2 = user_dict[pair.user2_id]
            ax.plot([u1.x, u2.x], [u1.y, u2.y], color="crimson", linestyle="-", linewidth=2.0, alpha=0.8, zorder=3)

        # Highlight unpaired users
        for u_id in pairing_plan.unpaired_users:
            unp = user_dict[u_id]
            ax.scatter([unp.x], [unp.y], facecolors="none", edgecolors="orange", s=250, linewidth=2.5, label="Unpaired User", zorder=4)

        ax.set_title(f"Hybrid NOMA Network Topology & Dynamic Pairing ({pairing_plan.method_name})")
        ax.set_xlabel("X Coordinate (meters)")
        ax.set_ylabel("Y Coordinate (meters)")
        ax.set_xlim(-cell_radius_max * 1.1, cell_radius_max * 1.1)
        ax.set_ylim(-cell_radius_max * 1.1, cell_radius_max * 1.1)
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300)
        self._show_if_possible(fig, str(out_path))
        return str(out_path)

    def plot_training_history(
        self,
        history: TrainingHistory,
        filename: str = "training_loss_curves.png",
    ) -> str:
        """Plots GNN training and validation loss convergence."""
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = range(1, history.epochs_trained + 1)
        ax.plot(epochs, history.train_losses, "b-", linewidth=2, label="Training Loss")
        ax.plot(epochs, history.val_losses, "r--", linewidth=2, label="Validation Loss")
        ax.set_title("GNN Training & Validation Convergence Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Joint Loss (BCE + Cross-Entropy)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300)
        self._show_if_possible(fig, str(out_path))
        return str(out_path)

    def plot_benchmark_comparison(
        self,
        metrics: Dict[str, MethodBenchmarkMetrics],
        filename: str = "sum_rate_benchmark.png",
    ) -> str:
        """Plots bar chart comparing Sum Rate and Jain's Fairness across algorithms."""
        methods = list(metrics.keys())
        # Shorten method names for clean x-axis
        labels = [m.replace("Orthogonal Multiple Access (OMA)", "OMA").replace("PD-NOMA", "NOMA").replace("Channel Gain", "Gain") for m in methods]
        sum_rates = [metrics[m].sum_rate_mean_mbps for m in methods]
        sum_stds = [metrics[m].sum_rate_std_mbps for m in methods]
        fairness = [metrics[m].jains_fairness_mean for m in methods]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bar chart for Sum Rate
        bars = ax1.bar(x - width/2, sum_rates, width, yerr=sum_stds, capsize=5, label="Sum Rate (Mbps)", color="#1f77b4", alpha=0.85)
        ax1.set_ylabel("Network Sum Rate (Mbps)", color="#1f77b4", fontweight="bold")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=20, ha="right", fontweight="bold")

        # Secondary y-axis for Jain's Fairness
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, fairness, width, label="Jain's Fairness", color="#2ca02c", alpha=0.85)
        ax2.set_ylabel("Jain's Fairness Index", color="#2ca02c", fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="#2ca02c")
        ax2.set_ylim(0, 1.1)

        # Annotate values
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        plt.title("Comparative Performance: Sum-Rate vs. Jain's Fairness Index", fontweight="bold")
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300)
        self._show_if_possible(fig, str(out_path))
        return str(out_path)

    def plot_scma_ber_curve(
        self,
        scma_results: List[SCMAResult],
        filename: str = "scma_ber_curve.png",
    ) -> str:
        """Plots SCMA Message Passing Algorithm (MPA) BER vs SNR curve."""
        snrs = [r.snr_db for r in scma_results]
        bers = [max(r.bit_error_rate, 1e-4) for r in scma_results]
        sers = [max(r.symbol_error_rate, 1e-4) for r in scma_results]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(snrs, sers, "s--", color="#ff7f0e", linewidth=2, label="Symbol Error Rate (SER)")
        ax.semilogy(snrs, bers, "o-", color="#d62728", linewidth=2, label="Bit Error Rate (BER)")
        ax.set_title("SCMA MPA Detector Error Rate vs. Transmit SNR (150% Overloading)")
        ax.set_xlabel("Signal-to-Noise Ratio (dB)")
        ax.set_ylabel("Error Probability (Log Scale)")
        ax.set_ylim(5e-5, 1.0)
        ax.legend()
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300)
        self._show_if_possible(fig, str(out_path))
        return str(out_path)
