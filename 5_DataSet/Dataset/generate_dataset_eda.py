#!/usr/bin/env python3
"""Rebuild a documented synthetic scenario dataset and its exploratory summaries.

Uses only the Python standard library. The project does not save the raw random
scenarios used by earlier runs, so this recreates their configured model and
seed plan; it cannot recover the exact NumPy-generated historical samples.
"""

from __future__ import annotations

import csv
import html
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
SEED = 42
NUM_USERS = 10
NUM_SLOTS = 5
TRAIN_COUNT = 200
VALIDATION_COUNT = 40
BENCHMARK_COUNT = 50
R_MIN_M = 20.0
R_MAX_M = 500.0
TOTAL_BANDWIDTH_HZ = 10.0e6
TOTAL_POWER_W = 1.0
NOISE_DENSITY_DBM_HZ = -174.0
NOISE_FIGURE_DB = 5.0
PATH_LOSS_EXPONENT = 3.5
REFERENCE_DISTANCE_M = 1.0
REFERENCE_PATH_LOSS_DB = 38.4
FADING_VARIANCE = 1.0
MIN_RATE_BPS = 1.0e5
ALPHA_WEAK = 0.75
ALPHA_STRONG = 0.25
SIC_RESIDUAL = 0.01
GAIN_RATIO_THRESHOLD = 1.5
FAIRNESS_WEIGHT = 0.2
NODE_FEATURES = [
    "node_x_norm",
    "node_y_norm",
    "node_distance_norm",
    "node_gain_db_zscore",
    "node_gain_relative_mean",
    "node_min_rate_mbps",
]
EDGE_FEATURES = [
    "edge_distance_norm",
    "edge_gain_disparity_norm",
    "edge_channel_correlation",
    "edge_interference_potential",
]
EDA_FEATURES = [
    "x_m",
    "y_m",
    "distance_m",
    "channel_gain_linear",
    "channel_gain_db",
    "snr_db",
    "min_rate_req_bps",
]


def noise_power_per_slot() -> float:
    density_w_hz = 10.0 ** ((NOISE_DENSITY_DBM_HZ - 30.0) / 10.0)
    return (
        density_w_hz
        * 10.0 ** (NOISE_FIGURE_DB / 10.0)
        * TOTAL_BANDWIDTH_HZ
        / NUM_SLOTS
    )


def make_scenario(seed: int) -> list[dict]:
    """Generate one area-uniform topology and i.i.d. complex Rayleigh fading."""
    rng = random.Random(seed)
    u_values = [rng.random() for _ in range(NUM_USERS)]
    radii = [
        math.sqrt(
            value * (R_MAX_M**2 - R_MIN_M**2) + R_MIN_M**2
        )
        for value in u_values
    ]
    angles = [rng.uniform(0.0, 2.0 * math.pi) for _ in range(NUM_USERS)]
    sigma = math.sqrt(FADING_VARIANCE / 2.0)
    fading_real = [rng.gauss(0.0, sigma) for _ in range(NUM_USERS)]
    fading_imag = [rng.gauss(0.0, sigma) for _ in range(NUM_USERS)]

    noise = noise_power_per_slot()
    slot_power = TOTAL_POWER_W / NUM_SLOTS
    users = []
    for user_id, radius in enumerate(radii):
        x = radius * math.cos(angles[user_id])
        y = radius * math.sin(angles[user_id])
        path_loss_db = REFERENCE_PATH_LOSS_DB + 10 * PATH_LOSS_EXPONENT * math.log10(
            max(radius, REFERENCE_DISTANCE_M) / REFERENCE_DISTANCE_M
        )
        path_loss_linear = 10.0 ** (-path_loss_db / 10.0)
        h_real = math.sqrt(path_loss_linear) * fading_real[user_id]
        h_imag = math.sqrt(path_loss_linear) * fading_imag[user_id]
        gain = h_real * h_real + h_imag * h_imag
        gain_db = 10.0 * math.log10(gain) if gain > 1.0e-30 else -300.0
        snr = slot_power * gain / noise if noise > 0.0 else 0.0
        snr_db = 10.0 * math.log10(snr) if snr > 1.0e-30 else -300.0
        users.append(
            {
                "user_id": user_id,
                "x_m": x,
                "y_m": y,
                "distance_m": radius,
                "path_loss_db": path_loss_db,
                "path_loss_linear": path_loss_linear,
                "fading_real": fading_real[user_id],
                "fading_imag": fading_imag[user_id],
                "channel_real": h_real,
                "channel_imag": h_imag,
                "channel_gain_linear": gain,
                "channel_gain_db": gain_db,
                "snr_linear": snr,
                "snr_db": snr_db,
                "min_rate_req_bps": MIN_RATE_BPS,
            }
        )

    mean_gain = max(statistics.fmean(u["channel_gain_linear"] for u in users), 1.0e-18)
    mean_db = statistics.fmean(u["channel_gain_db"] for u in users)
    std_db = max(statistics.pstdev(u["channel_gain_db"] for u in users), 1.0)
    for user in users:
        user.update(
            {
                "node_x_norm": user["x_m"] / R_MAX_M,
                "node_y_norm": user["y_m"] / R_MAX_M,
                "node_distance_norm": user["distance_m"] / R_MAX_M,
                "node_gain_db_zscore": (user["channel_gain_db"] - mean_db) / std_db,
                "node_gain_relative_mean": user["channel_gain_linear"] / mean_gain,
                "node_min_rate_mbps": user["min_rate_req_bps"] / 1.0e6,
            }
        )
        bandwidth = TOTAL_BANDWIDTH_HZ / NUM_SLOTS
        user["rate_target_bps"] = bandwidth * math.log2(1.0 + user["snr_linear"])
        user["outage_target"] = int(
            user["rate_target_bps"] < user["min_rate_req_bps"]
        )
    return users


def all_pair_partitions(users: tuple[int, ...]) -> Iterable[tuple[tuple[int, int], ...]]:
    if len(users) < 2:
        yield ()
        return
    first = users[0]
    for index in range(1, len(users)):
        second = users[index]
        remaining = users[1:index] + users[index + 1 :]
        for sub in all_pair_partitions(remaining):
            yield ((first, second),) + sub


def slot_quality(slot: int) -> float:
    phase = (slot + 1) * math.pi / NUM_SLOTS
    return 0.7 + 0.45 * math.sin(phase + 0.45)


def slot_noise_factor(slot: int) -> float:
    phase = (slot + 1) * math.pi / NUM_SLOTS
    return 1.0 + 0.35 * math.cos(phase + 1.1)


def pair_rates(
    first: dict, second: dict, slot: int
) -> tuple[str, dict[int, float], float]:
    gain_scale = slot_quality(slot)
    gain_a = max(first["channel_gain_linear"] * gain_scale, 1.0e-30)
    gain_b = max(second["channel_gain_linear"] * gain_scale, 1.0e-30)
    power = TOTAL_POWER_W / NUM_SLOTS
    bandwidth = TOTAL_BANDWIDTH_HZ / NUM_SLOTS
    noise = noise_power_per_slot() * slot_noise_factor(slot)
    weak_id, strong_id = (
        (first["user_id"], second["user_id"])
        if gain_a <= gain_b
        else (second["user_id"], first["user_id"])
    )
    weak_gain = min(gain_a, gain_b)
    strong_gain = max(gain_a, gain_b)
    sinr_weak = (
        ALPHA_WEAK * power * weak_gain
        / (ALPHA_STRONG * power * weak_gain + noise)
    )
    sinr_strong = (
        ALPHA_STRONG * power * strong_gain
        / (SIC_RESIDUAL * ALPHA_WEAK * power * strong_gain + noise)
    )
    weak_rate = bandwidth * math.log2(1.0 + sinr_weak)
    strong_rate = bandwidth * math.log2(1.0 + sinr_strong)

    scma_interference = 0.18 * (gain_a + gain_b) + noise
    scma_sinr = power * (gain_a + gain_b) / max(scma_interference, 1.0e-30)
    scma_rate = bandwidth * math.log2(1.0 + scma_sinr / 2.0)
    mode = "SCMA" if scma_rate > weak_rate + strong_rate else "PD-NOMA"

    if mode == "PD-NOMA":
        return mode, {weak_id: weak_rate, strong_id: strong_rate}, weak_rate + strong_rate
    return (
        mode,
        {first["user_id"]: scma_rate / 2.0, second["user_id"]: scma_rate / 2.0},
        scma_rate,
    )


def oracle_labels(users: list[dict]) -> tuple[dict, float, float, float, int]:
    """Match the project's exhaustive small-N oracle and slot schedule semantics."""
    by_id = {u["user_id"]: u for u in users}
    best_utility = -1.0e9
    best_pairs = None
    best_rates = None
    best_modes = None
    for pairs in all_pair_partitions(tuple(range(NUM_USERS))):
        rates = {user_id: 0.0 for user_id in range(NUM_USERS)}
        modes = []
        for slot, (user_a, user_b) in enumerate(pairs):
            mode, pair_user_rates, _ = pair_rates(
                by_id[user_a], by_id[user_b], slot
            )
            rates.update(pair_user_rates)
            modes.append(mode)
        total_rate = sum(rates.values())
        sum_squares = sum(value * value for value in rates.values())
        fairness = (
            total_rate**2 / (NUM_USERS * sum_squares)
            if sum_squares > 0.0
            else 1.0
        )
        utility = total_rate + FAIRNESS_WEIGHT * fairness * 1.0e6
        if utility > best_utility:
            best_utility = utility
            best_pairs = pairs
            best_rates = rates
            best_modes = modes
    assert best_pairs is not None and best_rates is not None and best_modes is not None
    partner_by_user = {}
    slot_by_user = {}
    mode_by_user = {}
    for slot, ((first, second), mode) in enumerate(zip(best_pairs, best_modes)):
        partner_by_user[first] = second
        partner_by_user[second] = first
        slot_by_user[first] = slot
        slot_by_user[second] = slot
        mode_by_user[first] = mode
        mode_by_user[second] = mode
    total_rate = sum(best_rates.values())
    sum_squares = sum(value * value for value in best_rates.values())
    fairness = total_rate**2 / (NUM_USERS * sum_squares) if sum_squares else 1.0
    outage_count = sum(
        best_rates[user_id] < by_id[user_id]["min_rate_req_bps"]
        for user_id in range(NUM_USERS)
    )
    return (
        {
            user_id: {
                "partner": partner_by_user[user_id],
                "slot": slot_by_user[user_id],
                "mode": mode_by_user[user_id],
                "rate": best_rates[user_id],
            }
            for user_id in range(NUM_USERS)
        },
        total_rate,
        fairness,
        best_utility,
        outage_count,
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict:
    return {
        "count": len(values),
        "missing": 0,
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
        "min": min(values),
        "q1": quantile(values, 0.25),
        "median": quantile(values, 0.5),
        "q3": quantile(values, 0.75),
        "max": max(values),
    }


def correlation(x: list[float], y: list[float]) -> float | None:
    sx = statistics.pstdev(x)
    sy = statistics.pstdev(y)
    if sx == 0.0 or sy == 0.0:
        return None
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    return statistics.fmean((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def svg_histogram(title: str, values: list[float], path: Path, *, log10: bool = False) -> None:
    if log10:
        values = [math.log10(max(value, 1.0e-300)) for value in values]
    width, height = 760, 390
    left, top, plot_w, plot_h = 78, 65, 630, 245
    low, high = min(values), max(values)
    if high <= low:
        high = low + 1.0
    bins = 24
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - low) / (high - low) * bins))
        counts[index] += 1
    max_count = max(counts) or 1
    bar_width = plot_w / bins
    bars = []
    for index, count in enumerate(counts):
        bar_h = count / max_count * plot_h
        x = left + index * bar_width + 1
        y = top + plot_h - bar_h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width-2:.1f}" '
            f'height="{bar_h:.1f}" fill="#2878a5"/>'
        )
    axis = (
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#34495e"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#34495e"/>'
        f'<text x="{left}" y="{top+plot_h+25}" font-size="12">{low:.3g}</text>'
        f'<text x="{left+plot_w}" y="{top+plot_h+25}" text-anchor="end" font-size="12">{high:.3g}</text>'
        f'<text x="{left-12}" y="{top+12}" text-anchor="end" font-size="12">{max_count}</text>'
        f'<text x="{left+plot_w/2}" y="{height-12}" text-anchor="middle" font-size="14">'
        f'{"log10(value)" if log10 else "Value"}</text>'
        f'<text transform="translate(20 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-size="14">User count</text>'
    )
    content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="18" font-weight="bold">{html.escape(title)}</text>'
        + axis
        + "".join(bars)
        + "</svg>"
    )
    path.write_text(content, encoding="utf-8")


def svg_scatter(users: list[dict], path: Path) -> None:
    width, height = 760, 510
    left, top, plot_w, plot_h = 90, 55, 420, 420
    max_r = R_MAX_M
    points = []
    for row in users:
        px = left + (row["x_m"] + max_r) / (2 * max_r) * plot_w
        py = top + (max_r - row["y_m"]) / (2 * max_r) * plot_h
        shade = max(0, min(255, int(255 * (row["channel_gain_db"] + 150) / 100)))
        points.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.3" fill="rgb(30,{shade},150)" opacity="0.52"/>'
        )
    circles = []
    for radius in (R_MIN_M, R_MAX_M):
        r_px = radius / (2 * max_r) * plot_w
        circles.append(
            f'<circle cx="{left+plot_w/2}" cy="{top+plot_h/2}" r="{r_px:.2f}" '
            'fill="none" stroke="#657786" stroke-dasharray="5,4"/>'
        )
    content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" font-weight="bold">'
        "User locations across training scenarios</text>"
        + "".join(circles)
        + "".join(points)
        + f'<polygon points="{left+plot_w/2-7},{top+plot_h/2+7} {left+plot_w/2+7},{top+plot_h/2+7} {left+plot_w/2},{top+plot_h/2-7}" fill="#c0392b"/>'
        + f'<text x="{left+plot_w+25}" y="95" font-size="13">N = {len(users)} users</text>'
        + f'<text x="{left+plot_w+25}" y="120" font-size="13">Cell: {R_MIN_M:g}-{R_MAX_M:g} m</text>'
        + f'<text x="{left+plot_w+25}" y="150" font-size="12">Color: channel gain (dB)</text>'
        + f'<text x="{left+plot_w/2}" y="{height-12}" text-anchor="middle" font-size="14">x (m)</text>'
        + f'<text transform="translate(20 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-size="14">y (m)</text>'
        + "</svg>"
    )
    path.write_text(content, encoding="utf-8")


def svg_correlation(names: list[str], matrix: list[list[float | None]], path: Path) -> None:
    cell, left, top = 105, 180, 80
    width, height = left + len(names) * cell + 25, top + len(names) * cell + 50
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="50%" y="30" text-anchor="middle" font-size="18" font-weight="bold">Pearson correlation (user rows pooled)</text>',
    ]
    for row, name in enumerate(names):
        y = top + row * cell
        parts.append(f'<text x="{left-8}" y="{y+cell/2+4}" text-anchor="end" font-size="12">{html.escape(name)}</text>')
        for col, column_name in enumerate(names):
            x = left + col * cell
            value = matrix[row][col]
            if value is None:
                color, label = "#d6dde2", "n/a"
            else:
                intensity = int(240 * abs(value))
                color = f"rgb({240 if value < 0 else 255-intensity},{255-intensity},{255-intensity if value < 0 else 240})"
                label = f"{value:.2f}"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="white"/>')
            parts.append(f'<text x="{x+cell/2}" y="{y+cell/2+5}" text-anchor="middle" font-size="14">{label}</text>')
            if row == 0:
                parts.append(f'<text transform="translate({x+cell/2} {top-8}) rotate(-35)" text-anchor="start" font-size="11">{html.escape(column_name)}</text>')
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


def write_documentation(summary: dict) -> None:
    splits = summary["splits"]
    stats = summary["feature_statistics_all_user_rows"]
    labels = summary["labels"]
    quality = summary["data_quality"]
    outliers = summary["iqr_outlier_flags"]
    stat_table = "\n".join(
        f"| `{name}` | {item['count']:,} | {item['mean']:.6g} | "
        f"{item['std_population']:.6g} | {item['min']:.6g} | {item['median']:.6g} | "
        f"{item['max']:.6g} |"
        for name, item in stats.items()
    )
    outlier_table = "\n".join(
        f"| `{name}` | {item['count_outside_fences']:,} | "
        f"{item['iqr_lower_fence']:.6g} | {item['iqr_upper_fence']:.6g} |"
        for name, item in outliers.items()
    )
    split_table = "\n".join(
        f"| {name} | {data['scenarios']:,} | {data['users']:,} | "
        f"{data['seed_min']}–{data['seed_max']} |"
        for name, data in splits.items()
    )
    split_feature_names = ["distance_m", "channel_gain_db", "snr_db"]
    split_feature_table = "\n".join(
        f"| {split} | "
        + " | ".join(
            f"{summary['mean']:.5g} / {summary['std_population']:.5g}"
            for summary in [
                summary["split_feature_statistics"][split][feature]
                for feature in split_feature_names
            ]
        )
        + " |"
        for split in splits
    )
    scenario_metric_names = [
        "oracle_sum_rate_mbps",
        "oracle_jains_fairness",
        "oracle_outage_users",
    ]
    scenario_metric_table = "\n".join(
        f"| {split} | "
        + " | ".join(
            f"{summary['mean']:.5g} / {summary['std_population']:.5g}"
            for summary in [
                summary["scenario_metric_statistics"][split][metric]
                for metric in scenario_metric_names
            ]
        )
        + " |"
        for split in splits
    )
    slot_table = "\n".join(
        f"| {slot} | {count:,} |" for slot, count in labels["slot_label_user_counts"].items()
    )
    mode_table = "\n".join(
        f"| {mode} | {count:,} |"
        for mode, count in labels["oracle_communication_mode_user_counts"].items()
    )
    corr_rows = []
    for record in summary["correlation_matrix"]:
        entries = [
            "n/a" if record[name] is None else f"{record[name]:.3f}"
            for name in summary["correlation_features"]
        ]
        corr_rows.append(f"| `{record['feature']}` | " + " | ".join(entries) + " |")
    corr_header = "| Feature | " + " | ".join(
        f"`{name}`" for name in summary["correlation_features"]
    ) + " |"
    corr_separator = "|" + "---|" * (len(summary["correlation_features"]) + 1)

    report = f"""# Dataset and Exploratory Data Analysis Report

## Project

**Hybrid GNN-Based NOMA for User Assignment and Slot Pairing**

## Executive summary

The implemented GNN is trained on **synthetic wireless-network graphs**, not measured or externally sourced radio data. The configured default is 200 training scenarios, 40 validation scenarios, and 50 separately generated benchmark scenarios; each contains 10 users and has oracle-derived pair and slot labels. This package contains **290 scenarios, 2,900 user rows, and 26,100 directed non-self edges**.

The raw historical scenarios are not saved by the project. Consequently, these files are a documented, reproducible **reconstruction** of the configured simulation and labels, not a claim to recover byte-for-byte NumPy samples from an earlier run. The reconstruction uses Python's standard-library random generator. See [README.md](README.md) and [dataset_metadata.json](dataset_metadata.json) before interpreting or citing the numeric results.

## 1. Dataset source and generation

The source project constructs users in memory in `src/channel_model.py`: users are uniform over an annulus with radii 20–500 m; path loss follows the configured log-distance law; independent complex Gaussian fading produces Rayleigh small-scale channel magnitudes. The GNN's dataset generator in `src/training.py` creates a graph per scenario and obtains labels from the exhaustive oracle in `src/optimization.py`. The graph builder in `src/graph_builder.py` forms a complete directed graph without self-loops and derives six node and four edge input features.

There are no real-world measurement records or imported external training dataset in this primary GNN path. `data/example_users.csv` in the source project is a small illustrative input and is not the default GNN training set. The separate optional classical-regression workflow creates rate/outage targets from simulator formulas; those are included here only as auxiliary reference columns, not as GNN labels.

### Provenance and reproducibility qualification

In the project, train and validation seeds are set by `cfg.system.seed + seed_offset + index`; `main.py` uses offsets 100 and 500, respectively. Evaluation uses seed `seed_offset + index` with offset 1000. This reconstruction follows those seed numbers and split sizes, along with the simulator equations, graph features, and exhaustive pairing/scheduling objective. It uses Python's `random.Random`, whereas the project uses NumPy `default_rng`; identical seed integers therefore **do not** produce the same random samples. No saved raw scenario cache was found to compare against. To export the exact samples of a future project run, modify/use a project-side exporter when its dependencies are available; do not treat these reconstructed draws as archived run data.

## 2. Sample units and split design

The independent observational unit is the **scenario/graph**, not the individual user row. Users within a scenario share a topology and are statistically dependent. The scenario seeds are disjoint across the three sets.

| Set | Scenarios (graphs) | User rows | Seed range |
|---|---:|---:|---:|
{split_table}

Training and validation are the GNN's supervised-learning splits. The 50 benchmark scenarios are a separately generated simulator benchmark set; they are not a classical-regression train/validation/test partition. The configuration has five slots and ten users per graph, so the exhaustive oracle assigns five pairs per graph and one pair to each slot.

### Split-level distribution check

The values below are mean / population SD across user rows in each split. These are a descriptive drift check; the scenario remains the independent unit.

| Split | Distance (m), mean / SD | Channel gain (dB), mean / SD | Per-slot SNR (dB), mean / SD |
|---|---|---|---|
{split_feature_table}

Oracle outcome summaries are scenario-level mean / SD:

| Split | Sum rate (Mbps), mean / SD | Jain fairness, mean / SD | Outage users/scenario, mean / SD |
|---|---|---|---|
{scenario_metric_table}

## 3. Variables and labels

### Physical/user-level fields

Each split's `*_users.csv` has one row per user and retains split/scenario/seed identifiers, coordinates, BS distance, path-loss values, complex channel components, linear and dB channel gain, and per-slot SNR. `rate_target_bps` and `outage_target` are auxiliary, simulator-derived single-user targets following `src/data_pipeline.py`; they are not GNN supervision.

### GNN inputs

**Node features:** `node_x_norm`, `node_y_norm`, `node_distance_norm`, `node_gain_db_zscore`, `node_gain_relative_mean`, `node_min_rate_mbps`.

**Directed-edge features:** `edge_distance_norm`, `edge_gain_disparity_norm`, `edge_channel_correlation`, `edge_interference_potential`. For 10 users, each graph has 10 × 9 = 90 directed non-self edges.

The configured YAML comment calls the sixth node feature “target_rate”, but the implementation uses `min_rate_req_bps / 1e6`. Here it is 0.1 Mbps for every user, so that feature is constant and does not convey a user-specific achieved/target throughput.

### Supervision

The oracle supplies a symmetric pair adjacency target and one slot ID per user. In the exported user rows, `oracle_partner_user_id` and `oracle_slot_id` encode those targets; `oracle_mode` and `oracle_user_rate_bps` record the oracle's selected link mode and achieved per-user rate. The edge file contains the pair target as `is_oracle_pair`. Labels are generated from the same simulator and optimization objective (sum rate plus weighted Jain fairness); they are optimization targets, not ground-truth radio measurements.

| Slot label | User rows |
|---:|---:|
{slot_table}

| Oracle selected pair mode | User rows |
|---|---:|
{mode_table}

Pair-label balance: **{labels['pair_labels']['positive_directed_edges']:,} positive** and **{labels['pair_labels']['negative_directed_edges']:,} negative** directed edges ({labels['pair_labels']['positive_edge_fraction']:.2%} positive). Each fully paired graph contributes 10 positive directed edges and 80 negatives. Any training interpretation should account for this imbalance (the project uses a positive-class weight in its BCE loss).

## 4. Descriptive statistics

Statistics below pool user rows across train, validation, and benchmark sets. Standard deviation is population SD. User-row summaries are descriptive; they do not make 2,900 independent scenarios.

| Variable | N | Mean | Population SD | Min | Median | Max |
|---|---:|---:|---:|---:|---:|---:|
{stat_table}

### Feature relationships

Pearson correlations use pooled user rows. Correlation is descriptive and does not imply causal effects; the physical generation process and within-scenario clustering matter.

{corr_header}
{corr_separator}
{chr(10).join(corr_rows)}

The edge feature named `edge_channel_correlation` is computed in the source as the magnitude of a normalized product of two **scalar** complex channel coefficients. For single-antenna scalar channels, this is 1 whenever both magnitudes exceed the numerical threshold (otherwise 0); it is not a spatial/antenna correlation model. It is 1 for every exported edge in this reconstruction, so it is a constant feature here. This deserves explicit mention in project interpretation and could be reconsidered in a future code revision; no source-code change is made here.

The `node_min_rate_mbps` feature is constant at 0.1 for the default configuration. Its correlation with nonconstant variables is undefined, so it is not included in the correlation matrix.

## 5. Figures

![User distance distribution](figures/distance_distribution.svg)

![Channel gain distribution](figures/channel_gain_distribution.svg)

![Per-slot SNR distribution](figures/snr_distribution.svg)

![Training user locations](figures/training_user_locations.svg)

![Feature correlation heatmap](figures/feature_correlation.svg)

Location plot colors encode channel gain dB; the center marker is the base station. Histograms describe per-user values, not scenario-level outcomes.

## 6. Data quality and outlier screening

Automated checks on the exported data found **{quality['user_rows']:,} user rows**, **{quality['scenario_count']:,} scenarios**, **{quality['duplicate_scenario_user_keys']} duplicate scenario/user keys**, no missing or non-finite values among checked numeric inputs/features/targets, distances within the configured annulus, reciprocal pair assignments, and all slot labels within 0–4. The largest absolute difference between `sqrt(x²+y²)` and stored distance is {quality['coordinate_distance_max_abs_error_m']:.3g} m (floating-point rounding).

An IQR 1.5×fence screen flags rows for review; it does not label observations as erroneous. In this simulator, large/small channel gains and SNRs can naturally occur under Rayleigh fading, and distance endpoints follow the configured annulus.

| Variable | Rows outside IQR fences | Lower fence | Upper fence |
|---|---:|---:|---:|
{outlier_table}

No flagged observation was automatically deleted or winsorized.

## 7. Interpretation and limitations

1. **Synthetic scope:** EDA describes the configured simulation, not deployment conditions, measured users, or a field trial. Do not present it as empirical radio data.
2. **Reconstruction, not historical recovery:** the project stores graph/metric outputs rather than raw generated train/validation/benchmark channel rows. This package is reproducible but uses a different RNG implementation from the project.
3. **Oracle dependence:** pair and slot labels are optimal under this simulator's finite-slot objective and assumptions. They are not independent annotations and may not transfer to another propagation, interference, or traffic model.
4. **Scenario clustering:** inferential statistics must resample or split by scenario, not by user/edge row.
5. **Rate/outage columns:** these are calculated from a single-user Shannon-rate proxy over one slot. They are not the selected NOMA pair rates and are not GNN targets.
6. **SCMA representation:** the hybrid schedule's pair-mode choice uses a simplified rate proxy in `src/slot_assignment.py`. It should not be interpreted as end-to-end SCMA message-passing detector evidence. The project's SCMA BER evaluation is a different experiment.
7. **Model feature semantics:** minimum-rate requirement is constant in the default setup; scalar channel “correlation” is nearly constant. Such features provide little information under this configuration.
8. **Hardware parameters:** 2.4 GHz is configured, but the current path-loss implementation uses a fixed reference loss and exponent; carrier frequency is not separately applied in the per-user gain formula.

## 8. Files and how to reproduce

- `train_users.csv`, `validation_users.csv`, `benchmark_users.csv`: split-specific user state, graph input features, and oracle labels.
- `all_splits_edges.csv`: directed non-self edge rows, edge inputs, and pair targets.
- `scenario_summary.csv`: one row per graph with oracle outcomes.
- `feature_correlations.csv`: correlation matrix values.
- `dataset_metadata.json`: full configuration, schema, computed statistics, labels, and quality checks.
- `figures/`: vector EDA plots.
- `generate_dataset_eda.py`: standard-library-only generator and report builder.

Run from any directory with Python 3.10 or later:

```powershell
python Dataset\\generate_dataset_eda.py
```

The script overwrites only generated files in this `Dataset` directory. For academically accurate reporting, cite the project implementation/configuration and explicitly describe these files as a simulator reconstruction, unless the historical raw samples are later exported from the exact project run.
"""
    (ROOT / "EDA_Report.md").write_text(report, encoding="utf-8")

    readme = f"""# Dataset package

This directory contains a reconstructed, synthetic dataset for the project's default GNN scenario setup plus a reproducible EDA report. It has {sum(item['scenarios'] for item in splits.values())} scenario graphs, {quality['user_rows']:,} user rows, and {labels['pair_labels']['directed_non_self_edges']:,} directed edge rows.

**Important provenance note:** the project does not save raw channel scenarios. The project code uses NumPy `default_rng`; this standard-library generator uses `random.Random`. The files reproduce configured distributions, seed indices, graph schema, and oracle label logic, but are not the exact numerical draws from an earlier run.

Start with [EDA_Report.md](EDA_Report.md), then see [dataset_metadata.json](dataset_metadata.json) for machine-readable statistics. Regenerate using `python Dataset\\generate_dataset_eda.py` from the repository root (Python 3.10+; standard library only).

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
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    specs = [
        ("train", TRAIN_COUNT, 100),
        ("validation", VALIDATION_COUNT, 500),
        ("benchmark", BENCHMARK_COUNT, 1000),
    ]
    all_users = []
    all_edges = []
    scenario_summaries = []
    split_info = {}
    split_users = {}
    split_seeds = {}

    for split, count, seed_offset in specs:
        users_rows = []
        seeds = []
        for index in range(count):
            seed = SEED + seed_offset + index if split != "benchmark" else seed_offset + index
            seeds.append(seed)
            scenario_id = f"{split}_{index:03d}"
            users = make_scenario(seed)
            labels, total_rate, fairness, utility, outage_count = oracle_labels(users)
            for user in users:
                label = labels[user["user_id"]]
                row = {
                    "split": split,
                    "scenario_id": scenario_id,
                    "scenario_index": index,
                    "random_seed": seed,
                    **user,
                    "oracle_partner_user_id": label["partner"],
                    "oracle_slot_id": label["slot"],
                    "oracle_mode": label["mode"],
                    "oracle_user_rate_bps": label["rate"],
                    "oracle_method": "exhaustive_pair_partition",
                    "scenario_oracle_sum_rate_bps": total_rate,
                    "scenario_oracle_jains_fairness": fairness,
                    "scenario_oracle_utility": utility,
                    "scenario_oracle_outage_users": outage_count,
                }
                users_rows.append(row)
                all_users.append(row)

            by_id = {user["user_id"]: user for user in users}
            positive_edges = {
                (user_id, label["partner"]) for user_id, label in labels.items()
            }
            for src in range(NUM_USERS):
                for dst in range(NUM_USERS):
                    if src == dst:
                        continue
                    first, second = by_id[src], by_id[dst]
                    distance = math.hypot(
                        first["x_m"] - second["x_m"],
                        first["y_m"] - second["y_m"],
                    )
                    corr = (
                        1.0
                        if math.sqrt(first["channel_gain_linear"]) > 1.0e-12
                        and math.sqrt(second["channel_gain_linear"]) > 1.0e-12
                        else 0.0
                    )
                    all_edges.append(
                        {
                            "split": split,
                            "scenario_id": scenario_id,
                            "random_seed": seed,
                            "src_user_id": src,
                            "dst_user_id": dst,
                            "edge_distance_norm": distance / (2.0 * R_MAX_M),
                            "edge_gain_disparity_norm": abs(
                                first["channel_gain_db"] - second["channel_gain_db"]
                            )
                            / 40.0,
                            "edge_channel_correlation": corr,
                            "edge_interference_potential": (
                                first["channel_gain_linear"]
                                + second["channel_gain_linear"]
                            )
                            / (
                                2.0
                                * max(
                                    statistics.fmean(
                                        user["channel_gain_linear"] for user in users
                                    ),
                                    1.0e-18,
                                )
                            ),
                            "is_oracle_pair": int((src, dst) in positive_edges),
                        }
                    )
            scenario_summaries.append(
                {
                    "split": split,
                    "scenario_id": scenario_id,
                    "random_seed": seed,
                    "users": NUM_USERS,
                    "oracle_pairs": NUM_USERS // 2,
                    "oracle_sum_rate_mbps": total_rate / 1.0e6,
                    "oracle_jains_fairness": fairness,
                    "oracle_outage_users": outage_count,
                    "oracle_utility": utility,
                    "scma_selected_pairs": sum(mode == "SCMA" for mode in labels.values()) // 2,
                }
            )
        split_users[split] = users_rows
        split_seeds[split] = seeds
        split_info[split] = {
            "scenarios": count,
            "users": count * NUM_USERS,
            "seed_min": min(seeds),
            "seed_max": max(seeds),
            "seed_rule": "base_seed + offset + index" if split != "benchmark" else "benchmark seed_offset + index",
        }
        user_fields = list(users_rows[0].keys())
        write_csv(ROOT / f"{split}_users.csv", user_fields, users_rows)

    edge_fields = list(all_edges[0].keys())
    write_csv(ROOT / "all_splits_edges.csv", edge_fields, all_edges)
    write_csv(ROOT / "scenario_summary.csv", list(scenario_summaries[0]), scenario_summaries)

    pooled = split_users["train"] + split_users["validation"]
    # Benchmark scenarios are deliberately summarized separately from GNN training.
    pooled = pooled + split_users["benchmark"]
    statistics_by_feature = {
        feature: summarize([float(row[feature]) for row in pooled])
        for feature in EDA_FEATURES
    }
    split_feature_statistics = {
        split: {
            feature: summarize([float(row[feature]) for row in rows])
            for feature in EDA_FEATURES
        }
        for split, rows in split_users.items()
    }
    scenario_metric_statistics = {
        split: {
            metric: summarize(
                [
                    float(row[metric])
                    for row in scenario_summaries
                    if row["split"] == split
                ]
            )
            for metric in (
                "oracle_sum_rate_mbps",
                "oracle_jains_fairness",
                "oracle_outage_users",
            )
        }
        for split, _, _ in specs
    }
    outlier_counts = {}
    for feature in EDA_FEATURES:
        values = [float(row[feature]) for row in pooled]
        q1, q3 = quantile(values, 0.25), quantile(values, 0.75)
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        outlier_counts[feature] = {
            "iqr_lower_fence": low,
            "iqr_upper_fence": high,
            "count_outside_fences": sum(value < low or value > high for value in values),
            "interpretation": "flag for inspection; not automatically invalid",
        }

    corr_names = ["x_m", "y_m", "distance_m", "channel_gain_db", "snr_db"]
    corr_values = {
        name: [float(row[name]) for row in pooled] for name in corr_names
    }
    corr_matrix = [
        [correlation(corr_values[row], corr_values[column]) for column in corr_names]
        for row in corr_names
    ]
    corr_records = []
    for row_name, values in zip(corr_names, corr_matrix):
        corr_records.append({"feature": row_name, **dict(zip(corr_names, values))})
    write_csv(
        ROOT / "feature_correlations.csv",
        ["feature"] + corr_names,
        corr_records,
    )

    label_summary = {
        "pair_labels": {
            "graphs": sum(item["scenarios"] for item in split_info.values()),
            "users": len(all_users),
            "directed_non_self_edges": len(all_edges),
            "positive_directed_edges": sum(edge["is_oracle_pair"] for edge in all_edges),
            "negative_directed_edges": sum(not edge["is_oracle_pair"] for edge in all_edges),
            "positive_edge_fraction": sum(edge["is_oracle_pair"] for edge in all_edges) / len(all_edges),
            "unique_pairs_per_graph": NUM_USERS // 2,
        },
        "slot_label_user_counts": dict(
            sorted(Counter(int(row["oracle_slot_id"]) for row in all_users).items())
        ),
        "oracle_communication_mode_user_counts": dict(
            sorted(Counter(row["oracle_mode"] for row in all_users).items())
        ),
        "rate_regression_target_outage_rows": sum(int(row["outage_target"]) for row in all_users),
    }
    # Under this even-user configuration all users are paired; counts per slot are user labels.
    for row in all_users:
        row["oracle_slot_id"] = int(row["oracle_slot_id"])

    data_quality = {
        "user_rows": len(all_users),
        "scenario_count": len(scenario_summaries),
        "duplicate_scenario_user_keys": len(all_users)
        - len({(row["scenario_id"], row["user_id"]) for row in all_users}),
        "missing_numeric_values": {
            key: sum(row.get(key) is None for row in all_users)
            for key in EDA_FEATURES + NODE_FEATURES + ["rate_target_bps", "oracle_slot_id"]
        },
        "non_finite_numeric_values": {
            key: sum(
                not math.isfinite(float(row[key]))
                for row in all_users
                if row.get(key) is not None
            )
            for key in EDA_FEATURES + NODE_FEATURES + ["rate_target_bps", "oracle_slot_id"]
        },
        "distance_range_valid": all(
            R_MIN_M <= row["distance_m"] <= R_MAX_M for row in all_users
        ),
        "coordinate_distance_max_abs_error_m": max(
            abs(math.hypot(row["x_m"], row["y_m"]) - row["distance_m"])
            for row in all_users
        ),
        "slot_labels_in_range": all(
            0 <= row["oracle_slot_id"] < NUM_SLOTS for row in all_users
        ),
        "pair_assignment_reciprocal": all(
            next(
                candidate["oracle_partner_user_id"]
                for candidate in all_users
                if candidate["scenario_id"] == row["scenario_id"]
                and candidate["user_id"] == row["oracle_partner_user_id"]
            )
            == row["user_id"]
            for row in all_users
        ),
        "edge_channel_correlation_unique_values": sorted(
            {edge["edge_channel_correlation"] for edge in all_edges}
        ),
    }

    figure_dir = ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    svg_histogram(
        "User distance distribution (all splits)",
        [float(row["distance_m"]) for row in pooled],
        figure_dir / "distance_distribution.svg",
    )
    svg_histogram(
        "Channel gain distribution (log scale)",
        [float(row["channel_gain_linear"]) for row in pooled],
        figure_dir / "channel_gain_distribution.svg",
        log10=True,
    )
    svg_histogram(
        "Per-slot SNR distribution",
        [float(row["snr_db"]) for row in pooled],
        figure_dir / "snr_distribution.svg",
    )
    svg_scatter(split_users["train"], figure_dir / "training_user_locations.svg")
    svg_correlation(
        corr_names, corr_matrix, figure_dir / "feature_correlation.svg"
    )

    summary = {
        "title": "Hybrid GNN-Based NOMA synthetic scenario dataset reconstruction",
        "provenance": {
            "type": "synthetic simulation",
            "measured_or_external_data": False,
            "reconstruction_notice": (
                "The project does not retain the historical raw NumPy-generated scenarios. "
                "This artifact reproduces the documented simulation distributions, configured "
                "parameters, split counts, seed offsets, graph features and exhaustive-oracle "
                "labeling rules using Python's standard-library random generator. Because its "
                "random number generator differs from NumPy default_rng, the generated numeric "
                "samples are not byte-for-byte the samples used in any previous run."
            ),
        },
        "configuration": {
            "base_seed": SEED,
            "users_per_scenario": NUM_USERS,
            "slots": NUM_SLOTS,
            "carrier_frequency_hz_configured": 2.4e9,
            "cell_radius_min_m": R_MIN_M,
            "cell_radius_max_m": R_MAX_M,
            "total_bandwidth_hz": TOTAL_BANDWIDTH_HZ,
            "total_power_w": TOTAL_POWER_W,
            "noise_density_dbm_hz": NOISE_DENSITY_DBM_HZ,
            "noise_figure_db": NOISE_FIGURE_DB,
            "path_loss_exponent": PATH_LOSS_EXPONENT,
            "reference_distance_m": REFERENCE_DISTANCE_M,
            "reference_path_loss_db": REFERENCE_PATH_LOSS_DB,
            "fading": "independent complex Gaussian Rayleigh; variance 1",
            "minimum_rate_requirement_bps": MIN_RATE_BPS,
            "noma_alpha_weak": ALPHA_WEAK,
            "noma_alpha_strong": ALPHA_STRONG,
            "sic_residual_factor": SIC_RESIDUAL,
            "fairness_weight": FAIRNESS_WEIGHT,
            "oracle": "exhaustive perfect pair partitions; slot order follows pair enumeration",
        },
        "splits": split_info,
        "split_independence": "Scenario seeds are disjoint across train, validation, and benchmark sets.",
        "node_features": NODE_FEATURES,
        "edge_features": EDGE_FEATURES,
        "eda_features": EDA_FEATURES,
        "feature_statistics_all_user_rows": statistics_by_feature,
        "split_feature_statistics": split_feature_statistics,
        "scenario_metric_statistics": scenario_metric_statistics,
        "iqr_outlier_flags": outlier_counts,
        "correlation_features": corr_names,
        "correlation_matrix": corr_records,
        "labels": label_summary,
        "data_quality": data_quality,
    }
    (ROOT / "dataset_metadata.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_documentation(summary)
    print(
        f"Wrote {len(scenario_summaries)} scenarios, {len(all_users)} user rows, "
        f"{len(all_edges)} directed edges to {ROOT}"
    )
    print(json.dumps(split_info, indent=2))
    print(json.dumps(data_quality, indent=2))
    print(json.dumps(label_summary, indent=2))


if __name__ == "__main__":
    main()
