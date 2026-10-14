"""Lightweight local prediction monitoring, not production observability."""

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict


class PredictionMonitor:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.invalid_count = 0
        self.latencies_ms = []

    def record(self, prediction: Any, latency_ms: float, valid: bool = True) -> None:
        self.count += 1
        self.invalid_count += int(not valid)
        self.latencies_ms.append(float(latency_ms))
        self.path.write_text(json.dumps({
            "prediction_count": self.count,
            "invalid_input_count": self.invalid_count,
            "mean_latency_ms": sum(self.latencies_ms) / len(self.latencies_ms),
            "last_prediction": prediction,
        }, indent=2), encoding="utf-8")

    def timed(self):
        return perf_counter()
