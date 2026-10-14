"""Model artifact metadata and safe local inference helpers."""

import json
from pathlib import Path
from typing import Any, Dict


def save_model_metadata(path: str, metadata: Dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_model_metadata(path: str) -> Dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Model metadata does not exist: {source}")
    return json.loads(source.read_text(encoding="utf-8"))
