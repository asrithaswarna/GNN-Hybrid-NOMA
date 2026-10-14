"""
Utility functions: random seeding, device management, and result persistence.
"""

import json
import random
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42) -> None:
    """Set global random seed across Python, NumPy, and PyTorch for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested: str = "cpu") -> str:
    """Select compute device ('cuda' if available and requested, else 'cpu')."""
    if requested.lower() == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary to formatted JSON file."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_dataframe_csv(df: pd.DataFrame, filepath: str) -> None:
    """Save pandas DataFrame to CSV file."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
