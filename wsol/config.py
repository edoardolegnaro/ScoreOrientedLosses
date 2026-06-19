from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np


@dataclass(frozen=True)
class WSOLConfig:
    """Configuration for binary weighted Score-Oriented Loss."""

    weights: tuple[float, ...] = (0.5, 0.25, 0.125)
    weight_mode: Literal["prod", "max"] = "prod"
    distribution: Literal["uniform", "cosine"] = "uniform"
    score: str = "tss"
    mu: float = 0.5
    delta: float = 0.1
    name: Optional[str] = None

    @property
    def T(self) -> int:
        return len(self.weights)

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        weights_str = "-".join(f"{weight:.3f}" for weight in self.weights)
        return f"T{self.T}_{self.weight_mode}_{self.score}_{weights_str}"


def normalize_score_name(score_name: str) -> str:
    aliases = {
        "f1_score": "f1",
        "f1score": "f1",
        "balanced_acc": "balanced_accuracy",
        "hss2": "hss",
    }
    normalized = score_name.lower()
    return aliases.get(normalized, normalized)


def validate_temporal_weights(weights: Iterable[float], weight_mode: str) -> np.ndarray:
    values = np.asarray(tuple(weights), dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty 1D sequence")
    if not np.isfinite(values).all():
        raise ValueError("weights must contain only finite numbers")
    if np.any(values <= 0.0):
        raise ValueError("weights must be strictly positive")
    if np.any(values[:-1] < values[1:]):
        raise ValueError("weights must be non-increasing in time")

    if weight_mode == "prod":
        if float(values.sum()) >= 1.0:
            raise ValueError("prod weights must sum to less than 1")
    elif weight_mode == "max":
        if float(values.max()) >= 1.0:
            raise ValueError("max weights must be strictly smaller than 1")
    else:
        raise ValueError(f"Unsupported weight_mode: {weight_mode}")

    return values
