from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

from wsol.config import WSOLConfig, normalize_score_name, validate_temporal_weights


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    safe_denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    quotient = numerator / safe_denominator
    return torch.where(denominator == 0, torch.zeros_like(quotient), quotient)


def _accuracy(tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
    return _safe_div(tp + tn, tp + tn + fp + fn)


def _precision(tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
    del tn, fn
    return _safe_div(tp, tp + fp)


def _recall(tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
    del tn, fp
    return _safe_div(tp, tp + fn)


def _specificity(tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
    del fn, tp
    return _safe_div(tn, tn + fp)


def _balanced_accuracy(tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
    return 0.5 * (_recall(tn, fp, fn, tp) + _specificity(tn, fp, fn, tp))


def _f1(tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
    precision = _precision(tn, fp, fn, tp)
    recall = _recall(tn, fp, fn, tp)
    return _safe_div(2.0 * precision * recall, precision + recall)


def _tss(tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
    return _recall(tn, fp, fn, tp) + _specificity(tn, fp, fn, tp) - 1.0


def _csi(tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
    del tn
    return _safe_div(tp, tp + fp + fn)


def _hss(tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
    numerator = 2.0 * (tp * tn - fp * fn)
    denominator = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    return _safe_div(numerator, denominator)


def _gmean(tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
    recall = _recall(tn, fp, fn, tp)
    specificity = _specificity(tn, fp, fn, tp)
    return torch.sqrt(torch.clamp(recall * specificity, min=0.0))


def _get_score(score_name: str):
    normalized = normalize_score_name(score_name)
    scores = {
        "accuracy": _accuracy,
        "precision": _precision,
        "recall": _recall,
        "specificity": _specificity,
        "balanced_accuracy": _balanced_accuracy,
        "f1": _f1,
        "tss": _tss,
        "csi": _csi,
        "hss": _hss,
        "gmean": _gmean,
    }
    if normalized not in scores:
        raise ValueError(f"Unsupported score: {score_name}")
    return scores[normalized]


def _cosine_distribution(values: torch.Tensor, mu: float, delta: float) -> torch.Tensor:
    if delta <= 0.0:
        raise ValueError("delta must be strictly positive for the cosine threshold distribution.")
    pi = torch.tensor(math.pi, dtype=values.dtype, device=values.device)
    scaled = (values - mu) / delta
    return torch.where(
        values < mu - delta,
        torch.zeros_like(values),
        torch.where(values > mu + delta, torch.ones_like(values), 0.5 * (1.0 + scaled + torch.sin(pi * scaled) / pi)),
    )


def _transform(outputs: torch.Tensor, config: WSOLConfig) -> torch.Tensor:
    probabilities = outputs.reshape(-1).to(dtype=torch.float32)
    if config.distribution == "uniform":
        return probabilities
    if config.distribution == "cosine":
        return _cosine_distribution(probabilities, mu=config.mu, delta=config.delta)
    raise ValueError(f"Unsupported distribution: {config.distribution}")


def _future_labels_matrix(labels: torch.Tensor, horizon: int) -> torch.Tensor:
    labels = labels.reshape(-1)
    padded = torch.cat([labels, torch.zeros(horizon, dtype=labels.dtype, device=labels.device)])
    return padded[1:].unfold(0, horizon, 1)[: labels.shape[0]]


def _past_outputs_matrix(outputs: torch.Tensor, horizon: int) -> torch.Tensor:
    outputs = outputs.reshape(-1)
    num_items = outputs.shape[0]
    indices = (
        torch.arange(num_items, device=outputs.device).unsqueeze(1)
        - 1
        - torch.arange(horizon, device=outputs.device).unsqueeze(0)
    )
    valid = indices >= 0
    clipped = indices.clamp(min=0)
    return outputs[clipped] * valid.to(outputs.dtype)


def _strict_running_max_indices(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    cumulative_max = torch.cummax(values, dim=0).values
    previous_cumulative_max = torch.cat(
        [
            torch.full((1,), float("-inf"), dtype=values.dtype, device=values.device),
            cumulative_max[:-1],
        ]
    )
    return torch.nonzero(values > previous_cumulative_max, as_tuple=False).squeeze(1)


class WeightedSOLLoss(nn.Module):
    """Binary weighted Score-Oriented Loss.

    `outputs` are expected to be probabilities in [0, 1]. If your model returns
    logits, apply a sigmoid before passing them to this loss.
    """

    def __init__(
        self,
        config: Optional[WSOLConfig] = None,
        *,
        weights: Sequence[float] = (0.5, 0.25, 0.125),
        weight_mode: str = "prod",
        distribution: str = "uniform",
        score: str = "tss",
        mu: float = 0.5,
        delta: float = 0.1,
    ):
        super().__init__()
        if config is None:
            config = WSOLConfig(
                weights=tuple(weights),
                weight_mode=weight_mode,  # type: ignore[arg-type]
                distribution=distribution,  # type: ignore[arg-type]
                score=score,
                mu=mu,
                delta=delta,
            )
        self.config = config
        temporal_weights = validate_temporal_weights(config.weights, config.weight_mode)
        self.register_buffer("weights", torch.tensor(temporal_weights, dtype=torch.float32))
        self.horizon = len(temporal_weights)
        self.score_fn = _get_score(config.score)

    def _weighted_false_positive_prod(self, transformed_outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(device=transformed_outputs.device, dtype=transformed_outputs.dtype)
        future_labels = _future_labels_matrix(labels, self.horizon)
        weighted_future = torch.sum(future_labels * weights.view(1, -1), dim=1)
        return torch.sum((1.0 - weighted_future) * (1.0 - labels) * transformed_outputs)

    def _weighted_false_negative_prod(self, transformed_outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(device=transformed_outputs.device, dtype=transformed_outputs.dtype)
        past_outputs = _past_outputs_matrix(transformed_outputs, self.horizon)
        deltas = torch.relu(past_outputs - transformed_outputs.unsqueeze(1))
        sigma = torch.sum(deltas * weights.view(1, -1), dim=1)
        return torch.sum(labels * (1.0 - transformed_outputs - sigma))

    def _weighted_false_positive_max(self, transformed_outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(device=transformed_outputs.device, dtype=transformed_outputs.dtype)
        future_labels = _future_labels_matrix(labels, self.horizon)
        weighted_future = torch.max(future_labels * weights.view(1, -1), dim=1).values
        return torch.sum((1.0 - weighted_future) * (1.0 - labels) * transformed_outputs)

    def _weighted_false_negative_max(self, transformed_outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(device=transformed_outputs.device, dtype=transformed_outputs.dtype)
        past_outputs = _past_outputs_matrix(transformed_outputs, self.horizon)
        deltas = torch.relu(past_outputs - transformed_outputs.unsqueeze(1))
        total = torch.zeros((), dtype=transformed_outputs.dtype, device=transformed_outputs.device)
        for index in range(transformed_outputs.shape[0]):
            row = past_outputs[index]
            active_indices = _strict_running_max_indices(row)
            if active_indices.numel() == 0:
                correction = torch.zeros((), dtype=row.dtype, device=row.device)
            else:
                active_weights = weights[active_indices]
                successor_weights = torch.zeros_like(active_weights)
                if active_indices.numel() > 1:
                    successor_weights[:-1] = weights[active_indices[1:]]
                weight_deltas = active_weights - successor_weights
                correction = torch.sum(weight_deltas * deltas[index, active_indices])
            total = total + labels[index] * (1.0 - transformed_outputs[index] - correction)
        return total

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels_flat = labels.reshape(-1).to(dtype=torch.float32, device=outputs.device)
        transformed_outputs = _transform(outputs, self.config)

        if self.config.weight_mode == "prod":
            fp = self._weighted_false_positive_prod(transformed_outputs, labels_flat)
            fn = self._weighted_false_negative_prod(transformed_outputs, labels_flat)
        elif self.config.weight_mode == "max":
            fp = self._weighted_false_positive_max(transformed_outputs, labels_flat)
            fn = self._weighted_false_negative_max(transformed_outputs, labels_flat)
        else:
            raise ValueError(f"Unsupported weight_mode: {self.config.weight_mode}")

        tp = torch.sum(labels_flat * transformed_outputs)
        tn = torch.sum((1.0 - labels_flat) * (1.0 - transformed_outputs))
        return 1.0 - self.score_fn(tn, fp, fn, tp)


wSOL = WeightedSOLLoss
