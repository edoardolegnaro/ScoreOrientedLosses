from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import tensorflow as tf

from wsol.config import WSOLConfig, normalize_score_name, validate_temporal_weights


def _safe_div(numerator: tf.Tensor, denominator: tf.Tensor) -> tf.Tensor:
    return tf.math.divide_no_nan(numerator, denominator)


def _accuracy(tn: tf.Tensor, fp: tf.Tensor, fn: tf.Tensor, tp: tf.Tensor) -> tf.Tensor:
    return _safe_div(tp + tn, tp + tn + fp + fn)


def _precision(tn: tf.Tensor, fp: tf.Tensor, fn: tf.Tensor, tp: tf.Tensor) -> tf.Tensor:
    del tn, fn
    return _safe_div(tp, tp + fp)


def _recall(tn: tf.Tensor, fp: tf.Tensor, fn: tf.Tensor, tp: tf.Tensor) -> tf.Tensor:
    del tn, fp
    return _safe_div(tp, tp + fn)


def _specificity(tn: tf.Tensor, fp: tf.Tensor, fn: tf.Tensor, tp: tf.Tensor) -> tf.Tensor:
    del fn, tp
    return _safe_div(tn, tn + fp)


def _balanced_accuracy(tn: tf.Tensor, fp: tf.Tensor, fn: tf.Tensor, tp: tf.Tensor) -> tf.Tensor:
    return 0.5 * (_recall(tn, fp, fn, tp) + _specificity(tn, fp, fn, tp))


def _f1(tn: tf.Tensor, fp: tf.Tensor, fn: tf.Tensor, tp: tf.Tensor) -> tf.Tensor:
    precision = _precision(tn, fp, fn, tp)
    recall = _recall(tn, fp, fn, tp)
    return _safe_div(2.0 * precision * recall, precision + recall)


def _tss(tn: tf.Tensor, fp: tf.Tensor, fn: tf.Tensor, tp: tf.Tensor) -> tf.Tensor:
    return _recall(tn, fp, fn, tp) + _specificity(tn, fp, fn, tp) - 1.0


def _csi(tn: tf.Tensor, fp: tf.Tensor, fn: tf.Tensor, tp: tf.Tensor) -> tf.Tensor:
    del tn
    return _safe_div(tp, tp + fp + fn)


def _hss(tn: tf.Tensor, fp: tf.Tensor, fn: tf.Tensor, tp: tf.Tensor) -> tf.Tensor:
    numerator = 2.0 * (tp * tn - fp * fn)
    denominator = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    return _safe_div(numerator, denominator)


def _gmean(tn: tf.Tensor, fp: tf.Tensor, fn: tf.Tensor, tp: tf.Tensor) -> tf.Tensor:
    recall = _recall(tn, fp, fn, tp)
    specificity = _specificity(tn, fp, fn, tp)
    return tf.sqrt(tf.maximum(recall * specificity, 0.0))


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


def _cosine_distribution(values: tf.Tensor, mu: float, delta: float) -> tf.Tensor:
    if delta <= 0.0:
        raise ValueError("delta must be strictly positive for the cosine threshold distribution.")
    pi = tf.constant(math.pi, dtype=tf.float32)
    values = tf.cast(values, tf.float32)
    scaled = (values - mu) / delta
    return tf.where(
        values < mu - delta,
        0.0,
        tf.where(values > mu + delta, 1.0, 0.5 * (1.0 + scaled + tf.math.sin(pi * scaled) / pi)),
    )


def _transform(outputs: tf.Tensor, config: WSOLConfig) -> tf.Tensor:
    probabilities = tf.cast(tf.reshape(outputs, [-1]), tf.float32)
    if config.distribution == "uniform":
        return probabilities
    if config.distribution == "cosine":
        return _cosine_distribution(probabilities, mu=config.mu, delta=config.delta)
    raise ValueError(f"Unsupported distribution: {config.distribution}")


def _future_labels_matrix(labels: tf.Tensor, horizon: int) -> tf.Tensor:
    labels = tf.reshape(labels, [-1])
    padded = tf.concat([labels, tf.zeros([horizon], dtype=labels.dtype)], axis=0)
    return tf.signal.frame(padded[1:], frame_length=horizon, frame_step=1)


def _past_outputs_matrix(outputs: tf.Tensor, horizon: int) -> tf.Tensor:
    outputs = tf.reshape(outputs, [-1])
    num_items = tf.shape(outputs)[0]
    indices = tf.expand_dims(tf.range(num_items), 1) - 1 - tf.expand_dims(tf.range(horizon), 0)
    valid = tf.cast(indices >= 0, outputs.dtype)
    clipped = tf.maximum(indices, 0)
    gathered = tf.gather(outputs, clipped)
    return gathered * valid


def _strict_running_max_indices(values: tf.Tensor) -> tf.Tensor:
    values = tf.reshape(values, [-1])
    cumulative_max = tf.scan(tf.maximum, values)
    previous_cumulative_max = tf.concat(
        [
            tf.fill([1], tf.cast(float("-inf"), values.dtype)),
            cumulative_max[:-1],
        ],
        axis=0,
    )
    return tf.reshape(tf.where(values > previous_cumulative_max), [-1])


class WeightedSOLLoss(tf.keras.losses.Loss):
    """Binary weighted Score-Oriented Loss for TensorFlow/Keras."""

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
        name: str = "weighted_sol",
    ):
        super().__init__(name=name)
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
        self.weights = tf.constant(temporal_weights, dtype=tf.float32)
        self.horizon = len(temporal_weights)
        self.score_fn = _get_score(config.score)

    def _weighted_false_positive_prod(self, transformed_outputs: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        future_labels = _future_labels_matrix(labels, self.horizon)
        weighted_future = tf.reduce_sum(future_labels * tf.reshape(self.weights, [1, -1]), axis=1)
        return tf.reduce_sum((1.0 - weighted_future) * (1.0 - labels) * transformed_outputs)

    def _weighted_false_negative_prod(self, transformed_outputs: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        past_outputs = _past_outputs_matrix(transformed_outputs, self.horizon)
        deltas = tf.nn.relu(past_outputs - tf.expand_dims(transformed_outputs, axis=1))
        sigma = tf.reduce_sum(deltas * tf.reshape(self.weights, [1, -1]), axis=1)
        return tf.reduce_sum(labels * (1.0 - transformed_outputs - sigma))

    def _weighted_false_positive_max(self, transformed_outputs: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        future_labels = _future_labels_matrix(labels, self.horizon)
        weighted_future = tf.reduce_max(future_labels * tf.reshape(self.weights, [1, -1]), axis=1)
        return tf.reduce_sum((1.0 - weighted_future) * (1.0 - labels) * transformed_outputs)

    def _weighted_false_negative_max(self, transformed_outputs: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        past_outputs = _past_outputs_matrix(transformed_outputs, self.horizon)
        deltas = tf.nn.relu(past_outputs - tf.expand_dims(transformed_outputs, axis=1))

        total = tf.constant(0.0, dtype=tf.float32)
        num_items = tf.shape(transformed_outputs)[0]
        for index in tf.range(num_items):
            row = past_outputs[index]
            active_indices = _strict_running_max_indices(row)

            def no_active_indices() -> tf.Tensor:
                return tf.constant(0.0, dtype=tf.float32)

            def some_active_indices() -> tf.Tensor:
                active_weights = tf.gather(self.weights, active_indices)
                successor_weights = tf.concat(
                    [
                        tf.gather(self.weights, active_indices[1:]),
                        tf.zeros([1], dtype=self.weights.dtype),
                    ],
                    axis=0,
                )
                weight_deltas = active_weights - successor_weights
                return tf.reduce_sum(weight_deltas * tf.gather(deltas[index], active_indices))

            correction = tf.cond(tf.equal(tf.size(active_indices), 0), no_active_indices, some_active_indices)
            total += labels[index] * (1.0 - transformed_outputs[index] - correction)
        return total

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        labels = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        transformed_outputs = _transform(y_pred, self.config)

        if self.config.weight_mode == "prod":
            fp = self._weighted_false_positive_prod(transformed_outputs, labels)
            fn = self._weighted_false_negative_prod(transformed_outputs, labels)
        elif self.config.weight_mode == "max":
            fp = self._weighted_false_positive_max(transformed_outputs, labels)
            fn = self._weighted_false_negative_max(transformed_outputs, labels)
        else:
            raise ValueError(f"Unsupported weight_mode: {self.config.weight_mode}")

        tp = tf.reduce_sum(labels * transformed_outputs)
        tn = tf.reduce_sum((1.0 - labels) * (1.0 - transformed_outputs))
        return 1.0 - self.score_fn(tn, fp, fn, tp)

    def get_config(self) -> Dict[str, object]:
        base_config = super().get_config()
        base_config.update(
            {
                "weights": self.config.weights,
                "weight_mode": self.config.weight_mode,
                "distribution": self.config.distribution,
                "score": self.config.score,
                "mu": self.config.mu,
                "delta": self.config.delta,
            }
        )
        return base_config


wSOL = WeightedSOLLoss
