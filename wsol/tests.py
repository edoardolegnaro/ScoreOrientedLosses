from __future__ import annotations

import numpy as np


def test_torch_wsol_smoke():
    import torch

    from wsol.torch import WeightedSOLLoss

    outputs = torch.tensor([0.1, 0.8, 0.3, 0.9, 0.2], dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    loss = WeightedSOLLoss(score="tss")(outputs, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert outputs.grad is not None


def test_tf_wsol_smoke():
    import tensorflow as tf

    from wsol.tf import WeightedSOLLoss

    outputs = tf.Variable(np.array([0.1, 0.8, 0.3, 0.9, 0.2], dtype=np.float32))
    labels = tf.constant([0.0, 1.0, 0.0, 1.0, 0.0], dtype=tf.float32)
    loss_fn = WeightedSOLLoss(score="tss")
    with tf.GradientTape() as tape:
        loss = loss_fn(labels, outputs)
    gradient = tape.gradient(loss, outputs)
    assert bool(tf.reduce_all(tf.math.is_finite(loss)).numpy())
    assert gradient is not None
