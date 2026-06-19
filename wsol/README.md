# wSOL

Binary weighted Score-Oriented Loss (wSOL) implementations for PyTorch and
TensorFlow/Keras.

## PyTorch

```python
from wsol.torch import WeightedSOLLoss

loss_fn = WeightedSOLLoss(
    weights=(0.5, 0.25, 0.125),
    weight_mode="prod",
    score="tss",
)
loss = loss_fn(y_pred, y_true)
```

`y_pred` is expected to contain probabilities in `[0, 1]`. Apply a sigmoid
before calling the loss if your model returns logits.

## TensorFlow

```python
from wsol.tf import WeightedSOLLoss

loss_fn = WeightedSOLLoss(
    weights=(0.5, 0.25, 0.125),
    weight_mode="prod",
    score="tss",
)
model.compile(optimizer="adam", loss=loss_fn)
```

## Options

- `weights`: non-increasing temporal weights for the look-ahead/look-back
  horizon.
- `weight_mode`: `"prod"` or `"max"`.
- `score`: `accuracy`, `precision`, `recall`, `specificity`,
  `balanced_accuracy`, `f1_score`, `tss`, `csi`, `hss`, or `gmean`.
- `distribution`: `"uniform"` or `"cosine"`.
