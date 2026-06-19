# Score-Oriented Losses

Implementations of Score-Oriented Loss (SOL) functions for deep learning
models. The repository includes:

- `multisol`: binary and multiclass SOL for PyTorch and TensorFlow.
- `wsol`: binary weighted SOL for temporally weighted event detection.

The reference paper for the `multisol` implementation is:

Marchetti, F., Legnaro, E., & Guastavino, S. (2025). The Multiclass
Score-Oriented Loss (MultiSOL) on the Simplex. arXiv preprint arXiv:2511.22587.

## Installation

```bash
git clone https://github.com/edoardolegnaro/ScoreOrientedLosses.git
cd ScoreOrientedLosses
pip install -e .
```

## Usage

### PyTorch Example

```python
import torch
import numpy as np
from multisol.torch.multisol import SOL

# Generate tau samples on the simplex
N = 1000  # Number of samples
m = 3     # Number of classes
tau_samples = np.random.rand(N, m)

# Create SOL loss with accuracy as target metric
sol_loss = SOL(score="accuracy", taus=tau_samples, lam=10.0)

# Use in training loop
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())

for X_batch, y_batch in data_loader:
    optimizer.zero_grad()
    y_pred = model(X_batch)
    loss = sol_loss(y_pred, y_batch)
    loss.backward()
    optimizer.step()
```

### TensorFlow Example

```python
import tensorflow as tf
import numpy as np
from multisol.tf.multisol import SOL

# Generate tau samples on the simplex
N = 1000  # Number of samples
m = 3     # Number of classes
tau_samples = np.random.rand(N, m)

# Create SOL loss with accuracy as target metric
sol_loss = SOL(score="accuracy", taus=tau_samples, lam=10.0)

# Create and compile model
model = tf.keras.Sequential([
    # your layers here
])
model.compile(optimizer='adam', loss=sol_loss)

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

## Weighted SOL (wSOL)

wSOL extends binary SOL with temporal weights. It is useful when detections
near future positive labels should be penalized differently from ordinary false
positives, and when missed positive labels should account for recent previous
outputs.

### PyTorch wSOL

```python
import torch
from wsol.torch import WeightedSOLLoss

loss_fn = WeightedSOLLoss(
    weights=(0.5, 0.25, 0.125),
    weight_mode="prod",
    score="tss",
)

y_pred = torch.tensor([0.1, 0.8, 0.3, 0.9])
y_true = torch.tensor([0.0, 1.0, 0.0, 1.0])
loss = loss_fn(y_pred, y_true)
```

`y_pred` must contain probabilities in `[0, 1]`. Apply a sigmoid before the
loss if your model returns logits.

### TensorFlow wSOL

```python
import tensorflow as tf
from wsol.tf import WeightedSOLLoss

loss_fn = WeightedSOLLoss(
    weights=(0.5, 0.25, 0.125),
    weight_mode="prod",
    score="tss",
)

model = tf.keras.Sequential([
    # your layers here
])
model.compile(optimizer="adam", loss=loss_fn)
```

### wSOL Options

- `weights`: non-increasing positive temporal weights.
- `weight_mode`: `"prod"` or `"max"`.
- `distribution`: `"uniform"` or `"cosine"`.
- `score`: target score optimized by the loss.

## Available Metrics

SOL supports the following metrics:

- accuracy: Standard classification accuracy
- precision: Precision (positive predictive value)
- recall: Recall (sensitivity, true positive rate)
- specificity: Specificity (true negative rate)
- f1_score: F1 score (harmonic mean of precision and recall)
- tss: True Skill Statistics (recall + specificity - 1)
- gmean: Geometric mean of sensitivity and specificity

wSOL supports all of the above plus `balanced_accuracy`, `csi`, and `hss`.
