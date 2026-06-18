# SOL

Implementation Score-Oriented Loss (SOL) functions for deep learning models.
Supports binary and multiclass classification settings.

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
from sol.torch.multisol import SOL

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
from sol.tf.multisol import SOL

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

## Available Metrics

Both implementations support the following metrics:

- accuracy: Standard classification accuracy
- precision: Precision (positive predictive value)
- recall: Recall (sensitivity, true positive rate)
- specificity: Specificity (true negative rate)
- f1_score: F1 score (harmonic mean of precision and recall)
- tss: True Skill Statistics (recall + specificity - 1)
- gmean: Geometric mean of sensitivity and specificity
