import tensorflow as tf

from multisol.tf.metrics import (
    accuracy,
    precision,
    recall,
    specificity,
    f1_score_fun,
    tss,
    gmean,
)


# ============================================================================
# Differentiable Monte Carlo indicator for multi-class assignment
#
# For a given sample with prediction vector y_pred (in S_m) and tau samples from
# a Dirichlet on the simplex, we approximate:
#
#    φ_j(y_pred, τ) = ∏_{k≠j} σ(λ [ y_pred^j - y_pred^k - (τ^j - τ^k) ])
#
# Averaging over N tau samples gives:
#
#    ψ_j(y_pred) = (1/N) ∑_{l=1}^{N} φ_j(y_pred, τ^(l)).
#
# This ψ_j is used in a one-vs-rest confusion matrix.
# ============================================================================
@tf.function
def multiclass_indicator(y_pred, taus, lam=10.0):
    """
    y_pred:  Tensor of shape (B, m) -- softmax outputs.
    taus:    Tensor of shape (N, m) -- tau samples from a distribution on S_m.
    lam:     Sigmoid steepness parameter.

    Returns:
       psi: Tensor of shape (B, m) with the differentiable assignment probabilities.
    """
    # Compute pairwise differences for predictions: shape (B, m, m).
    diff_y = y_pred[:, :, None] - y_pred[:, None, :]

    # Compute pairwise differences for taus: shape (N, m, m).
    diff_tau = taus[:, :, None] - taus[:, None, :]

    # Combine differences: shape (B, N, m, m).
    diff = diff_y[:, None, :, :] - diff_tau[None, :, :, :]

    # Apply sigmoid.
    s = tf.sigmoid(lam * diff)  # shape (B, N, m, m)

    # Set diagonal (j==k) to 1
    s = tf.linalg.set_diag(s, tf.ones(tf.shape(s)[:-1], dtype=s.dtype))

    # Product over k (axis=-1) and average over tau samples.
    prod = tf.reduce_prod(s, axis=-1)  # shape (B, N, m)
    psi = tf.reduce_mean(prod, axis=1)  # shape (B, m)
    return psi


# ============================================================================
# SOL Loss Function for Multi-class (one-vs-rest) case
# ============================================================================
# Avoid setting global XLA/JIT at import time; it can break gradients on some
# CPU setups. Enable it explicitly in your training script if desired.
# tf.config.optimizer.set_jit(True)


def SOL(
    score="accuracy",
    taus=None,
    lam=10.0,
    *,
    add_one=True,
):
    """
    Score-Oriented Loss (SOL) for multi-class classification.

    Computes a soft one-vs-rest confusion matrix by approximating the indicator
    1{ y_pred in R_j(τ) } via Monte Carlo integration over tau samples.

    Parameters:
      score:        String indicating the score (e.g., 'accuracy').
      taus:         A NumPy array or tensor of shape (N, m) containing tau samples.
      lam:          Sigmoid steepness parameter.
      add_one:      If True, returns (1 - score) instead of (-score). This is a
                    constant offset and does not change gradients/optima.

    Returns:
      A loss function SOL_(y_true, y_pred) for model.compile.
    """
    if taus is None:
        raise ValueError("`taus` must be provided (shape (N, m)).")

    # Select score function from metrics.
    if score == "accuracy":
        score_func = accuracy
    elif score == "precision":
        score_func = precision
    elif score == "recall":
        score_func = recall
    elif score == "specificity":
        score_func = specificity
    elif score == "f1_score":
        score_func = f1_score_fun
    elif score == "tss":
        score_func = tss
    elif score == "gmean":
        score_func = gmean
    else:
        score_func = accuracy  # default

    taus = tf.convert_to_tensor(taus, dtype=tf.float32)

    @tf.function
    def SOL_(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Support sparse labels (B,) or (B, 1):
        # - If y_pred is multiclass (B, m), convert indices -> one-hot (B, m).
        # - If y_pred is binary (B, 1), keep labels as (B, 1) float in {0,1}.
        num_classes = tf.shape(y_pred)[1]
        y_true_rank = tf.rank(y_true)

        def _from_rank1():
            return tf.cond(
                tf.equal(num_classes, 1),
                lambda: tf.cast(tf.expand_dims(y_true, axis=-1), tf.float32),
                lambda: tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes, dtype=tf.float32),
            )

        def _from_rank2():
            static_last_dim = y_true.shape[1]
            if static_last_dim is not None and static_last_dim != 1:
                return tf.cast(y_true, tf.float32)

            def _sparse_col():
                return tf.cond(
                    tf.equal(num_classes, 1),
                    lambda: tf.cast(y_true, tf.float32),
                    lambda: tf.one_hot(
                        tf.cast(tf.squeeze(y_true, axis=1), tf.int32),
                        depth=num_classes,
                        dtype=tf.float32,
                    ),
                )

            last_dim = tf.shape(y_true)[1]
            return tf.cond(tf.equal(last_dim, 1), _sparse_col, lambda: tf.cast(y_true, tf.float32))

        y_true = tf.cond(
            tf.equal(y_true_rank, 1),
            _from_rank1,
            lambda: tf.cond(tf.equal(y_true_rank, 2), _from_rank2, lambda: tf.cast(y_true, tf.float32)),
        )

        # Binary classification branch.
        if y_pred.shape[1] == 1:
            TN = tf.reduce_sum((1.0 - y_true) * (1.0 - y_pred))
            TP = tf.reduce_sum(y_true * y_pred)
            FP = tf.reduce_sum((1.0 - y_true) * y_pred)
            FN = tf.reduce_sum(y_true * (1.0 - y_pred))
            score_val = score_func(TN, FP, FN, TP)
            return (1.0 - score_val) if add_one else (-score_val)
        else:
            # Multi-class branch.
            psi = multiclass_indicator(y_pred, taus, lam=lam)  # shape (B, m)

            # Compute confusion matrix components for each class.
            TP = tf.reduce_sum(y_true * psi, axis=0)
            FN = tf.reduce_sum(y_true * (1.0 - psi), axis=0)
            FP = tf.reduce_sum((1.0 - y_true) * psi, axis=0)
            TN = tf.reduce_sum((1.0 - y_true) * (1.0 - psi), axis=0)

            # Compute the score per class and average.
            score_arr = score_func(TN, FP, FN, TP)
            final_score = tf.reduce_mean(score_arr)
            return (1.0 - final_score) if add_one else (-final_score)

    return SOL_
