import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from multisol.torch.metrics import (
	        accuracy as acc,
	        precision as prec,
	        recall as rec,
	        specificity as spec,
	        f1_score_fun as f1s,
	        tss as tss_score,
	        gmean as gmean_score,
	    )

# ============================================================================
# Differentiable Monte Carlo indicator for multi-class assignment
# ============================================================================
def multiclass_indicator(y_pred, taus, lam=10.0):
    """
    y_pred:  Tensor of shape (B, m) -- softmax outputs.
    taus:    Tensor of shape (N, m) -- tau samples from a distribution on the simplex.
    lam:     Sigmoid steepness parameter.

    Returns:
       psi: Tensor of shape (B, m) with the differentiable assignment probabilities.
    """
    # Compute pairwise differences for predictions: shape (B, m, m)
    diff_y = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)  # (B, m, m)
    # Compute pairwise differences for taus: shape (N, m, m)
    diff_tau = taus.unsqueeze(2) - taus.unsqueeze(1)      # (N, m, m)
    # Combine differences: shape (B, N, m, m)
    diff = diff_y.unsqueeze(1) - diff_tau.unsqueeze(0)    # (B, N, m, m)
    # Apply sigmoid.
    s = torch.sigmoid(lam * diff)  # shape (B, N, m, m)
    
    # Create a mask for the diagonal elements.
    diag_mask = torch.eye(s.size(-1), device=s.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
    # Use torch.where to set diagonal elements to 1.0, otherwise keep original values.
    s = torch.where(diag_mask, torch.ones_like(s), s)

    # Product over k (axis=-1) and average over tau samples (axis=1).
    prod = s.prod(dim=-1)         # (B, N, m)
    psi = prod.mean(dim=1)        # (B, m)
    return psi

# ============================================================================
# SOL Loss Function for Multi-class (one-vs-rest) case
# ============================================================================
class SOL(nn.Module):
    """
    Score-Oriented Loss (SOL) for multi-class classification.

    This loss computes a soft one-vs-rest confusion matrix by approximating the
    indicator 1{ y_pred in R_j(τ) } via Monte Carlo integration over tau samples.
    """
    def __init__(
        self,
        score="accuracy",
        distribution="uniform",
        mu=0.5,
        delta=0.1,
        mode="average",
        taus=None,
        lam=10.0,
        *,
        add_one=True,
    ):
        """
        Parameters:
          score:        String indicating the score (e.g., 'accuracy', 'precision',
                        'recall', 'specificity', 'f1_score').
          distribution: Not used (tau samples are provided).
          mu, delta:    Parameters for the cosine distribution (ignored if 'uniform').
          mode:         'average' for macro averaging.
	          taus:         A NumPy array or tensor of shape (N, m) containing tau samples.
	          lam:          Sigmoid steepness parameter.
	          add_one:      If True, returns (1 - score) instead of (-score). This is a
	                        constant offset and does not change gradients/optima.
	        """
        super(SOL, self).__init__()
        self.score = score
        self.distribution = distribution
        self.mu = mu
        self.delta = delta
        self.mode = mode
        self.lam = lam
        self.add_one = add_one

        # Set the score function.
        if score == "accuracy":
            self.score_func = acc
        elif score == "precision":
            self.score_func = prec
        elif score == "recall":
            self.score_func = rec
        elif score == "specificity":
            self.score_func = spec
        elif score == "f1_score":
            self.score_func = f1s
        elif score == "tss":
            self.score_func = tss_score
        elif score == "gmean":
            self.score_func = gmean_score
        else:
            self.score_func = acc  # default

        # Convert taus to tensor if necessary.
        if taus is not None:
            if not torch.is_tensor(taus):
                self.taus = torch.tensor(taus, dtype=torch.float32)
            else:
                self.taus = taus.float()
        else:
            raise ValueError("Tau samples must be provided for SOL loss.")

    def forward(self, y_pred, y_true):
        """
        y_pred: Tensor of shape (B, m) for multi-class or (B, 1) for binary.
        y_true: Tensor of the same shape as y_pred.
        """
        y_pred = y_pred.float()

        # Support sparse labels (B,) or (B, 1) by converting to one-hot.
        if y_true.dim() == 1:
            if y_pred.size(1) == 1:
                y_true = y_true.view(-1, 1)
            else:
                y_true = F.one_hot(y_true.to(torch.long), num_classes=y_pred.size(1))
        elif y_true.dim() == 2 and y_true.size(1) == 1 and y_pred.size(1) != 1:
            y_true = F.one_hot(y_true.squeeze(1).to(torch.long), num_classes=y_pred.size(1))

        y_true = y_true.to(device=y_pred.device, dtype=torch.float32)

        # Binary classification branch.
        if y_pred.size(1) == 1:
            TN = torch.sum((1.0 - y_true) * (1.0 - y_pred))
            TP = torch.sum(y_true * y_pred)
            FP = torch.sum((1.0 - y_true) * y_pred)
            FN = torch.sum(y_true * (1.0 - y_pred))
            score_val = self.score_func(TN, FP, FN, TP)
            loss = (1.0 - score_val) if self.add_one else (-score_val)
            return loss
        else:
            # Multi-class branch.
            # Ensure taus is on the same device as y_pred.
            taus = self.taus.to(y_pred.device)
            psi = multiclass_indicator(y_pred, taus, lam=self.lam)  # (B, m)

            # Compute confusion matrix components for each class.
            TP = torch.sum(y_true * psi, dim=0)
            FN = torch.sum(y_true * (1.0 - psi), dim=0)
            FP = torch.sum((1.0 - y_true) * psi, dim=0)
            TN = torch.sum((1.0 - y_true) * (1.0 - psi), dim=0)

            # Compute the score per class and average.
            score_arr = self.score_func(TN, FP, FN, TP)
            final_score = torch.mean(score_arr)
            loss = (1.0 - final_score) if self.add_one else (-final_score)
            return loss
