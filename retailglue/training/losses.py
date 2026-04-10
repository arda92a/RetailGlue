"""Loss functions for LightGlue training."""

import torch
import torch.nn as nn


def weight_loss(log_assignment, weights, gamma=0.0):
    """Compute weighted NLL loss over the log assignment matrix.

    Args:
        log_assignment: [B, M+1, N+1] log-probability assignment matrix.
        weights: [B, M+1, N+1] binary ground truth weights.
        gamma: Focal loss exponent. 0 disables focal weighting.

    Returns:
        Tuple of (nll_pos, nll_neg, num_pos, num_neg) per-batch.
    """
    b, m, n = log_assignment.shape
    m -= 1
    n -= 1

    if gamma > 0:
        probs = log_assignment.exp().clamp(min=1e-6, max=1.0)
        focal_weight = (1 - probs) ** gamma
        loss_sc = log_assignment * weights * focal_weight
    else:
        loss_sc = log_assignment * weights

    num_neg0 = weights[:, :m, -1].sum(-1).clamp(min=1.0)
    num_neg1 = weights[:, -1, :n].sum(-1).clamp(min=1.0)
    num_pos = weights[:, :m, :n].sum((-1, -2)).clamp(min=1.0)

    nll_pos = -loss_sc[:, :m, :n].sum((-1, -2)) / num_pos.clamp(min=1.0)

    nll_neg0 = -loss_sc[:, :m, -1].sum(-1)
    nll_neg1 = -loss_sc[:, -1, :n].sum(-1)
    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)

    return nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0


class NLLLoss(nn.Module):
    """Negative log-likelihood loss for match assignment matrices.

    Balances positive (matched) and negative (unmatched) product losses
    with configurable focal weighting.
    """

    def __init__(self, nll_balancing=0.5, gamma=0.0):
        super().__init__()
        self.nll_balancing = nll_balancing
        self.gamma = gamma

    def forward(self, pred, data, weights=None):
        log_assignment = pred["log_assignment"]
        if weights is None:
            weights = self._build_weights(log_assignment, data)

        nll_pos, nll_neg, num_pos, num_neg = weight_loss(
            log_assignment, weights, gamma=self.gamma
        )
        nll = self.nll_balancing * nll_pos + (1 - self.nll_balancing) * nll_neg

        metrics = {
            "assignment_nll": nll,
            "nll_pos": nll_pos,
            "nll_neg": nll_neg,
            "num_matchable": num_pos,
            "num_unmatchable": num_neg,
        }
        return nll, weights, metrics

    def _build_weights(self, log_assignment, data):
        """Build ground truth weight matrix from match annotations."""
        m = data["gt_matches0"].size(-1)
        n = data["gt_matches1"].size(-1)

        positive = data["gt_assignment"].float()
        neg0 = (data["gt_matches0"] == -1).float()
        neg1 = (data["gt_matches1"] == -1).float()

        weights = torch.zeros_like(log_assignment)
        weights[:, :m, :n] = positive
        weights[:, :m, -1] = neg0
        weights[:, -1, :n] = neg1
        return weights
