"""Evaluation metrics for product-level matching."""

import torch


@torch.no_grad()
def matcher_metrics(pred, data):
    """Compute matching metrics: recall, precision, accuracy, and AP.

    Args:
        pred: Dict with 'matches0', 'matches1', 'matching_scores0'.
        data: Dict with 'gt_matches0', 'gt_matches1'.

    Returns:
        Dict of per-batch metric tensors.
    """
    def recall(m, gt_m):
        mask = (gt_m > -1).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def precision(m, gt_m):
        mask = ((m > -1) & (gt_m >= -1)).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def accuracy(m, gt_m):
        mask = (gt_m >= -1).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def ranking_ap(m, gt_m, scores):
        p_mask = ((m > -1) & (gt_m >= -1)).float()
        r_mask = (gt_m > -1).float()
        sort_ind = torch.argsort(-scores)
        sorted_p_mask = torch.gather(p_mask, -1, sort_ind)
        sorted_r_mask = torch.gather(r_mask, -1, sort_ind)
        sorted_tp = torch.gather(m == gt_m, -1, sort_ind)
        p_pts = torch.cumsum(sorted_tp * sorted_p_mask, -1) / (
            1e-8 + torch.cumsum(sorted_p_mask, -1)
        )
        r_pts = torch.cumsum(sorted_tp * sorted_r_mask, -1) / (
            1e-8 + sorted_r_mask.sum(-1)[:, None]
        )
        r_pts_diff = r_pts[..., 1:] - r_pts[..., :-1]
        return torch.sum(r_pts_diff * p_pts[:, None, -1], dim=-1)

    return {
        "match_recall": recall(pred["matches0"], data["gt_matches0"]),
        "match_precision": precision(pred["matches0"], data["gt_matches0"]),
        "accuracy": accuracy(pred["matches0"], data["gt_matches0"]),
        "average_precision": ranking_ap(
            pred["matches0"], data["gt_matches0"], pred["matching_scores0"]
        ),
    }
