"""
Hausdorff distance for gene expression prediction.
"""
from scTimeBench.metrics.gex_prediction.ot_eval.base import OTLossMetric

import torch


class HausdorffLoss(OTLossMetric):
    """
    Computes Hausdorff distance between ground-truth and predicted next-timepoint
    gene expression.
    """

    def _defaults(self):
        return {
            "lognorm": False,
            "normalize_by_n_genes": False,
            "aggregate": "mean",
        }

    def _metric_solver_arrays(self, x_true, x_pred, n_genes):
        x_true_t = torch.as_tensor(x_true, dtype=torch.double)
        x_pred_t = torch.as_tensor(x_pred, dtype=torch.double)

        dist = torch.cdist(x_true_t, x_pred_t, p=2)
        if dist.numel() == 0:
            raise ValueError("Empty distance matrix for Hausdorff computation.")

        min_true = dist.min(dim=1).values
        min_pred = dist.min(dim=0).values
        out = torch.max(min_true.max(), min_pred.max()).item()

        if self.normalize_by_n_genes:
            out = out / n_genes
        return out
