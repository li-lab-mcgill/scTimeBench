"""
MMD loss for gene expression prediction.
"""
from scTimeBench.metrics.gex_prediction.ot_eval.base import OTLossMetric

import torch
from geomloss import SamplesLoss


class MMDLoss(OTLossMetric):
    """
    Computes MMD loss between ground-truth and predicted next-timepoint gene expression.
    """

    def _defaults(self):
        return {
            "lognorm": False,
            "mmd_kernel": "gaussian",
            "mmd_blur": 1.0,
            "mmd_debias": True,
            "mmd_backend": "tensorized",
            "normalize_by_n_genes": True,
            "aggregate": "mean",
        }

    def _metric_solver_arrays(self, x_true, x_pred, n_genes):
        x_true_t = torch.as_tensor(x_true, dtype=torch.double)
        x_pred_t = torch.as_tensor(x_pred, dtype=torch.double)

        mmd = SamplesLoss(
            self.mmd_kernel,
            blur=self.mmd_blur,
            debias=self.mmd_debias,
            backend=self.mmd_backend,
        )
        out = mmd(x_true_t, x_pred_t).item()
        if self.normalize_by_n_genes:
            out = out / n_genes
        return out
