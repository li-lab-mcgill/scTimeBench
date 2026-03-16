"""
Wasserstein (Sinkhorn) OT loss for gene expression prediction.
"""
from scTimeBench.metrics.gex_prediction.ot_eval.base import OTLossMetric

import torch
from geomloss import SamplesLoss


class WassersteinOTLoss(OTLossMetric):
    """
    Computes OT loss between ground-truth and predicted next-timepoint gene expression.
    """

    def _defaults(self):
        return {
            "lognorm": False,
            "ot_p": 2,
            "ot_blur": 0.05,
            "ot_scaling": 0.5,
            "ot_debias": True,
            "ot_backend": "tensorized",
            "normalize_by_n_genes": True,
            "aggregate": "mean",
        }

    def _metric_solver_arrays(self, x_true, x_pred, n_genes):
        x_true_t = torch.as_tensor(x_true, dtype=torch.double)
        x_pred_t = torch.as_tensor(x_pred, dtype=torch.double)

        ot1 = SamplesLoss(
            "sinkhorn",
            p=self.ot_p,
            blur=self.ot_blur,
            scaling=self.ot_scaling,
            debias=self.ot_debias,
            backend=self.ot_backend,
        )
        ot_out = ot1(x_true_t, x_pred_t).item()
        if self.normalize_by_n_genes:
            ot_out = ot_out / n_genes
        return ot_out
