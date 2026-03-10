"""
Energy distance for gene expression prediction.
"""
from scTimeBench.metrics.gex_prediction.ot_eval.base import OTLossMetric

import torch
from geomloss import SamplesLoss


class EnergyDistanceLoss(OTLossMetric):
    """
    Computes energy distance between ground-truth and predicted next-timepoint
    gene expression.
    """

    def _defaults(self):
        return {
            "lognorm": False,
            "energy_blur": 1.0,
            "energy_debias": True,
            "energy_backend": "tensorized",
            "normalize_by_n_genes": True,
            "aggregate": "mean",
        }

    def _metric_solver_arrays(self, x_true, x_pred, n_genes):
        x_true_t = torch.as_tensor(x_true, dtype=torch.double)
        x_pred_t = torch.as_tensor(x_pred, dtype=torch.double)

        energy = SamplesLoss(
            "energy",
            blur=self.energy_blur,
            debias=self.energy_debias,
            backend=self.energy_backend,
        )
        out = energy(x_true_t, x_pred_t).item()
        if self.normalize_by_n_genes:
            out = out / n_genes
        return out
