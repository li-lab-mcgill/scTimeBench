"""
Wasserstein (Sinkhorn) OT loss for gene expression prediction.
"""
from crispy_fishstick.metrics.gex_prediction.ot_eval.base import OTLossMetric
from crispy_fishstick.shared.constants import ObservationColumns

import os
import numpy as np
import scanpy as sc
import torch
from geomloss import SamplesLoss


class WassersteinOTLoss(OTLossMetric):
    """
    Computes OT loss between ground-truth and predicted next-timepoint gene expression.
    """

    def _get_timepoint_key(self, adata):
        if ObservationColumns.TIMEPOINT.value in adata.obs.columns:
            return ObservationColumns.TIMEPOINT.value
        if "timepoint" in adata.obs.columns:
            return "timepoint"
        raise ValueError("No timepoint column found in AnnData obs.")

    def _load_true_pred_adata(self, output_path, dataset):
        """
        Load ground-truth test data and predicted AnnData with held-out timepoints.
        """
        true_splits = dataset.load_data()
        if not isinstance(true_splits, tuple) or len(true_splits) != 2:
            raise ValueError("Dataset did not return (train, test) splits.")
        _, true_adata = true_splits

        model_output_file = os.path.join(output_path, self.output_path_name.value)
        pred_adata = sc.read_h5ad(model_output_file)

        return true_adata, pred_adata

    def _preprocess_basic(self, adata):
        """Normalize counts per cell to 1e4 and log1p."""
        adata = adata.copy()
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        sc.pp.log1p(adata)
        return adata

    def ot_solver(self, adata_true, adata_pred, lognorm=False):
        """
        Compute Sinkhorn (Wasserstein) distance between two expression matrices.
        """
        shared_genes = np.intersect1d(adata_true.var_names, adata_pred.var_names)
        if len(shared_genes) == 0:
            raise ValueError("No overlapping genes between the two AnnData objects.")
        at = adata_true[:, shared_genes]
        ap = adata_pred[:, shared_genes]

        if lognorm:
            at = self._preprocess_basic(at)
            ap = self._preprocess_basic(ap)

        x_true = at.X.A if hasattr(at.X, "A") else np.asarray(at.X)
        x_pred = ap.X.A if hasattr(ap.X, "A") else np.asarray(ap.X)

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
            ot_out = ot_out / len(shared_genes)
        return ot_out

    def parse_timepoints(self, adata_true, adata_pred, lognorm=False):
        """
        Get timepoint-specific OT results.
        """
        true_tp_key = self._get_timepoint_key(adata_true)
        pred_tp_key = self._get_timepoint_key(adata_pred)

        res = {}
        for tp in np.unique(adata_true.obs[true_tp_key]):
            if tp not in adata_pred.obs[pred_tp_key].values:
                raise ValueError(f"Timepoint {tp} missing in predicted data.")
            res[tp] = self.ot_solver(
                adata_true[adata_true.obs[true_tp_key] == tp],
                adata_pred[adata_pred.obs[pred_tp_key] == tp],
                lognorm=lognorm,
            )

        return res

    def _gex_eval(self, output_path, dataset):
        adata_true, adata_pred = self._load_true_pred_adata(output_path, dataset)
        ot_by_tp = self.parse_timepoints(
            adata_true, adata_pred, lognorm=self.lognorm
        )
        return self._aggregate_ot(ot_by_tp)
