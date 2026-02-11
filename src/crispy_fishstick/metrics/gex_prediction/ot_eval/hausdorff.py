"""
Hausdorff distance for gene expression prediction.
"""
from crispy_fishstick.metrics.gex_prediction.ot_eval.base import OTLossMetric
from crispy_fishstick.shared.constants import ObservationColumns, RequiredOutputColumns

import numpy as np
import scanpy as sc
import torch

import os


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

    def _to_dense(self, matrix):
        if hasattr(matrix, "A"):
            return matrix.A
        if hasattr(matrix, "toarray"):
            return matrix.toarray()
        return np.asarray(matrix)

    def _lognorm_matrix(self, matrix):
        totals = np.sum(matrix, axis=1)
        scale = np.where(totals == 0, 0.0, 1e4 / totals)
        normed = matrix * scale[:, None]
        return np.log1p(normed)

    def _hausdorff_solver_arrays(self, x_true, x_pred, n_genes):
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

    def _gex_eval(self, output_path, dataset):
        adata_true, adata_pred = self._load_true_pred_adata(output_path, dataset)

        if (
            RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION.value
            not in adata_pred.obsm
        ):
            raise ValueError(
                "Predicted gene expression not found in model output AnnData. "
                "Expected obsm key: "
                f"{RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION.value}."
            )

        true_tp_key = self._get_timepoint_key(adata_true)
        pred_tp_key = self._get_timepoint_key(adata_pred)

        true_tps = np.sort(np.unique(adata_true.obs[true_tp_key]))
        pred_tps = np.unique(adata_pred.obs[pred_tp_key])

        shared_genes = np.intersect1d(adata_true.var_names, adata_pred.var_names)
        if len(shared_genes) == 0:
            raise ValueError("No overlapping genes between the two AnnData objects.")

        true_gene_idx = np.where(np.isin(adata_true.var_names, shared_genes))[0]
        pred_gene_idx = np.where(np.isin(adata_pred.var_names, shared_genes))[0]

        pred_expr_all = self._to_dense(
            adata_pred.obsm[RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION.value]
        )

        res = {}
        for tp in pred_tps:
            next_idx = np.searchsorted(true_tps, tp, side="right")
            if next_idx >= len(true_tps):
                continue
            next_tp = true_tps[next_idx]

            true_mask = (adata_true.obs[true_tp_key] == next_tp).to_numpy()
            pred_mask = (adata_pred.obs[pred_tp_key] == tp).to_numpy()
            if not np.any(true_mask) or not np.any(pred_mask):
                continue

            true_expr = self._to_dense(adata_true.X[true_mask][:, true_gene_idx])
            pred_expr = pred_expr_all[pred_mask][:, pred_gene_idx]

            if self.lognorm:
                true_expr = self._lognorm_matrix(true_expr)
                pred_expr = self._lognorm_matrix(pred_expr)

            res[next_tp] = self._hausdorff_solver_arrays(
                true_expr, pred_expr, len(shared_genes)
            )

        if len(res) == 0:
            raise ValueError(
                "No overlapping timepoints between true and predicted data."
            )
        results = dict(res)
        results["All"] = self._aggregate_ot(res)
        return results
