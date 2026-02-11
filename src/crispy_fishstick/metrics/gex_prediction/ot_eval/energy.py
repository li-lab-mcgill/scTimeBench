"""
Energy distance for gene expression prediction.
"""
from crispy_fishstick.metrics.gex_prediction.ot_eval.base import OTLossMetric
from crispy_fishstick.shared.constants import ObservationColumns, RequiredOutputColumns

import numpy as np
import scanpy as sc
import torch
from geomloss import SamplesLoss

import os


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

    def _energy_solver_arrays(self, x_true, x_pred, n_genes):
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

            res[next_tp] = self._energy_solver_arrays(
                true_expr, pred_expr, len(shared_genes)
            )

        if len(res) == 0:
            raise ValueError(
                "No overlapping timepoints between true and predicted data."
            )
        results = dict(res)
        results["All"] = self._aggregate_ot(res)
        return results
