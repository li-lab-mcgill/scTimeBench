"""
OT-based gene expression prediction metric base class.
"""
from crispy_fishstick.metrics.gex_prediction.base import GexPredictionMetrics
from crispy_fishstick.shared.constants import ObservationColumns, RequiredOutputColumns

import numpy as np
import scanpy as sc
import torch
from geomloss import SamplesLoss

import logging
import os


class OTLossMetric(GexPredictionMetrics):
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

    def _gex_eval(self, output_path, dataset):
        adata_true, adata_pred = self._load_true_pred_adata(output_path, dataset)
        ot_by_tp = self._ot_by_timepoint(adata_true, adata_pred)
        results = dict(ot_by_tp)
        results["All"] = self._aggregate_ot(ot_by_tp)
        return results

    def _load_true_pred_adata(self, output_path, dataset):
        """
        Load ground-truth test data and construct predicted AnnData for next-timepoint
        gene expression.
        """
        model_output_file = os.path.join(output_path, self.output_path_name.value)
        pred_adata = sc.read_h5ad(model_output_file)

        if (
            RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION.value
            not in pred_adata.obsm.keys()
        ):
            raise ValueError(
                "Predicted gene expression not found in model output AnnData. "
                "Expected obsm key: "
                f"{RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION.value}."
            )

        data_splits = dataset.load_data()
        if not isinstance(data_splits, tuple) or len(data_splits) != 2:
            raise ValueError("Dataset did not return (train, test) splits.")
        _, true_adata = data_splits

        pred_next_tp_adata = self._make_pred_next_timepoint_adata(pred_adata)
        return true_adata, pred_next_tp_adata

    def _make_pred_next_timepoint_adata(self, pred_adata):
        """
        Construct an AnnData object of predicted next-timepoint gene expression.
        """
        timepoints = pred_adata.obs[ObservationColumns.TIMEPOINT.value].values
        unique_tps = np.sort(np.unique(timepoints))

        pred_expr = pred_adata.obsm[
            RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION.value
        ]
        pred_expr = pred_expr.A if hasattr(pred_expr, "A") else np.asarray(pred_expr)

        valid_indices = []
        next_timepoints = []
        for idx, tp in enumerate(timepoints):
            next_idx = np.searchsorted(unique_tps, tp, side="right")
            if next_idx >= len(unique_tps):
                continue
            row = pred_expr[idx]
            if np.isnan(row).all():
                continue
            valid_indices.append(idx)
            next_timepoints.append(unique_tps[next_idx])

        if len(valid_indices) == 0:
            raise ValueError("No valid predicted next-timepoint gene expression found.")

        pred_expr = pred_expr[valid_indices]
        pred_next_tp_adata = sc.AnnData(
            pred_expr,
            var=pred_adata.var.copy(),
            dtype=pred_expr.dtype,
        )
        pred_next_tp_adata.obs[ObservationColumns.TIMEPOINT.value] = np.array(
            next_timepoints
        )
        return pred_next_tp_adata

    def _preprocess_basic(self, adata):
        """Normalize counts per cell to 1e4 and log1p."""
        adata = adata.copy()
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        sc.pp.log1p(adata)
        return adata

    def _ot_solver(self, adata_true, adata_pred):
        """
        Compute Sinkhorn (Wasserstein) distance between two expression matrices.
        """
        shared_genes = np.intersect1d(adata_true.var_names, adata_pred.var_names)
        if len(shared_genes) == 0:
            raise ValueError("No overlapping genes between the two AnnData objects.")
        at = adata_true[:, shared_genes]
        ap = adata_pred[:, shared_genes]

        if self.lognorm:
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

    def _ot_by_timepoint(self, adata_true, adata_pred):
        """
        Compute OT per timepoint for shared timepoints.
        """
        true_tps = np.unique(adata_true.obs[ObservationColumns.TIMEPOINT.value])
        pred_tps = np.unique(adata_pred.obs[ObservationColumns.TIMEPOINT.value])
        shared_tps = np.intersect1d(true_tps, pred_tps)

        if len(shared_tps) == 0:
            raise ValueError(
                "No overlapping timepoints between true and predicted data."
            )

        results = {}
        for tp in shared_tps:
            true_tp = adata_true[
                adata_true.obs[ObservationColumns.TIMEPOINT.value] == tp
            ]
            pred_tp = adata_pred[
                adata_pred.obs[ObservationColumns.TIMEPOINT.value] == tp
            ]
            results[tp] = self._ot_solver(true_tp, pred_tp)

        logging.debug(f"OT by timepoint: {results}")
        return results

    def _aggregate_ot(self, ot_by_tp):
        values = np.array(list(ot_by_tp.values()), dtype=float)
        if len(values) == 0:
            raise ValueError("No OT values to aggregate.")
        if self.aggregate == "mean":
            return float(np.mean(values))
        if self.aggregate == "median":
            return float(np.median(values))
        if self.aggregate == "sum":
            return float(np.sum(values))
        raise ValueError(
            "Invalid aggregate method. Expected one of: mean, median, sum."
        )
