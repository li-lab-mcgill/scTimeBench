"""
OT-based gene expression prediction metric base class.
"""
from scTimeBench.metrics.gex_prediction.base import GexPredictionMetrics
from scTimeBench.shared.constants import ObservationColumns, RequiredOutputFiles
from scTimeBench.shared.utils import load_output_file

import numpy as np
import scanpy as sc

import logging


class OTLossMetric(GexPredictionMetrics):
    def _defaults(self):
        return {
            "lognorm": False,
            "normalize_by_n_genes": True,
            "aggregate": "mean",
        }

    def _setup_model_output_requirements(self):
        self.required_outputs = [
            RequiredOutputFiles.NEXT_TIMEPOINT_GENE_EXPRESSION,
        ]

    def _gex_eval(self, output_path, dataset):
        adata_true, adata_pred = self._load_true_pred_adata(output_path, dataset)
        metric_by_tp = self._metric_by_timepoint(adata_true, adata_pred)
        results = dict(metric_by_tp)
        results["All"] = self._aggregate_ot(metric_by_tp)
        return results

    def _metric_solver_arrays(self, x_true, x_pred, n_genes):
        raise NotImplementedError("Subclasses must implement _metric_solver_arrays.")

    def _load_true_pred_adata(self, output_path, dataset):
        """
        Load ground-truth test data and construct predicted AnnData for next-timepoint
        gene expression.
        """
        pred_expr = load_output_file(
            output_path, RequiredOutputFiles.NEXT_TIMEPOINT_GENE_EXPRESSION
        )

        data_splits = dataset.load_data()
        if not isinstance(data_splits, tuple) or len(data_splits) != 2:
            raise ValueError("Dataset did not return (train, test) splits.")
        _, true_adata = data_splits

        pred_next_tp_adata = self._make_pred_next_timepoint_adata(true_adata, pred_expr)
        return true_adata, pred_next_tp_adata

    def _make_pred_next_timepoint_adata(self, true_adata, pred_expr):
        """
        Construct an AnnData object of predicted next-timepoint gene expression.

        Supports two model output conventions for NEXT_TIMEPOINT_GENE_EXPRESSION:
        1) one row per test cell (rows for terminal timepoints may be NaN),
        2) one row per non-terminal test cell.
        """
        timepoints = true_adata.obs[ObservationColumns.TIMEPOINT.value].values
        unique_tps = np.sort(np.unique(timepoints))
        pred_expr = pred_expr.A if hasattr(pred_expr, "A") else np.asarray(pred_expr)

        if pred_expr.ndim != 2:
            raise ValueError(
                "Predicted next-timepoint gene expression must be a 2D array."
            )

        num_cells = len(timepoints)

        valid_base_indices = []
        inferred_next_timepoints = []
        for idx, tp in enumerate(timepoints):
            next_idx = np.searchsorted(unique_tps, tp, side="right")
            if next_idx >= len(unique_tps):
                continue
            valid_base_indices.append(idx)
            inferred_next_timepoints.append(unique_tps[next_idx])

        if pred_expr.shape[0] == num_cells:
            candidate_rows = list(range(num_cells))
            candidate_tps = []
            for idx, tp in enumerate(timepoints):
                next_idx = np.searchsorted(unique_tps, tp, side="right")
                if next_idx >= len(unique_tps):
                    candidate_tps.append(None)
                else:
                    candidate_tps.append(unique_tps[next_idx])
        elif pred_expr.shape[0] == len(valid_base_indices):
            candidate_rows = list(range(len(valid_base_indices)))
            candidate_tps = inferred_next_timepoints
        else:
            raise ValueError(
                "Unexpected shape for predicted next-timepoint expression. "
                f"Got {pred_expr.shape[0]} rows for {num_cells} test cells "
                f"({len(valid_base_indices)} non-terminal)."
            )

        valid_rows = []
        next_timepoints = []
        for row_idx, next_tp in zip(candidate_rows, candidate_tps):
            if next_tp is None:
                continue
            row = pred_expr[row_idx]
            if np.isnan(row).all():
                continue
            valid_rows.append(row_idx)
            next_timepoints.append(next_tp)

        if len(valid_rows) == 0:
            raise ValueError("No valid predicted next-timepoint gene expression found.")

        pred_expr = pred_expr[valid_rows]
        pred_next_tp_adata = sc.AnnData(
            pred_expr,
            var=true_adata.var.copy(),
            dtype=pred_expr.dtype,
        )
        pred_next_tp_adata.obs[ObservationColumns.TIMEPOINT.value] = np.array(
            next_timepoints
        )
        return pred_next_tp_adata

    def _get_timepoint_key(self, adata):
        if ObservationColumns.TIMEPOINT.value in adata.obs.columns:
            return ObservationColumns.TIMEPOINT.value
        if "timepoint" in adata.obs.columns:
            return "timepoint"
        raise ValueError("No timepoint column found in AnnData obs.")

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

    def _metric_by_timepoint(self, adata_true, adata_pred):
        true_tp_key = self._get_timepoint_key(adata_true)
        pred_tp_key = self._get_timepoint_key(adata_pred)

        true_tps = np.unique(adata_true.obs[true_tp_key])
        pred_tps = np.unique(adata_pred.obs[pred_tp_key])
        shared_tps = np.intersect1d(true_tps, pred_tps)
        if len(shared_tps) == 0:
            raise ValueError(
                "No overlapping timepoints between true and predicted data."
            )
        shared_genes = np.intersect1d(adata_true.var_names, adata_pred.var_names)
        if len(shared_genes) == 0:
            raise ValueError("No overlapping genes between the two AnnData objects.")

        true_gene_idx = np.where(np.isin(adata_true.var_names, shared_genes))[0]
        pred_gene_idx = np.where(np.isin(adata_pred.var_names, shared_genes))[0]

        results = {}
        for tp in shared_tps:
            true_mask = np.asarray(adata_true.obs[true_tp_key] == tp)
            pred_mask = np.asarray(adata_pred.obs[pred_tp_key] == tp)
            if not np.any(true_mask) or not np.any(pred_mask):
                continue

            true_expr = self._to_dense(adata_true.X[true_mask][:, true_gene_idx])
            pred_expr = self._to_dense(adata_pred.X[pred_mask][:, pred_gene_idx])

            if self.lognorm:
                true_expr = self._lognorm_matrix(true_expr)
                pred_expr = self._lognorm_matrix(pred_expr)

            results[str(tp)] = self._metric_solver_arrays(
                true_expr,
                pred_expr,
                len(shared_genes),
            )

        logging.debug(f"{self.__class__.__name__} by timepoint: {results}")
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
