"""
This script implements a simple correlation-based model for trajectory inference.
It does not actually train a model, but simply generates a predicted graph based
on the correlation of cell types at adjacent time points.
"""

import numpy as np
from scipy.sparse import issparse
from scipy.stats import rankdata
from scTimeBench.model_utils.model_runner import main, BaseMethod
from scTimeBench.shared.constants import RequiredOutputFiles
from scTimeBench.shared.constants import ObservationColumns
from enum import Enum
from tqdm import tqdm


class AveragingMethod(Enum):
    AVERAGE = "average"
    MAXIMUM = "maximum"
    NON_NEGATIVE_AVG = "non-negative-avg"


class CorrelationMethod(Enum):
    PEARSONR = "pearsonr"
    SPEARMANR = "spearmanr"


class Correlation(BaseMethod):
    def __init__(self, yaml_config):
        super().__init__(yaml_config)

        # select the option that has PRED_GRAPH
        for option in self.required_outputs_options:
            if RequiredOutputFiles.PRED_GRAPH in option:
                self.required_outputs = option
                break

    def train(self, ann_data, all_tps=None):
        """
        For correlation, we don't actually train a model :)
        """
        print(f"No such training exists for correlation :)")

    def generate_pred_graph(self, test_ann_data) -> np.ndarray:
        """
        Build a predicted graph from cell-level GEx correlations.

        1. For every adjacent (t, t + 1):
          a. Choose one correlation method from model metadata via
              metadata['correlation_method'] in {'pearsonr', 'spearmanr'}
              (default: 'spearmanr').
          b. Compute that correlation for all cell pairs between t and t + 1.
        2. For each source cell at t, compute a weighted average correlation for each
           destination cell type at t + 1 and pick the destination cell type with
           the maximum weighted average.

        Votes are accumulated at the source-cell-type -> destination-cell-type level,
        then each source row is normalized to sum to 1.
        """
        test_tps = (
            test_ann_data.obs[ObservationColumns.TIMEPOINT.value].unique().tolist()
        )
        test_tps.sort()

        # ** Important: we require the same ordering as the graphsimmetrics for this to work **
        cell_types = (
            test_ann_data.obs[ObservationColumns.CELL_TYPE.value].unique().tolist()
        )
        cell_type_to_id = {
            cell_type: idx for idx, cell_type in enumerate(sorted(cell_types))
        }

        time_col = ObservationColumns.TIMEPOINT.value
        obs = test_ann_data.obs
        graph_pred = np.zeros((len(cell_types), len(cell_types)))
        metadata = self.config.get("method", {}).get("metadata", {})
        corr_method = CorrelationMethod(metadata.get("correlation_method", "spearmanr"))
        # also get the averaging method of either the maximum, average, or non-negative average
        avg_method = AveragingMethod(metadata.get("averaging_method", "average"))

        for i in tqdm(range(len(test_tps) - 1)):
            t0 = test_tps[i]
            t1 = test_tps[i + 1]

            sorted_cell_types = sorted(cell_types)

            idx_t0 = np.where(obs[time_col] == t0)[0]
            idx_t1 = np.where(obs[time_col] == t1)[0]
            if idx_t0.size == 0 or idx_t1.size == 0:
                continue

            X_t0 = test_ann_data.X[idx_t0]
            X_t1 = test_ann_data.X[idx_t1]
            X_t0 = X_t0.toarray() if issparse(X_t0) else np.asarray(X_t0)
            X_t1 = X_t1.toarray() if issparse(X_t1) else np.asarray(X_t1)

            if X_t0.shape[1] == 0:
                continue

            # 1. Compute correlation for all cell pairs between t and t + 1.
            # Vectorized path: standardize rows then use matrix multiplication (BLAS-backed).
            if corr_method == CorrelationMethod.SPEARMANR:
                X_t0_corr = rankdata(X_t0, axis=1, method="average")
                X_t1_corr = rankdata(X_t1, axis=1, method="average")
            else:
                X_t0_corr = X_t0
                X_t1_corr = X_t1

            mean_t0 = X_t0_corr.mean(axis=1, keepdims=True)
            mean_t1 = X_t1_corr.mean(axis=1, keepdims=True)
            std_t0 = X_t0_corr.std(axis=1, keepdims=True)
            std_t1 = X_t1_corr.std(axis=1, keepdims=True)

            # Avoid division by zero for constant-expression cells.
            std_t0[std_t0 == 0] = 1.0
            std_t1[std_t1 == 0] = 1.0

            z_t0 = (X_t0_corr - mean_t0) / std_t0
            z_t1 = (X_t1_corr - mean_t1) / std_t1

            n_features = z_t0.shape[1]
            corr = (z_t0 @ z_t1.T) / n_features
            corr = np.nan_to_num(corr, nan=0.0)
            corr = np.clip(corr, -1.0, 1.0)

            # cell types for t0 and t1
            ct_t0 = obs.iloc[idx_t0][ObservationColumns.CELL_TYPE.value].to_numpy()
            ct_t1 = obs.iloc[idx_t1][ObservationColumns.CELL_TYPE.value].to_numpy()

            # cell types present in t1 (destination) - we only want to consider these as options for destination cell types
            present_ct_t1 = [ct for ct in sorted_cell_types if np.any(ct_t1 == ct)]

            # 2. For each cell in t0 compute the weighted average
            for cell_idx_t0, src_ct in enumerate(ct_t0):
                best_dst_ct = None
                best_score = -np.inf

                # iterate over the possible cell types
                for dst_ct in present_ct_t1:
                    # create a mask that finds all the cell types of dst_ct
                    dst_mask = ct_t1 == dst_ct
                    dst_corr = corr[cell_idx_t0, dst_mask]

                    # then take the average correlation for this destination cell type as the score
                    if avg_method == AveragingMethod.MAXIMUM:
                        score = dst_corr.max()
                    elif avg_method == AveragingMethod.NON_NEGATIVE_AVG:
                        # Use non-negative correlation as weights for the per-type average.
                        weights = np.maximum(dst_corr, 0.0)
                        if weights.sum() > 0:
                            score = np.average(dst_corr, weights=weights)
                        else:
                            score = dst_corr.mean()
                    else:
                        score = dst_corr.mean()

                    if score > best_score:
                        best_score = score
                        best_dst_ct = dst_ct

                # increment the predicted graph at the source cell type and best destination cell type
                graph_pred[cell_type_to_id[src_ct], cell_type_to_id[best_dst_ct]] += 1

        for i in range(graph_pred.shape[0]):
            if graph_pred[i, :].sum() > 0:
                graph_pred[i, :] = graph_pred[i, :] / graph_pred[i, :].sum()

        return graph_pred


if __name__ == "__main__":
    main(Correlation)
