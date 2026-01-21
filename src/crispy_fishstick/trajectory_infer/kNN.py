"""
kNN implementation for trajectory inference.
"""
from crispy_fishstick.trajectory_infer.base import BaseTrajectoryInferMethod
from sklearn.neighbors import NearestNeighbors
from crispy_fishstick.shared.constants import ObservationColumns, RequiredOutputColumns
import numpy as np
import json
import logging
from enum import Enum


class kNNStrategy(Enum):
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"


# TODO: build a unit test for this class, to ensure that we're doing this properly
class kNN(BaseTrajectoryInferMethod):
    def __init__(self, traj_config):
        super().__init__(traj_config)
        # sets the default number of neighbors
        self.n_neighbors = traj_config.get("n_neighbors", 5)
        self.strategy = kNNStrategy(
            traj_config.get("strategy", kNNStrategy.MAJORITY_VOTE.value)
        )

    def _method_infer_trajectory(self, ann_data):
        """
        Infer the trajectory using kNN graph-based method.

        1. We can accomplish this by first separating each embedding based on time.
        2. Then, for each time point, we find the k nearest neighbors in the next time point's
        embedding space.
        3. Finally, we consolidate the cell types per time point based on the kNN results.
        """
        # get the embeddings and timepoints
        embeddings = ann_data.obsm[RequiredOutputColumns.EMBEDDING.value]
        next_timepoint_embeddings = ann_data.obsm[
            RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING.value
        ]
        timepoints = ann_data.obs[ObservationColumns.TIMEPOINT.value]
        cell_types = ann_data.obs[ObservationColumns.CELL_TYPE.value]
        unique_timepoints = sorted(np.unique(timepoints))

        cell_lineage = {}
        for i in range(len(unique_timepoints) - 1):
            # get indices for current timepoint
            idx_current = np.where(timepoints == unique_timepoints[i])[0]
            # get indices for next timepoint
            idx_next = np.where(timepoints == unique_timepoints[i + 1])[0]
            # get the cell types for the next timepoint, where idx_next[i] corresponds to next_cell_types[i]
            next_cell_types = cell_types.iloc[idx_next]

            # get embeddings for current and next timepoints
            pred_emb_next = next_timepoint_embeddings[idx_current]
            embed_next = embeddings[idx_next]

            # build kNN model based on next timepoint embeddings
            if embed_next.shape[0] < self.n_neighbors:
                n_neighbors = embed_next.shape[0]
            else:
                n_neighbors = self.n_neighbors

            # now we fit a kNN on the true next timepoint embeddings
            knn_model = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric=self.traj_config.get("method", "minkowski"),
            )
            knn_model.fit(embed_next)

            # find kNN for each cell in current timepoint's predicted next embeddings
            # this results in indices of shape (num_cells_current_timepoint, n_neighbors)
            # where the n_neighbors are indices from embed_next
            _, indices = knn_model.kneighbors(pred_emb_next)

            # we want the number of rows to match the number of cells in the current timepoint
            assert indices.shape[0] == idx_current.shape[0]

            # map cell types from current to next timepoint
            for j, idx in enumerate(idx_current):
                # these are the original cell types
                source_cell_type = cell_types.iloc[idx]
                # then we want the cell types of the neighbours, by
                # getting the indices from the next timepoint which are closest

                # indices[j]: represents the embed_next indices in idx_next
                # next_cell_types: indexed by idx_next
                target_cell_types = next_cell_types.iloc[indices[j]].values

                # then each source cell type maps to multiple target cell types
                # where we'll be keeping count of each cell type to create a distribution later
                if source_cell_type not in cell_lineage:
                    cell_lineage[source_cell_type] = {}

                if self.strategy == kNNStrategy.MAJORITY_VOTE:
                    # majority vote: simply count each occurrence
                    cell_type_counts = {}
                    for target_cell_type in target_cell_types:
                        cell_type_counts[target_cell_type] = (
                            cell_type_counts.get(target_cell_type, 0) + 1
                        )

                    majority_vote_cell_type = max(
                        cell_type_counts.items(), key=lambda x: x[1]
                    )[0]
                    cell_lineage[source_cell_type][majority_vote_cell_type] = (
                        cell_lineage[source_cell_type].get(majority_vote_cell_type, 0)
                        + 1
                    )
                elif self.strategy == kNNStrategy.WEIGHTED_AVERAGE:
                    # weighted average: weight by inverse distance
                    cell_type_weights = {}

                    for rank, target_cell_type in enumerate(target_cell_types):
                        weight = 1 / (rank + 1)  # simple inverse rank weighting
                        cell_type_weights[target_cell_type] = (
                            cell_type_weights.get(target_cell_type, 0) + weight
                        )

                    # because k can differ (due to cell type counts), we normalize weights
                    total_weight = sum(cell_type_weights.values())
                    for target_cell_type in cell_type_weights:
                        cell_type_weights[target_cell_type] /= total_weight

                    for target_cell_type, weight in cell_type_weights.items():
                        cell_lineage[source_cell_type][target_cell_type] = (
                            cell_lineage[source_cell_type].get(target_cell_type, 0)
                            + weight
                        )

            logging.debug(
                f"Processed timepoint {unique_timepoints[i]} resulting in lineage: {cell_lineage}"
            )

        # then we should normalize the counts to get probabilities
        for source_cell_type in cell_lineage.keys():
            total_counts = sum(cell_lineage[source_cell_type].values())
            for target_cell_type in cell_lineage[source_cell_type]:
                cell_lineage[source_cell_type][target_cell_type] /= total_counts

        return cell_lineage

    def __str__(self):
        return json.dumps(
            {
                "method": "kNN",
                "n_neighbors": self.n_neighbors,
            }
        )
