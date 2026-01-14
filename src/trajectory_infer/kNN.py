"""
kNN implementation for trajectory inference.
"""
from trajectory_infer.base import BaseTrajectoryInferMethod
from sklearn.neighbors import NearestNeighbors
from shared.constants import ObservationColumns, RequiredOutputColumns
import numpy as np
import json


# TODO: build a unit test for this class, to ensure that we're doing this properly
class kNN(BaseTrajectoryInferMethod):
    def __init__(self, traj_config):
        super().__init__(traj_config)
        # sets the default number of neighbors
        self.n_neighbors = traj_config.get("n_neighbors", 5)

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
        timepoints = ann_data.obs[ObservationColumns.TIMEPOINT.value]
        cell_types = ann_data.obs[ObservationColumns.CELL_TYPE.value]
        unique_timepoints = sorted(np.unique(timepoints))
        print(cell_types.unique())

        cell_lineage = {}
        for i in range(len(unique_timepoints) - 1):
            tp_current = unique_timepoints[i]
            tp_next = unique_timepoints[i + 1]

            # get indices for current and next timepoints
            idx_current = np.where(timepoints == tp_current)[0]
            idx_next = np.where(timepoints == tp_next)[0]

            # get embeddings for current and next timepoints
            emb_current = embeddings[idx_current]
            emb_next = embeddings[idx_next]

            # build kNN model based on next timepoint embeddings
            if emb_next.shape[0] < self.n_neighbors:
                n_neighbors = emb_next.shape[0]
            else:
                n_neighbors = self.n_neighbors

            knn_model = NearestNeighbors(n_neighbors=n_neighbors)
            knn_model.fit(emb_next)

            # find kNN for each cell in current timepoint
            _, indices = knn_model.kneighbors(emb_current)

            # map cell types from current to next timepoint
            for j, idx in enumerate(idx_current):
                source_cell_type = cell_types.iloc[idx]
                target_cell_types = cell_types.iloc[idx_next[indices[j]]]

                # then each source cell type maps to multiple target cell types
                # where we'll be keeping count of each cell type to create a distribution later
                if source_cell_type not in cell_lineage:
                    cell_lineage[source_cell_type] = {}

                for target_cell_type in target_cell_types:
                    if target_cell_type not in cell_lineage[source_cell_type]:
                        cell_lineage[source_cell_type][target_cell_type] = 0
                    cell_lineage[source_cell_type][target_cell_type] += 1

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
