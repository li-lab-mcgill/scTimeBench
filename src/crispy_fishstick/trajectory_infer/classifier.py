"""
Classifier implementation for trajectory inference.
"""
from crispy_fishstick.trajectory_infer.base import BaseTrajectoryInferMethod
from crispy_fishstick.shared.constants import ObservationColumns, RequiredOutputColumns
from enum import Enum
import numpy as np
import logging

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split


class ClassifierTypes(Enum):
    RANDOM_FOREST = "random_forest"
    BOOSTING = "boosting"
    # Future classifier types can be added here


# TODO: build a unit test for this class, to ensure that we're doing this properly
class Classifier(BaseTrajectoryInferMethod):
    def __init__(self, traj_config):
        super().__init__(traj_config)
        # sets the default number of neighbors
        self.method_name = ClassifierTypes(
            traj_config.get("classifier", ClassifierTypes.RANDOM_FOREST.value)
        )
        if self.method_name == ClassifierTypes.RANDOM_FOREST:
            self.classifier = RandomForestClassifier(
                n_estimators=traj_config.get("n_estimators", 100),
                max_depth=traj_config.get("max_depth", None),
                random_state=traj_config.get("random_state", 42),
            )
        elif self.method_name == ClassifierTypes.BOOSTING:
            self.classifier = GradientBoostingClassifier(
                n_estimators=traj_config.get("n_estimators", 100),
                max_depth=traj_config.get("max_depth", 3),
                random_state=traj_config.get("random_state", 42),
            )
        else:
            raise ValueError(f"Unsupported classifier type: {self.method_name}")

    def _parameters(self):
        return {
            "classifier": self.method_name.value,
            "n_estimators": self.classifier.n_estimators,
            "max_depth": self.classifier.max_depth,
            "test_size": self.traj_config.get("test_size", 0.2),
            "random_state": self.classifier.random_state,
        }

    def _method_infer_trajectory(self, ann_data):
        """
        Infer the trajectory using kNN graph-based method.

        1. We can accomplish this by first separating each embedding based on time.
        2. Then, for each time point, we find the k nearest neighbors in the next time point's
        embedding space.
        3. Finally, we consolidate the cell types per time point based on the kNN results.
        """
        logging.debug(
            f"Inferring trajectory using Classifier: {self.method_name.value}"
        )

        # get the embeddings and timepoints
        embeddings = ann_data.obsm[RequiredOutputColumns.EMBEDDING.value]
        next_timepoint_embeddings = ann_data.obsm[
            RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING.value
        ]
        timepoints = ann_data.obs[ObservationColumns.TIMEPOINT.value]
        cell_types = ann_data.obs[ObservationColumns.CELL_TYPE.value]
        unique_timepoints = sorted(np.unique(timepoints))

        # first we simply build a classifier based on all the timepoints,
        # then we build the lineage mapping later
        # to do this, we first fit the classifier on a random 80% of the data, then test on the remaining 20%
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings,
            cell_types,
            test_size=self.traj_config.get("test_size", 0.2),
            random_state=self.traj_config.get("random_state", 42),
        )
        self.classifier.fit(X_train, y_train)

        # now let's log out the classifier's accuracy and other metrics:
        accuracy = self.classifier.score(X_test, y_test)
        logging.debug(f"Classifier test accuracy: {accuracy}")

        # now we build the lineage mapping
        cell_lineage = {}
        for i in range(len(unique_timepoints) - 1):
            # get indices for the current timepoint
            idx_current = np.where(timepoints == unique_timepoints[i])[0]

            # get embeddings for current and next timepoints
            emb_next = next_timepoint_embeddings[idx_current]

            # use the classifier to classify each cell in the next timepoint
            predicted_next = self.classifier.predict(emb_next)

            # now that we have the predicted next, we build the lineage mapping
            for cur_cell, next_cell in zip(
                cell_types.iloc[idx_current], predicted_next
            ):
                if cur_cell not in cell_lineage:
                    cell_lineage[cur_cell] = {}
                if next_cell not in cell_lineage[cur_cell]:
                    cell_lineage[cur_cell][next_cell] = 0
                cell_lineage[cur_cell][next_cell] += 1

        logging.debug(f"Constructed cell lineage (raw counts): {cell_lineage}")

        # then we should normalize the counts to get probabilities
        for source_cell_type in cell_lineage.keys():
            total_counts = sum(cell_lineage[source_cell_type].values())
            for target_cell_type in cell_lineage[source_cell_type]:
                cell_lineage[source_cell_type][target_cell_type] /= total_counts

        return cell_lineage
