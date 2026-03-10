"""
kNN implementation for trajectory inference.
"""
from scTimeBench.trajectory_infer.base import (
    BaseTrajectoryInferMethod,
)
from sklearn.neighbors import NearestNeighbors
import numpy as np
import logging
from enum import Enum
import os
import joblib

KNN_SAVE_FILE = "knn_model.pkl"
KNN_LABELS_FILE = "knn_labels.npy"


class kNNStrategy(Enum):
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"


# TODO: build a unit test for this class, to ensure that we're doing this properly
class kNN(BaseTrajectoryInferMethod):
    def __init__(self, traj_config):
        super().__init__(traj_config)
        # sets the default number of neighbors
        self.strategy = kNNStrategy(
            traj_config.get("strategy", kNNStrategy.MAJORITY_VOTE.value)
        )

    def _subclass_parameters(self):
        return {
            "n_neighbors": self.traj_config.get("n_neighbors", 5),
            "strategy": self.strategy.value,
            "metric": self.traj_config.get("metric", "minkowski"),
        }

    def _subclass_train(self, X_train, y_train, traj_infer_path):
        """
        kNN is an unsupervised method, so no training is needed.
        However, we can build the kNN graph here for later use.
        """
        save_path = os.path.join(traj_infer_path, KNN_SAVE_FILE)
        labels_path = os.path.join(traj_infer_path, KNN_LABELS_FILE)

        # build kNN model based on embeddings
        knn_model = NearestNeighbors(
            n_neighbors=self._parameters()["n_neighbors"],
            metric=self._parameters()["metric"],
        )

        if os.path.exists(save_path) and os.path.exists(labels_path):
            logging.debug("Loading existing kNN model from disk.")
            self.knn_model = joblib.load(save_path)
            self.knn_labels = np.load(labels_path, allow_pickle=True)
            return

        # save the kNN graph first and then save it and return it
        knn_model.fit(X_train)
        joblib.dump(knn_model, save_path)
        logging.debug(f"Saved kNN model to {save_path}")
        np.save(os.path.join(traj_infer_path, KNN_LABELS_FILE), y_train)

        self.knn_model = knn_model  # save for later use
        self.knn_labels = y_train  # save for later use

    def _predict_proba(self, embeds):
        """
        Returns the probability distribution over cell types for the given embeddings.
        """
        # find the k nearest neighbors for each test embedding
        _, indices = self.knn_model.kneighbors(embeds)

        probas = []
        for index_set in indices:
            # get the cell types of the neighbors
            neighbor_cell_types = self.knn_labels[index_set]
            cell_type_proba = {}

            for i, target_cell_type in enumerate(neighbor_cell_types):
                # here the probability added depends on the strategy
                if self.strategy == kNNStrategy.MAJORITY_VOTE:
                    cell_type_proba[target_cell_type] = (
                        cell_type_proba.get(target_cell_type, 0) + 1
                    )
                elif self.strategy == kNNStrategy.WEIGHTED_AVERAGE:
                    weight = 1 / (i + 1)  # simple inverse rank weighting
                    cell_type_proba[target_cell_type] = (
                        cell_type_proba.get(target_cell_type, 0) + weight
                    )

            # normalize to get probabilities
            total_neighbors = len(neighbor_cell_types)
            for cell_type in cell_type_proba:
                cell_type_proba[cell_type] /= total_neighbors

            probas.append(cell_type_proba)
        return probas

    def _subclass_predict_probs(self, embeds):
        """
        Given the test embeddings, predict their cell types based on kNN.
        """
        test_probas = self._predict_proba(embeds)

        # now let's turn test and next tp probas into a numpy array for easier handling
        labels = list(set(self.knn_labels))
        label_to_index = {label: i for i, label in enumerate(labels)}

        test_probas_array = np.zeros((len(test_probas), len(labels)))
        for i, proba_dict in enumerate(test_probas):
            for label, proba in proba_dict.items():
                test_probas_array[i, label_to_index[label]] = proba

        return test_probas_array, list(label_to_index.keys())

    def get_kNN_graph(self, output_path):
        """
        Function to get the kNN graph used in the trajectory inference.

        This can be useful for visualization or further analysis.
        """
        self.train_and_predict(output_path, train_only=True)
        assert hasattr(
            self, "knn_model"
        ), "kNN model not found, be sure to save it during training."
        return self.knn_model
