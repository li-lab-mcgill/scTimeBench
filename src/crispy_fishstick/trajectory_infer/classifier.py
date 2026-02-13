"""
Classifier implementation for trajectory inference.
"""
from crispy_fishstick.trajectory_infer.base import (
    BaseTrajectoryInferMethod,
)
from crispy_fishstick.shared.constants import ObservationColumns
from enum import Enum
import logging

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import os
import numpy as np
import anndata as ad
import scanpy as sc


CLASSIFIER_SAVE_FILE = "classifier_model.pkl"


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

    def _subclass_parameters(self):
        return {
            "classifier": self.method_name.value,
            "n_estimators": self.classifier.n_estimators,
            "max_depth": self.classifier.max_depth,
        }

    def _subclass_train(self, X_train, y_train, traj_infer_path):
        """
        Classification entropy is simply the fitted model's entropy over the predicted
        trajectories.
        """
        if os.path.exists(os.path.join(traj_infer_path, CLASSIFIER_SAVE_FILE)):
            logging.debug("Loading existing classifier model from disk.")
            self.classifier = joblib.load(
                os.path.join(traj_infer_path, CLASSIFIER_SAVE_FILE)
            )
        else:
            self.classifier.fit(X_train, y_train)
            # save the classifier for future use
            joblib.dump(
                self.classifier,
                os.path.join(traj_infer_path, CLASSIFIER_SAVE_FILE),
            )

    def _subclass_predict_probs(self, embeds):
        """
        Perform prediction using the trained classifier and return probabilities.
        """
        assert self.classifier is not None, "Classifier model is not trained."

        # then let's get the logits on the test dataset, and calculate the entropy
        test_probas = self.classifier.predict_proba(embeds)
        logging.debug(f"Predicted probabilities on test set: {test_probas}")

        # turn the index to cell types mapping
        return test_probas, list(self.classifier.classes_)


class CellTypist(BaseTrajectoryInferMethod):
    def __init__(self, traj_config):
        super().__init__(traj_config)
        self.label_key = ObservationColumns.CELL_TYPE.value
        self.classifier = None

    def _subclass_parameters(self):
        return {
            "n_jobs": self.traj_config.get("n_jobs", 10),
            "max_iter": self.traj_config.get("max_iter", 1000),
        }

    def _preprocess(self, data):
        sc.pp.normalize_total(data, target_sum=1e4)
        sc.pp.log1p(data)

    def _subclass_train(self, X_train, y_train, traj_infer_path):
        """
        Train a CellTypist model and cache it to disk.
        """
        model_path = os.path.join(traj_infer_path, CLASSIFIER_SAVE_FILE)

        if os.path.exists(model_path):
            logging.debug("Loading existing CellTypist model from disk.")
            self.classifier = joblib.load(model_path)
            return

        import celltypist

        train_adata = ad.AnnData(X=X_train)
        train_adata.obs[self.label_key] = np.array(y_train)

        # turn off numba
        logging.getLogger("numba").setLevel(logging.WARNING)
        try:
            self.classifier = celltypist.train(
                train_adata,
                labels=self.label_key,
                n_jobs=self._parameters()["n_jobs"],
                max_iter=self._parameters()["max_iter"],
            )
        except ValueError as e:
            logging.error(f"Error during CellTypist training: {e}")
            # this likely happens because of preprocessing issue, so let's try training with preprocessing as a fallback
            self._preprocess(train_adata)
            self.classifier = celltypist.train(
                train_adata,
                labels=self.label_key,
                n_jobs=self._parameters()["n_jobs"],
                max_iter=self._parameters()["max_iter"],
            )

        joblib.dump(self.classifier, model_path)
        logging.debug(f"Saved CellTypist model to {model_path}")

    def _subclass_predict_probs(self, embeds):
        """
        Predict probabilities using a trained CellTypist model.
        """
        assert self.classifier is not None, "CellTypist model is not trained."

        import celltypist

        # turn off numba
        logging.getLogger("numba").setLevel(logging.WARNING)

        pred_adata = ad.AnnData(X=embeds)
        self._preprocess(pred_adata)
        predictions = celltypist.annotate(
            pred_adata,
            model=self.classifier,
            majority_voting=True,
        )

        probs = predictions.probability_matrix.to_numpy()
        labels = predictions.probability_matrix.columns.tolist()
        logging.debug(f"Probability matrix: {predictions.probability_matrix}")
        logging.debug(f"Probability matrix labels: {labels}")
        return probs, labels
