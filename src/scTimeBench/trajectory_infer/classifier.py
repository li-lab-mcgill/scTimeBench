"""
Classifier implementation for trajectory inference.
"""
from scTimeBench.trajectory_infer.base import (
    BaseTrajectoryInferMethod,
)
from scTimeBench.shared.constants import ObservationColumns
from scTimeBench.shared.utils import is_log_normalized_to_counts
from enum import Enum
import logging

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import os
import numpy as np
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
            "use_SGD": self.traj_config.get("use_SGD", True),
            "mini_batch": self.traj_config.get("mini_batch", True),
            # decides whether or not to take the gex from the original data and renormalize it for CellTypist, or to just use the predicted values as they are
            # by default we turn it off because the model itself should be providing gex that's close to it
            "renormalize": self.traj_config.get("renormalize", False),
        }

    def _preprocess(self, data):
        # detect if the data is already normalized, and if so, check to see if it's
        # CP10K, and if not, then we warn the user that this might not be ideal for CellTypist performance
        if is_log_normalized_to_counts(data):
            logging.debug("Data appears to be already normalized to CP10K.")
            return data

        if data.X.max() <= 20:
            logging.debug(
                "Data appears to be log-transformed but not normalized. We need to normalize in any case"
                " for CellTypist to work. Proceed with caution as this might not be ideal for CellTypist performance."
            )

            logging.debug(
                f"Average summed expression per cell: {np.mean(np.sum(np.expm1(data.X), axis=1))}... Rescaling for CellTypist."
            )
            if not self._parameters()["renormalize"]:
                logging.debug(
                    "Renormalization is turned off, so we will not rescale the data for CellTypist. This might lead to suboptimal performance, so proceed with caution."
                )
                return data

            # first we clip any negative values to 0, as CellTypist doesn't handle negative values well
            data.X = np.clip(data.X, a_min=0, a_max=None)
            # then we rescale the data to have a total count of 1e4 per cell, which is what CellTypist expects
            data.X = np.expm1(data.X)  # undo log1p transformation
            sc.pp.normalize_total(data, target_sum=1e4)
            sc.pp.log1p(data.X)  # log-transform again after normalization
            return data

        # should not get here...
        raise ValueError(
            "Data should either be log-normalized to 10_000 counts or be log-transformed but not normalized (due to predicted values). "
            "Please check the input data and ensure it is properly preprocessed for CellTypist performance. "
            "Hint: You should be training and running with LogNormPreprocessor for your particular dataset."
        )

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

        train_adata = sc.AnnData(X=X_train)
        train_adata.obs[self.label_key] = np.array(y_train)

        # turn off numba
        logging.getLogger("numba").setLevel(logging.WARNING)
        train_adata = self._preprocess(train_adata)
        self.classifier = celltypist.train(
            train_adata,
            labels=self.label_key,
            n_jobs=self._parameters()["n_jobs"],
            max_iter=self._parameters()["max_iter"],
            use_SGD=self._parameters()["use_SGD"],
            mini_batch=self._parameters()["mini_batch"],
            balance_cell_type=True,
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

        # we want to create a new anndata with the same gene names
        pred_adata = sc.AnnData(X=embeds)
        pred_adata = self._preprocess(pred_adata)
        predictions = celltypist.annotate(
            pred_adata,
            model=self.classifier,
            majority_voting=True,  # added because otherwise rare populations are gone
        )

        probs = predictions.probability_matrix.to_numpy()
        labels = predictions.probability_matrix.columns.tolist()
        logging.debug(f"Probability matrix: {predictions.probability_matrix}")
        logging.debug(f"Probability matrix labels: {labels}")
        return probs, labels
