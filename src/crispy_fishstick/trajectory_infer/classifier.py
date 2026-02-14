"""
Classifier implementation for trajectory inference.
"""
from crispy_fishstick.trajectory_infer.base import (
    BaseTrajectoryInferMethod,
)
from enum import Enum
import logging

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import os


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
