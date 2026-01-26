"""
Embedding-based metrics.
"""
from crispy_fishstick.metrics.base import OutputPathName
from crispy_fishstick.metrics.embeddings.base import EmbeddingMetrics
from crispy_fishstick.shared.constants import RequiredOutputColumns
from crispy_fishstick.trajectory_infer.base import (
    TrajectoryInferenceMethodFactory,
    BaseTrajectoryInferMethod,
)
from crispy_fishstick.trajectory_infer.classifier import Classifier

import logging
import os
import json
import numpy as np


class TrajectoryEmbeddingMetrics(EmbeddingMetrics):
    def _setup_model_output_requirements(self):
        # ** NOTE: must define the following attributes **
        # where we define the output embedding name
        # as well as the required features and outputs
        self.output_path_name = OutputPathName.GRAPH_SIM
        self.required_outputs = [
            RequiredOutputColumns.EMBEDDING,
            RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING,
        ]


# now let's create the other metrics such as the gini index or classification entropy, etc.
class ClassificationEntropy(TrajectoryEmbeddingMetrics):
    def _setup_trajectory_inference_model(self):
        # by default we use the classifier trajectory inference model
        logging.debug(
            "Setting up trajectory inference model for classification entropy."
        )

        self.trajectory_infer_model: BaseTrajectoryInferMethod = (
            TrajectoryInferenceMethodFactory().get_trajectory_infer_method(
                self.metric_config.get(
                    "trajectory_infer_model", {"name": Classifier.__name__}
                )
            )
        )
        self.params["trajectory_infer_model"] = str(self.trajectory_infer_model)

    def _embedding_eval(self, output_path):
        model_output_file = os.path.join(output_path, self.output_path_name.value)
        probas, accuracy = self.trajectory_infer_model.train_and_predict(
            model_output_file
        )

        entropy = -np.sum(probas * np.log(probas + 1e-10), axis=1)  # avoid log(0)
        logging.debug(f"Average classification entropy: {np.mean(entropy)}")

        num_classes = probas.shape[1]
        normalized_entropy = entropy / np.log(num_classes)

        return json.dumps(
            {
                "avg_entropy": np.mean(entropy).item(),
                "std_entropy": np.std(entropy).item(),
                "avg_normalized_entropy": np.mean(normalized_entropy).item(),
                "std_normalized_entropy": np.std(normalized_entropy).item(),
                "num_classes": num_classes,
                "classifier_accuracy": accuracy,
            }
        )
