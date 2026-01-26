"""
Embedding-based metrics.
"""
from crispy_fishstick.metrics.base import OutputPathName
from crispy_fishstick.metrics.embeddings.base import EmbeddingMetrics
from crispy_fishstick.shared.constants import RequiredOutputColumns
from crispy_fishstick.trajectory_infer.base import TrajectoryInferenceMethodFactory
from crispy_fishstick.trajectory_infer.classifier import Classifier


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
        self.trajectory_infer_model = (
            TrajectoryInferenceMethodFactory().get_trajectory_infer_method(
                self.metric_config.get(
                    "trajectory_infer_model", {"name": Classifier.__name__}
                )
            )
        )
        self.params["trajectory_infer_model"] = str(self.trajectory_infer_model)

    def _embedding_eval(self, output_path):
        return self.trajectory_infer_model.evaluate_classification_entropy(output_path)
