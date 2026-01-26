"""
Embedding-based metrics.
"""
from crispy_fishstick.metrics.base import OutputPathName
from crispy_fishstick.metrics.embeddings.base import EmbeddingMetrics
from crispy_fishstick.shared.constants import RequiredOutputColumns


class AggregateEmbeddingMetrics(EmbeddingMetrics):
    def _setup_model_output_requirements(self):
        # ** NOTE: must define the following attributes **
        # where we define the output embedding name
        # as well as the required features and outputs
        self.output_path_name = OutputPathName.EMBEDDING
        self.required_outputs = [
            RequiredOutputColumns.EMBEDDING,
        ]

    def _embedding_eval(self, output_path):
        return None
