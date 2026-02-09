"""
Embedding-based metrics.
"""
from crispy_fishstick.metrics.embeddings.base import EmbeddingMetrics
from crispy_fishstick.shared.constants import RequiredOutputFiles


class AggregateEmbeddingMetrics(EmbeddingMetrics):
    def _setup_model_output_requirements(self):
        # ** NOTE: must define the following attributes **
        # where we define the output embedding name
        # as well as the required features and outputs
        self.required_outputs = [
            RequiredOutputFiles.EMBEDDING,
        ]
