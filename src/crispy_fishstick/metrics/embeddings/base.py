"""
Embedding-based metrics.
"""
from crispy_fishstick.metrics.base import BaseMetric
from crispy_fishstick.shared.dataset.registry import SuoDataset, GarciaAlonsoDataset
from crispy_fishstick.metrics.base import OutputPathName
from crispy_fishstick.shared.constants import RequiredOutputColumns

import os


class EmbeddingMetrics(BaseMetric):
    def _setup_model_output_requirements(self):
        # ** NOTE: must define the following attributes **
        # where we define the output embedding name
        # as well as the required features and outputs
        self.output_path_name = OutputPathName.GRAPH_SIM
        self.required_outputs = [
            RequiredOutputColumns.EMBEDDING,
        ]

    def _setup_supported_datasets(self):
        # ** NOTE: must define the following two attributes, though each subclass **
        # ** Must also define required_feature_specs and output_path_name individually, as they likely require **
        # ** different output files. **
        self.supported_datasets = [
            SuoDataset.__name__,
            GarciaAlonsoDataset.__name__,
        ]

        # get the path to the default datasets, under ./default_datasets.yaml
        self.default_datasets_path = os.path.join(
            os.path.dirname(__file__), "default_datasets.yaml"
        )

    def _defaults(self):
        """The default parameters for ontology-based metrics."""
        return {}

    def _prep_kwargs_for_submetric_eval(self, output_path, dataset, model):
        return {
            "output_path": output_path,
            "model": model,
        }

    def _submetric_eval(self, output_path, model):
        """
        Wrapper function to call the graph similarity evaluation, and handle database
        logging.
        """
        self.db_manager.insert_eval(
            model,
            self.__class__.__name__,
            self._get_param_encoding(),
            self._embedding_eval(output_path),
        )

    def _embedding_eval(self, output_path):
        raise NotImplementedError("Subclasses should implement this method.")
