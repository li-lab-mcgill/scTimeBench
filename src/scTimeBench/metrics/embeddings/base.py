"""
Embedding-based metrics.
"""
from scTimeBench.metrics.base import BaseMetric
from scTimeBench.shared.dataset.registry import (
    SuoDataset,
    GarciaAlonsoDataset,
    MaDataset,
)

import os


class EmbeddingMetrics(BaseMetric):
    def _setup_supported_datasets(self):
        # ** NOTE: must define the following two attributes, though each subclass **
        # ** Must also define required_feature_specs and output_path_name individually, as they likely require **
        # ** different output files. **
        self.supported_datasets = [
            SuoDataset.__name__,
            GarciaAlonsoDataset.__name__,
            MaDataset.__name__,
        ]

        self.default_dataset_group = "embeddings"

        # get the path to the shared default datasets config
        self.default_datasets_path = os.path.join(
            os.path.dirname(__file__), "..", "shared", "default_datasets.yaml"
        )

    def _defaults(self):
        """The default parameters for ontology-based metrics."""
        return {}

    def _prep_kwargs_for_submetric_eval(self, output_path, dataset, model):
        return {
            "output_path": output_path,
            "model": model,
            "dataset": dataset,
        }

    def _submetric_eval(self, output_path, model, dataset):
        """
        Wrapper function to call the graph similarity evaluation, and handle database
        logging.
        """
        self.db_manager.insert_eval(
            model,
            self.__class__.__name__,
            self._get_param_encoding(),
            self._embedding_eval(output_path, dataset),
        )

    def _embedding_eval(self, output_path, dataset):
        raise NotImplementedError("Subclasses should implement this method.")
