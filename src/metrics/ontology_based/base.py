"""
Ontology-Based Metrics.
"""
from metrics.base import BaseMetric
from shared.dataset.registry import SuoDataset, GarciaAlonsoDataset

import os


class OntologyBasedMetrics(BaseMetric):
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

    def _setup_model_output_requirements(self):
        """Skip this, as it's a higher level class."""
        self.required_outputs = (
            f"See requirements of submetrics: {self.__class__.submetrics}"
        )
