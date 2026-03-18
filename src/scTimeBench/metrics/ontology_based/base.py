"""
Ontology-Based Metrics.
"""
from scTimeBench.metrics.base import BaseMetric
from scTimeBench.shared.dataset.registry import (
    SuoDataset,
    GarciaAlonsoDataset,
    MaDataset,
)

import os


class OntologyBasedMetrics(BaseMetric):
    def _setup_supported_datasets(self):
        # ** NOTE: must define the following two attributes, though each subclass **
        # ** Must also define required_feature_specs and output_path_name individually, as they likely require **
        # ** different output files. **
        self.supported_datasets = [
            SuoDataset.__name__,
            GarciaAlonsoDataset.__name__,
            MaDataset.__name__,
        ]

        self.default_dataset_group = "ontology_based"

        # get the path to the shared default datasets config
        self.default_datasets_path = os.path.join(
            os.path.dirname(__file__), "..", "shared", "default_datasets.yaml"
        )

        self.optional_datasets_path = os.path.join(
            os.path.dirname(__file__), "..", "shared", "optional_datasets.yaml"
        )

    def _defaults(self):
        """The default parameters for ontology-based metrics."""
        return {}

    # TODO: build the proper hierarchy for this all? or maybe just force graph sim only? not too sure...
    def _setup_method_output_requirements(self):
        """Skip this, as it's a higher level class."""
        self.required_outputs = (
            f"See requirements of submetrics: {self.__class__.submetrics}"
        )
