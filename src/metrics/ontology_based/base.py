"""
Ontology-Based Metrics.
"""
from metrics.base import BaseMetric, FeatureSpec
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

    def _setup_required_feature_specs(self):
        # ** NOTE: must define the following attribute **
        self.required_feature_specs = [FeatureSpec.TRAJECTORY]
