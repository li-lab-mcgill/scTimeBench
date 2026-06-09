"""
Perturbation-Based Metrics.
"""
from scTimeBench.metrics.base import BaseMetric
from scTimeBench.shared.constants import (
    RequiredOutputFiles,
    PERTURBATION_SET_CONFIG_FILENAME,
)
from scTimeBench.shared.dataset.registry import (
    SuoDataset,
    GarciaAlonsoDataset,
    MaDataset,
)
from scTimeBench.shared.perturbation_set import PerturbationSet

import os
import yaml


class PerturbationBasedMetrics(BaseMetric):
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
        """The default parameters for perturbation-based metrics."""
        return {}

    def _setup_method_output_requirements(self):
        """Skip this, as it's a higher level class."""
        # build the required outputs based on the perturbation set defined
        self.perturbation_set = PerturbationSet(
            self.metric_config.get("perturbation_set_config", {})
        )
        self.required_outputs = [RequiredOutputFiles.PERTURBED_TEST_ANN_DATA]

    def _get_relative_output_path(self):
        """
        Returns the relative output path, which is based on the perturbation set hash.
        """
        assert (
            self.perturbation_set is not None
        ), "Perturbation set must be defined before getting the output path."
        return self.perturbation_set.perturbation_path()

    def _submetric_eval_setup(self, eval_output_path):
        """
        Optional setup steps that give the output path to the submetric evaluation function,
        in case there are some steps needed for training/evaluation specifically.

        Here, we will write out the perturbation set config to the output path.
        So this can be reconstructed during submetric evaluation.
        """
        # Write out the perturbation set config
        perturbation_config_path = os.path.join(
            eval_output_path, PERTURBATION_SET_CONFIG_FILENAME
        )
        with open(perturbation_config_path, "w") as f:
            yaml.safe_dump(self.metric_config.get("perturbation_set_config", {}), f)
