"""
Meta-Based Metrics.

This is a metric that operates by running multiple submetrics.

It takes in as its parameters a set of submetrics to run, and returns this
to the metric to figure out what to do with that information.

This is useful for things such as calculating the error bar of a metric,
by running it multiple times with different seeds, or also for calculating
the perturbation's baselines as well.
"""
from scTimeBench.metrics.base import BaseMetric, create_submetric_instance
from scTimeBench.shared.constants import RequiredOutputFiles
from scTimeBench.shared.dataset.registry import GarciaAlonsoDataset

import logging
import os


class MetaMetric(BaseMetric):
    def _setup_supported_datasets(self):
        # ** NOTE: must define the following two attributes, though each subclass **
        # ** Must also define required_feature_specs and output_path_name individually, as they likely require **
        # ** different output files. **
        self.supported_datasets = [
            GarciaAlonsoDataset.__name__,
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
        """The default parameters for meta-based metrics."""
        return {}

    def _setup_method_output_requirements(self):
        """Skip this, as it's a higher level class."""
        # because we're running submetrics, we just ignore this
        self.required_outputs = [RequiredOutputFiles.META_FLAG]

    def _prep_kwargs_for_submetric_eval(self, output_path, dataset, method):
        return {"output_path": output_path, "dataset": dataset, "method": method}

    def _run_submetric(self, submetric_config):
        """
        Run a submetric and return the result from eval.
        """
        # here, we'll change config to force a rerun because
        # the submetric will likely require different outputs and require
        # redoing the metric entirely
        self.config.force_rerun = True
        submetric_instance = create_submetric_instance(
            self.config, self.db_manager, submetric_config
        )
        logging.debug(
            f"Running submetric {submetric_config['name']} with config: {submetric_config}"
        )
        return submetric_instance.eval()
