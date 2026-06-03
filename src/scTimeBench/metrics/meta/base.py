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

import logging
import json
import os


class MetaMetric(BaseMetric):
    def _setup_supported_datasets(self):
        # ** NOTE: must define the following two attributes, though each subclass **
        # ** Must also define required_feature_specs and output_path_name individually, as they likely require **
        # ** different output files. **
        self.supported_datasets = self.metric_config.get("supported_datasets", [])
        self.default_dataset_group = self.metric_config.get(
            "default_dataset_group", None
        )

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
        return {"method": method}

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
        return submetric_instance.eval()


class MetaPerturbation(MetaMetric):
    def _submetric_eval(self, method):
        # for each submetric, we need to run it and get the result
        submetric_results = {}

        assert (
            "cell_type" in self.metric_config
        ), "Cell type must be specified in the config for MetaPerturbation metrics."
        cell_type = self.metric_config["cell_type"]

        aliases = ["production", "elimination", "control"]

        for submetric_config in self.metric_config.get("submetrics", []):
            # check to make sure this is a valid submetric, i.e. of the form
            # PerturbationCellTypeProportion
            submetric_alias = submetric_config.get("alias", None)
            assert (
                submetric_alias is not None
            ), "Submetric must have an alias defined in the config."
            assert (
                submetric_alias in aliases
            ), f"Submetric alias must be one of {aliases}."

            result = self._run_submetric(submetric_config)
            logging.debug(f"Result of submetric {submetric_alias}: {result}")
            submetric_results[submetric_alias] = result[cell_type]

        # now we have the submetric results, we can calculate the final result based on the submetric results
        eval = {
            "increase_from_baseline": submetric_results["production"]
            - submetric_results["control"],
            "decrease_from_baseline": submetric_results["elimination"]
            - submetric_results["control"],
            "production_elimination_delta": submetric_results["production"]
            - submetric_results["elimination"],
            "results": submetric_results,
        }

        logging.debug(f"Results of evaluation: {eval}")

        eval_json = json.dumps(eval, sort_keys=True)
        self.db_manager.insert_eval(
            method, self.__class__.__name__, self._get_param_encoding(), eval_json
        )
