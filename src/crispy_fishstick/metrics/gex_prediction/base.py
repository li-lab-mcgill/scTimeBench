"""
Gene expression prediction metrics.
"""
from crispy_fishstick.metrics.base import BaseMetric
from crispy_fishstick.shared.dataset.registry import (
    MaDataset,
    OlaniruDataset,
    MaOlaniruDataset,
    ZebrafishDataset,
    DrosophilaDataset,
    MEFDataset,
)


import json
import os


class GexPredictionMetrics(BaseMetric):
    def _setup_supported_datasets(self):
        # ** NOTE: must define the following two attributes, though each subclass **
        self.supported_datasets = [
            MaDataset.__name__,
            OlaniruDataset.__name__,
            MaOlaniruDataset.__name__,
            ZebrafishDataset.__name__,
            DrosophilaDataset.__name__,
            MEFDataset.__name__,
        ]

        self.default_dataset_group = "gex_prediction"

        # get the path to the shared default datasets config
        self.default_datasets_path = os.path.join(
            os.path.dirname(__file__), "..", "shared", "default_datasets.yaml"
        )

    def _defaults(self):
        """The default parameters for gene expression prediction metrics."""
        return {}

    def _setup_model_output_requirements(self):
        """Skip this, as it's a higher level class."""
        self.required_outputs = (
            f"See requirements of submetrics: {self.__class__.submetrics}"
        )

    def _prep_kwargs_for_submetric_eval(self, output_path, dataset, model):
        return {
            "output_path": output_path,
            "dataset": dataset,
            "model": model,
        }

    def _submetric_eval(self, output_path, dataset, model):
        """
        Wrapper function to call the gene-expression metric evaluation, and handle database
        logging.
        """
        result = self._gex_eval(output_path, dataset)
        aggregate = result
        if isinstance(result, dict):
            aggregate = result.get("All")

        self.db_manager.insert_eval(
            model,
            self.__class__.__name__,
            self._get_param_encoding(),
            aggregate,
        )

        if isinstance(result, dict):
            for tp, score in result.items():
                if tp == "All":
                    continue
                tp_params = dict(self.params)
                tp_params["timepoint"] = str(tp)
                tp_params_json = json.dumps(tp_params)
                if not self.db_manager.has_metric(
                    self.__class__.__name__, tp_params_json
                ):
                    self.db_manager.insert_metric(
                        self.__class__.__name__, tp_params_json
                    )
                if not self.db_manager.has_eval(
                    model, self.__class__.__name__, tp_params_json
                ):
                    self.db_manager.insert_eval(
                        model,
                        self.__class__.__name__,
                        tp_params_json,
                        float(score),
                    )

    def _gex_eval(self, output_path, dataset):
        raise NotImplementedError("Subclasses should implement this method.")
