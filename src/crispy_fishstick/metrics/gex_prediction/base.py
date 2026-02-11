"""
Gene expression prediction metrics.
"""
from crispy_fishstick.metrics.base import BaseMetric, OutputPathName
from crispy_fishstick.shared.constants import RequiredOutputColumns
from crispy_fishstick.shared.dataset.registry import SuoDataset, GarciaAlonsoDataset

import json
import os


class GexPredictionMetrics(BaseMetric):
    def _setup_supported_datasets(self):
        # ** NOTE: must define the following two attributes, though each subclass **
        self.supported_datasets = [
            SuoDataset.__name__,
            GarciaAlonsoDataset.__name__,
        ]

        # get the path to the default datasets, under ./default_datasets.yaml
        self.default_datasets_path = os.path.join(
            os.path.dirname(__file__), "default_datasets.yaml"
        )

    def _defaults(self):
        """The default parameters for gene expression prediction metrics."""
        return {}

    def _setup_model_output_requirements(self):
        # ** NOTE: must define the following attributes **
        self.output_path_name = OutputPathName.GEX_PREDICTION
        self.required_outputs = [
            RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION,
        ]

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
