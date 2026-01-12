"""
Model Base Class.
"""
from typing import final
from enum import Enum
import yaml
import json
import hashlib
import subprocess
from shared.dataset.base import BaseDataset


class FeatureSpec(Enum):
    """Enum for different feature specifications of models, and required features for metrics."""

    CONTINUOUS = "continuous"
    EMBEDDING = "embedding"
    TRAJECTORY = "trajectory"
    GENE_EXPRESSION = "gene_expression"
    GRN_INFERENCE = "grn_inference"


class BaseModel:
    def __init__(self, config, dataset: BaseDataset):
        self.config = config
        self._check_feature_specs()

        # the model should be parametrized by a dataset
        assert isinstance(
            dataset, BaseDataset
        ), "Model must be initialized with a BaseDataset instance"
        self.dataset = dataset

    @final
    def _check_feature_specs(self):
        """
        Populate the feature specifications required for the metric.
        """
        self.required_feature_specs = None

        # let's use the defined features.yaml to get the features for this model
        with open(self.config.model_features_path, "r") as f:
            features_config = yaml.safe_load(f)

        model_name = self.config.model["name"]

        for model in features_config:
            if model["name"] == model_name:
                self.required_feature_specs = [
                    FeatureSpec(feature) for feature in model["features"]
                ]
                return

        raise ValueError(f"Model features not defined for model: {model_name}")

    def train_and_test(self, yaml_config_path):
        """
        Runs the train and test script provided in the config.
        """
        # start a subprocess to run the script and wait for it to finish
        script_path = self.config.model["train_and_test_script"]
        subprocess.run(["bash", script_path, yaml_config_path], check=True)

    def _get_name(self) -> str:
        """
        Get the name of the model from the configuration.
        """
        return self.config.model["name"]

    def _encode_metadata(self) -> str:
        """
        Generate a string representation of the model metadata.

        This can be used to cache model outputs.
        """
        return json.dumps(self.config.model.get("metadata", {}), sort_keys=True)

    def _encode_output_path(self) -> str:
        """
        Encode the output path based on:
        1) the dataset config
        2) the dataset filters applied
        3) the output file name required by the metric
        and return the full output path as a hashed string.
        """
        filters = self.dataset.encode_filters()
        unique_string = json.dumps(
            {
                "name": self._get_name(),
                "metadata": self._encode_metadata(),
                "dataset_config": self.config.dataset,
                "filters": filters,
            },
            sort_keys=True,
        )
        # Generate a base64 encoded string of the unique string
        return hashlib.sha256(unique_string.encode()).hexdigest()
