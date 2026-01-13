"""
Model Base Class.
"""
import json
import hashlib
import subprocess
from shared.dataset.base import BaseDataset


class ModelManager:
    def __init__(self, config, dataset: BaseDataset):
        self.config = config

        # the model should be parametrized by a dataset
        assert isinstance(
            dataset, BaseDataset
        ), "Model must be initialized with a BaseDataset instance"
        self.dataset = dataset

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
                "dataset_dict": self.dataset.dataset_dict,
                "filters": filters,
            },
            sort_keys=True,
        )
        # Generate a base64 encoded string of the unique string
        return hashlib.sha256(unique_string.encode()).hexdigest()
