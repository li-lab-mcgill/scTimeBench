"""
Method Base Class.
"""
import json
import hashlib
import subprocess
import logging
from scTimeBench.shared.dataset.base import BaseDataset


class MethodManager:
    def __init__(self, config, dataset: BaseDataset):
        self.config = config

        # the method should be parametrized by a dataset
        assert isinstance(
            dataset, BaseDataset
        ), "Method must be initialized with a BaseDataset instance"
        self.dataset = dataset

    def train_and_test(self, yaml_config_path):
        """
        Runs the train and test script provided in the config.
        """
        # start a subprocess to run the script and wait for it to finish
        script_path = self.config.method["train_and_test_script"]
        process = subprocess.Popen(
            ["bash", script_path, yaml_config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # This loop effectively "waits" for the process output to finish
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                clean_line = line.strip()
                if clean_line:
                    logging.debug(clean_line)

        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                f"Train and test script failed with return code {return_code}"
            )

    def _get_name(self) -> str:
        """
        Get the name of the method from the configuration.
        """
        return self.config.method["name"]

    def _encode_metadata(self) -> str:
        """
        Generate a string representation of the method metadata.

        This can be used to cache method outputs.
        """
        return json.dumps(self.config.method.get("metadata", {}), sort_keys=True)

    def _encode_output_path(self) -> str:
        """
        Encode the output path based on:
        1) the dataset config
        2) the dataset filters applied
        3) the output file name required by the metric
        and return the full output path as a hashed string.
        """
        unique_string = json.dumps(
            {
                "name": self._get_name(),
                "metadata": self._encode_metadata(),
                "dataset_dict": self.dataset.encode_dataset_dict(),
                "filters": self.dataset.encode_filters(),
            },
            sort_keys=True,
        )
        # Generate a base64 encoded string of the unique string
        return hashlib.sha256(unique_string.encode()).hexdigest()
