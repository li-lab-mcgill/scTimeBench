"""
Method Base Class.
"""
import os
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

    def _encode_train_output_path(self) -> str:
        """
        Encode the output path based on:
        1) the dataset config
        2) the dataset preprocessors applied
        3) the output file name required by the metric
        and return the full output path as a hashed string.

        This is used to cache the outputs of the method on the training dataset, which can be shared across multiple metrics.
        """
        dataset_to_encode = (
            self.dataset.cached_train_dataset
            if self.dataset.cached_train_dataset is not None
            else self.dataset
        )

        unique_string = json.dumps(
            {
                "name": self._get_name(),
                "metadata": self._encode_metadata(),
                # if the dataset has a cached_train_dataset,
                # then we should only encode the dataset_dict and preprocessors
                # of the cached_train_dataset, since that is what is used for training the model
                "dataset_dict": dataset_to_encode.encode_dataset_dict(),
                "preprocessors": dataset_to_encode.encode_preprocessors(),
            },
            sort_keys=True,
        )
        # Generate a base64 encoded string of the unique string
        return hashlib.sha256(unique_string.encode()).hexdigest()

    def _encode_output_path(self) -> str:
        """
        Just like the train and test datasets will now be split,
        so that the method can have a separate cache for the test dataset outputs,
        we should also have a separate encoding for the test output path per model.
        """
        # if there is no cached_train_dataset, then we will just use the old way
        # of encoding the output path, since the train and test datasets are not really separated
        train_output_path = self._encode_train_output_path()
        if self.dataset.cached_train_dataset is None:
            return train_output_path

        unique_string = json.dumps(
            {
                "name": self._get_name(),
                "metadata": self._encode_metadata(),
                # always encode the proper dataset, as the test is not shared
                "dataset_dict": self.dataset.encode_dataset_dict(),
                "preprocessors": self.dataset.encode_preprocessors(),
            },
            sort_keys=True,
        )

        # Generate a base64 encoded string of the unique string
        test_hash = hashlib.sha256(unique_string.encode()).hexdigest()
        return os.path.join(train_output_path, "test_outputs", test_hash)

    def is_equiv(self, other_method) -> bool:
        """
        Check if two methods are equivalent based on their name and metadata.
        """
        return self._encode_output_path() == other_method._encode_output_path()
