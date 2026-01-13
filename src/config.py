"""
config.py

Configuration management for YAML-based configs, similar to the tf-binding project.
Handles both YAML file loading and command-line argument parsing.
"""

import argparse
import os
import yaml

from enum import Enum


# enum for the different run types, primarily:
# 1) auto_train_test: automatically run training and testing for models that support it,
# by running the training and testing script specified.
# 2) preprocess: we preprocess the data and then save out a yaml file specifying requirements.
# The user handles training and testing outside of this framework.
# 3) eval_only: we only evaluate the metric on already generated data based on step 2).
class RunType(Enum):
    AUTO_TRAIN_TEST = "auto_train_test"
    PREPROCESS = "preprocess"
    EVAL_ONLY = "eval_only"


class Config:
    """Config class for both yaml and cli arguments."""

    def __init__(self):
        """
        Initialize config by parsing YAML file and command-line arguments.
        CLI arguments override YAML settings.
        """
        # Initiate parser and parse arguments
        parser = argparse.ArgumentParser(
            description="Single-cell trajectory analysis configuration"
        )

        # Config file argument
        parser.add_argument(
            "-c", "--config", type=str, help="Path to YAML configuration file"
        )

        # add metrics argument
        parser.add_argument(
            "--metrics",
            type=str,
            nargs="+",
            help="List of metrics to compute",
        )

        parser.add_argument(
            "--available",
            action="store_true",
            help="Show available models, datasets, and metrics",
        )

        parser.add_argument(
            "--database_path",
            type=str,
            help="Path to the SQLite database file for storing results",
        )

        parser.add_argument(
            "--model_features_path",
            type=str,
            help="Path to the YAML file defining model features",
        )

        parser.add_argument(
            "--run_type",
            type=str,
            choices=[rt.value for rt in RunType],
            help="Type of run to perform: auto_train_test, preprocess, or eval_only. Defaults to preprocess.",
        )

        parser.add_argument(
            "--output_dir",
            type=str,
            help="Directory to store outputs",
        )

        # Parse known arguments
        args = parser.parse_args()

        # Get all config keys
        config_keys = list(args.__dict__.keys())

        # other keys to add from the yaml file
        config_keys.extend(["model", "datasets"])

        # First read the config file if provided
        assert (
            args.config is not None
        ), "Config file path must be provided with --config"
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")

        with open(args.config, "r") as file:
            data = yaml.safe_load(file)

        # Set attributes from YAML file
        for key in config_keys:
            if key in data.keys():
                setattr(self, key, data[key])

        # Override with command-line arguments
        for key, value in args._get_kwargs():
            if value is not None:
                setattr(self, key, value)

        # Set defaults for optional parameters
        defaults = {
            "database_path": "crispy_fishstick.db",
            "run_type": RunType.PREPROCESS.value,
            "model_features_path": "model_utils/features.yaml",
            "output_dir": "outputs/",
            "datasets": [],
        }

        for key, value in defaults.items():
            if not hasattr(self, key) or getattr(self, key) is None:
                setattr(self, key, value)

        # Validate required fields
        required_fields = ["model", "metrics"]
        for field in required_fields:
            assert (
                hasattr(self, field) and getattr(self, field) is not None
            ), f"Required field '{field}' must be specified in config file or as --{field}"

        dataset_required_fields = ["data_path", "name", "filters"]
        dataset_alternate_field = "tag"
        model_required_fields = ["name"]

        for dataset in self.datasets:
            # we want to make sure either all of the required fields are specified,
            # or the alternate field is specified, but not a mix of both
            if dataset_alternate_field in dataset:
                if any([field in dataset for field in dataset_required_fields]):
                    raise ValueError(
                        f"Dataset config cannot have both '{dataset_alternate_field}' and any other fields {dataset_required_fields}."
                    )
                continue

            for field in dataset_required_fields:
                assert (
                    field in dataset
                ), f"Required dataset field '{field}' must be specified in config file."

            # also ensure that all the fields found in the dataset are only of the required fields
            # this is to ensure caching also works properly
            for field in dataset.keys():
                if field not in dataset_required_fields:
                    raise ValueError(
                        f"Unknown field '{field}' found in dataset config. Allowed fields are {dataset_required_fields} or '{dataset_alternate_field}'."
                    )

        for field in model_required_fields:
            assert (
                field in self.model
            ), f"Required model field '{field}' must be specified in config file"

        # Validate paths exist
        dataset_path_keys = [
            "data_path",
            "cell_lineage_file",
            "cell_equivalence_file",
            "model_features_path",
        ]
        model_path_keys = [
            "train_and_test_script",
        ]
        paths = {
            *{
                value
                for dataset in self.datasets
                for key, value in dataset.items()
                if key in dataset_path_keys
            },
            *{value for key, value in self.model.items() if key in model_path_keys},
        }

        for path in paths:
            assert os.path.exists(path), f"Path for '{path}' does not exist: {path}"

        # set the run type
        self.run_type = RunType(self.run_type)

        # verify that the train and test script is specified if auto_train_test is set
        if self.run_type == RunType.AUTO_TRAIN_TEST:
            assert (
                "train_and_test_script" in self.model
            ), "Model must specify 'train_and_test_script' to use --auto_train_test"

        print(f"Configuration successfully loaded: {self.__dict__}")
