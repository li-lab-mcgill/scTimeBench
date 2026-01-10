"""
config.py

Configuration management for YAML-based configs, similar to the tf-binding project.
Handles both YAML file loading and command-line argument parsing.
"""

import argparse
import os
import yaml

from enum import Enum


class FeatureSpec(Enum):
    """Enum for different feature specifications of models, and required features for metrics."""

    CONTINUOUS = "continuous"
    EMBEDDING = "embedding"
    TRAJECTORY = "trajectory"
    GENE_EXPRESSION = "gene_expression"
    GRN_INFERENCE = "grn_inference"


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

        # Parse known arguments
        args = parser.parse_args()

        # Get all config keys
        config_keys = list(args.__dict__.keys())

        # other keys to add from the yaml file
        config_keys.extend(["model", "dataset"])

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
        }

        for key, value in defaults.items():
            if not hasattr(self, key) or getattr(self, key) is None:
                setattr(self, key, value)

        # Validate required fields
        required_fields = ["dataset", "model", "metrics"]
        for field in required_fields:
            assert (
                hasattr(self, field) and getattr(self, field) is not None
            ), f"Required field '{field}' must be specified in config file or as --{field}"

        dataset_required_fields = ["data_path", "preprocessed_dir", "name"]
        model_required_fields = ["name"]

        for field in dataset_required_fields:
            assert (
                field in self.dataset
            ), f"Required dataset field '{field}' must be specified in config file"

        for field in model_required_fields:
            assert (
                field in self.model
            ), f"Required model field '{field}' must be specified in config file"

        # Validate paths exist
        dataset_path_keys = ["data_path", "cell_lineage_file", "cell_equivalence_file"]
        paths = {
            *{value for key, value in self.dataset.items() if key in dataset_path_keys},
        }

        for path in paths:
            assert os.path.exists(path), f"Path for '{path}' does not exist: {path}"

        print(f"Configuration successfully loaded: {self.__dict__}")
