"""
config.py

Configuration management for YAML-based configs, similar to the tf-binding project.
Handles both YAML file loading and command-line argument parsing.
"""

import argparse
import os
import yaml


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

        # Required arguments
        parser.add_argument(
            "--data_path", type=str, help="Path to the AnnData h5ad file"
        )

        parser.add_argument(
            "--cell_lineage_file",
            type=str,
            help="Path to the cell lineage file (=> separated)",
        )

        parser.add_argument(
            "--cell_equivalence_file",
            type=str,
            help="Path to the cell equivalence mapping file",
        )

        # Parse known arguments
        args = parser.parse_args()

        # Get all config keys
        config_keys = list(args.__dict__.keys())

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
        defaults = {}

        for key, value in defaults.items():
            if not hasattr(self, key) or getattr(self, key) is None:
                setattr(self, key, value)

        # Validate required fields
        required_fields = ["data_path"]
        for field in required_fields:
            assert (
                hasattr(self, field) and getattr(self, field) is not None
            ), f"Required field '{field}' must be specified in config file or as --{field}"

        # Validate paths exist
        paths = ["data_path", "cell_lineage_file", "cell_equivalence_file"]

        for path in paths:
            if hasattr(self, path) and getattr(self, path):
                assert os.path.exists(
                    getattr(self, path)
                ), f"Path for '{path}' does not exist: {getattr(self, path)}"
