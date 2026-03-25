"""
config.py

Configuration management for YAML-based configs, similar to the tf-binding project.
Handles both YAML file loading and command-line argument parsing.
"""

import argparse
import logging
import os
import yaml

from enum import Enum


# enum for the different run types, primarily:
# 1) auto_train_test: automatically run training and testing for methods that support it,
# by running the training and testing script specified.
# 2) preprocess: we preprocess the data and then save out a yaml file specifying requirements.
# The user handles training and testing outside of this framework.
# 3) eval_only: we only evaluate the metric on already generated data based on step 2).
class RunType(Enum):
    AUTO_TRAIN_TEST = "auto_train_test"
    PREPROCESS = "preprocess"
    EVAL_ONLY = "eval_only"
    TRAIN_ONLY = "train_only"


class CsvExportType(Enum):
    GRAPH_SIM = "graph_sim"
    EMBEDDING = "embedding"
    GEX_PRED = "gex_pred"


class CsvWriteMode(Enum):
    MERGE = "merge"
    SEPARATE = "separate"


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
            help="Show available methods, datasets, and metrics",
        )

        parser.add_argument(
            "--print_all",
            action="store_true",
            help="Print all entries in the database tables",
        )

        parser.add_argument(
            "--to_csv",
            nargs="*",
            choices=[csv_type.value for csv_type in CsvExportType],
            help="Export results to CSV. Use '--to_csv' with no values to export all csvs, or provide any number of values explicitly.",
        )

        parser.add_argument(
            "--output_csv_path",
            type=str,
            default="csv_results",
            help="Directory where CSV outputs are saved (default: csv_results)",
        )

        parser.add_argument(
            "--csv_write_mode",
            type=str,
            choices=[mode.value for mode in CsvWriteMode],
            default=CsvWriteMode.SEPARATE.value,
            help="CSV write mode: 'merge' appends into existing shared files, 'separate' (default) writes using the name of your db file as a stem.",
        )

        parser.add_argument(
            "--clear_tables",
            action="store_true",
            help="Clear all entries in the database tables",
        )

        parser.add_argument(
            "--view_evals_by_method",
            action="store_true",
            help="View existing evaluations of all metrics in the database per method set in the configuration",
        )

        parser.add_argument(
            "--view_evals_by_metric",
            action="store_true",
            help="View existing evaluations of all methods in the database per metric set in the configuration",
        )

        parser.add_argument(
            "--database_path",
            type=str,
            help="Path to the SQLite database file for storing results",
        )

        parser.add_argument(
            "--run_type",
            type=str,
            choices=[rt.value for rt in RunType],
            help="Type of run to perform: (default) auto_train_test, preprocess, eval_only, train_only. Defaults to auto_train_test.",
        )

        parser.add_argument(
            "--output_dir",
            type=str,
            help="Directory to store outputs",
        )

        parser.add_argument(
            "--log_level",
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level for the run (default: INFO)",
        )

        parser.add_argument(
            "--log_file",
            type=str,
            help="Optional path to a log file; if omitted logs only go to stdout",
        )

        parser.add_argument(
            "--data_dir",
            type=str,
            help="Optional base directory for dataset files, otherwise uses root paths specified in the config. If used, treats paths in config as either absolute or relative to this directory.",
        )

        parser.add_argument(
            "--force_rerun",
            action="store_true",
            help="Usually duplicate method evaluations are skipped. This flag forces re-running even if evaluations already exist.",
        )

        parser.add_argument(
            "-cf",
            "--crispy-fishstick",
            action="store_true",
            help=argparse.SUPPRESS,  # hide this from help since it's an Easter egg
        )

        # Parse known arguments
        args = parser.parse_args()

        # If --to_csv is provided without values, export all supported CSV outputs.
        if args.to_csv is not None and len(args.to_csv) == 0:
            args.to_csv = [csv_type.value for csv_type in CsvExportType]

        # first handle the Easter egg
        if args.crispy_fishstick:
            from scTimeBench.shared.utils import animate, restore_interrupts
            import sys

            try:
                animate()
            finally:
                restore_interrupts()
                sys.stdout.write("\033[?25h")  # Re-show the cursor if you hid it
            exit()

        # Get all config keys
        config_keys = list(args.__dict__.keys())

        # other keys to add from the yaml file
        config_keys.extend(["method", "datasets"])

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
            "database_path": "scTimeBench.db",
            "run_type": RunType.AUTO_TRAIN_TEST.value,
            "output_dir": "outputs/",
            "datasets": [],
            "log_level": "INFO",
            "log_file": None,
        }

        for key, value in defaults.items():
            if not hasattr(self, key) or getattr(self, key) is None:
                setattr(self, key, value)

        # Normalize csv export selections to enum values for consistent downstream use.
        if not hasattr(self, "to_csv"):
            self.to_csv = None
        elif self.to_csv is not None:
            self.to_csv = [
                item if isinstance(item, CsvExportType) else CsvExportType(item)
                for item in self.to_csv
            ]

        if hasattr(self, "csv_write_mode") and self.csv_write_mode is not None:
            if not isinstance(self.csv_write_mode, CsvWriteMode):
                self.csv_write_mode = CsvWriteMode(self.csv_write_mode)

        # Configure logging with stdout always enabled and optional file output.
        resolved_log_level = getattr(logging, str(self.log_level).upper(), logging.INFO)
        handlers = [logging.StreamHandler()]
        if self.log_file:
            log_dir = os.path.dirname(self.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            handlers.append(logging.FileHandler(self.log_file))

        logging.basicConfig(
            level=resolved_log_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=handlers,
        )

        # Validate required fields
        required_fields = ["method", "metrics"]
        for field in required_fields:
            assert (
                hasattr(self, field) and getattr(self, field) is not None
            ), f"Required field '{field}' must be specified in config file or as --{field}"

        # make sure no other fields exist besides this + datasets in the yaml
        allowed_fields = set(required_fields + ["datasets", "metrics_skiplist"])
        for field in data.keys():
            if field not in allowed_fields:
                raise ValueError(
                    f"Unknown field '{field}' found in config. Allowed fields are {allowed_fields}."
                )

        # validate the fields within each larger section
        # N.B.: we don't need them to specify preprocessors because it might already be preprocessed
        # ** DATASETS **
        dataset_required_fields = ["data_path", "name"]
        dataset_optional_fields = ["data_preprocessing_steps"]
        dataset_alternate_field = "tag"

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
                ), f"Required dataset field '{field}' must be specified in config file, or use alternate flag \"tag\" instead."

            # also ensure that all the fields found in the dataset are only of the required fields
            # this is to ensure caching also works properly
            for field in dataset.keys():
                if (
                    field not in dataset_required_fields
                    and field not in dataset_optional_fields
                ):
                    raise ValueError(
                        f"Unknown field '{field}' found in dataset config. Allowed fields are {dataset_required_fields} or '{dataset_alternate_field}'."
                    )

        # ** METHOD **
        method_required_fields = ["name"]

        for field in method_required_fields:
            assert (
                field in self.method
            ), f"Required method field '{field}' must be specified in config file"

        # Validate paths exist
        dataset_path_keys = [
            "data_path",
            "cell_lineage_file",
            "cell_equivalence_file",
        ]
        method_path_keys = [
            "train_and_test_script",
        ]
        paths = {
            *{
                value
                for dataset in self.datasets
                for key, value in dataset.items()
                if key in dataset_path_keys
            },
            *{value for key, value in self.method.items() if key in method_path_keys},
        }

        for path in paths:
            assert os.path.exists(path), f"Path for '{path}' does not exist: {path}"

        # set the run type
        self.run_type = RunType(self.run_type)

        # verify that the train and test script is specified if auto_train_test is set
        if self.run_type == RunType.AUTO_TRAIN_TEST:
            assert (
                "train_and_test_script" in self.method
            ), "Method must specify 'train_and_test_script' to use --auto_train_test"

        # finally, to be used later for saving the original config
        # into the method output yaml config:
        self.method_yaml_data = data["method"]

        # then let's also add in the metric skip list
        self.metrics_skiplist = data.get("metrics_skiplist", [])
        self.metrics_skiplist = [
            metric if isinstance(metric, str) else metric.get("name", "")
            for metric in self.metrics_skiplist
        ]

        logging.info("Configuration successfully loaded")
        logging.debug("Configuration details: %s", self.__dict__)
