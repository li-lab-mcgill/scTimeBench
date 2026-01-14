"""
Base class for all metrics. They should all implement the eval method, and
depend on the dataset that they belong to.
"""
from config import Config, RunType
from typing import final
from metrics.model_manager import ModelManager
from shared.dataset.base import DATASET_REGISTRY, DATASET_FILTER_REGISTRY
from database import DatabaseManager
from enum import Enum

import os
import pickle
import yaml
import logging


# feature specifications that the metrics can require from models
class FeatureSpec(Enum):
    """Enum for different feature specifications of models, and required features for metrics."""

    CONTINUOUS = "continuous"
    EMBEDDING = "embedding"
    TRAJECTORY = "trajectory"
    GENE_EXPRESSION = "gene_expression"
    GRN_INFERENCE = "grn_inference"


class OutputPathName(Enum):
    EMBEDDING = "embedding.pkl"
    GRAPH_SIM = "graph_sim.h5ad"


METRIC_REGISTRY = {}


def register_metric(cls):
    """Decorator to register a metric class in the METRIC_REGISTRY."""
    METRIC_REGISTRY[cls.__name__] = cls


# also store a registry of metrics of name to class
class BaseMetric:
    def __init__(self, config: Config, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.config = config
        self.MODEL_CONFIG_FILENAME = "model_config.yaml"
        self.PICKLED_DATASET_FILENAME = "dataset.pkl"

    def __init_subclass__(cls):
        """
        Automatically register subclasses in the METRIC_REGISTRY.

        This allows for hierarchical structuring of metrics, where
        subclasses are tracked under their parent classes.
        """
        register_metric(cls)

        # we want to clear the submetrics list for each subclass
        cls.submetrics = []

        for base in cls.__bases__:
            if hasattr(base, "submetrics"):
                base.submetrics.append(cls)

    # ** METRIC EVALUATION SECTION, main function to be called (no others should be called) **
    def _eval(self, output_path):
        """
        Subclasses should implement this method to evaluate the metric.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _preprocess(self, dataset):
        """
        Preprocessing steps required before evaluating the metric.
        """
        # 1) we check that the subclasses have defined the required feature specs
        assert (
            hasattr(self, "output_path_name")
            and type(self.output_path_name) is OutputPathName
        ), "Subclasses must define output_path_name of type OutputPathName"

        self.model = ModelManager(self.config, dataset)

        # ** NOTE **
        # because it doesn't cost much (as it did before with the dataset preprocessing)
        # we'll simply always preprocess the model output directory (here used to be caching)
        # but we still should save it to the database for future reference

        # 2) create the output directory for this model that is parametrized by
        # this dataset. So that if we run the same model on the same dataset
        # with the same filters, we will get the same output directory
        # Each output directory can contain multiple files required by different metrics
        # (e.g., embedding.pkl, graph_sim.pkl, etc.)
        # we will first insert a yaml config file that the model can use to
        # train and test on this dataset
        hash_output_dir = self.model._encode_output_path()
        output_path = os.path.join(self.config.output_dir, hash_output_dir)

        os.makedirs(output_path, exist_ok=True)

        # to make our lives easier, we will also pickle our current dataset object
        # and store this in the output directory as well
        # so that the model can load this dataset object directly for training and testing
        pickled_dataset_path = os.path.join(output_path, self.PICKLED_DATASET_FILENAME)
        with open(pickled_dataset_path, "wb") as f:
            pickle.dump(dataset, f)

        yaml_config = {
            "output_path": output_path,
            "output_file_name": self.output_path_name.value,
            "dataset_pkl_path": pickled_dataset_path,
            "model": self.config.model_yaml_data,
        }

        # write out the yaml config file for the model
        yaml_config_path = os.path.join(output_path, self.MODEL_CONFIG_FILENAME)
        with open(yaml_config_path, "w") as f:
            yaml.safe_dump(yaml_config, f)

        # now let's save this hash output dir to the database as well, only if it doesn't exist
        if self.db_manager.get_model_output_path(self.model) is None:
            self.db_manager.insert_model_output(self.model, output_path)

        return output_path

    @final
    def eval(self):
        """
        Evaluates the model on the dataset according to the metric.

        First, however, this method will:
        1) preprocess the dataset according to the metric's needs
        2) train the model on the preprocess dataset and test it
        3) evaluate the metric
        """
        # always have to preprocess - self.datasets is required for eval

        # 1) initialize the dataset splits dependent on the metric and initialize the model
        # with the dataset as its parameter
        self._init_datasets()

        # 2) for each dataset, we preprocess the output model directory and dataset,
        # train/test, and evaluate
        for dataset in self.datasets:
            output_path = self._preprocess(dataset)
            if self.config.run_type == RunType.PREPROCESS:
                logging.debug(
                    "Run type is PREPROCESS. Skipping model training and metric evaluation."
                )
                logging.info(f"Output path for model: {output_path}")
            elif self.config.run_type == RunType.AUTO_TRAIN_TEST:
                self.model.train_and_test(
                    os.path.join(output_path, self.MODEL_CONFIG_FILENAME)
                )

            if self.config.run_type in [RunType.EVAL_ONLY, RunType.AUTO_TRAIN_TEST]:
                # verify that there is the model output where expected
                assert os.path.exists(
                    os.path.join(output_path, self.output_path_name.value)
                ), f"Model output file not found: {os.path.join(output_path, self.output_path_name.value)}"

                # finally, we evaluate on the test data (ground truth)
                # and the predicted data from the model
                self._eval(output_path, dataset)

    # ** PREPROCESSING DATASET SECTION **
    @final
    def _init_datasets(self):
        """
        Initializes the dataset by first checking if the processed dataset is
        already cached, and if not, to go through the processing steps.

        Steps:
        1. Checks if datasets is specified. If not, then read from the default datasets.
            a. Datasets are not specified: Read from the default datasets path defined in the metric subclass.
            b. Datasets are specified: Use the datasets provided in the config. If only "tag" is provided,
                read from the default datasets and find the matching tag.

        Then we do the following to all of the above cases:
            1. Checks if the datasets are supported by this metric.
            2. Initializes the dataset filters per dataset based on the config.
            3. Creates the dataset instances with the appropriate filters applied.
        """
        # first we create default datasets to be used
        assert hasattr(
            self, "default_datasets_path"
        ), "If datasets are not specified in the config, the metric subclass must define default_datasets_path."

        with open(self.default_datasets_path, "r") as f:
            default_datasets = yaml.safe_load(f)["datasets"]

        # now we check if datasets are specified in the config
        if not hasattr(self.config, "datasets") or len(self.config.datasets) == 0:
            # datasets are not specified, we use the default datasets
            self.config.datasets = default_datasets

        # then we check if any dataset only has a "tag" specified
        new_datasets = []
        for dataset in self.config.datasets:
            if "tag" in dataset:
                # we need to find the matching dataset from the default datasets
                matching_datasets = [
                    d for d in default_datasets if d.get("tag", None) == dataset["tag"]
                ]
                assert (
                    len(matching_datasets) == 1
                ), f"Dataset with tag {dataset['tag']} not found or multiple found in default datasets."
                new_datasets.append(matching_datasets[0])
            else:
                new_datasets.append(dataset)

        # finally, we want to remove the tag associated with each dataset
        # to ensure that the model caches are consistent
        self.config.datasets = [
            {k: v for k, v in dataset.items() if k != "tag"} for dataset in new_datasets
        ]

        logging.debug("-" * 50 + "Datasets" + "-" * 50)
        logging.debug(self.config.datasets)
        logging.debug("-" * 100)

        # 1) check that all the specified datasets are supported by this metric
        for dataset in self.config.datasets:
            assert (
                dataset["name"] in self.supported_datasets
            ), f"Dataset {dataset} not supported by this metric."

        # 2) initialize the dataset and the dataset filters based on the config
        # for some reason, we need to create a wrapper function here, as lambdas don't work well...
        # searching it up, it's because of late binding
        def dataset_filters_builder_wrapper(dataset_filter):
            def builder(dataset_dict):
                return DATASET_FILTER_REGISTRY[dataset_filter["name"]](
                    dataset_dict,
                    **{k: v for k, v in dataset_filter.items() if k != "name"},
                )

            return builder

        # start with a list of list of dataset filter builders
        # where each inner list corresponds to the filters for a dataset,
        # and the outer list corresponds to the datasets
        self.dataset_filters_builders_list = []

        for dataset in self.config.datasets:
            builders = []
            for dataset_filter in dataset["filters"]:
                builders.append(dataset_filters_builder_wrapper(dataset_filter))
            self.dataset_filters_builders_list.append(builders)

        assert len(self.dataset_filters_builders_list) == len(
            self.config.datasets
        ), "Mismatch in number of datasets and dataset filter builders."

        # 3) finally, with the dataset filters built, we create all the filters that we need
        self.datasets = []
        for dataset, builders in zip(
            self.config.datasets, self.dataset_filters_builders_list
        ):
            # now we create a dataset instance with the appropriate filters
            self.datasets.append(
                DATASET_REGISTRY[dataset["name"]](
                    dataset, [builder(dataset) for builder in builders]
                )
            )

        # verify that the datasets are properly initialized
        # TODO: add a unit test for this!
        for dataset in self.datasets:
            logging.debug("-" * 100)
            logging.debug(dataset)
