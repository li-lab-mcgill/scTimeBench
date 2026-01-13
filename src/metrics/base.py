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
    GRAPH_SIM = "graph_sim.pkl"


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

        # 2) we check if the database already has this model output cached
        cached_output_path = self.db_manager.get_model_output_path(self.model)
        if cached_output_path is not None:
            assert os.path.exists(
                cached_output_path
            ), f"Cached model output path not found: {cached_output_path}"
            print("Model output cache found. Loading from cache.")
            return cached_output_path

        # 3) create the output directory for this model that is parametrized by
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
        }

        # write out the yaml config file for the model
        yaml_config_path = os.path.join(output_path, self.MODEL_CONFIG_FILENAME)
        with open(yaml_config_path, "w") as f:
            yaml.safe_dump(yaml_config, f)

        # now let's save this hash output dir to the database as well
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
                print(
                    "Run type is PREPROCESS. Skipping model training and metric evaluation."
                )
                print(f"Output path for model: {output_path}")
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
                # TODO: change this so that it's the test data that's loaded instead
                self._eval(output_path)

    # ** PREPROCESSING DATASET SECTION **
    @final
    def _init_datasets(self):
        """
        Initializes the dataset by first checking if the processed dataset is
        already cached, and if not, to go through the processing steps.

        Steps:
        1. Calls _populate_dataset_filters to grab the necessary dataset filters.
        2. Checks if the cache exists based on the dataset filters and dataset.
        3. Applies each filter to the dataset in sequence.
        4. Caches the processed dataset for future use.
        5. Returns the processed dataset path
        """
        # 1) check that all the specified datasets are supported by this metric
        for dataset in self.config.datasets:
            assert (
                dataset["name"] in self.supported_datasets
            ), f"Dataset {dataset} not supported by this metric."

        # 2) set the defaults to just be the supported datasets instead
        if self.config.datasets == []:
            self.config.datasets = self.supported_datasets

        # 3) initialize the dataset and the dataset filters based on the config
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
            print("-" * 100)
            dataset.print()
