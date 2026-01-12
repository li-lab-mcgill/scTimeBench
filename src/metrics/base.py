"""
Base class for all metrics. They should all implement the eval method, and
depend on the dataset that they belong to.
"""
from config import Config, RunType
from typing import final
from models.base import BaseModel
from shared.dataset.base import DATASET_REGISTRY
from database import DatabaseManager
from enum import Enum

import os
import pickle
import yaml


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

    def _preprocess(self):
        """
        Preprocessing steps required before evaluating the metric.
        """
        # first we check that the subclasses have defined the required feature specs
        assert (
            hasattr(self, "output_path_name")
            and type(self.output_path_name) is OutputPathName
        ), "Subclasses must define output_path_name of type OutputPathName"

        # 1) initialize the dataset splits dependent on the metric and initialize the model
        # with the dataset as its parameter
        self._init_dataset()
        self._init_model()

        # now we check if the database already has this model output cached
        cached_output_path = self.db_manager.get_model_output_path(self.model)
        if cached_output_path is not None:
            assert os.path.exists(
                cached_output_path
            ), f"Cached model output path not found: {cached_output_path}"
            print("Model output cache found. Loading from cache.")
            return cached_output_path

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
            pickle.dump(self.dataset, f)

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
        # always have to preprocess - self.dataset is required for eval
        output_path = self._preprocess()

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

    # ** METRIC FEATURE SPECS SECTION **
    @final
    def _init_model(self):
        """
        Initialize the model based on the configuration.
        """
        model_name = self.config.model["name"]
        if model_name not in self.config.get_available_models():
            raise ValueError(f"Model {model_name} not found in registry.")

        self.model = BaseModel(self.config, self.dataset)

        # Check if the model's feature specifications satisfy the metric's requirements.
        model_feature_specs = self.model.required_feature_specs
        required_specs = self.required_feature_specs
        assert (
            self.required_feature_specs is not None
        ), "Subclasses must define required_feature_specs"

        for spec in required_specs:
            if spec not in model_feature_specs:
                raise ValueError(
                    f"Model {self.config.model['name']} does not satisfy the required feature spec: {spec}"
                )

    # ** PREPROCESSING DATASET SECTION **
    @final
    def _init_dataset(self):
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
        # make sure that the dataset filters are populated
        assert (
            self.dataset_filters is not None
        ), "Subclasses must define dataset_filters"

        # 1) initialize the dataset and the dataset filters based on the metric
        self.dataset = DATASET_REGISTRY[self.config.dataset["name"]](
            self.config, self.dataset_filters
        )
