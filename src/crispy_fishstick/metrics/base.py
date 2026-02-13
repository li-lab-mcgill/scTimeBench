"""
Base class for all metrics. They should all implement the eval method, and
depend on the dataset that they belong to.
"""

from typing import final

from crispy_fishstick.config import Config, RunType
from crispy_fishstick.metrics.model_manager import ModelManager
from crispy_fishstick.shared.dataset.base import (
    BaseDataset,
    DATASET_REGISTRY,
    DATASET_FILTER_REGISTRY,
)
from crispy_fishstick.shared.constants import (
    RequiredOutputFiles,
    PICKLED_DATASET_FILENAME,
    MODEL_CONFIG_FILENAME,
)
from crispy_fishstick.shared.utils import load_test_dataset
from crispy_fishstick.database import DatabaseManager

import os
import pickle
import yaml
import logging
import json


METRIC_REGISTRY = {}


def register_metric(cls):
    """Decorator to register a metric class in the METRIC_REGISTRY."""
    METRIC_REGISTRY[cls.__name__] = cls
    return cls


SKIP_METRIC_REGISTRY = {}


def skip_metric(cls):
    """Decorator to register a skip metric class in the SKIP_METRIC_REGISTRY."""
    SKIP_METRIC_REGISTRY[cls.__name__] = cls
    return cls


# also store a registry of metrics of name to class
@register_metric
class BaseMetric:
    def __init__(
        self,
        config: Config,
        db_manager: DatabaseManager,
        metric_config: dict,
    ):
        self.db_manager = db_manager
        self.config = config
        self.metric_config = metric_config
        self.params = {}
        self.optional_datasets_path = None

        # skip the preprocessing steps if it has submetrics, as they will handle it themselves
        if len(self.submetrics) > 0:
            logging.debug("Metric has submetrics, skipping preprocessing.")
            return

        # then we set the defaults if not provided
        # and also store them in params for database logging
        for key, value in self._defaults().items():
            setattr(self, key, metric_config.get(key, value))
            self.params[key] = getattr(self, key)

        # now we call the setups that need to be defined by subclasses
        self._setup_trajectory_inference_model()
        self._setup_supported_datasets()
        self._setup_model_output_requirements()

        # finally we setup the datasets and metrics db
        # insert the metric if it's not already in the database
        if self.db_manager is None:
            # skip here because it might be just getting information
            return

        if not self.db_manager.has_metric(
            self.__class__.__name__, self._get_param_encoding()
        ):
            self.db_manager.insert_metric(
                self.__class__.__name__, self._get_param_encoding()
            )

        # initialize the dataset splits dependent on the metric and initialize the model
        # with the dataset as its parameter, also create its preprocessing output path
        self._init_datasets()
        self.models = []
        for dataset in self.datasets:
            output_path = self._preprocess(dataset)
            logging.info(f"Output path for model: {output_path}")

    def _setup_trajectory_inference_model(self):
        """By default do nothing"""
        logging.debug(
            f"No trajectory inference model specified for this metric {self.__class__.__name__}."
        )

    def _setup_supported_datasets(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _setup_model_output_requirements(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _defaults(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _get_param_encoding(self):
        return json.dumps(self.params)

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
            if not hasattr(base, "submetrics"):
                base.submetrics = []
            if hasattr(base, "submetrics"):
                base.submetrics.append(cls)

    # ** METRIC EVALUATION SECTION, main function to be called (no others should be called) **
    def _eval(self):
        """
        Main evaluation function which calls all the submetrics if applicable.

        This function will be called after the model has been trained and tested,
        and the model outputs are available at output_path.

        Subclasses should implement the method `_submetric_eval` to evaluate the metric
        based on the model outputs and the dataset.
        """
        # skip the metric if it's in the skip registry, unless it's force rerun
        if (
            self.__class__.__name__ in SKIP_METRIC_REGISTRY
            and not self.config.force_rerun
        ):
            logging.info(
                f"Skipping metric {self.__class__.__name__} as it is marked to be skipped."
            )
            return

        logging.info(f"Evaluating metric {self.__class__.__name__}")

        # assert that the preprocessing was done correctly
        # we assume that each model corresponds to a dataset
        assert len(self.models) == len(
            self.datasets
        ), "Number of models and datasets must be the same."

        for model, dataset in zip(self.models, self.datasets):
            # first we skip if we already have the evaluation in the database
            if (
                self.db_manager.has_eval(
                    model,
                    self.__class__.__name__,
                    self._get_param_encoding(),
                )
                and not self.config.force_rerun
            ):
                logging.info(
                    f"Evaluation for metric {self.__class__.__name__} with params {self.params} already exists for model {model}. Skipping evaluation."
                )
                continue

            # this preprocessing step already happens during creation, skip this here!
            output_path = self.db_manager.get_model_output_path(model)

            if self.config.run_type == RunType.PREPROCESS:
                logging.debug(
                    "Run type is PREPROCESS. Skipping model training and metric evaluation."
                )
            elif (
                self.config.run_type == RunType.AUTO_TRAIN_TEST
                or self.config.run_type == RunType.TRAIN_ONLY
            ):
                # ** Note: we always rerun if some list is not complete this because of issue: https://github.com/ehuan2/crispy-fishstick/issues/53 **
                # ** This will not necessarily re-run the model, and the only time that is not saved is the activation of the venv **
                # ** But this should be okay because that time is negligible, and this ensures that **
                # ** The model outputs give what are expected. The model outputs should still be cached however. **
                if all(
                    isinstance(outputs_list, list)
                    for outputs_list in self.required_outputs
                ):
                    # list of list case -- require all of them to exist
                    required_outputs_exist = all(
                        all(
                            os.path.exists(os.path.join(output_path, output.value))
                            for output in output_set
                        )
                        for output_set in self.required_outputs
                    )
                else:
                    # list case
                    required_outputs_exist = all(
                        os.path.exists(os.path.join(output_path, output.value))
                        for output in self.required_outputs
                    )

                if not required_outputs_exist:
                    # before running to train and test, we check if the dataset
                    # requires caching, and if so, then we run to cache ahead of time
                    # We do this because some filters (e.g., psupertime)
                    # require the dataset to be preprocessed with a certain module
                    # which may not exist in other models.
                    if dataset.requires_caching():
                        logging.info(
                            f"Dataset {dataset} requires caching. Caching now before training and testing the model."
                        )
                        # we still load the test dataset for a couple of reasons:
                        # 1. we want to store this into cache so the model can load it
                        # 2. we still want the requires_caching however to avoid the extra load
                        # before training, unless required. This also helps us avoid
                        # creating an extra cache that is not necessary.
                        _ = load_test_dataset(output_path)

                    model.train_and_test(
                        os.path.join(output_path, MODEL_CONFIG_FILENAME)
                    )
                else:
                    logging.info(
                        f"Model output already exists at {output_path}. Skipping training and generation."
                    )

            if self.config.run_type in [RunType.EVAL_ONLY, RunType.AUTO_TRAIN_TEST]:
                # verify that required output files exist
                if all(
                    isinstance(outputs_list, list)
                    for outputs_list in self.required_outputs
                ):
                    # list of list case - check that at least one set exists
                    outputs_valid = any(
                        all(
                            os.path.exists(os.path.join(output_path, output.value))
                            for output in output_set
                        )
                        for output_set in self.required_outputs
                    )
                else:
                    # list case - check all required outputs exist
                    outputs_valid = all(
                        os.path.exists(os.path.join(output_path, output.value))
                        for output in self.required_outputs
                    )

                assert (
                    outputs_valid
                ), f"Required model output files not found in: {output_path}"

                # finally, we evaluate on the test data (ground truth)
                # and the predicted data from the model
                self._submetric_eval(
                    **self._prep_kwargs_for_submetric_eval(output_path, dataset, model)
                )

    def _prep_kwargs_for_submetric_eval(self, output_path, dataset: BaseDataset, model):
        """
        Prepares the keyword arguments required for submetric evaluation.

        Subclasses can override this method to provide specific arguments
        needed for their submetric evaluations.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _submetric_eval(self, **kwargs):
        """
        Subclasses can implement this method to evaluate submetrics.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _preprocess(self, dataset):
        """
        Preprocessing steps required before evaluating the metric.
        """
        # 1) we check that the subclasses have defined the required feature specs
        model = ModelManager(self.config, dataset)
        self.models.append(model)

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
        output_path = os.path.join(self.config.output_dir, model._encode_output_path())
        os.makedirs(output_path, exist_ok=True)

        # to make our lives easier, we will also pickle our current dataset object
        # and store this in the output directory as well
        # so that the model can load this dataset object directly for training and testing
        assert hasattr(
            self, "required_outputs"
        ), "Subclasses must define required_outputs attribute."
        if all(
            isinstance(output, RequiredOutputFiles) for output in self.required_outputs
        ):
            required_outputs_serialized = [
                output.value for output in self.required_outputs
            ]
        elif all(isinstance(output, list) for output in self.required_outputs):
            if not all(
                all(isinstance(item, RequiredOutputFiles) for item in output_set)
                for output_set in self.required_outputs
            ):
                raise AssertionError(
                    "All required_outputs entries must be RequiredOutputFiles"
                )
            required_outputs_serialized = [
                [output.value for output in output_set]
                for output_set in self.required_outputs
            ]
        else:
            raise AssertionError(
                "required_outputs must be a list or list of lists of RequiredOutputFiles"
            )

        logging.debug(f"Required outputs serialized: {required_outputs_serialized}")

        yaml_config = {
            "output_path": output_path,
            "dataset_pkl_path": os.path.join(
                dataset.get_dataset_dir(), PICKLED_DATASET_FILENAME
            ),
            "model": self.config.model_yaml_data,
            "required_outputs": required_outputs_serialized,
            "datasets": dataset.encode_dataset_dict(),
            "filters": dataset.encode_filters(),
        }

        # write out the yaml config file for the model
        yaml_config_path = os.path.join(output_path, MODEL_CONFIG_FILENAME)
        with open(yaml_config_path, "w") as f:
            yaml.safe_dump(yaml_config, f)

        # now let's save this hash output dir to the database as well, only if it doesn't exist
        if self.db_manager.get_model_output_path(model) is None:
            self.db_manager.insert_model_output(model, output_path)

        return output_path

    @final
    def eval(self):
        """
        Evaluation function that handles the calling of submetrics if applicable.

        Basically it happens as follows:
        1. If there are submetrics defined, we create an instance of each submetric
        2. We call the _eval function of each submetric. This ensures that each submetric
           can handle its own evaluation logic, datasets that it chooses, and models that it runs on.
        3. From this _eval function, we further call the _submetric_eval function that each subclass
           must implement to handle the actual evaluation logic.
        """
        if self.submetrics:
            for submetric in self.submetrics:
                if (
                    submetric.__name__ in SKIP_METRIC_REGISTRY
                    and not self.config.force_rerun
                ):
                    logging.info(
                        f"Skipping metric {submetric.__name__} as it is marked to be skipped."
                    )
                    continue

                submetric_instance: BaseMetric = submetric(
                    self.config,
                    self.db_manager,
                    self.metric_config,
                )
                submetric_instance.eval()
        else:
            self._eval()

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

        optional_datasets = []
        if self.optional_datasets_path is not None:
            with open(self.optional_datasets_path, "r") as f:
                optional_datasets = yaml.safe_load(f)["datasets"]

        # now we check if datasets are specified in the config
        if not hasattr(self.config, "datasets") or len(self.config.datasets) == 0:
            # datasets are not specified, we use the default datasets
            self.config.datasets = default_datasets

        # then we check if any dataset only has a "tag" specified
        new_datasets = []
        for dataset in self.config.datasets:
            if "tag" in dataset:
                # we need to find the matching dataset from the default datasets or optional datasets
                to_match = default_datasets + optional_datasets
                matching_datasets = [
                    d for d in to_match if d.get("tag", None) == dataset["tag"]
                ]
                assert (
                    len(matching_datasets) == 1
                ), f"Dataset with tag {dataset['tag']} not found or multiple found in default or optional datasets."
                new_datasets.append(matching_datasets[0])
            else:
                new_datasets.append(dataset)

        # now we modify the data paths if a data_dir is specified
        if hasattr(self.config, "data_dir"):
            for dataset in new_datasets:
                # if the data_path is not absolute, we join it with data_dir
                if os.path.isabs(dataset["data_path"]):
                    continue
                dataset["data_path"] = os.path.join(
                    self.config.data_dir, dataset["data_path"]
                )

        # finally, we want to remove the tag associated with each dataset
        # to ensure that the model caches are consistent
        self.config.datasets = [
            {k: v for k, v in dataset.items() if k != "tag"} for dataset in new_datasets
        ]

        logging.debug("-" * 50 + "Datasets" + "-" * 50)
        logging.debug(self.config.datasets)
        logging.debug("-" * 100)

        # 1) check that all the specified datasets are supported by this metric
        # if not, we simply take the ones that are supported
        dataset_for_metric = []
        for dataset in self.config.datasets:
            if dataset["name"] in self.supported_datasets:
                dataset_for_metric.append(dataset)
            else:
                logging.warning(
                    f"Dataset {dataset} not supported by this metric {self.__class__.__name__}."
                )

        logging.debug(
            "-" * 50 + f"Datasets for metric {self.__class__.__name__}" + "-" * 50
        )
        logging.debug(dataset_for_metric)
        logging.debug("-" * 100)

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

        for dataset in dataset_for_metric:
            builders = []
            for dataset_filter in dataset["filters"]:
                builders.append(dataset_filters_builder_wrapper(dataset_filter))
            self.dataset_filters_builders_list.append(builders)

        assert len(self.dataset_filters_builders_list) == len(
            dataset_for_metric
        ), "Mismatch in number of datasets and dataset filter builders."

        # 3) finally, with the dataset filters built, we create all the filters that we need
        self.datasets: list[BaseDataset] = []
        for dataset, builders in zip(
            dataset_for_metric, self.dataset_filters_builders_list
        ):
            # now we create a dataset instance with the appropriate filters
            self.datasets.append(
                DATASET_REGISTRY[dataset["name"]](
                    dataset,
                    [builder(dataset) for builder in builders],
                    self.config.output_dir,
                )
            )

        # 4) Then we insert these datasets into the database if they don't already exist,
        # and we also create their dataset directories
        for dataset in self.datasets:
            self.db_manager.insert_dataset(dataset)
            logging.debug(
                f"Processing dataset: {dataset} to {dataset.get_dataset_dir()}"
            )
            dataset.create_dataset_dir()
            # TODO: in this dataset directory, we can then store the base metrics we have
            # such as the base visualization, etc.
            # for now, let's just store the dataset object itself, and the model can load this for its own use
            pickled_dataset_path = os.path.join(
                dataset.get_dataset_dir(), PICKLED_DATASET_FILENAME
            )
            with open(pickled_dataset_path, "wb") as f:
                pickle.dump(dataset, f)

            # write out the dataset information to the directory as well
            with open(
                os.path.join(dataset.get_dataset_dir(), "dataset_info.yaml"), "w"
            ) as f:
                yaml.safe_dump(
                    {
                        "dataset_dict": dataset.dataset_dict,
                        "dataset_filters": [
                            {
                                "name": type(f).__name__,
                                "parameters": f._parameters(),
                            }
                            for f in dataset.dataset_filters
                        ],
                    },
                    f,
                )

        # verify that the datasets are properly initialized
        # TODO: add a unit test for this!
        for dataset in self.datasets:
            logging.debug("-" * 100)
            logging.debug(dataset)
