"""
Base class for all metrics. They should all implement the eval method, and
depend on the dataset that they belong to.
"""
from config import Config
from typing import final
from models.base import MODEL_REGISTRY
from dataset.base import DATASET_REGISTRY
from database import DatabaseManager


METRIC_REGISTRY = {}


def register_metric(cls):
    """Decorator to register a metric class in the METRIC_REGISTRY."""
    METRIC_REGISTRY[cls.__name__] = cls


# also store a registry of metrics of name to class
class BaseMetric:
    def __init__(self, config: Config, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.config = config
        # 1) check the required feature specs match up
        self._check_feature_specs()
        self._init_model()
        self._check_model_feature_specs()

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
    @final
    def eval(self, *args, **kwargs):
        """
        Evaluates the model on the dataset according to the metric.

        First, however, this method will:
        1) preprocess the dataset according to the metric's needs
        2) train the model on the preprocess dataset
        3) grab whatever the model needs to evaluate the metric
        4) evaluate the metric
        """
        # 1) initialize the dataset splits dependent on the metric
        self._init_dataset()

        # TODO: figure this out below
        # # 2) train the model on the dataset
        # self.model.train(self.dataset)

        # # 3) evaluate the model according to the metric
        # self.model.test(self.dataset)

        # 4) evaluate the metric based on the model outputs
        self._eval()

    # ** METRIC FEATURE SPECS SECTION **
    @final
    def _check_feature_specs(self):
        """
        Populate the feature specifications required for the metric.
        """
        self.required_feature_specs = None
        self._populate_feature_specs()
        assert (
            self.required_feature_specs is not None
        ), "Subclasses must define required_feature_specs"

    def _populate_feature_specs(self):
        """
        Subclasses should implement this method to define
        their required feature specifications.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @final
    def _init_model(self):
        """
        Initialize the model based on the configuration.
        """
        model_name = self.config.model["name"]
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model {model_name} not found in registry.")
        model_class = MODEL_REGISTRY[model_name]
        self.model = model_class(self.config)

    @final
    def _check_model_feature_specs(self):
        """
        Check if the model's feature specifications satisfy the metric's requirements.
        """
        model_feature_specs = self.model.required_feature_specs
        required_specs = self.required_feature_specs

        for spec in required_specs:
            if spec not in model_feature_specs:
                raise ValueError(
                    f"Model {self.config.model['name']} does not satisfy the required feature spec: {spec}"
                )

    # ** PREPROCESSING DATASET SECTION **
    def _populate_dataset_filters(self):
        """
        Populate the dataset filters required for the metric.
        Expect a list of dataset filter instances.
        """
        raise NotImplementedError("Subclasses should implement this method.")

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
        """
        # 1) initialize the dataset and the dataset filters based on the metric
        self.dataset = DATASET_REGISTRY[self.config.dataset["name"]](self.config)
        self._populate_dataset_filters()

        # now we check the cache based on the set preprocessed directory
        # stop if the cache exists
        cached_dataset_path = self.db_manager.get_processed_dataset_path(
            self.dataset.config.dataset.get("name"), self.dataset_filters
        )

        if cached_dataset_path is not None:
            print("Processed dataset cache found. Loading from cache.")
            self.dataset.load_cached_data(cached_dataset_path)
            return

        # if the cache does not exist, we go ahead and process the dataset
        # this will save out a cached version as well, which we will then
        # insert into the database
        print("No processed dataset cache found. Processing dataset.")
        save_path = self.dataset.load_data(self.dataset_filters)

        self.db_manager.insert_processed_dataset(
            self.dataset.config.dataset["name"], self.dataset_filters, save_path
        )
