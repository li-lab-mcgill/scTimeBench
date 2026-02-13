"""
Base filter for datasets. Every metric will likely require different splits of the
data, so this base class will define the necessary interface for dataset preprocessing.
"""
from crispy_fishstick.shared.constants import ObservationColumns, DATASET_DIR
import json
import hashlib
import os

# ** DATASET FILTERING SECTION **
DATASET_FILTER_REGISTRY = {}


def register_dataset_filter(cls):
    """Decorator to register a dataset filter class in the DATASET_FILTER_REGISTRY."""
    DATASET_FILTER_REGISTRY[cls.__name__] = cls


class BaseDatasetFilter:
    def __init__(self, dataset_dict):
        self.dataset_dict = dataset_dict
        self.splits = False  # whether this filter produces train-test splits

    def requires_caching(self):
        """
        By default, most filters should be simple and not require external packages.
        """
        return False

    def __init_subclass__(cls):
        """
        Automatically register subclasses in the DATASET_FILTER_REGISTRY.
        """
        register_dataset_filter(cls)

    def _parameters(self):
        """
        Subclasses should implement this method to return filter-specific parameters.
        e.g.: timesplit parameters for time-based filters, or cell-lineage file for
        lineage-based filters.
        """
        return {}

    def filter(self, ann_data, **kwargs):
        """
        Subclasses should implement this method to filter and split the dataset
        according to the metric's requirements.
        """
        raise NotImplementedError("Subclasses should implement this method.")


# ** DATASET INFORMATION **
DATASET_REGISTRY = {}


def register_dataset(cls):
    """Decorator to register a dataset class in the DATASET_REGISTRY."""
    DATASET_REGISTRY[cls.__name__] = cls


class BaseDataset:
    def __init__(
        self, dataset_dict, dataset_filters: list[BaseDatasetFilter], output_dir
    ):
        self.dataset_dict = dataset_dict
        self.dataset_filters = dataset_filters
        self.output_dir = output_dir
        self.TRAIN_PROCESSED_DATA_FILE = "train_processed_data.h5ad"
        self.TEST_PROCESSED_DATA_FILE = "test_processed_data.h5ad"

    def __init_subclass__(cls):
        register_dataset(cls)

    def _load_data(self):
        """
        Subclasses should implement this method to load the dataset.
        Each dataset might require its own loading mechanism, as well as preprocessing
        mechanisms, but the BaseDatasetFilter should hopefully work on all datasets.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def requires_caching(self):
        """
        Some datasets might require caching because they have filters that take a long time to run (e.g., pseudotime estimation).
        By default, we assume that datasets do not require caching, but this can be overridden by specific datasets if necessary.
        """
        return any([f.requires_caching() for f in self.dataset_filters])

    def encode_filters(self, i=None):
        """
        Generate a string representation of the applied dataset filters
        and their parameters.

        This can be used to cache processed datasets.
        """

        filters_to_encode = (
            self.dataset_filters if i is None else self.dataset_filters[:i]
        )

        filter_names = [
            {"name": type(f).__name__, "parameters": f._parameters()}
            for f in filters_to_encode
        ]
        return json.dumps(filter_names, sort_keys=True)

    def encode_dataset_dict(self):
        """
        Generate a string representation of the dataset configuration.

        This can be used to cache processed datasets.
        """
        # we exclude data_path to avoid path differences across systems
        # we also exclude requires_caching and filters since
        # requires_caching should not affect the processing itself
        # and the filters are encoded elsewhere
        blocklist = ["data_path", "requires_caching", "filters"]
        return json.dumps(
            {
                k: v
                for k, v in self.dataset_dict.items()
                if k
                not in blocklist  # we exclude data_path to avoid path differences across systems
            },
            sort_keys=True,
        )

    def get_name(self):
        """
        Get the name of the dataset from the configuration.
        """
        return self.dataset_dict["name"]

    def load_data(self):
        """
        This ensures that the dataset loading is done properly.

        We require the following:
        1. Load the data from the source.
        2. Include observation metadata of cell_type, and timepoint.
        3. Drop everything else not required, to speed up processing.
        4. Apply the dataset filters provided.
        5. Return the train and test splits.

        Update:
        > Because I'm getting annoyed about the dependency hell we need for psupertime...
        > I've decided that the best way forward is to simply add pypsupertime as a possible
        > thing to have, but not necessary. Instead, we would require them to run the preprocessing
        > ahead of time, which is what this function does -- loads the data (running them through the filter)
        > and saving them to their respective output directory.
        """
        self._load_data()

        # now let's verify that the necessary columns exist
        assert hasattr(
            self, "data"
        ), "Dataset must have a 'data' attribute after loading."
        assert (
            ObservationColumns.CELL_TYPE.value in self.data.obs.columns
        ), f"Dataset must have '{ObservationColumns.CELL_TYPE.value}' in observation metadata."
        assert (
            ObservationColumns.TIMEPOINT.value in self.data.obs.columns
        ), f"Dataset must have '{ObservationColumns.TIMEPOINT.value}' in observation metadata."

        # then we put this through the dataset filtering process
        encountered_split = False
        for i, dataset_filter in enumerate(self.dataset_filters):
            # error out if we have multiple split filters
            if dataset_filter.splits and encountered_split:
                raise ValueError(
                    "Multiple dataset filters producing splits are not supported."
                )

            checkpoint_dir = self.get_checkpoint_dir(i)

            # otherwise, if this is the first time we're seeing split, we handle it differently
            if dataset_filter.splits:
                encountered_split = True
                train_data, test_data = dataset_filter.filter(
                    self.data, checkpoint_dir=checkpoint_dir
                )
                self.data = (train_data, test_data)

            # if we have already encountered a split, we need to apply the filter to both splits
            elif encountered_split:
                train_data = dataset_filter.filter(
                    self.data[0], checkpoint_dir=checkpoint_dir
                )
                test_data = dataset_filter.filter(
                    self.data[1], checkpoint_dir=checkpoint_dir
                )
                self.data = (train_data, test_data)

            # otherwise, just apply the filter normally
            else:
                self.data = dataset_filter.filter(
                    self.data, checkpoint_dir=checkpoint_dir
                )

        assert (
            encountered_split
        ), "At least one dataset filter must produce train-test splits."

        return self.data

    def __str__(self):
        return (
            f"Dataset Name: {self.get_name()}\n"
            f"Dataset Config: {self.dataset_dict}\n"
            f"Applied Filters: {[type(f).__name__ + ', parameters: ' + str(f._parameters()) for f in self.dataset_filters]}"
        )

    def get_dataset_dir(self):
        """
        Get a unique directory name for this dataset configuration, which can be used for caching.
        This is based on the dataset name, the encoded dataset dictionary, and the encoded filters.

        It should be a hashable string that uniquely identifies the dataset configuration and applied filters,
        so that we can cache processed datasets effectively.
        """
        unique_string = json.dumps(
            {
                "dataset_dict": self.encode_dataset_dict(),
                "filters": self.encode_filters(),
            },
            sort_keys=True,
        )

        # Generate a base64 encoded string of the unique string
        return os.path.join(
            self.output_dir,
            DATASET_DIR,
            hashlib.sha256(unique_string.encode()).hexdigest(),
        )

    def get_checkpoint_dir(self, i):
        """
        We define a checkpoint as the ith filter in the pipeline.
        This is used to save intermediate results that take a while to get to (such as pseudotime estimation).
        """
        unique_string = json.dumps(
            {
                "dataset_dict": self.encode_dataset_dict(),
                "filters": self.encode_filters(i),
            },
            sort_keys=True,
        )
        return os.path.join(
            self.output_dir,
            DATASET_DIR,
            "checkpoints",
            hashlib.sha256(unique_string.encode()).hexdigest(),
        )

    def create_dataset_dir(self):
        """
        Create a directory for this dataset configuration under the given base path.
        """
        os.makedirs(self.get_dataset_dir(), exist_ok=True)
