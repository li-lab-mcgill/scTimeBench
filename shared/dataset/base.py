"""
Base filter for datasets. Every metric will likely require different splits of the
data, so this base class will define the necessary interface for dataset preprocessing.
"""
from shared.constants import ObservationColumns
import hashlib
import json

# ** DATASET FILTERING SECTION **
DATASET_FILTER_REGISTRY = {}


def register_dataset_filter(cls):
    """Decorator to register a dataset filter class in the DATASET_FILTER_REGISTRY."""
    DATASET_FILTER_REGISTRY[cls.__name__] = cls


class BaseDatasetFilter:
    def __init__(self, config):
        self.config = config
        self.splits = False  # whether this filter produces train-test splits

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

    def filter(self, ann_data):
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
    def __init__(self, config, dataset_filters):
        self.config = config
        self.dataset_filters = dataset_filters

    def __init_subclass__(cls):
        register_dataset(cls)

    def _load_data(self):
        """
        Subclasses should implement this method to load the dataset.
        Each dataset might require its own loading mechanism, as well as preprocessing
        mechanisms, but the BaseDatasetFilter should hopefully work on all datasets.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def encode_filters(self):
        """
        Generate a string representation of the applied dataset filters
        and their parameters.

        This can be used to cache processed datasets.
        """
        filter_names = [
            {"name": type(f).__name__, "parameters": f._parameters()}
            for f in self.dataset_filters
        ]
        return json.dumps(filter_names, sort_keys=True)

    def encode_config(self):
        """
        Generate a string representation of the dataset configuration.

        This can be used to cache processed datasets.
        """
        return json.dumps(self.config.dataset, sort_keys=True)

    def get_name(self):
        """
        Get the name of the dataset from the configuration.
        """
        return self.config.dataset["name"]

    def encode_dataset_path(self):
        """
        Generate a hash for the processed dataset based on the applied filters
        and the original dataset configuration.

        This can be used to cache processed datasets.
        """
        # Create a unique string based on dataset config and filter names
        filter_names = self.encode_filters()
        unique_string = json.dumps(
            {
                "dataset_config": self.config.dataset,
                "filters": filter_names,
            },
            sort_keys=True,
        )
        # Generate a base64 encoded string of the unique string
        return hashlib.sha256(unique_string.encode()).hexdigest() + ".h5ad"

    def load_data(self):
        """
        This ensures that the dataset loading is done properly.

        We require the following:
        1. Load the data from the source.
        2. Include observation metadata of cell_type, and timepoint.
        3. Drop everything else not required, to speed up processing.
        4. Apply the dataset filters provided.
        5. Return the train and test splits.
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
        for dataset_filter in self.dataset_filters:
            # error out if we have multiple split filters
            if dataset_filter.splits and encountered_split:
                raise ValueError(
                    "Multiple dataset filters producing splits are not supported."
                )

            # otherwise, if this is the first time we're seeing split, we handle it differently
            if dataset_filter.splits:
                encountered_split = True
                train_data, test_data = dataset_filter.filter(self.data)
                self.data = (train_data, test_data)

            # if we have already encountered a split, we need to apply the filter to both splits
            elif encountered_split:
                train_data = dataset_filter.filter(self.data[0])
                test_data = dataset_filter.filter(self.data[1])
                self.data = (train_data, test_data)

            # otherwise, just apply the filter normally
            else:
                self.data = dataset_filter.filter(self.data)

        assert (
            encountered_split
        ), "At least one dataset filter must produce train-test splits."
        return self.data
