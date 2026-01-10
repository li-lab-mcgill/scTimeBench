"""
Base filter for datasets. Every metric will likely require different splits of the
data, so this base class will define the necessary interface for dataset preprocessing.
"""

from enum import Enum
import scanpy as sc
import os
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

    def __init_subclass__(cls):
        """
        Automatically register subclasses in the DATASET_FILTER_REGISTRY.
        """
        register_dataset_filter(cls)

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


class ObservationColumns(Enum):
    CELL_TYPE = "cell_type"
    TIMEPOINT = "timepoint"


class BaseDataset:
    def __init__(self, config):
        self.config = config

    def __init_subclass__(cls):
        register_dataset(cls)

    def _load_data(self):
        """
        Subclasses should implement this method to load the dataset.
        Each dataset might require its own loading mechanism, as well as preprocessing
        mechanisms, but the BaseDatasetFilter should hopefully work on all datasets.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def encode_dataset_path(self, dataset_filters):
        """
        Generate a hash for the processed dataset based on the applied filters
        and the original dataset configuration.

        This can be used to cache processed datasets.
        """
        # Create a unique string based on dataset config and filter names
        filter_names = [type(f).__name__ for f in dataset_filters]
        unique_string = json.dumps(
            {
                "dataset_config": self.config.dataset,
                "filters": filter_names,
            },
            sort_keys=True,
        )
        # Generate a base64 encoded string of the unique string
        return hashlib.sha256(unique_string.encode()).hexdigest() + ".h5ad"

    def load_data(self, filters):
        """
        This ensures that the dataset loading is done properly.

        We require the following:
        1. Load the data from the source.
        2. Include observation metadata of cell_type, and timepoint.
        3. Drop everything else not required, to speed up processing.
        4. Apply the dataset filters provided.
        5. Write to cache.
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
        for dataset_filter in filters:
            self.data = dataset_filter.filter(self.data)

        # now we cache the processed dataset for future use
        os.makedirs(self.config.dataset["preprocessed_dir"], exist_ok=True)
        dataset_hash = self.encode_dataset_path(filters)
        cached_dataset_path = (
            f"{self.config.dataset['preprocessed_dir']}/{dataset_hash}"
        )
        self.data.write_h5ad(cached_dataset_path)
        return cached_dataset_path

    def load_cached_data(self, cached_dataset_path):
        """
        This ensures that the dataset loading is done properly.

        We require the following:
        1. Load the data from the source.
        2. Include observation metadata of cell_type, and timepoint.
        3. Drop everything else not required, to speed up processing.
        """
        self.data = sc.read_h5ad(cached_dataset_path)

        # now let's verify that the necessary columns exist
        assert (
            ObservationColumns.CELL_TYPE.value in self.data.obs.columns
        ), f"Dataset must have '{ObservationColumns.CELL_TYPE.value}' in observation metadata."
        assert (
            ObservationColumns.TIMEPOINT.value in self.data.obs.columns
        ), f"Dataset must have '{ObservationColumns.TIMEPOINT.value}' in observation metadata."
