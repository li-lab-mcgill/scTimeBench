"""
Base preprocessor for datasets. Every metric will likely require different splits of the
data, so this base class will define the necessary interface for dataset preprocessing.
"""
from scTimeBench.shared.constants import ObservationColumns, DATASET_DIR
from scTimeBench.shared.helpers import _resolve_shared_resource_path
import json
import hashlib
import os
import yaml

# ** DATASET PREPROCESSOR SECTION **
DATASET_PREPROCESSOR_REGISTRY = {}


def register_dataset_preprocessor(cls):
    """Decorator to register a dataset preprocessor class in the DATASET_PREPROCESSOR_REGISTRY."""
    DATASET_PREPROCESSOR_REGISTRY[cls.__name__] = cls


class BaseDatasetPreprocessor:
    def __init__(self, dataset_dict):
        self.dataset_dict = dataset_dict
        self.splits = False  # whether this preprocessor produces train-test splits

    def requires_caching(self):
        """
        By default, most preprocessors should be simple and not require external packages.
        """
        return False

    def __init_subclass__(cls):
        """
        Automatically register subclasses in the DATASET_PREPROCESSOR_REGISTRY.
        """
        register_dataset_preprocessor(cls)

    def _parameters(self):
        """
        Subclasses should implement this method to return preprocessor-specific parameters.
        e.g.: timesplit parameters for time-based preprocessors, or cell-lineage file for
        lineage-based preprocessors.
        """
        return {}

    def preprocess(self, ann_data, **kwargs):
        """
        Subclasses should implement this method to preprocessor and split the dataset
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
        self,
        dataset_dict,
        dataset_preprocessors: list[BaseDatasetPreprocessor],
        output_dir,
        cached_train_dataset=None,  # this should be BaseDataset, but Python 3.10 has no typing for this
    ):
        """
        Initialize the dataset.

        Args:
            dataset_dict (dict): The dictionary containing the dataset information.
            dataset_preprocessors (list): A list of dataset preprocessors.
            output_dir (str): The directory where the processed dataset will be saved.
            cached_train_dataset (BaseDataset, optional):
                The cached train dataset.

                If this is provided, then this is what is used for training the model,
                and the test dataset is generated from this using the dataset_preprocessors.
        """
        self.dataset_dict = dataset_dict
        self.dataset_preprocessors = dataset_preprocessors
        self.output_dir = output_dir
        self.cached_train_dataset = cached_train_dataset
        self.TRAIN_PROCESSED_DATA_FILE = "train_processed_data.h5ad"
        self.TEST_PROCESSED_DATA_FILE = "test_processed_data.h5ad"

        # load into this the cell lineages genes if it exists
        # which specify the genes involved in the lineage transitions
        if "cell_lineage_genes_file" in self.dataset_dict:
            with open(
                _resolve_shared_resource_path(
                    self.dataset_dict["cell_lineage_genes_file"]
                )
            ) as f:
                self.cell_lineage_genes = yaml.safe_load(f)
        else:
            self.cell_lineage_genes = None

    def __init_subclass__(cls):
        register_dataset(cls)

    def _load_data(self):
        """
        Subclasses should implement this method to load the dataset.
        Each dataset might require its own loading mechanism, as well as preprocessing
        mechanisms, but the BaseDatasetPreprocessor should hopefully work on all datasets.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def requires_caching(self):
        """
        Some datasets might require caching because they have preprocessors that take a long time to run (e.g., pseudotime estimation).
        By default, we assume that datasets do not require caching, but this can be overridden by specific datasets if necessary.
        """
        return any([f.requires_caching() for f in self.dataset_preprocessors])

    def encode_preprocessors(self, i=None):
        """
        Generate a string representation of the applied dataset preprocessors
        and their parameters.

        This can be used to cache processed datasets.
        """

        preprocessors_to_encode = (
            self.dataset_preprocessors if i is None else self.dataset_preprocessors[:i]
        )

        preprocessor_names = [
            {"name": type(f).__name__, "parameters": f._parameters()}
            for f in preprocessors_to_encode
        ]
        return json.dumps(preprocessor_names, sort_keys=True)

    def encode_dataset_dict(self):
        """
        Generate a string representation of the dataset configuration.

        This can be used to cache processed datasets.
        """
        # we exclude data_path to avoid path differences across systems
        # we also exclude requires_caching and preprocessors since
        # requires_caching should not affect the processing itself
        # and the preprocessors are encoded elsewhere
        # then, whether the user decides to include the train dataset caching or not
        # should not affect the hash of the dataset
        # same goes for the cell lineage genes file

        # TODO: consider changing the blocklist to an includes list
        # TODO: I really think it should be the other way around...
        blocklist = [
            "data_path",
            "requires_caching",
            "data_preprocessing_steps",
            "tag",
            "equiv_train_dataset_tag",
            "cell_lineage_genes_file",
        ]
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
        4. Apply the dataset preprocessors provided.
        5. Return the train and test splits.

        Update:
        > Because I'm getting annoyed about the dependency hell we need for psupertime...
        > I've decided that the best way forward is to simply add pypsupertime as a possible
        > thing to have, but not necessary. Instead, we would require them to run the preprocessing
        > ahead of time, which is what this function does -- loads the data (running them through the preprocessor)
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

        # then we put this through the dataset preprocessoring process
        encountered_split = False
        for i, dataset_preprocessor in enumerate(self.dataset_preprocessors):
            # error out if we have multiple split preprocessors
            if dataset_preprocessor.splits and encountered_split:
                raise ValueError(
                    "Multiple dataset preprocessors producing splits are not supported."
                )

            checkpoint_dir = self.get_checkpoint_dir(i)

            # otherwise, if this is the first time we're seeing split, we handle it differently
            if dataset_preprocessor.splits:
                encountered_split = True
                train_data, test_data = dataset_preprocessor.preprocess(
                    self.data, checkpoint_dir=checkpoint_dir
                )
                self.data = (train_data, test_data)

            # if we have already encountered a split, we need to apply the preprocessor to both splits
            elif encountered_split:
                train_data = dataset_preprocessor.preprocess(
                    self.data[0], checkpoint_dir=checkpoint_dir
                )
                test_data = dataset_preprocessor.preprocess(
                    self.data[1], checkpoint_dir=checkpoint_dir
                )
                self.data = (train_data, test_data)

            # otherwise, just apply the preprocessor normally
            else:
                self.data = dataset_preprocessor.preprocess(
                    self.data, checkpoint_dir=checkpoint_dir
                )

        assert (
            encountered_split
        ), "At least one dataset preprocessor must produce train-test splits."

        return self.data

    def __str__(self):
        return (
            f"Dataset Name: {self.get_name()}\n"
            f"Dataset Config: {self.dataset_dict}\n"
            f"Applied preprocessors: {[type(f).__name__ + ', parameters: ' + str(f._parameters()) for f in self.dataset_preprocessors]}"
        )

    def _get_unique_dataset_dir(self, output_dir):
        """
        Get a unique directory name for this dataset configuration, which can be used for caching.
        This is based on the dataset name, the encoded dataset dictionary, and the encoded preprocessors.

        It should be a hashable string that uniquely identifies the dataset configuration and applied preprocessors,
        so that we can cache processed datasets effectively.
        """
        unique_string = json.dumps(
            {
                "dataset_dict": self.encode_dataset_dict(),
                "preprocessors": self.encode_preprocessors(),
            },
            sort_keys=True,
        )

        # Generate a base64 encoded string of the unique string
        return os.path.join(
            output_dir,
            DATASET_DIR,
            hashlib.sha256(unique_string.encode()).hexdigest(),
        )

    def get_train_dataset_dir(self):
        """
        Get the directory where the processed train dataset is stored.

        If train_output_dir is provided, we will use that directly, otherwise,
        we will use the old way of generating a unique directory based on the dataset
        configuration and preprocessors.
        """
        if self.cached_train_dataset is not None:
            return self.cached_train_dataset.get_train_dataset_dir()
        else:
            return self._get_unique_dataset_dir(self.output_dir)

    def get_test_dataset_dir(self):
        """
        Get the directory where the processed test dataset is stored.

        If cached_train_dataset is provided, we will use a new output directory
        which is <cached_train_dataset_dir>/tests/<hash>, where <hash> is a hash of the dataset configuration and preprocessors.
        """
        if self.cached_train_dataset is not None:
            return self._get_unique_dataset_dir(
                os.path.join(self.cached_train_dataset.get_train_dataset_dir(), "tests")
            )
        else:
            return self._get_unique_dataset_dir(self.output_dir)

    def get_checkpoint_dir(self, i):
        """
        We define a checkpoint as the ith preprocessor in the pipeline.
        This is used to save intermediate results that take a while to get to (such as pseudotime estimation).
        """
        unique_string = json.dumps(
            {
                "dataset_dict": self.encode_dataset_dict(),
                "preprocessors": self.encode_preprocessors(i),
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
        os.makedirs(self.get_train_dataset_dir(), exist_ok=True)
        os.makedirs(self.get_test_dataset_dir(), exist_ok=True)

    def is_equiv(self, other_dataset) -> bool:
        """
        Check if two datasets are equivalent based on their configuration and applied preprocessors.
        """
        return (
            self.encode_dataset_dict() == other_dataset.encode_dataset_dict()
            and self.encode_preprocessors() == other_dataset.encode_preprocessors()
        )
