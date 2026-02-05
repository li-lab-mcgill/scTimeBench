"""
Shared utility functions for loading datasets and output files.
"""
import os
import pickle
import numpy as np
import pandas as pd

from crispy_fishstick.shared.constants import (
    RequiredOutputFiles,
    PICKLED_DATASET_FILENAME,
)


def load_test_dataset(output_path):
    """
    Load the test dataset from the pickled dataset file in output_path.

    Args:
        output_path: Path to the model output directory

    Returns:
        The test AnnData object from the dataset
    """
    dataset_pkl_path = os.path.join(output_path, PICKLED_DATASET_FILENAME)
    if not os.path.exists(dataset_pkl_path):
        raise FileNotFoundError(f"Dataset pickle not found: {dataset_pkl_path}")

    with open(dataset_pkl_path, "rb") as f:
        dataset = pickle.load(f)

    _, test_ann_data = dataset.load_data()
    return test_ann_data


def load_output_file(output_path, required_output: RequiredOutputFiles):
    """
    Load a model output file from output_path.

    Args:
        output_path: Path to the model output directory
        required_output: RequiredOutputFiles enum value specifying which file to load

    Returns:
        For .npy files: numpy array
        For .parquet files: pandas DataFrame
    """
    file_path = os.path.join(output_path, required_output.value)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Output file not found: {file_path}")

    if required_output.value.endswith(".npy"):
        return np.load(file_path)
    elif required_output.value.endswith(".parquet"):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unknown file type: {required_output.value}")
