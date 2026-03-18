"""
Shared utility functions for loading datasets and output files.
"""
import os
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import logging
import yaml

from scTimeBench.shared.constants import (
    RequiredOutputFiles,
    METHOD_CONFIG_FILENAME,
)

DATASET_CACHE_LIMIT = 3  # max number of datasets to cache in memory
DATASET_IN_MEM_CACHE = {}

OUTPUT_FILE_CACHE_LIMIT = 3  # max number of output files to cache in memory
OUTPUT_FILE_CACHE = {}


def clear_dataset_cache():
    """
    Clear the in-memory dataset cache.
    """
    DATASET_IN_MEM_CACHE.clear()


def get_dataset(output_path):
    """
    Get the dataset from the pickled dataset file in output_path.

    Args:
        output_path: Path to the method output directory
    Returns:
        The dataset object loaded from the pickled file
    """
    # first let's read the method yaml to get the dataset pickle path
    method_config_path = os.path.join(output_path, METHOD_CONFIG_FILENAME)
    if not os.path.exists(method_config_path):
        raise FileNotFoundError(f"method config file not found: {method_config_path}")
    logging.debug(f"Loading method config from {method_config_path}")
    with open(method_config_path, "r") as f:
        method_config = yaml.safe_load(f)

    dataset_pkl_path = method_config.get("dataset_pkl_path", None)
    if not os.path.exists(dataset_pkl_path):
        raise FileNotFoundError(f"Dataset pickle not found: {dataset_pkl_path}")

    logging.debug(f"Loading dataset from {dataset_pkl_path}")
    with open(dataset_pkl_path, "rb") as f:
        dataset = pickle.load(f)

    return dataset, dataset_pkl_path


def load_test_dataset(output_path):
    """
    Load the test dataset from the pickled dataset file in output_path.

    Args:
        output_path: Path to the method output directory

    Returns:
        The test AnnData object from the dataset
    """
    dataset, dataset_pkl_path = get_dataset(output_path)
    test_ann_data = DATASET_IN_MEM_CACHE.get(dataset_pkl_path, None)

    if test_ann_data is None:
        _, test_ann_data = dataset.load_data()
        DATASET_IN_MEM_CACHE[dataset_pkl_path] = test_ann_data
        if len(DATASET_IN_MEM_CACHE) > DATASET_CACHE_LIMIT:
            # Remove an arbitrary item (not the most efficient, but simple)
            DATASET_IN_MEM_CACHE.pop(next(iter(DATASET_IN_MEM_CACHE)))
    else:
        logging.debug("Loaded test dataset from in-memory cache.")

    return test_ann_data


def load_output_file(output_path, required_output: RequiredOutputFiles):
    """
    Load a method output file from output_path.

    Args:
        output_path: Path to the method output directory
        required_output: RequiredOutputFiles enum value specifying which file to load

    Returns:
        For .npy files: numpy array
        For .parquet files: pandas DataFrame
    """
    file_path = os.path.join(output_path, required_output.value)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Output file not found: {file_path}")

    if file_path in OUTPUT_FILE_CACHE:
        logging.debug(f"Loaded {required_output.value} from in-memory cache.")
        return OUTPUT_FILE_CACHE[file_path]

    if required_output.value.endswith(".npy"):
        output_file = np.load(file_path)
    elif required_output.value.endswith(".parquet"):
        output_file = pd.read_parquet(file_path)
    elif required_output.value.endswith(".h5ad"):
        output_file = sc.read_h5ad(file_path)
    else:
        raise ValueError(f"Unknown file type: {required_output.value}")

    # Cache the output file if we haven't exceeded the limit
    if len(OUTPUT_FILE_CACHE) >= OUTPUT_FILE_CACHE_LIMIT:
        # Remove an arbitrary item (not the most efficient, but simple)
        OUTPUT_FILE_CACHE.pop(next(iter(OUTPUT_FILE_CACHE)))

    OUTPUT_FILE_CACHE[file_path] = output_file
    return output_file


def is_raw(ann_data: sc.AnnData):
    """
    Returns whether the data is raw (i.e. not log-normalized) by checking that:
    1. All the data is non-negative
    2. All the data is integer-valued
    """
    gex = ann_data.X if isinstance(ann_data.X, np.ndarray) else ann_data.X.toarray()
    return np.all(gex >= 0) and np.all(np.mod(gex, 1) == 0)


def is_log_normalized_to_counts(ann_data, counts=10_000):
    """
    Heuristic to determine if the data is log-normalized to a certain counts threshold.
    Checks if ann_data.X is raw and if not, then checks to see that the data is
    log-normalized to counts=10_000.

    Args:
        ann_data: The AnnData object to check
        counts: The expected counts value (default is 10_000)

    Returns:
        True if the data is log-normalized to the expected counts, False otherwise
    """
    return not is_raw(ann_data) and np.allclose(
        np.sum(np.expm1(ann_data.X), axis=1), counts
    )


# Easter egg: Crispy Fishstick was our old name, and we want to pay homage to it
# with the following animation
import time
import os
import sys
import signal


def cheeky_message(sig, frame):
    sys.stdout.write("\n\r  \033[91mNO ESCAPE. ENJOY THE FISHSTICK.\033[0m  ")
    sys.stdout.flush()


def block_interrupts():
    signal.signal(signal.SIGINT, cheeky_message)


def restore_interrupts():
    signal.signal(signal.SIGINT, signal.SIG_DFL)


# --- THE ART ---
fishstick_art = [
    r"      .   ____   .        ",
    r"       \ /    \ /       / ",
    r"     _  |(o) (o)|  _   [ The ultimate sc-timebench mascot, the Crispy Fishstick! ]",
    r"    ( ) |   ^  | ( )   \_________________________________________________________/",
    r"     ¯  |  ~~  |  ¯                        ",
    r"       / \____/ \                         ",
    r"      '          '                        ",
    r"     CRISPY FISHSTICK                      ",
]


def animate():
    block_interrupts()

    try:
        cols, rows = os.get_terminal_size()
    except OSError:
        cols, rows = (80, 24)

    art_width = max(len(line) for line in fishstick_art)
    speed = 0.02

    # Move Right to Left
    # We add art_width to the range so it starts fully off-screen right
    for i in range(cols, -art_width, -1):
        # Move cursor to Home
        output = ["\033[H"]
        # Add top padding
        output.append("\n" * (rows // 3))

        for line in fishstick_art:
            if i >= 0:
                # Leading spaces + the art
                full_line = (" " * i) + line
            else:
                # Sliced art for when it's exiting left
                full_line = line[abs(i) :]

            # THE FIX: Pad the end of the line with spaces to clear old ']' and '/'
            # Then truncate to 'cols' to prevent wrapping
            cleaned_line = full_line.ljust(cols)[:cols]
            output.append(cleaned_line)

        sys.stdout.write("\n".join(output))
        sys.stdout.flush()
        time.sleep(speed)

    # Clean clear-up and release the user
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()
    print("\n   [ SC-TIMEBENCH RESUMING... ]\n")
    restore_interrupts()
