"""
Note: for this file only, this will be used by other models as a base class
And so its context is outside the src/ folder, so we need to use crispy_fishstick.*
imports instead of relative imports.
"""

import argparse
import os
import pickle
import yaml
import numpy as np
import pandas as pd

from crispy_fishstick.shared.constants import RequiredOutputFiles
from crispy_fishstick.shared.constants import ObservationColumns


def get_parser():
    # parser that will read the input data path and the model output path
    parser = argparse.ArgumentParser(description="Train ExampleRandomSampler model.")
    parser.add_argument(
        "--yaml_config", type=str, help="Path to YAML configuration file"
    )
    return parser


def process_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    paths = ["dataset_pkl_path", "output_path"]
    for path in paths:
        if path not in yaml_config:
            raise ValueError(f"YAML configuration missing required path: {path}")

        # verify the data paths exist:
        if path in [
            "output_path",
            "dataset_pkl_path",
        ] and not os.path.exists(yaml_config[path]):
            raise FileNotFoundError(f"Data file not found: {yaml_config[path]}")

    # now let's load the dataset from the pickled path
    with open(yaml_config["dataset_pkl_path"], "rb") as f:
        yaml_config["dataset"] = pickle.load(f)

    return yaml_config


# Class to be inherited by all models
class BaseModel:
    def __init__(self, yaml_config):
        self.config = yaml_config
        self.output_path = yaml_config["output_path"]

        # normalize required outputs: list or list of lists
        raw_required_outputs = self.config["required_outputs"]
        if not raw_required_outputs:
            raise ValueError("required_outputs must not be empty.")

        if all(isinstance(item, list) for item in raw_required_outputs):
            required_output_options = [
                [RequiredOutputFiles(output) for output in option]
                for option in raw_required_outputs
            ]
        else:
            required_output_options = [
                [RequiredOutputFiles(output) for output in raw_required_outputs]
            ]

        self.required_outputs_options = required_output_options

        # by default since we are not an OT method, we just select the option without NEXT_CELLTYPE
        for option in required_output_options:
            if RequiredOutputFiles.NEXT_CELLTYPE not in option:
                self.required_outputs = option
                break

        print(f"Required outputs: {self.required_outputs}")

    def train(self, ann_data, all_tps=None):
        raise NotImplementedError("Subclasses should implement this method.")

    def generate(self, test_ann_data):
        """
        Main generation method that dispatches to individual output generators.
        Each output is saved to its own file under self.output_path.
        """
        for required_output in self.required_outputs:
            output_file = os.path.join(self.output_path, required_output.value)
            if os.path.exists(output_file):
                print(f"Output file {output_file} already exists, skipping generation.")
                continue

            print(f"Generating {required_output.value}...")
            if required_output == RequiredOutputFiles.EMBEDDING:
                result = self.generate_embedding(test_ann_data)
                np.save(output_file, result)
            elif required_output == RequiredOutputFiles.NEXT_TIMEPOINT_EMBEDDING:
                result = self.generate_next_tp_embedding(test_ann_data)
                np.save(output_file, result)
            elif required_output == RequiredOutputFiles.NEXT_TIMEPOINT_GENE_EXPRESSION:
                result = self.generate_next_tp_gex(test_ann_data)
                np.save(output_file, result)
            elif required_output == RequiredOutputFiles.NEXT_CELLTYPE:
                result = self.generate_next_cell_type(test_ann_data)
                # result should be a pandas DataFrame or Series
                result.to_parquet(output_file)
            else:
                raise ValueError(f"Unknown required output: {required_output}")

            print(f"Saved {required_output.value} to {output_file}")

    def generate_embedding(self, test_ann_data) -> np.ndarray:
        """
        Generate embeddings for the current timepoint.
        Returns: np.ndarray of shape (n_cells, embedding_dim)
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def generate_next_tp_embedding(self, test_ann_data) -> np.ndarray:
        """
        Generate embeddings for the next timepoint.
        Returns: np.ndarray of shape (n_cells, embedding_dim)
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def generate_next_tp_gex(self, test_ann_data) -> np.ndarray:
        """
        Generate gene expression for the next timepoint.
        Returns: np.ndarray of shape (n_cells, n_genes)
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def generate_next_cell_type(self, test_ann_data) -> pd.DataFrame:
        """
        Generate next cell type predictions.
        Returns: pd.DataFrame with cell type predictions
        """
        raise NotImplementedError("Subclasses should implement this method.")


def main(model_class: BaseModel):
    print(f"Starting train and testing for model...")
    parser = get_parser()
    args = parser.parse_args()
    yaml_config = process_yaml(args.yaml_config)

    output_path = yaml_config["output_path"]

    # Initialize the model
    model: BaseModel = model_class(yaml_config)

    # first let's check if the required outputs already exist -- and skip the whole process if so
    if all(
        [
            os.path.exists(os.path.join(output_path, required_output.value))
            for required_output in model.required_outputs
        ]
    ):
        print(
            "All required output files already exist, skipping training and generation."
        )
        return

    # Otherwise we have to load the data and train/test the model
    print("Loading dataset...")
    train_ann_data, test_ann_data = yaml_config["dataset"].load_data()

    # Some methods map the tps to indices, argument all used for pertinent methods.
    # Providing it to train argument for processing to be handled within the subclasses.
    time_col = ObservationColumns.TIMEPOINT.value
    all_tps = (
        train_ann_data.obs[time_col].unique().tolist()
        + test_ann_data.obs[time_col].unique().tolist()
    )
    all_tps = list(set(all_tps))

    print(f"Training and/or loading the model: {model_class.__name__}")
    # let's let the train() function handle the caching as well
    model.train(train_ann_data, all_tps=all_tps)
    print("Training/loading complete.")

    # Generate outputs - each required output saved to its own file
    print(f"Starting generation to {output_path}")
    model.generate(test_ann_data)
    print("Generation complete.")

    # Verify that all required output files were created
    print(f"Verifying generated outputs at {output_path}")
    for required_output in model.required_outputs:
        output_file = os.path.join(output_path, required_output.value)
        if not os.path.exists(output_file):
            raise RuntimeError(f"Required output file was not created: {output_file}")
        print(f"    Found {required_output.value}")
