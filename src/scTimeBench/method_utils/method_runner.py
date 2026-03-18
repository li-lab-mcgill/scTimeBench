"""
Note: for this file only, this will be used by other methods as a base class
And so its context is outside the src/ folder, so we need to use scTimeBench.*
imports instead of relative imports.
"""

import argparse
import os
import pickle
import yaml
import numpy as np
import pandas as pd
import scanpy as sc

from scTimeBench.shared.constants import RequiredOutputFiles
from scTimeBench.shared.constants import ObservationColumns


def get_parser():
    # parser that will read the input data path and the method output path
    parser = argparse.ArgumentParser(description="Train ExampleRandomSampler method.")
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


# Class to be inherited by all methods
class BaseMethod:
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

            print(f"Generating {required_output.value}...", flush=True)
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
            elif required_output == RequiredOutputFiles.PRED_GRAPH:
                result = self.generate_pred_graph(test_ann_data)
                np.save(output_file, result)
            elif required_output == RequiredOutputFiles.FROM_ZERO_TO_END_PRED_GEX:
                # first let's preprocess test_ann_data to only provide the first timepoint
                # then we also require that the resultant ann data file has (tps - 1) * n_cells from tp0
                # and that it populates the column ObservationColumns.TIMEPOINT with the correct time
                first_tp = test_ann_data.obs[ObservationColumns.TIMEPOINT.value].min()
                first_tp_cells = test_ann_data[
                    test_ann_data.obs[ObservationColumns.TIMEPOINT.value] == first_tp
                ].copy()

                all_tps = sorted(
                    test_ann_data.obs[ObservationColumns.TIMEPOINT.value].unique()
                )

                result = self.generate_zero_to_end_pred_gex(first_tp_cells, all_tps)
                # result should be an AnnData object
                # now check that the result has the correct shape and the correct timepoints in the obs column
                expected_n_cells = len(first_tp_cells) * len(all_tps)
                assert (
                    result.shape[0] == expected_n_cells
                ), f"Expected {expected_n_cells} cells in the result, but got {result.shape[0]}"
                result_tps = sorted(
                    result.obs[ObservationColumns.TIMEPOINT.value].unique()
                )
                assert (
                    result_tps == all_tps
                ), f"Expected timepoints {all_tps} in the result, but got {result_tps}"
                result.write_h5ad(output_file)
            else:
                raise ValueError(f"Unknown required output: {required_output}")

            print(f"Saved {required_output.value} to {output_file}", flush=True)

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

    def generate_pred_graph(self, test_ann_data) -> np.ndarray:
        """
        Generate predicted graph.
        Returns: np.ndarray representing the predicted graph
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def generate_zero_to_end_pred_gex(self, first_tp_cells, all_tps) -> sc.AnnData:
        """
        Generate predicted gene expression from the first to the last timepoint.
        Returns: AnnData object with predicted gene expression across all timepoints
        """
        raise NotImplementedError("Subclasses should implement this method.")


def main(method_class: BaseMethod):
    print(f"Starting train and testing for method...")
    parser = get_parser()
    args = parser.parse_args()
    yaml_config = process_yaml(args.yaml_config)

    output_path = yaml_config["output_path"]

    # Initialize the method
    method: BaseMethod = method_class(yaml_config)

    # first let's check if the required outputs already exist -- and skip the whole process if so
    if all(
        [
            os.path.exists(os.path.join(output_path, required_output.value))
            for required_output in method.required_outputs
        ]
    ):
        print(
            "All required output files already exist, skipping training and generation."
        )
        return

    # Otherwise we have to load the data and train/test the method
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

    print(f"Training and/or loading the method: {method_class.__name__}", flush=True)
    # let's let the train() function handle the caching as well
    method.train(train_ann_data, all_tps=all_tps)
    print("Training/loading complete.")

    # Generate outputs - each required output saved to its own file
    print(f"Starting generation to {output_path}", flush=True)
    method.generate(test_ann_data)
    print("Generation complete.", flush=True)

    # Verify that all required output files were created
    print(f"Verifying generated outputs at {output_path}")
    for required_output in method.required_outputs:
        output_file = os.path.join(output_path, required_output.value)
        if not os.path.exists(output_file):
            raise RuntimeError(f"Required output file was not created: {output_file}")
        print(f"    Found {required_output.value}")
