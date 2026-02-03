"""
Note: for this file only, this will be used by other models as a base class
And so its context is outside the src/ folder, so we need to use crispy_fishstick.*
imports instead of relative imports.
"""

import argparse
import os
import pickle
import yaml
import scanpy as sc

from crispy_fishstick.shared.constants import RequiredOutputColumns
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

    paths = ["dataset_pkl_path", "output_path", "output_file_name"]
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

        # normalize required outputs: list or list of lists
        raw_required_outputs = self.config["required_outputs"]
        if not raw_required_outputs:
            raise ValueError("required_outputs must not be empty.")

        if all(isinstance(item, list) for item in raw_required_outputs):
            required_output_options = [
                [RequiredOutputColumns(output) for output in option]
                for option in raw_required_outputs
            ]
        else:
            required_output_options = [
                [RequiredOutputColumns(output) for output in raw_required_outputs]
            ]

        self.required_outputs_options = required_output_options

        # select the option that has NEXT_CELLTYPE if it's an OT method
        if self.is_ot_method():
            for option in required_output_options:
                if RequiredOutputColumns.NEXT_CELLTYPE in option:
                    self.required_outputs = option
                    break

            if not hasattr(self, "required_outputs"):
                print(
                    f"Warning: OT method but no NEXT_CELLTYPE in required outputs. Using first option."
                )
                self.required_outputs = required_output_options[0]

    def train(self, ann_data, all_tps=None):
        raise NotImplementedError("Subclasses should implement this method.")

    def generate(self, test_ann_data, expected_output_path):
        raise NotImplementedError("Subclasses should implement this method.")

    def is_ot_method(self) -> bool:
        """
        Check if the model is an OT-based method. Defaults to False.
        Subclasses representing OT methods should override this method to return True.
        """
        return False


def main(model_class: BaseModel):
    print(f"Starting train and testing for model...")
    parser = get_parser()
    args = parser.parse_args()
    yaml_config = process_yaml(args.yaml_config)

    # if the model outputs already exist, then we skip generation
    model_output_path = os.path.join(
        yaml_config["output_path"], yaml_config["output_file_name"]
    )
    if os.path.exists(model_output_path):
        print(
            f'Generated samples found at {os.path.join(yaml_config["output_path"], yaml_config["output_file_name"])}. Skipping generation.'
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

    # Initialize the model
    model: BaseModel = model_class(yaml_config)

    print(f"Training and/or loading the model: {model_class.__name__}")
    # let's let the train() function handle the caching as well
    model.train(train_ann_data, all_tps=all_tps)
    print("Training/loading complete.")

    # Generate samples -- we'll move the saving of generated samples outside of this script
    print(f"Starting generation to {model_output_path}")
    model.generate(test_ann_data, expected_output_path=model_output_path)
    print("Generation complete.")

    # verify that the output file was created, and that it contains the required columns
    if not os.path.exists(model_output_path):
        raise RuntimeError(f"Model output file was not created at {model_output_path}")
    print(f"Verifying generated output at {model_output_path}")

    # TODO: add generate and train under a try catch which will clean the model output path if anything fails
    # load the ann data and check for required columns
    generated_ann_data = sc.read_h5ad(model_output_path)
    for required_output in model.required_outputs:
        if required_output.value not in generated_ann_data.obsm.keys():
            # delete the generated file to avoid confusion
            os.remove(model_output_path)
            raise RuntimeError(
                f"Generated output missing required column: {required_output.value}"
            )
