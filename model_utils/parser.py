import argparse
import os
import pickle
import sys
import yaml
from shared.constants import RequiredOutputColumns

# ** The following is needed for src/config imports!**
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


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

        # let's set the required columns properly
        self.required_outputs = [
            RequiredOutputColumns(output) for output in self.config["required_outputs"]
        ]

    def train(self, ann_data):
        raise NotImplementedError("Subclasses should implement this method.")

    def generate(self, test_ann_data, expected_output_path):
        raise NotImplementedError("Subclasses should implement this method.")


def main(model_class: BaseModel):
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
    train_ann_data, test_ann_data = yaml_config["dataset"].load_data()

    # Initialize the model
    model: BaseModel = model_class(yaml_config)

    print(f"Training and/or loading the model: {model_class.__name__}")
    # let's let the train() function handle the caching as well
    model.train(train_ann_data)
    print("Training/loading complete.")

    # Generate samples -- we'll move the saving of generated samples outside of this script
    print(f"Starting generation to {model_output_path}")
    model.generate(test_ann_data, expected_output_path=model_output_path)
    print("Generation complete.")
