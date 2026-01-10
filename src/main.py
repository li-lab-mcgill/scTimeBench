"""
main.py. Entrypoint for measuring trajectories in single-cell data,
particularly involving gene regulatory networks and cell lineage information.
"""
from config import Config

# required to register metrics
import metrics
import dataset
import models

if False:
    metrics  # to avoid unused import warning
    dataset  # to avoid unused import warning
    models  # to avoid unused import warning

from metrics.base import METRIC_REGISTRY
from models.base import MODEL_REGISTRY
from dataset.base import DATASET_REGISTRY

import database


def print_available():
    """
    Print available models, datasets, and metrics.
    """
    print("Available Models:")
    for model_name in MODEL_REGISTRY.keys():
        print(f" - {model_name}")

    print("\nAvailable Datasets:")
    for dataset_name in DATASET_REGISTRY.keys():
        print(f" - {dataset_name}")

    print("\nAvailable Metrics:")
    for metric_name in METRIC_REGISTRY.keys():
        print(f" - {metric_name}")


def run_metrics(config: Config):
    """
    Run the specified metrics based on the provided configuration.
    """
    # initialize the database connection
    db_manager = database.DatabaseManager(config)

    for metric_name in config.metrics:
        if metric_name not in METRIC_REGISTRY:
            raise ValueError(f"Metric {metric_name} not found in registry.")
        metric_class = METRIC_REGISTRY[metric_name]
        metric_instance = metric_class(config=config, db_manager=db_manager)
        metric_instance.eval()

    db_manager.close()


if __name__ == "__main__":
    config = Config()

    if config.available:
        print_available()
        exit()

    run_metrics(config)
