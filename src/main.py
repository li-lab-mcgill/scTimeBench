"""
main.py. Entrypoint for measuring trajectories in single-cell data,
particularly involving gene regulatory networks and cell lineage information.
"""
# first let's add the model utils path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from config import Config

# required to register metrics
import metrics
import shared.dataset

if False:
    metrics  # to avoid unused import warning
    shared.dataset  # to avoid unused import warning

from metrics.base import METRIC_REGISTRY
from shared.dataset.base import DATASET_REGISTRY

import database


def print_available(config: Config):
    """
    Print available models, datasets, and metrics.
    """
    print("\nAvailable Datasets:")
    for dataset_name in DATASET_REGISTRY.keys():
        print(f" - {dataset_name}")

    print("\nAvailable Metrics:")
    for metric_name in METRIC_REGISTRY.keys():
        metric_inst = METRIC_REGISTRY[metric_name](config, None)
        print(f" - {metric_name}")
        print(f"   Required Feature Specs: {metric_inst.required_feature_specs}")
        print(f"   Supported datasets: {metric_inst.supported_datasets}")


def run_metrics(config: Config):
    """
    Run the specified metrics based on the provided configuration.
    """
    # initialize the database connection
    db_manager = database.DatabaseManager(config)

    for metric in config.metrics:
        metric_name = metric["name"]
        if metric_name not in METRIC_REGISTRY:
            raise ValueError(f"Metric {metric_name} not found in registry.")
        metric_class = METRIC_REGISTRY[metric_name]
        metric_instance = metric_class(
            config=config, db_manager=db_manager, metric_config=metric
        )
        metric_instance.eval()

    db_manager.close()


if __name__ == "__main__":
    config = Config()

    if config.available:
        print_available(config)
        exit()

    if config.print_all:
        db_manager = database.DatabaseManager(config)
        db_manager.print_all()
        db_manager.close()
        exit()

    run_metrics(config)
