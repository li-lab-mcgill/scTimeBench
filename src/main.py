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

from metrics.base import METRIC_REGISTRY, BaseMetric
from metrics.model_manager import ModelManager
from shared.dataset.base import DATASET_REGISTRY

from pprint import pprint

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
        metric_inst = METRIC_REGISTRY[metric_name](config, None, {})
        print(f" - {metric_name}")
        print(f"   Required Outputs: {metric_inst.required_outputs}")
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


def view_evals_by_model(config: Config):
    """
    View evaluations grouped by model.
    """
    db_manager = database.DatabaseManager(config)

    # define a single metric to get the evals
    simple_metric = config.metrics[0]
    metric_name = simple_metric["name"]
    if metric_name not in METRIC_REGISTRY:
        raise ValueError(f"Metric {metric_name} not found in registry.")

    metric_class = METRIC_REGISTRY[metric_name]
    metric_instance: BaseMetric = metric_class(
        config=config, db_manager=db_manager, metric_config=simple_metric
    )

    # now instead of evaluating, we will just view the evals:
    for dataset in metric_instance.datasets:
        model = ModelManager(config, dataset)
        print(
            f"""
----------------------------------------------------------------------------------------------------
Evals for Model: {model._get_name()}
Dataset: {model.dataset.get_name()}, {model.dataset.encode_dataset_dict()}, {model.dataset.encode_filters()}
Metadata: {model._encode_metadata()}"""
        )

        evals = db_manager.get_evals_per_model(model)
        print(f" Metric: {metric_name} with params {metric_instance.params}")
        for eval in evals:
            pprint(eval)

    db_manager.close()


def view_evals_by_metric(config: Config):
    """
    View evaluations grouped by metric.
    """
    db_manager = database.DatabaseManager(config)
    # for each of the metrics defined, view their existing evals
    for metric in config.metrics:
        metric_name = metric["name"]
        if metric_name not in METRIC_REGISTRY:
            raise ValueError(f"Metric {metric_name} not found in registry.")
        metric_class = METRIC_REGISTRY[metric_name]
        metric_instance: BaseMetric = metric_class(
            config=config, db_manager=db_manager, metric_config=metric
        )

        # now instead of evaluating, we will just view the evals:
        evals = db_manager.get_evals_per_metric(
            metric_instance.__class__.__name__, metric_instance._get_param_encoding()
        )
        print(f"\nEvals for Metric: {metric_name} with params {metric_instance.params}")
        for eval in evals:
            pprint(eval)

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

    if config.view_evals_by_model:
        view_evals_by_model(config)
        exit()

    if config.view_evals_by_metric:
        view_evals_by_metric(config)
        exit()

    run_metrics(config)
