"""
main.py. Entrypoint for measuring trajectories in single-cell data,
particularly involving gene regulatory networks and cell lineage information.
"""

from crispy_fishstick.config import Config

# required to register metrics
import crispy_fishstick.metrics
import crispy_fishstick.shared.dataset

if False:
    crispy_fishstick.metrics  # to avoid unused import warning
    crispy_fishstick.shared.dataset  # to avoid unused import warning

from crispy_fishstick.metrics.base import METRIC_REGISTRY, BaseMetric
from crispy_fishstick.metrics.model_manager import ModelManager
from crispy_fishstick.shared.dataset.base import DATASET_REGISTRY

from pprint import pprint

import crispy_fishstick.database as database


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
        if hasattr(metric_inst, "required_outputs"):
            print(f"   Required Outputs: {metric_inst.required_outputs}")
        if hasattr(metric_inst, "supported_datasets"):
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


def main():
    """
    Main entrypoint for the crispy-fishstick package.
    """
    config = Config()

    if config.available:
        print_available(config)
        exit()

    if config.print_all:
        db_manager = database.DatabaseManager(config)
        db_manager.print_all()
        db_manager.close()
        exit()

    if config.graph_sim_to_csv:
        db_manager = database.DatabaseManager(config)
        db_manager.graph_sim_to_csv(config.output_csv_path)
        db_manager.close()
        exit()

    if config.view_evals_by_model:
        view_evals_by_model(config)
        exit()

    if config.view_evals_by_metric:
        view_evals_by_metric(config)
        exit()

    if config.clear_tables:
        db_manager = database.DatabaseManager(config)
        db_manager.clear_tables()
        print("All database tables have been cleared.")
        db_manager.close()
        exit()

    run_metrics(config)


if __name__ == "__main__":
    main()
