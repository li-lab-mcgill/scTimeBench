"""
main.py. Entrypoint for measuring trajectories in single-cell data,
particularly involving gene regulatory networks and cell lineage information.
"""

from scTimeBench.config import Config, CsvExportType, CsvWriteMode

# required to register metrics
import scTimeBench.metrics
import scTimeBench.shared.dataset

# plotting import
from scTimeBench.plotting import Plotting

if False:
    scTimeBench.metrics  # to avoid unused import warning
    scTimeBench.shared.dataset  # to avoid unused import warning

from scTimeBench.metrics.base import METRIC_REGISTRY, BaseMetric
from scTimeBench.metrics.method_manager import MethodManager
from scTimeBench.shared.dataset.base import DATASET_REGISTRY

from pprint import pprint
import os
from pathlib import Path

import scTimeBench.database as database


def print_available(config: Config):
    """
    Print available datasets and metrics.
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


def view_evals_by_method(config: Config):
    """
    View evaluations grouped by method.
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
        method = MethodManager(config, dataset)
        print(
            f"""
----------------------------------------------------------------------------------------------------
Evals for method: {method._get_name()}
Dataset: {method.dataset.get_name()}, {method.dataset.encode_dataset_dict()}, {method.dataset.encode_preprocessors()}
Metadata: {method._encode_metadata()}"""
        )

        evals = db_manager.get_evals_per_method(method)
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


def iterate_csv_types(config: Config):
    db_stem = Path(config.database_path).stem
    merge_mode = config.csv_write_mode == CsvWriteMode.MERGE
    for csv_type in config.to_csv if config.to_csv is not None else CsvExportType:
        yield csv_type, os.path.join(
            config.csv_dir,
            f"{csv_type.value}.csv"
            if merge_mode
            else f"{db_stem}_{csv_type.value}.csv",
        )


def to_csv(config: Config):
    os.makedirs(config.csv_dir, exist_ok=True)
    merge_mode = config.csv_write_mode == CsvWriteMode.MERGE
    if config.to_csv is not None:
        db_manager = database.DatabaseManager(config)
        for csv_type, output_file in iterate_csv_types(config):
            if csv_type == CsvExportType.GRAPH_SIM:
                db_manager.graph_sim_to_csv(output_file, append=merge_mode)
            if csv_type == CsvExportType.EMBEDDING:
                db_manager.embedding_to_csv(output_file, append=merge_mode)
            if csv_type == CsvExportType.GEX_PRED:
                db_manager.gex_pred_to_csv(output_file, append=merge_mode)
        db_manager.close()


def plot(config: Config):
    # now let's plot if needed
    plotting = Plotting(config)
    for csv_type, output_file in iterate_csv_types(config):
        if csv_type == CsvExportType.GRAPH_SIM:
            plotting.plot_graph_sim_from_csv(output_file)
        # TODO: finish these next steps!
        # if csv_type == CsvExportType.EMBEDDING:
        # plotting.plot_embedding_from_csv(output_file)
        # if csv_type == CsvExportType.GEX_PRED:
        # plotting.plot_gex_pred_from_csv(output_file)


def main():
    """
    Main entrypoint for the scTimeBench (crispy-fishstick) package.
    """
    config = Config()

    exit_on_output = [
        config.available,
        config.print_all,
        config.to_csv is not None,
        config.plot_from_csv,
        config.view_evals_by_method,
        config.view_evals_by_metric,
        config.clear_tables,
    ]

    if config.available:
        print_available(config)

    if config.print_all:
        db_manager = database.DatabaseManager(config)
        db_manager.print_all()
        db_manager.close()

    if config.to_csv is not None:
        to_csv(config)
    if config.plot_from_csv:
        plot(config)

    if config.view_evals_by_method:
        view_evals_by_method(config)

    if config.view_evals_by_metric:
        view_evals_by_metric(config)

    if config.clear_tables:
        db_manager = database.DatabaseManager(config)
        db_manager.clear_tables()
        print("All database tables have been cleared.")
        db_manager.close()

    if any(exit_on_output):
        print(
            "Output generated based on provided flags. Exiting without running metrics."
        )
        exit()

    run_metrics(config)


if __name__ == "__main__":
    main()
