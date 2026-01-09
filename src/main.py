"""
main.py. Entrypoint for measuring trajectories in single-cell data,
particularly involving gene regulatory networks and cell lineage information.
"""
from config import Config, METRIC_REGISTRY

# required to register metrics
import metrics

if False:
    metrics  # to avoid unused import warning


def run_metrics(config: Config):
    """
    Run the specified metrics based on the provided configuration.
    """
    for metric_name in config.metrics:
        if metric_name not in METRIC_REGISTRY:
            raise ValueError(f"Metric {metric_name} not found in registry.")
        metric_class = METRIC_REGISTRY[metric_name]
        metric_instance = metric_class(config=config)
        metric_instance.eval()


if __name__ == "__main__":
    config = Config()
    run_metrics(config)
