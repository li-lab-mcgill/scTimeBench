"""
Base class for all metrics. They should all implement the eval method, and
depend on the dataset that they belong to.
"""
from config import Config, register_metric


# also store a registry of metrics of name to class
class BaseMetric:
    def __init__(self, config: Config):
        self.config = config
        self.populate_feature_specs()

    def __init_subclass__(cls):
        """
        Automatically register subclasses in the METRIC_REGISTRY.

        This allows for hierarchical structuring of metrics, where
        subclasses are tracked under their parent classes.
        """
        register_metric(cls)

        # we want to clear the submetrics list for each subclass
        cls.submetrics = []

        for base in cls.__bases__:
            if hasattr(base, "submetrics"):
                base.submetrics.append(cls)

    def eval(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

    def populate_feature_specs(self):
        """
        Populate the feature specifications required for the metric.
        """
        raise NotImplementedError("Subclasses should implement this method.")
