"""
Graph Similarity Metric Base Class
"""
from metrics.base import BaseMetric


class GraphSimMetric(BaseMetric):
    def eval(self):
        """
        The graph similarity metrics we will be using will take in
        """
        if self.submetrics:
            for submetric in self.submetrics:
                submetric_instance = submetric(self.config)
                submetric_instance._eval(self.graph_pred, self.graph_ref)
        else:
            self._eval(self.graph_pred, self.graph_ref)

    def populate_feature_specs(self):
        """
        Populate the feature specifications required for graph similarity metrics
        """
        self.graph_pred = None
        self.graph_ref = None

    def _eval(self, graph_pred, graph_ref):
        raise NotImplementedError("Subclasses should implement this method.")
