"""
Graph Similarity Metric Base Class
"""
from src.metrics.base import BaseMetric


class GraphSimMetric(BaseMetric):
    def __init__(self, name: str, dataset: str):
        super().__init__(name, dataset=dataset)

    def eval(self, graph1, graph2):
        """
        The graph similarity metrics we will be using will take in
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def build_reference_objects(self):
        """
        Build the reference graph object needed for graph similarity metrics
        """
