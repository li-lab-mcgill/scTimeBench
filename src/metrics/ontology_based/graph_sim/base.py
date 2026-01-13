"""
Graph Similarity Metric Base Class
"""
from metrics.base import BaseMetric, OutputPathName, FeatureSpec
from shared.dataset.registry import SuoDataset, GarciaAlonsoDataset


class GraphSimMetric(BaseMetric):
    def __init__(self, config, db_manager):
        super().__init__(config, db_manager)

        # ** NOTE: must define the following three attributes **
        # ** Particularly, we require a filter builder which is a set of functions **
        # ** that take in a dataset_dict and return a dataset filter instance. **
        self.required_feature_specs = [FeatureSpec.TRAJECTORY]
        self.output_path_name = OutputPathName.GRAPH_SIM
        # this needs to be the name of the class
        self.supported_datasets = [
            SuoDataset.__name__,
            GarciaAlonsoDataset.__name__,
        ]

    def _eval(self, output_path):
        """
        The graph similarity metrics we will be using will take in
        """
        # TODO: build the predicted and reference graphs from the given data
        print(f"Model outputs are found: {output_path}")
        self.graph_pred = None
        self.graph_ref = None

        if self.submetrics:
            for submetric in self.submetrics:
                submetric_instance = submetric(self.config)
                submetric_instance._graph_sim_eval(self.graph_pred, self.graph_ref)
        else:
            self._graph_sim_eval(self.graph_pred, self.graph_ref)

    def _graph_sim_eval(self, graph_pred, graph_ref):
        raise NotImplementedError("Subclasses should implement this method.")
