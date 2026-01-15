from metrics.ontology_based.graph_sim.base import GraphSimMetric, AdjacencyMatrixType

import networkx as nx


class GraphEditDistance(GraphSimMetric):
    def _graph_sim_eval(self, graph_pred, graph_ref):
        """
        The graph similarity metrics we will be using will take in
        """
        graph_pred_unweighted = graph_pred[AdjacencyMatrixType.UNWEIGHTED]
        return nx.graph_edit_distance(
            nx.from_numpy_array(graph_pred_unweighted),
            nx.from_numpy_array(graph_ref),
        )
