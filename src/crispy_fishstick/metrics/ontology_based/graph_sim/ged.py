from crispy_fishstick.metrics.ontology_based.graph_sim.base import (
    GraphSimMetric,
    AdjacencyMatrixType,
)

import networkx as nx
import logging


class GraphEditDistance(GraphSimMetric):
    def _graph_sim_eval(self, graph_pred, graph_ref):
        """
        The graph similarity metrics we will be using will take in
        """
        graph_pred_unweighted = graph_pred[AdjacencyMatrixType.UNWEIGHTED]
        graph_ref_adj = graph_ref[AdjacencyMatrixType.UNWEIGHTED]

        G_pred = nx.from_numpy_array(graph_pred_unweighted, create_using=nx.DiGraph)
        G_ref = nx.from_numpy_array(graph_ref_adj, create_using=nx.DiGraph)

        for i in G_pred.nodes:
            G_pred.nodes[i]["id"] = i
        for i in G_ref.nodes:
            G_ref.nodes[i]["id"] = i

        edit_distance = nx.graph_edit_distance(
            G_pred,
            G_ref,
            node_match=lambda n1, n2: n1["id"] == n2["id"],
        )

        logging.debug(
            f"Predicted Graph: {graph_pred_unweighted}, Reference Graph: {graph_ref_adj}"
        )
        logging.debug(f"Graph Edit Distance: {edit_distance}")
        return edit_distance
