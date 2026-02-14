from crispy_fishstick.metrics.ontology_based.graph_sim.base import (
    GraphSimMetric,
    AdjacencyMatrixType,
)

import numpy as np
import logging


class GraphEditDistance(GraphSimMetric):
    def _defaults(self):
        return {
            **super()._defaults(),
            "weighted": False,
            "norm": None,  # by default we use the Frobenius norm
        }

    def _graph_sim_eval(self, graph_pred, graph_ref):
        """
        The graph similarity metrics we will be using will take in
        """
        graph_pred_adj = (
            graph_pred[AdjacencyMatrixType.WEIGHTED]
            if self.weighted
            else graph_pred[AdjacencyMatrixType.UNWEIGHTED]
        )
        graph_ref_adj = graph_ref[AdjacencyMatrixType.UNWEIGHTED]

        # it's quite simply the difference between the predicted and reference adjacency matrices
        if self.norm or self.norm == "frobenius":
            edit_distance = np.linalg.norm(
                graph_pred_adj - graph_ref_adj, ord=self.norm
            ).item()
        else:
            edit_distance = np.linalg.norm(graph_pred_adj - graph_ref_adj).item()

        logging.debug(
            f"Predicted Graph: {graph_pred_adj}, Reference Graph: {graph_ref_adj}"
        )
        logging.debug(f"Graph Edit Distance: {edit_distance}")
        return edit_distance
