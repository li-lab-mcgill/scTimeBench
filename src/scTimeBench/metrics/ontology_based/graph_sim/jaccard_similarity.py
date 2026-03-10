from scTimeBench.metrics.ontology_based.graph_sim.base import (
    GraphSimMetric,
    AdjacencyMatrixType,
)

import numpy as np
import logging


class JaccardSimilarity(GraphSimMetric):
    def _graph_sim_eval(self, graph_pred, graph_ref, criteria):
        """
        Calculate Jaccard similarity between predicted and reference graphs.
        Jaccard similarity = |intersection| / |union|
        """
        graph_pred_adj = graph_pred[AdjacencyMatrixType.UNWEIGHTED]
        graph_ref_adj = graph_ref[AdjacencyMatrixType.UNWEIGHTED]

        # Convert to binary (0 or 1)
        pred_binary = (graph_pred_adj > 0).astype(int)
        ref_binary = (graph_ref_adj > 0).astype(int)

        # Calculate intersection and union
        intersection = np.sum(pred_binary & ref_binary)
        union = np.sum(pred_binary | ref_binary)

        # Handle edge case where union is 0
        if union == 0:
            jaccard_sim = 1.0 if intersection == 0 else 0.0
        else:
            jaccard_sim = intersection / union

        logging.debug(f"Jaccard Similarity: {jaccard_sim}")
        return jaccard_sim
