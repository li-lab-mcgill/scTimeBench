from scTimeBench.metrics.ontology_based.graph_sim.base import (
    GraphSimMetric,
    AdjacencyMatrixType,
)
from scTimeBench.metrics.ontology_based.graph_sim.utils import floyd_warshall
from scTimeBench.metrics.base import skip_metric

import numpy as np
import logging


# TODO: add some unit tests for this metric!
# skipping this metric because it doesn't really tell you much...
@skip_metric
class AverageShortestPathDiff(GraphSimMetric):
    def _graph_sim_eval(self, graph_pred, graph_ref, criteria):
        """
        Calculate average shortest path difference between predicted and reference graphs.

        Steps:
        1. Calculate shortest paths in reference graph using Floyd-Warshall
        2. Calculate shortest paths in predicted graph
        3. Calculate the number of false positives (paths in predicted but not in reference) and false negatives (paths in reference but not in predicted)
        4. Compute average difference on valid entries, outputting total number of valid entries
        """
        graph_pred_adj = graph_pred[AdjacencyMatrixType.UNWEIGHTED].astype(np.float32)
        graph_ref_adj = graph_ref[AdjacencyMatrixType.UNWEIGHTED].astype(np.float32)

        # Step 1: Compute Floyd-Warshall on reference graph
        ref_shortest_paths = floyd_warshall(graph_ref_adj)
        logging.debug(f"Reference shortest paths:\n{ref_shortest_paths}")

        # Step 2: Compute Floyd-Warshall on predicted graph
        pred_shortest_paths = floyd_warshall(graph_pred_adj)
        logging.debug(f"Predicted shortest paths:\n{pred_shortest_paths}")

        # Step 3: Create masks from both the reference and predicted shortest path matrices
        ref_invalid_mask = np.isinf(ref_shortest_paths)
        pred_invalid_mask = np.isinf(pred_shortest_paths)

        # Step 4: Calculate the average shortest path difference only on valid entries
        both_valid = ~ref_invalid_mask & ~pred_invalid_mask
        ref_shortest_paths[~both_valid] = np.inf
        pred_shortest_paths[~both_valid] = np.inf
        logging.debug(f"Reference shortest paths (masked):\n{ref_shortest_paths}")
        logging.debug(f"Predicted shortest paths (masked):\n{pred_shortest_paths}")

        # Compute absolute differences and average only on valid predicted paths
        overlap_paths = (
            np.sum(both_valid).item() - ref_shortest_paths.shape[0]
        )  # exclude self-paths
        logging.debug(
            f"Number of overlapping valid paths for difference calculation: {overlap_paths}"
        )

        # finally, compute their average difference
        differences = []
        for i in range(ref_shortest_paths.shape[0]):
            for j in range(ref_shortest_paths.shape[1]):
                if i == j:
                    continue  # skip self-paths

                if both_valid[i][j]:
                    differences.append(
                        np.abs(pred_shortest_paths[i][j] - ref_shortest_paths[i][j])
                    )

        assert (
            len(differences) == overlap_paths
        ), "Mismatch in counted overlapping paths and collected differences."

        # Average difference
        if len(differences) > 0:
            avg_diff = np.mean(differences).item()
        else:
            avg_diff = 0.0

        logging.debug(f"Average Shortest Path Difference: {avg_diff:.4f}")

        return avg_diff
