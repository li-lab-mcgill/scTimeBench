from crispy_fishstick.metrics.ontology_based.graph_sim.base import (
    GraphSimMetric,
    AdjacencyMatrixType,
)

import numpy as np
import logging
import json


# TODO: add some unit tests for this metric!
class AverageShortestPathDiff(GraphSimMetric):
    def _floyd_warshall(self, adj_matrix):
        """
        Compute shortest paths using Floyd-Warshall algorithm.

        Args:
            adj_matrix: Adjacency matrix where 0 means no edge

        Returns:
            Distance matrix with shortest paths
        """
        n = adj_matrix.shape[0]

        # Initialize distance matrix
        # Set to infinity where there's no edge, keep actual weights otherwise
        dist = np.full((n, n), np.inf)

        # Set distances based on adjacency matrix
        for i in range(n):
            for j in range(n):
                if i == j:
                    dist[i][j] = 0
                elif adj_matrix[i][j] > 0:
                    dist[i][j] = adj_matrix[i][j]

        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        return dist

    def _graph_sim_eval(self, graph_pred, graph_ref):
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
        ref_shortest_paths = self._floyd_warshall(graph_ref_adj)
        logging.debug(f"Reference shortest paths:\n{ref_shortest_paths}")

        # Step 2: Compute Floyd-Warshall on predicted graph
        pred_shortest_paths = self._floyd_warshall(graph_pred_adj)
        logging.debug(f"Predicted shortest paths:\n{pred_shortest_paths}")

        # Step 3: Create masks from both the reference and predicted shortest path matrices
        ref_invalid_mask = np.isinf(ref_shortest_paths)
        pred_invalid_mask = np.isinf(pred_shortest_paths)

        # then calculate the number of false positives -- paths that exist in predicted but not in reference
        false_positives = np.sum(ref_invalid_mask & ~pred_invalid_mask).item()
        # and the number of false negatives -- paths that exist in reference but not in predicted
        false_negatives = np.sum(~ref_invalid_mask & pred_invalid_mask).item()

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

        return json.dumps(
            {
                "false_positives": false_positives,
                "avg_diff": avg_diff,
                "false_negatives": false_negatives,
                "overlap_paths": overlap_paths,
                # total paths should just be from reference, the total number of actual paths
                # you can take
                "total_paths": np.sum(~ref_invalid_mask).item()
                - ref_shortest_paths.shape[0],  # exclude self-paths
            }
        )
