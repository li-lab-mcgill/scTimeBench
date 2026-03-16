import numpy as np


def floyd_warshall(adj_matrix):
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


def modified_floyd_warshall(adj_matrix):
    """
    A modified version of Floyd-Warshall that is meant to compute the maximum
    minimum probability path between all pairs of nodes, rather than the shortest path.
    This is useful for the area under threshold curve, where different thresholds
    will affect the graph that we get out.

    See: https://en.wikipedia.org/wiki/Widest_path_problem for more information.
    """
    n = adj_matrix.shape[0]
    prob_matrix = adj_matrix.copy()

    for k in range(n):
        for i in range(n):
            for j in range(n):
                # if the prob between i and k or k and j is 0, then it won't create a path
                # otherwise, we take the max of the existing probability and the min of the two new paths through k
                prob_matrix[i, j] = max(
                    prob_matrix[i, j], min(prob_matrix[i, k], prob_matrix[k, j])
                )

    return prob_matrix
