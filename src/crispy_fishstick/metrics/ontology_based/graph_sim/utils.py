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
