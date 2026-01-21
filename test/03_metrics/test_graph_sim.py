from crispy_fishstick.metrics.ontology_based.graph_sim.base import AdjacencyMatrixType
from crispy_fishstick.metrics.ontology_based.graph_sim.ged import GraphEditDistance
import numpy as np


def test_graph_edit_distance():
    """
    Tests that the Graph Edit Distance metric gives expected results.
    """
    ged = GraphEditDistance(None, None, {})

    def wrapper_graph_sim_eval(adj_pred, adj_ref):
        return ged._graph_sim_eval(
            {
                AdjacencyMatrixType.UNWEIGHTED: adj_pred,
            },
            {
                AdjacencyMatrixType.UNWEIGHTED: adj_ref,
            },
        )

    assert (
        wrapper_graph_sim_eval(
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                ]
            ),
        )
        == 2
    ), "Graph Edit Distance calculation is incorrect."

    assert (
        wrapper_graph_sim_eval(
            np.array(
                [
                    [0, 1],
                    [0, 0],
                ]
            ),
            np.array(
                [
                    [0, 1],
                    [0, 0],
                ]
            ),
        )
        == 0
    ), "Graph Edit Distance for identical graphs should be zero."

    assert (
        wrapper_graph_sim_eval(
            np.array(
                [
                    [0, 1],
                    [0, 0],
                ]
            ),
            np.array(
                [
                    [0, 1],
                    [1, 0],
                ]
            ),
        )
        == 1
    ), "Graph Edit Distance is incorrect for single edge difference."
