from crispy_fishstick.metrics.ontology_based.graph_sim.base import AdjacencyMatrixType
from crispy_fishstick.metrics.ontology_based.graph_sim.ged import GraphEditDistance
from crispy_fishstick.metrics.ontology_based.graph_sim.jaccard_similarity import (
    JaccardSimilarity,
)
from crispy_fishstick.metrics.ontology_based.graph_sim.confusion_matrix import (
    GraphPrecision,
    GraphRecall,
    GraphF1,
)
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

    assert wrapper_graph_sim_eval(
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
    ) == np.sqrt(2), "Graph Edit Distance calculation is incorrect."

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


def test_jaccard_similarity():
    """
    Tests that the Jaccard Similarity metric gives expected results.
    """
    js = JaccardSimilarity(None, None, {})

    def wrapper_graph_sim_eval(adj_pred, adj_ref):
        return js._graph_sim_eval(
            {
                AdjacencyMatrixType.UNWEIGHTED: adj_pred,
            },
            {
                AdjacencyMatrixType.UNWEIGHTED: adj_ref,
            },
        )

    # Test identical graphs
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
                    [0, 0, 1],
                    [0, 0, 0],
                ]
            ),
        )
        == 1.0
    ), "Jaccard Similarity for identical graphs should be 1.0."

    # Test completely different graphs
    assert (
        wrapper_graph_sim_eval(
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                ]
            ),
        )
        == 0.0
    ), "Jaccard Similarity for completely different graphs should be 0.0."

    # Test partial overlap
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
        == 0.5
    ), "Jaccard Similarity with 1 common edge and 2 total edges should be 0.5."


def test_graph_precision():
    """
    Tests that the Graph Precision metric gives expected results.
    Precision = TP / (TP + FP)
    """
    precision = GraphPrecision(None, None, {})

    def wrapper_graph_sim_eval(adj_pred, adj_ref):
        return precision._graph_sim_eval(
            {
                AdjacencyMatrixType.UNWEIGHTED: adj_pred,
            },
            {
                AdjacencyMatrixType.UNWEIGHTED: adj_ref,
            },
        )

    # Test perfect precision (all predicted edges are correct)
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
        == 1.0
    ), "Precision for perfect prediction should be 1.0."

    # Test zero precision (no predicted edges are correct)
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
                    [0, 0],
                    [1, 0],
                ]
            ),
        )
        == 0.0
    ), "Precision with no correct predictions should be 0.0."

    # Test partial precision (1 TP, 1 FP out of 2 predicted edges)
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
                    [0, 0, 0],
                ]
            ),
        )
        == 0.5
    ), "Precision with 1 TP and 1 FP should be 0.5."


def test_graph_recall():
    """
    Tests that the Graph Recall metric gives expected results.
    Recall = TP / (TP + FN)
    """
    recall = GraphRecall(None, None, {})

    def wrapper_graph_sim_eval(adj_pred, adj_ref):
        return recall._graph_sim_eval(
            {
                AdjacencyMatrixType.UNWEIGHTED: adj_pred,
            },
            {
                AdjacencyMatrixType.UNWEIGHTED: adj_ref,
            },
        )

    # Test perfect recall (all reference edges are predicted)
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
        == 1.0
    ), "Recall for perfect prediction should be 1.0."

    # Test zero recall (no reference edges are predicted)
    assert (
        wrapper_graph_sim_eval(
            np.array(
                [
                    [0, 0],
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
        == 0.0
    ), "Recall with no correct predictions should be 0.0."

    # Test partial recall (1 TP, 1 FN out of 2 reference edges)
    assert (
        wrapper_graph_sim_eval(
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0],
                ]
            ),
        )
        == 0.5
    ), "Recall with 1 TP and 1 FN should be 0.5."


def test_graph_f1():
    """
    Tests that the Graph F1 metric gives expected results.
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    f1 = GraphF1(None, None, {})

    def wrapper_graph_sim_eval(adj_pred, adj_ref):
        return f1._graph_sim_eval(
            {
                AdjacencyMatrixType.UNWEIGHTED: adj_pred,
            },
            {
                AdjacencyMatrixType.UNWEIGHTED: adj_ref,
            },
        )

    # Test perfect F1 (identical graphs)
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
        == 1.0
    ), "F1 for identical graphs should be 1.0."

    # Test zero F1 (no overlap)
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
                    [0, 0],
                    [1, 0],
                ]
            ),
        )
        == 0.0
    ), "F1 with no correct predictions should be 0.0."

    # Test F1 with equal precision and recall
    # (1 TP, 1 FP, 1 FN -> precision=0.5, recall=0.5, F1=0.5)
    assert (
        wrapper_graph_sim_eval(
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0],
                ]
            ),
        )
        == 2.0 / 3.0
    ), "F1 with precision=1.0 and recall=0.5 should be 0.67."
