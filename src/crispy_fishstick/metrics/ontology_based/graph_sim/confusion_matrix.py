from crispy_fishstick.metrics.ontology_based.graph_sim.base import (
    GraphSimMetric,
    AdjacencyMatrixType,
    DATASET_NAME_KEY,
)
from crispy_fishstick.metrics.ontology_based.graph_sim.utils import floyd_warshall
from sklearn.metrics import classification_report, roc_auc_score

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import numpy as np
import logging
import json


def get_confusion_metrics(graph_pred_adj, graph_ref_adj):
    # flatten the adjacency matrices
    report = classification_report(
        graph_ref_adj.flatten().astype(int),
        graph_pred_adj.flatten().astype(int),
        output_dict=True,
    )

    # just keep all the fields in '1' and the overall accuracy:
    report = {
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
    }
    logging.debug(f"Classification Report:\n{report}")
    return report


def get_threshold_roc(graph_weighted_pred_adj, graph_ref_adj, title, output_file):
    """
    Gets the threshold ROC curve and AUC-ROC score for the predicted graph
    against the reference graph, treating the weighted adjacency matrix as
    predicted probabilities and the reference adjacency matrix as true labels.
    """
    # now let's get the auc roc score using the weighted adjacency matrix as the predicted probabilities
    # and the reference adjacency matrix as the true labels
    # let's first do some more preprocessing onto the weighted, where we get rid of the self loops
    for i in range(graph_weighted_pred_adj.shape[0]):
        graph_weighted_pred_adj[i, i] = 0.0

    auc_roc = roc_auc_score(
        graph_ref_adj.flatten().astype(int),
        graph_weighted_pred_adj.flatten(),
    )
    logging.debug(f"Graph AUC ROC: {auc_roc}")

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    RocCurveDisplay.from_predictions(
        graph_ref_adj.flatten().astype(int), graph_weighted_pred_adj.flatten()
    )
    plt.title(title)
    plt.savefig(output_file)
    logging.debug(f"Saved ROC curve to {output_file}")

    return auc_roc


class GraphClassificationReport(GraphSimMetric):
    def _graph_sim_eval(self, graph_pred, graph_ref):
        """
        The graph similarity metrics we will be using will take in
        """
        graph_weighted_pred_adj = graph_pred[AdjacencyMatrixType.WEIGHTED]
        graph_pred_adj = graph_pred[AdjacencyMatrixType.UNWEIGHTED]
        graph_ref_adj = graph_ref[AdjacencyMatrixType.UNWEIGHTED]

        logging.debug(f"Predicted paths:\n{graph_pred_adj}")
        logging.debug(f"Weighted predicted paths:\n{graph_weighted_pred_adj}")
        logging.debug(f"Reference paths:\n{graph_ref_adj}")

        return json.dumps(
            {
                **get_confusion_metrics(graph_pred_adj, graph_ref_adj),
                "auc_roc": get_threshold_roc(
                    graph_weighted_pred_adj,
                    graph_ref_adj,
                    title=f"Threshold ROC Curve of {self.config.model['name']} on {graph_ref[DATASET_NAME_KEY]}",
                    output_file=graph_pred["output_path"] + "/roc_curve.png",
                ),
            }
        )


class GraphClassificationReportAllPaths(GraphSimMetric):
    def _graph_sim_eval(self, graph_pred, graph_ref):
        """
        This metric calculates the confusion matrix metrics (accuracy, precision, recall, F1) not just on the direct edges in the predicted and reference graphs, but also on all paths between cell types. So if there is a path from A to B in the reference graph but not in the predicted graph, that would count as a false negative.
        """
        graph_weighted_pred_adj = graph_pred[AdjacencyMatrixType.WEIGHTED]
        graph_ref_adj = graph_ref[AdjacencyMatrixType.UNWEIGHTED]

        # we can use Floyd-Warshall to calculate the transitive closure of both graphs, which will give us all paths between cell types
        # binarize the paths for this
        graph_pred_paths = (floyd_warshall(graph_weighted_pred_adj) < np.inf).astype(
            int
        )
        graph_ref_paths = (floyd_warshall(graph_ref_adj) < np.inf).astype(int)

        graph_weighted_pred_paths = self._modified_floyd_warshall(
            graph_weighted_pred_adj
        )

        logging.debug(f"Predicted paths (transitive closure):\n{graph_pred_paths}")
        logging.debug(
            f"Weighted predicted paths (modified Floyd-Warshall):\n{graph_weighted_pred_paths}"
        )
        logging.debug(f"Reference paths (transitive closure):\n{graph_ref_paths}")

        return json.dumps(
            {
                **get_confusion_metrics(graph_pred_paths, graph_ref_paths),
                "auc_roc": get_threshold_roc(
                    graph_weighted_pred_paths,
                    graph_ref_paths,
                    title=f"All Paths Threshold ROC Curve of {self.config.model['name']} on {graph_ref[DATASET_NAME_KEY]}",
                    output_file=graph_pred["output_path"] + "/all_paths_roc_curve.png",
                ),
            }
        )

    def _modified_floyd_warshall(self, adj_matrix):
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
