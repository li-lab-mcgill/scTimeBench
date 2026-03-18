from scTimeBench.metrics.ontology_based.graph_sim.base import (
    GraphSimMetric,
    AdjacencyMatrixType,
    ThresholdCriteria,
)
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
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


def get_threshold_prc(graph_weighted_pred_adj, graph_ref_adj, title, output_file):
    """
    Gets the threshold PRC curve and AUC-PRC score for the predicted graph
    against the reference graph, treating the weighted adjacency matrix as
    predicted probabilities and the reference adjacency matrix as true labels.
    """
    # now let's get the auc roc score using the weighted adjacency matrix as the predicted probabilities
    # and the reference adjacency matrix as the true labels
    # let's first do some more preprocessing onto the weighted, where we get rid of the self loops
    for i in range(graph_weighted_pred_adj.shape[0]):
        graph_weighted_pred_adj[i, i] = 0.0

    auc_prc = average_precision_score(
        graph_ref_adj.flatten().astype(int),
        graph_weighted_pred_adj.flatten(),
    )

    logging.debug(f"Graph AUC PRC: {auc_prc}")

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    PrecisionRecallDisplay.from_predictions(
        graph_ref_adj.flatten().astype(int),
        graph_weighted_pred_adj.flatten(),
        plot_chance_level=True,
    )
    plt.title(title)
    plt.savefig(output_file)
    logging.debug(f"Saved ROC curve to {output_file}")
    return auc_prc


class GraphClassificationReport(GraphSimMetric):
    def _graph_sim_eval(self, graph_pred, graph_ref, criteria):
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
                    title=f"Threshold ROC Curve of {self.config.method['name']} on {self.dataset_name}{'(All Paths)' if criteria == ThresholdCriteria.ALL_PATHS.value else ''}",
                    output_file=self.traj_dir + f"/roc_curve_{criteria}.png",
                ),
                "auc_prc": get_threshold_prc(
                    graph_weighted_pred_adj,
                    graph_ref_adj,
                    title=f"Threshold PRC Curve of {self.config.method['name']} on {self.dataset_name}{'(All Paths)' if criteria == ThresholdCriteria.ALL_PATHS.value else ''}",
                    output_file=self.traj_dir + f"/prc_curve_{criteria}.png",
                ),
            }
        )
