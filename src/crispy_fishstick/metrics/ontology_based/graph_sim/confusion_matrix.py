from crispy_fishstick.metrics.ontology_based.graph_sim.base import (
    GraphSimMetric,
    AdjacencyMatrixType,
)

import numpy as np
import logging


class GraphAccuracy(GraphSimMetric):
    def _graph_sim_eval(self, graph_pred, graph_ref):
        """
        The graph similarity metrics we will be using will take in
        """
        graph_pred_adj = graph_pred[AdjacencyMatrixType.UNWEIGHTED]
        graph_ref_adj = graph_ref[AdjacencyMatrixType.UNWEIGHTED]

        # calculate the accuracy as the number of correct edges over total edges
        correct_edges = np.sum(graph_pred_adj == graph_ref_adj)
        total_edges = graph_ref_adj.shape[0] * graph_ref_adj.shape[1]
        accuracy = correct_edges / total_edges

        logging.debug(f"Graph Accuracy: {accuracy}")
        return accuracy


class GraphPrecision(GraphSimMetric):
    def _graph_sim_eval(self, graph_pred, graph_ref):
        """
        Calculate precision for edge prediction.
        Precision = TP / (TP + FP)
        """
        graph_pred_adj = graph_pred[AdjacencyMatrixType.UNWEIGHTED]
        graph_ref_adj = graph_ref[AdjacencyMatrixType.UNWEIGHTED]

        # Convert to binary
        pred_binary = (graph_pred_adj > 0).astype(int)
        ref_binary = (graph_ref_adj > 0).astype(int)

        # Calculate true positives and false positives
        tp = np.sum(pred_binary & ref_binary)
        fp = np.sum(pred_binary & ~ref_binary)

        # Handle edge case where no edges are predicted
        if tp + fp == 0:
            precision = 1.0 if tp == 0 else 0.0
        else:
            precision = tp / (tp + fp)

        logging.debug(f"Graph Precision: {precision}")
        return precision


class GraphRecall(GraphSimMetric):
    def _graph_sim_eval(self, graph_pred, graph_ref):
        """
        Calculate recall for edge prediction.
        Recall = TP / (TP + FN)
        """
        graph_pred_adj = graph_pred[AdjacencyMatrixType.UNWEIGHTED]
        graph_ref_adj = graph_ref[AdjacencyMatrixType.UNWEIGHTED]

        # Convert to binary
        pred_binary = (graph_pred_adj > 0).astype(int)
        ref_binary = (graph_ref_adj > 0).astype(int)

        # Calculate true positives and false negatives
        tp = np.sum(pred_binary & ref_binary)
        fn = np.sum(~pred_binary & ref_binary)

        # Handle edge case where no reference edges exist
        if tp + fn == 0:
            recall = 1.0 if tp == 0 else 0.0
        else:
            recall = tp / (tp + fn)

        logging.debug(f"Graph Recall: {recall}")
        return recall


class GraphF1(GraphSimMetric):
    def _graph_sim_eval(self, graph_pred, graph_ref):
        """
        Calculate F1 score for edge prediction.
        F1 = 2 * (precision * recall) / (precision + recall)
        """
        graph_pred_adj = graph_pred[AdjacencyMatrixType.UNWEIGHTED]
        graph_ref_adj = graph_ref[AdjacencyMatrixType.UNWEIGHTED]

        # Convert to binary
        pred_binary = (graph_pred_adj > 0).astype(int)
        ref_binary = (graph_ref_adj > 0).astype(int)

        # Calculate TP, FP, FN
        tp = np.sum(pred_binary & ref_binary)
        fp = np.sum(pred_binary & ~ref_binary)
        fn = np.sum(~pred_binary & ref_binary)

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Calculate F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        logging.debug(f"Graph F1: {f1}")
        return f1
