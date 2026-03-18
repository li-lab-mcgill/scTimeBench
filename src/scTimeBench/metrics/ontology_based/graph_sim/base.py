"""
Graph Similarity Metric Base Class
"""
from scTimeBench.metrics.ontology_based.base import OntologyBasedMetrics
from scTimeBench.metrics.ontology_based.graph_sim.utils import (
    floyd_warshall,
    modified_floyd_warshall,
)
from scTimeBench.shared.constants import RequiredOutputFiles
from scTimeBench.shared.helpers import parse_cell_lineage
from scTimeBench.shared.dataset.filters.lineage import LineageDatasetFilter
from scTimeBench.shared.dataset.base import BaseDataset
from scTimeBench.shared.dataset.filters.pseudotime_filter import (
    BasePseudotimeFilter,
)
from enum import Enum
from sklearn.metrics import roc_curve, precision_recall_curve
from scTimeBench.trajectory_infer.base import TrajectoryInferenceMethodFactory
import numpy as np
import logging
import os
import json


class AdjacencyMatrixType:
    UNWEIGHTED = "adjacency_matrix"
    WEIGHTED = "weighted_adjacency_matrix"


class ThresholdCriteria(Enum):
    ALL_PATHS = "all_paths"  # consider after the Floyd-Warshall
    SIMPLE = "simple"  # just consider the adjacency matrix entries


class GraphSimMetric(OntologyBasedMetrics):
    def _defaults(self):
        return {
            "threshold_criterion": [
                ThresholdCriteria.ALL_PATHS.value,
                ThresholdCriteria.SIMPLE.value,
            ],
            "auto_threshold": True,
            "edge_threshold": 0.1,
            "from_tp_zero": False,
            # by default we choose PRC threshold if auto_threshold is on
            "prc_threshold": True,
        }

    def _setup_trajectory_inference_model(self):
        traj_infer_config = self.metric_config.get("trajectory_infer_model", {})

        assert (
            "from_tp_zero" not in traj_infer_config
            or traj_infer_config["from_tp_zero"] == self.params["from_tp_zero"]
        ), "from_tp_zero in trajectory inference config must either not be defined, or match from_tp_zero in metric config."
        traj_infer_config["from_tp_zero"] = self.params["from_tp_zero"]

        self.trajectory_infer_model = (
            TrajectoryInferenceMethodFactory().get_trajectory_infer_method(
                traj_infer_config
            )
        )
        self.params["trajectory_infer_model"] = str(self.trajectory_infer_model)

    def _setup_method_output_requirements(self):
        # ** NOTE: must define the following attributes **
        # where we define the output embedding name
        # as well as the required features and outputs
        if self.trajectory_infer_model.uses_gene_expr():
            primary_outputs = (
                [
                    RequiredOutputFiles.NEXT_TIMEPOINT_GENE_EXPRESSION,
                ]
                if not self.params["from_tp_zero"]
                else [
                    RequiredOutputFiles.FROM_ZERO_TO_END_PRED_GEX,
                ]
            )
        else:
            assert not self.params["from_tp_zero"], (
                "from_tp_zero can only be True if the "
                "trajectory inference model uses gene expression."
            )
            primary_outputs = [
                RequiredOutputFiles.EMBEDDING,
                RequiredOutputFiles.NEXT_TIMEPOINT_EMBEDDING,
            ]

        # allow alternate outputs for OT-based methods
        self.required_outputs = [
            primary_outputs,
            [RequiredOutputFiles.NEXT_CELLTYPE],
            # option for correlation metric: give the pred graph directly
            [RequiredOutputFiles.PRED_GRAPH],
        ]

    def _build_ref_graph(self, dataset: BaseDataset):
        """
        Build the reference graph from the cell lineage tree.
        """
        # iterate over the filters until you find the one that contains the lineage information
        # we will assume there is only one such filter
        cell_lineage = None

        for filter in dataset.dataset_filters:
            if isinstance(filter, LineageDatasetFilter):
                if cell_lineage is not None:
                    raise ValueError(
                        "Multiple LineageDatasetFilter found in dataset filters."
                    )
                cell_lineage = parse_cell_lineage(
                    filter.cell_lineage_file, filter.cell_equivalence_file
                )

        if cell_lineage is None:
            raise ValueError("No LineageDatasetFilter found in dataset filters.")

        self.cell_lineage = cell_lineage

        logging.debug(f"Found cell lineage: {cell_lineage}")
        # from this cell lineage, let's:
        # 1) create an index of the cell types (so that we can keep track of the cell types to ids mapping)
        # 2) create the adjacency matrix representation of the graph
        cell_types = set()
        for source, targets in cell_lineage.items():
            cell_types.add(source)
            for target in targets:
                cell_types.add(target)

        cell_type_to_id = {
            cell_type: idx for idx, cell_type in enumerate(sorted(cell_types))
        }
        num_cell_types = len(cell_types)
        logging.debug(f"Cell type to ID mapping: {cell_type_to_id}")

        # initialize adjacency matrix
        adjacency_matrix = np.zeros((num_cell_types, num_cell_types), dtype=np.float32)
        for source, targets in cell_lineage.items():
            source_id = cell_type_to_id[source]
            for target in targets:
                target_id = cell_type_to_id[target]
                adjacency_matrix[source_id, target_id] = 1.0

        logging.debug(f"Reference graph adjacency matrix:\n{adjacency_matrix}")
        return adjacency_matrix, cell_type_to_id

    def _build_pred_graph_with_threshold(self, weighted_adjacency_matrix, threshold):
        """
        Returns a binarized adjacency matrix based on the provided threshold.
        """
        num_cell_types = weighted_adjacency_matrix.shape[0]
        adjacency_matrix = np.zeros((num_cell_types, num_cell_types), dtype=np.float32)
        # now we can iterate over the predicted trajectory to fill in the adjacency matrix
        for source_id in range(num_cell_types):
            for target_id in range(num_cell_types):
                prob = weighted_adjacency_matrix[source_id, target_id]
                # avoid self-loops and any edges below the threshold
                if source_id != target_id and prob >= threshold:
                    adjacency_matrix[source_id, target_id] = 1.0

        return adjacency_matrix

    def _build_pred_graph(self, output_path, cell_type_to_id):
        # we need to check if they provided a predicted graph, then we ignore the trajectory
        # inference step, and directly use the provided predicted graph for evaluation.
        if RequiredOutputFiles.PRED_GRAPH.value in os.listdir(output_path):
            logging.debug(
                "Found predicted graph in output, skipping trajectory inference step."
            )
            pred_graph_path = os.path.join(
                output_path, RequiredOutputFiles.PRED_GRAPH.value
            )
            weighted_adjacency_matrix = np.load(pred_graph_path)
            # no special trajectory dir needed here
            self.traj_dir = output_path
            return weighted_adjacency_matrix

        # otherwise, we need to go through the trajectory inference step
        traj_dir, _ = self.trajectory_infer_model._get_traj_infer_path(output_path)
        self.traj_dir = traj_dir

        # first let's ensure that it's in the right format
        # we expect it to have the true embeddings and predicted embeddings
        # for timepoints (1, ..., n) in separate output files
        pred_trajectory = self.trajectory_infer_model.infer_trajectory(output_path)

        logging.debug(f"Predicted trajectory: {pred_trajectory}")

        # based on the predicted trajectory, we can build the adjacency matrix
        # let's use the config's threshold for an edge to be created
        num_cell_types = len(cell_type_to_id)
        weighted_adjacency_matrix = np.zeros(
            (num_cell_types, num_cell_types), dtype=np.float32
        )

        # now we can iterate over the predicted trajectory to fill in the adjacency matrix
        for source_cell_type, target_distribution in pred_trajectory.items():
            source_id = cell_type_to_id[source_cell_type]

            for target_cell_type, prob in target_distribution.items():
                target_id = cell_type_to_id[target_cell_type]
                weighted_adjacency_matrix[source_id, target_id] = prob

        return weighted_adjacency_matrix

    def _prepare_final_graphs(
        self, weighted_adjacency_matrix, unweighted_ref, cell_type_to_id
    ):
        """
        Prepares the final predicted graphs based on the weighted adjacency matrix
        and the reference unweighted adjacency matrix, using different thresholding
        criteria if specified.
        """
        # now we find the best threshold if it's automatically found, otherwise
        # we use the provided one
        pred_graphs = []
        ref_graphs = []
        thresholds = []
        criterions = []

        for criteria in self.params["threshold_criterion"]:
            # first get the threshold that we want
            if self.auto_threshold:
                # TODO: this is not expensive because we have relatively small n, but in the future
                # TODO: it would be good to cache the result from these threshold calculations!
                # now, using the AUROC curve, we calculate the best threshold
                # using Youden's J statistic.
                if criteria == ThresholdCriteria.ALL_PATHS.value:
                    pred_paths = modified_floyd_warshall(weighted_adjacency_matrix)
                    ref_paths = (floyd_warshall(unweighted_ref) < np.inf).astype(int)
                    threshold = self._calculate_best_threshold(pred_paths, ref_paths)
                elif criteria == ThresholdCriteria.SIMPLE.value:
                    threshold = self._calculate_best_threshold(
                        weighted_adjacency_matrix, unweighted_ref
                    )
                else:
                    raise ValueError(f"Invalid threshold criterion: {criteria}")
            else:
                threshold = self.edge_threshold

            # then build the predicted and reference graphs based on this threshold
            pred_graph = self._build_pred_graph_with_threshold(
                weighted_adjacency_matrix, threshold
            ).astype(int)
            if criteria == ThresholdCriteria.ALL_PATHS.value:
                logging.debug(f"Using ALL_PATHS criterion with threshold: {threshold}")
                pred_graph = (floyd_warshall(pred_graph) < np.inf).astype(int)
                ref_graph = (floyd_warshall(unweighted_ref) < np.inf).astype(int)
            elif criteria == ThresholdCriteria.SIMPLE.value:
                logging.debug(f"Using SIMPLE criterion with threshold: {threshold}")
                ref_graph = unweighted_ref

            logging.debug(f"Threshold: {threshold}")
            logging.debug(f"Predicted Unweighted: {pred_graph}")
            logging.debug(f"Predicted Weighted: {weighted_adjacency_matrix}")
            logging.debug(f"Reference Graph: {ref_graph}")

            pred_graphs.append(
                {
                    AdjacencyMatrixType.UNWEIGHTED: pred_graph,
                    AdjacencyMatrixType.WEIGHTED: weighted_adjacency_matrix,
                }
            )
            ref_graphs.append(
                {
                    AdjacencyMatrixType.UNWEIGHTED: ref_graph,
                }
            )
            thresholds.append(threshold)
            criterions.append(criteria)

        # let's print out what the predicted trajectory (with thresholding) looks like
        # so we need to map the adjacency matrix back to cell types
        if self.config.log_level == "DEBUG":
            ids_to_cell_types = {v: k for k, v in cell_type_to_id.items()}
            for pred_graph in pred_graphs:
                pred_lineage = {}
                for i in range(pred_graph[AdjacencyMatrixType.UNWEIGHTED].shape[0]):
                    for j in range(pred_graph[AdjacencyMatrixType.UNWEIGHTED].shape[1]):
                        if pred_graph[AdjacencyMatrixType.UNWEIGHTED][i, j] == 1.0:
                            source_cell_type = ids_to_cell_types[i]
                            target_cell_type = ids_to_cell_types[j]
                            if source_cell_type not in pred_lineage:
                                pred_lineage[source_cell_type] = []
                            pred_lineage[source_cell_type].append(target_cell_type)
                logging.debug(
                    f"Predicted cell lineage (after thresholding): {pred_lineage}"
                )

        return pred_graphs, ref_graphs, thresholds, criterions

    def _calculate_best_threshold(self, pred, ref):
        """
        Based on the number of true positives and false negatives,
        chooses the best threshold that maximizes Youden's J statistic (sensitivity + specificity - 1).

        However, we want to choose it such that it is not at (1, 1) because that is trivial.
        In this case, we select the best threshold besides this one.
        """
        logging.debug(
            f"Calculating threshold for {ref}, {pred} using {'prc' if self.params['prc_threshold'] else 'roc'} curve."
        )
        if self.params["prc_threshold"]:
            precision, recall, thresholds = precision_recall_curve(
                ref.flatten().astype(int), pred.flatten()
            )
            j_scores = precision + recall
            best_idx = j_scores.argmax()
            logging.debug(
                f"Selected best threshold at (precision, recall): ({precision[best_idx]}, {recall[best_idx]}) with threshold: {thresholds[best_idx]}"
            )
        else:
            fpr, tpr, thresholds = roc_curve(ref.flatten(), pred.flatten())
            j_scores = tpr - fpr
            best_idx = j_scores.argmax()
            logging.debug(
                f"Selected best threshold at (fpr, tpr): ({fpr[best_idx]}, {tpr[best_idx]}) with threshold: {thresholds[best_idx]}"
            )
        return thresholds[best_idx]

    def _prep_kwargs_for_submetric_eval(self, output_path, dataset, method):
        self.output_path = output_path
        self.dataset_dir = dataset.get_dataset_dir()
        self.dataset_name = dataset.get_name()
        self.time_label = "Time"
        for dataset_filter in dataset.dataset_filters:
            if isinstance(dataset_filter, BasePseudotimeFilter):
                self.time_label = dataset_filter.label()
                break

        # first build the original reference and predicted graphs
        graph_ref, cell_type_to_id = self._build_ref_graph(dataset)
        self.cell_type_to_id = cell_type_to_id
        weighted_adjacency_matrix = self._build_pred_graph(
            output_path, self.cell_type_to_id
        )

        # then based on the threshold criteria, we build both the all-paths/simple
        # and with original thresholds or not
        graph_preds, graph_refs, thresholds, criterions = self._prepare_final_graphs(
            weighted_adjacency_matrix, graph_ref, self.cell_type_to_id
        )
        return {
            "graphs": [
                {
                    "graph_preds": pred_graph,
                    "graph_refs": ref_graph,
                    "threshold": threshold,
                    "criterion": criterion,
                }
                for (pred_graph, ref_graph, threshold, criterion) in zip(
                    graph_preds, graph_refs, thresholds, criterions
                )
            ],
            "method": method,
        }

    def _submetric_eval(self, graphs, method):
        """
        Wrapper function to call the graph similarity evaluation, and handle database
        logging.
        """
        # graph pred is made up of an array of adjacency matrices
        for graph_dict in graphs:
            graph_pred = graph_dict["graph_preds"]
            graph_ref = graph_dict["graph_refs"]
            self.threshold = graph_dict["threshold"]
            eval = self._graph_sim_eval(graph_pred, graph_ref, graph_dict["criterion"])
            if eval is not None:
                eval = json.dumps(
                    {
                        "eval": eval,
                        "threshold": str(graph_dict.get("threshold")),
                        "criteria": graph_dict.get("criterion"),
                    }
                )
                self.db_manager.insert_eval(
                    method, self.__class__.__name__, self._get_param_encoding(), eval
                )

    def _graph_sim_eval(self, graph_pred, graph_ref, criteria):
        raise NotImplementedError("Subclasses should implement this method.")
