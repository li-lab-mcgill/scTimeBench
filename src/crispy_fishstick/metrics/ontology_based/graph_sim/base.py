"""
Graph Similarity Metric Base Class
"""
from crispy_fishstick.metrics.ontology_based.base import OntologyBasedMetrics
from crispy_fishstick.shared.constants import RequiredOutputFiles
from crispy_fishstick.shared.helpers import parse_cell_lineage
from crispy_fishstick.shared.dataset.filters.lineage import LineageDatasetFilter
from crispy_fishstick.shared.dataset.base import BaseDataset
from crispy_fishstick.shared.dataset.filters.pseudotime_filter import (
    BasePseudotimeFilter,
)
from crispy_fishstick.trajectory_infer.base import TrajectoryInferenceMethodFactory
import numpy as np
import logging


class AdjacencyMatrixType:
    UNWEIGHTED = "adjacency_matrix"
    WEIGHTED = "weighted_adjacency_matrix"


CELL_TYPE_TO_ID_KEY = "cell_type_to_id"


class GraphSimMetric(OntologyBasedMetrics):
    def _defaults(self):
        return {
            "edge_threshold": 0.1,
        }

    def _setup_trajectory_inference_model(self):
        self.trajectory_infer_model = (
            TrajectoryInferenceMethodFactory().get_trajectory_infer_method(
                self.metric_config.get("trajectory_infer_model", {})
            )
        )
        self.params["trajectory_infer_model"] = str(self.trajectory_infer_model)

    def _setup_model_output_requirements(self):
        # ** NOTE: must define the following attributes **
        # where we define the output embedding name
        # as well as the required features and outputs
        if self.trajectory_infer_model.uses_gene_expr():
            primary_outputs = [
                RequiredOutputFiles.NEXT_TIMEPOINT_GENE_EXPRESSION,
            ]
        else:
            primary_outputs = [
                RequiredOutputFiles.EMBEDDING,
                RequiredOutputFiles.NEXT_TIMEPOINT_EMBEDDING,
            ]

        # allow alternate outputs for OT-based methods
        self.required_outputs = [
            primary_outputs,
            [RequiredOutputFiles.NEXT_CELLTYPE],
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
        return {
            AdjacencyMatrixType.UNWEIGHTED: adjacency_matrix,
            CELL_TYPE_TO_ID_KEY: cell_type_to_id,
        }

    def _build_pred_graph(self, output_path, cell_type_to_id):
        """
        Builds the predicted graph structure based on the output.
        """
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
        adjacency_matrix = np.zeros((num_cell_types, num_cell_types), dtype=np.float32)

        # now we can iterate over the predicted trajectory to fill in the adjacency matrix
        for source_cell_type, target_distribution in pred_trajectory.items():
            source_id = cell_type_to_id[source_cell_type]

            for target_cell_type, prob in target_distribution.items():
                target_id = cell_type_to_id[target_cell_type]
                weighted_adjacency_matrix[source_id, target_id] = prob

                # avoid self-loops and any edges below the threshold
                if source_id != target_id and prob >= self.edge_threshold:
                    adjacency_matrix[source_id, target_id] = 1.0

        # let's print out what the predicted trajectory (with thresholding) looks like
        # so we need to map the adjacency matrix back to cell types
        pred_lineage = {}
        ids_to_cell_types = {v: k for k, v in cell_type_to_id.items()}

        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j] == 1.0:
                    source_cell_type = ids_to_cell_types[i]
                    target_cell_type = ids_to_cell_types[j]
                    if source_cell_type not in pred_lineage:
                        pred_lineage[source_cell_type] = []
                    pred_lineage[source_cell_type].append(target_cell_type)

        logging.debug(f"Predicted cell lineage (after thresholding): {pred_lineage}")

        return {
            AdjacencyMatrixType.WEIGHTED: weighted_adjacency_matrix,
            AdjacencyMatrixType.UNWEIGHTED: adjacency_matrix,
        }

    def _prep_kwargs_for_submetric_eval(self, output_path, dataset, model):
        graph_ref = self._build_ref_graph(dataset)
        self.output_path = output_path

        traj_dir, classifier_dir = self.trajectory_infer_model._get_traj_infer_path(
            output_path
        )
        self.traj_dir = traj_dir
        self.classifier_dir = classifier_dir

        self.dataset_name = dataset.get_name()
        self.time_label = "Time"
        for dataset_filter in dataset.dataset_filters:
            if isinstance(dataset_filter, BasePseudotimeFilter):
                self.time_label = dataset_filter.label()
                break

        return {
            # build the reference graph
            "graph_ref": graph_ref,
            # build the predicted graph
            "graph_pred": self._build_pred_graph(
                output_path, graph_ref[CELL_TYPE_TO_ID_KEY]
            ),
            "model": model,
        }

    def _submetric_eval(self, graph_pred, graph_ref, model):
        """
        Wrapper function to call the graph similarity evaluation, and handle database
        logging.
        """
        eval = self._graph_sim_eval(graph_pred, graph_ref)
        if eval is not None:
            self.db_manager.insert_eval(
                model, self.__class__.__name__, self._get_param_encoding(), eval
            )

    def _graph_sim_eval(self, graph_pred, graph_ref):
        raise NotImplementedError("Subclasses should implement this method.")
