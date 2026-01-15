"""
Graph Similarity Metric Base Class
"""
from metrics.base import OutputPathName
from metrics.ontology_based.base import OntologyBasedMetrics
from shared.constants import RequiredOutputColumns
from shared.helpers import parse_cell_lineage
from shared.dataset.filters.lineage import LineageDatasetFilter

import numpy as np

import os
import logging


class AdjacencyMatrixType:
    UNWEIGHTED = "adjacency_matrix"
    WEIGHTED = "weighted_adjacency_matrix"


class GraphSimMetric(OntologyBasedMetrics):
    def _defaults(self):
        return {
            "edge_threshold": 0.1,
        }

    def _setup_model_output_requirements(self):
        # ** NOTE: must define the following attributes **
        # where we define the output embedding name
        # as well as the required features and outputs
        self.output_path_name = OutputPathName.GRAPH_SIM
        self.required_outputs = [
            RequiredOutputColumns.EMBEDDING,
            RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING,
        ]

    def _build_ref_graph(self, dataset):
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
                cell_lineage = parse_cell_lineage(filter.cell_lineage_file)

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

        self.graph_ref = adjacency_matrix
        self.cell_type_to_id = cell_type_to_id
        logging.debug(f"Reference graph adjacency matrix:\n{adjacency_matrix}")

    def _build_pred_graph(self, output_path):
        """
        Builds the predicted graph structure based on the output.
        """
        # first let's ensure that it's in the right format
        # we expect it to have the true embeddings and predicted embeddings
        # for timepoints (1, ..., n) in h5ad format, where we save new embeddings
        model_output_file = os.path.join(output_path, self.output_path_name.value)
        pred_trajectory = self.trajectory_infer_model.infer_trajectory(
            model_output_file
        )

        # based on the predicted trajectory, we can build the adjacency matrix
        # let's use the config's threshold for an edge to be created
        num_cell_types = len(self.cell_type_to_id)
        weighted_adjacency_matrix = np.zeros(
            (num_cell_types, num_cell_types), dtype=np.float32
        )
        adjacency_matrix = np.zeros((num_cell_types, num_cell_types), dtype=np.float32)

        # now we can iterate over the predicted trajectory to fill in the adjacency matrix
        for source_cell_type, target_distribution in pred_trajectory.items():
            source_id = self.cell_type_to_id[source_cell_type]

            for target_cell_type, prob in target_distribution.items():
                target_id = self.cell_type_to_id[target_cell_type]
                weighted_adjacency_matrix[source_id, target_id] = prob

                # avoid self-loops and any edges below the threshold
                if source_id != target_id and prob >= self.edge_threshold:
                    adjacency_matrix[source_id, target_id] = 1.0

        self.graph_pred = {
            AdjacencyMatrixType.WEIGHTED: weighted_adjacency_matrix,
            AdjacencyMatrixType.UNWEIGHTED: adjacency_matrix,
        }

    def _eval(self, output_path, dataset):
        """
        The graph similarity metrics we will be using will take in
        """
        print(f"Model outputs are found: {output_path}")
        # build the reference graph
        self._build_ref_graph(dataset)

        # build the predicted graph
        self._build_pred_graph(output_path)

        if self.submetrics:
            for submetric in self.submetrics:
                submetric_instance = submetric(self.config)
                submetric_instance._graph_sim_eval_wrapper(
                    self.graph_pred, self.graph_ref
                )
        else:
            self._graph_sim_eval_wrapper(self.graph_pred, self.graph_ref)

    def _graph_sim_eval_wrapper(self, graph_pred, graph_ref):
        """
        Wrapper function to call the graph similarity evaluation, and handle database
        logging.
        """
        self.db_manager.insert_eval(
            self.model,
            self.__class__.__name__,
            self._get_param_encoding(),
            self._graph_sim_eval(graph_pred, graph_ref),
        )

    def _graph_sim_eval(self, graph_pred, graph_ref):
        raise NotImplementedError("Subclasses should implement this method.")
