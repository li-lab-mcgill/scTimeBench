"""
Graph Similarity Metric Base Class
"""
from metrics.base import OutputPathName
from metrics.ontology_based.base import OntologyBasedMetrics
from shared.helpers import parse_cell_lineage
from shared.dataset.filters.lineage import LineageDatasetFilter

import numpy as np
import logging


class GraphSimMetric(OntologyBasedMetrics):
    def __init__(self, config, db_manager):
        super().__init__(config, db_manager)

        # ** NOTE: must define the following attribute **
        self.output_path_name = OutputPathName.GRAPH_SIM

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

        self.graph_ref = {
            "adjacency_matrix": adjacency_matrix,
            "cell_type_to_id": cell_type_to_id,
        }
        logging.debug(f"Reference graph adjacency matrix:\n{adjacency_matrix}")

    def _eval(self, output_path, dataset):
        """
        The graph similarity metrics we will be using will take in
        """
        print(f"Model outputs are found: {output_path}")
        # build the reference graph
        self._build_ref_graph(dataset)

        # build the predicted graph
        self.graph_pred = None

        if self.submetrics:
            for submetric in self.submetrics:
                submetric_instance = submetric(self.config)
                submetric_instance._graph_sim_eval(self.graph_pred, self.graph_ref)
        else:
            self._graph_sim_eval(self.graph_pred, self.graph_ref)

    def _graph_sim_eval(self, graph_pred, graph_ref):
        raise NotImplementedError("Subclasses should implement this method.")
