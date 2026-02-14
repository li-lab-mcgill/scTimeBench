from crispy_fishstick.metrics.ontology_based.graph_sim.base import (
    GraphSimMetric,
    AdjacencyMatrixType,
    CELL_TYPE_TO_ID_KEY,
)
import os
import logging


class GraphVisualization(GraphSimMetric):
    def _graph_sim_eval(self, graph_pred, graph_ref):
        """
        This is a special metric that uses graphviz to generate visualizations of the trajectory

        We build both the predicted and reference graphs and save them as images.
        """
        import graphviz

        def build_graph_image(adj_matrix, cell_id_to_type, output_path):
            dot = graphviz.Digraph(format="png")
            num_nodes = adj_matrix.shape[0]

            # Add nodes with labels
            for node_id in range(num_nodes):
                cell_type = cell_id_to_type.get(node_id, "Unknown")
                dot.node(str(node_id), label=cell_type)

            # Add edges
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if adj_matrix[i, j] > 0:
                        dot.edge(str(i), str(j), label=f"{adj_matrix[i, j]:.2f}")

            dot.render(output_path, cleanup=True)
            logging.info(f"Graph visualization saved to {output_path}.png")

        # take the reverse dictionary
        cell_id_to_type = {v: k for k, v in graph_ref[CELL_TYPE_TO_ID_KEY].items()}

        build_graph_image(
            graph_ref[AdjacencyMatrixType.UNWEIGHTED],
            cell_id_to_type,
            os.path.join(self.output_path, "reference_graph"),
        )
        build_graph_image(
            graph_pred[AdjacencyMatrixType.WEIGHTED],
            cell_id_to_type,
            os.path.join(self.output_path, "predicted_graph"),
        )
        build_graph_image(
            graph_pred[AdjacencyMatrixType.UNWEIGHTED],
            cell_id_to_type,
            os.path.join(self.output_path, "predicted_unweighted_graph"),
        )

        return  # Visualization metric does not return a numeric score
