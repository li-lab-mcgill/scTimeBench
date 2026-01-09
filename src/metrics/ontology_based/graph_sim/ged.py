from metrics.ontology_based.graph_sim.base import GraphSimMetric


class GraphEditDistance(GraphSimMetric):
    def _eval(self, graph_pred, graph_ref):
        """
        The graph similarity metrics we will be using will take in
        """
        print(
            f"Running GraphEditDistance evaluation on predicted graph: {graph_pred} and reference graph: {graph_ref}"
        )
