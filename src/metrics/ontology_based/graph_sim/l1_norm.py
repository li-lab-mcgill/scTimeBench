from metrics.ontology_based.graph_sim.base import GraphSimMetric


class L1Norm(GraphSimMetric):
    def _eval(self, graph_pred, graph_ref):
        """
        The graph similarity metrics we will be using will take in
        """
        print(
            f"Running L1Norm evaluation on predicted graph: {graph_pred} and reference graph: {graph_ref}"
        )
