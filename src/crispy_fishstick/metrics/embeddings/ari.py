"""
Graph Similarity Metric Base Class
"""
from crispy_fishstick.metrics.embeddings.base import EmbeddingMetrics
from crispy_fishstick.shared.constants import ObservationColumns, RequiredOutputColumns

import os
import logging
import scanpy as sc
from sklearn.metrics import adjusted_rand_score


class ARI(EmbeddingMetrics):
    def _defaults(self):
        return {
            "n_neighbors": 15,
            "resolution": 0.5,
        }

    def _embedding_eval(self, output_path):
        """
        The embedding-based metric evaluation function.
        """
        model_output_file = os.path.join(output_path, self.output_path_name.value)
        adata = sc.read_h5ad(model_output_file)

        true_labels = adata.obs[ObservationColumns.CELL_TYPE.value].values

        # silence the numba warnings
        logging.getLogger("numba").setLevel(logging.WARNING)

        sc.pp.neighbors(
            adata,
            use_rep=RequiredOutputColumns.EMBEDDING.value,
            n_neighbors=self.n_neighbors,
        )
        sc.tl.leiden(adata, key_added="leiden_clusters", resolution=self.resolution)

        ari_score = adjusted_rand_score(true_labels, adata.obs["leiden_clusters"])

        logging.debug(f"Adjusted Rand Index: {ari_score}")

        return ari_score
