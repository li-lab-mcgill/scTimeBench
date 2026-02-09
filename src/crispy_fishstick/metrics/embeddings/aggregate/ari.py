"""
Graph Similarity Metric Base Class
"""
from crispy_fishstick.metrics.embeddings.aggregate.base import AggregateEmbeddingMetrics
from crispy_fishstick.shared.constants import ObservationColumns, RequiredOutputFiles
from crispy_fishstick.shared.utils import load_test_dataset, load_output_file

import logging
import scanpy as sc
import anndata
from sklearn.metrics import adjusted_rand_score


class ARI(AggregateEmbeddingMetrics):
    def _defaults(self):
        return {
            "n_neighbors": 15,
            "resolution": 0.5,
        }

    def _embedding_eval(self, output_path):
        """
        The embedding-based metric evaluation function.
        """
        # Load embeddings from the output file
        embeddings = load_output_file(output_path, RequiredOutputFiles.EMBEDDING)

        # Load test dataset to get true cell type labels
        test_ann_data = load_test_dataset(output_path)
        true_labels = test_ann_data.obs[ObservationColumns.CELL_TYPE.value].values

        # Create an AnnData object with embeddings for scanpy operations
        adata = anndata.AnnData(X=embeddings)
        adata.obs[ObservationColumns.CELL_TYPE.value] = true_labels
        adata.obsm["X_embedding"] = embeddings

        # silence the numba warnings
        logging.getLogger("numba").setLevel(logging.WARNING)

        sc.pp.neighbors(
            adata,
            use_rep="X_embedding",
            n_neighbors=self.n_neighbors,
        )
        sc.tl.leiden(adata, key_added="leiden_clusters", resolution=self.resolution)

        ari_score = adjusted_rand_score(true_labels, adata.obs["leiden_clusters"])

        logging.debug(f"Adjusted Rand Index: {ari_score}")

        return ari_score
