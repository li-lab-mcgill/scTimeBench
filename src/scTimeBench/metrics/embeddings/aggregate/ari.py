"""
Graph Similarity Metric Base Class
"""

from scTimeBench.metrics.embeddings.aggregate.base import AggregateEmbeddingMetrics
from scTimeBench.shared.constants import ObservationColumns, RequiredOutputFiles
from scTimeBench.shared.utils import load_test_dataset, load_output_file

import json
import logging

import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse


class ARI(AggregateEmbeddingMetrics):
    def _setup_method_output_requirements(self):
        self.required_outputs = [
            RequiredOutputFiles.EMBEDDING,
            RequiredOutputFiles.NEXT_TIMEPOINT_EMBEDDING,
            RequiredOutputFiles.NEXT_TIMEPOINT_GENE_EXPRESSION,
        ]

    def _defaults(self):
        return {
            "n_neighbors": 15,
            "resolution": 0.5,
        }

    def _embedding_eval(self, output_path, dataset):
        """
        The embedding-based metric evaluation function. The function works as follows:
        1. Load and use the embeddings to calculate a knn graph.
        2. Use the inferred knn graph to transfer cell type labels.
        3. Calculate the ARI on these cell type labels.
        4. Compare ARI for ground truth cell embeddings vs next timepoint embeddings.
        5. Compute ARI for GEx + PCA ground truth and next timepoint.
        """
        embeddings = load_output_file(output_path, RequiredOutputFiles.EMBEDDING)
        next_timepoint_embeddings = load_output_file(
            output_path, RequiredOutputFiles.NEXT_TIMEPOINT_EMBEDDING
        )

        # Load test dataset to get true labels and timepoints
        test_ann_data = load_test_dataset(output_path)
        cell_types = test_ann_data.obs[ObservationColumns.CELL_TYPE.value].to_numpy()

        if embeddings.shape[0] != cell_types.shape[0]:
            raise ValueError(
                "Embedding count does not match number of cell type labels."
            )

        # silence the numba warnings
        logging.getLogger("numba").setLevel(logging.WARNING)

        def compute_ari(embeds, labels):
            if embeds.shape[0] == 0:
                raise ValueError("No embeddings available for ARI computation.")
            n_neighbors = min(self.n_neighbors, max(1, embeds.shape[0] - 1))
            adata_eval = sc.AnnData(X=embeds)
            sc.pp.neighbors(adata_eval, n_neighbors=n_neighbors)
            sc.tl.leiden(
                adata_eval, key_added="leiden_clusters", resolution=self.resolution
            )
            return adjusted_rand_score(labels, adata_eval.obs["leiden_clusters"])

        # compute ground truth ARI, excluding the first timepoint
        ari_ground_truth = compute_ari(embeddings, cell_types)

        # assign next timepoint labels using kNN to all ground truth embeddings
        valid_mask = ~np.isnan(next_timepoint_embeddings).any(axis=1)
        pred_embeddings = next_timepoint_embeddings[valid_mask]

        if pred_embeddings.shape[0] == 0:
            raise ValueError("No valid next timepoint embeddings for ARI computation.")

        n_neighbors = min(self.n_neighbors, embeddings.shape[0])
        knn_model = NearestNeighbors(n_neighbors=n_neighbors)
        knn_model.fit(embeddings)
        _, neighbor_indices = knn_model.kneighbors(pred_embeddings)

        pred_labels = []
        for neighbors in neighbor_indices:
            neighbor_labels = cell_types[neighbors]
            labels, counts = np.unique(neighbor_labels, return_counts=True)
            pred_labels.append(labels[counts.argmax()])

        pred_labels = np.array(pred_labels)
        ari_next_timepoint = compute_ari(pred_embeddings, pred_labels)

        logging.debug(f"Adjusted Rand Index (ground truth): {ari_ground_truth}")
        logging.debug(f"Adjusted Rand Index (next timepoint): {ari_next_timepoint}")

        next_timepoint_gex = load_output_file(
            output_path, RequiredOutputFiles.NEXT_TIMEPOINT_GENE_EXPRESSION
        )
        gex = (
            test_ann_data.X.toarray() if issparse(test_ann_data.X) else test_ann_data.X
        )
        n_components = min(50, gex.shape[0], gex.shape[1])
        if n_components < 1:
            raise ValueError("No features available for GEx + PCA ARI computation.")

        pca = PCA(n_components=n_components)
        gex_embeddings = pca.fit_transform(gex)
        gex_pca_ground_truth = compute_ari(gex_embeddings, cell_types)

        gex_valid_mask = ~np.isnan(next_timepoint_gex).any(axis=1)
        pred_gex = next_timepoint_gex[gex_valid_mask]
        if pred_gex.shape[0] == 0:
            raise ValueError(
                "No valid next timepoint gene expression for GEx + PCA ARI computation."
            )

        pred_gex_embeddings = pca.transform(pred_gex)
        n_neighbors = min(self.n_neighbors, gex_embeddings.shape[0])
        knn_model = NearestNeighbors(n_neighbors=n_neighbors)
        knn_model.fit(gex_embeddings)
        _, neighbor_indices = knn_model.kneighbors(pred_gex_embeddings)

        pred_labels = []
        for neighbors in neighbor_indices:
            neighbor_labels = cell_types[neighbors]
            labels, counts = np.unique(neighbor_labels, return_counts=True)
            pred_labels.append(labels[counts.argmax()])

        pred_labels = np.array(pred_labels)
        gex_pca_next_timepoint = compute_ari(pred_gex_embeddings, pred_labels)

        logging.debug(
            "Adjusted Rand Index (GEx + PCA, ground truth): %s",
            gex_pca_ground_truth,
        )
        logging.debug(
            "Adjusted Rand Index (GEx + PCA, next timepoint): %s",
            gex_pca_next_timepoint,
        )

        return json.dumps(
            {
                "ari_ground_truth": float(ari_ground_truth),
                "ari_next_timepoint": float(ari_next_timepoint),
                "ari_ground_truth_gex_pca": float(gex_pca_ground_truth),
                "ari_next_timepoint_gex_pca": float(gex_pca_next_timepoint),
            }
        )
