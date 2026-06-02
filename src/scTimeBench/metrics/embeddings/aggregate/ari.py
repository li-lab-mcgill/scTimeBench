"""
Adjusted Rand Index metrics on embeddings and gene expression (GEx + PCA).
"""

from scTimeBench.metrics.embeddings.aggregate.base import AggregateEmbeddingMetrics
from scTimeBench.metrics.embeddings.base import EmbeddingMetrics
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


def _compute_ari(embeds, labels, n_neighbors: int, resolution: float) -> float:
    if embeds.shape[0] == 0:
        raise ValueError("No embeddings available for ARI computation.")
    n_neighbors = min(n_neighbors, max(1, embeds.shape[0] - 1))
    adata_eval = sc.AnnData(X=embeds)
    sc.pp.neighbors(adata_eval, n_neighbors=n_neighbors)
    sc.tl.leiden(adata_eval, key_added="leiden_clusters", resolution=resolution)
    return adjusted_rand_score(labels, adata_eval.obs["leiden_clusters"])


def _knn_majority_labels(
    reference_embeds, reference_labels, query_embeds, n_neighbors: int
) -> np.ndarray:
    n_neighbors = min(n_neighbors, reference_embeds.shape[0])
    knn_model = NearestNeighbors(n_neighbors=n_neighbors)
    knn_model.fit(reference_embeds)
    _, neighbor_indices = knn_model.kneighbors(query_embeds)

    pred_labels = []
    for neighbors in neighbor_indices:
        neighbor_labels = reference_labels[neighbors]
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        pred_labels.append(labels[counts.argmax()])
    return np.array(pred_labels)


class ARI(AggregateEmbeddingMetrics):
    def _defaults(self):
        return {
            "n_neighbors": 15,
            "resolution": 0.5,
        }

    def _setup_method_output_requirements(self):
        self.required_outputs = [
            RequiredOutputFiles.EMBEDDING,
            RequiredOutputFiles.NEXT_TIMEPOINT_EMBEDDING,
        ]

    def _embedding_eval(self, output_path, dataset):
        """
        Compare ARI on ground-truth embeddings vs next-timepoint embeddings.
        """
        embeddings = load_output_file(output_path, RequiredOutputFiles.EMBEDDING)
        next_timepoint_embeddings = load_output_file(
            output_path, RequiredOutputFiles.NEXT_TIMEPOINT_EMBEDDING
        )

        test_ann_data = load_test_dataset(output_path)
        cell_types = test_ann_data.obs[ObservationColumns.CELL_TYPE.value].to_numpy()

        if embeddings.shape[0] != cell_types.shape[0]:
            raise ValueError(
                "Embedding count does not match number of cell type labels."
            )

        logging.getLogger("numba").setLevel(logging.WARNING)

        ari_ground_truth = _compute_ari(
            embeddings, cell_types, self.n_neighbors, self.resolution
        )

        valid_mask = ~np.isnan(next_timepoint_embeddings).any(axis=1)
        pred_embeddings = next_timepoint_embeddings[valid_mask]
        if pred_embeddings.shape[0] == 0:
            raise ValueError("No valid next timepoint embeddings for ARI computation.")

        pred_labels = _knn_majority_labels(
            embeddings, cell_types, pred_embeddings, self.n_neighbors
        )
        ari_next_timepoint = _compute_ari(
            pred_embeddings, pred_labels, self.n_neighbors, self.resolution
        )

        logging.debug(f"Adjusted Rand Index (ground truth): {ari_ground_truth}")
        logging.debug(f"Adjusted Rand Index (next timepoint): {ari_next_timepoint}")

        return json.dumps(
            {
                "ari_ground_truth": float(ari_ground_truth),
                "ari_next_timepoint": float(ari_next_timepoint),
            }
        )


class ARIGEx(EmbeddingMetrics):
    def _defaults(self):
        return {
            "n_neighbors": 15,
            "resolution": 0.5,
        }

    def _setup_method_output_requirements(self):
        self.required_outputs = [
            RequiredOutputFiles.NEXT_TIMEPOINT_GENE_EXPRESSION,
        ]

    def _embedding_eval(self, output_path, dataset):
        """
        Compare ARI on ground-truth GEx (PCA) vs next-timepoint predicted GEx (PCA).
        """
        test_ann_data = load_test_dataset(output_path)
        cell_types = test_ann_data.obs[ObservationColumns.CELL_TYPE.value].to_numpy()
        next_timepoint_gex = load_output_file(
            output_path, RequiredOutputFiles.NEXT_TIMEPOINT_GENE_EXPRESSION
        )
        gex = (
            test_ann_data.X.toarray() if issparse(test_ann_data.X) else test_ann_data.X
        )

        n_components = min(50, gex.shape[0], gex.shape[1])
        if n_components < 1:
            raise ValueError("No features available for GEx + PCA ARI computation.")

        logging.getLogger("numba").setLevel(logging.WARNING)

        pca = PCA(n_components=n_components)
        gex_embeddings = pca.fit_transform(gex)
        ari_ground_truth = _compute_ari(
            gex_embeddings, cell_types, self.n_neighbors, self.resolution
        )

        gex_valid_mask = ~np.isnan(next_timepoint_gex).any(axis=1)
        pred_gex = next_timepoint_gex[gex_valid_mask]
        if pred_gex.shape[0] == 0:
            raise ValueError(
                "No valid next timepoint gene expression for GEx + PCA ARI computation."
            )

        pred_gex_embeddings = pca.transform(pred_gex)
        pred_labels = _knn_majority_labels(
            gex_embeddings, cell_types, pred_gex_embeddings, self.n_neighbors
        )
        ari_next_timepoint = _compute_ari(
            pred_gex_embeddings, pred_labels, self.n_neighbors, self.resolution
        )

        logging.debug(
            "Adjusted Rand Index (GEx + PCA, ground truth): %s", ari_ground_truth
        )
        logging.debug(
            "Adjusted Rand Index (GEx + PCA, next timepoint): %s", ari_next_timepoint
        )

        return json.dumps(
            {
                "ari_ground_truth": float(ari_ground_truth),
                "ari_next_timepoint": float(ari_next_timepoint),
            }
        )
