"""
Graph Similarity Metric Base Class
"""

from scTimeBench.metrics.embeddings.aggregate.base import AggregateEmbeddingMetrics
from scTimeBench.shared.constants import ObservationColumns, RequiredOutputFiles
from scTimeBench.shared.utils import load_test_dataset, load_output_file

import json
import logging

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors
from enum import Enum


class VisualizeType(Enum):
    UMAP = "umap"
    TSNE = "tsne"
    PHATE = "phate"
    PCA = "pca"


class VisualizeEmbeds(AggregateEmbeddingMetrics):
    def _setup_method_output_requirements(self):
        self.required_outputs = [
            RequiredOutputFiles.EMBEDDING,
            RequiredOutputFiles.NEXT_TIMEPOINT_EMBEDDING,
        ]

    def _defaults(self):
        return {
            "n_neighbors": 15,
            "type": VisualizeType.UMAP.value,
        }

    def _embedding_eval(self, output_path, dataset):
        """
        The embedding-based metric evaluation function. The function works as follows:
        1. Load and use the embeddings to calculate a knn graph.
        2. Use the inferred knn graph to transfer cell type labels.
        3. Calculate the ARI on these cell type labels.
        4. Compare ARI for ground truth cell embeddings vs next timepoint embeddings.
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

        # if self.type is not an enum's value, raise an error
        if not self.type in [val.value for val in VisualizeType]:
            raise ValueError(f"Unsupported visualization type: {self.type}")

        (
            graph_path,
            next_tp_graph_path,
            orig_graph_path,
        ) = self.visualize_embedding_projection(
            embeddings, pred_embeddings, cell_types, pred_labels, output_path
        )

        logging.debug(
            f"Saved {self.type} visualizations to {graph_path}, {next_tp_graph_path}, and {orig_graph_path}"
        )

        return json.dumps(
            {
                "graph_path": graph_path,
                "next_timepoint_graph_path": next_tp_graph_path,
                "orig_graph_path": orig_graph_path,
            }
        )

    def visualize_embedding_projection(
        self, embeddings, pred_embeddings, cell_types, pred_labels, output_path
    ):
        # Use Scanpy to compute a 2D projection on combined embeddings and plot
        logging.getLogger("numba").setLevel(logging.WARNING)

        n_orig = embeddings.shape[0]
        n_pred = pred_embeddings.shape[0]
        total_embeddings = np.vstack([embeddings, pred_embeddings])

        # Build AnnData with obs for origin and cell type labels
        obs = pd.DataFrame(
            {
                "origin": ["orig"] * n_orig + ["pred"] * n_pred,
                "cell_type": np.concatenate(
                    [cell_types.astype(str), pred_labels.astype(str)]
                ),
            }
        )

        adata = sc.AnnData(X=total_embeddings, obs=obs)

        # Compute neighbors and the requested projection using Scanpy
        n_neighbors = min(self.n_neighbors, max(1, total_embeddings.shape[0] - 1))
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")

        if VisualizeType(self.type) == VisualizeType.UMAP:
            sc.tl.umap(adata, random_state=42)
            coord_key = "X_umap"
        elif VisualizeType(self.type) == VisualizeType.TSNE:
            sc.tl.tsne(adata, random_state=42)
            coord_key = "X_tsne"
        elif VisualizeType(self.type) == VisualizeType.PHATE:
            import phate

            phate_operator = phate.PHATE(
                knn=n_neighbors, n_jobs=-2, decay=15, verbose=True
            )
            adata.obsm["X_phate"] = phate_operator.fit_transform(adata.X)
            coord_key = "X_phate"
        elif VisualizeType(self.type) == VisualizeType.PCA:
            sc.tl.pca(adata, n_comps=2, random_state=42)
            coord_key = "X_pca"
        else:
            raise ValueError(f"Unsupported visualization type: {self.type}")

        coords = adata.obsm[coord_key]

        # Prepare color mapping across all labels so colors are consistent
        all_labels = np.unique(adata.obs["cell_type"].values)
        categories = list(all_labels)
        cat = pd.Categorical(adata.obs["cell_type"], categories=categories)
        codes = cat.codes
        cmap = plt.get_cmap("tab20")
        colors = cmap(codes % cmap.N)

        # Prepare output file paths
        graph_path = os.path.join(output_path, f"{self.type}.png")
        next_tp_graph_path = os.path.join(output_path, f"next_tp_{self.type}.png")
        orig_graph_path = os.path.join(output_path, f"orig_{self.type}.png")

        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        # Plot everything together: originals faint, preds highlighted
        plt.figure(figsize=(6, 5))
        mask_orig = adata.obs["origin"] == "orig"
        mask_pred = adata.obs["origin"] == "pred"

        # Originals: small, semi-transparent
        plt.scatter(
            coords[mask_orig, 0],
            coords[mask_orig, 1],
            c=colors[mask_orig],
            s=8,
            alpha=0.5,
        )

        # Predicted: larger with black edge
        plt.scatter(
            coords[mask_pred, 0],
            coords[mask_pred, 1],
            c=colors[mask_pred],
            s=8,
            edgecolor="k",
            linewidths=0.3,
            zorder=2,
        )

        plt.title(f"{self.type} of embeddings (orig + pred)")
        plt.xlabel(f"{self.type.upper()}1")
        plt.ylabel(f"{self.type.upper()}2")
        # Legend for cell types
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=str(cat),
                markerfacecolor=cmap(i % cmap.N),
                markersize=6,
            )
            for i, cat in enumerate(categories)
        ]
        plt.legend(
            handles=handles,
            title="cell_type",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize="small",
        )
        plt.tight_layout()
        plt.savefig(graph_path, dpi=150)
        plt.close()

        # Second plot: originals greyed, pred colored
        plt.figure(figsize=(6, 5))
        plt.scatter(
            coords[mask_orig, 0], coords[mask_orig, 1], c="#dddddd", s=6, alpha=0.6
        )
        plt.scatter(
            coords[mask_pred, 0],
            coords[mask_pred, 1],
            c=colors[mask_pred],
            s=8,
            edgecolor="k",
            linewidths=0.4,
        )
        plt.title(f"{self.type} of next timepoint embeddings (highlighted)")
        plt.xlabel(f"{self.type.upper()}1")
        plt.ylabel(f"{self.type.upper()}2")
        plt.legend(
            handles=handles,
            title="cell_type",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize="small",
        )
        plt.tight_layout()
        plt.savefig(next_tp_graph_path, dpi=150)
        plt.close()

        # Third plot: predicted greyed out, originals coloured (highlight originals)
        plt.figure(figsize=(6, 5))
        plt.scatter(
            coords[mask_pred, 0], coords[mask_pred, 1], c="#dddddd", s=6, alpha=0.6
        )
        plt.scatter(
            coords[mask_orig, 0],
            coords[mask_orig, 1],
            c=colors[mask_orig],
            s=8,
            edgecolor="k",
            linewidths=0.4,
        )
        plt.title(f"{self.type} of original embeddings (highlighted)")
        plt.xlabel(f"{self.type.upper()}1")
        plt.ylabel(f"{self.type.upper()}2")
        plt.legend(
            handles=handles,
            title="cell_type",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize="small",
        )
        plt.tight_layout()
        plt.savefig(orig_graph_path, dpi=150)
        plt.close()

        return graph_path, next_tp_graph_path, orig_graph_path
