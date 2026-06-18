from enum import Enum
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.sparse import issparse

import numpy as np
import pickle
import logging


class EmbeddingType(Enum):
    GEX = "gex"
    PCA = "PCA"


class EmbeddingSpace:
    def train(self, ann_data):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def encode(self, ann_data):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def decode(self, embedding):
        raise NotImplementedError("This method should be implemented by subclasses.")


class GExSpace(EmbeddingSpace):
    def _to_dense_float32(self, X):
        data = X.toarray() if issparse(X) else X
        return np.asarray(data, dtype=np.float32)

    def train(self, ann_data):
        pass

    def encode(self, ann_data):
        return self._to_dense_float32(ann_data.X)

    def decode(self, embedding):
        return embedding


class PCASpace(EmbeddingSpace):
    def __init__(self, embedding_dim, cache_dir):
        self.embedding_dim = embedding_dim
        self.cache_dir = Path(cache_dir)
        self.pca_cache_path = self.cache_dir / "pca_model.pkl"

    def _to_dense_float32(self, X):
        data = X.toarray() if issparse(X) else X
        return np.asarray(data, dtype=np.float32)

    def train(self, ann_data):
        # Implement PCA training logic here, using self.embedding_dim and self.cache_dir
        if self.pca_cache_path.exists():
            with open(self.pca_cache_path, "rb") as f:
                self._pca_model = pickle.load(f)
        else:
            train_gex = self._to_dense_float32(ann_data.X)
            n_components = min(
                self.embedding_dim,
                train_gex.shape[0],
                train_gex.shape[1],
            )
            if n_components < 1:
                raise ValueError("PCA requires at least one component.")
            self._pca_model = PCA(n_components=n_components)
            self._pca_model.fit(train_gex)
            with open(self.pca_cache_path, "wb") as f:
                pickle.dump(self._pca_model, f)

    def encode(self, ann_data):
        train_gex = self._to_dense_float32(ann_data.X)
        return self._pca_model.transform(train_gex)

    def decode(self, embedding):
        return self._pca_model.inverse_transform(embedding)


def generate_embedding_space(
    embedding_type: EmbeddingType, embedding_dim: int, cache_dir: str
) -> EmbeddingSpace:
    logging.debug(
        f"Generating embedding space with type: {embedding_type}, dimension: {embedding_dim}"
    )
    if embedding_type == EmbeddingType.GEX:
        return GExSpace(embedding_dim)
    elif embedding_type == EmbeddingType.PCA:
        return PCASpace(embedding_dim, cache_dir)
    else:
        raise ValueError("Unsupported embedding type")
