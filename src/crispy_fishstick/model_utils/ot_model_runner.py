from crispy_fishstick.model_utils.model_runner import BaseModel
from crispy_fishstick.shared.constants import RequiredOutputFiles
from crispy_fishstick.shared.constants import ObservationColumns
from scipy.sparse import issparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class BaseOTModel(BaseModel):
    """
    Base class for OT-based models.
    """

    def __init__(self, yaml_config):
        super().__init__(yaml_config)

        # select the option that has NEXT_CELLTYPE if it's an OT method
        for option in self.required_outputs_options:
            if RequiredOutputFiles.NEXT_CELLTYPE in option:
                self.required_outputs = option
                break

        if not hasattr(self, "required_outputs"):
            print(
                f"Warning: OT method but no NEXT_CELLTYPE in required outputs. Using first option."
            )
            self.required_outputs = self.required_outputs_options[0]

        print(f"Required outputs for OT method: {self.required_outputs}")

        # Cache for computed transport plans and intermediate results
        self._transport_plans_cache = {}
        self._test_ann_data = None
        self._pca_model = None

    def train(self, ann_data, all_tps=None):
        print(
            f"OT-based model training not part of OT paradigm. Skipping training step."
        )

    def _prepare_generate(self, test_ann_data):
        """
        By default we don't need to do anything here. Subclasses representing OT methods can override this method if needed.
        """

    def _ensure_transport_plans(self, test_ann_data):
        """
        Compute and cache transport plans for all timepoint transitions.
        """
        if self._test_ann_data is not None:
            return  # Already computed

        self._prepare_generate(test_ann_data)
        self._test_ann_data = test_ann_data

        test_tps = test_ann_data.obs[ObservationColumns.TIMEPOINT.value].to_numpy()
        unique_tps = sorted(np.unique(test_tps))

        # Compute transport plans for each transition
        for i, tp in enumerate(unique_tps[:-1]):
            next_tp = unique_tps[i + 1]
            print(f"Computing transport plan: {tp} -> {next_tp}")
            transport_plan = self.get_transport_plan(test_ann_data, tp, next_tp)
            self._transport_plans_cache[(tp, next_tp)] = transport_plan

    def get_transport_plan(self, source_data, target_data):
        """
        Given source and target data, compute the transport plan.
        Subclasses representing OT methods should implement this method.

        Parameters:
        -----------
        source_data : np.ndarray
            Source data matrix (cells x features)
        target_data : np.ndarray
            Target data matrix (cells x features)

        Returns:
        --------
        np.ndarray
            Transport plan matrix (source cells x target cells)
        """
        raise NotImplementedError(
            "Subclasses representing OT methods should implement this method."
        )

    def generate_embedding(self, test_ann_data) -> np.ndarray:
        """
        Generate PCA embeddings from gene expression data.
        """
        print("Computing PCA embedding from gene expression data...")
        data = (
            test_ann_data.X.toarray() if issparse(test_ann_data.X) else test_ann_data.X
        )
        self._pca_model = PCA(n_components=50)
        embedding = self._pca_model.fit_transform(data)
        return embedding

    def generate_next_tp_embedding(self, test_ann_data) -> np.ndarray:
        """
        Generate embeddings for the next timepoint using transport plan.
        """
        self._ensure_transport_plans(test_ann_data)

        # First ensure we have next timepoint gene expression
        next_tp_gex = self.generate_next_tp_gex(test_ann_data)

        # Use the same PCA model to transform next timepoint data
        if self._pca_model is None:
            # Need to fit PCA first
            data = (
                test_ann_data.X.toarray()
                if issparse(test_ann_data.X)
                else test_ann_data.X
            )
            self._pca_model = PCA(n_components=50)
            self._pca_model.fit(data)

        # zero out the NaN rows before transforming
        nan_rows = np.isnan(next_tp_gex).any(axis=1)
        next_tp_gex[nan_rows, :] = 0.0
        next_tp_embedding = self._pca_model.transform(next_tp_gex)

        # then nan out the rows that were NaN originally
        next_tp_embedding[nan_rows, :] = np.nan
        return next_tp_embedding

    def generate_next_tp_gex(self, test_ann_data) -> np.ndarray:
        """
        Generate gene expression for the next timepoint using transport plan.
        """
        self._ensure_transport_plans(test_ann_data)

        test_tps = test_ann_data.obs[ObservationColumns.TIMEPOINT.value].to_numpy()
        unique_tps = sorted(np.unique(test_tps))

        next_tp_gex = np.full(
            (test_ann_data.n_obs, test_ann_data.n_vars), np.nan, dtype=np.float32
        )

        for i, tp in enumerate(unique_tps[:-1]):
            next_tp = unique_tps[i + 1]
            transport_plan = self._transport_plans_cache[(tp, next_tp)]

            source_mask = test_ann_data.obs[ObservationColumns.TIMEPOINT.value] == tp
            dest_mask = test_ann_data.obs[ObservationColumns.TIMEPOINT.value] == next_tp

            inferred_gex = self._get_next_cell_gex_from_transport_plan(
                transport_plan, test_ann_data, dest_mask
            )

            # Assign to source cells
            source_indices = np.where(source_mask)[0]
            for j, cell_idx in enumerate(source_indices):
                next_tp_gex[cell_idx, :] = inferred_gex[j]

        return next_tp_gex

    def generate_next_cell_type(self, test_ann_data) -> pd.DataFrame:
        """
        Generate next cell type predictions using transport plan.
        """
        self._ensure_transport_plans(test_ann_data)

        test_tps = test_ann_data.obs[ObservationColumns.TIMEPOINT.value].to_numpy()
        unique_tps = sorted(np.unique(test_tps))

        next_cell_types = np.full((test_ann_data.n_obs,), "unknown", dtype=object)

        for i, tp in enumerate(unique_tps[:-1]):
            next_tp = unique_tps[i + 1]
            transport_plan = self._transport_plans_cache[(tp, next_tp)]

            source_mask = test_ann_data.obs[ObservationColumns.TIMEPOINT.value] == tp
            dest_mask = test_ann_data.obs[ObservationColumns.TIMEPOINT.value] == next_tp

            inferred_types = self._get_next_cell_type_from_transport_plan(
                transport_plan, test_ann_data, dest_mask
            )

            # Assign to source cells
            source_indices = np.where(source_mask)[0]
            for j, cell_idx in enumerate(source_indices):
                next_cell_types[cell_idx] = inferred_types[j]

        # TODO: change this to return a DataFrame with proper column name
        return pd.DataFrame({"next_cell_type": next_cell_types})

    def _get_next_cell_type_from_transport_plan(
        self, transport_plan, ann_data, dest_mask
    ) -> list:
        """
        Given a transport plan, infer the next cell types.
        """
        dest_cell_types = ann_data.obs[ObservationColumns.CELL_TYPE.value].to_numpy()[
            dest_mask
        ]

        # One hot encoding of dest cell types
        unique_cell_types = sorted(np.unique(dest_cell_types))
        cell_type_to_index = {ct: i for i, ct in enumerate(unique_cell_types)}
        one_hot_dest = np.zeros((len(dest_cell_types), len(unique_cell_types)))
        for i, ct in enumerate(dest_cell_types):
            one_hot_dest[i, cell_type_to_index[ct]] = 1.0

        # Matrix multiplication to get distribution
        cell_type_distribution = transport_plan @ one_hot_dest

        # Argmax to get target cell type
        target_cell_types = []
        total_no_weight = 0
        for i in range(cell_type_distribution.shape[0]):
            distribution = cell_type_distribution[i, :]
            if np.sum(distribution) == 0:
                total_no_weight += 1
            target_cell_types.append(unique_cell_types[np.argmax(distribution)])

        if total_no_weight > 0:
            print(
                f"Warning: {total_no_weight} source cells had zero transport weights."
            )

        assert (
            len(target_cell_types) == transport_plan.shape[0]
        ), f"Length of target cell types {len(target_cell_types)} does not match number of source cells {transport_plan.shape[0]}"
        return target_cell_types

    def _get_next_cell_gex_from_transport_plan(
        self, transport_plan, ann_data, dest_mask
    ):
        """
        Given a transport plan, compute expected gene expression at next timepoint.
        """
        data = ann_data.X.toarray() if issparse(ann_data.X) else ann_data.X
        dest_data = data[dest_mask, :]

        BATCH_SIZE = 1000
        num_source_cells = transport_plan.shape[0]
        target_cell_expression = np.zeros((num_source_cells, dest_data.shape[1]))

        for start_idx in range(0, num_source_cells, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, num_source_cells)
            batch_transport_plan = transport_plan[start_idx:end_idx, :]
            batch_target_expression = batch_transport_plan @ dest_data
            target_cell_expression[start_idx:end_idx, :] = batch_target_expression

        return target_cell_expression
