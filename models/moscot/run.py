"""
Moscot runner script.

This script trains and evaluates Moscot (Multiomics Single-cell Optimal Transport)
for trajectory inference on an AnnData dataset.
It uses the TemporalProblem from moscot to compute optimal transport maps between time points.
It keeps the BaseModel runner structure used across the project.
"""

import os
from typing import List, Optional

import numpy as np
import anndata
import scanpy as sc
from scipy.sparse import issparse

from crispy_fishstick.model_utils.model_runner import main, BaseModel
from crispy_fishstick.shared.constants import ObservationColumns, RequiredOutputColumns

from moscot.problems.time import TemporalProblem


class Moscot(BaseModel):
    def __init__(self, yaml_config):
        super().__init__(yaml_config)
        self.train_ann_data = None
        self.temporal_problems = (
            {}
        )  # Cache for solved problems: (t_i, t_{i+1}) -> TemporalProblem

    def is_ot_method(self):
        return True

    def train(self, ann_data: anndata.AnnData, all_tps: Optional[List] = None):
        """
        Store the training data. Temporal problems will be solved during generation.
        """
        print(
            f"Storing training data with shape {ann_data.shape}, though in Moscot training is deferred to generation."
        )

    def generate(self, test_ann_data: anndata.AnnData, expected_output_path: str):
        """
        Generation logic for Moscot.

        For each timepoint transition t_i -> t_{i+1}:
        1. Create a TemporalProblem with test(t_i) and test(t_{i+1})
        2. Solve and cache the problem
        3. Push test from t_i through the map
        4. Extract test cell predictions at t_{i+1}
        """
        print("Starting generation with Moscot...")
        final_ann_data = test_ann_data.copy()

        # Get timepoint information
        test_tps = test_ann_data.obs[ObservationColumns.TIMEPOINT.value].to_numpy()
        unique_tps = sorted(np.unique(test_tps))

        print(f"Processing required outputs: {self.required_outputs}")

        if RequiredOutputColumns.NEXT_CELLTYPE in self.required_outputs:
            # Compute next timepoint expression using sequential OT problems
            _, next_cell_types = self._compute_next_timepoint_expression(
                test_ann_data, test_tps, unique_tps
            )
            final_ann_data.obsm[
                RequiredOutputColumns.NEXT_CELLTYPE.value
            ] = next_cell_types

        if (
            RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION
            in self.required_outputs
            or RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING in self.required_outputs
        ):
            # Compute next timepoint expression using sequential OT problems
            next_expression, _ = self._compute_next_timepoint_expression(
                test_ann_data, test_tps, unique_tps
            )
            final_ann_data.obsm[
                RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION.value
            ] = next_expression

            if RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING in self.required_outputs:
                next_embeddings = self._compute_next_timepoint_embeddings_from_data(
                    next_expression, test_tps, unique_tps
                )
                final_ann_data.obsm[
                    RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING.value
                ] = next_embeddings

        if RequiredOutputColumns.EMBEDDING in self.required_outputs:
            print(f"Computing embeddings for all timepoints...")
            final_ann_data.obsm[
                RequiredOutputColumns.EMBEDDING.value
            ] = self._compute_per_timepoint_embeddings(test_ann_data)
            print(f"Finished computing embeddings for all timepoints.")

        # Write output
        print(f"Writing output to {expected_output_path}")
        final_ann_data.write_h5ad(expected_output_path)
        print("Generation complete.")

    def _solve_temporal_problem(self, ann_data: anndata.AnnData, source_tp, target_tp):
        """
        Solve a single temporal problem for transition source_tp -> target_tp.
        Check cache first, then solve and cache if needed.

        Args:
            source_tp: Source timepoint
            target_tp: Target timepoint

        Returns:
            Solved TemporalProblem
        """
        cache_key = (source_tp, target_tp)

        # Check in-memory cache
        if cache_key in self.temporal_problems:
            return self.temporal_problems[cache_key]

        # Check file cache
        problems_dir = os.path.join(self.config["output_path"], "problems")
        cache_file = os.path.join(problems_dir, f"{source_tp}_{target_tp}.pkl")

        if os.path.exists(cache_file):
            print(f"Loading cached problem for {source_tp} -> {target_tp}")
            tp = TemporalProblem.load(cache_file)
            self.temporal_problems[cache_key] = tp
            return tp

        # Need to create and solve
        print(f"Creating and solving temporal problem for {source_tp} -> {target_tp}")

        # Create subset with only these two timepoints
        time_key = ObservationColumns.TIMEPOINT.value
        mask = ann_data.obs[time_key].isin([source_tp, target_tp])
        print(mask.value_counts())
        subset_data = ann_data[mask].copy()

        # Set up and solve
        metadata = self.config.get("model", {}).get("metadata", {})
        epsilon = metadata.get("epsilon", 1e-3)
        scale_cost = metadata.get("scale_cost", "mean")
        max_iterations = metadata.get("max_iterations", int(1e7))

        tp = TemporalProblem(subset_data).prepare(time_key=time_key)
        tp = tp.solve(
            epsilon=epsilon, scale_cost=scale_cost, max_iterations=max_iterations
        )

        # Cache to file
        os.makedirs(problems_dir, exist_ok=True)
        tp.save(cache_file)

        # Cache in memory
        self.temporal_problems[cache_key] = tp
        return tp

    def _compute_per_timepoint_embeddings(
        self, ann_data: anndata.AnnData, n_comps: int = 30
    ) -> np.ndarray:
        """
        Compute PCA embeddings separately for each timepoint.
        """
        time_key = ObservationColumns.TIMEPOINT.value
        cell_tps = ann_data.obs[time_key].to_numpy()
        unique_tps = sorted(np.unique(cell_tps))

        data = ann_data.X.toarray() if issparse(ann_data.X) else ann_data.X
        n_cells = data.shape[0]
        embeddings = np.zeros((n_cells, n_comps), dtype=np.float32)

        for tp in unique_tps:
            tp_indices = np.where(cell_tps == tp)[0]
            tp_data = data[tp_indices, :]

            if tp_data.shape[0] < 2:
                print(
                    f"Warning: Only {tp_data.shape[0]} cells at timepoint {tp}, skipping PCA"
                )
                continue

            # Compute PCA for this timepoint
            pca_adata = anndata.AnnData(tp_data)
            sc.pp.pca(pca_adata, n_comps=min(n_comps, tp_data.shape[0] - 1))
            tp_embeddings = pca_adata.obsm["X_pca"]

            # Pad if needed
            if tp_embeddings.shape[1] < n_comps:
                padded = np.zeros((tp_embeddings.shape[0], n_comps), dtype=np.float32)
                padded[:, : tp_embeddings.shape[1]] = tp_embeddings
                tp_embeddings = padded

            embeddings[tp_indices] = tp_embeddings.astype(np.float32)

        return embeddings

    def _compute_next_timepoint_embeddings_from_data(
        self,
        data: np.ndarray,
        cell_tps: np.ndarray,
        unique_tps: List,
        n_comps: int = 30,
    ) -> np.ndarray:
        """
        Compute PCA embeddings for expression data (not an AnnData object).
        """
        n_cells = data.shape[0]
        embeddings = np.full((n_cells, n_comps), np.nan, dtype=np.float32)

        for tp in unique_tps[:-1]:
            tp_indices = np.where(cell_tps == tp)[0]
            tp_data = data[tp_indices, :]

            if tp_data.shape[0] < 2:
                print(
                    f"Warning: Only {tp_data.shape[0]} cells at timepoint {tp}, skipping PCA"
                )
                continue

            pca_adata = anndata.AnnData(tp_data)
            sc.pp.pca(pca_adata, n_comps=min(n_comps, tp_data.shape[0] - 1))
            tp_embeddings = pca_adata.obsm["X_pca"]

            if tp_embeddings.shape[1] < n_comps:
                padded = np.zeros((tp_embeddings.shape[0], n_comps), dtype=np.float32)
                padded[:, : tp_embeddings.shape[1]] = tp_embeddings
                tp_embeddings = padded

            embeddings[tp_indices] = tp_embeddings.astype(np.float32)

        return embeddings

    def _compute_next_timepoint_expression(
        self, ann_data: anndata.AnnData, cell_tps: np.ndarray, unique_tps: List
    ) -> np.ndarray:
        """
        Compute predicted gene expression at the next timepoint.

        For each transition t_i -> t_{i+1}:
        - Solve the temporal problem for that pair
        - Push combined (train + test) cells from t_i
        - Extract test cell predictions at t_{i+1}
        """
        cell_type_data = ann_data.obs[ObservationColumns.CELL_TYPE.value].to_numpy()
        data = ann_data.X.toarray() if issparse(ann_data.X) else ann_data.X

        n_cells, n_genes = data.shape
        next_timepoint_expr = np.full((n_cells, n_genes), np.nan, dtype=np.float32)
        next_tp_cell_types = np.full((n_cells,), "unknown", dtype=object)

        print(f"Unique timepoints: {unique_tps}")

        # Process each timepoint transition
        for i, tp in enumerate(unique_tps[:-1]):
            next_tp = unique_tps[i + 1]

            # Get cells at current timepoint
            tp_indices = np.where(cell_tps == tp)[0]
            next_tp_indices = np.where(cell_tps == next_tp)[0]

            if len(tp_indices) == 0:
                continue

            tp_data = data[tp_indices]
            next_tp_data = data[next_tp_indices]

            print(f"Pushing {len(tp_indices)} cells from {tp} to {next_tp}")
            print(f"  Data shape: {tp_data.shape}")

            # Solve temporal problem for this transition
            tp_problem = self._solve_temporal_problem(ann_data, tp, next_tp)

            # now let's go through the temporal problem matrix and print out its
            assert (
                len(tp_problem.solutions) == 1
            ), "Expected exactly one solution per temporal problem"
            print(f"Solutions: {tp_problem.solutions}")

            for _, solution in tp_problem.solutions.items():
                # now let's use the transport matrix's argmax to get the pushed expression
                next_cell_indices = np.argmax(solution.transport_matrix, axis=1)
                next_timepoint_expr[tp_indices] = next_tp_data[
                    next_cell_indices
                ].astype(np.float32)

                # Also get the predicted cell types if available
                next_tp_cell_types[tp_indices] = cell_type_data[next_tp_indices][
                    next_cell_indices
                ]

        return next_timepoint_expr, next_tp_cell_types


if __name__ == "__main__":
    main(Moscot)
