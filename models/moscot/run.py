"""
Moscot runner script.

This script trains and evaluates Moscot (Multiomics Single-cell Optimal Transport)
for trajectory inference on an AnnData dataset.
It uses the TemporalProblem from moscot to compute optimal transport maps between time points.
It keeps the BaseMethod runner structure used across the project.
"""

import os

import anndata

from scTimeBench.model_utils.model_runner import main
from scTimeBench.model_utils.ot_model_runner import BaseOTMethod
from scTimeBench.shared.constants import ObservationColumns

from moscot.problems.time import TemporalProblem


class Moscot(BaseOTMethod):
    def __init__(self, yaml_config):
        super().__init__(yaml_config)
        self.train_ann_data = None
        self.temporal_problems = (
            {}
        )  # Cache for solved problems: (t_i, t_{i+1}) -> TemporalProblem

    def get_transport_plan(self, ann_data: anndata.AnnData, source_tp, target_tp):
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
            assert (
                len(tp.solutions) == 1
            ), "Expected exactly one solution per temporal problem"
            print(f"Solutions: {tp.solutions}")

            solution = list(tp.solutions.values())[0]
            return solution.transport_matrix

        # Need to create and solve
        print(f"Creating and solving temporal problem for {source_tp} -> {target_tp}")

        # Create subset with only these two timepoints
        time_key = ObservationColumns.TIMEPOINT.value
        mask = ann_data.obs[time_key].isin([source_tp, target_tp])
        print(mask.value_counts())
        subset_data = ann_data[mask].copy()

        # Set up and solve
        metadata = self.config.get("method", {}).get("metadata", {})
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

        # now let's go through the temporal problem matrix and print out its
        assert (
            len(tp.solutions) == 1
        ), "Expected exactly one solution per temporal problem"
        print(f"Solutions: {tp.solutions}")

        solution = list(tp.solutions.values())[0]
        return solution.transport_matrix


if __name__ == "__main__":
    main(Moscot)
