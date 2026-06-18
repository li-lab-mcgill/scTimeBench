"""
Cell type proportion perturbation metric.
"""
from scTimeBench.metrics.perturbation.base import PerturbationBasedMetrics
from scTimeBench.trajectory_infer.base import TrajectoryInferenceMethodFactory
from scTimeBench.shared.utils import load_output_file
from scTimeBench.shared.constants import RequiredOutputFiles, ObservationColumns

import logging
import json
import os


class PerturbationCellTypeProportionBase(PerturbationBasedMetrics):
    def _setup_trajectory_inference_model(self):
        traj_infer_config = self.metric_config.get("trajectory_infer_model", {})

        if "perturbation_data" in traj_infer_config:
            assert (
                traj_infer_config["perturbation_data"] == True
            ), "Trajectory inference model must be configured to use perturbation data for this metric."

        traj_infer_config["perturbation_data"] = True

        self.trajectory_infer_model = (
            TrajectoryInferenceMethodFactory().get_trajectory_infer_method(
                traj_infer_config
            )
        )
        self.params["trajectory_infer_model"] = str(self.trajectory_infer_model)

    def _prep_kwargs_for_submetric_eval(self, output_path, dataset, method):
        # TODO: change this as the training is currently the same!
        # We need to modify this so it saves it to eval output path
        return {
            "output_path": output_path,
            "trajectory": self.trajectory_infer_model.infer_trajectory(
                output_path,
                per_tp=True,
                eval_output_path=os.path.join(
                    output_path, self._get_relative_output_path()
                ),
            ),
            "method": method,
        }


class PerturbationCellTypeProportion(PerturbationCellTypeProportionBase):
    def _submetric_eval(self, output_path, trajectory, method):
        logging.debug(f"Trajectory for method {method}: {trajectory}")

        # now let's recalculate the total number of cells of each cell type
        # we have a dictionary of:
        # timepoint:
        #   { cell type -> { cell_type 1: count, cell type 2: count, ...} }
        # we want to convert this to a dictionary of:
        # cell type -> count
        # cell type -> count
        # so we need to sum up over all the timepoints, i.e. we need to do
        # all the target timepoints cell type counts
        # because we don't want to include the src cell counts
        cell_type_dist = {}

        timepoints = sorted(trajectory.keys())

        total_count = 0

        for tp in timepoints:
            logging.debug(f"Timepoint {tp}: cell type distribution {trajectory[tp]}")
            cell_type_counts = trajectory[tp]
            for target_cell_counts in cell_type_counts.values():
                for cell_type, count in target_cell_counts.items():
                    if cell_type not in cell_type_dist:
                        cell_type_dist[cell_type] = 0
                    cell_type_dist[cell_type] += count
                    total_count += count

        # now let's normalize by the total count:
        logging.debug(
            f"Total cell type distribution across all timepoints (raw): {cell_type_dist}"
        )
        for cell_type in cell_type_dist:
            cell_type_dist[cell_type] /= total_count

        logging.debug(
            f"Total cell type distribution across all timepoints (normalized): {cell_type_dist}"
        )
        eval = json.dumps(cell_type_dist, sort_keys=True)
        self.db_manager.insert_eval(
            method, self.__class__.__name__, self._get_param_encoding(), eval
        )

        return cell_type_dist


class PerturbationCellTypeProportionPerTp(PerturbationCellTypeProportionBase):
    def _submetric_eval(self, output_path, trajectory, method):
        logging.debug(f"Trajectory for method {method}: {trajectory}")

        # finally, we want to get all the timepoints from the output file
        perturbed_data = load_output_file(
            os.path.join(output_path, self._get_relative_output_path()),
            RequiredOutputFiles.PERTURBED_TEST_ANN_DATA,
        )
        all_tps = sorted(
            list(perturbed_data.obs[ObservationColumns.TIMEPOINT.value].unique())
        )

        return trajectory, all_tps
