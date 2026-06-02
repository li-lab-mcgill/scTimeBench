"""
Cell type proportion perturbation metric.
"""
from scTimeBench.metrics.perturbation.base import PerturbationBasedMetrics


class PerturbationCellTypeProportion(PerturbationBasedMetrics):
    # TODO: make this below work!
    # def _setup_trajectory_inference_model(self):
    # traj_infer_config = self.metric_config.get("trajectory_infer_model", {})

    # assert (
    #     "from_tp_zero" not in traj_infer_config
    #     or traj_infer_config["from_tp_zero"] == self.params["from_tp_zero"]
    # ), "from_tp_zero in trajectory inference config must either not be defined, or match from_tp_zero in metric config."
    # traj_infer_config["from_tp_zero"] = self.params["from_tp_zero"]

    # assert (
    #     "infer_first_tp" not in traj_infer_config
    #     or traj_infer_config["infer_first_tp"] == self.params["infer_first_tp"]
    # ), "infer_first_tp in trajectory inference config must either not be defined, or match infer_first_tp in metric config."
    # traj_infer_config["infer_first_tp"] = self.params["infer_first_tp"]

    # self.trajectory_infer_model = (
    #     TrajectoryInferenceMethodFactory().get_trajectory_infer_method(
    #         traj_infer_config
    #     )
    # )
    # self.params["trajectory_infer_model"] = str(self.trajectory_infer_model)

    def _prep_kwargs_for_submetric_eval(self, output_path, dataset, method):
        pass

    def _submetric_eval(self, graphs, method):
        pass
