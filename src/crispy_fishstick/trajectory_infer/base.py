"""
Base trajectory inference model.

This is the base class for all trajectory inference models, i.e. given an ann data
and its timepoints, we want to infer the trajectory structure.

Examples are the kNN graph-based methods, or the optimal transport based methods.
"""
from crispy_fishstick.shared.constants import ObservationColumns, RequiredOutputColumns
from typing import final
import scanpy as sc
import logging
import json

DEFAULT_METHOD = "kNN"
TRAJECTORY_INFER_METHOD_REGISTRY = {}


def register_trajectory_inference_method(cls):
    """
    Decorator to register a trajectory inference method.
    """
    TRAJECTORY_INFER_METHOD_REGISTRY[cls.__name__] = cls


class BaseTrajectoryInferMethod:
    def __init__(self, traj_config):
        # very simply, this will have the ann_data object
        self.traj_config = traj_config
        # we want this to run either that the method support gene expression
        # if we are using gene expression, or if we're not using gene expression,
        # then it doesn't error out
        assert self.uses_gene_expr() or not self.traj_config.get(
            "use_gene_expr", False
        ), f"Trajectory inference method {self.__class__.__name__} does not support gene expression data."

    def __init_subclass__(cls):
        register_trajectory_inference_method(cls)

    def uses_gene_expr(self):
        """
        Function to be overwritten if the trajectory inference method uses gene expression data.
        By default, we assume it does not.
        """
        return False

    def _method_infer_trajectory(self, ann_data):
        raise NotImplementedError("Subclasses should implement this method.")

    @final
    def infer_trajectory(self, model_output_file):
        ann_data = sc.read_h5ad(model_output_file)

        if self.uses_gene_expr():
            required_obs_columns = [
                ObservationColumns.CELL_TYPE.value,
                ObservationColumns.TIMEPOINT.value,
            ]

            required_obsm_columns = [
                RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION.value,
            ]
        else:
            required_obs_columns = [
                ObservationColumns.CELL_TYPE.value,
                ObservationColumns.TIMEPOINT.value,
            ]

            required_obsm_columns = [
                RequiredOutputColumns.EMBEDDING.value,
                RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING.value,
            ]

        for col in required_obs_columns:
            if col not in ann_data.obs.columns:
                raise ValueError(
                    f"Predicted graph data must have '{col}' in observation metadata."
                )
        for col in required_obsm_columns:
            if col not in ann_data.obsm.keys():
                raise ValueError(
                    f"Predicted graph data must have '{col}' in observation embeddings."
                )

        logging.debug(
            f"Inferring trajectory with method: {self.__class__.__name__} and config: {self.traj_config}"
        )
        return self._method_infer_trajectory(ann_data)

    def __str__(self):
        return json.dumps(
            {
                "method": self.__class__.__name__,
                "config": self.traj_config,
            }
        )


class TrajectoryInferenceMethodFactory:
    def get_trajectory_infer_method(self, traj_config):
        # returns the trajectory inference model based on the config
        method_name = traj_config.get("name", DEFAULT_METHOD)
        if method_name not in TRAJECTORY_INFER_METHOD_REGISTRY:
            raise ValueError(f"Trajectory inference method {method_name} not found.")

        method_class = TRAJECTORY_INFER_METHOD_REGISTRY[method_name]
        logging.debug(f"Using trajectory inference method: {method_name}")
        return method_class(
            traj_config=traj_config,
        )
