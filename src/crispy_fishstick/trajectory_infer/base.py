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
import hashlib
from pathlib import Path
import os

DEFAULT_METHOD = "kNN"
TRAJECTORY_INFER_METHOD_REGISTRY = {}
INFERRED_TRAJ_DIR = "trajectory_infer"
INFERRED_TRAJ_FILE = "inferred_trajectory.json"
TRAJ_CONFIG_FILE = "traj_config.json"


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

    def _parameters(self):
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

        # cache the inferred trajectory in a new folder under
        # trajectory_infer/<hash-of-traj-model>/
        # because the hash should be the same, and we write out the traj config
        # there is no need to store in the database
        # and also store under trajectory_infer/<hash-of-traj-model>/traj_config.yaml
        # and trajectory_infer/<hash-of-traj-model>/inferred_trajectory.json
        model_output_path = Path(model_output_file).parent
        traj_infer_path = os.path.join(
            model_output_path, INFERRED_TRAJ_DIR, self.encode()
        )

        os.makedirs(traj_infer_path, exist_ok=True)

        if os.path.exists(os.path.join(traj_infer_path, INFERRED_TRAJ_FILE)):
            logging.info(
                f"Inferred trajectory already exists at {traj_infer_path}, loading from file."
            )
            with open(os.path.join(traj_infer_path, INFERRED_TRAJ_FILE), "r") as f:
                inferred_traj = json.load(f)
            return inferred_traj

        # now we also write the traj_config to file for future reference
        with open(os.path.join(traj_infer_path, TRAJ_CONFIG_FILE), "w") as f:
            json.dump(self.traj_config, f)

        inferred_traj = self._method_infer_trajectory(ann_data)

        with open(os.path.join(traj_infer_path, INFERRED_TRAJ_FILE), "w") as f:
            json.dump(inferred_traj, f)

        return inferred_traj

    def __str__(self):
        return json.dumps(
            {
                "method": self.__class__.__name__,
                "parameters": self._parameters(),
            }
        )

    def encode(self):
        """
        Hash the trajectory inference method based on its class name and parameters.
        """
        return hashlib.md5(str(self).encode()).hexdigest()

    def _classification_entropy(self, ann_data):
        return 1  # placeholder implementation
        raise NotImplementedError("Subclasses should implement this method.")

    @final
    def evaluate_classification_entropy(self, model_output_file):
        """
        Evaluate the classification entropy of the inferred trajectory.

        This function computes the classification entropy based on the predicted
        trajectories and the cell type distributions at each time point.
        """
        ann_data = sc.read_h5ad(model_output_file)

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
            f"Evaluating classification entropy with method: {self.__class__.__name__} and config: {self.traj_config}"
        )

        return self._classification_entropy(ann_data)


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
