"""
Base trajectory inference model.

This is the base class for all trajectory inference models, i.e. given an ann data
and its timepoints, we want to infer the trajectory structure.

Examples are the kNN graph-based methods, or the optimal transport based methods.
"""
from crispy_fishstick.shared.constants import ObservationColumns, RequiredOutputFiles
from crispy_fishstick.shared.utils import load_test_dataset, load_output_file
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from typing import final
import numpy as np
import pandas as pd
import logging
import json
import hashlib
import os

DEFAULT_METHOD = "Classifier"
TRAJECTORY_INFER_METHOD_REGISTRY = {}
INFERRED_TRAJ_DIR = "trajectory_infer"
INFERRED_TRAJ_FILE = "inferred_trajectory.json"
TRAJ_CONFIG_FILE = "traj_config.json"
NEXT_TP_PROBS_FILE = "next_timepoint_probs.npy"
NEXT_TP_INDICES_FILE = "next_timepoint_indices.npy"
IDX_TO_CELLTYPE_FILE = "index_to_cell_type.json"
INFERRED_TRAJ_PER_TP_FILE = "inferred_trajectory_per_tp.json"


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
        """
        Full parameter set used for hashing and caching.

        Shared fields live here; subclasses should implement `_subclass_parameters`
        to return their custom fields. This keeps hashing consistent without
        requiring subclasses to remember to call `super()`.
        """
        return {
            "test_size": self.traj_config.get("test_size", 0.2),
            "random_state": self.traj_config.get("random_state", 42),
            "use_gene_expr": self.traj_config.get("use_gene_expr", False),
            **self._subclass_parameters(),
        }

    def _subclass_parameters(self):
        """Override in subclasses to add method-specific parameters."""
        return {}

    def _subclass_train(self, X_train, y_train, traj_infer_path):
        """Override in subclasses to implement training logic."""
        raise NotImplementedError("Subclasses should implement this method.")

    def _subclass_predict_probs(self, embeds):
        """
        Override in subclasses to implement training and prediction logic.

        This needs to return two items:
        - predicted probabilities for each cell type
        - index to cell type mapping
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _get_next_tp_tensors(self, output_path, test_ann_data):
        """
        Based on the use_gene_expr property, get the proper tensors for trajectory inference.

        We want to return:
        - Predicted gene expr/embedding at time t (for (1, last t))
        """
        timepoints = test_ann_data.obs[ObservationColumns.TIMEPOINT.value]
        valid_timepoints = np.where(timepoints < timepoints.max())[0]

        if self._parameters()["use_gene_expr"]:
            next_tp_gex = load_output_file(
                output_path, RequiredOutputFiles.NEXT_TIMEPOINT_GENE_EXPRESSION
            )
            return (next_tp_gex[valid_timepoints], valid_timepoints)
        else:
            next_tp_embed = load_output_file(
                output_path, RequiredOutputFiles.NEXT_TIMEPOINT_EMBEDDING
            )
            return (next_tp_embed[valid_timepoints], valid_timepoints)

    def _get_cur_tp_tensors(self, output_path, test_ann_data):
        """
        Based on the use_gene_expr property, get the proper tensors for trajectory inference.

        We want to return:
        - Original gene expr/embedding at time t (for all t)
        """
        if self._parameters()["use_gene_expr"]:
            return test_ann_data.X.toarray()
        else:
            return load_output_file(output_path, RequiredOutputFiles.EMBEDDING)

    def _get_traj_infer_path(self, output_path):
        """
        Get the trajectory inference path based on the hashed config.
        """
        traj_infer_path = os.path.join(output_path, INFERRED_TRAJ_DIR, self.encode())
        os.makedirs(traj_infer_path, exist_ok=True)
        return traj_infer_path

    def _prep_data(self, output_path):
        """
        Prepares the data for trajectory inference by loading from output files.
        Returns the test AnnData and the trajectory inference path.
        """
        test_ann_data = load_test_dataset(output_path)

        required_obs_columns = [
            ObservationColumns.CELL_TYPE.value,
            ObservationColumns.TIMEPOINT.value,
        ]

        for col in required_obs_columns:
            if col not in test_ann_data.obs.columns:
                raise ValueError(
                    f"Test data must have '{col}' in observation metadata."
                )

        # Verify required output files exist
        if self.uses_gene_expr():
            required_files = [RequiredOutputFiles.NEXT_TIMEPOINT_GENE_EXPRESSION]
        else:
            required_files = [
                RequiredOutputFiles.EMBEDDING,
                RequiredOutputFiles.NEXT_TIMEPOINT_EMBEDDING,
            ]

        # Also check for alternative (OT-based) output
        next_celltype_path = os.path.join(
            output_path, RequiredOutputFiles.NEXT_CELLTYPE.value
        )

        if not os.path.exists(next_celltype_path):
            for required_file in required_files:
                file_path = os.path.join(output_path, required_file.value)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Required output file not found: {file_path}"
                    )

        return test_ann_data

    @final
    def train_and_predict(
        self, output_path, test_ann_data=None, traj_infer_path=None, train_only=False
    ):
        """
        Trains and predicts using the trajectory inference model.

        Note that we can specify test_ann_data and traj_infer_path directly to avoid
        re-loading and re-prepping the data if already done, as well as train_only.
        """
        if test_ann_data is None or traj_infer_path is None:
            test_ann_data = self._prep_data(output_path)
            traj_infer_path = self._get_traj_infer_path(output_path)

            # now we also write the traj_config to file for future reference
            with open(os.path.join(traj_infer_path, TRAJ_CONFIG_FILE), "w") as f:
                f.write(str(self))

        # we use the same cached trajectory path so that way we can save classifiers
        # in the future if needed, as it takes time to fit
        # get the embeddings and timepoints
        cell_types = test_ann_data.obs[ObservationColumns.CELL_TYPE.value]

        # filter next timepoint embeddings to only include the valid timepoints
        embeddings = self._get_cur_tp_tensors(output_path, test_ann_data)

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings,
            cell_types,
            test_size=self._parameters()["test_size"],
            random_state=self._parameters()["random_state"],
        )

        # load the classifier model or train a new one if not exists
        logging.debug(f"Training trajectory inference model {self.__class__.__name__}.")
        self._subclass_train(X_train, y_train, traj_infer_path)
        logging.debug(
            f"Predicting trajectory inference model {self.__class__.__name__}."
        )

        return (
            (self._subclass_predict_probs(X_test), y_test) if not train_only else None
        )

    def train_and_predict_k_fold_cv(self, output_path, k):
        """
        Does the train and predict with k-fold cross validation.

        We store everything under traj_infer_path/k_fold_<k>/fold_<i>/
        """
        test_ann_data = self._prep_data(output_path)
        traj_infer_path = self._get_traj_infer_path(output_path)
        k_fold_path = os.path.join(traj_infer_path, f"k_fold_{k}")
        os.makedirs(k_fold_path, exist_ok=True)

        # we use the same cached trajectory path so that way we can save classifiers
        # in the future if needed, as it takes time to fit
        # get the embeddings and timepoints
        cell_types = test_ann_data.obs[ObservationColumns.CELL_TYPE.value]

        # filter next timepoint embeddings to only include the valid timepoints
        embeddings = self._get_cur_tp_tensors(output_path, test_ann_data)

        kf = KFold(
            n_splits=k, shuffle=True, random_state=self._parameters()["random_state"]
        )

        predictions = []
        for train_index, test_index in kf.split(embeddings):
            X_train, X_test = embeddings[train_index], embeddings[test_index]
            y_train, y_test = cell_types.iloc[train_index], cell_types.iloc[test_index]

            fold_path = os.path.join(
                k_fold_path, f"fold_{len(os.listdir(k_fold_path))}"
            )
            os.makedirs(fold_path, exist_ok=True)

            # load the classifier model or train a new one if not exists
            logging.debug(
                f"Training trajectory inference model {self.__class__.__name__} k-fold {len(os.listdir(k_fold_path))}."
            )
            self._subclass_train(X_train, y_train, fold_path)
            logging.debug(
                f"Predicting trajectory inference model {self.__class__.__name__} k-fold {len(os.listdir(k_fold_path))}."
            )
            predictions.append((self._subclass_predict_probs(X_test), y_test))

        return predictions

    def predict_next_tp(self, output_path, test_ann_data=None, traj_infer_path=None):
        """
        Predict the next timepoint cell types using the trajectory inference model.
        """
        if test_ann_data is None or traj_infer_path is None:
            test_ann_data = self._prep_data(output_path)
            traj_infer_path = self._get_traj_infer_path(output_path)

        logging.debug(
            f"Predicting next timepoint with method: {self.__class__.__name__} and config: {self.traj_config}"
        )

        # get the embeddings and timepoints
        next_timepoint_embeddings, indices = self._get_next_tp_tensors(
            output_path, test_ann_data
        )

        next_tp_probs_path = os.path.join(traj_infer_path, NEXT_TP_PROBS_FILE)
        next_tp_idxs_path = os.path.join(traj_infer_path, NEXT_TP_INDICES_FILE)
        idx_to_celltype_path = os.path.join(traj_infer_path, IDX_TO_CELLTYPE_FILE)

        # cache the result of the prediction for faster access later
        if (
            os.path.exists(next_tp_probs_path)
            and os.path.exists(next_tp_idxs_path)
            and os.path.exists(idx_to_celltype_path)
        ):
            logging.debug("Loading cached next timepoint probabilities from disk.")
            return (
                np.load(next_tp_probs_path),
                np.load(next_tp_idxs_path),
                json.load(open(idx_to_celltype_path)),
            )

        next_tp_embed_probs, idx_to_cell_types = self._subclass_predict_probs(
            next_timepoint_embeddings
        )
        np.save(next_tp_probs_path, next_tp_embed_probs)
        np.save(next_tp_idxs_path, indices)
        with open(idx_to_celltype_path, "w") as f:
            json.dump(idx_to_cell_types, f)

        return next_tp_embed_probs, indices, idx_to_cell_types

    @final
    def infer_trajectory(self, output_path, per_tp=False):
        """
        Infer the trajectory using kNN graph-based method.

        1. We can accomplish this by first separating each embedding based on time.
        2. Then, for each time point, we find the k nearest neighbors in the next time point's
        embedding space.
        3. Finally, we consolidate the cell types per time point based on the kNN results.
        """
        traj_infer_path = self._get_traj_infer_path(output_path)
        logging.debug(
            f"Inferring trajectory with method: {self.__class__.__name__} and config: {self.traj_config}"
        )

        # cache the inferred trajectory in a new folder under
        # trajectory_infer/<hash-of-traj-model>/
        # because the hash should be the same, and we write out the traj config
        # there is no need to store in the database
        # and also store under trajectory_infer/<hash-of-traj-model>/traj_config.yaml
        # and trajectory_infer/<hash-of-traj-model>/inferred_trajectory.json
        if not per_tp and os.path.exists(
            os.path.join(traj_infer_path, INFERRED_TRAJ_FILE)
        ):
            logging.info(
                f"Inferred trajectory already exists at {traj_infer_path}, loading from file."
            )
            with open(os.path.join(traj_infer_path, INFERRED_TRAJ_FILE), "r") as f:
                inferred_traj = json.load(f)
            return inferred_traj
        elif per_tp and os.path.exists(
            os.path.join(traj_infer_path, INFERRED_TRAJ_PER_TP_FILE)
        ):
            logging.info(
                f"Inferred trajectory already exists at {traj_infer_path}, loading from file."
            )
            with open(
                os.path.join(traj_infer_path, INFERRED_TRAJ_PER_TP_FILE), "r"
            ) as f:
                inferred_traj = json.load(f)
            return inferred_traj

        test_ann_data = self._prep_data(output_path)

        # now we also write the traj_config to file for future reference
        with open(os.path.join(traj_infer_path, TRAJ_CONFIG_FILE), "w") as f:
            f.write(str(self))

        # subset for only cells that are not at the last timepoint
        timepoints = test_ann_data.obs[ObservationColumns.TIMEPOINT.value]
        valid_timepoints = np.where(timepoints < timepoints.max())[0]
        cell_tps = test_ann_data.obs[ObservationColumns.TIMEPOINT.value].iloc[
            valid_timepoints
        ]
        cell_types = test_ann_data.obs[ObservationColumns.CELL_TYPE.value].iloc[
            valid_timepoints
        ]

        # ** Note: if the NEXT_CELLTYPE file already exists, then use that instead (OT methods) **
        next_celltype_path = os.path.join(
            output_path, RequiredOutputFiles.NEXT_CELLTYPE.value
        )
        # this next block gets the next cell type for us, matching the same indices as valid_timepoints
        if os.path.exists(next_celltype_path):
            logging.debug(
                f"Next cell type file found at {next_celltype_path}. Using it for trajectory inference."
            )
            # load the next cell types from the parquet file
            next_celltype_df = pd.read_parquet(next_celltype_path)
            next_cell_types = next_celltype_df[
                ObservationColumns.CELL_TYPE.value
            ].values
            next_cell_types_valid = next_cell_types[valid_timepoints]

        else:
            logging.debug(
                f"No next cell type file found at {next_celltype_path}. Using trajectory inference model to predict next cell types."
            )
            # always start by training to ensure that the model is fitted
            # we need to do the train_only option here instead of calling _train directly
            # because it does some preprocessing we don't want to repeat
            # this trains a classifier on all the data points
            self.train_and_predict(
                output_path, test_ann_data, traj_infer_path, train_only=True
            )
            logging.debug(
                f"Inferring trajectory using trajectory inference model: {self.__class__.__name__}"
            )
            # then we run the predict next timepoint to get the embeddings
            next_tp_embed_probs, indices, idx_to_cell_types = self.predict_next_tp(
                output_path, test_ann_data, traj_infer_path
            )
            next_cell_types_valid = [
                idx_to_cell_types[np.argmax(probs)] for probs in next_tp_embed_probs
            ]
            assert all(
                indices == valid_timepoints.tolist()
            ), "Indices from trajectory inference do not match valid timepoints."

        inferred_traj = {}

        if not per_tp:
            for cur_cell, next_cell in zip(cell_types, next_cell_types_valid):
                if cur_cell not in inferred_traj:
                    inferred_traj[cur_cell] = {}
                if next_cell not in inferred_traj[cur_cell]:
                    inferred_traj[cur_cell][next_cell] = 0
                inferred_traj[cur_cell][next_cell] += 1

            logging.debug(f"Constructed cell lineage (raw counts): {inferred_traj}")

            # then we should normalize the counts to get probabilities
            for source_cell_type in inferred_traj.keys():
                total_counts = sum(inferred_traj[source_cell_type].values())
                for target_cell_type in inferred_traj[source_cell_type]:
                    inferred_traj[source_cell_type][target_cell_type] /= total_counts

            with open(os.path.join(traj_infer_path, INFERRED_TRAJ_FILE), "w") as f:
                json.dump(inferred_traj, f)

            return inferred_traj

        # now it's per tp
        for cur_cell, next_cell, cell_tp in zip(
            cell_types, next_cell_types_valid, cell_tps
        ):
            if cell_tp not in inferred_traj:
                inferred_traj[cell_tp] = {}
            if cur_cell not in inferred_traj[cell_tp]:
                inferred_traj[cell_tp][cur_cell] = {}
            if next_cell not in inferred_traj[cell_tp][cur_cell]:
                inferred_traj[cell_tp][cur_cell][next_cell] = 0

            inferred_traj[cell_tp][cur_cell][next_cell] += 1

        logging.debug(f"Constructed cell lineage (raw counts): {inferred_traj}")

        # we don't normalize the per-tp counts
        with open(os.path.join(traj_infer_path, INFERRED_TRAJ_PER_TP_FILE), "w") as f:
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


class TrajectoryInferenceMethodFactory:
    def get_trajectory_infer_method(self, traj_config) -> BaseTrajectoryInferMethod:
        # returns the trajectory inference model based on the config
        method_name = traj_config.get("name", DEFAULT_METHOD)
        if method_name not in TRAJECTORY_INFER_METHOD_REGISTRY:
            raise ValueError(f"Trajectory inference method {method_name} not found.")

        method_class = TRAJECTORY_INFER_METHOD_REGISTRY[method_name]
        logging.debug(f"Using trajectory inference method: {method_name}")
        return method_class(
            traj_config=traj_config,
        )
