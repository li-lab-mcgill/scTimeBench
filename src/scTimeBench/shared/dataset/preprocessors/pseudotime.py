"""
Filter that replaces the time column with a psupertime.
"""

from scTimeBench.shared.dataset.base import BaseDatasetPreprocessor
from scTimeBench.shared.constants import ObservationColumns
import scanpy as sc
from sklearn.decomposition import PCA
import logging
import numpy as np
import joblib
import os

from enum import Enum


class PreprocessType(Enum):
    NONE = "none"
    PCA = "pca"
    HVG = "hvg"
    ZHENG_HVG = "zheng_hvg"


class BasePseudotimePreprocessor(BaseDatasetPreprocessor):
    def __init__(self, dataset_dict, preprocess_type, **kwargs):
        super().__init__(dataset_dict)
        self.PCA_FILE = "pca_model.pkl"
        self.preprocess_type = PreprocessType(preprocess_type)
        self.PSEUDOTIME_FILE = self.label() + ".npy"

    def label(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _parameters(self):
        """
        Return filter-specific parameters.
        """
        params = {
            "preprocess_type": self.preprocess_type.value,
        }

        if self.preprocess_type == PreprocessType.PCA:
            params["pca_components"] = self.dataset_dict.get("pca_components", 50)
        elif self.preprocess_type == PreprocessType.HVG:
            params["n_top_genes"] = self.dataset_dict.get("n_top_genes", 1000)
            params["n_cells_train"] = self.dataset_dict.get("n_cells_train", 1000)
        elif self.preprocess_type == PreprocessType.ZHENG_HVG:
            params["n_top_genes"] = self.dataset_dict.get("n_top_genes", 1000)

        return {
            **super()._parameters(),
            **params,
        }

    def requires_caching(self):
        """
        Some of these packages might not be installed elsewhere and/or will take
        a long time to load. Pseudotime is one such filter, and so we cache it ahead of time.
        """
        return True

    def preprocess(self, ann_data, checkpoint_dir):
        """
        Use some pseudotime estimation method (to be implemented by subclasses) to replace the time column with a pseudotime.

        Preprocessing steps:
        1) PCA
        Because the gene space tends to be way too big, and this is slowing things down tremendously,
        we perform PCA on the data and use the top n components as input to the pseudotime estimation.

        We also save this to a file under the dataset directory to be used later.

        2) Subset the data per timepoint so that we only use n_cells_train cells.
        """
        from scipy.stats import spearmanr

        # by default we will cache to dataset_dir/pseudotime.npy
        cache_path = os.path.join(checkpoint_dir, self.PSEUDOTIME_FILE)
        if os.path.exists(cache_path):
            logging.debug(
                f"Cached pseudotime file already exists at {cache_path}. Loading pseudotime from cache."
            )
            pseudotime = np.load(cache_path)

            # now to get a good idea on how well the pseudotime estimation is doing, let's check the spearman correlation
            spearman_corr = spearmanr(
                ann_data.obs[ObservationColumns.TIMEPOINT.value],
                pseudotime,
            )
            logging.debug(f"Spearman correlation: {spearman_corr}")

            ann_data.obs[ObservationColumns.TIMEPOINT.value] = pseudotime
            return ann_data

        # 1) PCA and/or HVG
        # let's turn off numba
        logging.getLogger("numba").setLevel(logging.WARNING)
        if self._parameters()["preprocess_type"] == PreprocessType.HVG.value:
            sc.pp.highly_variable_genes(
                ann_data, n_top_genes=self._parameters()["n_top_genes"], inplace=True
            )
            preprocessed_ann_data = ann_data[:, ann_data.var.highly_variable].copy()
            logging.debug(
                f"Selected {preprocessed_ann_data.n_vars} highly variable genes for pseudotime estimation.",
            )

        elif self._parameters()["preprocess_type"] == PreprocessType.PCA.value:
            # we perform PCA and use the top n components as input to the pseudotime estimation.
            pca_path = os.path.join(checkpoint_dir, self.PCA_FILE)
            if os.path.exists(pca_path):
                logging.debug(
                    f"PCA model already exists at {pca_path}. Loading PCA model."
                )
                pca_model = joblib.load(pca_path)
            else:
                pca_model = PCA(n_components=self._parameters()["pca_components"]).fit(
                    ann_data.X
                )
                # now we save the pca data to a file under the dataset directory to be used later
                joblib.dump(pca_model, pca_path)
            pca_data = pca_model.transform(ann_data.X)
            preprocessed_ann_data = sc.AnnData(X=pca_data, obs=ann_data.obs.copy())

        elif self._parameters()["preprocess_type"] == PreprocessType.ZHENG_HVG.value:
            preprocessed_ann_data = sc.pp.recipe_zheng17(
                ann_data, n_top_genes=self._parameters()["n_top_genes"], copy=True
            )

        else:
            preprocessed_ann_data = ann_data.copy()

        pseudotime = self._filter_pseudotime(preprocessed_ann_data)

        # now to get a good idea on how well the pseudotime estimation is doing, let's check the spearman correlation
        spearman_corr = spearmanr(
            ann_data.obs[ObservationColumns.TIMEPOINT.value],
            pseudotime,
        )
        logging.debug(f"Spearman correlation: {spearman_corr}")

        ann_data.obs[ObservationColumns.TIMEPOINT.value] = pseudotime
        os.makedirs(checkpoint_dir, exist_ok=True)
        np.save(cache_path, pseudotime)
        logging.debug(f"Caching to {cache_path}...")
        return ann_data

    def _filter_pseudotime(self, preprocessed_ann_data):
        """
        Filter the dataset to replace its time column with a pseudotime.
        This should return an ann_data with the same number of cells,
        but with the time column replaced by the pseudotime.

        This is a placeholder method to be implemented by subclasses, as different pseudotime estimation methods may have different requirements for the input data and the output pseudotime.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class Psupertime(BasePseudotimePreprocessor):
    """
    @DEPRECATION: This is a deprecated filter and will break if used. See the setup at commit history:
    c501310953af8391ddfdaa2b73af324f73fdd9c3
    for a reference on how to set things up.
    """

    def label(self):
        return "Psupertime"

    def _parameters(self):
        """
        Return filter-specific parameters.
        """
        return {
            **super()._parameters(),
            "n_cpus": self.dataset_dict.get("n_cpus", 10),
        }

    def _filter_pseudotime(self, preprocessed_ann_data):
        """
        Filter the dataset to replace its time column with a psupertime.
        """
        from pypsupertime import Psupertime
        from pypsupertime.model import SGDModel

        logging.warning(
            f"""
            WARNING: Psupertime is very slow, and does not produce the necessary time labels that is useful for downstream tasks.
            We recommend using Pseudotime instead.
            """
        )

        # let's turn off numba
        logging.getLogger("numba").setLevel(logging.WARNING)

        # then let's run psupertime!
        # let's first preprocess on all the data
        psup = Psupertime(
            n_jobs=self._parameters()["n_cpus"],
            n_folds=2,
            n_batches=1,
            preprocessing_params={"select_genes": "hvg", "hvg_n_top_genes": 1000},
            estimator_class=SGDModel,
            estimator_params={
                "max_iter": 25,
                "n_iter_no_change": 4,
                "early_stopping": False,
            },
            regularization_params={
                "n_params": 5,
            },
        )

        preprocessed_ann_data = psup.preprocessing.fit_transform(
            preprocessed_ann_data,
        )

        # now let's avoid preprocessing during the run
        psup.preprocessing = None
        preprocessed_ann_data = psup.run(
            preprocessed_ann_data,
            ObservationColumns.TIMEPOINT.value,
        )

        logging.debug(
            f'Psupertime observation: {preprocessed_ann_data.obs["psupertime"]}'
        )
        return preprocessed_ann_data.obs["psupertime"]


class Sceptic(BasePseudotimePreprocessor):
    """
    @DEPRECATION: This is a deprecated filter and will break if used. See the setup at commit history:
    c501310953af8391ddfdaa2b73af324f73fdd9c3
    for a reference on how to set things up.

    Implementation of sceptic psuedotime filter. See more: https://github.com/Noble-Lab/Sceptic
    and the paper: https://link.springer.com/article/10.1186/s13059-025-03679-3

    NOTE: this works pretty well with PCA = 50, doesn't take too long and
    gives reasonable spearman correlations (0.8) on GarciaAlonso.
    """

    def label(self):
        return "Sceptic"

    def _filter_pseudotime(self, preprocessed_ann_data):
        """
        Filter the dataset to replace its time column with a psupertime.
        """
        from sceptic import run_sceptic_and_evaluate
        import torch

        logging.warning(
            f"""
            WARNING: Sceptic does not produce the necessary time labels that is useful for downstream tasks.
            We recommend using Pseudotime instead.
            """
        )

        # Option 1: Pass actual time values directly (easiest!)
        time_labels = preprocessed_ann_data.obs[
            ObservationColumns.TIMEPOINT.value
        ].values

        # let's create the unique times and sort them too
        unique_time_labels = np.sort(np.unique(time_labels))
        logging.debug(f"Unique time labels: {unique_time_labels}")

        # now we need to transform these time labels to categorical ones and pass in the
        # ordinal list as well
        # Convert labels to categorical values
        time_to_categorical_dict = {
            label: idx for idx, label in enumerate(unique_time_labels)
        }
        logging.debug(f"Time to categorical mapping: {time_to_categorical_dict}")

        label = np.array([time_to_categorical_dict[t] for t in time_labels])
        logging.debug(f"Encoded labels (first 10): {label[:10]}")
        logging.debug(f"Non-encoded labels (first 10): {time_labels[:10]}")

        data = (
            preprocessed_ann_data.X
            if isinstance(preprocessed_ann_data.X, np.ndarray)
            else preprocessed_ann_data.X.toarray()
        )

        cm, pred, pseudotime, prob = run_sceptic_and_evaluate(
            data,
            labels=label,
            label_list=unique_time_labels,
            method="xgboost",
            use_gpu=torch.cuda.is_available(),
        )

        logging.debug(f"Sceptic confusion matrix:\n{cm}")
        logging.debug(f"Sceptic predicted labels:\n{pred}")
        logging.debug(f"\nAccuracy: {np.sum(np.diag(cm)) / np.sum(cm):.3f}")
        logging.debug(f"Sceptic pseudotime values:\n{pseudotime}")
        logging.debug(f"Sceptic class probabilities:\n{prob}")

        return pseudotime


class Pseudotime(BasePseudotimePreprocessor):
    def label(self):
        return "Pseudotime"

    def _filter_pseudotime(self, preprocessed_ann_data):
        """
        Filter the dataset to replace its time column with a pseudotime.
        We will use scanpy's DFT pseudotime: https://scanpy.readthedocs.io/en/latest/tutorials/trajectories/paga-paul15.html
        See the above as an example tutorial.
        """
        # assume that we have pca embeddings
        # 1. Compute neighbors using your denoised representation (scVI)
        sc.pp.neighbors(preprocessed_ann_data, n_neighbors=15)
        sc.tl.diffmap(preprocessed_ann_data)

        # 3. Identify a 'root cell' (the start of your trajectory)
        # You can do this by picking a cell from your earliest timepoint
        earliest_tp = preprocessed_ann_data.obs[
            ObservationColumns.TIMEPOINT.value
        ].min()

        root_idx = preprocessed_ann_data.obs[
            preprocessed_ann_data.obs[ObservationColumns.TIMEPOINT.value] == earliest_tp
        ].index[0]
        preprocessed_ann_data.uns["iroot"] = preprocessed_ann_data.obs_names.get_loc(
            root_idx
        )

        # 4. Calculate Diffusion Pseudotime
        sc.tl.dpt(preprocessed_ann_data)
        logging.debug(f'Pseudotime: {preprocessed_ann_data.obs["dpt_pseudotime"]}')
        return preprocessed_ann_data.obs["dpt_pseudotime"]
