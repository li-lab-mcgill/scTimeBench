"""
Filter that replaces the time column with a psupertime.
"""

from crispy_fishstick.shared.dataset.base import BaseDatasetFilter
from crispy_fishstick.shared.constants import ObservationColumns
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


class BasePseudotimeFilter(BaseDatasetFilter):
    def __init__(self, dataset_dict, preprocess_type):
        super().__init__(dataset_dict)
        self.PCA_FILE = "pca_model.pkl"
        self.preprocess_type = PreprocessType(preprocess_type)
        self.PSEUDOTIME_FILE = "pseudotime.npy"

    def _parameters(self):
        """
        Return filter-specific parameters.
        """
        params = {
            "preprocess_type": self.preprocess_type.value,
            "n_cells_train": self.dataset_dict.get("n_cells_train", 1000),
        }

        if self.preprocess_type == PreprocessType.PCA:
            params["pca_components"] = self.dataset_dict.get("pca_components", 50)
        elif self.preprocess_type == PreprocessType.HVG:
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

    def filter(self, ann_data, dataset_dir):
        """
        Use some pseudotime estimation method (to be implemented by subclasses) to replace the time column with a pseudotime.

        Preprocessing steps:
        1) PCA
        Because the gene space tends to be way too big, and this is slowing things down tremendously,
        we perform PCA on the data and use the top n components as input to the pseudotime estimation.

        We also save this to a file under the dataset directory to be used later.

        2) Subset the data per timepoint so that we only use n_cells_train cells.
        """

        # by default we will cache to dataset_dir/pseudotime.npy
        cache_path = os.path.join(dataset_dir, self.PSEUDOTIME_FILE)
        if os.path.exists(cache_path):
            logging.debug(
                f"Cached pseudotime file already exists at {cache_path}. Loading pseudotime from cache."
            )
            pseudotime = np.load(cache_path)
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
            pca_path = os.path.join(dataset_dir, self.PCA_FILE)
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

        pseudotime = self._filter_pseudotime(preprocessed_ann_data)
        ann_data.obs[ObservationColumns.TIMEPOINT.value] = pseudotime
        np.save(cache_path, pseudotime)
        return ann_data

    def _select_train_data(self, ann_data):
        # 2) Subset the data per timepoint so that we only use n_cells_train cells
        # for the pseudotime estimation to speed things up (since some pseudotime estimation methods can be quite slow, especially on large datasets).
        # Then we would apply this model on the full data to get the pseudotime for all cells.
        train_ann_data = ann_data.copy()
        if ann_data.n_obs > self._parameters()["n_cells_train"]:
            # we would want to make sure that we have a balanced number of cells
            # from each timepoint, so we would sample n_cells_train / n_timepoints
            # cells from each timepoint to make sure that we have a balanced representation
            # of each timepoint in the training data for pseudotime estimation.
            logging.debug(
                f"Filtering for {self._parameters()['n_cells_train']} random cells out of {ann_data.n_obs} to speed up pseudotime estimation for debugging..."
            )

            time_col = ObservationColumns.TIMEPOINT.value
            timepoints = list(ann_data.obs[time_col].unique())
            n_timepoints = len(timepoints)
            n_cells_train = self._parameters()["n_cells_train"]
            base_per_tp = int(n_cells_train // n_timepoints)

            rng = np.random.default_rng()
            selected_indices = []
            # keep track of which timepoints have remaining cells to sample from
            # in case we need to do additional rounds of sampling to fill the remaining budget
            remaining_by_tp = {}

            for tp in timepoints:
                tp_mask = ann_data.obs[ObservationColumns.TIMEPOINT.value] == tp
                tp_indices = np.where(tp_mask)[0]
                permuted = rng.permutation(tp_indices)

                take = min(base_per_tp, permuted.size)
                if take > 0:
                    selected_indices.extend(permuted[:take].tolist())

                if permuted.size > take:
                    remaining_by_tp[tp] = permuted[take:]

            remaining_budget = n_cells_train - len(selected_indices)
            while remaining_budget > 0 and remaining_by_tp:
                available_tps = list(remaining_by_tp.keys())
                base_take = remaining_budget // len(available_tps)
                extra = remaining_budget % len(available_tps)
                to_remove = []
                any_taken = False

                for i, tp in enumerate(available_tps):
                    tp_remaining = remaining_by_tp[tp]
                    take = base_take + (1 if i < extra else 0)
                    if take == 0 and remaining_budget > 0:
                        take = 1
                    take = min(take, tp_remaining.size)

                    if take > 0:
                        selected_indices.extend(tp_remaining[:take].tolist())
                        remaining_budget -= take
                        any_taken = True

                    if tp_remaining.size == take:
                        to_remove.append(tp)
                    else:
                        remaining_by_tp[tp] = tp_remaining[take:]

                    if remaining_budget == 0:
                        break

                for tp in to_remove:
                    remaining_by_tp.pop(tp, None)

                if not any_taken:
                    break

            # finally let's verify that there are no duplicated entries:
            assert len(selected_indices) == len(
                set(selected_indices)
            ), "There are duplicated entries in the selected indices for pseudotime estimation training data."
            train_ann_data = ann_data[np.array(selected_indices, dtype=int)].copy()

            # finally logging.debug the new counts
            logging.debug(
                f"Cell counts by timepoint: {train_ann_data.obs[ObservationColumns.TIMEPOINT.value].value_counts()}",
            )

        return train_ann_data

    def _filter_pseudotime(self, preprocessed_ann_data):
        """
        Filter the dataset to replace its time column with a pseudotime.
        This should return an ann_data with the same number of cells,
        but with the time column replaced by the pseudotime.

        This is a placeholder method to be implemented by subclasses, as different pseudotime estimation methods may have different requirements for the input data and the output pseudotime.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class PsupertimeFilter(BasePseudotimeFilter):
    def _parameters(self):
        """
        Return filter-specific parameters.
        """
        return {
            **super()._parameters(),
            "n_cpus": self.dataset_dict.get("n_cpus", 20),
        }

    def _filter_pseudotime(self, preprocessed_ann_data):
        """
        Filter the dataset to replace its time column with a psupertime.
        """
        from pypsupertime import Psupertime
        from scipy.stats import spearmanr

        # let's turn off numba
        logging.getLogger("numba").setLevel(logging.WARNING)

        # then let's run psupertime!
        psup = Psupertime(
            n_jobs=self._parameters()["n_cpus"],
            n_folds=3,
        )

        # let's first preprocess on all the data
        preprocessed_ann_data = psup.preprocessing.fit_transform(preprocessed_ann_data)
        train_preprocessed_ann_data = self._select_train_data(preprocessed_ann_data)

        # now let's avoid preprocessing during the run
        psup.preprocessing = None
        train_preprocessed_ann_data = psup.run(
            train_preprocessed_ann_data,
            ObservationColumns.TIMEPOINT.value,
        )

        # now that we've done the training, let's apply it to the full data to get the pseudotime for all cells
        # then let's first preprocess the data as the train data
        psup.predict_psuper(preprocessed_ann_data, inplace=True)

        # now to get a good idea on how well the pseudotime estimation is doing, let's check the spearman correlation
        spearman_corr = spearmanr(
            preprocessed_ann_data.obs[ObservationColumns.TIMEPOINT.value],
            preprocessed_ann_data.obs["psupertime"],
        )
        logging.debug(f"Spearman correlation: {spearman_corr}")

        logging.debug(
            f'Psupertime observation: {preprocessed_ann_data.obs["psupertime"]}'
        )
        return preprocessed_ann_data.obs["psupertime"]
