from scTimeBench.shared.dataset.base import BaseDatasetFilter
from scTimeBench.shared.constants import ObservationColumns
import random


class NaiveSplitDatasetFilter(BaseDatasetFilter):
    def __init__(self, config, train_pct):
        super().__init__(config)
        self.train_pct = train_pct
        self.splits = True

    def _parameters(self):
        """
        Return filter-specific parameters.
        """
        return {
            "train_pct": self.train_pct,
        }

    def filter(self, ann_data, **kwargs):
        """
        Filter the dataset to only include cells present in the lineage information.
        """
        # naive split the dataset based on the given train_pct, per time point
        # to ensure that each time point has representation in both train and test sets
        train_indices = []
        test_indices = []
        tp_column = ObservationColumns.TIMEPOINT.value
        timepoints = ann_data.obs[tp_column].unique()
        for tp in timepoints:
            tp_data = ann_data[ann_data.obs[tp_column] == tp]
            n_train = int(tp_data.n_obs * self.train_pct)
            indices = list(range(tp_data.n_obs))
            random.shuffle(indices)
            train_indices.extend(tp_data.obs.index[indices[:n_train]])
            test_indices.extend(tp_data.obs.index[indices[n_train:]])
        train_data = ann_data[train_indices].copy()
        test_data = ann_data[test_indices].copy()
        return train_data, test_data
