"""
Split anndata object into training and test sets based on timepoints.
"""

from scTimeBench.shared.dataset.base import BaseDatasetPreprocessor
from scTimeBench.shared.constants import ObservationColumns


class TestTimepointSelection(BaseDatasetPreprocessor):
    def __init__(self, config, test_tps, **kwargs):
        super().__init__(config)
        self.test_tps = test_tps
        self.splits = True

    def _parameters(self):
        """
        Return filter-specific parameters.
        """
        return {
            "test_tps": self.test_tps,
            "use_time_indices": self.dataset_dict.get("use_time_indices", False),
        }

    def preprocess(self, ann_data, **kwargs):
        """
        Split the dataset based on the given test timepoints. Must add starting timepoint for proper behaviour.
        """
        # Split the dataset based on the given test_tps
        tp_column = ObservationColumns.TIMEPOINT.value
        timepoints = ann_data.obs[tp_column].unique()
        timepoints = sorted(timepoints)
        start_tp = timepoints[0]

        # Avoid mutating the configured list to keep filter parameters stable.
        test_tps = list(self.test_tps)
        if self._parameters().get("use_time_indices", False):
            # if we're using time indices, we need to convert the test_tps from indices to actual timepoints
            test_tps = [timepoints[tp] for tp in self.test_tps]

        train_tps = [tp for tp in timepoints if tp not in test_tps]

        # add start_tp to test_tps if not already included
        if start_tp not in test_tps:
            test_tps.append(start_tp)

        train_indices = ann_data.obs[ann_data.obs[tp_column].isin(train_tps)].index
        test_indices = ann_data.obs[ann_data.obs[tp_column].isin(test_tps)].index

        train_data = ann_data[train_indices].copy()
        test_data = ann_data[test_indices].copy()

        return train_data, test_data
