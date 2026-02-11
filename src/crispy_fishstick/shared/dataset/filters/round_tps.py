"""
Filter that replaces the time column with rounded values
to ensure that there are enough cells per timepoint.
"""

from crispy_fishstick.shared.dataset.base import BaseDatasetFilter
from crispy_fishstick.shared.constants import ObservationColumns


class RoundingFilter(BaseDatasetFilter):
    def __init__(self, dataset_dict, min_cells_per_timepoint=None, round_to_k=None):
        super().__init__(dataset_dict)
        self.min_cells_per_timepoint = min_cells_per_timepoint
        self.round_to_k = round_to_k

    def _parameters(self):
        """
        Return filter-specific parameters.
        """
        params = {}
        if self.min_cells_per_timepoint is not None:
            params["min_cells_per_timepoint"] = self.min_cells_per_timepoint
        if self.round_to_k is not None:
            params["round_to_k"] = self.round_to_k
        if self.min_cells_per_timepoint is None and self.round_to_k is None:
            # set default to rounding to nearest 1
            params["round_to_k"] = 1
        return params

    def filter(self, ann_data, **kwargs):
        """
        We round until there is a sufficient number of cells per timepoint
        (doing a simple linear approach) where we merge t_i with t_{i + 1}
        if t_i has too few cells, and repeating this.

        If the last timepoint has too few cells, we merge it with the previous timepoint.
        """
        # 1) First we round to the nearest k
        if self.round_to_k is not None:
            ann_data.obs[ObservationColumns.TIMEPOINT.value] = ann_data.obs[
                ObservationColumns.TIMEPOINT.value
            ].apply(lambda x: round(x / self.round_to_k) * self.round_to_k)

        # 2) Then we rounding to the nearest timepoint with enough cells
        if self.min_cells_per_timepoint is not None:
            timepoints = ann_data.obs[ObservationColumns.TIMEPOINT.value].unique()
            timepoints_counts = ann_data.obs[
                ObservationColumns.TIMEPOINT.value
            ].value_counts()

            # build a mapping of timepoints to its rounded timepoints
            # probably easiest to build by copying the timepoints and
            # then merging the ones with too few cells until all have enough cells
            while (
                timepoints_counts.min() < self._parameters()["min_cells_per_timepoint"]
            ):
                # find the timepoint with the fewest cells
                min_timepoint = timepoints_counts.idxmin()

                # if the last timepoint has too few cells, merge it with the previous timepoint
                if min_timepoint == timepoints.max():
                    merge_timepoint = timepoints[timepoints < min_timepoint].max()
                else:
                    merge_timepoint = timepoints[timepoints > min_timepoint].min()

                # merge the two timepoints by replacing the min_timepoint with the merge_timepoint
                ann_data.obs.loc[
                    ann_data.obs[ObservationColumns.TIMEPOINT.value] == min_timepoint,
                    ObservationColumns.TIMEPOINT.value,
                ] = merge_timepoint

                # recalculate the timepoints and their counts
                timepoints = ann_data.obs[ObservationColumns.TIMEPOINT.value].unique()
                timepoints_counts = ann_data.obs[
                    ObservationColumns.TIMEPOINT.value
                ].value_counts()

            print(f"Timepoint counts: {timepoints_counts}")

        return ann_data
