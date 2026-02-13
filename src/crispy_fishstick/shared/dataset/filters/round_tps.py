"""
Filter that replaces the time column with rounded values
to ensure that there are enough cells per timepoint.
"""

from crispy_fishstick.shared.dataset.base import BaseDatasetFilter
from crispy_fishstick.shared.constants import ObservationColumns
import logging


class RoundingFilter(BaseDatasetFilter):
    def __init__(
        self,
        dataset_dict,
        min_cells_per_timepoint=None,
        round_to_k=None,
        num_tps=None,
        even_cells_per_tp=True,
    ):
        super().__init__(dataset_dict)
        self.min_cells_per_timepoint = min_cells_per_timepoint
        self.round_to_k = round_to_k
        self.num_tps = num_tps
        self.even_cells_per_tp = even_cells_per_tp

    def _parameters(self):
        """
        Return filter-specific parameters.
        """
        params = {}
        if self.min_cells_per_timepoint is not None:
            params["min_cells_per_timepoint"] = self.min_cells_per_timepoint
        if self.round_to_k is not None:
            params["round_to_k"] = self.round_to_k
        if self.num_tps is not None:
            # round the minimum cells until num tps is achieved
            params["num_tps"] = self.num_tps
            params["even_cells_per_tp"] = self.even_cells_per_tp
        if (
            self.min_cells_per_timepoint is None
            and self.round_to_k is None
            and self.num_tps is None
        ):
            # set the default to round to 10 tps
            params["num_tps"] = 10
        return params

    def filter(self, ann_data, **kwargs):
        """
        We round until there is a sufficient number of cells per timepoint
        (doing a simple linear approach) where we merge t_i with t_{i + 1}
        if t_i has too few cells, and repeating this.

        If the last timepoint has too few cells, we merge it with the previous timepoint.
        """
        # 1) First we round to the nearest k
        if self.round_to_k is not None or (
            self.num_tps is not None and not self.even_cells_per_tp
        ):
            # if we are doing even tps and not even cells, we set the round to k to
            # the range of tps / num_tps to get an even distribution of tps
            if self.num_tps is not None:
                round_to_k = (
                    ann_data.obs[ObservationColumns.TIMEPOINT.value].max()
                    - ann_data.obs[ObservationColumns.TIMEPOINT.value].min()
                ) / self.num_tps
            else:
                round_to_k = self.round_to_k

            ann_data.obs[ObservationColumns.TIMEPOINT.value] = ann_data.obs[
                ObservationColumns.TIMEPOINT.value
            ].apply(lambda x: round(x / round_to_k) * round_to_k)

        # 2) Then we round to the nearest timepoint with enough cells
        if self.min_cells_per_timepoint is not None or (
            self.num_tps is not None and self.even_cells_per_tp
        ):
            timepoints = ann_data.obs[ObservationColumns.TIMEPOINT.value].unique()
            timepoints_counts = ann_data.obs[
                ObservationColumns.TIMEPOINT.value
            ].value_counts()

            # build a mapping of timepoints to its rounded timepoints
            # probably easiest to build by copying the timepoints and
            # then merging the ones with too few cells until all have enough cells
            # until either we have enough cells per timepoint, or we have the desired number of timepoints
            while (
                self.min_cells_per_timepoint is not None
                and timepoints_counts.min()
                < self._parameters()["min_cells_per_timepoint"]
            ) or (
                self.num_tps is not None
                and len(timepoints) > self._parameters()["num_tps"]
            ):
                # find the timepoint with the fewest cells
                min_timepoint = timepoints_counts.idxmin()

                # if the last timepoint has too few cells, merge it with the previous timepoint
                if min_timepoint == timepoints.max():
                    merge_timepoint = timepoints[timepoints < min_timepoint].max()
                elif min_timepoint == timepoints.min():
                    merge_timepoint = timepoints[timepoints > min_timepoint].min()
                else:
                    # here we merge with either the next or the earlier timepoint, depending on which one
                    # has fewer cells (we want to merge with the one with fewer cells to get more balanced timepoints)
                    prev_timepoint = timepoints[timepoints < min_timepoint].max()
                    next_timepoint = timepoints[timepoints > min_timepoint].min()
                    merge_timepoint = (
                        next_timepoint
                        if timepoints_counts[next_timepoint]
                        < timepoints_counts[prev_timepoint]
                        else prev_timepoint
                    )

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

            logging.debug(f"Timepoint counts: {timepoints_counts}")

        return ann_data
