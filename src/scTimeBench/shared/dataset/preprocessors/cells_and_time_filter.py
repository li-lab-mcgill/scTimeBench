from scTimeBench.shared.dataset.base import BaseDatasetPreprocessor
from scTimeBench.shared.constants import ObservationColumns
import scanpy as sc


class CellsAndTimeFilter(BaseDatasetPreprocessor):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.params = kwargs
        self.splits = True

    def _parameters(self):
        """
        Return filter-specific parameters.
        """
        return {
            "start_tp_idx": self.params.get("start_tp_idx", 0),
            "end_tp_idx": self.params.get("end_tp_idx", None),
            "cell_type": self.params.get("cell_type", None),
        }

    def preprocess(self, ann_data: sc.AnnData, **kwargs):
        """
        Filter the dataset to only include cells present in the lineage information.
        """
        train_ann_data = ann_data.copy()

        start_tp_idx = self._parameters()["start_tp_idx"]
        end_tp_idx = self._parameters()["end_tp_idx"]
        cell_type = self._parameters()["cell_type"]

        tps = ann_data.obs[ObservationColumns.TIMEPOINT.value].unique()
        tps = sorted(tps)
        start_tp = tps[start_tp_idx]
        end_tp = tps[end_tp_idx] if end_tp_idx is not None else None

        if cell_type is not None:
            ann_data = ann_data[
                ann_data.obs[ObservationColumns.CELL_TYPE.value] == cell_type
            ]

        if end_tp is not None:
            ann_data = ann_data[
                (ann_data.obs[ObservationColumns.TIMEPOINT.value] >= start_tp)
                & (ann_data.obs[ObservationColumns.TIMEPOINT.value] <= end_tp)
            ]
        else:
            ann_data = ann_data[
                ann_data.obs[ObservationColumns.TIMEPOINT.value] >= start_tp
            ]

        return train_ann_data, ann_data
