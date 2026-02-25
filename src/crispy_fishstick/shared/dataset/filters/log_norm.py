from crispy_fishstick.shared.dataset.base import BaseDatasetFilter
from crispy_fishstick.shared.utils import is_log_normalized_to_counts
import scanpy as sc
import logging


class LogNormFilter(BaseDatasetFilter):
    """
    Filter that normalizes all the raw counts to 10^4, then log norms it,
    and puts it into X.
    """

    def __init__(self, config):
        super().__init__(config)

    def _parameters(self):
        """
        Return filter-specific parameters.
        """
        return {
            "counts": 10_000,  # default is 10^4
        }

    def filter(self, ann_data: sc.AnnData, **kwargs):
        """
        Filter the dataset to only include cells present in the lineage information.
        """
        if is_log_normalized_to_counts(ann_data, counts=self._parameters()["counts"]):
            logging.debug(
                "Data appears to be log-normalized to counts=10^4. No further normalization will be applied."
            )
            return ann_data

        if ann_data.raw is None:
            raise ValueError(
                "Data appears to be normalized some other way and does not provide raw data. This could lead to errors down the line "
                "with cell type lineage prediction. Please ensure that the data is log-normalized to counts=10^4, or that the raw data is provided for proper CellTypist performance."
            )

        # we want all the vars, etc. to stay the same, so we just replace X with the raw data
        data = ann_data.raw.to_adata()
        sc.pp.filter_genes(
            data, min_cells=1
        )  # filter out genes that are not expressed in any cells
        sc.pp.normalize_total(data, target_sum=self._parameters()["counts"])
        sc.pp.log1p(data)
        print(
            f'Finished running normalization, check {is_log_normalized_to_counts(data, counts=self._parameters()["counts"])}'
        )
        return data
