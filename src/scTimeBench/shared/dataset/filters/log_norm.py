from scTimeBench.shared.dataset.base import BaseDatasetFilter
from scTimeBench.shared.utils import is_log_normalized_to_counts, is_raw
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

        if not is_raw(ann_data) and ann_data.raw is None:
            raise ValueError(
                "Data appears to be normalized some other way and does not provide raw data. This could lead to errors down the line "
                "with cell type lineage prediction. Please ensure that the data is log-normalized to counts=10^4, or that the raw data is provided for proper CellTypist performance."
            )

        # we want all the vars, etc. to stay the same, so we just replace X with the raw data
        if not is_raw(ann_data):
            logging.debug(
                "Data appears to be log-transformed but not normalized. Using raw data for normalization."
            )
            data = ann_data.raw.to_adata()
        else:
            logging.debug(
                "Data appears to be raw counts. Normalizing to counts=10^4 and log-transforming."
            )
            data = ann_data.copy()

        sc.pp.filter_genes(
            data, min_cells=1
        )  # filter out genes that are not expressed in any cells
        sc.pp.normalize_total(data, target_sum=self._parameters()["counts"])
        sc.pp.log1p(data)
        logging.debug(
            f'Finished running normalization, final check: {is_log_normalized_to_counts(data, counts=self._parameters()["counts"])}'
        )
        return data
