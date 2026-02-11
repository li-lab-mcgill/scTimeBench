"""
Filter that replaces the time column with a psupertime.
"""

from crispy_fishstick.shared.dataset.base import BaseDatasetFilter
from crispy_fishstick.shared.constants import ObservationColumns
import scanpy as sc
import logging
import numpy as np


class PsupertimeFilter(BaseDatasetFilter):
    def _parameters(self):
        """
        Return filter-specific parameters.
        """
        return {
            "num_hvgs": self.dataset_dict.get("num_hvgs", 2000),
        }

    def filter(self, ann_data):
        from pypsupertime import Psupertime

        """
        Filter the dataset to replace its time column with a psupertime.
        """
        # 1. Load your data (standard AnnData object)
        # 2. Initialize and run
        # 'time' should be the column in adata.obs containing your sequential labels

        # let's turn off numba
        logging.getLogger("numba").setLevel(logging.WARNING)

        # ann data is a bit slow... so let's filter for the top highly variable genes
        sc.pp.highly_variable_genes(
            ann_data, n_top_genes=self._parameters()["num_hvgs"]
        )
        ann_data = ann_data[:, ann_data.var.highly_variable].copy()

        # let's randomly filter for 1000 cells to speed up the process for debugging...
        if ann_data.n_obs > 1000:
            logging.info(
                f"Filtering for 1000 random cells out of {ann_data.n_obs} to speed up psupertime for debugging..."
            )
            random_indices = np.random.choice(ann_data.n_obs, size=1000, replace=False)
            ann_data = ann_data[random_indices].copy()

        psup = Psupertime(n_jobs=5, n_folds=3)
        output_data = psup.run(ann_data, ObservationColumns.TIMEPOINT.value)

        logging.debug(f'Psupertime observation: {output_data.obs["psupertime"]}')

        # now it's under obs["psupertime"], which should then be renamed back to TIMEPOINT
        output_data.obs[ObservationColumns.TIMEPOINT.value] = output_data.obs[
            "psupertime"
        ]
        return output_data
