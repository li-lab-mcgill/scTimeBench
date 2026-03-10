"""
Suo et al. (2022) dataset.
"""

from scTimeBench.shared.dataset.base import BaseDataset, ObservationColumns
import scanpy as sc


class SuoDataset(BaseDataset):
    def _load_data(self):
        """
        Load the Suo et al. dataset.
        """
        print("Loading Suo et al. dataset...")
        # read from the config data_path
        data_path = self.dataset_dict["data_path"]
        self.data = sc.read_h5ad(data_path)

        # now let's filter out all the datapoints that are low quality
        # i.e. nan age and nan cell type
        # rename these columns to standard names
        self.data.obs = self.data.obs.rename(
            columns={
                "celltype_annotation": ObservationColumns.CELL_TYPE.value,
                "age": ObservationColumns.TIMEPOINT.value,
            }
        )
        print("Suo et al. dataset loaded successfully.")
