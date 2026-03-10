"""
Garcia-Alonso et al. (2022) dataset.
"""

from scTimeBench.shared.dataset.base import BaseDataset, ObservationColumns
import scanpy as sc


class GarciaAlonsoDataset(BaseDataset):
    def _load_data(self):
        """
        Load the Garcia-Alonso et al. dataset.
        """
        print("Loading Garcia-Alonso et al. dataset...")
        # read from the dataset data_path
        data_path = self.dataset_dict["data_path"]
        self.data = sc.read_h5ad(data_path)

        # now let's filter out all the datapoints that are low quality
        # i.e. nan age and nan cell type
        # rename these columns to standard names
        self.data.obs = self.data.obs.rename(
            columns={
                "celltype": ObservationColumns.CELL_TYPE.value,
                "PCW": ObservationColumns.TIMEPOINT.value,
            }
        )
        print("Garcia-Alonso et al. dataset loaded successfully.")

        print(
            f"Cell types: {self.data.obs[ObservationColumns.CELL_TYPE.value].unique()}"
        )
