"""
Ma et al. (2023) dataset.
"""

from scTimeBench.shared.dataset.base import BaseDataset, ObservationColumns
import scanpy as sc


class MaDataset(BaseDataset):
    def _load_data(self):
        """
        Load the Ma et al. dataset.
        """
        print("Loading Ma et al. dataset...")
        # read from the dataset data_path
        data_path = self.dataset_dict["data_path"]
        self.data = sc.read_h5ad(data_path)

        self.data.obs = self.data.obs.rename(
            columns={
                "cell_type": ObservationColumns.CELL_TYPE.value,
                "timepoint": ObservationColumns.TIMEPOINT.value,
            }
        )
        print("Ma et al. dataset loaded successfully.")

        print(
            f"Cell types: {self.data.obs[ObservationColumns.CELL_TYPE.value].unique()}"
        )
