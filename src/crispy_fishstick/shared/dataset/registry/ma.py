"""
Ma et al. (2023) dataset.
"""

from crispy_fishstick.shared.dataset.base import BaseDataset, ObservationColumns
import scanpy as sc


class OlaniruDataset(BaseDataset):
    def _load_data(self):
        """
        Load the Ma et al. datasett.
        """
        print("Loading Ma et al. dataset...")
        # read from the dataset data_path
        data_path = self.dataset_dict["data_path"]
        self.data = sc.read_h5ad(data_path)

        print("Ma et al. dataset loaded successfully.")

        # commented out until annotations included (optional)
        # self.data.obs = self.data.obs.rename(
        #     columns={
        #         "celltype": ObservationColumns.CELL_TYPE.value,
        #         "PCW": ObservationColumns.TIMEPOINT.value,
        #     }
        # )
        print(
            f"Cell types: {self.data.obs[ObservationColumns.CELL_TYPE.value].unique()}"
        )
