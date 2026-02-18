"""
Olaniru et al. (2023) dataset.
"""

from crispy_fishstick.shared.dataset.base import BaseDataset, ObservationColumns
import scanpy as sc


class OlaniruDataset(BaseDataset):
    def _load_data(self):
        """
        Load the Olaniru et al. dataset.
        """
        print("Loading Olaniru et al. dataset...")
        # read from the dataset data_path
        data_path = self.dataset_dict["data_path"]
        self.data = sc.read_h5ad(data_path)

        self.data.obs = self.data.obs.rename(
            columns={
                "timepoint": ObservationColumns.TIMEPOINT.value,
            }
        )
        self.data.obs[ObservationColumns.CELL_TYPE.value] = "unknown"
        print("Olaniru et al. dataset loaded successfully.")

        print(
            f"Timepoints: {self.data.obs[ObservationColumns.TIMEPOINT.value].unique()}"
        )
        print("No celltypes available for this dataset as of yet.")
        print("Assigned dummy cell type: unknown")
