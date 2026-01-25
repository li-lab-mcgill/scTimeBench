"""
Drosophila dataset.
"""

from shared.dataset.base import BaseDataset, ObservationColumns
import scanpy as sc


class DrosophilaDataset(BaseDataset):
    def _load_data(self):
        """
        Load the Drosophila datasett.
        """
        print("Loading Drosophila dataset...")
        # read from the dataset data_path
        data_path = self.dataset_dict["data_path"]
        self.data = sc.read_h5ad(data_path)

        print("Drosophila dataset loaded successfully.")
