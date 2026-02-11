"""
MEF dataset.
"""

from crispy_fishstick.shared.dataset.base import BaseDataset
import scanpy as sc


class MEFDataset(BaseDataset):
    def _load_data(self):
        """
        Load the MEF datasett.
        """
        print("Loading MEF dataset...")
        # read from the dataset data_path
        data_path = self.dataset_dict["data_path"]
        self.data = sc.read_h5ad(data_path)

        print("MEF dataset loaded successfully.")
