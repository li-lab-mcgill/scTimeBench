"""
Filter based on only the cells existing in the lineage information.
"""

from scTimeBench.shared.dataset.base import BaseDatasetPreprocessor
from scTimeBench.shared.constants import ObservationColumns
from scTimeBench.shared.helpers import parse_cell_lineage, parse_equivalence


class LineageDatasetFilter(BaseDatasetPreprocessor):
    def __init__(self, dataset_dict, cell_lineage_file, cell_equivalence_file=None):
        super().__init__(dataset_dict)
        self.cell_lineage_file = cell_lineage_file
        self.cell_equivalence_file = cell_equivalence_file

    def _parameters(self):
        """
        Return preprocessor-specific parameters.
        """
        return {
            "cell_lineage_file": self.cell_lineage_file,
            "cell_equivalence_file": self.cell_equivalence_file,
        }

    def preprocess(self, ann_data, **kwargs):
        """
        Filter the dataset to only include cells present in the lineage information.
        """

        ann_data = ann_data.copy()

        # first normalize dataset labels to canonical names from equivalence mapping
        equivalence_dict = parse_equivalence(self.cell_equivalence_file)
        if len(equivalence_dict) > 0:
            cell_type_col = ObservationColumns.CELL_TYPE.value
            original_labels = ann_data.obs[cell_type_col]
            mapped_labels = original_labels.map(equivalence_dict)
            ann_data.obs[cell_type_col] = mapped_labels.where(
                mapped_labels.notna(), original_labels
            )

        # based off the config, we should filter our dataset
        # for only the cells that are in the lineage information
        lineage_dict = parse_cell_lineage(
            self.cell_lineage_file, self.cell_equivalence_file
        )

        print(f"Lineage: {lineage_dict}")

        # now let's filter the dataset based on this lineage information
        cells_in_lineage = set(lineage_dict.keys())
        for targets in lineage_dict.values():
            for target in targets:
                cells_in_lineage.add(target)

        # let's warn the user if there is a cell type in the lineage that is not in the dataset
        dataset_cell_types = set(
            ann_data.obs[ObservationColumns.CELL_TYPE.value].unique()
        )

        for cell_type in cells_in_lineage:
            if cell_type not in dataset_cell_types:
                print(
                    f"Warning: Cell type {cell_type} in lineage information not found in dataset. Create equivalence mapping if necessary."
                )

        # filter the dataset
        return ann_data[
            ann_data.obs[ObservationColumns.CELL_TYPE.value].isin(cells_in_lineage)
        ].copy()
