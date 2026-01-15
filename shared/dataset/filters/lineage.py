"""
Filter based on only the cells existing in the lineage information.
"""

from shared.dataset.base import BaseDatasetFilter
from shared.constants import ObservationColumns
from shared.helpers import parse_cell_lineage


class LineageDatasetFilter(BaseDatasetFilter):
    def __init__(self, dataset_dict, cell_lineage_file, cell_equivalence_file=None):
        super().__init__(dataset_dict)
        self.cell_lineage_file = cell_lineage_file
        self.cell_equivalence_file = cell_equivalence_file

    def _parameters(self):
        """
        Return filter-specific parameters.
        """
        return {
            "cell_lineage_file": self.cell_lineage_file,
            "cell_equivalence_file": self.cell_equivalence_file,
        }

    def filter(self, ann_data):
        """
        Filter the dataset to only include cells present in the lineage information.
        """

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

        # filter the dataset
        return ann_data[
            ann_data.obs[ObservationColumns.CELL_TYPE.value].isin(cells_in_lineage)
        ].copy()
