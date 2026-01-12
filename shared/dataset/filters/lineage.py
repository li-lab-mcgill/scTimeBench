"""
Filter based on only the cells existing in the lineage information.
"""

from shared.dataset.base import BaseDatasetFilter
from shared.constants import ObservationColumns


class LineageDatasetFilter(BaseDatasetFilter):
    def _parameters(self):
        """
        Return filter-specific parameters.
        """
        return {
            "cell_lineage_file": self.config.dataset.get("cell_lineage_file", None),
            "cell_equivalence_file": self.config.dataset.get(
                "cell_equivalence_file", None
            ),
        }

    def filter(self, ann_data):
        """
        Filter the dataset to only include cells present in the lineage information.
        """

        # based off the config, we should filter our dataset
        # for only the cells that are in the lineage information
        lineage_file = self.config.dataset.get("cell_lineage_file", None)
        if lineage_file is None:
            raise ValueError(
                "Cell lineage file must be specified in the config for LineageDatasetFilter."
            )

        equivalence_file = self.config.dataset.get("cell_equivalence_file", None)

        lineage_dict = self._parse_cell_lineage(lineage_file, equivalence_file)

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

    def _parse_equivalence(self, file_path):
        """
        Parse a cell equivalence file and create a dictionary mapping equivalent names.

        Parameters:
        -----------
        file_path : str
            Path to the equivalence file (split by ,)

        Returns:
        --------
        dict
            Dictionary mapping normalized cell type names to their equivalent names
        """
        if file_path is None:
            return {}

        with open(file_path, "r") as f:
            content = f.read().strip()

        equivalence = {}

        # for each row in the file:
        for row in content.splitlines():
            # Split by , and clean up first
            cells = row.split(",")

            for i in range(len(cells)):
                # first we replace the names with what we have in
                cells[i] = cells[i].strip()

            # then let's build the equivalence mapping
            equivalence[cells[0]] = cells[1]

        return equivalence

    def _parse_cell_lineage(self, file_path, equivalence_file_path=None):
        """
        Parse a cell lineage file and create a dictionary mapping source to root.

        Parameters:
        -----------
        file_path : str
            Path to the lineage file (split by =>)

        Returns:
        --------
        dict
            Dictionary mapping normalized cell types to their root
        """
        equivalence_dict = self._parse_equivalence(equivalence_file_path)

        with open(file_path, "r") as f:
            content = f.read().strip()

        lineage = {}

        # for each row in the file:
        for row in content.splitlines():
            # Split by => and clean up first
            cells = row.split("=>")
            for i in range(len(cells)):
                # first we replace the names with what we have in
                cells[i] = cells[i].strip()
                if cells[i] in equivalence_dict:
                    cells[i] = equivalence_dict[cells[i]]
                cells[i] = cells[i].upper().replace(" ", "_").replace("-", "_")

            # then let's build the lineage mapping
            for i in range(len(cells) - 1):
                source = cells[i]
                target = cells[i + 1]

                if source not in lineage:
                    lineage[source] = []
                lineage[source].append(target)

        return lineage
