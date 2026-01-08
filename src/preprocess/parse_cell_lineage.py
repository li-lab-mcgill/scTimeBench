"""
Preprocesses the cell lineage file to create a mapping from source cell types to root cell types.
"""

import argparse


def parse_equivalence(file_path):
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


def parse_cell_lineage(file_path, equivalence_file_path=None):
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
    equivalence_dict = parse_equivalence(equivalence_file_path)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell_lineage_file",
        type=str,
        default="./examples/cell_lineages/b_cells/b_cell_line.txt",
    )
    parser.add_argument(
        "--cell_equivalence_file",
        type=str,
        default="./examples/cell_lineages/b_cells/equal_names.txt",
    )
    args = parser.parse_args()
    print(parse_cell_lineage(args.cell_lineage_file, args.cell_equivalence_file))
