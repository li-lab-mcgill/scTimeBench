"""
Helper function for miscellaneous tasks.
"""


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
        Dictionary mapping alias cell type names to their canonical name.
    """
    if file_path is None:
        return {}

    with open(file_path, "r") as f:
        content = f.read().strip()

    equivalence = {}

    # for each row in the file:
    for row in content.splitlines():
        # Split by , and clean up first
        canonical, aliases = [cell.strip() for cell in row.split("<=") if cell.strip()]
        aliases = [cell.strip() for cell in aliases.split(",") if cell.strip()]

        if not aliases:
            continue

        for alias in aliases:
            if alias:
                equivalence[alias] = canonical

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
        Dictionary mapping canonicalized cell types to their descendants
    """
    equivalence_dict = parse_equivalence(equivalence_file_path)

    with open(file_path, "r") as f:
        content = f.read().strip()

    lineage = {}

    # for each row in the file:
    for row in content.splitlines():
        # Split by => and clean up first
        cells = [cell.strip() for cell in row.split("=>") if cell.strip()]

        # replace aliases with canonical names
        normalized_cells = [equivalence_dict.get(cell, cell) for cell in cells]

        # then let's build the lineage mapping
        for i in range(len(normalized_cells) - 1):
            source = normalized_cells[i]
            target = normalized_cells[i + 1]

            if source not in lineage:
                lineage[source] = []

            if target not in lineage[source]:
                lineage[source].append(target)

    return lineage
