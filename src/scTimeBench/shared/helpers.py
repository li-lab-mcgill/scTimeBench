"""
Helper function for miscellaneous tasks.
"""

from pathlib import Path


def _resolve_shared_resource_path(file_path):
    """
    Resolve lineage/equivalence file paths in a package-safe way.

    Resolution order:
    1) keep absolute paths
    2) keep existing cwd-relative paths
    3) resolve relative to scTimeBench/shared/dataset
    """
    if file_path is None:
        return None

    raw_path = Path(file_path)
    if raw_path.is_absolute() or raw_path.exists():
        return str(raw_path)

    package_root = Path(__file__).resolve().parent
    shared_root = package_root / "dataset"

    normalized_rel = str(file_path).lstrip("./")
    rel_path = Path(normalized_rel)
    candidate = shared_root / rel_path
    if candidate.exists():
        return str(candidate)

    return str(raw_path)


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

    file_path = _resolve_shared_resource_path(file_path)

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
    file_path = _resolve_shared_resource_path(file_path)
    equivalence_file_path = _resolve_shared_resource_path(equivalence_file_path)
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
