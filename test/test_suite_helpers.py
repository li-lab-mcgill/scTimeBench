from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sqlite3

import yaml

from scTimeBench.shared.constants import RequiredOutputFiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_E2E_CONFIG_ROOT = (
    PROJECT_ROOT / "test" / "01_end_to_end" / "configs" / "all_models"
)

# Explicit per-model fast configs (checked into test/01_end_to_end/configs/all_models).
MODEL_TEMPLATE_PATHS = {
    "Artemis": TEST_E2E_CONFIG_ROOT / "Artemis.yaml",
    "CellMNN": TEST_E2E_CONFIG_ROOT / "CellMNN.yaml",
    "Cooccurrence": TEST_E2E_CONFIG_ROOT / "Cooccurrence.yaml",
    "Correlation": TEST_E2E_CONFIG_ROOT / "Correlation.yaml",
    "MIOFlow": TEST_E2E_CONFIG_ROOT / "MIOFlow.yaml",
    "Moscot": TEST_E2E_CONFIG_ROOT / "Moscot.yaml",
    "PISDE": TEST_E2E_CONFIG_ROOT / "PISDE.yaml",
    "PRESCIENT": TEST_E2E_CONFIG_ROOT / "PRESCIENT.yaml",
    "Squidiff": TEST_E2E_CONFIG_ROOT / "Squidiff.yaml",
    "WOT": TEST_E2E_CONFIG_ROOT / "WOT.yaml",
    "scIMF": TEST_E2E_CONFIG_ROOT / "scIMF.yaml",
    "scNODE": TEST_E2E_CONFIG_ROOT / "scNODE.yaml",
}

OT_METHODS = {"Moscot", "WOT"}
PRED_GRAPH_METHODS = {"Cooccurrence", "Correlation"}
GRAPH_ONLY_METHODS = OT_METHODS | PRED_GRAPH_METHODS

FULL_NON_OT_OUTPUTS = {
    output.value
    for output in RequiredOutputFiles
    if output
    not in {
        RequiredOutputFiles.NEXT_CELLTYPE,
        RequiredOutputFiles.PRED_GRAPH,
        RequiredOutputFiles.FROM_ZERO_TO_END_PRED_GEX,
    }
}


def load_yaml(path: Path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def build_method_config(
    method_name: str, datasets: list[dict], metrics: list[dict]
) -> dict:
    template = load_yaml(MODEL_TEMPLATE_PATHS[method_name])
    method = template["method"]

    method_payload = {
        "name": method["name"],
        "train_and_test_script": method["train_and_test_script"],
        "metadata": deepcopy(method.get("metadata", {})),
    }

    if not method_payload["metadata"]:
        method_payload.pop("metadata")

    return {
        "datasets": deepcopy(datasets),
        "method": method_payload,
        "metrics": deepcopy(metrics),
    }


def write_config(config_data: dict, workspace: Path, name: str) -> Path:
    config_path = workspace / name
    with open(config_path, "w") as handle:
        yaml.safe_dump(config_data, handle, sort_keys=False)
    return config_path


def sql_fetchone(db_path: Path, query: str, params: tuple = ()):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchone()


def sql_fetchall(db_path: Path, query: str, params: tuple = ()):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()


def get_output_paths_for_method(db_path: Path, method_name: str) -> list[Path]:
    rows = sql_fetchall(
        db_path,
        "SELECT path FROM method_outputs WHERE name = ?",
        (method_name,),
    )
    return [Path(row[0]) for row in rows]


def get_output_records_for_method(db_path: Path, method_name: str) -> list[dict]:
    rows = sql_fetchall(
        db_path,
        """
        SELECT mo.path, d.name
        FROM method_outputs mo
        JOIN datasets d ON d.id = mo.dataset_id
        WHERE mo.name = ?
        """,
        (method_name,),
    )
    return [
        {"path": Path(path), "dataset_name": dataset_name}
        for path, dataset_name in rows
    ]


def load_shared_datasets() -> list[dict]:
    default_file = (
        PROJECT_ROOT
        / "src"
        / "scTimeBench"
        / "shared"
        / "dataset"
        / "default_datasets.yaml"
    )
    optional_file = (
        PROJECT_ROOT
        / "src"
        / "scTimeBench"
        / "shared"
        / "dataset"
        / "optional_datasets.yaml"
    )

    default_data = load_yaml(default_file).get("datasets", [])
    optional_data = load_yaml(optional_file).get("datasets", [])
    return default_data + optional_data


def dataset_tag_to_name() -> dict[str, str]:
    tag_map: dict[str, str] = {}
    for dataset in load_shared_datasets():
        tag = dataset.get("tag")
        if tag:
            tag_map[tag] = dataset["name"]
    return tag_map


def expected_outputs_for_method(method_name: str) -> set[str]:
    if method_name in OT_METHODS:
        return {RequiredOutputFiles.NEXT_CELLTYPE.value}
    if method_name in PRED_GRAPH_METHODS:
        return {RequiredOutputFiles.PRED_GRAPH.value}
    return FULL_NON_OT_OUTPUTS


def expected_outputs_for_method_dataset(
    method_name: str, dataset_name: str
) -> set[str]:
    # Keep method-specific special behavior first.
    if method_name in OT_METHODS:
        return {RequiredOutputFiles.NEXT_CELLTYPE.value}
    if method_name in PRED_GRAPH_METHODS:
        return {RequiredOutputFiles.PRED_GRAPH.value}

    # Dataset-aware behavior for regular methods.
    if dataset_name in GEX_DATASET_NAMES:
        return {RequiredOutputFiles.NEXT_TIMEPOINT_GENE_EXPRESSION.value}

    return {
        RequiredOutputFiles.EMBEDDING.value,
        RequiredOutputFiles.NEXT_TIMEPOINT_EMBEDDING.value,
    }


def config_metrics_for_model(method_name: str) -> list[dict]:
    if method_name in GRAPH_ONLY_METHODS:
        return [{"name": "JaccardSimilarity"}]

    return [
        {"name": "ARI"},
        {"name": "JaccardSimilarity"},
        {"name": "WassersteinOTLoss"},
    ]


def config_datasets_for_model(method_name: str) -> list[dict]:
    if method_name in GRAPH_ONLY_METHODS:
        return [{"tag": "defaultGarciaAlonso"}]
    return [{"tag": "defaultGarciaAlonso"}, {"tag": "EasyZebrafish"}]


def get_leaf_metric_names() -> list[str]:
    import scTimeBench.metrics  # noqa: F401
    from scTimeBench.metrics.base import METRIC_REGISTRY

    leaf_names = []
    for metric_name, metric_cls in METRIC_REGISTRY.items():
        if len(getattr(metric_cls, "submetrics", [])) == 0:
            leaf_names.append(metric_name)
    return sorted(leaf_names)


METRIC_IMAGE_PATTERNS = {
    "GraphClassificationReport": [
        "**/roc_curve_*.png",
        "**/prc_curve_*.png",
    ],
    "GraphVisualization": [
        "**/predicted_graph*.svg",
        "**/predicted_unweighted_graph*.svg",
    ],
    "StackedBarPlot": [
        "**/target_stacked_bar_plot.svg",
    ],
}

GEX_DATASET_NAMES = {
    "MaDataset",
    "OlaniruDataset",
    "MaOlaniruDataset",
    "ZebrafishDataset",
    "DrosophilaDataset",
    "MEFDataset",
}
