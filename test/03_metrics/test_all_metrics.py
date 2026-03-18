from pathlib import Path

import pytest

from test.test_suite_helpers import (
    build_method_config,
    get_leaf_metric_names,
    sql_fetchone,
    sql_fetchall,
    write_config,
    METRIC_IMAGE_PATTERNS,
    SCNODE_FASTEST_METADATA,
)


SCNODE_BASE_DATASETS = [
    {"tag": "defaultGarciaAlonso"},
    {"tag": "EasyZebrafish"},
]
SCNODE_BASE_METHOD = "scNODE"


@pytest.mark.skip(
    reason="This test tends to be quite long, and the next test covers this, but this is good to test to make sure things are going smoothly."
)
@pytest.mark.parametrize("metric_name", get_leaf_metric_names())
def test_metric_populates_db_and_outputs(metric_name, workspace, run_bench):
    """
    Validate every leaf metric with the simplest scNODE setup.

    Checks:
    1) metric inserts evals into DB,
    2) expected plot/image artifacts are generated when applicable.
    """
    metric_config = {"name": metric_name}
    config = build_method_config(
        SCNODE_BASE_METHOD,
        SCNODE_BASE_DATASETS,
        [metric_config],
        method_metadata_overrides=SCNODE_FASTEST_METADATA,
    )

    config_path = write_config(config, workspace, f"metric_{metric_name}.yaml")

    result = run_bench(
        config_path,
        "auto_train_test",
        workspace,
        f"metric_{metric_name}.log",
        extra_args=["--force_rerun"],
    )
    assert result.returncode == 0

    db_path = workspace / "scTimeBench.db"
    # 1) check whether or not the images are populated
    eval_count = sql_fetchone(
        db_path,
        """
        SELECT COUNT(*)
        FROM evals e
        JOIN metrics m ON m.id = e.metric_id
        WHERE m.name = ?
        """,
        (metric_name,),
    )[0]
    assert eval_count > 0, f"No eval rows found for metric {metric_name}."

    # 2) now let's check if the images were populated as expected
    image_patterns = METRIC_IMAGE_PATTERNS.get(metric_name, [])
    if not image_patterns:
        return

    output_paths = [
        Path(row[0])
        for row in sql_fetchall(
            db_path,
            "SELECT path FROM method_outputs WHERE name = ?",
            (SCNODE_BASE_METHOD,),
        )
    ]

    assert output_paths, "No method output paths recorded for scNODE."

    for pattern in image_patterns:
        found = []
        for output_path in output_paths:
            found.extend(output_path.glob(pattern))
        assert found, f"Expected image pattern '{pattern}' not found for {metric_name}."


def test_base_metric_populates_all_leaf_metrics(workspace, run_bench):
    """
    Run BaseMetric once and verify that every leaf metric is populated in evals.
    """
    config = build_method_config(
        SCNODE_BASE_METHOD,
        SCNODE_BASE_DATASETS,
        [{"name": "BaseMetric"}],
        method_metadata_overrides=SCNODE_FASTEST_METADATA,
    )

    config_path = write_config(config, workspace, "metric_BaseMetric.yaml")

    result = run_bench(
        config_path,
        "auto_train_test",
        workspace,
        "metric_BaseMetric.log",
        extra_args=["--force_rerun"],
    )
    assert result.returncode == 0

    db_path = workspace / "scTimeBench.db"
    rows = sql_fetchall(
        db_path,
        """
        SELECT DISTINCT m.name
        FROM evals e
        JOIN metrics m ON m.id = e.metric_id
        """,
    )
    populated_metric_names = {row[0] for row in rows}
    leaf_metric_names = set(get_leaf_metric_names())

    missing_leaf_metrics = leaf_metric_names - populated_metric_names
    assert not missing_leaf_metrics, (
        "BaseMetric did not populate the following leaf metrics in evals: "
        f"{sorted(missing_leaf_metrics)}"
    )
