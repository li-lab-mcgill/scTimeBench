import pytest
import yaml
from pathlib import Path

from test.test_suite_helpers import (
    expected_outputs_for_method_dataset,
    get_output_records_for_method,
    sql_fetchone,
)

CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.name)
def test_all_models_fast_end_to_end(config_path, workspace, run_bench):
    """
    End-to-end smoke for each model with fast metadata.

    Required checks:
    1) requested dataset/metric combinations run,
    2) SQL tables are populated,
    3) expected required output files exist for method category.
    """
    with open(config_path, "r") as handle:
        config_data = yaml.safe_load(handle)

    method_name = config_data["method"]["name"]
    metrics = config_data["metrics"]

    log_name = f"e2e_{config_path.stem}.log"
    result = run_bench(
        config_path,
        "auto_train_test",
        workspace,
        log_name,
        extra_args=["--force_rerun"],
    )
    assert result.returncode == 0

    db_path = workspace / "scTimeBench.db"

    # 1) SQL: method_outputs inserted for this method
    method_output_count = sql_fetchone(
        db_path,
        "SELECT COUNT(*) FROM method_outputs WHERE name = ?",
        (method_name,),
    )[0]
    assert method_output_count > 0, f"No method_outputs row inserted for {method_name}."

    # 2) SQL: eval rows exist for this method
    eval_count = sql_fetchone(
        db_path,
        """
        SELECT COUNT(*)
        FROM evals e
        JOIN method_outputs mo ON mo.id = e.method_output_id
        WHERE mo.name = ?
        """,
        (method_name,),
    )[0]
    assert eval_count > 0, f"No eval row inserted for {method_name}."

    # 3) SQL: all requested metric names show up at least once for this method
    expected_metric_names = {metric["name"] for metric in metrics}
    metric_concat = sql_fetchone(
        db_path,
        """
        SELECT GROUP_CONCAT(DISTINCT m.name)
        FROM evals e
        JOIN method_outputs mo ON mo.id = e.method_output_id
        JOIN metrics m ON m.id = e.metric_id
        WHERE mo.name = ?
        """,
        (method_name,),
    )
    metric_csv = metric_concat[0] if metric_concat and metric_concat[0] else ""
    found_metric_names = {name for name in metric_csv.split(",") if name}
    missing_metrics = expected_metric_names - found_metric_names
    assert (
        not missing_metrics
    ), f"Method {method_name} did not populate eval rows for metrics: {missing_metrics}"

    # 4) File outputs: verify required outputs by method+dataset category
    output_records = get_output_records_for_method(db_path, method_name)
    assert (
        output_records
    ), f"No method output directories found in DB for {method_name}."

    for record in output_records:
        output_path = record["path"]
        dataset_name = record["dataset_name"]
        expected_output_files = expected_outputs_for_method_dataset(
            method_name, dataset_name
        )

        assert output_path.exists(), f"Output directory missing: {output_path}"
        found_files = {path.name for path in output_path.iterdir() if path.is_file()}
        missing_files = expected_output_files - found_files
        assert (
            not missing_files
        ), f"Method {method_name} on dataset {dataset_name} missing required outputs {missing_files} in {output_path}"
