import pytest
from pathlib import Path
from scTimeBench.metrics.base import METRIC_REGISTRY

# 1. Define the path to your configs
CONFIG_DIR = Path(__file__).parent / "configs"
# 2. Automatically find all .yaml files
CONFIG_FILES = list(CONFIG_DIR.glob("*.yaml"))

TEST_DIR = Path(__file__).parent.parent


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.name)
def test_config_execution(config_path, workspace, run_bench):
    """
    Test that a given config file can be executed end-to-end
    and produces the expected outputs.
    """
    log_file = Path(config_path).name + ".log"
    result = run_bench(config_path, "auto_train_test", workspace, log_file)

    assert result.returncode == 0

    output_files = list((workspace / "outputs").iterdir())
    assert len(output_files) > 0, f"Config {config_path.name} produced no output."

    # let's verify that we have:
    # 1. database object
    # 2. metric outputs
    # 3. dataset pickle
    # 4. model config

    # then for each metric collect a set of all required output files
    with open(config_path, "r") as f:
        import yaml

        config_data = yaml.safe_load(f)

    metric_to_run = config_data.get("metrics", [])[0]
    top_metric_name = metric_to_run["name"]
    top_metric_inst = METRIC_REGISTRY[top_metric_name](None, None, {})

    # need to traverse the tree of submetrics to get all the leaf metrics
    def get_leaf_metrics(metric_inst):
        if len(metric_inst.submetrics) == 0:
            return [metric_inst]
        else:
            leaves = []
            for submetric in metric_inst.submetrics:
                leaves.extend(get_leaf_metrics(submetric(None, None, {})))
            return leaves

    output_required_files = set()
    for metric in get_leaf_metrics(top_metric_inst):
        # Collect required outputs from each metric
        if hasattr(metric, "required_outputs"):
            for output in metric.required_outputs:
                if isinstance(output, list):
                    # list of list case - add all from the inner list
                    for inner_output in output:
                        output_required_files.add(inner_output.value)
                else:
                    output_required_files.add(output.value)

    output_required_files.update(
        [
            "model_config.yaml",
        ]
    )

    # here we parse for the proper output path:
    # by looking under the logs for the line: Output path for model: <path>
    with open(TEST_DIR / "logs" / log_file, "r") as log_file_handle:
        for line in log_file_handle:
            if "Output path for" in line:
                parts = line.strip().split("Output path for model: ")
                if len(parts) == 2:
                    output_path = parts[1]
                    output_model_dir = Path(output_path).name
                    break

    found_output_files = set(
        f.name for f in (workspace / "outputs" / output_model_dir).iterdir()
    )

    missing_files = output_required_files - found_output_files
    assert (
        len(missing_files) == 0
    ), f"Running config {config_path.name} missing output files: {missing_files}"

    # finally, verify that the database holds the expected evaluations
    # this is good to test hierarchical metric evaluations as well

    # create a minimal config object for database access
    from scTimeBench.database import DatabaseManager

    class dummy_class:
        def __init__(self, database_path):
            self.database_path = database_path

    test_config = dummy_class(str(workspace / "scTimeBench.db"))
    db_manager = DatabaseManager(test_config)

    db_contents = db_manager.return_all()

    # now we parse line by line of the db_contents to make sure that every single
    # metric gets hit, as long as they are non-hierarchical because we will be testing that
    # config is going to include BaseMetric (i.e. all of them)
    print(db_contents)

    for metric in get_leaf_metrics(top_metric_inst):
        metric_name = metric.__class__.__name__
        print("Metric name:", metric_name)

        evals = db_manager.get_evals_per_metric(
            metric_name,
            metric._get_param_encoding(),  # ** Note: have to use default parametrization **
        )
        print(
            f"Metric {metric_name} produced {len(evals)} evaluations in the database."
        )
        assert (
            len(evals) > 0
        ), f"Metric {metric_name} produced no evaluations in the database."

    db_manager.close()


MISC_TEST_DIR = Path(__file__).parent / "misc_test_configs"
MISC_CONFIG_FILES = list(MISC_TEST_DIR.glob("*.yaml"))


@pytest.mark.parametrize("config_path", MISC_CONFIG_FILES, ids=lambda p: p.name)
def test_dummy_dataset(config_path, workspace, run_bench):
    """
    Tests that the dummy dataset is not run in this example because it is not supported.
    And so no tests should run here.
    """
    log_file = Path(config_path).name + ".log"
    result = run_bench(config_path, "auto_train_test", workspace, log_file)

    assert result.returncode == 0

    # finally, let's verify that in the output that we get the line
    # Dataset DummyDataset not supported by this metric <metric name>.
    # for all the metrics, which should be every metric in the metric registry that is not a hierarchical metric
    # due to us using the base metric class
    with open(TEST_DIR / "logs" / log_file, "r") as log_file_handle:
        log_contents = log_file_handle.read()

    # we should be doing this based off of the submetric tree for the metric that we're looking at
    with open(config_path, "r") as f:
        import yaml

        config_data = yaml.safe_load(f)
    metric_to_run = config_data.get("metrics", [])[0]
    top_metric_name = metric_to_run["name"]

    top_metric_inst = METRIC_REGISTRY[top_metric_name](None, None, {})

    # need to traverse the tree of submetrics to get all the leaf metrics
    def get_leaf_metrics(metric_inst):
        if len(metric_inst.submetrics) == 0:
            return [metric_inst]
        else:
            leaves = []
            for submetric in metric_inst.submetrics:
                leaves.extend(get_leaf_metrics(submetric(None, None, {})))
            return leaves

    print(f"Existing leaf metrics: {get_leaf_metrics(top_metric_inst)}")
    for submetric in get_leaf_metrics(top_metric_inst):
        if len(submetric.submetrics) == 0:
            assert (
                "Dataset {'name': 'DummyDataset', 'data_path': './data/garcia-alonso/human_germ.h5ad'} not supported by this metric "
                + f"{submetric.__class__.__name__}."
                in log_contents
            ), f"Expected line not found for metric {submetric.__class__.__name__}."

    # finally we should verify that the database is completely empty
    from scTimeBench.database import DatabaseManager

    class dummy_class:
        def __init__(self, database_path):
            self.database_path = database_path

    test_config = dummy_class(str(workspace / "scTimeBench.db"))
    db_manager = DatabaseManager(test_config)

    db_contents = db_manager.return_all()

    # verify that the contents are as follows:
    # 1. Contents of table: model_outputs followed by nothing
    # 2. Contents of table: metrics followed by a certain number of metrics but no evals
    # 3. Contents of table: evals followed by nothing
    stage = 0
    for line in db_contents.splitlines():
        if stage == 0:
            assert (
                line.strip() == "Contents of table: model_outputs"
            ), "Unexpected content in database."
            stage = 1
        elif stage == 1:
            assert (
                line.strip() == "Contents of table: metrics"
            ), "Unexpected content in output database, there exists some model_outputs created that should not have been."
            stage = 2
        elif stage == 2:
            if line.strip() == "Contents of table: evals":
                stage = 3
            else:
                continue
        elif stage == 3:
            assert (
                line.strip() == ""
            ), "Unexpected content in output database, there exists some evals created that should not have been."
