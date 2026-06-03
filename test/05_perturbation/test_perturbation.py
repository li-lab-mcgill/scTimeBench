import pytest
from pathlib import Path

# 1. Define the path to your configs
CONFIG_DIR = Path(__file__).parent / "configs"
# 2. Automatically find all .yaml files
CONFIG_FILES = list(CONFIG_DIR.glob("*.yaml"))

TEST_DIR = Path(__file__).parent.parent


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.name)
def test_perturbation(config_path, workspace, run_bench):
    """
    Test that a given trajectory inference method config file can be executed end-to-end.

    The config file contains everything that we need, so this is not that bad.
    """
    log_file = f"05_perturbation_{config_path.stem}.log"
    # run it at the root of the project
    result = run_bench(
        config_path,
        "auto_train_test",
        workspace,
        log_file,
        extra_args=["--force_rerun"],
    )
    assert result.returncode == 0
