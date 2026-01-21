import pytest
from pathlib import Path

# 1. Define the path to your configs
CONFIG_DIR = Path(__file__).parent / "configs"
# 2. Automatically find all .yaml files
CONFIG_FILES = list(CONFIG_DIR.glob("*.yaml"))

TEST_DIR = Path(__file__).parent.parent


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.name)
def test_inference_method(config_path, run_bench):
    """
    Test that a given trajectory inference method config file can be executed end-to-end.
    Basically, just make sure that it can be run!
    """
    log_file = Path(config_path).name + ".log"
    # run it at the root of the project
    result = run_bench(config_path, "auto_train_test", TEST_DIR.parent, log_file)

    assert result.returncode == 0
