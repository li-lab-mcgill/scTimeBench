import pytest
import subprocess
from pathlib import Path
import sys


@pytest.fixture
def run_bench():
    def _run(config_path, run_type, workspace, log_name, extra_args=None):
        project_root = Path(__file__).parent.parent

        # delete any previous log file
        log_file_path = str(project_root / "test" / "logs" / log_name)
        if Path(log_file_path).exists():
            Path(log_file_path).unlink()

        cmd = [
            "scTimeBench",
            "--config",
            str(config_path),
            "--run_type",
            run_type,
            "--output_dir",
            str(workspace / "outputs"),
            "--database_path",
            str(workspace / "scTimeBench.db"),
            "--log_file",
            log_file_path,
            "--log_level",
            "DEBUG",
        ]

        if extra_args:
            cmd.extend(extra_args)

        # Use subprocess.Popen to stream logs in real-time
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Print each line to stdout so pytest can capture/display it
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        return process

    return _run


@pytest.fixture
def workspace(tmp_path):
    """Creates and cleans up directories for every test."""
    out = tmp_path / "outputs"
    out.mkdir()
    return tmp_path
