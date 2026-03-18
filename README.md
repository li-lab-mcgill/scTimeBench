# scTimeBench

![scTimeBench Overview](./assets/scTimeBench.png)



## Setup

If the external dependencies such as pypsupertime or sceptic are not used (which they are not used by default), you can install using pip as follows:

```
pip install -e ".[benchmark]"
```

to run the benchmark. For your own method, simply install without the extra benchmarking requirements with

```
pip install -e .
```

There are extra dependencies that can be found under `pyproject.toml`.

## Setup: UV

Due to external dependencies and a more complex setup, we have decided to package everything under `uv` (see: https://github.com/astral-sh/uv). To start with, install `uv` then run the following:

```
uv sync
```

This allows you to create model run files that use the necessary constants and model runners under the scTimeBench packages.

Note: for the other models, we don't need all the dependencies so for example, if you're setting up the moscot environment,

```
uv sync --no-dev
```

is enough.

For other packages depending on the metrics used, it would be useful to install them with:

```
uv sync --extra <dependency-group>
```

e.g.:

```
uv sync --extra test --extra dev --extra benchmark
```

or simply

```
uv sync --all-extras
```

## Python Version

We also set the Python version to be 3.10. This will likely cause issues in other python versions, so do try to use:

```
uv python install 3.10
uv python pin 3.10
```

before running uv sync.

## Detailed Layout of File Structure

* `examples/` defines the examples

  * `configs/` possible yaml config files to use as a starting point
* `models/` defines the different models that are possible to use, including defined submodules. Add your own methodology here.
* `src/` where the scTimeBench package lies. See `src/ReadMe.md` for more documentation on the modules that exist there.
* `test/` unit tests for each model, each metric, and other important modules.

## Example Run

Run either using the package itself with:

```
scTimeBench --config examples/configs/scNODE_user_defined.yaml --run_type auto_train_test
```

or with:

```
python src/scTimeBench/main.py --config examples/configs/scNODE_user_defined.yaml --run_type auto_train_test
```

## Contributing

If you want to contribute, please install both the test and the dev environments with:

```
pip install -e ".[test, dev, benchmark]"
```

Then for our autoformatting, please run:

```
pre-commit install
```

## Testing

To run a test simply run:

```
pytest test
```

under the root directory or move to `test` and run:

```
pytest
```

See more information on the pytest documentation: https://docs.pytest.org/en/stable/. A useful flag is `-s` to view the entire output of the test.

