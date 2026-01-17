# Crispy-Fishstick
## Setup
To start, please install the crispy-fishstick package with:
```
pip install -e .
```

This allows you to create model run files that use the necessary constants and model runners under the crispy fishstick packages.

## Detailed Layout of File Structure
- `examples/` defines the examples
    -  `configs/` possible yaml config files to use as a starting point
- `models/` defines the different models that are possible to use, including defined submodules. Add your own methodology here.
- `src/` where the crispy-fishstick package lies. See `src/ReadMe.md` for more documentation on the modules that exist there.
- `test/` unit tests for each model, each metric, and other important modules.

## Example Run
Run either using the package itself with:
```
crispy_fishstick --config examples/configs/scNODE_user_defined.yaml --run_type auto_train_test
```

or with:
```
python src/crispy_fishstick/main.py --config examples/configs/scNODE_user_defined.yaml --run_type auto_train_test
```

## Contributing
If you want to contribute, please install both the test and the dev environments with:
```
pip install -e ".[test, dev]"
```
