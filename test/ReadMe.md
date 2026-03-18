# Testing Framework

## Setup
To ensure that this is setup correctly, make sure to add `--extra dev` when running `uv sync`.

## End-to-end testing
When testing end-to-end we need to ensure that the following works:
1. Database is populated correctly with its method outputs, evaluation counts, and metrics.
2. The method outputs are also populated as expected depending on the method type.

To run a specific config file for testing, run the following command by replacing the `scNODE.yaml`:
```
pytest "01_end_to_end/test_all_models.py::test_all_models_fast_end_to_end[scNODE.yaml]"
```

## Trajectory Inference Module
This requires its own separate testing to make sure that the implementations are working as expected.
1. Create your own 2D data that is easy to reason about.
2. Run it through the trajectory inference module and make sure that it works.

## Metrics
We need to test each metric that we create and ensure that they work as expected.
