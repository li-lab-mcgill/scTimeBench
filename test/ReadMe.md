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



## Metrics
We need to test each metric that we create and ensure that they work as expected. In particular, we are testing that BaseMetric will populate the evaluation table with something given all the leaf metrics.

Test with
```
pytest 03_metrics
```
This should run within ~20 minutes, so this is a long test.

## Datasets
We test that all the datasets work with scNODE (which paired with the end to end should mean that everything works as expected).

Test with
```
pytest 04_datasets
```
