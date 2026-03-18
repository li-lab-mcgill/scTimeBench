# scTimeBench Package (formely Crispy Fishstick)
## Metrics
Under `src/scTimeBench/metrics` you can define separate metrics as you want! The structure of the folders here are very hierarchical, and you can see them separated into larger categories of trajectory inference, cell ontology, GRN inference and perturbation analysis.

## Model Utils
Under `src/scTimeBench/method_utils`, we define a `method_runner.py` class that is very useful for running your own external modules such as scNODE. See `models/scNODE/run.py` for an example use case.

## Shared
Under `src/scTimeBench/shared`, we define different constants, and datasets to be used. This includes important constants such as the required model outputs.

## Trajectory Inference
Under `src/scTimeBench/trajectory_infer`, we define the different methods that can be used to infer trajectories from embeddings.

## Others (database, config)
Under `src/scTimeBench/database.py` we have a SQL database manager. Feel free to modify and/or edit for your own purpose such as running different queries for post-hoc analysis.

We also define `src/scTimeBench/config.py` for the different types of models that are required for running.
