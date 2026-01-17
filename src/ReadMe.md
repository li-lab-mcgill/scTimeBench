# Crispy Fishstick Package
## Metrics
Under `src/crispy_fishstick/metrics` you can define separate metrics as you want! The structure of the folders here are very hierarchical, and you can see them separated into larger categories of trajectory inference, cell ontology, GRN inference and perturbation analysis.

## Model Utils
Under `src/crispy_fishstick/model_utils`, we define a `model_runner.py` class that is very useful for running your own external modules such as scNODE. See `models/scNODE/run.py` for an example use case.

## Shared
Under `src/crispy_fishstick/shared`, we define different constants, and datasets to be used. This includes important constants such as the required model outputs.

## Trajectory Inference
Under `src/crispy_fishstick/trajectory_infer`, we define the different methods that can be used to infer trajectories from embeddings.

## Others (database, config)
Under `src/crispy_fishstick/database.py` we have a SQL database manager. Feel free to modify and/or edit for your own purpose such as running different queries for post-hoc analysis.

We also define `src/crispy_fishstick/config.py` for the different types of models that are required for running.
