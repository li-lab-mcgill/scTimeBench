"""
Definitions to be shared by the benchmark and the model implementations.
"""
from enum import Enum


class ObservationColumns(Enum):
    # need to prefix it so that it does not conflict with user data
    CELL_TYPE = "crispy_fishstick_cell_type"
    TIMEPOINT = "crispy_fishstick_timepoint"


# TODO: add in required constraints for what the model outputs
# TODO: should look like per metric
