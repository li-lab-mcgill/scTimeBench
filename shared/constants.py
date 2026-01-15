"""
Definitions to be shared by the benchmark and the model implementations.
"""
from enum import Enum


class ObservationColumns(Enum):
    # need to prefix it so that it does not conflict with user data
    CELL_TYPE = "crispy_fishstick_cell_type"
    TIMEPOINT = "crispy_fishstick_timepoint"


class RequiredOutputColumns(Enum):
    EMBEDDING = "crispy_fishstick_embedding"
    NEXT_TIMEPOINT_EMBEDDING = "crispy_fishstick_next_timepoint_embedding"
