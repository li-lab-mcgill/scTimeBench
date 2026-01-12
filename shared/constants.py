"""
Definitions to be shared by the benchmark and the model implementations.
"""
from enum import Enum


class ObservationColumns(Enum):
    CELL_TYPE = "cell_type"
    TIMEPOINT = "timepoint"


class RequiredGeneExpressionColumns(Enum):
    EXPRESSION = "expression"
    # so we can map back to original cells after processing
    # in case the model changes the order of cells
    CELL_ID = "cell_id"
