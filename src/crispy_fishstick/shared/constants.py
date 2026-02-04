"""
Definitions to be shared by the benchmark and the model implementations.
"""
from enum import Enum


class ObservationColumns(Enum):
    # need to prefix it so that it does not conflict with user data
    CELL_TYPE = "crispy_fishstick_cell_type"
    TIMEPOINT = "crispy_fishstick_timepoint"


class RequiredOutputFiles(Enum):
    EMBEDDING = "embedding.npy"
    NEXT_TIMEPOINT_EMBEDDING = "next_timepoint_embedding.npy"
    NEXT_TIMEPOINT_GENE_EXPRESSION = "next_timepoint_gene_expression.npy"
    # This is to be used only for the OT methods which can directly correlate
    # cells to cells, and thus build their lineage this way
    NEXT_CELLTYPE = "next_cell_type.parquet"
