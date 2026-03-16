"""
Definitions to be shared by the benchmark and the model implementations.
"""
from enum import Enum


class ObservationColumns(Enum):
    # need to prefix it so that it does not conflict with user data
    CELL_TYPE = "scTimeBench_cell_type"
    TIMEPOINT = "scTimeBench_timepoint"


class RequiredOutputFiles(Enum):
    EMBEDDING = "embedding.npy"
    NEXT_TIMEPOINT_EMBEDDING = "next_timepoint_embedding.npy"
    NEXT_TIMEPOINT_GENE_EXPRESSION = "next_timepoint_gene_expression.npy"
    # This is to be used only for the OT methods which can directly correlate
    # cells to cells, and thus build their lineage this way
    NEXT_CELLTYPE = "next_cell_type.parquet"
    PRED_GRAPH = "predicted_graph.npy"
    FROM_ZERO_TO_END_PRED_GEX = "from_zero_to_end_predicted_gene_expression.h5ad"


DATASET_DIR = "datasets"
PICKLED_DATASET_FILENAME = "dataset.pkl"
MODEL_CONFIG_FILENAME = "model_config.yaml"
