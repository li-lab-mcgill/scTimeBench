"""
Definitions to be shared by the benchmark and the method implementations.
"""
from enum import Enum


class ObservationColumns(Enum):
    # need to prefix it so that it does not conflict with user data
    CELL_TYPE = "scTimeBench_cell_type"
    TIMEPOINT = "scTimeBench_timepoint"
    FUTURE_TIMEPOINTS = "scTimeBench_future_timepoints"


class RequiredOutputFiles(Enum):
    EMBEDDING = "embedding.npy"
    NEXT_TIMEPOINT_EMBEDDING = "next_timepoint_embedding.npy"
    NEXT_TIMEPOINT_GENE_EXPRESSION = "next_timepoint_gene_expression.npy"
    # This is to be used only for the OT methods which can directly correlate
    # cells to cells, and thus build their lineage this way
    NEXT_CELLTYPE = "next_cell_type.parquet"
    PRED_GRAPH = "predicted_graph.npy"
    FROM_ZERO_TO_END_PRED_GEX = "from_zero_to_end_predicted_gene_expression.h5ad"
    # perturbation ann data
    PERTURBED_TEST_ANN_DATA = "perturbed_test_ann_data.h5ad"
    # perturbation ann data for t to t + 1
    PERTURBED_TEST_ANN_DATA_T_TO_T_PLUS_ONE = (
        "perturbed_test_ann_data_t_to_t_plus_one.h5ad"
    )
    # flag to make sure that the meta metric can run
    META_FLAG = "meta_flag.txt"


DATASET_DIR = "datasets"
PICKLED_DATASET_FILENAME = "dataset.pkl"
METHOD_CONFIG_FILENAME = "method_config.yaml"
PERTURBATION_SET_CONFIG_FILENAME = "perturbation_config.yaml"
GLOBAL_PERTURBATION_SET_CONFIG_FILENAME = "global_perturbation_config.yaml"
