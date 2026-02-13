"""
This script implements a simple cooccurrence-based model for trajectory inference.
It does not actually train a model, but simply generates a predicted graph based
on the cooccurrence of cell types at adjacent time points.
"""

import numpy as np
from crispy_fishstick.model_utils.model_runner import main, BaseModel
from crispy_fishstick.shared.constants import RequiredOutputFiles
from crispy_fishstick.shared.constants import ObservationColumns


class Cooccurrence(BaseModel):
    def __init__(self, yaml_config):
        super().__init__(yaml_config)

        # select the option that has PRED_GRAPH
        for option in self.required_outputs_options:
            if RequiredOutputFiles.PRED_GRAPH in option:
                self.required_outputs = option
                break

    def train(self, ann_data, all_tps=None):
        """
        For cooccurrence, we don't actually train a model :)
        """
        print(f"No such training exists for cooccurrence :)")

    def generate_pred_graph(self, test_ann_data) -> np.ndarray:
        """
        Here we create the predicted graph by doing a simple cooccurrence-based approach.
        For every (t, t + 1), we calculate the cooccurrence between cells types at time t and cells types at time t + 1,
        then we 0 out the rows for cells types that don't exist in t, so that we only have edges from cells at t to cells at t + 1.

        Then, we sum this up for all (t, t + 1) pairs to get the final predicted graph,
        and normalize it to be between 0 and 1.
        """
        test_tps = (
            test_ann_data.obs[ObservationColumns.TIMEPOINT.value].unique().tolist()
        )
        test_tps.sort()

        # ** Important: we require the same ordering as the graphsimmetrics for this to work **
        cell_types = (
            test_ann_data.obs[ObservationColumns.CELL_TYPE.value].unique().tolist()
        )
        cell_type_to_id = {
            cell_type: idx for idx, cell_type in enumerate(sorted(cell_types))
        }

        time_col = ObservationColumns.TIMEPOINT.value
        cell_type_col = ObservationColumns.CELL_TYPE.value
        obs = test_ann_data.obs
        graph_pred = np.zeros((len(cell_types), len(cell_types)))

        for i in range(len(test_tps) - 1):
            t0 = test_tps[i]
            t1 = test_tps[i + 1]

            sorted_cell_types = sorted(cell_types)

            idx_t0 = np.where(obs[time_col] == t0)[0]
            idx_t1 = np.where(obs[time_col] == t1)[0]
            if idx_t0.size == 0 or idx_t1.size == 0:
                continue

            ct_counts_t0 = obs.iloc[idx_t0][cell_type_col].value_counts().to_dict()
            ct_counts_t1 = obs.iloc[idx_t1][cell_type_col].value_counts().to_dict()

            # now let's turn these dictionaries to vectors where the order is the same as cell_type_to_id
            ct_counts_t0_arr = np.array(
                [ct_counts_t0.get(cell_type, 0) for cell_type in sorted_cell_types]
            )
            ct_counts_t1_arr = np.array(
                [ct_counts_t1.get(cell_type, 0) for cell_type in sorted_cell_types]
            )

            # now let's first normalize these counts to get distributions
            ct_counts_t0_arr = ct_counts_t0_arr / np.linalg.norm(ct_counts_t0_arr)
            ct_counts_t1_arr = ct_counts_t1_arr / np.linalg.norm(ct_counts_t1_arr)

            cooccurrence = np.outer(ct_counts_t0_arr, ct_counts_t1_arr)

            # now let's only keep the rows that correspond to cell types that exist in t0
            for ct0 in ct_counts_t0.keys():
                graph_pred[cell_type_to_id[ct0], :] += cooccurrence[
                    cell_type_to_id[ct0], :
                ]

        for i in range(graph_pred.shape[0]):
            if graph_pred[i, :].sum() > 0:
                graph_pred[i, :] = graph_pred[i, :] / graph_pred[i, :].sum()

        return graph_pred


if __name__ == "__main__":
    main(Cooccurrence)
