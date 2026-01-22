"""
kNN implementation for trajectory inference.
"""
from crispy_fishstick.trajectory_infer.base import BaseTrajectoryInferMethod
from crispy_fishstick.shared.constants import ObservationColumns, RequiredOutputColumns
import numpy as np
import logging
import torch

from geomloss import SamplesLoss
from pykeops.torch import generic_sum


# TODO: build a unit test for this class, to ensure that we're doing this properly
class OptimalTransport(BaseTrajectoryInferMethod):
    def __init__(self, traj_config):
        super().__init__(traj_config)

    def uses_gene_expr(self):
        """
        OT method is the only one that we allow to
        """
        return self.traj_config.get("use_gene_expr", False)

    def _get_tensors_for_traj(self, ann_data):
        """
        Based on the _use_gene_expr function, get the proper tensors for trajectory inference.

        We want to return:
        - Original gene expr/embedding at time t (for all t)
        - Predicted gene expr/embedding at time t (for (1, last t))
        """
        if self.uses_gene_expr():
            return (
                torch.from_numpy(ann_data.X.toarray()),
                ann_data.obsm[
                    RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION.value
                ],
            )
        else:
            return (
                ann_data.obsm[RequiredOutputColumns.EMBEDDING.value],
                ann_data.obsm[RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING.value],
            )

    def _method_infer_trajectory(self, ann_data):
        """
        Infer the trajectory using an OT method. There are two types, one where we
        use the gene expression:
        1. This follows the STORIES paper, where we do the OT between gene expression
        2. Based on the transport plan we then transfer the cell type labels through kNN
        Note: They do it slightly differently, where they take the mass and then only take the top k,
        whereas we simply calculate the direct label transfer through OT.

        Otherwise, we just simply use the OT on the embeddings and transfer the labels
        based on the transport plan.
        """
        cur_tensor, next_tensor = self._get_tensors_for_traj(ann_data)
        timepoints = ann_data.obs[ObservationColumns.TIMEPOINT.value]
        cell_types = ann_data.obs[ObservationColumns.CELL_TYPE.value]
        unique_timepoints = sorted(np.unique(timepoints))

        one_hot_encoding, index_to_type = self.cell_types_to_one_hot(cell_types)
        logging.debug(f"One-hot encoding index to type: {index_to_type}")

        total_pred_cells = 0
        cell_lineage = {}
        for i in range(len(unique_timepoints) - 1):
            # get indices for current timepoint
            idx_current = np.where(timepoints == unique_timepoints[i])[0]
            # get indices for next timepoint
            idx_next = np.where(timepoints == unique_timepoints[i + 1])[0]
            # get the cell types for the next timepoint, where idx_next[i] corresponds to next_cell_types[i]
            next_cell_types = one_hot_encoding[idx_next]

            # later for logging purposes
            total_pred_cells += next_cell_types.shape[0]

            # now let's do the label transfer using optimal transport
            pred_tensor = torch.FloatTensor(next_tensor[idx_current])
            true_next_tensor = torch.FloatTensor(cur_tensor[idx_next])

            labels = (
                self.get_ot_labels(true_next_tensor, pred_tensor, next_cell_types)
                .detach()
                .numpy()
            )

            # now let's map the soft labels to hard cell type labels
            cell_type_labels = self.soft_labels_to_cell_types(labels, index_to_type)

            # now based on these cell type labels, let's build the lineage
            for cur_cell, next_cell in zip(
                cell_types.iloc[idx_current], cell_type_labels
            ):
                if cur_cell not in cell_lineage:
                    cell_lineage[cur_cell] = {}
                if next_cell not in cell_lineage[cur_cell]:
                    cell_lineage[cur_cell][next_cell] = 0
                cell_lineage[cur_cell][next_cell] += 1

            logging.debug(
                f"Cell lineage after timepoint {unique_timepoints[i]}: {cell_lineage}"
            )

        # finally what we do is count the total number of "unknown" transitions and remove them
        total_unknowns = 0
        for source_cell_type in cell_lineage.keys():
            if "unknown" in cell_lineage[source_cell_type]:
                total_unknowns += cell_lineage[source_cell_type]["unknown"]
                del cell_lineage[source_cell_type]["unknown"]
        logging.debug(
            f"Total unknown transitions removed: {total_unknowns} out of {total_pred_cells} cells"
        )

        logging.debug(f"Constructed cell lineage (raw counts): {cell_lineage}")
        # finally, we normalize the counts
        for source_cell_type in cell_lineage.keys():
            total_counts = sum(cell_lineage[source_cell_type].values())
            for target_cell_type in cell_lineage[source_cell_type]:
                cell_lineage[source_cell_type][target_cell_type] /= total_counts

        return cell_lineage

    def cell_types_to_one_hot(self, cell_types):
        """
        Given a list of cell types, convert to one-hot encoding
        """
        unique_clusters = sorted(np.unique(cell_types).tolist())
        type_to_index = {tp: i for i, tp in enumerate(unique_clusters)}
        # ! Important: This needs to be torch and not numpy or else it causes issues!
        one_hot = torch.zeros((len(cell_types), len(unique_clusters)))
        for i, tp in enumerate(cell_types):
            one_hot[i][type_to_index[tp]] = 1
        return one_hot, unique_clusters

    def soft_labels_to_cell_types(self, labels, index_to_type):
        """
        Given the labels from get_ot_labels, and the index to type mapping,
        convert the soft labels to hard cell type labels
        """
        cell_type_labels = np.full(labels.shape[0], "", dtype=object)
        for i in range(labels.shape[0]):
            if sum(labels[i]) == 0 or any(labels[i] == np.nan):
                cell_type_labels[i] = "unknown"
            else:
                cell_type_labels[i] = index_to_type[np.argmax(labels[i]).item()]

        return cell_type_labels

    def get_ot_labels(self, true_embed, pred_embed, one_hot_labels):
        """
        Given the true embeddings, predicted embeddings and one-hot encoding of true cell types,
        get the transport plan using optimal transport
        """
        ot_solver = SamplesLoss(
            "sinkhorn",
            p=2,
            blur=self.traj_config.get("unbalanced_ot_blur", 0.05),
            debias=True,
            backend="tensorized",
            scaling=self.traj_config.get("unbalanced_ot_scaling", 0.5),
            potentials=True,
            reach=self.traj_config.get(
                "unbalanced_ot_reach", None
            ),  # set this to be None for balanced OT, otherwise a float value
        )

        F, G = ot_solver(pred_embed, true_embed)

        # ! Important: everything that is inputted to KeOps needs to be torch FloatTensor
        # ! Or else it just won't work... I hate this sometimes
        transfer = generic_sum(
            "Exp( (F_i + G_j - IntInv(2)*SqDist(X_i,Y_j)) / E ) * L_j",  # See the formula above
            f"Lab = Vi({one_hot_labels.shape[1]})",  # Output:  one vector of size one_hot_labels per line
            "E   = Pm(1)",  # 1st arg: a scalar parameter, the temperature
            f"X_i = Vi({pred_embed.shape[1]})",  # 2nd arg: one 2d-point per line
            f"Y_j = Vj({true_embed.shape[1]})",  # 3rd arg: one 2d-point per column
            "F_i = Vi(1)",  # 4th arg: one scalar value per line
            "G_j = Vj(1)",  # 5th arg: one scalar value per column
            f"L_j = Vj({one_hot_labels.shape[1]})",
        )  # 6th arg: one vector of size 3 per column

        # And apply it on the data (KeOps is pretty picky on the input shapes...):
        labels_i = (
            transfer(
                torch.Tensor(
                    [self.traj_config.get("unbalanced_ot_blur", 0.05) ** 2]
                ).type(torch.FloatTensor),
                pred_embed,
                true_embed,
                F.view(-1, 1),
                G.view(-1, 1),
                one_hot_labels,
            )
            / true_embed.shape[0]
        )

        return labels_i
