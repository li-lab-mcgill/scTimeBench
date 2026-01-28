"""
kNN implementation for trajectory inference.
"""
from crispy_fishstick.trajectory_infer.base import BaseTrajectoryInferMethod
import numpy as np
import logging
import torch

from geomloss import SamplesLoss
from pykeops.torch import generic_sum


# TODO: build a unit test for this class, to ensure that we're doing this properly
class OptimalTransport(BaseTrajectoryInferMethod):
    def __init__(self, traj_config):
        super().__init__(traj_config)

    def _subclass_parameters(self):
        return {
            "use_gene_expr": self.traj_config.get("use_gene_expr", False),
            "unbalanced_ot_blur": self.traj_config.get("unbalanced_ot_blur", 0.05),
            "unbalanced_ot_scaling": self.traj_config.get("unbalanced_ot_scaling", 0.5),
            "unbalanced_ot_reach": self.traj_config.get("unbalanced_ot_reach", None),
        }

    def _subclass_train(self, X_train, y_train, traj_infer_path):
        # here it's simple, we just save the training data for later use
        self.train_tensor = torch.FloatTensor(X_train)

        one_hot_encoding, index_to_type = self.cell_types_to_one_hot(y_train)
        logging.debug(f"One-hot encoding index to type: {index_to_type}")
        self.train_labels = one_hot_encoding
        self.index_to_type = index_to_type
        self.traj_infer_path = traj_infer_path

    def _subclass_predict_probs(self, embeds):
        """
        Infer the trajectory using an OT method. There are two types, one where we
        use the gene expression:
        1. This follows the STORIES paper, where we do the OT between gene expression
        2. Based on the transport plan we then transfer the cell type labels through kNN
        Note: They do it slightly differently, where they take the mass and then only take the top k,
        whereas we simply calculate the direct label transfer through OT.

        Otherwise, we just simply use the OT on the embeddings and transfer the labels
        based on the transport plan. We want to transfer from all embeddings to the unknown cells
        """
        # cache labels for faster access
        test_labels = (
            self.get_ot_labels(
                self.train_tensor, torch.FloatTensor(embeds), self.train_labels
            )
            .detach()
            .numpy()
        )

        # these test labels then need to be normalized to represent probabilities
        # sum(axis = 1) means sum across the columns, then keepdims to maintain the 2D shape
        logging.debug(f"Raw OT test labels: {test_labels}")
        test_labels = test_labels / (test_labels.sum(axis=1, keepdims=True) + 1e-8)
        logging.debug(f"Normalized OT test labels: {test_labels}")
        return test_labels, self.index_to_type

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
