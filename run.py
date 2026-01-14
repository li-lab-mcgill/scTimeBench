"""
ExampleRandomSampler script.

We use this script to train the ExampleRandomSampler model on a dataset.
Where we simply memorize a random sample from each time point.
"""

# let's add the model_utils path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "./scNODE_module"))

from model_utils.parser import main, BaseModel
from shared.constants import ObservationColumns
import numpy as np
import torch
from optim.running import constructscNODEModel, scNODETrainWithPreTrain


# next we want to prepare the data for training
def prepare_data(ann_data):
    """
    Prepare data for scNODE training.
    """
    # first we get the data
    data = ann_data.X

    # get the time points
    cell_tps = ann_data.obs[ObservationColumns.TIMEPOINT.value].to_numpy()
    unique_tps = sorted(np.unique(cell_tps))

    # Convert to torch project
    # so right now, we have it s.t. if the time points do match up, we get the data
    # np.where returns a tuple, the array we care about is the first element
    traj_data = [
        torch.FloatTensor(data[np.where(cell_tps == t)[0], :].toarray())
        for t in unique_tps
    ]

    tps = torch.FloatTensor(unique_tps)
    n_cells = [each.shape[0] for each in traj_data]
    for cell, tp in zip(n_cells, unique_tps):
        print(f"Time point {tp} has {cell} cells.")

    return traj_data, tps


def model_setup(n_genes):
    # Model setup
    latent_dim = 50
    drift_latent_size = [50, 50]
    enc_latent_list = [64, 64]
    dec_latent_list = [64, 64]

    latent_ode_model = constructscNODEModel(
        n_genes,
        latent_dim=latent_dim,
        enc_latent_list=enc_latent_list,
        dec_latent_list=dec_latent_list,
        drift_latent_size=drift_latent_size,
        latent_enc_act="none",
        latent_dec_act="relu",
        drift_act="relu",
        ode_method="euler",
    )

    return latent_ode_model


def model_training(
    train_data,
    train_tps,
):
    # Model training
    pretrain_iters = 200
    pretrain_lr = 1e-3
    epochs = 10
    iters = 100
    batch_size = 32
    lr = 1e-3
    latent_ode_model = model_setup(train_data[0].shape[1])

    return scNODETrainWithPreTrain(
        train_data,
        train_tps,
        latent_ode_model,
        latent_coeff=1.0,
        epochs=epochs,
        iters=iters,
        batch_size=batch_size,
        lr=lr,
        kl_coeff=0.0,
        pretrain_iters=pretrain_iters,
        pretrain_lr=pretrain_lr,
    )


class scNODE(BaseModel):
    def train(self, ann_data):
        """
        Training logic for scNODE.
        """
        # so we already have the preprocessed ann_data with columns
        # ObservationColumns.TIMEPOINT and ObservationColumns.CELL_TYPE
        cache_path = os.path.join(
            self.config["output_path"], "trained_scNODE_model.pth"
        )

        if os.path.exists(cache_path):
            print("Trained scNODE model found, loading from file.")
            self.latent_ode_model = model_setup(ann_data.X.shape[1])
            self.latent_ode_model.load_state_dict(torch.load(cache_path))
            return

        # then, we need to prepare this for scNODE training
        traj_data, tps = prepare_data(ann_data)
        latent_ode_model, _, _, _, _ = model_training(traj_data, tps)
        # now let's cache the trained model
        self.latent_ode_model = latent_ode_model

        # store to .pth file
        torch.save(self.latent_ode_model.state_dict(), cache_path)

    def generate(self, test_ann_data, expected_output_path):
        """
        Generation logic with interpolation.
        Returns an AnnData object containing the generated samples.
        """
        # now let's evaluate the data
        self.latent_ode_model.eval()
        traj_data, tps = prepare_data(test_ann_data)


if __name__ == "__main__":
    main(scNODE)
