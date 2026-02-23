"""
ExampleRandomSampler script.

We use this script to train the ExampleRandomSampler model on a dataset.
Where we simply memorize a random sample from each time point.
"""

# let's add the model_utils path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "./scNODE_module"))

from crispy_fishstick.model_utils.model_runner import main, BaseModel
from crispy_fishstick.shared.constants import ObservationColumns
import numpy as np
import torch
import scanpy as sc
import random
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
    metadata,
):
    # Model training
    pretrain_iters = metadata.get("pretrain_iters", 200)
    pretrain_lr = 1e-3
    epochs = metadata.get("epochs", 10)
    seed = metadata.get("seed", 42)
    iters = 100
    batch_size = 32
    lr = 1e-3
    latent_ode_model = model_setup(train_data[0].shape[1])

    # now let's set the seed for reproducibility -- not quite sure which one, but let's do all
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
    def train(self, ann_data, all_tps=None):
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
        latent_ode_model, _, _, _, _ = model_training(
            traj_data, tps, self.config["model"].get("metadata", {})
        )
        # now let's cache the trained model
        self.latent_ode_model = latent_ode_model

        # store to .pth file
        torch.save(self.latent_ode_model.state_dict(), cache_path)

    def generate_embedding(self, test_ann_data) -> np.ndarray:
        """
        Generate embeddings for the current timepoint using VAE reconstruction.
        """
        self.latent_ode_model.eval()

        data = test_ann_data.X.toarray()
        embeds, _ = self.latent_ode_model.vaeReconstruct([data])
        embeds = embeds[0]

        print(f"Embeddings shape: {embeds.shape}")
        return embeds.detach().numpy()

    def generate_next_tp_embedding(self, test_ann_data) -> np.ndarray:
        """
        Generate embeddings for the next timepoint.
        """
        self.latent_ode_model.eval()

        data = test_ann_data.X.toarray()
        cell_tps = test_ann_data.obs[ObservationColumns.TIMEPOINT.value].to_numpy()
        unique_tps = sorted(np.unique(cell_tps))

        next_timepoint_embeds = []
        for cell, tp in zip(data, cell_tps):
            # given the unique_tps that we have, find the next timepoint
            next_tps = [t for t in unique_tps if t > tp]
            if not next_tps:
                # if there is no next timepoint, we just return a NaN object
                next_timepoint_embeds.append(
                    np.full((self.latent_ode_model.latent_dim,), np.nan)
                )
                continue

            cell = cell.reshape(1, -1)

            _, pred_embed, _ = self.latent_ode_model.predict(
                torch.FloatTensor(cell),
                torch.FloatTensor([tp, next_tps[0]]),
                n_cells=1,
            )
            pred_embed = pred_embed[0]  # get the first (and only) batch
            next_timepoint_embeds.append(pred_embed[1].detach().numpy())

        return np.vstack(next_timepoint_embeds)

    def generate_next_tp_gex(self, test_ann_data) -> np.ndarray:
        """
        Generate gene expression for the next timepoint.
        """
        self.latent_ode_model.eval()

        data = test_ann_data.X.toarray()
        cell_tps = test_ann_data.obs[ObservationColumns.TIMEPOINT.value].to_numpy()
        unique_tps = sorted(np.unique(cell_tps))

        next_timepoint_gene_expr = []
        for cell, tp in zip(data, cell_tps):
            # given the unique_tps that we have, find the next timepoint
            next_tps = [t for t in unique_tps if t > tp]
            if not next_tps:
                # if there is no next timepoint, we just return a NaN object
                next_timepoint_gene_expr.append(np.full((data.shape[1],), np.nan))
                continue

            cell = cell.reshape(1, -1)

            _, _, recon_obs = self.latent_ode_model.predict(
                torch.FloatTensor(cell),
                torch.FloatTensor([tp, next_tps[0]]),
                n_cells=1,
            )
            recon_obs = recon_obs[0]  # get the first (and only) batch
            next_timepoint_gene_expr.append(recon_obs[1].detach().numpy())

        return np.vstack(next_timepoint_gene_expr)

    def generate_zero_to_end_pred_gex(self, first_tp_cells, all_tps) -> sc.AnnData:
        """
        Generate predicted gene expression from the first to the last timepoint.
        Returns: AnnData object with predicted gene expression across all timepoints
        """
        self.latent_ode_model.eval()

        # first off let's get the cells at tp0
        # and then project it forward using latent_ode_model.predict
        data = first_tp_cells.X.toarray()
        _, _, recon_obs = self.latent_ode_model.predict(
            torch.FloatTensor(data),
            torch.FloatTensor(all_tps),
            n_cells=data.shape[0],
        )

        recon_obs = recon_obs.detach().numpy()

        # recon_obs is cells by tps by genes
        # which we can recover cells by genes with:
        # the timepoint information should be simply [:, i, :]
        pred_ann_data = first_tp_cells.copy()
        for i, tp in enumerate(all_tps[1:], 1):
            # create a new AnnData object for each timepoint
            tp_ann_data = first_tp_cells.copy()
            tp_ann_data.X = recon_obs[:, i, :]
            tp_ann_data.obs[ObservationColumns.TIMEPOINT.value] = tp
            pred_ann_data = sc.concat([pred_ann_data, tp_ann_data], axis=0)
            print(f"Shape of {i}th timepoint: {tp_ann_data.shape}")

        return pred_ann_data


if __name__ == "__main__":
    main(scNODE)
