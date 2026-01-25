'''
Description:
    Run MIOFlow model.

Reference:
    [1] https://github.com/KrishnaswamyLab/MIOFlow/tree/main
'''
from MIOFlow.utils import generate_steps, set_seeds, config_criterion
from MIOFlow.models import make_model, Autoencoder
from MIOFlow.plots import plot_comparision, plot_losses
from MIOFlow.train import train_ae, training_regimen
from MIOFlow.constants import ROOT_DIR, DATA_DIR, NTBK_DIR, IMGS_DIR, RES_DIR

from MIOFlow.geo import setup_distance
from MIOFlow.exp import setup_exp
from MIOFlow.eval import generate_plot_data

import os, pandas as pd, numpy as np, \
    seaborn as sns, matplotlib as mpl, matplotlib.pyplot as plt, \
    torch, torch.nn as nn

import time


def trainModel(
        df, train_tps, test_tps, n_genes, n_epochs_emb=1000, samples_size_emb = (30,), gae_embedded_dim = 50,
        encoder_layers = [50, 50, 50], layers = [50, 50, 50],
        batch_size=32, n_local_epochs=40, n_global_epochs=40, n_post_local_epochs=0,
        lambda_density=35, pca_dims=50
):
    use_cuda = True
    hold_out = test_tps
    groups = train_tps
    # =================================
    # GAE hyperparameter
    distance_type = 'alpha_decay' # gaussian, alpha_decay
    rbf_length_scale = 0.001 # 0.1
    knn = 5
    t_max = 5
    dist = setup_distance(distance_type, rbf_length_scale=rbf_length_scale, knn=knn, t_max=t_max)
    # Construct GAE
    gae = Autoencoder(
        encoder_layers=encoder_layers,
        decoder_layers=encoder_layers[::-1],
        activation='ReLU', use_cuda=use_cuda
    ) # [model_features, hidden layer, gae_embedded_dim]
    optimizer = torch.optim.AdamW(gae.parameters())
    # GAE training
    recon = True # use reconstruction loss
    gae_losses = train_ae(
        gae, df, train_tps, optimizer,
        n_epochs=n_epochs_emb, sample_size=samples_size_emb,
        noise_min_scale=0.09, noise_max_scale=0.15,
        dist=dist, recon=recon, use_cuda=use_cuda
    )
    autoencoder = gae
    # plt.title("GAE Loss Curve")
    # plt.plot(gae_losses)
    # plt.show()
    # =================================
    # ODE hyperparameters
    use_density_loss = True
    activation = 'CELU' # LeakyReLU, ReLU, CELU
    sde_scales = (len(train_tps) + len(test_tps)) * [0.2]
    # Construct ODE model
    model_features = n_genes  # Use the actual number of features (50), not the GAE embedded dimension
    model = make_model(
        model_features, layers,
        activation=activation, scales=sde_scales, use_cuda=use_cuda
    )
    # ODE training
    sample_size = (batch_size,)
    n_local_epochs = n_local_epochs
    n_epochs = n_global_epochs
    n_post_local_epochs = n_post_local_epochs
    reverse_schema = True
    reverse_n = 2 # each reverse_n epoch
    criterion_name = 'ot'
    criterion = config_criterion(criterion_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    local_losses, batch_losses, globe_losses = training_regimen(
        # local, global, local train structure
        n_local_epochs=n_local_epochs,
        n_epochs=0,  # Disable global epochs to avoid KeyError
        n_post_local_epochs=n_post_local_epochs,
        exp_dir = "./",
        # BEGIN: train params
        model=model, df=df, groups=train_tps, optimizer=optimizer,
        criterion=criterion, use_cuda=use_cuda,
        use_density_loss=False,  # Disable density loss to avoid KeyError
        lambda_density=0,
        autoencoder=autoencoder,
        sample_size=sample_size,
        reverse_schema=False, reverse_n=reverse_n,  # Disable reverse schema
        # END: train params
    )
    opts = {"use_cuda": use_cuda, "autoencoder": autoencoder, "recon": recon}
    return model, gae_losses, local_losses, batch_losses, globe_losses, opts




def makeSimulation(df, model, tps, opts, n_sim_cells, n_trajectories=100, n_bins=100):
    use_cuda = opts["use_cuda"]
    autoencoder = opts["autoencoder"]
    recon = opts["recon"]
    n_points = n_sim_cells
    generated = generate_plot_data(
        model,
        df,
        n_points,
        n_trajectories,
        n_bins,
        sample_with_replacement=True,
        use_cuda=use_cuda,
        samples_key="samples",
        autoencoder=autoencoder,
        recon=recon,
    )
    if isinstance(generated, tuple):
        generated = generated[0]
    return generated


