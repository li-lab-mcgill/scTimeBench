"""
MIOFlow runner script.

This script trains and evaluates MIOFlow on an AnnData dataset.
It keeps the BaseModel runner structure used across the project.
"""

# let's add the model_utils path
import os
from typing import Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA

from crispy_fishstick.model_utils.model_runner import main, BaseModel
from crispy_fishstick.shared.constants import ObservationColumns

from MIOFlow.utils import set_seeds, config_criterion
from MIOFlow.models import make_model, Autoencoder
from MIOFlow.train import training_regimen, train_ae
from MIOFlow.geo import setup_distance
from MIOFlow.eval import generate_points


# next we want to prepare the data for training
def _to_dense(x):
    return x.toarray() if sp.issparse(x) else x


def prepare_data(ann_data, all_tps: List, pca_dims: int, seed: int):
    """
    Prepare data for MIOFlow training with PCA embeddings.
    """
    # get the time points
    cell_tps = ann_data.obs[ObservationColumns.TIMEPOINT.value].to_numpy()
    if not all_tps:
        raise ValueError(
            "all_tps must be provided for MIOFlow to build a stable timepoint mapping across splits."
        )
    # Use train+test timepoints to create mapping
    unique_tps = sorted(np.unique(all_tps))

    # Fit PCA on all available training data in this runner
    X = _to_dense(ann_data.X)
    pca = PCA(n_components=pca_dims, svd_solver="arpack", random_state=seed)
    X_pca = pca.fit_transform(X)

    # Map timepoints to contiguous indices (required by MIOFlow)
    tp_to_idx: Dict = {tp: i for i, tp in enumerate(unique_tps)}

    df = pd.DataFrame(X_pca, columns=[f"d{i}" for i in range(1, pca_dims + 1)])
    df.insert(0, "samples", [tp_to_idx[tp] for tp in cell_tps])

    return df, unique_tps, tp_to_idx, pca


def model_setup(
    model_features: int, layers: List[int], n_timepoints: int, use_cuda: bool
):
    # Model setup
    sde_scales = [0.2] * n_timepoints
    return make_model(
        model_features,
        layers,
        activation="CELU",
        scales=sde_scales,
        use_cuda=use_cuda,
    )


def model_training(
    df_train: pd.DataFrame,
    groups: List[int],
    metadata: Dict,
    model_features: int,
    n_timepoints: int,
    use_cuda: bool,
):
    # Model training
    n_local_epochs = metadata.get("n_local_epochs", 1)  # TODO updated back to 20
    n_epochs = metadata.get("n_epochs", 1)  # TODO  updated back to 80
    n_post_local_epochs = metadata.get("n_post_local_epochs", 0)
    batch_size = metadata.get(
        "batch_size", 64
    )  # TODO re-evaluate appropriate batch size
    layers = [
        int(x) for x in metadata.get("layers", "16,32,16").split(",") if x.strip() != ""
    ]
    lambda_density = metadata.get("lambda_density", 5.0)
    use_density_loss = metadata.get("use_density_loss", False)

    model = model_setup(model_features, layers, n_timepoints, use_cuda)
    criterion = config_criterion("ot")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    local_losses = defaultdict(list)
    ot_lambda_global = {i: 1.0 for i in range(len(groups))}

    training_regimen(
        sample_with_replacement=True,  ## TODO assess if this should be used or addressed elsewhere
        n_local_epochs=n_local_epochs,
        n_epochs=n_epochs,
        n_post_local_epochs=n_post_local_epochs,
        exp_dir=metadata.get("output_path", "./Output/AnnData"),
        model=model,
        df=df_train,
        groups=groups,
        optimizer=optimizer,
        criterion=criterion,
        use_cuda=use_cuda,
        hold_one_out=False,
        hold_out="random",
        use_density_loss=use_density_loss,
        lambda_density=lambda_density,
        autoencoder=metadata.get("autoencoder", None),
        use_emb=False,
        use_gae=metadata.get("use_gae", False),
        sample_size=(batch_size,),
        reverse_schema=True,
        reverse_n=2,
        plot_every=100,
        n_points=100,
        n_trajectories=100,
        n_bins=100,
        local_losses=local_losses,
        batch_losses=[],
        globe_losses=[],
        ot_lambda_global=ot_lambda_global,
    )

    return model


class MIOFlow(BaseModel):
    def train(self, ann_data, all_tps=None):
        """
        Training logic for MIOFlow.
        """
        cache_path = os.path.join(
            self.config["output_path"], "trained_mioflow_model.pth"
        )

        metadata = self.config.get("model", {}).get("metadata", {})
        pca_dims = metadata.get("pca_dims", 50)
        seed = metadata.get("seed", 0)
        use_gae = metadata.get("use_gae", False)
        gae_embedded_dim = metadata.get("gae_embedded_dim", 32)
        gae_encoder_layers = metadata.get("gae_encoder_layers", "")
        n_epochs_gae = metadata.get("n_epochs_gae", 10)
        gae_sample_size = metadata.get("gae_sample_size", 30)
        distance_type = metadata.get("distance_type", "gaussian")
        rbf_length_scale = metadata.get("rbf_length_scale", 0.5)
        knn = metadata.get("knn", 5)
        t_max = metadata.get("t_max", 5)

        set_seeds(seed)
        use_cuda = torch.cuda.is_available()

        expected_tps = sorted(np.unique(all_tps)) if all_tps else None

        if os.path.exists(cache_path):
            print("Trained MIOFlow model found, loading from file.")
            cache = torch.load(cache_path, map_location="cpu")
            cached_tps = cache.get("all_tps")
            if (
                expected_tps is not None
                and cached_tps is not None
                and list(cached_tps) != list(expected_tps)
            ):
                print("Cached MIOFlow model timepoints mismatch; retraining.")
            elif expected_tps is not None and cached_tps is None:
                print("Cached MIOFlow model missing timepoint mapping; retraining.")
            else:
                self.pca = cache["pca"]
                self.tp_to_idx = cache["tp_to_idx"]
                self.idx_to_tp = cache["idx_to_tp"]
                model_features = cache["model_features"]
                layers = cache["layers"]
                self.model = model_setup(
                    model_features, layers, len(self.tp_to_idx), use_cuda
                )
                self.model.load_state_dict(cache["model_state"])
                self.use_gae = cache.get("use_gae", False)
                self.autoencoder = None
                if self.use_gae:
                    encoder_layers = cache["gae_encoder_layers"]
                    gae = Autoencoder(
                        encoder_layers=encoder_layers,
                        decoder_layers=encoder_layers[::-1],
                        activation="ReLU",
                        use_cuda=use_cuda,
                    )
                    gae.load_state_dict(cache["autoencoder_state"])
                    self.autoencoder = gae
                self.df_train = None
                return

        df_train, unique_tps, tp_to_idx, pca = prepare_data(
            ann_data,
            all_tps=all_tps,
            pca_dims=pca_dims,
            seed=seed,
        )
        self.df_train = df_train
        self.pca = pca
        self.tp_to_idx = tp_to_idx
        self.idx_to_tp = {i: tp for tp, i in tp_to_idx.items()}

        autoencoder = None
        if use_gae:
            if gae_encoder_layers.strip() == "":
                encoder_layers = [pca_dims, 8, gae_embedded_dim]
            else:
                encoder_layers = [
                    int(x) for x in gae_encoder_layers.split(",") if x.strip() != ""
                ]
                if encoder_layers[0] != pca_dims:
                    raise ValueError("First GAE encoder layer must match pca_dims.")
                if encoder_layers[-1] != gae_embedded_dim:
                    raise ValueError(
                        "Last GAE encoder layer must match gae_embedded_dim."
                    )

            dist = setup_distance(
                distance_type,
                rbf_length_scale=rbf_length_scale,
                knn=knn,
                t_max=t_max,
            )

            gae = Autoencoder(
                encoder_layers=encoder_layers,
                decoder_layers=encoder_layers[::-1],
                activation="ReLU",
                use_cuda=use_cuda,
            )
            gae_optimizer = torch.optim.AdamW(gae.parameters())
            train_ae(
                gae,
                df_train,
                groups=sorted(df_train.samples.unique()),
                optimizer=gae_optimizer,
                n_epochs=n_epochs_gae,
                sample_size=(gae_sample_size,),
                noise_min_scale=0.09,
                noise_max_scale=0.15,
                dist=dist,
                recon=True,
                use_cuda=use_cuda,
                hold_one_out=False,
                hold_out="random",
            )
            autoencoder = gae
            gae_encoder_layers_cache = encoder_layers
        else:
            gae_encoder_layers_cache = None

        model_features = gae_embedded_dim if use_gae else pca_dims
        layers = [
            int(x)
            for x in metadata.get("layers", "16,32,16").split(",")
            if x.strip() != ""
        ]

        metadata["output_path"] = self.config["output_path"]
        metadata["autoencoder"] = autoencoder
        metadata["use_gae"] = use_gae
        self.model = model_training(
            df_train,
            groups=sorted(df_train.samples.unique()),
            metadata=metadata,
            model_features=model_features,
            n_timepoints=len(self.tp_to_idx),
            use_cuda=use_cuda,
        )
        self.autoencoder = autoencoder
        self.use_gae = use_gae

        cache_payload = {
            "pca": self.pca,
            "tp_to_idx": self.tp_to_idx,
            "idx_to_tp": self.idx_to_tp,
            "model_state": self.model.state_dict(),
            "model_features": model_features,
            "layers": layers,
            "use_gae": self.use_gae,
            "all_tps": expected_tps,
        }
        if self.use_gae and self.autoencoder is not None:
            cache_payload["autoencoder_state"] = self.autoencoder.state_dict()
            cache_payload["gae_encoder_layers"] = gae_encoder_layers_cache

        torch.save(cache_payload, cache_path)

    def _generate_all_embeddings(self, test_ann_data):
        """
        Generate embeddings for all timepoints using trained model.
        Returns tuple of (current_embeds, next_embeds, cell_tps, unique_tps).
        Caches results for reuse.
        """
        if hasattr(self, "_cached_embeddings"):
            return self._cached_embeddings

        self.model.eval()

        all_tp_indices = [self.tp_to_idx[tp] for tp in sorted(self.tp_to_idx.keys())]
        cell_tps = test_ann_data.obs[ObservationColumns.TIMEPOINT.value].to_numpy()
        unique_tps = sorted(np.unique(cell_tps))
        n_sim_cells = test_ann_data[
            test_ann_data.obs[ObservationColumns.TIMEPOINT.value] == unique_tps[0]
        ].shape[0]

        if self.df_train is None:
            # Rebuild a minimal df for generation when model loaded from cache
            X = self.pca.transform(_to_dense(test_ann_data.X))
            df_tmp = pd.DataFrame(
                X, columns=[f"d{i}" for i in range(1, X.shape[1] + 1)]
            )
            df_tmp.insert(
                0,
                "samples",
                [self.tp_to_idx[tp] for tp in cell_tps],
            )
            df_for_gen = df_tmp
        else:
            df_for_gen = self.df_train

        generated = generate_points(
            self.model,
            df_for_gen,
            n_points=n_sim_cells,
            sample_with_replacement=True,
            use_cuda=torch.cuda.is_available(),
            samples_key="samples",
            sample_time=all_tp_indices,
            autoencoder=self.autoencoder,
            recon=self.use_gae,
        )

        generated = np.asarray(generated)
        tp_to_gen_pos = {tp_idx: i for i, tp_idx in enumerate(all_tp_indices)}
        rng = np.random.default_rng(0)

        def _select_generated(gen, gen_pos, n_cells):
            if gen.ndim == 3:
                gen_slice = gen[gen_pos]
                if gen_slice.shape[0] == n_cells:
                    return gen_slice
                if gen_slice.shape[0] < n_cells:
                    idx = rng.choice(gen_slice.shape[0], size=n_cells, replace=True)
                    return gen_slice[idx]
                return gen_slice[:n_cells]
            if gen.ndim == 2:
                gen_vec = gen[gen_pos]
                return np.repeat(gen_vec[None, :], n_cells, axis=0)
            if gen.ndim == 1:
                return np.repeat(gen[None, :], n_cells, axis=0)
            raise ValueError(f"Unexpected generated shape: {gen.shape}")

        if generated.ndim == 3:
            n_features = generated.shape[2]
        elif generated.ndim == 2:
            n_features = generated.shape[1]
        elif generated.ndim == 1:
            n_features = generated.shape[0]
        else:
            raise ValueError(f"Unexpected generated shape: {generated.shape}")

        # Current timepoint embeddings
        embeds = np.empty((test_ann_data.n_obs, n_features), dtype=generated.dtype)
        for tp in unique_tps:
            mask = cell_tps == tp
            tp_idx = self.tp_to_idx.get(tp)
            if tp_idx is None or tp_idx not in tp_to_gen_pos:
                raise ValueError(f"Timepoint {tp} not found in generated samples.")
            gen_pos = tp_to_gen_pos[tp_idx]
            embeds[mask] = _select_generated(generated, gen_pos, int(mask.sum()))

        # Next timepoint embeddings
        next_embeds = np.full(
            (test_ann_data.n_obs, n_features), np.nan, dtype=generated.dtype
        )
        for i, tp in enumerate(unique_tps):
            mask = cell_tps == tp
            if i + 1 >= len(unique_tps):
                continue
            next_tp = unique_tps[i + 1]
            next_idx = self.tp_to_idx.get(next_tp)
            if next_idx is None or next_idx not in tp_to_gen_pos:
                continue
            gen_pos = tp_to_gen_pos[next_idx]
            next_embeds[mask] = _select_generated(generated, gen_pos, int(mask.sum()))

        self._cached_embeddings = (embeds, next_embeds, cell_tps, unique_tps)
        return self._cached_embeddings

    def generate_embedding(self, test_ann_data) -> np.ndarray:
        """
        Generate embeddings for the current timepoint.
        """
        embeds, _, _, _ = self._generate_all_embeddings(test_ann_data)
        return embeds

    def generate_next_tp_embedding(self, test_ann_data) -> np.ndarray:
        """
        Generate embeddings for the next timepoint.
        """
        _, next_embeds, _, _ = self._generate_all_embeddings(test_ann_data)
        return next_embeds


if __name__ == "__main__":
    main(MIOFlow)
