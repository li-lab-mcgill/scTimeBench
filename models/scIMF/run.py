"""
scIMF runner script.

This script trains and evaluates scIMF on an AnnData dataset.
It keeps the BaseModel runner structure used across the project.
"""

import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
import tqdm
import geomloss
from sklearn.decomposition import PCA

from scTimeBench.model_utils.model_runner import main, BaseModel
from scTimeBench.shared.constants import ObservationColumns


_SCIMF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "scIMF_module"))
if os.path.isdir(_SCIMF_PATH) and _SCIMF_PATH not in sys.path:
    sys.path.append(_SCIMF_PATH)

from config import config as config_builder, init_config  # type: ignore
from utils import init_all  # type: ignore
from model import MultiCNet  # type: ignore
from load_Data import constructOutDir  # type: ignore


def _ensure_dense_float32(x):
    try:
        import scipy.sparse as sp  # type: ignore

        if sp.issparse(x):
            x = x.toarray()
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)


def _sorted_unique(values):
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.number):
        return list(np.sort(np.unique(values)))
    try:
        import natsort  # type: ignore

        return list(natsort.natsorted(np.unique(values)))
    except Exception:
        return list(sorted(np.unique(values).tolist()))


def _map_leaveouts(leaveouts, mapping, n_tps):
    mapped = []
    for lo in leaveouts:
        if isinstance(lo, (int, np.integer)) and 0 <= int(lo) < n_tps:
            mapped.append(int(lo))
        elif lo in mapping:
            mapped.append(mapping[lo])
        else:
            raise ValueError(f"Invalid leaveout timepoint: {lo}")
    return sorted(set(mapped))


def _select_generated(gen, gen_pos, n_cells, rng):
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


class scIMF(BaseModel):
    def _build_config(self, metadata: Dict, all_tps: Optional[List] = None):
        args = config_builder()

        if "epochs" in metadata and "train_epochs" not in metadata:
            metadata = dict(metadata)
            metadata["train_epochs"] = metadata["epochs"]
        if "batch_size" in metadata and "train_batch" not in metadata:
            metadata = dict(metadata)
            metadata["train_batch"] = metadata["batch_size"]
        if "lr" in metadata and "train_lr" not in metadata:
            metadata = dict(metadata)
            metadata["train_lr"] = metadata["lr"]

        for key, value in metadata.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"[warn] Ignoring unknown scIMF config parameter: {key}")

        args = init_all(args)
        config = init_config(args)

        if all_tps is None:
            raise ValueError("all_tps is required to build a stable timepoint mapping.")

        unique_tps = _sorted_unique(all_tps)
        if len(unique_tps) < 2:
            raise ValueError("At least two timepoints are required for training")

        mapping = {tp: idx for idx, tp in enumerate(unique_tps)}
        n_tps = len(unique_tps)

        leaveouts = metadata.get("leaveouts", None)
        if leaveouts is not None:
            config.leaveouts = _map_leaveouts(leaveouts, mapping, n_tps)
        else:
            config.leaveouts = []

        if 0 in config.leaveouts:
            config.leaveouts = [t for t in config.leaveouts if t != 0]

        config.task = "leaveout" if config.leaveouts else "fate"
        config.dataset = metadata.get("dataset", "AnnData")
        config.out_dir = self.config["output_path"]
        config.for_train = True

        config.split_type = (
            "all_times"
            if len(config.leaveouts) == 0
            else f"leaveout_{'_'.join(map(str, config.leaveouts))}"
        )

        config.Train_ts = list(range(1, n_tps))
        config.train_t = list(sorted(set(config.Train_ts) - set(config.leaveouts)))
        config.test_t = list(config.leaveouts)

        config.out_dir = "{}/{}/{}/seed_{}/Ours".format(
            config.out_dir,
            config.dataset,
            config.split_type,
            config.seed,
        )
        config = constructOutDir(config)

        return config, mapping

    def _resolve_device(self, metadata: Dict):
        use_cuda = metadata.get("use_cuda", True)
        cuda_device = metadata.get("cuda_device", 0)
        auto_device = metadata.get("auto_device", True)

        if use_cuda and torch.cuda.is_available():
            device = torch.device(f"cuda:{cuda_device}")
            if auto_device:
                try:
                    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
                    if free_bytes < 512 * 1024 * 1024:
                        print(
                            f"[warn] Low free GPU memory on {device} ({free_bytes / 1024**2:.2f} MiB). Falling back to CPU."
                        )
                        device = torch.device("cpu")
                except Exception:
                    pass
        else:
            device = torch.device("cpu")

        return device

    def train(self, ann_data, all_tps=None):
        """
        Training logic for scIMF.
        """
        cache_path = os.path.join(self.config["output_path"], "trained_scIMF_model.pth")
        metadata = self.config.get("model", {}).get("metadata", {})

        self.device = self._resolve_device(metadata)

        if os.path.exists(cache_path):
            print("Trained scIMF model found, loading from file.")
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.tp_to_idx = cache["tp_to_idx"]
            self.idx_to_tp = {i: tp for tp, i in self.tp_to_idx.items()}
            self.pca_model = cache["pca_model"]
            self.data_t0 = (
                torch.FloatTensor(cache["data_t0"]) if "data_t0" in cache else None
            )

            config_dict = cache["config"]
            args = SimpleNamespace(**config_dict)
            args.device = self.device
            self.model = MultiCNet(args)
            self.model.load_state_dict(cache["model_state"])
            self.model.to(self.device)
            return

        config, mapping = self._build_config(metadata, all_tps=all_tps)
        config.device = self.device

        time_col = ObservationColumns.TIMEPOINT.value
        if time_col not in ann_data.obs.columns:
            raise ValueError(f"Missing obs column '{time_col}' in AnnData")

        tp_values = ann_data.obs[time_col].values
        cell_tps = np.array([mapping[v] for v in tp_values], dtype=int)
        n_tps = len(mapping)

        data_log = _ensure_dense_float32(ann_data.X)
        pca = PCA(n_components=config.latent_dim)
        data = pca.fit_transform(data_log)
        self.pca_model = pca

        data_listAllT = [
            torch.FloatTensor(data[np.where(cell_tps == t)[0], :])
            for t in range(0, n_tps)
        ]

        available_tps = sorted(set(cell_tps.tolist()))
        if 0 not in available_tps:
            raise ValueError("scIMF training requires timepoint 0 in training data.")
        config.train_t = [
            t for t in available_tps if t != 0 and t not in config.leaveouts
        ]
        config.Train_ts = [t for t in available_tps if t != 0]

        model = MultiCNet(config)
        model.to(config.device)
        model.zero_grad()
        model.train()

        ot_loss = geomloss.SamplesLoss(
            "sinkhorn",
            p=2,
            blur=config.sinkhorn_blur,
            scaling=config.sinkhorn_scaling,
        )
        torch.save(config.__dict__, config.config_pt)

        optimizer = torch.optim.Adam(list(model.parameters()), lr=config.train_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        pbar = tqdm.tqdm(range(config.train_epochs))

        best_train_loss = np.inf
        with open(config.train_log, "w") as log_handle:
            for epoch in pbar:
                losses_list = []
                losses_energy = []
                config.train_epoch = epoch
                optimizer.zero_grad()

                ts = [0] + config.train_t
                x0 = data_listAllT[0].to(config.device)
                latent_xs_energy_predict = model(ts, x0, batch_size=config.train_batch)

                num_train_t = len(config.train_t)
                for jj, train_t in enumerate(config.train_t):
                    loss_train_t = ot_loss(
                        data_listAllT[int(train_t)].to(config.device),
                        latent_xs_energy_predict[jj + 1][:, 0:-1],
                    )
                    losses_list.append(loss_train_t.item())

                    if (train_t == config.train_t[-1]) and config.use_intLoss:
                        loss_energy = (
                            torch.mean(latent_xs_energy_predict[-1][:, -1])
                        ) / train_t
                        losses_energy.append(loss_energy.item())
                        loss_all = (
                            (loss_train_t * config.lambda_marginal) / num_train_t
                        ) + loss_energy
                    else:
                        loss_all = (loss_train_t * config.lambda_marginal) / num_train_t

                    loss_all.backward(retain_graph=True)

                train_loss = np.mean(losses_list) if losses_list else 0.0
                if config.use_intLoss and losses_energy:
                    train_loss_energy = np.mean(losses_energy)

                if config.train_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.train_clip
                    )
                optimizer.step()
                scheduler.step()

                desc = "[train] {}".format(epoch + 1)
                desc += " {:.6f}".format(train_loss)
                if config.use_intLoss and losses_energy:
                    desc += " {:.6f}".format(train_loss_energy)
                desc += " {:.6f}".format(best_train_loss)
                pbar.set_description(desc)
                log_handle.write(desc + "\n")
                log_handle.flush()

                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "epoch": config.train_epoch + 1,
                        },
                        config.train_pt.format("best"),
                    )

                if (config.train_epoch + 1) % config.save == 0:
                    epoch_ = str(config.train_epoch + 1).rjust(6, "0")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "epoch": config.train_epoch + 1,
                        },
                        config.train_pt.format("epoch_{}".format(epoch_)),
                    )

        config.done_log = os.path.join(config.out_dir, "done.log")
        with open(config.done_log, "w") as log_handle:
            log_handle.write("done\n")

        self.model = model
        self.tp_to_idx = mapping
        self.idx_to_tp = {i: tp for tp, i in mapping.items()}

        data_t0 = data_listAllT[0].cpu().numpy()
        max_cache_cells = metadata.get("cache_n_cells", 2000)
        if data_t0.shape[0] > max_cache_cells:
            rng = np.random.default_rng(int(metadata.get("seed", 0)))
            idx = rng.choice(data_t0.shape[0], size=max_cache_cells, replace=False)
            data_t0_cache = data_t0[idx]
        else:
            data_t0_cache = data_t0

        cache_payload = {
            "tp_to_idx": self.tp_to_idx,
            "pca_model": self.pca_model,
            "model_state": self.model.state_dict(),
            "config": config.__dict__,
            "data_t0": data_t0_cache,
        }
        torch.save(cache_payload, cache_path)

    def _get_data_t0(self, test_ann_data):
        if getattr(self, "data_t0", None) is not None:
            return self.data_t0

        time_col = ObservationColumns.TIMEPOINT.value
        if time_col not in test_ann_data.obs.columns:
            return None

        tp_values = test_ann_data.obs[time_col].values
        if len(self.tp_to_idx) == 0:
            return None

        tp0_label = None
        for tp, idx in self.tp_to_idx.items():
            if idx == 0:
                tp0_label = tp
                break
        if tp0_label is None:
            return None

        mask = tp_values == tp0_label
        if mask.sum() == 0:
            return None

        data_log = _ensure_dense_float32(test_ann_data.X)
        data_pca = self.pca_model.transform(data_log)
        return torch.FloatTensor(data_pca[mask])

    def _generate_outputs(self, test_ann_data):
        if hasattr(self, "_cached_outputs"):
            return self._cached_outputs

        self.model.eval()

        time_col = ObservationColumns.TIMEPOINT.value
        cell_tps = test_ann_data.obs[time_col].values
        unique_tps = _sorted_unique(cell_tps)
        missing = [tp for tp in unique_tps if tp not in self.tp_to_idx]
        if missing:
            raise ValueError(f"Unknown timepoints found in test data: {missing}")
        ordered_unique_tps = sorted(unique_tps, key=lambda x: self.tp_to_idx[x])
        n_tps = len(self.tp_to_idx)

        data_t0 = self._get_data_t0(test_ann_data)
        if data_t0 is None:
            raise ValueError("Unable to resolve timepoint 0 data for scIMF generation.")

        # tp_counts = [int((cell_tps == tp).sum()) for tp in ordered_unique_tps]
        # n_sim_cells = max(tp_counts) if tp_counts else data_t0.shape[0]
        n_sim_cells = int(data_t0.shape[0])

        ts = torch.FloatTensor(list(range(n_tps))).to(self.device)
        with torch.no_grad():
            latent_pred = self.model.predict(ts, data_t0, n_cells=n_sim_cells)
        latent_pred = latent_pred.detach().cpu().numpy()

        rng = np.random.default_rng(0)
        n_features = latent_pred.shape[-1]
        embeds = np.empty((test_ann_data.n_obs, n_features), dtype=latent_pred.dtype)

        for tp in ordered_unique_tps:
            mask = cell_tps == tp
            tp_idx = self.tp_to_idx.get(tp)
            if tp_idx is None:
                raise ValueError(f"Timepoint {tp} not found in generated samples.")
            embeds[mask] = _select_generated(latent_pred, tp_idx, int(mask.sum()), rng)

        next_embeds = np.full(
            (test_ann_data.n_obs, n_features), np.nan, dtype=embeds.dtype
        )
        for i, tp in enumerate(ordered_unique_tps):
            mask = cell_tps == tp
            if i + 1 >= len(ordered_unique_tps):
                continue
            next_tp = ordered_unique_tps[i + 1]
            next_idx = self.tp_to_idx.get(next_tp)
            if next_idx is None:
                continue
            next_embeds[mask] = _select_generated(
                latent_pred, next_idx, int(mask.sum()), rng
            )

        next_gene_expr = np.full(
            (test_ann_data.n_obs, test_ann_data.n_vars),
            np.nan,
            dtype=np.float32,
        )
        for i, tp in enumerate(ordered_unique_tps):
            mask = cell_tps == tp
            if i + 1 >= len(ordered_unique_tps):
                continue
            next_tp = ordered_unique_tps[i + 1]
            next_idx = self.tp_to_idx.get(next_tp)
            if next_idx is None:
                continue
            next_latent = _select_generated(latent_pred, next_idx, int(mask.sum()), rng)
            next_gene_expr[mask] = self.pca_model.inverse_transform(next_latent)

        self._cached_outputs = (embeds, next_embeds, next_gene_expr)
        return self._cached_outputs

    def generate_embedding(self, test_ann_data) -> np.ndarray:
        """
        Generate embeddings for the current timepoint.
        """
        embeds, _, _ = self._generate_outputs(test_ann_data)
        return embeds

    def generate_next_tp_embedding(self, test_ann_data) -> np.ndarray:
        """
        Generate embeddings for the next timepoint.
        """
        _, next_embeds, _ = self._generate_outputs(test_ann_data)
        return next_embeds

    def generate_next_tp_gex(self, test_ann_data) -> np.ndarray:
        """
        Generate gene expression for the next timepoint.
        """
        _, _, next_gene_expr = self._generate_outputs(test_ann_data)
        return next_gene_expr


if __name__ == "__main__":
    main(scIMF)
