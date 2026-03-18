"""
PI-SDE runner script.

This script trains and evaluates PI-SDE on an AnnData dataset.
It keeps the BaseMethod runner structure used across the project.
"""

import os
import sys
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA

from scTimeBench.method_utils.method_runner import main, BaseMethod
from scTimeBench.shared.constants import ObservationColumns


_PISDE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "PISDE_module"))
if os.path.isdir(_PISDE_PATH) and _PISDE_PATH not in sys.path:
    sys.path.append(_PISDE_PATH)

from src.config_Veres import config as pisde_config, init_config, load_data  # type: ignore
import src.train as pisde_train  # type: ignore
from src.model import ForwardSDE  # type: ignore


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


def _ensure_dense(x):
    try:
        import scipy.sparse as sp  # type: ignore

        if sp.issparse(x):
            return x.toarray()
    except Exception:
        pass
    return np.asarray(x)


def _build_pisde_data(ann_data, unique_tps, tp_to_idx, time_col):
    cell_tps = ann_data.obs[time_col].to_numpy()
    X = _ensure_dense(ann_data.X)

    xp = []
    for tp in unique_tps:
        mask = cell_tps == tp
        data_tp = X[mask]
        xp.append(torch.FloatTensor(data_tp))

    if np.issubdtype(np.asarray(unique_tps).dtype, np.number):
        y = [float(tp) for tp in unique_tps]
    else:
        y = [float(i) for i in range(len(unique_tps))]

    return {
        "xp": xp,
        "y": y,
        "tp_to_idx": tp_to_idx,
        "unique_tps": unique_tps,
    }


def _filter_timepoints_with_data(ann_data, unique_tps, time_col):
    cell_tps = ann_data.obs[time_col].to_numpy()
    counts = {tp: int(np.sum(cell_tps == tp)) for tp in unique_tps}
    filtered = [tp for tp in unique_tps if counts.get(tp, 0) > 0]
    return filtered, counts


def _clamp_train_batch_ratio(train_batch, min_count):
    if min_count <= 0:
        return train_batch
    try:
        train_batch = float(train_batch)
    except Exception:
        return train_batch
    if train_batch <= 0:
        return train_batch
    min_ratio = 1.0 / float(min_count)
    return max(train_batch, min_ratio)


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :]
    return x


def _fill_from_sim(sim_list: List[np.ndarray], tp_idx: np.ndarray) -> np.ndarray:
    if not sim_list:
        raise ValueError("Simulation results are empty.")
    first = _ensure_2d(sim_list[0])
    out = np.full((tp_idx.shape[0], first.shape[1]), np.nan, dtype=np.float32)

    for t in range(len(sim_list)):
        mask = tp_idx == t
        if not np.any(mask):
            continue
        sim_mat = _ensure_2d(sim_list[t])
        positions = np.arange(np.sum(mask)) % sim_mat.shape[0]
        out[mask] = sim_mat[positions]

    return out


def _fill_from_next(sim_list: List[np.ndarray], tp_idx: np.ndarray) -> np.ndarray:
    if not sim_list:
        raise ValueError("Simulation results are empty.")
    first = _ensure_2d(sim_list[0])
    out = np.full((tp_idx.shape[0], first.shape[1]), np.nan, dtype=np.float32)

    for t in range(len(sim_list)):
        mask = tp_idx == t
        if not np.any(mask):
            continue
        next_t = t + 1
        if next_t >= len(sim_list):
            continue
        sim_mat = _ensure_2d(sim_list[next_t])
        positions = np.arange(np.sum(mask)) % sim_mat.shape[0]
        out[mask] = sim_mat[positions]

    return out


def _fit_pca_projection(x: np.ndarray, embedding_dim: int):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D data for PCA fit, got shape {x.shape}")

    n_samples, n_features = x.shape
    max_components = min(int(embedding_dim), n_features, max(1, n_samples - 1))
    if max_components <= 0:
        raise ValueError("Unable to fit PCA: invalid number of components.")

    svd_solver = "arpack" if max_components < min(n_samples, n_features) else "full"
    pca = PCA(n_components=max_components, svd_solver=svd_solver)
    pca.fit(x)
    return pca.components_.astype(np.float32), pca.mean_.astype(np.float32)


def _project_with_pca(
    x: np.ndarray, components: np.ndarray, mean: np.ndarray
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - mean) @ components.T


def _select_checkpoint(config):
    best_path = config.train_pt.format("best")
    if os.path.exists(best_path):
        return best_path
    train_dir = os.path.dirname(config.train_pt)
    candidates = sorted(
        [
            p
            for p in os.listdir(train_dir)
            if p.startswith("train.") and p.endswith(".pt")
        ]
    )
    if not candidates:
        raise FileNotFoundError("No PI-SDE checkpoints found.")
    return os.path.join(train_dir, candidates[-1])


class PISDE(BaseMethod):
    def train(self, ann_data, all_tps: Optional[List] = None):
        """
        Training logic for PI-SDE.
        """
        cache_path = os.path.join(self.config["output_path"], "trained_pisde_model.pt")
        metadata = self.config.get("method", {}).get("metadata", {})

        if os.path.exists(cache_path):
            print("Trained PI-SDE model cache found, loading from file.")
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.data_path = cache["data_path"]
            self.config_dir = cache["config_dir"]
            self.unique_tps = cache["unique_tps"]
            self.tp_to_idx = cache["tp_to_idx"]
            self.embedding_dim = int(metadata.get("embedding_dim", 50))
            self.pca_components = cache.get("pca_components")
            self.pca_mean = cache.get("pca_mean")
            return

        time_col = ObservationColumns.TIMEPOINT.value
        if time_col not in ann_data.obs.columns:
            raise ValueError(f"Missing obs column '{time_col}' in AnnData")

        if not all_tps:
            all_tps = ann_data.obs[time_col].unique().tolist()

        unique_tps_all = _sorted_unique(all_tps)
        train_tps, tp_counts = _filter_timepoints_with_data(
            ann_data, unique_tps_all, time_col
        )
        dropped = [tp for tp in unique_tps_all if tp_counts.get(tp, 0) == 0]
        if dropped:
            print(
                "Training has no cells for timepoints; keeping for generation: "
                + ", ".join(map(str, dropped))
            )
        if len(train_tps) < 2:
            raise ValueError("At least two timepoints are required for training")

        tp_to_idx = {tp: idx for idx, tp in enumerate(unique_tps_all)}

        data_dict = _build_pisde_data(ann_data, unique_tps_all, tp_to_idx, time_col)

        data_path = os.path.join(self.config["output_path"], "pisde_data.pt")
        torch.save({"xp": data_dict["xp"], "y": data_dict["y"]}, data_path)

        args = pisde_config()
        args.data_path = data_path
        args.data = metadata.get("dataset", "AnnData")
        args.out_dir = self.config["output_path"]

        args.train_epochs = int(metadata.get("train_epochs", args.train_epochs))
        args.train_lr = float(metadata.get("train_lr", args.train_lr))
        args.train_lambda = float(metadata.get("train_lambda", args.train_lambda))
        args.train_batch = float(metadata.get("train_batch", args.train_batch))
        args.train_clip = float(metadata.get("train_clip", args.train_clip))
        args.save = int(metadata.get("save", args.save))

        args.sinkhorn_scaling = float(
            metadata.get("sinkhorn_scaling", args.sinkhorn_scaling)
        )
        args.sinkhorn_blur = float(metadata.get("sinkhorn_blur", args.sinkhorn_blur))
        args.ns = float(metadata.get("ns", args.ns))

        args.use_cuda = bool(metadata.get("use_cuda", args.use_cuda))
        args.device = int(metadata.get("device", args.device))
        args.seed = int(metadata.get("seed", args.seed))
        if args.use_cuda and not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU for PI-SDE.")
            args.use_cuda = False

        k_dims = metadata.get("k_dims", args.k_dims)
        if isinstance(k_dims, str):
            k_dims = [int(x) for x in k_dims.split(",") if x.strip() != ""]
        args.k_dims = k_dims

        args.activation = metadata.get("activation", args.activation)
        args.sigma_type = metadata.get("sigma_type", args.sigma_type)
        args.sigma_const = float(metadata.get("sigma_const", args.sigma_const))

        min_count = min(tp_counts[tp] for tp in train_tps) if train_tps else 0
        args.train_batch = _clamp_train_batch_ratio(args.train_batch, min_count)

        n_tps = len(unique_tps_all)
        trainable_idx = sorted({tp_to_idx[tp] for tp in train_tps})
        train_t = metadata.get("train_t", list(range(1, n_tps)))
        args.train_t = sorted(
            [int(t) for t in train_t if 0 <= int(t) < n_tps and int(t) in trainable_idx]
        )
        if not args.train_t:
            raise ValueError("Training timepoint list is empty after filtering.")
        args.start_t = int(metadata.get("start_t", 0))
        if args.start_t < 0 or args.start_t >= n_tps:
            raise ValueError("start_t is out of range after filtering timepoints.")
        if args.start_t not in trainable_idx:
            args.start_t = trainable_idx[0]
        print(
            "PI-SDE timepoint mapping: all=%s, trainable_idx=%s, start_t=%s"
            % (unique_tps_all, trainable_idx, args.start_t)
        )

        leaveouts = metadata.get("leaveouts", None)
        try:
            if leaveouts is not None:
                mapped_leaveouts = _map_leaveouts(leaveouts, tp_to_idx, n_tps)
                config = pisde_train.run_leaveout(
                    args, init_config, leaveouts=mapped_leaveouts
                )
            else:
                config = pisde_train.run(args, init_config)
        except RuntimeError as exc:
            msg = str(exc)
            if args.use_cuda and (
                "CUBLAS_STATUS_NOT_INITIALIZED" in msg or "CUDA error" in msg
            ):
                print("CUDA initialization failed; retrying PI-SDE training on CPU.")
                args.use_cuda = False
                if leaveouts is not None:
                    mapped_leaveouts = _map_leaveouts(leaveouts, tp_to_idx, n_tps)
                    config = pisde_train.run_leaveout(
                        args, init_config, leaveouts=mapped_leaveouts
                    )
                else:
                    config = pisde_train.run(args, init_config)
            else:
                raise

        self.data_path = data_path
        self.config_dir = config.out_dir
        self.unique_tps = unique_tps_all
        self.tp_to_idx = tp_to_idx
        self.embedding_dim = int(metadata.get("embedding_dim", 50))

        train_x = _ensure_dense(ann_data.X).astype(np.float32)
        self.pca_components, self.pca_mean = _fit_pca_projection(
            train_x, self.embedding_dim
        )

        torch.save(
            {
                "data_path": data_path,
                "config_dir": self.config_dir,
                "unique_tps": self.unique_tps,
                "tp_to_idx": self.tp_to_idx,
                "pca_components": self.pca_components,
                "pca_mean": self.pca_mean,
            },
            cache_path,
        )

    def _ensure_pca_projection(self, test_ann_data):
        if (
            getattr(self, "pca_components", None) is not None
            and getattr(self, "pca_mean", None) is not None
        ):
            return

        print(
            "[warn] PI-SDE cache missing PCA projection; fitting PCA on test data for embeddings."
        )
        fallback_dim = int(getattr(self, "embedding_dim", 50))
        x = _ensure_dense(test_ann_data.X).astype(np.float32)
        self.pca_components, self.pca_mean = _fit_pca_projection(x, fallback_dim)

    def _simulate_tp_series(self, test_ann_data):
        if hasattr(self, "_cached_sim_tp"):
            return self._cached_sim_tp, self._cached_tp_idx

        config_pt = os.path.join(self.config_dir, "config.pt")
        config_dict = torch.load(config_pt, weights_only=False)
        config = SimpleNamespace(**config_dict)

        device = torch.device(
            f"cuda:{config.device}"
            if config.use_cuda and torch.cuda.is_available()
            else "cpu"
        )

        model = ForwardSDE(config)
        ckpt_path = _select_checkpoint(config)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        x, y, _ = load_data(config)
        #  n_sim_cells = int(metadata.get("n_sim_cells", 2000))
        n_sim_cells = int(x[config.start_t].shape[0])
        x_0, _ = pisde_train.p_samp(x[config.start_t], n_sim_cells)
        r_0 = torch.zeros(int(n_sim_cells)).unsqueeze(1)
        x_r_0 = torch.cat([x_0, r_0], dim=1).to(device)

        ts = [np.float64(y[0])] + [np.float64(val) for val in y[1:]]
        x_r_s = model(ts, x_r_0)

        sim_tp = [x_r_s[t][:, 0:-1].detach().cpu().numpy() for t in range(len(y))]

        time_col = ObservationColumns.TIMEPOINT.value
        cell_tps = test_ann_data.obs[time_col].to_numpy()
        missing_tps = set(cell_tps.tolist()) - set(self.tp_to_idx.keys())
        if missing_tps:
            raise ValueError(
                "PI-SDE cannot generate for unseen timepoints: "
                + ", ".join(map(str, sorted(missing_tps)))
            )
        tp_idx = np.array([self.tp_to_idx[t] for t in cell_tps], dtype=int)

        self._cached_sim_tp = sim_tp
        self._cached_tp_idx = tp_idx
        return sim_tp, tp_idx

    def generate_embedding(self, test_ann_data) -> np.ndarray:
        """
        Generate embeddings for the current timepoint.
        """
        self._ensure_pca_projection(test_ann_data)
        sim_tp, tp_idx = self._simulate_tp_series(test_ann_data)
        sim_current = _fill_from_sim(sim_tp, tp_idx)
        return _project_with_pca(sim_current, self.pca_components, self.pca_mean)

    def generate_next_tp_embedding(self, test_ann_data) -> np.ndarray:
        """
        Generate embeddings for the next timepoint.
        """
        self._ensure_pca_projection(test_ann_data)
        sim_tp, tp_idx = self._simulate_tp_series(test_ann_data)
        sim_next = _fill_from_next(sim_tp, tp_idx)
        return _project_with_pca(sim_next, self.pca_components, self.pca_mean)

    def generate_next_tp_gex(self, test_ann_data) -> np.ndarray:
        """
        Generate gene expression for the next timepoint.
        """
        sim_tp, tp_idx = self._simulate_tp_series(test_ann_data)
        return _fill_from_next(sim_tp, tp_idx)


if __name__ == "__main__":
    main(PISDE)
