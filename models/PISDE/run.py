"""
PI-SDE runner script.

This script trains and evaluates PI-SDE on an AnnData dataset.
It keeps the BaseModel runner structure used across the project.
"""

import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch

from crispy_fishstick.model_utils.model_runner import main, BaseModel
from crispy_fishstick.shared.constants import ObservationColumns, RequiredOutputColumns


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


def _select_checkpoint(config):
    best_path = config.train_pt.format("best")
    if os.path.exists(best_path):
        return best_path
    train_dir = os.path.dirname(config.train_pt)
    candidates = sorted(
        [p for p in os.listdir(train_dir) if p.startswith("train.") and p.endswith(".pt")]
    )
    if not candidates:
        raise FileNotFoundError("No PI-SDE checkpoints found.")
    return os.path.join(train_dir, candidates[-1])


class PISDE(BaseModel):
    def train(self, ann_data, all_tps: Optional[List] = None):
        """
        Training logic for PI-SDE.
        """
        cache_path = os.path.join(self.config["output_path"], "trained_pisde_model.pt")
        metadata = self.config.get("model", {}).get("metadata", {})

        if os.path.exists(cache_path):
            print("Trained PI-SDE model cache found, loading from file.")
            cache = torch.load(cache_path, map_location="cpu",weights_only=False)
            self.data_path = cache["data_path"]
            self.config_dir = cache["config_dir"]
            self.unique_tps = cache["unique_tps"]
            self.tp_to_idx = cache["tp_to_idx"]
            return

        time_col = ObservationColumns.TIMEPOINT.value
        if time_col not in ann_data.obs.columns:
            raise ValueError(f"Missing obs column '{time_col}' in AnnData")

        if not all_tps:
            all_tps = ann_data.obs[time_col].unique().tolist()

        unique_tps = _sorted_unique(all_tps)
        unique_tps, tp_counts = _filter_timepoints_with_data(ann_data, unique_tps, time_col)
        dropped = [tp for tp, count in tp_counts.items() if count == 0]
        if dropped:
            print(
                "Dropping timepoints with no training cells for PI-SDE: "
                + ", ".join(map(str, dropped))
            )
        if len(unique_tps) < 2:
            raise ValueError("At least two timepoints are required for training")

        tp_to_idx = {tp: idx for idx, tp in enumerate(unique_tps)}

        data_dict = _build_pisde_data(
            ann_data, unique_tps, tp_to_idx, time_col
        )

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

        args.sinkhorn_scaling = float(metadata.get("sinkhorn_scaling", args.sinkhorn_scaling))
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

        min_count = min(tp_counts[tp] for tp in unique_tps) if unique_tps else 0
        args.train_batch = _clamp_train_batch_ratio(args.train_batch, min_count)

        n_tps = len(unique_tps)
        train_t = metadata.get("train_t", list(range(1, n_tps)))
        args.train_t = sorted([int(t) for t in train_t if 0 <= int(t) < n_tps])
        if not args.train_t:
            raise ValueError("Training timepoint list is empty after filtering.")
        args.start_t = int(metadata.get("start_t", 0))
        if args.start_t < 0 or args.start_t >= n_tps:
            raise ValueError("start_t is out of range after filtering timepoints.")

        leaveouts = metadata.get("leaveouts", None)
        try:
            if leaveouts is not None:
                mapped_leaveouts = _map_leaveouts(leaveouts, tp_to_idx, n_tps)
                config = pisde_train.run_leaveout(args, init_config, leaveouts=mapped_leaveouts)
            else:
                config = pisde_train.run(args, init_config)
        except RuntimeError as exc:
            msg = str(exc)
            if args.use_cuda and ("CUBLAS_STATUS_NOT_INITIALIZED" in msg or "CUDA error" in msg):
                print("CUDA initialization failed; retrying PI-SDE training on CPU.")
                args.use_cuda = False
                if leaveouts is not None:
                    mapped_leaveouts = _map_leaveouts(leaveouts, tp_to_idx, n_tps)
                    config = pisde_train.run_leaveout(args, init_config, leaveouts=mapped_leaveouts)
                else:
                    config = pisde_train.run(args, init_config)
            else:
                raise

        self.data_path = data_path
        self.config_dir = config.out_dir
        self.unique_tps = unique_tps
        self.tp_to_idx = tp_to_idx

        torch.save(
            {
                "data_path": data_path,
                "config_dir": self.config_dir,
                "unique_tps": self.unique_tps,
                "tp_to_idx": self.tp_to_idx,
            },
            cache_path,
        )

    def generate(self, test_ann_data, expected_output_path):
        """
        Generation logic with interpolation.
        Returns an AnnData object containing the generated samples.
        """
        metadata = self.config.get("model", {}).get("metadata", {})
        n_sim_cells = int(metadata.get("n_sim_cells", 2000))

        config_pt = os.path.join(self.config_dir, "config.pt")
        config_dict = torch.load(config_pt,weights_only=False)
        config = SimpleNamespace(**config_dict)

        device = torch.device(f"cuda:{config.device}" if config.use_cuda and torch.cuda.is_available() else "cpu")

        model = ForwardSDE(config)
        ckpt_path = _select_checkpoint(config)
        checkpoint = torch.load(ckpt_path, map_location="cpu",weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        x, y, _ = load_data(config)
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

        final_ann_data = test_ann_data.copy()
        print(f"Now populating: {self.required_outputs}")

        for output in self.required_outputs:
            if output == RequiredOutputColumns.EMBEDDING:
                embeds = _fill_from_sim(sim_tp, tp_idx)
                final_ann_data.obsm[RequiredOutputColumns.EMBEDDING.value] = embeds
            elif output == RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING:
                next_embeds = _fill_from_next(sim_tp, tp_idx)
                final_ann_data.obsm[RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING.value] = next_embeds
            elif output == RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION:
                next_expr = _fill_from_next(sim_tp, tp_idx)
                final_ann_data.obsm[RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION.value] = next_expr

        final_ann_data.write_h5ad(expected_output_path)


if __name__ == "__main__":
    main(PISDE)
