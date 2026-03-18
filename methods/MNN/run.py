"""
Cell-MNN runner script.

This script trains and evaluates Cell-MNN on an AnnData dataset.
It keeps the BaseMethod runner structure used across the project.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

from scTimeBench.method_utils.method_runner import main, BaseMethod
from scTimeBench.shared.constants import ObservationColumns


_CELL_MNN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "MNN_module"))
if os.path.isdir(_CELL_MNN_PATH) and _CELL_MNN_PATH not in sys.path:
    sys.path.append(_CELL_MNN_PATH)

from data.snapshot_sampler import build_snapshot_sampler_with_pca  # type: ignore
from fit.cell_mnn import train_cell_mnn  # type: ignore
from model.cell_mnn import CellMNN  # type: ignore


def _ensure_dense_float32(x: np.ndarray) -> np.ndarray:
    try:
        import scipy.sparse as sp  # type: ignore

        if sp.issparse(x):
            x = x.toarray()
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)


def _sorted_unique(values: List) -> List:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.number):
        return list(np.sort(np.unique(values)))
    try:
        import natsort  # type: ignore

        return list(natsort.natsorted(np.unique(values)))
    except Exception:
        return list(sorted(np.unique(values).tolist()))


def _build_time_mapping(all_tps: List) -> Tuple[Dict, List]:
    ordered = _sorted_unique(all_tps)
    if np.issubdtype(np.asarray(ordered).dtype, np.number):
        mapping = {tp: float(tp) for tp in ordered}
    else:
        mapping = {tp: float(idx) for idx, tp in enumerate(ordered)}
    return mapping, ordered


def _build_time_to_data(ann_data, time_mapping: Dict) -> Dict[float, torch.Tensor]:
    time_col = ObservationColumns.TIMEPOINT.value
    if time_col not in ann_data.obs.columns:
        raise ValueError(f"Missing obs column '{time_col}' in AnnData")

    data = _ensure_dense_float32(ann_data.X)
    times = ann_data.obs[time_col].values
    time_to_indices: Dict[float, List[int]] = {}
    for idx, t in enumerate(times):
        if t not in time_mapping:
            raise ValueError(f"Timepoint '{t}' not found in training mapping")
        mapped = float(time_mapping[t])
        time_to_indices.setdefault(mapped, []).append(idx)

    time_to_data = {
        t: torch.tensor(data[idxs], dtype=torch.float32)
        for t, idxs in time_to_indices.items()
    }
    return time_to_data


def _pca_transform(
    x: np.ndarray, components: np.ndarray, mean: np.ndarray
) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    transformed = (x - mean) @ components
    return np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)


def _pca_inverse(z: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    inverse = z @ components.T + mean
    return np.nan_to_num(inverse, nan=0.0, posinf=0.0, neginf=0.0)


def _model_state_is_finite(state_dict: Dict[str, torch.Tensor]) -> bool:
    for tensor in state_dict.values():
        if not torch.isfinite(tensor).all():
            return False
    return True


def _resolve_device(metadata: Dict) -> torch.device:
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


class CellMNNRunner(BaseMethod):
    def train(self, ann_data, all_tps=None):
        cache_path = os.path.join(
            self.config["output_path"], "trained_CellMNN_model.pth"
        )
        metadata = self.config.get("method", {}).get("metadata", {})

        self.device = _resolve_device(metadata)

        if os.path.exists(cache_path):
            print("Trained Cell-MNN model found, loading from file.")
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.time_mapping = cache["time_mapping"]
            self.ordered_time_labels = cache["ordered_time_labels"]
            self.latent_dim = int(cache["latent_dim"])

            if not _model_state_is_finite(cache["model_state"]):
                print(
                    "[warn] Cached Cell-MNN weights contain non-finite values; "
                    "ignoring cache and retraining."
                )
            else:
                pca_components = torch.tensor(
                    cache["pca_components"], dtype=torch.float32
                )
                pca_mean = torch.tensor(cache["pca_mean"], dtype=torch.float32)

                self.model = CellMNN(
                    latent_dim=self.latent_dim,
                    pca_components=pca_components,
                    pca_mean=pca_mean,
                )
                self.model.load_state_dict(cache["model_state"])
                self.model.to(self.device)
                return

        if all_tps is None:
            raise ValueError("all_tps is required to build timepoint mapping.")

        self.time_mapping, self.ordered_time_labels = _build_time_mapping(all_tps)

        latent_dim = int(metadata.get("latent_dim", 5))
        if latent_dim != 5:
            print("[warn] Cell-MNN requires latent_dim=5; overriding.")
            latent_dim = 5

        time_to_data = _build_time_to_data(ann_data, self.time_mapping)

        sampler, pca = build_snapshot_sampler_with_pca(
            time_to_data,
            latent_dim=latent_dim,
            dtype=torch.float32,
            device=self.device,
        )

        self.model = CellMNN(
            latent_dim=latent_dim,
            pca_components=pca.components_,
            pca_mean=pca.mean_,
        ).to(self.device)

        timepoints = sorted(time_to_data.keys())
        if len(timepoints) < 2:
            raise ValueError("At least two timepoints are required for training.")

        epochs = int(metadata.get("epochs", 100))
        steps_per_epoch = int(metadata.get("steps_per_epoch", 100))
        batch_size = int(metadata.get("batch_size", 256))
        gamma = float(metadata.get("gamma", 0.1))
        kinetic_weight = float(metadata.get("kinetic_weight", 1.0))
        inverse_weight = float(metadata.get("inverse_weight", 1.0))
        learning_rate = float(metadata.get("learning_rate", 2e-4))
        weight_decay = float(metadata.get("weight_decay", 1e-5))

        train_cell_mnn(
            self.model,
            sampler,
            timepoints,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            gamma=gamma,
            kinetic_weight=kinetic_weight,
            inverse_weight=inverse_weight,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=self.device,
        )

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "pca_components": pca.components_.cpu().numpy(),
                "pca_mean": pca.mean_.cpu().numpy(),
                "time_mapping": self.time_mapping,
                "ordered_time_labels": self.ordered_time_labels,
                "latent_dim": latent_dim,
            },
            cache_path,
        )

    def _generate_all_outputs(self, test_ann_data):
        if hasattr(self, "_cached_generation"):
            return self._cached_generation

        if not hasattr(self, "model"):
            raise ValueError("Model not trained or loaded; cannot generate outputs.")

        self.model.eval()

        data = _ensure_dense_float32(test_ann_data.X)
        time_col = ObservationColumns.TIMEPOINT.value
        cell_tps = test_ann_data.obs[time_col].to_numpy()

        ordered = list(self.ordered_time_labels)
        label_to_pos = {label: idx for idx, label in enumerate(ordered)}

        pca_components = self.model.pca_components.detach().cpu().numpy()
        pca_mean = self.model.pca_mean.detach().cpu().numpy()

        embeds = _pca_transform(data, pca_components, pca_mean)

        next_embeds = np.full_like(embeds, np.nan, dtype=np.float32)
        next_expr = np.full_like(data, np.nan, dtype=np.float32)
        invalid_next = 0

        with torch.no_grad():
            for idx, (cell, tp_label) in enumerate(zip(embeds, cell_tps)):
                if tp_label not in label_to_pos:
                    invalid_next += 1
                    continue
                pos = label_to_pos[tp_label]
                if pos + 1 >= len(ordered):
                    invalid_next += 1
                    continue

                next_label = ordered[pos + 1]
                t_val = float(self.time_mapping[tp_label])
                next_t_val = float(self.time_mapping[next_label])
                delta_t = next_t_val - t_val

                z = torch.tensor(cell[None, :], dtype=torch.float32, device=self.device)
                t_tensor = torch.tensor(
                    [[t_val]], dtype=torch.float32, device=self.device
                )
                dt_tensor = torch.tensor(
                    [[delta_t]], dtype=torch.float32, device=self.device
                )

                z_pred, _p, _lam = self.model(z, t_tensor, dt_tensor)
                z_pred_np = z_pred.cpu().numpy()[0]
                next_embeds[idx] = z_pred_np
                next_expr[idx] = _pca_inverse(
                    z_pred_np[None, :], pca_components, pca_mean
                )[0]

        fill_mask = ~np.isfinite(next_embeds).all(axis=1)
        if fill_mask.any():
            print(
                f"[warn] CellMNN produced non-finite next-timepoint values for {fill_mask.sum()} cells; "
                "falling back to current embeddings/expressions for those rows."
            )
            next_embeds[fill_mask] = embeds[fill_mask]
            next_expr[fill_mask] = data[fill_mask]

        if invalid_next > 0:
            print(
                f"[warn] CellMNN could not infer next timepoint for {invalid_next} cells; "
                "filling with current embeddings to avoid NaNs."
            )
            tp_fill_mask = np.isnan(next_embeds).all(axis=1)
            next_embeds[tp_fill_mask] = embeds[tp_fill_mask]
            next_expr[tp_fill_mask] = data[tp_fill_mask]

        self._cached_generation = (embeds, next_embeds, next_expr)
        return self._cached_generation

    def generate_embedding(self, test_ann_data) -> np.ndarray:
        embeds, _, _ = self._generate_all_outputs(test_ann_data)
        return embeds

    def generate_next_tp_embedding(self, test_ann_data) -> np.ndarray:
        _, next_embeds, _ = self._generate_all_outputs(test_ann_data)
        return next_embeds

    def generate_next_tp_gex(self, test_ann_data) -> np.ndarray:
        _, _, next_expr = self._generate_all_outputs(test_ann_data)
        return next_expr


if __name__ == "__main__":
    main(CellMNNRunner)
