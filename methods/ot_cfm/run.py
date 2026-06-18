"""Notebook-style OT-CFM runner based on the single-cell example.

This implementation intentionally stays simple:
- Uses `get_batch` to sample minibatches across adjacent timepoint pairs.
- Trains an MLP velocity field with OT-CFM loss.
- Predicts next-timepoint gene expression via NeuralODE integration.
- Uses method.metadata for MLP/training hyperparameters.
"""

from pathlib import Path
import pickle
import sys

import numpy as np
import scanpy as sc
import torch
from sklearn.decomposition import PCA
from scipy.sparse import issparse
from tqdm import tqdm

from scTimeBench.method_utils.method_runner import BaseMethod, main
from scTimeBench.shared.constants import ObservationColumns
from scTimeBench.shared.utils import (
    undo_log_normalization,
    log_normalize_to_counts,
)

try:
    from torchcfm.conditional_flow_matching import (
        ExactOptimalTransportConditionalFlowMatcher,
    )
    from torchcfm.models import MLP
    from torchcfm.utils import torch_wrapper
except ImportError:
    _MODULE_PATH = Path(__file__).resolve().parent / "ot_cfm_module"
    if str(_MODULE_PATH) not in sys.path:
        sys.path.insert(0, str(_MODULE_PATH))
    from torchcfm.conditional_flow_matching import (  # type: ignore
        ExactOptimalTransportConditionalFlowMatcher,
    )
    from torchcfm.models import MLP  # type: ignore
    from torchcfm.utils import torch_wrapper  # type: ignore

from torchdyn.core import NeuralODE


def get_batch(fm, x_by_time, batch_size, n_times, device, return_noise=False):
    """Construct a minibatch from each adjacent timepoint pair."""
    ts = []
    xts = []
    uts = []
    noises = []

    for t_start in range(n_times - 1):
        x0_np = x_by_time[t_start]
        x1_np = x_by_time[t_start + 1]
        idx0 = np.random.randint(x0_np.shape[0], size=batch_size)
        idx1 = np.random.randint(x1_np.shape[0], size=batch_size)
        x0 = torch.from_numpy(x0_np[idx0]).float().to(device)
        x1 = torch.from_numpy(x1_np[idx1]).float().to(device)

        if return_noise:
            t, xt, ut, eps = fm.sample_location_and_conditional_flow(
                x0, x1, return_noise=True
            )
            noises.append(eps)
        else:
            t, xt, ut = fm.sample_location_and_conditional_flow(
                x0, x1, return_noise=False
            )

        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)

    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)

    if return_noise:
        return t, xt, ut, torch.cat(noises)
    return t, xt, ut


class OTCFM(BaseMethod):
    def __init__(self, yaml_config):
        super().__init__(yaml_config)
        metadata = self.config["method"].get("metadata", {})

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(metadata.get("batch_size", 256))
        self.sigma = float(metadata.get("sigma", 0.1))
        self.train_steps = int(metadata.get("train_steps", 10000))
        self.learning_rate = float(metadata.get("learning_rate", 1e-4))
        self.mlp_width = int(metadata.get("mlp_width", 64))
        self.ode_solver = str(metadata.get("ode_solver", "dopri5"))
        self.ode_sensitivity = str(metadata.get("ode_sensitivity", "adjoint"))
        self.embedding_space = str(metadata.get("embedding_space", "PCA")).upper()
        if self.embedding_space not in {"GEX", "PCA"}:
            raise ValueError("metadata.embedding_space must be either 'GEX' or 'PCA'.")
        self.pca_components = int(metadata.get("pca_components", 50))

        self._unique_train_tps = []
        self._tp_to_index = {}
        self._x_by_time = []
        self._model = None
        self._pca_model = None
        self._node = None

    def _to_dense_float32(self, X):
        data = X.toarray() if issparse(X) else X
        return np.asarray(data, dtype=np.float32)

    def train(self, ann_data, all_tps=None, train_output_path=None):
        time_col = ObservationColumns.TIMEPOINT.value
        train_tps = ann_data.obs[time_col].to_numpy()
        self._unique_train_tps = sorted(np.unique(train_tps))
        self._tp_to_index = {tp: i for i, tp in enumerate(self._unique_train_tps)}
        if len(self._unique_train_tps) < 2:
            raise ValueError("OT-CFM training needs at least 2 train timepoints.")

        output_dir = Path(train_output_path)
        cache_path = (
            output_dir / f"trained_ot_cfm_model_{self.embedding_space.lower()}.pth"
        )
        pca_cache_path = output_dir / "trained_ot_cfm_pca_model.pkl"

        train_gex = self._to_dense_float32(ann_data.X)
        if self.embedding_space == "PCA":
            if pca_cache_path.exists():
                with open(pca_cache_path, "rb") as f:
                    self._pca_model = pickle.load(f)
            else:
                n_components = min(
                    self.pca_components,
                    train_gex.shape[0],
                    train_gex.shape[1],
                )
                if n_components < 1:
                    raise ValueError("PCA requires at least one component.")
                self._pca_model = PCA(n_components=n_components)
                self._pca_model.fit(train_gex)
                with open(pca_cache_path, "wb") as f:
                    pickle.dump(self._pca_model, f)

            train_x = self._pca_model.transform(train_gex).astype(np.float32)
        else:
            train_x = train_gex

        self._x_by_time = [
            train_x[np.where(train_tps == tp)[0], :] for tp in self._unique_train_tps
        ]

        dim = int(train_x.shape[1])
        cache_path = Path(train_output_path) / "trained_ot_cfm_model.pth"
        self._model = MLP(dim=dim, time_varying=True, w=self.mlp_width).to(self.device)

        if cache_path.exists():
            print("Trained OT-CFM model found, loading from file.")
            state_dict = torch.load(cache_path, map_location=self.device)
            self._model.load_state_dict(state_dict)
            self._model.eval()
            self._node = NeuralODE(
                torch_wrapper(self._model),
                solver=self.ode_solver,
                sensitivity=self.ode_sensitivity,
            )
            return

        optimizer = torch.optim.Adam(self._model.parameters(), self.learning_rate)
        fm = ExactOptimalTransportConditionalFlowMatcher(sigma=self.sigma)

        self._model.train()
        for _ in tqdm(range(self.train_steps)):
            optimizer.zero_grad()
            t, xt, ut = get_batch(
                fm,
                self._x_by_time,
                self.batch_size,
                len(self._x_by_time),
                self.device,
            )
            vt = self._model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()

        self._model.eval()
        self._node = NeuralODE(
            torch_wrapper(self._model),
            solver=self.ode_solver,
            sensitivity=self.ode_sensitivity,
        )
        torch.save(self._model.state_dict(), cache_path)

    def _interpolate_index_from_tp(self, tp):
        """
        Interpolate the index from the timepoint.

        Interpolate what the time indices would be by finding the closest lower timepoint to from_tp
        and the closest higher timepoint to to_tp.
        e.g.: if I have train timepoints of 8.0, 8.8, 9.2, 9.6 => 0, 1, 2
        and I want to calculate the timepoint for 9.0 => 9.4, it would be 1.5 => 2.5

        If we're extrapolating, simply take the last timepoint difference as the scale.
        """
        if len(self._unique_train_tps) < 2:
            raise ValueError("Need at least 2 train timepoints to interpolate index.")

        if tp in self._tp_to_index:
            return float(self._tp_to_index[tp])

        tps = np.asarray(self._unique_train_tps, dtype=np.float64)
        tp = float(tp)

        # Extrapolation on the right uses the final observed timepoint spacing.
        if tp > tps[-1]:
            dt = tps[-1] - tps[-2]
            if dt <= 0:
                raise ValueError("Train timepoints must be strictly increasing.")
            return float((len(tps) - 1) + (tp - tps[-1]) / dt)

        # Extrapolation on the left uses the first observed timepoint spacing.
        if tp < tps[0]:
            dt = tps[1] - tps[0]
            if dt <= 0:
                raise ValueError("Train timepoints must be strictly increasing.")
            return float((tp - tps[0]) / dt)

        # Interpolate between the closest lower and higher train timepoints.
        upper = int(np.searchsorted(tps, tp, side="right"))
        lower = upper - 1
        lower_tp = tps[lower]
        upper_tp = tps[upper]
        span = upper_tp - lower_tp
        if span <= 0:
            raise ValueError("Train timepoints must be strictly increasing.")

        frac = (tp - lower_tp) / span
        return float(lower + frac)

    def _predict_one_step(self, source_x, from_tp, to_tp):
        if self._node is None:
            raise RuntimeError("Model was not trained. Call train() first.")

        from_idx = self._interpolate_index_from_tp(from_tp)
        to_idx = self._interpolate_index_from_tp(to_tp)
        print(
            f"Predicting from {from_tp} (index {from_idx}) to {to_tp} (index {to_idx})."
        )
        print(
            f"Train timepoints: {self._unique_train_tps}, train indices: {[self._tp_to_index[tp] for tp in self._unique_train_tps]}"
        )

        if to_idx <= from_idx:
            raise ValueError(
                f"Target timepoint must be after source timepoint, got {from_tp} -> {to_tp}."
            )

        x0 = torch.from_numpy(np.asarray(source_x, dtype=np.float32)).to(self.device)
        t_span = torch.tensor([float(from_idx), float(to_idx)], device=self.device)
        with torch.no_grad():
            traj = self._node.trajectory(x0, t_span=t_span)
        return traj[-1].detach().cpu().numpy().astype(np.float32)

    def generate_next_tp_gex(self, test_ann_data) -> np.ndarray:
        """Predict next-timepoint gene expression via one-step NeuralODE flow."""
        time_col = ObservationColumns.TIMEPOINT.value
        test_tps = test_ann_data.obs[time_col].to_numpy()
        unique_test_tps = sorted(np.unique(test_tps))

        test_gex = self._to_dense_float32(test_ann_data.X)
        if self.embedding_space == "PCA":
            if self._pca_model is None:
                raise RuntimeError("PCA model not available. Call train() first.")
            test_x = self._pca_model.transform(test_gex).astype(np.float32)
            out = np.full(
                (test_ann_data.n_obs, test_x.shape[1]), np.nan, dtype=np.float32
            )
        else:
            test_x = test_gex
            out = np.full(
                (test_ann_data.n_obs, test_ann_data.n_vars), np.nan, dtype=np.float32
            )

        for tp in unique_test_tps:
            candidate_next_tps = [x for x in unique_test_tps if x > tp]
            if not candidate_next_tps:
                continue
            next_tp = candidate_next_tps[0]

            source_idx = np.where(test_tps == tp)[0]
            source_x = test_x[source_idx]

            out[source_idx] = self._predict_one_step(source_x, tp, next_tp)

        if self.embedding_space == "PCA":
            out_gex = np.full(
                (test_ann_data.n_obs, test_ann_data.n_vars), np.nan, dtype=np.float32
            )
            valid_rows = ~np.isnan(out).any(axis=1)
            if np.any(valid_rows):
                out_gex[valid_rows] = self._pca_model.inverse_transform(out[valid_rows])
            return out_gex.astype(np.float32)

        return out

    def generate_embedding(self, test_ann_data) -> np.ndarray:
        """Generate embeddings for the current timepoint using the configured embedding space.

        If embedding_space is "PCA", returns PCA-transformed embeddings.
        Otherwise returns the gene expression directly.
        """
        test_gex = self._to_dense_float32(test_ann_data.X)

        if self.embedding_space == "PCA":
            if self._pca_model is None:
                raise RuntimeError("PCA model not available. Call train() first.")
            return self._pca_model.transform(test_gex).astype(np.float32)
        else:
            raise NotImplementedError(
                "generate_embedding is only implemented for PCA embedding space."
            )

    def generate_next_tp_embedding(self, test_ann_data) -> np.ndarray:
        """Generate embeddings for the next timepoint.

        Returns embeddings in the configured embedding space (PCA or GEX).
        """
        time_col = ObservationColumns.TIMEPOINT.value
        test_tps = test_ann_data.obs[time_col].to_numpy()
        unique_test_tps = sorted(np.unique(test_tps))

        test_gex = self._to_dense_float32(test_ann_data.X)
        if self.embedding_space == "PCA":
            if self._pca_model is None:
                raise RuntimeError("PCA model not available. Call train() first.")
            test_x = self._pca_model.transform(test_gex).astype(np.float32)
            out = np.full(
                (test_ann_data.n_obs, test_x.shape[1]), np.nan, dtype=np.float32
            )
        else:
            raise NotImplementedError(
                "generate_next_tp_embedding is only implemented for PCA embedding space."
            )

        for tp in unique_test_tps:
            candidate_next_tps = [x for x in unique_test_tps if x > tp]
            if not candidate_next_tps:
                continue
            next_tp = candidate_next_tps[0]

            source_idx = np.where(test_tps == tp)[0]
            source_x = test_x[source_idx]

            out[source_idx] = self._predict_one_step(source_x, tp, next_tp)

        return out.astype(np.float32)

    def generate_zero_to_end_pred_gex(self, first_tp_cells, all_tps) -> sc.AnnData:
        """Project first timepoint cells to every later timepoint and return a stacked AnnData.

        The result keeps the original first timepoint cells unchanged and appends one
        predicted block for each subsequent timepoint in `all_tps`.
        """
        time_col = ObservationColumns.TIMEPOINT.value
        first_tp = all_tps[0]

        first_gex = self._to_dense_float32(first_tp_cells.X)
        if self.embedding_space == "PCA":
            if self._pca_model is None:
                raise RuntimeError("PCA model not available. Call train() first.")
            source_x = self._pca_model.transform(first_gex).astype(np.float32)
        else:
            source_x = first_gex

        pred_ann_data = first_tp_cells.copy()
        for tp in all_tps[1:]:
            predicted_x = self._predict_one_step(source_x, first_tp, tp)
            if self.embedding_space == "PCA":
                predicted_x = self._pca_model.inverse_transform(predicted_x)

            # now let's clip this and re-log normalize
            predicted_x = np.clip(predicted_x, a_min=0, a_max=20)
            # put it in an ann data and then grab it out
            predicted_ann_data = sc.AnnData(predicted_x)
            predicted_ann_data = log_normalize_to_counts(
                undo_log_normalization(predicted_ann_data)
            )
            predicted_x = predicted_ann_data.X

            tp_ann_data = first_tp_cells.copy()
            tp_ann_data.X = np.asarray(predicted_x, dtype=np.float32)
            tp_ann_data.obs[time_col] = tp
            pred_ann_data = sc.concat([pred_ann_data, tp_ann_data], axis=0)
            print(f"Shape of projected timepoint {tp}: {tp_ann_data.shape}")

        return pred_ann_data

    def generate_gex_from_t_to_t1(self, test_ann_data, t, t1):
        """Generate predicted gene expression from timepoint t to t1."""
        # assert that all the tps are of timepoint t
        test_tps = test_ann_data.obs[ObservationColumns.TIMEPOINT.value].to_numpy()
        assert np.all(
            test_tps == t
        ), f"All cells must be from timepoint {t}, but found timepoints: {np.unique(test_tps)}"
        first_gex = self._to_dense_float32(test_ann_data.X)

        if self.embedding_space == "PCA":
            if self._pca_model is None:
                raise RuntimeError("PCA model not available. Call train() first.")
            source_x = self._pca_model.transform(first_gex).astype(np.float32)
        else:
            source_x = first_gex

        t1_cells = self._predict_one_step(source_x, t, t1)
        if self.embedding_space == "PCA":
            t1_cells = self._pca_model.inverse_transform(t1_cells)

        pred_ann_data = test_ann_data.copy()
        pred_ann_data.X = t1_cells
        pred_ann_data.obs[ObservationColumns.TIMEPOINT.value] = t1
        return pred_ann_data


if __name__ == "__main__":
    main(OTCFM)
