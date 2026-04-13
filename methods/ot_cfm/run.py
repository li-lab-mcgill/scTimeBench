"""Notebook-style OT-CFM runner based on the single-cell example.

This implementation intentionally stays simple:
- Uses `get_batch` to sample minibatches across adjacent timepoint pairs.
- Trains an MLP velocity field with OT-CFM loss.
- Predicts next-timepoint gene expression via NeuralODE integration.
- Uses method.metadata for MLP/training hyperparameters.
"""

from pathlib import Path
import sys

import numpy as np
import torch
from scipy.sparse import issparse
from tqdm import tqdm

from scTimeBench.method_utils.method_runner import BaseMethod, main
from scTimeBench.shared.constants import ObservationColumns

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


def label_pseudotimes_global(x_by_time):
    """Compute one joint diffusion pseudotime over all cells and split by timepoint."""
    try:
        import scanpy as sc
    except ImportError as e:
        raise ImportError(
            "scanpy is required for pseudotime_uniform prior. Please install scanpy."
        ) from e

    if len(x_by_time) < 2:
        raise ValueError("Need at least two timepoints to compute pseudotime labels.")

    sizes = [x.shape[0] for x in x_by_time]
    if any(s == 0 for s in sizes):
        raise ValueError("Each timepoint must contain at least one cell.")

    x_joint = np.concatenate(x_by_time, axis=0).astype(np.float32, copy=False)
    adata = sc.AnnData(x_joint)

    n_cells = adata.n_obs
    n_neighbors = min(15, max(1, n_cells - 1))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.diffmap(adata)

    first_tp_count = sizes[0]
    root_pool = np.arange(first_tp_count)
    dc1 = adata.obsm["X_diffmap"][root_pool, 0]
    adata.uns["iroot"] = int(root_pool[np.argmin(dc1)])

    sc.tl.dpt(adata)
    t_joint = adata.obs["dpt_pseudotime"].to_numpy(dtype=np.float32)
    t_joint = np.nan_to_num(t_joint, nan=0.0)

    splits = np.cumsum(sizes[:-1])
    return np.split(t_joint, splits)


def get_batch(
    fm,
    x_by_time,
    pseudotime_by_time,
    batch_size,
    n_times,
    device,
    return_noise=False,
):
    """Construct a minibatch from each adjacent timepoint pair."""
    ts = []
    xts = []
    uts = []
    noises = []

    for t_start in range(n_times - 1):
        x0_np = x_by_time[t_start]
        x1_np = x_by_time[t_start + 1]
        y0_np = pseudotime_by_time[t_start]
        y1_np = pseudotime_by_time[t_start + 1]
        idx0 = np.random.randint(x0_np.shape[0], size=batch_size)
        idx1 = np.random.randint(x1_np.shape[0], size=batch_size)
        x0 = torch.from_numpy(x0_np[idx0]).float().to(device)
        x1 = torch.from_numpy(x1_np[idx1]).float().to(device)
        y0 = torch.from_numpy(y0_np[idx0]).float().to(device)
        y1 = torch.from_numpy(y1_np[idx1]).float().to(device)

        if return_noise:
            t, xt, ut, eps = fm.sample_location_and_conditional_flow(
                x0, x1, y0=y0, y1=y1, return_noise=True
            )
            noises.append(eps)
        else:
            t, xt, ut = fm.sample_location_and_conditional_flow(
                x0, x1, y0=y0, y1=y1, return_noise=False
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

        self.method = str(metadata.get("method", "exact"))
        self.prior_method = str(metadata.get("prior_method", "to_first"))
        self.reg = float(metadata.get("reg", 0.1))

        self._unique_train_tps = []
        self._tp_to_index = {}
        self._x_by_time = []
        self._pseudotime_by_time = []
        self._model = None
        self._node = None

    def train(self, ann_data, all_tps=None):
        time_col = ObservationColumns.TIMEPOINT.value
        train_tps = ann_data.obs[time_col].to_numpy()
        self._unique_train_tps = sorted(np.unique(train_tps))
        self._tp_to_index = {tp: i for i, tp in enumerate(self._unique_train_tps)}
        if len(self._unique_train_tps) < 2:
            raise ValueError("OT-CFM training needs at least 2 train timepoints.")

        train_x = ann_data.X.toarray() if issparse(ann_data.X) else ann_data.X
        train_x = np.asarray(train_x, dtype=np.float32)
        self._x_by_time = [
            train_x[np.where(train_tps == tp)[0], :] for tp in self._unique_train_tps
        ]
        if self.prior_method == "pseudotime_uniform":
            self._pseudotime_by_time = label_pseudotimes_global(self._x_by_time)
        else:
            self._pseudotime_by_time = [
                np.zeros(x.shape[0], dtype=np.float32) for x in self._x_by_time
            ]

        dim = int(train_x.shape[1])
        cache_path = Path(self.config["output_path"]) / "trained_ot_cfm_model.pth"
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
        fm = ExactOptimalTransportConditionalFlowMatcher(
            sigma=self.sigma,
            method=self.method,
            prior_method=self.prior_method,
            reg=self.reg,
        )

        self._model.train()
        for _ in tqdm(range(self.train_steps)):
            optimizer.zero_grad()
            t, xt, ut = get_batch(
                fm,
                self._x_by_time,
                self._pseudotime_by_time,
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

        test_x = (
            test_ann_data.X.toarray() if issparse(test_ann_data.X) else test_ann_data.X
        )
        test_x = np.asarray(test_x, dtype=np.float32)

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

        return out


if __name__ == "__main__":
    main(OTCFM)
