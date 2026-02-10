"""
PRESCIENT runner script.

This script trains and evaluates PRESCIENT on an AnnData dataset
using the BaseModel runner structure used across the project.
"""

import os
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from crispy_fishstick.model_utils.model_runner import main, BaseModel
from crispy_fishstick.shared.constants import ObservationColumns, RequiredOutputColumns


timestamp = datetime.now().strftime("%Y%m%d%H%M%S")


def _sorted_unique(values):
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.number):
        return list(np.sort(np.unique(values)))
    try:
        import natsort  # type: ignore
        return list(natsort.natsorted(np.unique(values)))
    except Exception:
        return list(sorted(np.unique(values).tolist()))


def _import_prescient():
    try:
        from prescient.train.run import run as prescient_train_run  # type: ignore
        from prescient.train.model import AutoGenerator  # type: ignore
        from prescient.simulate import sim as prescient_sim  # type: ignore
        return prescient_train_run, AutoGenerator, prescient_sim
    except Exception as exc:
        raise ImportError(
            "Unable to import PRESCIENT modules from the installed package. "
            "Ensure 'pip install prescient' completed successfully and that the "
            "active environment matches your runner."
            f" Import error: {exc}"
        )


def _prescient_init_config(args: SimpleNamespace) -> SimpleNamespace:
    config = SimpleNamespace(
        seed=args.seed,
        timestamp=timestamp,
        data_path=args.data_path,
        weight=args.weight,
        activation=args.activation,
        layers=args.layers,
        k_dim=args.k_dim,
        pretrain_burnin=50,
        pretrain_sd=0.1,
        pretrain_lr=1e-9,
        pretrain_epochs=args.pretrain_epochs,
        train_dt=args.train_dt,
        train_sd=args.train_sd,
        train_batch_size=args.train_batch,
        ns=2000,
        train_burnin=100,
        train_tau=args.train_tau,
        train_epochs=args.train_epochs,
        train_lr=args.train_lr,
        train_clip=args.train_clip,
        save=args.save,
        sinkhorn_scaling=0.7,
        sinkhorn_blur=0.1,
        out_dir=args.out_dir,
        out_name=os.path.basename(args.out_dir),
        pretrain_pt=os.path.join(args.out_dir, "pretrain.pt"),
        train_pt=os.path.join(args.out_dir, "train.{}.pt"),
        train_log=os.path.join(args.out_dir, "train.log"),
        done_log=os.path.join(args.out_dir, "done.log"),
        config_pt=os.path.join(args.out_dir, "config.pt"),
    )

    config.train_t = []
    config.test_t = []

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    return config


def _prescient_train_init(args: SimpleNamespace):
    data_pt = torch.load(args.data_path, weights_only=False)
    x = data_pt["xp"]
    y = data_pt["y"]
    weight = data_pt["w"]

    a = SimpleNamespace(**args.__dict__)
    if a.weight_name is not None:
        a.weight = a.weight_name

    name = (
        "{weight}-"
        "{activation}_{layers}_{k_dim}-"
        "{train_tau}"
    ).format(**a.__dict__)

    a.out_dir = os.path.join(args.out_dir, name, f"seed_{a.seed}")
    config = _prescient_init_config(a)

    config.x_dim = x[0].shape[-1]
    config.t = y[-1] - y[0]

    start_idx = 0
    if isinstance(y[0], (int, np.integer)):
        start_idx = int(y[0]) if int(y[0]) < len(y) else 0
    config.start_t = y[0]
    config.train_t = y[1:]

    y_start = y[start_idx]
    y_ = [yy for yy in y if yy > y_start]
    w_ = weight[start_idx]
    w = {(y_start, yy): torch.from_numpy(np.exp((yy - y_start) * w_)) for yy in y_}

    return x, y, w, config


def _ensure_dense(x):
    try:
        import scipy.sparse as sp  # type: ignore
        if sp.issparse(x):
            return x.toarray()
    except Exception:
        pass
    return np.asarray(x)


def _prepare_prescient_data(ann_data, time_col: str, k_dim: int, num_neighbors_umap: int):
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore

    X = _ensure_dense(ann_data.X)
    genes = ann_data.var.index.values
    tps_raw = ann_data.obs[time_col].to_numpy()
    unique_tps = _sorted_unique(tps_raw)
    tp_to_idx = {tp: idx for idx, tp in enumerate(unique_tps)}
    tps_idx = np.array([tp_to_idx[v] for v in tps_raw], dtype=int)

    celltype_col = ObservationColumns.CELL_TYPE.value
    if celltype_col in ann_data.obs.columns:
        celltype = ann_data.obs[celltype_col].to_numpy()
    else:
        celltype = np.repeat("NAN", ann_data.shape[0])

    scaler = StandardScaler()
    x = scaler.fit_transform(X)

    max_pcs = min(k_dim, x.shape[1], max(1, x.shape[0] - 1))
    pca = PCA(n_components=max_pcs, svd_solver="arpack")
    xp = pca.fit_transform(x)

    um = None
    try:
        import umap  # type: ignore
        um = umap.UMAP(n_components=2, metric="euclidean", n_neighbors=num_neighbors_umap)
        xu = um.fit_transform(xp)
    except Exception:
        if xp.shape[1] >= 2:
            xu = xp[:, :2]
        else:
            xu = np.zeros((xp.shape[0], 2))

    x_list = [torch.from_numpy(x[tps_idx == i]).float() for i in range(len(unique_tps))]
    xp_list = [torch.from_numpy(xp[tps_idx == i]).float() for i in range(len(unique_tps))]
    xu_list = [torch.from_numpy(xu[tps_idx == i]).float() for i in range(len(unique_tps))]
    w_list = [np.ones(x_list[i].shape[0], dtype=np.float32) for i in range(len(unique_tps))]

    data_pt = {
        "data": X,
        "genes": genes,
        "celltype": celltype,
        "tps": tps_idx,
        "x": x_list,
        "xp": xp_list,
        "xu": xu_list,
        "y": list(range(len(unique_tps))),
        "pca": pca,
        "um": um,
        "w": w_list,
    }

    return data_pt, scaler, pca, um, unique_tps, tp_to_idx


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


class PRESCIENT(BaseModel):
    def train(self, ann_data, all_tps: Optional[List] = None):
        """
        Training logic for PRESCIENT.
        """
        cache_path = os.path.join(self.config["output_path"], "trained_prescient_model.pth")
        metadata = self.config.get("model", {}).get("metadata", {})

        if os.path.exists(cache_path):
            print("Trained PRESCIENT model found, loading from file.")
            cache = torch.load(cache_path, map_location="cpu")
            self.data_path = cache["data_path"]
            self.scaler = cache["scaler"]
            self.pca = cache["pca"]
            self.um = cache.get("um")
            self.unique_tps = cache["unique_tps"]
            self.tp_to_idx = cache["tp_to_idx"]
            self.model_dir = cache.get("model_dir", self.config["output_path"])
            return

        time_col = ObservationColumns.TIMEPOINT.value
        if time_col not in ann_data.obs.columns:
            raise ValueError(f"Missing obs column '{time_col}' in AnnData")

        if not all_tps:
            all_tps = ann_data.obs[time_col].unique().tolist()

        self.unique_tps = _sorted_unique(all_tps)
        if len(self.unique_tps) < 2:
            raise ValueError("At least two timepoints are required for training")

        self.tp_to_idx = {tp: idx for idx, tp in enumerate(self.unique_tps)}

        k_dim = int(metadata.get("k_dim", 50))
        num_neighbors_umap = int(metadata.get("num_neighbors_umap", 10))

        data_pt, scaler, pca, um, unique_tps, tp_to_idx = _prepare_prescient_data(
            ann_data,
            time_col=time_col,
            k_dim=k_dim,
            num_neighbors_umap=num_neighbors_umap,
        )

        self.unique_tps = unique_tps
        self.tp_to_idx = tp_to_idx

        data_path = os.path.join(self.config["output_path"], "prescient_data.pt")
        torch.save(data_pt, data_path, pickle_protocol=4)

        min_cells = min(len(x) for x in data_pt["xp"])
        if min_cells == 0:
            raise ValueError("PRESCIENT training data has an empty timepoint; cannot train.")

        prescient_train_run, _auto_gen, _sim = _import_prescient()

        train_batch = float(metadata.get("train_batch", 0.1))
        min_fraction = 1.0 / float(min_cells)
        if train_batch < min_fraction:
            train_batch = min_fraction

        args = SimpleNamespace(
            data_path=data_path,
            out_dir=self.config["output_path"],
            weight=metadata.get("weight", "weight"),
            weight_name=metadata.get("weight_name", None),
            loss=metadata.get("loss", "euclidean"),
            k_dim=int(metadata.get("k_dim", 500)),
            activation=metadata.get("activation", "softplus"),
            layers=int(metadata.get("layers", 1)),
            pretrain_epochs=int(metadata.get("pretrain_epochs", 500)),
            train_epochs=int(metadata.get("train_epochs", 1)), #TODO set to be 2500
            train_lr=float(metadata.get("train_lr", 0.01)),
            train_dt=float(metadata.get("train_dt", 0.1)),
            train_sd=float(metadata.get("train_sd", 0.5)),
            train_tau=float(metadata.get("train_tau", 1e-6)),
            train_batch=train_batch,
            train_clip=float(metadata.get("train_clip", 0.25)),
            save=int(metadata.get("save", 100)),
            pretrain=bool(metadata.get("pretrain", True)),
            train=bool(metadata.get("train", True)),
            seed=int(metadata.get("seed", 2)),
            gpu=int(metadata.get("gpu", 0)),
        )

        prescient_train_run(args, _prescient_train_init)
        _x, _y, _w, config = _prescient_train_init(args)

        self.data_path = data_path
        self.scaler = scaler
        self.pca = pca
        self.um = um
        self.unique_tps = unique_tps
        self.tp_to_idx = tp_to_idx
        self.model_dir = config.out_dir

        torch.save(
            {
                "data_path": data_path,
                "scaler": scaler,
                "pca": pca,
                "um": um,
                "unique_tps": self.unique_tps,
                "tp_to_idx": self.tp_to_idx,
                "model_dir": self.model_dir,
            },
            cache_path,
            pickle_protocol=4,
        )

    def generate(self, test_ann_data, expected_output_path):
        """
        Generation logic with interpolation.
        Returns an AnnData object containing the generated samples.
        """
        metadata = self.config.get("model", {}).get("metadata", {})
        n_sim_cells = int(metadata.get("n_sim_cells", 2000))

        time_col = ObservationColumns.TIMEPOINT.value
        cell_tps = test_ann_data.obs[time_col].to_numpy()
        tp_idx = np.array([self.tp_to_idx[t] for t in cell_tps], dtype=int)

        _prescient_train_run, AutoGenerator, prescient_sim = _import_prescient()

        data_pt = torch.load(self.data_path, weights_only=False)
        config_path = os.path.join(self.model_dir, "config.pt")
        config = SimpleNamespace(**torch.load(config_path, weights_only=False))

        train_pt = os.path.join(
            self.model_dir,
            "train.{}.pt".format(metadata.get("epoch", "best")),
        )
        if not os.path.exists(train_pt):
            best_pt = os.path.join(self.model_dir, "train.best.pt")
            if os.path.exists(best_pt):
                train_pt = best_pt
            else:
                raise FileNotFoundError(f"PRESCIENT checkpoint not found at {train_pt}")

        gpu_id = int(metadata.get("gpu", 0))
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        model = AutoGenerator(config)
        checkpoint = torch.load(train_pt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        num_steps = int(round((len(self.unique_tps) - 1) / config.train_dt))
        all_sims = prescient_sim.simulate(
            np.concatenate([x.numpy() for x in data_pt["xp"]], axis=0),
            data_pt["tps"],
            data_pt["celltype"],
            data_pt["w"],
            model,
            config,
            num_sims=1,
            num_cells=n_sim_cells,
            num_steps=num_steps,
            device=device,
            tp_subset=None,
            celltype_subset=None,
        )

        sim_steps = all_sims[0]
        step_indices = [int(round(i / config.train_dt)) for i in range(len(self.unique_tps))]
        sim_tp_latent = [sim_steps[min(idx, sim_steps.shape[0] - 1)] for idx in step_indices]
        sim_tp_recon = [
            self.scaler.inverse_transform(self.pca.inverse_transform(each))
            for each in sim_tp_latent
        ]

        final_ann_data = test_ann_data.copy()
        print(f"Now populating: {self.required_outputs}")

        for output in self.required_outputs:
            if output == RequiredOutputColumns.EMBEDDING:
                embeds = _fill_from_sim(sim_tp_latent, tp_idx)
                final_ann_data.obsm[RequiredOutputColumns.EMBEDDING.value] = embeds
            elif output == RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING:
                next_embeds = _fill_from_next(sim_tp_latent, tp_idx)
                final_ann_data.obsm[
                    RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING.value
                ] = next_embeds
            elif output == RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION:
                next_expr = _fill_from_next(sim_tp_recon, tp_idx)
                final_ann_data.obsm[
                    RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION.value
                ] = next_expr

        final_ann_data.write_h5ad(expected_output_path)


if __name__ == "__main__":
    main(PRESCIENT)
