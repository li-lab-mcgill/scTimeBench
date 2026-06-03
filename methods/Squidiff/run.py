"""
Squidiff runner script.

This script trains and evaluates Squidiff on an AnnData dataset.
It keeps the BaseMethod runner structure used across the project.
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.distributed as dist
import torch

from scTimeBench.method_utils.method_runner import main, BaseMethod
from scTimeBench.shared.constants import ObservationColumns


def _single_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        return torch.device("cuda:0")
    return torch.device("cpu")


def _sorted_unique(values: List) -> List:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.number):
        return list(np.sort(np.unique(values)))
    try:
        import natsort  # type: ignore

        return list(natsort.natsorted(np.unique(values)))
    except Exception:
        return list(sorted(np.unique(values).tolist()))


def _ensure_dense(x) -> np.ndarray:
    try:
        import scipy.sparse as sp  # type: ignore

        if sp.issparse(x):
            return x.toarray()
    except Exception:
        pass
    return np.asarray(x)


def _setup_single_process_dist() -> None:
    if not dist.is_available() or dist.is_initialized():
        return

    store_dir = tempfile.mkdtemp(prefix="squidiff-dist-")
    store_path = os.path.join(store_dir, "init")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_path}",
        rank=0,
        world_size=1,
    )


def _prepare_grouped_adata(ann_data, all_tps: Optional[List]) -> Tuple:
    time_col = ObservationColumns.TIMEPOINT.value
    if time_col not in ann_data.obs.columns:
        raise ValueError(f"Missing obs column '{time_col}' in AnnData")

    if not all_tps:
        all_tps = ann_data.obs[time_col].unique().tolist()
    unique_tps = _sorted_unique(all_tps)
    if len(unique_tps) < 2:
        raise ValueError("At least two timepoints are required for Squidiff training")

    tp_to_idx = {tp: idx for idx, tp in enumerate(unique_tps)}
    grouped = ann_data.copy()
    grouped.obs["Group"] = [tp_to_idx[t] for t in grouped.obs[time_col].to_numpy()]
    return grouped, unique_tps, tp_to_idx


def _build_args(metadata: Dict, data_path: str, output_path: str, n_genes: int) -> Dict:
    try:
        from Squidiff.script_util import model_and_diffusion_defaults  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Unable to import Squidiff from the installed package. "
            "Ensure 'pip install Squidiff' completed successfully and that the "
            "active environment matches your runner."
            f" Import error: {exc}"
        )

    args = dict(model_and_diffusion_defaults())

    def _as_bool(value, default=False):
        if value is None:
            return default
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _as_int(value, default):
        try:
            return int(value)
        except Exception:
            return default

    def _as_float(value, default):
        try:
            return float(value)
        except Exception:
            return default

    args.update(
        {
            "data_path": data_path,
            "control_data_path": metadata.get("control_data_path", ""),
            "schedule_sampler": metadata.get("schedule_sampler", "uniform"),
            "lr": _as_float(metadata.get("lr", 1e-4), 1e-4),
            "weight_decay": _as_float(metadata.get("weight_decay", 0.0), 0.0),
            "lr_anneal_steps": _as_int(metadata.get("lr_anneal_steps", 100000), 100000),
            "batch_size": _as_int(metadata.get("batch_size", 64), 64),
            "microbatch": _as_int(metadata.get("microbatch", -1), -1),
            "ema_rate": str(metadata.get("ema_rate", "0.9999")),
            "log_interval": _as_int(metadata.get("log_interval", 10000), 10000),
            "save_interval": _as_int(metadata.get("save_interval", 10000), 10000),
            "resume_checkpoint": metadata.get(
                "resume_checkpoint", os.path.join(output_path, "squidiff_checkpoints")
            ),
            "use_fp16": _as_bool(metadata.get("use_fp16", False), False),
            "fp16_scale_growth": _as_float(
                metadata.get("fp16_scale_growth", 1e-3), 1e-3
            ),
            "gene_size": _as_int(metadata.get("gene_size", n_genes), n_genes),
            "output_dim": _as_int(metadata.get("output_dim", n_genes), n_genes),
            "num_layers": _as_int(metadata.get("num_layers", 3), 3),
            "class_cond": _as_bool(metadata.get("class_cond", False), False),
            "use_encoder": _as_bool(metadata.get("use_encoder", True), True),
            "diffusion_steps": _as_int(metadata.get("diffusion_steps", 1000), 1000),
            "logger_path": metadata.get(
                "logger_path", os.path.join(output_path, "squidiff_logs")
            ),
            "use_drug_structure": _as_bool(
                metadata.get("use_drug_structure", False), False
            ),
            "comb_num": _as_int(metadata.get("comb_num", 1), 1),
            "use_ddim": _as_bool(metadata.get("use_ddim", True), True),
            "drug_dimension": _as_int(metadata.get("drug_dimension", 1024), 1024),
        }
    )

    os.makedirs(args["logger_path"], exist_ok=True)
    os.makedirs(args["resume_checkpoint"], exist_ok=True)
    return args


def _run_training(args: Dict) -> None:
    from Squidiff import dist_util, logger  # type: ignore

    # from Squidiff import dist_util, logger
    from Squidiff.scrna_datasets import prepared_data  # type: ignore
    from Squidiff.resample import create_named_schedule_sampler  # type: ignore
    from Squidiff.script_util import (  # type: ignore
        create_model_and_diffusion,
        args_to_dict,
        model_and_diffusion_defaults,
    )
    from Squidiff.train_util import TrainLoop, plot_loss  # type: ignore

    _setup_single_process_dist()
    device = _single_device()
    logger.configure(dir=args["logger_path"])

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)
    # model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(
        args["schedule_sampler"], diffusion
    )

    try:
        data = prepared_data(
            data_dir=args["data_path"],
            control_data_dir=args.get("control_data_path", ""),
            batch_size=args["batch_size"],
            use_drug_structure=args["use_drug_structure"],
            comb_num=args["comb_num"],
        )
    except TypeError as exc:
        if "control_data_dir" in str(exc):
            data = prepared_data(
                data_dir=args["data_path"],
                batch_size=args["batch_size"],
                use_drug_structure=args["use_drug_structure"],
                comb_num=args["comb_num"],
            )
        else:
            raise

    train_loop = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args["batch_size"],
        microbatch=args["microbatch"],
        lr=args["lr"],
        ema_rate=args["ema_rate"],
        log_interval=args["log_interval"],
        save_interval=args["save_interval"],
        resume_checkpoint=args["resume_checkpoint"],
        use_fp16=args["use_fp16"],
        fp16_scale_growth=args["fp16_scale_growth"],
        schedule_sampler=schedule_sampler,
        weight_decay=args["weight_decay"],
        lr_anneal_steps=args["lr_anneal_steps"],
        use_drug_structure=args["use_drug_structure"],
        comb_num=args["comb_num"],
    )

    try:
        train_loop.run_loop()
        plot_loss(train_loop.loss_list, args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _load_model(args: Dict, model_path: str):
    # from Squidiff import dist_util
    from Squidiff.script_util import (  # type: ignore
        create_model_and_diffusion,
        args_to_dict,
        model_and_diffusion_defaults,
    )

    def _safe_load_state_dict(path: str):
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            if "state_dict" in state:
                return state["state_dict"]
            if "model" in state:
                return state["model"]
        return state

    # world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # use_dist = world_size > 1

    # if use_dist:
    #     dist_util.setup_dist()

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # if use_dist:
    #     try:
    #         state_dict = dist_util.load_state_dict(model_path, map_location="cpu")
    #     except RuntimeError as exc:
    #         if "No backend type associated with device type cpu" in str(exc):
    #             state_dict = _safe_load_state_dict(model_path)
    #         else:
    #             raise
    # else:
    #     state_dict = _safe_load_state_dict(model_path)
    state_dict = _safe_load_state_dict(model_path)
    model.load_state_dict(state_dict)
    device = _single_device()
    model.to(device)
    model.eval()
    return model, diffusion, device

    # model.to(dist_util.dev())
    # model.eval()
    # return model, diffusion, dist_util.dev()


def _encode_latent(
    model,
    x: np.ndarray,
    device,
    batch_size: int,
    use_encoder: bool,
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not use_encoder or not hasattr(model, "encoder"):
        return x.astype(np.float32)

    requires_labels = getattr(model, "num_classes", None) is not None
    if requires_labels and labels is None:
        raise ValueError("Squidiff encoder requires labels when class_cond is enabled.")

    def _encoder_forward(
        batch_tensor: torch.Tensor, label_tensor: Optional[torch.Tensor] = None
    ):
        encoder = model.encoder
        try:
            import inspect

            sig = inspect.signature(encoder.forward)
            kwargs = {"drug_dose": None, "control_feature": None}
            if label_tensor is not None and "label" in sig.parameters:
                kwargs["label"] = label_tensor
            filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return encoder(batch_tensor, **filtered)
        except Exception:
            if label_tensor is not None:
                return encoder(batch_tensor, label=label_tensor)
            return encoder(batch_tensor)

    embeddings = []
    label_values = None if labels is None else np.asarray(labels)
    for start in range(0, x.shape[0], batch_size):
        batch = torch.tensor(
            x[start : start + batch_size], dtype=torch.float32, device=device
        )
        label_batch = None
        if requires_labels and label_values is not None:
            label_batch = torch.tensor(
                label_values[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            if label_batch.ndim == 1:
                label_batch = label_batch[:, None]
        with torch.no_grad():
            z_sem = _encoder_forward(batch, label_batch)
        embeddings.append(z_sem.detach().cpu().numpy())
    return np.vstack(embeddings)


def _sample_around_point(
    point: np.ndarray, num_samples: int, scale: float
) -> np.ndarray:
    if num_samples <= 0:
        return np.zeros((0, point.shape[0]), dtype=np.float32)
    noise = np.random.randn(num_samples, point.shape[0]).astype(np.float32)
    return point.astype(np.float32) + scale * noise


def _compute_latent_means(
    model,
    ann_data,
    device,
    batch_size: int,
    use_encoder: bool,
    labels: Optional[np.ndarray] = None,
) -> Dict:
    if not use_encoder:
        return {}

    data = _ensure_dense(ann_data.X).astype(np.float32)
    z_sem = _encode_latent(model, data, device, batch_size, use_encoder, labels=labels)
    time_col = ObservationColumns.TIMEPOINT.value
    tps = ann_data.obs[time_col].to_numpy()
    means: Dict = {}
    for tp in _sorted_unique(tps):
        mask = tps == tp
        if np.any(mask):
            means[tp] = z_sem[mask].mean(axis=0)
    return means


def _compute_global_direction(
    latent_means: Dict, ordered_tps: List
) -> Tuple[np.ndarray, Optional[object], Optional[object]]:
    available = [tp for tp in ordered_tps if tp in latent_means]
    if len(available) < 2:
        return np.array([]), None, None
    start_tp = available[0]
    end_tp = available[-1]
    direction = latent_means[end_tp] - latent_means[start_tp]
    return direction, start_tp, end_tp


def _sample_from_latent(
    model,
    diffusion,
    z_sem: np.ndarray,
    device,
    gene_size: int,
    use_ddim: bool,
    batch_size: int,
) -> np.ndarray:
    sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop
    outputs = []
    for start in range(0, z_sem.shape[0], batch_size):
        batch = torch.tensor(
            z_sem[start : start + batch_size], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            pred = sample_fn(
                model,
                shape=(batch.shape[0], gene_size),
                model_kwargs={"z_mod": batch},
                noise=None,
            )
        outputs.append(pred.detach().cpu().numpy())
    return np.vstack(outputs)


class Squidiff(BaseMethod):
    def train(self, ann_data, all_tps: Optional[List] = None, train_output_path=None):
        """
        Training logic for Squidiff.
        """
        cache_path = os.path.join(train_output_path, "trained_squidiff_model.pt")
        metadata = self.config.get("method", {}).get("metadata", {})

        if os.path.exists(cache_path):
            print("Trained Squidiff model found, loading from file.")
            try:
                cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            except TypeError:
                cache = torch.load(cache_path, map_location="cpu")
            self.model_path = cache["model_path"]
            self.args = cache["args"]
            self.tp_to_idx = cache.get("tp_to_idx", {})
            self.unique_tps = cache.get("unique_tps", [])
            self.latent_direction = cache.get("latent_direction")
            self.latent_start_tp = cache.get("latent_start_tp")
            self.latent_end_tp = cache.get("latent_end_tp")
            self.class_cond = bool(cache.get("class_cond", False))
            return

        grouped_adata, unique_tps, tp_to_idx = _prepare_grouped_adata(ann_data, all_tps)
        self.unique_tps = unique_tps
        self.tp_to_idx = tp_to_idx

        train_data_path = os.path.join(train_output_path, "squidiff_train.h5ad")
        grouped_adata.write_h5ad(train_data_path)

        n_genes = grouped_adata.X.shape[1]
        args = _build_args(metadata, train_data_path, train_output_path, n_genes)

        _run_training(args)

        model_path = os.path.join(args["resume_checkpoint"], "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Squidiff training did not produce a model.pt checkpoint."
            )

        self.model_path = model_path
        self.args = args
        self.class_cond = bool(args.get("class_cond", False))

        self.latent_direction = None
        self.latent_start_tp = None
        self.latent_end_tp = None

        use_encoder = bool(args.get("use_encoder", True))
        batch_size = int(args.get("batch_size", 64))
        if use_encoder:
            model, _, device = _load_model(args, self.model_path)
            latent_means = _compute_latent_means(
                model,
                grouped_adata,
                device,
                batch_size,
                use_encoder,
                labels=grouped_adata.obs["Group"].to_numpy(dtype=np.float32).reshape(-1, 1),
            )
            direction, start_tp, end_tp = _compute_global_direction(
                latent_means, unique_tps
            )
            if direction.size > 0:
                self.latent_direction = direction
                self.latent_start_tp = start_tp
                self.latent_end_tp = end_tp

        torch.save(
            {
                "model_path": self.model_path,
                "args": self.args,
                "tp_to_idx": self.tp_to_idx,
                "unique_tps": self.unique_tps,
                "latent_direction": self.latent_direction,
                "latent_start_tp": self.latent_start_tp,
                "latent_end_tp": self.latent_end_tp,
                "class_cond": self.class_cond,
            },
            cache_path,
        )

    def _generate_outputs(self, test_ann_data):
        if hasattr(self, "_cached_outputs"):
            return self._cached_outputs

        if not hasattr(self, "model_path"):
            raise ValueError("Model not trained or loaded; cannot generate outputs.")

        model, diffusion, device = _load_model(self.args, self.model_path)

        data = _ensure_dense(test_ann_data.X).astype(np.float32)
        batch_size = int(self.args.get("batch_size", 64))
        gene_size = int(self.args.get("gene_size", data.shape[1]))
        use_ddim = bool(self.args.get("use_ddim", True))
        use_encoder = bool(self.args.get("use_encoder", True))

        if not use_encoder:
            raise ValueError(
                "Squidiff temporal projection follows the published encoder-driven sampling path and requires use_encoder=True."
            )

        if getattr(self, "class_cond", False):
            raise ValueError(
                "Squidiff temporal inference in this runner follows the published reproduction scripts, which do not use class-conditional labels. Set class_cond=false."
            )

        time_col = ObservationColumns.TIMEPOINT.value
        cell_tps = test_ann_data.obs[time_col].to_numpy()

        metadata = self.config.get("method", {}).get("metadata", {})
        latent_noise_scale = float(metadata.get("latent_noise_scale", 0.7))

        embeds = _encode_latent(
            model,
            data,
            device,
            batch_size,
            use_encoder,
        )

        latent_direction = getattr(self, "latent_direction", None)
        latent_start_tp = getattr(self, "latent_start_tp", None)
        latent_end_tp = getattr(self, "latent_end_tp", None)

        if latent_direction is None or getattr(latent_direction, "size", 0) == 0:
            raise ValueError(
                "Squidiff latent direction is unavailable; train() must complete successfully and cache the reference direction before generation."
            )

        test_tps = _sorted_unique(cell_tps)
        global_tps = self.unique_tps or test_tps
        tp_positions = {tp: idx for idx, tp in enumerate(global_tps)}

        start_tp = latent_start_tp if latent_start_tp in tp_positions else global_tps[0]
        end_tp = latent_end_tp if latent_end_tp in tp_positions else global_tps[-1]
        denom = tp_positions[end_tp] - tp_positions[start_tp]
        if denom == 0:
            raise ValueError(
                "Squidiff latent direction cannot be scaled because the training time axis collapsed to a single point."
            )

        next_expr = np.full((test_ann_data.n_obs, gene_size), np.nan, dtype=np.float32)
        next_latent = np.full_like(embeds, np.nan, dtype=np.float32)

        for idx, tp in enumerate(test_tps[:-1]):
            next_tp = test_tps[idx + 1]
            mask = cell_tps == tp
            if not np.any(mask):
                continue

            # Mirror Squidiff's published interpolation path: move the mean latent
            # state along a reference direction, then add local noise and sample.
            delta_scale = (tp_positions[next_tp] - tp_positions[tp]) / denom
            source_center = embeds[mask].mean(axis=0)
            interp_center = source_center + latent_direction * delta_scale
            z_next = _sample_around_point(
                interp_center, int(np.sum(mask)), latent_noise_scale
            )

            preds_tp = _sample_from_latent(
                model,
                diffusion,
                z_next,
                device,
                gene_size,
                use_ddim,
                batch_size,
            )
            next_expr[mask] = preds_tp
            next_latent[mask] = z_next

        self._cached_outputs = (embeds, next_latent, next_expr)
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
        _, next_latent, _ = self._generate_outputs(test_ann_data)
        return next_latent

    def generate_next_tp_gex(self, test_ann_data) -> np.ndarray:
        """
        Generate gene expression for the next timepoint.
        """
        _, _, next_expr = self._generate_outputs(test_ann_data)
        return next_expr


if __name__ == "__main__":
    main(Squidiff)
