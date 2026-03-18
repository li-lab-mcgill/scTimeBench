"""
Squidiff runner script.

This script trains and evaluates Squidiff on an AnnData dataset.
It keeps the BaseMethod runner structure used across the project.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
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
    from Squidiff import logger  # type: ignore

    # from Squidiff import dist_util, logger
    from Squidiff.scrna_datasets import prepared_data  # type: ignore
    from Squidiff.resample import create_named_schedule_sampler  # type: ignore
    from Squidiff.script_util import (  # type: ignore
        create_model_and_diffusion,
        args_to_dict,
        model_and_diffusion_defaults,
    )
    from Squidiff.train_util import TrainLoop, plot_loss  # type: ignore

    device = _single_device()
    # dist_util.setup_dist()
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

    train_loop.run_loop()
    plot_loss(train_loop.loss_list, args)


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
    model, x: np.ndarray, device, batch_size: int, use_encoder: bool
) -> np.ndarray:
    if not use_encoder or not hasattr(model, "encoder"):
        return x.astype(np.float32)

    def _encoder_forward(batch_tensor: torch.Tensor):
        encoder = model.encoder
        try:
            import inspect

            sig = inspect.signature(encoder.forward)
            kwargs = {
                "label": None,
                "drug_dose": None,
                "control_feature": None,
            }
            filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return encoder(batch_tensor, **filtered)
        except Exception:
            return encoder(batch_tensor)

    embeddings = []
    for start in range(0, x.shape[0], batch_size):
        batch = torch.tensor(
            x[start : start + batch_size], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            z_sem = _encoder_forward(batch)
        embeddings.append(z_sem.detach().cpu().numpy())
    return np.vstack(embeddings)


def _sample_outputs(
    model,
    diffusion,
    x: np.ndarray,
    device,
    gene_size: int,
    use_ddim: bool,
    batch_size: int,
    use_encoder: bool,
) -> np.ndarray:
    sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop

    outputs = []
    for start in range(0, x.shape[0], batch_size):
        batch = torch.tensor(
            x[start : start + batch_size], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            if use_encoder and hasattr(model, "encoder"):
                try:
                    z_sem = model.encoder(
                        batch, label=None, drug_dose=None, control_feature=None
                    )
                except TypeError:
                    z_sem = model.encoder(batch)
                pred = sample_fn(
                    model,
                    shape=(batch.shape[0], gene_size),
                    model_kwargs={"z_mod": z_sem},
                    noise=None,
                )
            else:
                pred = sample_fn(
                    model,
                    shape=(batch.shape[0], gene_size),
                    model_kwargs={},
                    noise=None,
                )
        outputs.append(pred.detach().cpu().numpy())
    return np.vstack(outputs)


def _next_timepoint_mask(cell_tps: np.ndarray, unique_tps: List) -> np.ndarray:
    ordered = _sorted_unique(unique_tps)
    next_map = {
        ordered[i]: (ordered[i + 1] if i + 1 < len(ordered) else None)
        for i in range(len(ordered))
    }
    return np.array([next_map.get(tp) is not None for tp in cell_tps], dtype=bool)


class Squidiff(BaseMethod):
    def train(self, ann_data, all_tps: Optional[List] = None):
        """
        Training logic for Squidiff.
        """
        cache_path = os.path.join(
            self.config["output_path"], "trained_squidiff_model.pt"
        )
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
            return

        grouped_adata, unique_tps, tp_to_idx = _prepare_grouped_adata(ann_data, all_tps)
        self.unique_tps = unique_tps
        self.tp_to_idx = tp_to_idx

        train_data_path = os.path.join(
            self.config["output_path"], "squidiff_train.h5ad"
        )
        grouped_adata.write_h5ad(train_data_path)

        n_genes = grouped_adata.X.shape[1]
        args = _build_args(
            metadata, train_data_path, self.config["output_path"], n_genes
        )

        _run_training(args)

        model_path = os.path.join(args["resume_checkpoint"], "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Squidiff training did not produce a model.pt checkpoint."
            )

        self.model_path = model_path
        self.args = args

        torch.save(
            {
                "model_path": self.model_path,
                "args": self.args,
                "tp_to_idx": self.tp_to_idx,
                "unique_tps": self.unique_tps,
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

        embeds = _encode_latent(model, data, device, batch_size, use_encoder)
        preds = _sample_outputs(
            model,
            diffusion,
            data,
            device,
            gene_size,
            use_ddim,
            batch_size,
            use_encoder,
        )
        next_embeds = _encode_latent(model, preds, device, batch_size, use_encoder)

        final_ann_data = test_ann_data.copy()
        time_col = ObservationColumns.TIMEPOINT.value
        cell_tps = final_ann_data.obs[time_col].to_numpy()
        unique_tps = _sorted_unique(cell_tps)
        has_next = _next_timepoint_mask(cell_tps, unique_tps)

        next_expr = np.full_like(preds, np.nan, dtype=np.float32)
        next_expr[has_next] = preds[has_next]

        next_latent = np.full_like(next_embeds, np.nan, dtype=np.float32)
        next_latent[has_next] = next_embeds[has_next]

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
