"""
ARTEMIS runner script.

This script trains and evaluates ARTEMIS on an AnnData dataset.
It keeps the BaseMethod runner structure used across the project.
"""

from __future__ import annotations

import os
import sys
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import jax

from scTimeBench.model_utils.model_runner import main, BaseMethod
from scTimeBench.shared.constants import ObservationColumns, RequiredOutputColumns


_ARTEMIS_SRC_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "Artemis_module", "src")
)
if os.path.isdir(_ARTEMIS_SRC_PATH) and _ARTEMIS_SRC_PATH not in sys.path:
    sys.path.append(_ARTEMIS_SRC_PATH)

from datasets import Input_Dataset  # type: ignore
from training_setup import Training_Setup  # type: ignore
from training import Trainer  # type: ignore
from analysis_utils import (
    get_model_latents_single_data,  # type: ignore
    get_latent_trajectories,  # type: ignore
    get_reconstructed_trajectory,  # type: ignore
)


def _ensure_dense_float32(x):
    try:
        import scipy.sparse as sp  # type: ignore

        if sp.issparse(x):
            x = x.toarray()
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)


def _parse_hvg_n_top(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        if value.upper() == "ALL":
            return None
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(
                "ARTEMIS hvg_n_top must be 'ALL' or a positive integer."
            ) from exc
        value = parsed
    if isinstance(value, (int, np.integer)):
        if int(value) <= 0:
            raise ValueError("ARTEMIS hvg_n_top must be a positive integer.")
        return int(value)
    raise ValueError("ARTEMIS hvg_n_top must be 'ALL' or a positive integer.")


def _select_hvg(ann_data, hvg_n_top):
    n_top = _parse_hvg_n_top(hvg_n_top)
    if n_top is None:
        return ann_data
    try:
        import scanpy as sc  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "scanpy is required for HVG selection. Install it or set hvg_n_top: ALL."
        ) from exc

    adata = ann_data.copy()
    try:
        import scipy.sparse as sp  # type: ignore

        if sp.issparse(adata.X):
            data = adata.X.data
            adata.X.data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top)
    if "highly_variable" not in adata.var:
        raise ValueError("HVG selection failed to populate 'highly_variable'.")
    hv_mask = adata.var["highly_variable"].values
    if int(np.sum(hv_mask)) == 0:
        raise ValueError("HVG selection produced zero features.")
    return adata[:, hv_mask].copy()


def _sorted_unique(values):
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.number):
        return list(np.sort(np.unique(values)))
    try:
        import natsort  # type: ignore

        return list(natsort.natsorted(np.unique(values)))
    except Exception:
        return list(sorted(np.unique(values).tolist()))


def _ensure_list(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _build_time_mapping(all_tps: List) -> Tuple[Dict, List]:
    ordered = _sorted_unique(all_tps)
    mapping = {tp: float(idx) for idx, tp in enumerate(ordered)}
    return mapping, ordered


def _build_dataframe(ann_data, time_mapping: Dict) -> pd.DataFrame:
    time_col = ObservationColumns.TIMEPOINT.value
    if time_col not in ann_data.obs.columns:
        raise ValueError(f"Missing obs column '{time_col}' in AnnData")

    X = _ensure_dense_float32(ann_data.X)
    n_features = X.shape[1]

    if (
        getattr(ann_data, "var_names", None) is not None
        and len(ann_data.var_names) == n_features
    ):
        columns = list(ann_data.var_names)
    else:
        columns = [f"gene_{i}" for i in range(n_features)]

    df = pd.DataFrame(X, columns=columns)
    times = ann_data.obs[time_col].values

    mapped_times = []
    for t in times:
        if t not in time_mapping:
            raise ValueError(f"Timepoint '{t}' not found in training mapping")
        mapped_times.append(time_mapping[t])

    df["time"] = np.asarray(mapped_times, dtype=float)
    return df


def _sample_rows(
    values: np.ndarray, n_rows: int, rng: np.random.Generator
) -> np.ndarray:
    if values.shape[0] == n_rows:
        return values
    if values.shape[0] == 0:
        return np.zeros((n_rows, values.shape[1]), dtype=values.dtype)
    if values.shape[0] < n_rows:
        idx = rng.choice(values.shape[0], size=n_rows, replace=True)
        return values[idx]
    idx = rng.choice(values.shape[0], size=n_rows, replace=False)
    return values[idx]


def _build_time_value_list(mapping: Dict, ordered_labels: List) -> List[float]:
    return [mapping[label] for label in ordered_labels]


def _nan_inf_stats(name: str, values: np.ndarray) -> None:
    if values.size == 0:
        print(f"ARTEMIS diag: {name} is empty.")
        return
    finite_mask = np.isfinite(values)
    finite_count = int(np.sum(finite_mask))
    total = int(values.size)
    nan_count = int(np.sum(np.isnan(values)))
    inf_count = int(np.sum(np.isinf(values)))
    msg = (
        "ARTEMIS diag: "
        f"{name} shape={values.shape} finite={finite_count}/{total} "
        f"nan={nan_count} inf={inf_count}"
    )
    if finite_count > 0:
        finite_vals = values[finite_mask]
        msg += (
            f" min={float(np.min(finite_vals)):.4g}"
            f" max={float(np.max(finite_vals)):.4g}"
            f" mean={float(np.mean(finite_vals)):.4g}"
        )
    print(msg)


class Artemis(BaseMethod):
    def _build_dataset(
        self, ann_data, time_mapping: Dict, metadata: Dict
    ) -> Input_Dataset:
        disable_birth_death = bool(metadata.get("disable_birth_death", False))
        splitting_births_frac = (
            0.0 if disable_birth_death else metadata.get("splitting_births_frac", 0.2)
        )
        death_importance_rate = (
            0 if disable_birth_death else metadata.get("death_importance_rate", 100)
        )
        mb_prior = 0 if disable_birth_death else metadata.get("mb_prior", 5.0)
        hvg_n_top = metadata.get("hvg_n_top", "ALL")
        ann_data = _select_hvg(ann_data, hvg_n_top)
        train_df = _build_dataframe(ann_data, time_mapping)
        dataset = Input_Dataset(
            train_df,
            meta=None,
            meta_celltype_column=metadata.get("meta_celltype_column", None),
            splitting_births_frac=splitting_births_frac,
            eps=metadata.get("eps", 1e-7),
            steps_num=int(metadata.get("steps_num", 100)),
            val_split=bool(metadata.get("val_split", False)),
            death_importance_rate=death_importance_rate,
            f_val=metadata.get("f_val", None),
            std_threshold=metadata.get("std_threshold", 2.0),
            cutoff=metadata.get("cutoff", 0.2),
            mb_prior=mb_prior,
        )
        return dataset

    def _build_training_setup(self, dataset: Input_Dataset, metadata: Dict, key):
        hidden_dim = _ensure_list(metadata.get("hidden_dim", [64, 64]))

        raw_dec_hidden = metadata.get("dec_hidden_size", None)
        if raw_dec_hidden is None:
            raw_dec_hidden = metadata.get("dec_hidden_dim", None)
        dec_hidden_size = _ensure_list(raw_dec_hidden, default=list(hidden_dim))
        if dec_hidden_size and dec_hidden_size[-1] != 1:
            dec_hidden_size = list(dec_hidden_size) + [1]

        vae_input_dim = metadata.get("vae_input_dim", dataset.input_dim)
        if vae_input_dim is None:
            vae_input_dim = dataset.input_dim

        training_setup_kwargs = {
            "dataset_name": metadata.get("dataset", "AnnData"),
            "steps_num": int(metadata.get("steps_num", dataset.steps_num)),
            "epochs": int(metadata.get("epochs", 5)),
            "vae_epochs": int(metadata.get("vae_epochs", 100)),
            "key": key,
            "params": None,
            "objective": metadata.get("objective", "divergence"),
            "batch_size": int(metadata.get("batch_size", 512)),
            "hidden_dim": hidden_dim,
            "dec_hidden_size": dec_hidden_size,
            "ferryman_hidden_dim": _ensure_list(
                metadata.get("ferryman_hidden_dim", [64])
            ),
            "ferryman_activate_final": bool(
                metadata.get("ferryman_activate_final", True)
            ),
            "ipf_mask_dead": bool(metadata.get("ipf_mask_dead", False)),
            "reality_coefficient": float(metadata.get("reality_coefficient", 0.1)),
            "paths_reuse": int(metadata.get("paths_reuse", 5)),
            "num_sde": int(metadata.get("num_sde", 10)),
            "resnet": bool(metadata.get("resnet", False)),
            "feature_spatial_loss": bool(metadata.get("feature_spatial_loss", False)),
            "t_dim": int(metadata.get("t_dim", 16)),
            "vae_input_dim": int(vae_input_dim),
            "vae_enc_hidden_dim": _ensure_list(
                metadata.get("vae_enc_hidden_dim", [512, 512])
            ),
            "vae_dec_hidden_dim": _ensure_list(
                metadata.get("vae_dec_hidden_dim", [512, 512])
            ),
            "vae_t_dim": int(metadata.get("vae_t_dim", 8)),
            "calc_latent_loss": bool(metadata.get("calc_latent_loss", True)),
            "calc_recon_loss": bool(metadata.get("calc_recon_loss", True)),
            "vae_latent_dim": int(metadata.get("vae_latent_dim", 64)),
            "vae_batch_size": int(metadata.get("vae_batch_size", 64)),
            "use_sinkhorn_recon_loss": bool(
                metadata.get("use_sinkhorn_recon_loss", True)
            ),
            "use_sinkhorn_latent_loss": bool(
                metadata.get("use_sinkhorn_latent_loss", True)
            ),
        }
        training_setup = Training_Setup(dataset=dataset, **training_setup_kwargs)
        return training_setup, training_setup_kwargs

    def _build_trainer(
        self,
        dataset: Input_Dataset,
        training_setup: Training_Setup,
        metadata: Dict,
        key,
    ):
        beta1 = float(metadata.get("beta1", 0.9))
        beta2 = float(metadata.get("beta2", 0.999))
        if not (0.0 < beta1 < 1.0) or not (0.0 < beta2 < 1.0):
            raise ValueError(
                "ARTEMIS optimizer betas must be in (0, 1). "
                f"Got beta1={beta1}, beta2={beta2}."
            )
        trainer_kwargs = {
            "lr": float(metadata.get("lr", 1e-3)),
            "ferryman_lr": float(metadata.get("ferryman_lr", 1e-3)),
            "vae_lr": float(metadata.get("vae_lr", 1e-3)),
            "beta1": beta1,
            "beta2": beta2,
            "ferryman_coeff": float(metadata.get("ferryman_coeff", 1.0)),
        }
        trainer = Trainer(dataset, training_setup, key, **trainer_kwargs)
        return trainer, trainer_kwargs

    def _load_cache(self, cache_path: str) -> bool:
        if not os.path.exists(cache_path):
            return False
        with open(cache_path, "rb") as handle:
            payload = pickle.load(handle)

        dataset = payload["dataset"]
        time_mapping = payload["time_mapping"]
        ordered_timepoints = payload["ordered_timepoints"]
        training_setup_kwargs = payload["training_setup_kwargs"]
        trainer_kwargs = payload["trainer_kwargs"]
        key = payload["trainer_key"]
        params = payload["trainer_params"]
        vae_params = payload["vae_params"]

        training_setup = Training_Setup(dataset=dataset, **training_setup_kwargs)
        trainer = Trainer(dataset, training_setup, key, **trainer_kwargs)
        trainer.training_setup.state = (key, params)
        trainer.vae_params = vae_params
        disable_birth_death = bool(
            self.config.get("method", {})
            .get("metadata", {})
            .get("disable_birth_death", False)
        )
        if not disable_birth_death:
            trainer.training_setup.sde.killer = dataset.killing_function()

        self.trainer = trainer
        self.time_mapping = time_mapping
        self.ordered_timepoints = ordered_timepoints
        self.training_setup_kwargs = training_setup_kwargs
        return True

    def _save_cache(
        self,
        cache_path: str,
        dataset: Input_Dataset,
        time_mapping: Dict,
        ordered_timepoints: List,
        training_setup_kwargs: Dict,
        trainer_kwargs: Dict,
        trainer: Trainer,
    ) -> None:
        payload = {
            "dataset": dataset,
            "time_mapping": time_mapping,
            "ordered_timepoints": ordered_timepoints,
            "training_setup_kwargs": training_setup_kwargs,
            "trainer_kwargs": trainer_kwargs,
            "trainer_key": trainer.training_setup.state[0],
            "trainer_params": trainer.training_setup.state[1],
            "vae_params": trainer.vae_params,
        }
        with open(cache_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def train(self, ann_data, all_tps: Optional[List] = None):
        cache_path = os.path.join(self.config["output_path"], "trained_artemis.pkl")
        metadata = self.config.get("method", {}).get("metadata", {})
        disable_birth_death = bool(metadata.get("disable_birth_death", False))

        if self._load_cache(cache_path):
            print("Trained ARTEMIS model cache found, loading from file.")
            return

        if not all_tps:
            time_col = ObservationColumns.TIMEPOINT.value
            if time_col not in ann_data.obs.columns:
                raise ValueError(f"Missing obs column '{time_col}' in AnnData")
            all_tps = ann_data.obs[time_col].unique().tolist()

        time_mapping, ordered_timepoints = _build_time_mapping(all_tps)

        seed = int(metadata.get("seed", 0))
        key = jax.random.PRNGKey(seed)

        dataset = self._build_dataset(ann_data, time_mapping, metadata)
        training_setup, training_setup_kwargs = self._build_training_setup(
            dataset, metadata, key
        )
        trainer, trainer_kwargs = self._build_trainer(
            dataset, training_setup, metadata, key
        )

        td_schedule = metadata.get("td_schedule", None)
        if td_schedule is None:
            td_schedule = [metadata.get("td_coeff", None)]
        if not isinstance(td_schedule, list):
            td_schedule = [td_schedule]

        trainer.train(
            td_schedule=td_schedule,
            project_name=metadata.get("project_name", "benchmark"),
        )

        if not disable_birth_death:
            trainer.training_setup.sde.killer = dataset.killing_function()
        else:
            trainer.training_setup.sde.killer = None

        self.trainer = trainer
        self.time_mapping = time_mapping
        self.ordered_timepoints = ordered_timepoints
        self.training_setup_kwargs = training_setup_kwargs

        self._save_cache(
            cache_path,
            dataset,
            time_mapping,
            ordered_timepoints,
            training_setup_kwargs,
            trainer_kwargs,
            trainer,
        )

    def _build_prediction_maps(
        self,
        rng: np.random.Generator,
    ):
        metadata = self.config.get("method", {}).get("metadata", {})
        disable_birth_death = bool(metadata.get("disable_birth_death", False))
        fallback_to_train_latent = bool(
            metadata.get("fallback_to_train_latent_on_nan", False)
        )
        diagnostics = bool(metadata.get("diagnostics", False))
        dataset = self.trainer.dataset
        time_mapping = self.time_mapping

        train_df = dataset.x.copy()
        _, train_latent = get_model_latents_single_data(
            train_df, self.trainer.training_setup, self.trainer
        )
        if diagnostics:
            _nan_inf_stats("train_df", _ensure_dense_float32(train_df.values))
            _nan_inf_stats("train_latent", train_latent.iloc[:, :-1].values)

        timepoints = _build_time_value_list(time_mapping, self.ordered_timepoints)
        if not timepoints:
            raise ValueError("No timepoints found for ARTEMIS generation.")

        t0 = timepoints[0]
        val_data_init = train_latent[train_latent["time"] == t0].iloc[:, :-1].values
        if val_data_init.size == 0:
            raise ValueError(
                "Unable to resolve timepoint 0 latent data for ARTEMIS generation."
            )

        max_cells = int(
            self.config.get("method", {}).get("metadata", {}).get("max_sim_cells", 2000)
        )

        (
            pred_latent,
            pred_latent_alive,
            timepoints_all,
            timepoints_alive,
            _,
            _,
        ) = get_latent_trajectories(
            dataset,
            train_latent,
            timepoints,
            self.trainer,
            self.trainer.training_setup,
            val_data_init=val_data_init,
            t_0_orig=t0,
            max_size=max_cells,
            test=False,
        )

        if diagnostics:
            _nan_inf_stats("pred_latent_raw", np.asarray(pred_latent))
            _nan_inf_stats("pred_latent_alive_raw", np.asarray(pred_latent_alive))

        if disable_birth_death:
            pred_latent = np.asarray(pred_latent)
            timepoints_all = np.asarray(timepoints_all)
        else:
            pred_latent = np.asarray(pred_latent_alive)
            timepoints_all = np.asarray(timepoints_alive)

        if pred_latent.size == 0:
            raise ValueError("ARTEMIS generated empty latent trajectories.")

        latent_map = {}
        for tp in np.unique(timepoints_all):
            mask = timepoints_all == tp
            latent_map[tp] = pred_latent[mask]

        recon = get_reconstructed_trajectory(
            pred_latent,
            train_df,
            int(self.training_setup_kwargs["vae_input_dim"]),
            _ensure_list(self.training_setup_kwargs["vae_dec_hidden_dim"], [512, 512]),
            int(self.training_setup_kwargs["vae_latent_dim"]),
            self.trainer,
            timepoints_all,
        )
        recon = np.asarray(recon)
        if diagnostics:
            _nan_inf_stats("recon", recon)
        if recon.size == 0:
            raise ValueError("ARTEMIS reconstructed empty trajectories.")

        # drop any rows with NaNs/Infs across latent or reconstruction
        finite_latent = np.all(np.isfinite(pred_latent), axis=1)
        finite_recon = np.all(np.isfinite(recon), axis=1)
        finite_mask = finite_latent & finite_recon
        if not np.all(finite_mask):
            dropped = int(np.sum(~finite_mask))
            print(
                f"ARTEMIS generated {dropped} cells with NaNs/Infs. Dropping them from outputs."
            )
            pred_latent = pred_latent[finite_mask]
            recon = recon[finite_mask]
            timepoints_all = timepoints_all[finite_mask]

        if pred_latent.size == 0 or recon.size == 0:
            if not fallback_to_train_latent:
                raise ValueError("All ARTEMIS generated cells contained NaNs/Infs.")
            print(
                "ARTEMIS fallback: using encoded train latents after NaN/Inf generation."
            )
            pred_latent = train_latent.iloc[:, :-1].values
            timepoints_all = train_latent["time"].values
            if pred_latent.size == 0:
                raise ValueError("ARTEMIS fallback produced empty train latents.")

            recon = get_reconstructed_trajectory(
                pred_latent,
                train_df,
                int(self.training_setup_kwargs["vae_input_dim"]),
                _ensure_list(
                    self.training_setup_kwargs["vae_dec_hidden_dim"], [512, 512]
                ),
                int(self.training_setup_kwargs["vae_latent_dim"]),
                self.trainer,
                timepoints_all,
            )
            recon = np.asarray(recon)
            if recon.size == 0:
                raise ValueError("ARTEMIS fallback reconstructed empty trajectories.")

            finite_latent = np.all(np.isfinite(pred_latent), axis=1)
            finite_recon = np.all(np.isfinite(recon), axis=1)
            finite_mask = finite_latent & finite_recon
            if not np.all(finite_mask):
                dropped = int(np.sum(~finite_mask))
                print(
                    "ARTEMIS fallback produced "
                    f"{dropped} cells with NaNs/Infs. Dropping them from outputs."
                )
                pred_latent = pred_latent[finite_mask]
                recon = recon[finite_mask]
                timepoints_all = timepoints_all[finite_mask]

            if pred_latent.size == 0 or recon.size == 0:
                raise ValueError("ARTEMIS fallback still contains only NaNs/Infs.")

        recon_map = {}
        for tp in np.unique(timepoints_all):
            mask = timepoints_all == tp
            recon_map[tp] = recon[mask]

        return latent_map, recon_map

    def generate(self, test_ann_data, expected_output_path):
        # check if disable_birth_death is set
        disable_birth_death = bool(
            self.config.get("method", {})
            .get("metadata", {})
            .get("disable_birth_death", False)
        )
        print(f"ARTEMIS generation with disable_birth_death={disable_birth_death}")

        if self.trainer is None:
            raise RuntimeError("ARTEMIS model is not trained.")

        final_ann_data = test_ann_data.copy()
        time_col = ObservationColumns.TIMEPOINT.value
        if time_col not in test_ann_data.obs.columns:
            raise ValueError(f"Missing obs column '{time_col}' in AnnData")

        time_mapping = self.time_mapping
        cell_tps = test_ann_data.obs[time_col].values
        unknown = [tp for tp in cell_tps if tp not in time_mapping]
        if unknown:
            raise ValueError(
                f"Unknown timepoints found in test data: {sorted(set(unknown))}"
            )

        ordered_unique_tps = sorted(
            set(cell_tps.tolist()), key=lambda t: time_mapping[t]
        )

        rng = np.random.default_rng(0)
        allow_resample = bool(
            self.config.get("method", {})
            .get("metadata", {})
            .get("allow_resample_candidates", False)
        )
        latent_map, recon_map = self._build_prediction_maps(rng)

        any_latent = next(iter(latent_map.values()))
        n_features = any_latent.shape[1]
        embeds = np.full(
            (test_ann_data.n_obs, n_features), np.nan, dtype=any_latent.dtype
        )

        for tp_label in ordered_unique_tps:
            mask = cell_tps == tp_label
            tp_val = time_mapping[tp_label]
            candidates = latent_map.get(tp_val)
            if candidates is None or candidates.size == 0:
                raise ValueError(
                    f"No alive latent candidates for timepoint {tp_label} (mapped to {tp_val})."
                )
            if np.any(~np.isfinite(candidates)):
                raise ValueError(
                    f"NaNs/Infs found in alive latent candidates for timepoint {tp_label}."
                )
            if candidates.shape[0] != int(mask.sum()):
                if allow_resample:
                    candidates = _sample_rows(candidates, int(mask.sum()), rng)
                else:
                    raise ValueError(
                        f"Alive latent count mismatch for timepoint {tp_label}: "
                        f"expected {int(mask.sum())}, got {candidates.shape[0]}."
                    )
            embeds[mask] = candidates

        if RequiredOutputColumns.EMBEDDING in self.required_outputs:
            final_ann_data.obsm[RequiredOutputColumns.EMBEDDING.value] = embeds

        if RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING in self.required_outputs:
            next_embeds = np.full_like(embeds, np.nan)
            for i, tp_label in enumerate(ordered_unique_tps):
                mask = cell_tps == tp_label
                if i + 1 >= len(ordered_unique_tps):
                    # for final timepoint, keep alive embeddings from current timepoint
                    next_embeds[mask] = embeds[mask]
                    continue
                next_tp_label = ordered_unique_tps[i + 1]
                next_tp_val = time_mapping[next_tp_label]
                candidates = latent_map.get(next_tp_val)
                if candidates is None or candidates.size == 0:
                    raise ValueError(
                        f"No alive latent candidates for next timepoint {next_tp_label} (mapped to {next_tp_val})."
                    )
                if np.any(~np.isfinite(candidates)):
                    raise ValueError(
                        f"NaNs/Infs found in alive latent candidates for next timepoint {next_tp_label}."
                    )
                if candidates.shape[0] != int(mask.sum()):
                    if allow_resample:
                        candidates = _sample_rows(candidates, int(mask.sum()), rng)
                    else:
                        raise ValueError(
                            f"Alive latent count mismatch for next timepoint {next_tp_label}: "
                            f"expected {int(mask.sum())}, got {candidates.shape[0]}."
                        )
                next_embeds[mask] = candidates
            final_ann_data.obsm[
                RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING.value
            ] = next_embeds

        if (
            RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION
            in self.required_outputs
        ):
            next_expr = np.full(
                (test_ann_data.n_obs, test_ann_data.n_vars), np.nan, dtype=np.float32
            )
            for i, tp_label in enumerate(ordered_unique_tps):
                mask = cell_tps == tp_label
                if i + 1 >= len(ordered_unique_tps):
                    # for final timepoint, keep reconstructed expression from current timepoint
                    current_tp_val = time_mapping[tp_label]
                    candidates = recon_map.get(current_tp_val)
                    if candidates is None or candidates.size == 0:
                        raise ValueError(
                            f"No reconstructed candidates for timepoint {tp_label} (mapped to {current_tp_val})."
                        )
                    if np.any(~np.isfinite(candidates)):
                        raise ValueError(
                            f"NaNs/Infs found in reconstructed candidates for timepoint {tp_label}."
                        )
                    if candidates.shape[0] != int(mask.sum()):
                        if allow_resample:
                            candidates = _sample_rows(candidates, int(mask.sum()), rng)
                        else:
                            raise ValueError(
                                f"Reconstructed count mismatch for timepoint {tp_label}: "
                                f"expected {int(mask.sum())}, got {candidates.shape[0]}."
                            )
                    next_expr[mask] = candidates
                    continue
                next_tp_label = ordered_unique_tps[i + 1]
                next_tp_val = time_mapping[next_tp_label]
                candidates = recon_map.get(next_tp_val)
                if candidates is None or candidates.size == 0:
                    raise ValueError(
                        f"No reconstructed candidates for next timepoint {next_tp_label} (mapped to {next_tp_val})."
                    )
                if np.any(~np.isfinite(candidates)):
                    raise ValueError(
                        f"NaNs/Infs found in reconstructed candidates for next timepoint {next_tp_label}."
                    )
                if candidates.shape[0] != int(mask.sum()):
                    if allow_resample:
                        candidates = _sample_rows(candidates, int(mask.sum()), rng)
                    else:
                        raise ValueError(
                            f"Reconstructed count mismatch for next timepoint {next_tp_label}: "
                            f"expected {int(mask.sum())}, got {candidates.shape[0]}."
                        )
                next_expr[mask] = candidates
            final_ann_data.obsm[
                RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION.value
            ] = next_expr

        final_ann_data.write_h5ad(expected_output_path)


if __name__ == "__main__":
    main(Artemis)
