"""scVI / scANVI reference training and scArches query mapping.

API verified against the current scvi-tools reference-mapping tutorial
("Reference mapping with SCVI-Tools", docs.scvi-tools.org, checked
2026-09-21) on scvi-tools 1.3.3 (Python 3.10 pin):

* ``SCVI.setup_anndata(layer=..., batch_key=...)``
* ``SCVI(use_layer_norm='both', use_batch_norm='none', encode_covariates=True,
  dropout_rate=0.2, n_layers=2)``
* ``SCANVI.from_scvi_model(scvi_model, unlabeled_category='Unknown',
  labels_key=...)`` then ``train(max_epochs=20, n_samples_per_label=100)``
* ``SCANVI.prepare_query_anndata(query, ref_path)``
* ``SCANVI.load_query_data(query, ref_path)`` then
  ``train(max_epochs=100, plan_kwargs={'weight_decay': 0.0},
  check_val_every_n_epoch=10)``
* ``model.predict(soft=True)`` for per-class scores.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import LABELS_KEY
from .config import Config

METHOD_NAME = "scanvi_scarches"


def resolve_accelerator(cfg: Config) -> dict[str, Any]:
    """Resolve training device.

    'auto' -> CUDA when available, otherwise CPU. Apple MPS is never chosen
    automatically: official scvi-tools/Lightning support for MPS in this
    workflow is not reliable.
    """
    import torch

    requested = str(cfg.get("accelerator", "auto")).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return {"accelerator": "gpu", "devices": int(cfg.get("devices", 1))}
        return {"accelerator": "cpu", "devices": 1}
    if requested in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU/CUDA requested but torch.cuda.is_available() is False")
        return {"accelerator": "gpu", "devices": int(cfg.get("devices", 1))}
    if requested == "mps":
        raise RuntimeError(
            "MPS is intentionally disabled: use cpu or a CUDA GPU (Colab T4)."
        )
    return {"accelerator": "cpu", "devices": 1}


def set_seed(cfg: Config) -> None:
    import scvi

    scvi.settings.seed = int(cfg["seed"])
    # Keep CPU thread usage explicit/bounded for reproducibility on laptops.
    torch = __import__("torch")
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))


def subset_hvg(adata, cfg: Config):
    """Subset an AnnData to the frozen reference HVG list in frozen order."""
    hvgs = [g for g in cfg.hvg_path.read_text().splitlines() if g.strip()]
    missing = [g for g in hvgs if g not in adata.var_names]
    if missing:
        raise RuntimeError(
            f"{len(missing)} HVGs missing from AnnData; query alignment must "
            "go through prepare_query_anndata, not this helper."
        )
    return adata[:, hvgs].copy()


def _history_to_frame(history) -> pd.DataFrame:
    """Convert scvi's history (metrics at possibly different cadences) to a
    single DataFrame, aligning on the epoch index with an outer join."""
    frames = []
    for key, val in dict(history).items():
        df = val if isinstance(val, pd.DataFrame) else pd.DataFrame({key: np.asarray(val).ravel()})
        if not isinstance(val, pd.DataFrame):
            df.columns = [key]
        else:
            df = df.copy()
            if list(df.columns) == [0] or len(df.columns) == 1:
                df.columns = [key]
            else:
                df.columns = [f"{key}__{c}" for c in df.columns]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.join(frame, how="outer")
    out.index.name = "epoch_index"
    return out.sort_index()


def _train_kwargs(cfg_section: dict, device: dict, **extra) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "max_epochs": int(cfg_section["max_epochs"]),
        "check_val_every_n_epoch": int(cfg_section.get("check_val_every_n_epoch", 1)),
        "early_stopping": bool(cfg_section.get("early_stopping", False)),
        "accelerator": device["accelerator"],
        "devices": device["devices"],
    }
    if kw["early_stopping"]:
        kw["early_stopping_patience"] = int(
            cfg_section.get("early_stopping_patience", 10)
        )
    kw.update(extra)
    return kw


# --------------------------------------------------------------------------
# Reference models
# --------------------------------------------------------------------------
def train_scvi_reference(reference, cfg: Config):
    import scvi

    device = resolve_accelerator(cfg)
    set_seed(cfg)
    scfg = cfg["scvi"]

    scvi.model.SCVI.setup_anndata(
        reference,
        layer=cfg["counts_layer"],
        batch_key=cfg["batch_key"],
    )
    model = scvi.model.SCVI(
        reference,
        n_hidden=int(scfg.get("n_hidden", 128)),
        n_latent=int(scfg["n_latent"]),
        n_layers=int(scfg["n_layers"]),
        dropout_rate=float(scfg["dropout_rate"]),
        use_layer_norm=scfg["use_layer_norm"],
        use_batch_norm=scfg["use_batch_norm"],
        encode_covariates=bool(scfg["encode_covariates"]),
        gene_likelihood=scfg.get("gene_likelihood", "zinb"),
    )
    kwargs = _train_kwargs(
        scfg,
        device,
        train_size=float(scfg.get("train_size", 0.9)),
        batch_size=int(scfg.get("batch_size", 256)),
    )
    t0 = time.perf_counter()
    model.train(**kwargs)
    wall = time.perf_counter() - t0
    summary = {
        "stage": "scvi_reference",
        "run_tag": cfg.run_tag,
        "device": device,
        "wallclock_seconds": round(wall, 2),
        "n_params_total": sum(p.numel() for p in model.module.parameters()),
        "n_params_trainable": sum(
            p.numel() for p in model.module.parameters() if p.requires_grad
        ),
        "is_trained": bool(model.is_trained),
        "train_kwargs": kwargs,
        "history": {
            "n_epochs_recorded": int(len(_history_to_frame(model.history_))),
            "columns": list(_history_to_frame(model.history_).columns),
        },
    }
    return model, summary


def train_scanvi_reference(scvi_model, reference, cfg: Config):
    import scvi

    device = resolve_accelerator(cfg)
    set_seed(cfg)
    xcfg = cfg["scanvi"]

    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        unlabeled_category=cfg["unlabeled_category"],
        labels_key=LABELS_KEY,
    )
    kwargs = _train_kwargs(xcfg, device)
    t0 = time.perf_counter()
    scanvi_model.train(
        n_samples_per_label=int(xcfg["n_samples_per_label"]),
        **kwargs,
    )
    wall = time.perf_counter() - t0
    summary = {
        "stage": "scanvi_reference",
        "run_tag": cfg.run_tag,
        "device": device,
        "wallclock_seconds": round(wall, 2),
        "n_params_total": sum(p.numel() for p in scanvi_model.module.parameters()),
        "n_params_trainable": sum(
            p.numel()
            for p in scanvi_model.module.parameters()
            if p.requires_grad
        ),
        "is_trained": bool(scanvi_model.is_trained),
        "n_label_classes": int(scanvi_model.summary_stats.get("n_labels", -1)),
        "train_kwargs": {**kwargs, "n_samples_per_label": int(xcfg["n_samples_per_label"])},
        "history": {
            "n_epochs_recorded": int(len(_history_to_frame(scanvi_model.history_))),
            "columns": list(_history_to_frame(scanvi_model.history_).columns),
        },
    }
    return scanvi_model, summary


# --------------------------------------------------------------------------
# scArches query mapping
# --------------------------------------------------------------------------
def prepare_and_load_query(query, ref_model_path: str, cfg: Config):
    """Align genes, instantiate the frozen query model and report params."""
    import scvi

    scvi.model.SCANVI.prepare_query_anndata(query, ref_model_path)
    query_model = scvi.model.SCANVI.load_query_data(query, ref_model_path)
    n_total = sum(p.numel() for p in query_model.module.parameters())
    n_trainable = sum(
        p.numel() for p in query_model.module.parameters() if p.requires_grad
    )
    return query_model, {"n_params_total": n_total, "n_params_trainable": n_trainable}


def train_query_model(query_model, cfg: Config):
    device = resolve_accelerator(cfg)
    set_seed(cfg)
    qcfg = cfg["query_mapping"]
    kwargs = _train_kwargs(qcfg, device, plan_kwargs={"weight_decay": 0.0})
    # weight_decay is pinned to 0.0 by config validation; never override from
    # query performance.
    t0 = time.perf_counter()
    query_model.train(**kwargs)
    wall = time.perf_counter() - t0
    summary = {
        "stage": "scanvi_query_mapping",
        "run_tag": cfg.run_tag,
        "device": device,
        "wallclock_seconds": round(wall, 2),
        "is_trained": bool(query_model.is_trained),
        "train_kwargs": kwargs,
        "history": {
            "n_epochs_recorded": int(len(_history_to_frame(query_model.history_))),
            "columns": list(_history_to_frame(query_model.history_).columns),
        },
    }
    return query_model, summary


def predict_query(query_model, query, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hard labels + soft scores. Confidence = max soft score."""
    scores = query_model.predict(query, soft=True)
    if isinstance(scores, pd.DataFrame):
        scores_df = scores.copy()
    else:
        scores_df = pd.DataFrame(np.asarray(scores))
    scores_df.index = query.obs_names
    class_cols = list(scores_df.columns)
    proba = scores_df[class_cols].to_numpy(dtype=float)
    predicted_idx = proba.argmax(axis=1)
    predicted = np.asarray(class_cols)[predicted_idx]
    confidence = proba.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(proba * np.log(np.clip(proba, 1e-300, 1.0))).sum(axis=1)

    predictions = pd.DataFrame(
        {
            "cell_id": query.obs_names.to_numpy(),
            "predicted_label": predicted.astype(str),
            "confidence": confidence.astype(float),
            "entropy": entropy.astype(float),
            "query_donor": str(query.obs[cfg["donor_key"]].iloc[0]),
            "method": METHOD_NAME,
            "status": "predicted",
        }
    )
    scores_df = scores_df.reset_index().rename(columns={"index": "cell_id"})
    if "cell_id" not in scores_df.columns:
        scores_df.columns = ["cell_id", *class_cols]
    return predictions, scores_df


def save_history(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _history_to_frame(model.history_).to_csv(path, index=False)
