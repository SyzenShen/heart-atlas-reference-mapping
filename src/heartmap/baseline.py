"""Reference-only PCA + distance-weighted kNN baseline.

Every learned quantity (HVGs, library-normalisation convention, PCA axes, the
kNN classifier) is fitted on reference cells alone. Query cells only ever call
``transform`` / ``predict``. Query true labels are not read here.

Densification note: scanpy/sklearn PCA on 15,632 x 2,000 float32 is ~125 MB;
we deliberately densify in float32 rather than running a sparse SVD with
different semantics, and record this decision in the run metadata.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import LABELS_KEY
from .config import Config
from .data import apply_reference_genes

METHOD_NAME = "pca_knn"


def load_hvgs(cfg: Config) -> list[str]:
    return [g for g in cfg.hvg_path.read_text().splitlines() if g.strip()]


def _normalized_matrix(adata, hvgs: list[str], counts_layer: str,
                       target_sum: float) -> np.ndarray:
    """HVG subset (reference-defined) -> per-dataset library norm + log1p.

    Library-size normalisation has no fitted global parameters; it is computed
    independently within each dataset (scanpy's documented query workflow).
    """
    import scanpy as sc

    sub = apply_reference_genes(adata, hvgs, counts_layer, fill_missing=True)
    # Drive normalisation from the verified counts layer; never mutate it.
    sub.X = sub.layers[counts_layer].copy()
    sc.pp.normalize_total(sub, target_sum=target_sum)
    sc.pp.log1p(sub)
    return sub.X.toarray().astype(np.float32, copy=False), sub.obs.copy()


def run_baseline(reference, query, cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier

    bcfg = cfg["baseline"]
    hvgs = load_hvgs(cfg)
    target_sum = float(bcfg.get("target_sum", 1e4))
    n_components = int(bcfg["n_components"])
    k = int(bcfg["n_neighbors"])
    weights = str(bcfg["weights"])
    metric = str(bcfg.get("metric", "euclidean"))

    t0 = time.perf_counter()
    x_ref, ref_obs = _normalized_matrix(
        reference, hvgs, cfg["counts_layer"], target_sum
    )
    x_query, q_obs = _normalized_matrix(
        query, hvgs, cfg["counts_layer"], target_sum
    )

    pca = PCA(
        n_components=min(n_components, x_ref.shape[1], x_ref.shape[0] - 1),
        random_state=int(cfg["seed"]),
    )
    z_ref = pca.fit_transform(x_ref)
    z_query = pca.transform(x_query)
    t_fit = time.perf_counter()

    ref_labels = reference.obs[LABELS_KEY].astype(str).to_numpy()
    knn = KNeighborsClassifier(
        n_neighbors=min(k, z_ref.shape[0]), weights=weights, metric=metric
    )
    knn.fit(z_ref, ref_labels)
    predicted = knn.predict(z_query)
    proba = knn.predict_proba(z_query)
    t_predict = time.perf_counter()

    confidence = proba.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log(np.clip(proba, 1e-300, 1.0))
        entropy = -(proba * log_p).sum(axis=1)  # nats

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

    metadata = {
        "method": METHOD_NAME,
        "run_tag": cfg.run_tag,
        "n_hvg": len(hvgs),
        "pca_n_components_fitted": int(pca.n_components_),
        "pca_explained_variance_ratio": [
            float(v) for v in pca.explained_variance_ratio_
        ],
        "pca_total_explained_variance": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "knn_k": int(knn.n_neighbors),
        "knn_metric": metric,
        "knn_weights": weights,
        "target_sum": target_sum,
        "normalization": "normalize_total(target_sum) + log1p, per dataset",
        "densification": (
            "HVG matrix densified to float32 before sklearn PCA "
            "(15632x2000 ~= 0.125 GB); no reference/query joint processing"
        ),
        "reference_cell_count": int(z_ref.shape[0]),
        "query_cell_count": int(z_query.shape[0]),
        "preprocess_fit_seconds": round(t_fit - t0, 3),
        "knn_fit_seconds": round(t_predict - t_fit, 3),
        "wallclock_seconds": round(t_predict - t0, 3),
        "reference_label_classes": knn.classes_.tolist(),
        "labels_used_for_fit": LABELS_KEY,
        "query_true_labels_read": False,
    }
    return predictions, metadata


def prediction_columns() -> list[str]:
    return [
        "cell_id",
        "predicted_label",
        "confidence",
        "entropy",
        "query_donor",
        "method",
        "status",
    ]
