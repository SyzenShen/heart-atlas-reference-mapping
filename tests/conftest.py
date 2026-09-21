"""Shared test fixtures: small synthetic count AnnData, no real data needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.stats import nbinom


def _make_counts(n_cells: int, labels: np.ndarray, donor_effect: float,
                 seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    n_genes = 120
    n_types = 3
    # Type-specific marker programs on disjoint gene blocks.
    programs = np.zeros((n_types, n_genes))
    for t in range(n_types):
        programs[t, t * 30 : (t + 1) * 30] = 3.0
    mu = np.exp(0.3 + programs[labels] + donor_effect)  # (cells, genes)
    p = 0.3
    r = 2.0
    counts = nbinom.rvs(r, p, size=mu.shape, random_state=rng).astype("float32")
    # modulate means: nbinom above ignores mu; build mean-scaled draws
    counts = nbinom.rvs(r, r / (r + mu), random_state=rng).astype("float32")
    return sp.csr_matrix(counts)


@pytest.fixture
def synthetic_atlas():
    """3 cell types, 3 donors (A,B,C); donor C held out as query."""
    import anndata as ad

    rng = np.random.default_rng(7)
    donors = ["A", "B", "C"]
    n_per = {"A": 240, "B": 240, "C": 200}
    obs_parts, x_parts = [], []
    for d in donors:
        n = n_per[d]
        labels = rng.integers(0, 3, size=n)
        effect = {"A": 0.0, "B": 0.25, "C": -0.2}[d]
        x_parts.append(_make_counts(n, labels, effect, seed=hash(d) % 2**32))
        obs_parts.append(
            pd.DataFrame(
                {
                    "donor": pd.Categorical([d] * n),
                    "cell_type": pd.Categorical(labels.astype(str)),
                    "cell_states": pd.Categorical([f"s{l}" for l in labels]),
                },
                index=[f"{d}_{i}" for i in range(n)],
            )
        )
    X = sp.vstack(x_parts).tocsr()
    obs = pd.concat(obs_parts)
    adata = ad.AnnData(X=X, obs=obs)
    adata.layers["counts"] = adata.X.copy()
    adata.var_names = [f"g{i}" for i in range(adata.n_vars)]
    return adata


@pytest.fixture
def synthetic_config(tmp_path):
    from heartmap.config import Config

    raw = {
        "_project_root": str(tmp_path),
        "experiment_name": "synthetic_test",
        "run_tag": "smoke",
        "seed": 0,
        "dataset": "synthetic",
        "donor_key": "donor",
        "batch_key": "donor",
        "cell_type_key": "cell_type",
        "counts_layer": "counts",
        "unlabeled_category": "Unknown",
        "n_hvg": 60,
        "hvg_flavor": "seurat_v3",
        "query_selection_rule": "largest_eligible_donor",
        "minimum_query_cells": 100,
        "minimum_query_cell_types": 3,
        "exclude_labels": [],
        "accelerator": "cpu",
        "devices": 1,
        "baseline": {
            "method": "pca_knn", "n_components": 10, "n_neighbors": 5,
            "weights": "distance", "metric": "euclidean", "target_sum": 1e4,
        },
        "scvi": {
            "n_latent": 8, "n_hidden": 16, "n_layers": 1, "dropout_rate": 0.1,
            "use_layer_norm": "both", "use_batch_norm": "none",
            "encode_covariates": True, "gene_likelihood": "zinb",
            "max_epochs": 1, "early_stopping": False,
            "check_val_every_n_epoch": 1, "train_size": 0.9, "batch_size": 64,
        },
        "scanvi": {
            "max_epochs": 1, "n_samples_per_label": 20, "early_stopping": False,
            "check_val_every_n_epoch": 1,
        },
        "query_mapping": {
            "max_epochs": 1, "weight_decay": 0.0, "early_stopping": False,
            "check_val_every_n_epoch": 1,
        },
        "confidence_thresholds": [0.0, 0.5, 0.9],
        "evaluable_class_min_support": 5,
        "umap": {"n_neighbors": 10, "min_dist": 0.3, "seed": 0},
        "figure_dpi": 80,
        "smoke_reference_max_cells": 10_000,
        "smoke_query_max_cells": 10_000,
    }
    cfg = Config(raw=raw)
    # Redirect on-disk locations into a temp project tree.
    return cfg
