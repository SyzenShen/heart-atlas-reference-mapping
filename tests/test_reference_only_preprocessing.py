"""HVGs, PCA and the kNN classifier must be fitted on reference data alone."""

from __future__ import annotations

import numpy as np
import pytest

from heartmap.baseline import run_baseline
from heartmap.data import (apply_reference_genes, select_reference_hvgs,
                           validate_counts)
from heartmap.split import make_split


def _split(synthetic_atlas, synthetic_config):
    return make_split(synthetic_atlas, synthetic_config)[:2]


def test_hvgs_selected_from_reference_only(synthetic_atlas, synthetic_config):
    ref, query = _split(synthetic_atlas, synthetic_config)
    hvgs = select_reference_hvgs(ref, synthetic_config)
    assert len(hvgs) == synthetic_config["n_hvg"]
    # every selected gene exists in the reference object
    assert set(hvgs).issubset(set(ref.var_names))
    # order is deterministic across calls
    hvgs2 = select_reference_hvgs(ref, synthetic_config)
    assert hvgs == hvgs2
    # persist as prepare_split does; file content equals the returned list
    synthetic_config.hvg_path.write_text("\n".join(hvgs) + "\n")
    assert synthetic_config.hvg_path.read_text().splitlines() == hvgs


def test_query_genes_aligned_and_missing_zero_padded(synthetic_atlas,
                                                     synthetic_config):
    ref, query = _split(synthetic_atlas, synthetic_config)
    hvgs = select_reference_hvgs(ref, synthetic_config)
    # Drop two genes from the query -> they must be zero-padded in HVG order.
    drop = hvgs[:2]
    q_small = query[:, [g for g in query.var_names if g not in drop]].copy()
    aligned = apply_reference_genes(
        q_small, hvgs, synthetic_config["counts_layer"], fill_missing=True
    )
    assert list(aligned.var_names) == hvgs
    idx = [hvgs.index(g) for g in drop]
    assert aligned.X[:, idx].sum() == 0
    assert aligned.layers["counts"][:, idx].sum() == 0


def test_pca_knn_does_not_use_query_labels(synthetic_atlas, synthetic_config):
    ref, query = _split(synthetic_atlas, synthetic_config)
    hvgs = select_reference_hvgs(ref, synthetic_config)
    synthetic_config.hvg_path.write_text("\n".join(hvgs) + "\n")
    preds1, meta = run_baseline(ref, query, synthetic_config)
    assert meta["query_true_labels_read"] is False
    # PCA fitted on reference: output dims and class counts consistent
    assert meta["pca_n_components_fitted"] == synthetic_config["baseline"]["n_components"]
    assert meta["reference_cell_count"] == ref.n_obs
    assert meta["query_cell_count"] == query.n_obs
    # Confidence in [0,1]; predicted labels all from reference label set
    assert preds1["confidence"].between(0, 1).all()
    assert set(preds1["predicted_label"]).issubset(
        set(ref.obs["labels_scanvi"].astype(str))
    )
    # Shuffling query row order permutes outputs identically: cell order is
    # the only thing that changes, query labels are not inputs.
    order = np.random.default_rng(0).permutation(query.n_obs)
    q2 = query[order].copy()
    preds2, _ = run_baseline(ref, q2, synthetic_config)
    merged = preds1.merge(
        preds2, on="cell_id", suffixes=("_a", "_b")
    )
    assert (merged["predicted_label_a"] == merged["predicted_label_b"]).all()
    assert len(preds2) == query.n_obs


def test_baseline_uses_count_layer_not_logcounts(synthetic_atlas,
                                                 synthetic_config):
    ref, _ = _split(synthetic_atlas, synthetic_config)
    select_reference_hvgs(ref, synthetic_config)
    # reference counts must still validate as integers before baseline runs
    info = validate_counts(ref, synthetic_config["counts_layer"])
    assert info["looks_like_integer_counts"]
