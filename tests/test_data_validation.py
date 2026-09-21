"""Raw-count validation must accept integer counts and reject log values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from heartmap.data import audit_adata, validate_counts


def test_integer_sparse_counts_pass(synthetic_atlas):
    info = validate_counts(synthetic_atlas, "counts")
    assert info["looks_like_integer_counts"]
    assert info["n_negative"] == 0
    assert info["is_sparse"]


def test_negative_values_fail(synthetic_atlas):
    bad = synthetic_atlas.copy()
    bad.layers["counts"] = bad.layers["counts"].copy()
    bad.layers["counts"].data[0] = -5.0
    with pytest.raises(RuntimeError):
        validate_counts(bad, "counts")


def test_log_normalized_values_fail(synthetic_atlas):
    import scanpy as sc

    bad = synthetic_atlas.copy()
    bad.X = bad.layers["counts"].copy()
    sc.pp.normalize_total(bad, target_sum=1e4)
    sc.pp.log1p(bad)
    with pytest.raises(RuntimeError):
        validate_counts(bad, None)


def test_missing_layer_raises(synthetic_atlas):
    with pytest.raises(RuntimeError):
        validate_counts(synthetic_atlas, "does_not_exist")


def test_audit_reports_structure(synthetic_atlas):
    a = audit_adata(synthetic_atlas)
    assert a["n_obs"] == synthetic_atlas.n_obs
    assert "donor" in a["candidate_donor_columns"]
    assert "cell_type" in a["candidate_label_columns"]
    assert "cell_states" in a["candidate_label_columns"]
    assert a["obs_names_unique"]


def test_dense_integer_counts_pass():
    import anndata as ad

    x = np.array([[0, 1, 2], [3, 0, 0]], dtype="float32")
    adata = ad.AnnData(X=x)
    info = validate_counts(adata, None)
    assert info["looks_like_integer_counts"]
    assert not info["is_sparse"]
