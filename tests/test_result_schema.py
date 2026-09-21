"""Result schemas, config validation and a full synthetic CPU pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heartmap.baseline import prediction_columns, run_baseline
from heartmap.config import ConfigError, load_config
from heartmap.models import (METHOD_NAME, prepare_and_load_query,
                             predict_query, save_history, subset_hvg,
                             train_query_model, train_scanvi_reference,
                             train_scvi_reference)
from heartmap.split import make_split
from heartmap.metrics import evaluate_method

PROJECT_CONFIGS = [
    "configs/main.yaml",
    "configs/smoke.yaml",
]


def test_shipped_configs_validate():
    for p in PROJECT_CONFIGS:
        cfg = load_config(p)
        assert cfg["query_selection_rule"] == "largest_donor_by_cells"


def test_config_rejects_bad_threshold(tmp_path):
    bad = load_config("configs/main.yaml")
    bad.raw["confidence_thresholds"] = [0.5]  # missing 0.0
    with pytest.raises(ConfigError):
        bad.validate()


def test_config_rejects_nonzero_weight_decay():
    bad = load_config("configs/main.yaml")
    bad.raw["query_mapping"]["weight_decay"] = 1e-4
    with pytest.raises(ConfigError):
        bad.validate()


def test_config_rejects_unknown_rule_and_tag():
    bad = load_config("configs/main.yaml")
    bad.raw["query_selection_rule"] = "pick_best_after_scoring"
    with pytest.raises(ConfigError):
        bad.validate()
    bad2 = load_config("configs/main.yaml")
    bad2.raw["run_tag"] = "holiday"
    with pytest.raises(ConfigError):
        bad2.validate()


def test_prediction_schema_baseline(synthetic_atlas, synthetic_config):
    ref, query = make_split(synthetic_atlas, synthetic_config)[:2]
    _write_simple_hvgs(ref, synthetic_config)
    preds, meta = run_baseline(ref, query, synthetic_config)
    assert list(preds.columns) == prediction_columns()
    assert preds.cell_id.is_unique
    assert preds["method"].iloc[0] == "pca_knn"
    assert preds["query_donor"].iloc[0] == "A"


def _write_simple_hvgs(reference, cfg, n=60) -> list[str]:
    """Reference-only top-variance genes (avoids loess in tiny synthetic data;
    HVG flavor itself is tested separately with seurat_v3)."""
    mat = reference.layers["counts"]
    mean = np.asarray(mat.mean(axis=0)).ravel()
    var = np.asarray(mat.multiply(mat).mean(axis=0)).ravel() - mean**2
    top = np.argsort(var)[::-1][:n]
    genes = reference.var_names[top].astype(str).tolist()
    cfg.data_processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.hvg_path.write_text("\n".join(genes) + "\n")
    return genes


def test_full_synthetic_scvi_scanvi_scarches_pipeline(synthetic_atlas,
                                                      synthetic_config):
    ref, query, sealed, manifest, _, _ = make_split(
        synthetic_atlas, synthetic_config
    )
    _write_simple_hvgs(ref, synthetic_config)
    ref_h = subset_hvg(ref, synthetic_config)
    q_h = subset_hvg(query, synthetic_config)

    scvi_model, _ = train_scvi_reference(ref_h, synthetic_config)
    scanvi_model, _ = train_scanvi_reference(
        scvi_model, ref_h, synthetic_config
    )
    ref_path = synthetic_config.models_dir / "scanvi_reference_synth"
    synthetic_config.models_dir.mkdir(exist_ok=True)
    scanvi_model.save(str(ref_path), overwrite=True)

    qmodel, params = prepare_and_load_query(q_h, str(ref_path), synthetic_config)
    # scArches freezes reference weights: only a subset is trainable.
    assert params["n_params_trainable"] < params["n_params_total"]
    qmodel, _ = train_query_model(qmodel, synthetic_config)
    save_history(qmodel, synthetic_config.results_dir / "training" / "qhist.csv")

    preds, scores = predict_query(qmodel, q_h, synthetic_config)
    assert list(preds.columns) == prediction_columns()
    assert preds.cell_id.is_unique
    assert preds["method"].iloc[0] == METHOD_NAME
    assert preds["confidence"].between(0, 1).all()
    assert (preds["entropy"] >= -1e-9).all()
    assert set(scores.columns) - {"cell_id"} == {"0", "1", "2"}
    np.testing.assert_allclose(
        scores[["0", "1", "2"]].to_numpy().sum(axis=1), 1.0, atol=1e-5
    )

    result = evaluate_method(
        preds, sealed, manifest["reference_label_set"], METHOD_NAME,
        min_support=5,
    )
    assert set(result.summary.scope) == {"all", "closed_set"}
    assert result.summary.n_query_cells.max() == query.n_obs
    cm = result.confusion[f"{METHOD_NAME}__all__counts"].to_numpy().sum()
    assert cm == query.n_obs
