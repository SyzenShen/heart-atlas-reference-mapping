"""Metric correctness vs sklearn, missing classes, out-of-reference types."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score)

from heartmap.metrics import evaluate_method, join_predictions


def _frames(true, pred, conf=None, method="m"):
    cells = [f"c{i}" for i in range(len(true))]
    if conf is None:
        conf = np.ones(len(true)) * 0.8
    preds = pd.DataFrame({
        "cell_id": cells,
        "predicted_label": pred,
        "confidence": conf,
        "entropy": 0.1,
        "query_donor": "Q",
        "method": method,
        "status": "predicted",
    })
    sealed = pd.DataFrame({"cell_id": cells, "true_cell_type": true})
    return preds, sealed


def test_metrics_match_sklearn_closed_set():
    true = np.array(["a", "b", "c", "a", "b"])
    pred = np.array(["a", "a", "c", "a", "b"])
    preds, sealed = _frames(true, pred)
    res = evaluate_method(preds, sealed, ["a", "b", "c"], "m", min_support=1)
    row = res.summary[res.summary.scope == "closed_set"].iloc[0]
    assert row.accuracy == pytest.approx(accuracy_score(true, pred))
    assert row.balanced_accuracy == pytest.approx(
        balanced_accuracy_score(true, pred)
    )
    assert row.macro_f1 == pytest.approx(
        f1_score(true, pred, average="macro")
    )
    assert row.weighted_f1 == pytest.approx(
        f1_score(true, pred, average="weighted")
    )


def test_out_of_reference_type_separated():
    true = np.array(["a", "a", "newtype"])
    pred = np.array(["a", "b", "a"])
    preds, sealed = _frames(true, pred)
    res = evaluate_method(preds, sealed, ["a", "b"], "m", min_support=1)
    assert res.out_of_reference_types == ["newtype"]
    all_row = res.summary[res.summary.scope == "all"].iloc[0]
    closed = res.summary[res.summary.scope == "closed_set"].iloc[0]
    assert all_row.n_query_cells == 3
    assert all_row.n_out_of_reference_cells == 1
    assert closed.n_query_cells == 2
    assert closed.accuracy == pytest.approx(0.5)  # a vs [a,b]
    # newtype is kept in the all-query per-class table with zero precision etc.
    all_pc = res.per_class[res.per_class.scope == "all"]
    assert "newtype" in set(all_pc.label)
    new = all_pc[all_pc.label == "newtype"].iloc[0]
    assert new.precision == 0 and new.recall == 0 and new.support == 1
    # confusion count totals preserve every cell
    cm = res.confusion["m__all__counts"]
    assert cm.to_numpy().sum() == 3


def test_confusion_matrix_row_normalization():
    true = np.array(["a", "a", "b"])
    pred = np.array(["a", "b", "b"])
    preds, sealed = _frames(true, pred)
    res = evaluate_method(preds, sealed, ["a", "b"], "m", min_support=1)
    norm = res.confusion["m__closed_set__row_normalized"]
    np.testing.assert_allclose(norm.sum(axis=1), [1.0, 1.0])


def test_join_detects_duplicate_predictions():
    preds, sealed = _frames(["a", "a"], ["a", "b"])
    preds.loc[1, "cell_id"] = preds.loc[0, "cell_id"]
    with pytest.raises(ValueError):
        join_predictions(preds, sealed)


def test_evaluable_class_threshold(synthetic_config):
    # rare class below threshold excluded only from evaluable macro-F1
    true = np.array(["a"] * 50 + ["rare"])
    pred = np.array(["a"] * 50 + ["a"])
    preds, sealed = _frames(true, pred)
    res = evaluate_method(preds, sealed, ["a", "rare"], "m", min_support=5)
    row = res.summary[res.summary.scope == "all"].iloc[0]
    pc = res.per_class[res.per_class.scope == "all"]
    assert set(pc.label) == {"a", "rare"}  # never dropped from full table
    assert row.n_evaluable_classes == 1
    assert row.macro_f1_evaluable == pytest.approx(
        pc[pc.label == "a"].f1.iloc[0]
    )
