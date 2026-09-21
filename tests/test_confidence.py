"""Confidence-coverage sweep invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd

from heartmap.confidence import confidence_coverage


def _joined(conf, true, pred):
    return pd.DataFrame({
        "confidence": conf,
        "true_cell_type": true,
        "predicted_label": pred,
    })


def test_zero_threshold_full_coverage():
    df = _joined(
        np.array([0.1, 0.5, 0.9]),
        np.array(["a", "a", "b"]),
        np.array(["a", "b", "b"]),
    )
    cov, _ = confidence_coverage(df, [0.0, 0.5, 0.9], ["a", "b"])
    row0 = cov.sort_values("threshold").iloc[0]
    assert row0.coverage == 1.0
    assert row0.retained_cells == 3


def test_coverage_never_increases():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, size=500)
    true = rng.choice(["a", "b"], size=500)
    df = _joined(conf, true, true)
    cov, _ = confidence_coverage(df, [0.0, 0.3, 0.6, 0.9, 0.95], ["a", "b"])
    coverages = cov.sort_values("threshold").coverage.to_numpy()
    assert np.all(np.diff(coverages) <= 0)


def test_accuracy_on_retained_matches_retained_subset():
    conf = np.array([1.0, 1.0, 0.2])
    true = np.array(["a", "a", "b"])
    pred = np.array(["a", "b", "b"])  # 2 high-conf: 1 correct
    df = _joined(conf, true, pred)
    cov, rej = confidence_coverage(df, [0.0, 0.5], ["a", "b"])
    hi = cov[cov.threshold == 0.5].iloc[0]
    assert hi.retained_cells == 2
    assert hi.coverage == 2 / 3
    assert hi.accuracy_on_retained == 0.5
    assert hi.error_rate_on_retained == 0.5
    # rejected composition records the rejected cell truth
    r = rej[rej.threshold == 0.5]
    assert r.iloc[0]["true_cell_type"] == "b"
    assert r.iloc[0]["n_rejected"] == 1


def test_monotonicity_assertion_design():
    # Function itself never receives unsorted thresholds wrongly; it sorts.
    df = _joined(np.array([0.2, 0.8]), np.array(["a", "a"]), np.array(["a", "a"]))
    cov, _ = confidence_coverage(df, [0.9, 0.0, 0.5], ["a"])
    assert cov.sort_values("threshold").coverage.is_monotonic_decreasing
