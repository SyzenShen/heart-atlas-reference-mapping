"""Confidence-coverage (abstention) analysis.

Confidence here is the maximum class score emitted by a model. It is a
*prediction confidence score*, NOT a calibrated probability: no calibration
analysis has been performed unless explicitly added.

For every threshold t we retain cells with confidence >= t and report
coverage and accuracy/F1 on the retained set. Higher threshold => lower
coverage; this is a hard monotonicity guarantee checked by the tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score)


def confidence_coverage(
    joined: pd.DataFrame, thresholds: list[float], reference_labels: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (coverage_table, rejected_composition_table).

    ``joined`` must contain confidence, predicted_label, true_cell_type.
    """
    y_true = joined["true_cell_type"].astype(str).to_numpy()
    y_pred = joined["predicted_label"].astype(str).to_numpy()
    conf = joined["confidence"].astype(float).to_numpy()
    in_ref = joined["true_cell_type"].astype(str).isin(set(reference_labels)).to_numpy()
    n_total = len(joined)

    rows = []
    rejected_parts = []
    prev_coverage = 1.0 + 1e-12
    for t in sorted(float(x) for x in thresholds):
        retained = conf >= t
        n_ret = int(retained.sum())
        coverage = n_ret / n_total if n_total else 0.0
        if coverage > prev_coverage + 1e-12:
            raise AssertionError(
                "coverage must be non-increasing as threshold rises"
            )
        prev_coverage = coverage

        if n_ret > 0:
            acc = float(accuracy_score(y_true[retained], y_pred[retained]))
            bacc = float(
                balanced_accuracy_score(y_true[retained], y_pred[retained])
            )
            mf1 = float(
                f1_score(
                    y_true[retained], y_pred[retained],
                    average="macro", zero_division=0,
                )
            )
            error_rate = 1.0 - acc
        else:
            acc = bacc = mf1 = float("nan")
            error_rate = float("nan")

        rejected = ~retained
        rows.append(
            {
                "threshold": t,
                "retained_cells": n_ret,
                "rejected_cells": int(rejected.sum()),
                "coverage": coverage,
                "accuracy_on_retained": acc,
                "balanced_accuracy_on_retained": bacc,
                "macro_f1_on_retained": mf1,
                "error_rate_on_retained": error_rate,
                "retained_out_of_reference_cells": int(
                    (retained & ~in_ref).sum()
                ),
                "mean_confidence": float(np.mean(conf[retained])) if n_ret else float("nan"),
            }
        )

        comp = (
            pd.DataFrame({"true_cell_type": y_true[rejected]})
            .value_counts("true_cell_type")
            .rename_axis("true_cell_type")
            .reset_index(name="n_rejected")
        )
        if not comp.empty:
            type_totals = pd.Series(y_true).value_counts()
            comp["fraction_of_type_rejected"] = comp.apply(
                lambda r: r["n_rejected"] / float(type_totals[r["true_cell_type"]]),
                axis=1,
            )
        comp["threshold"] = t
        rejected_parts.append(comp)

    coverage_df = pd.DataFrame(rows)
    rejected_df = (
        pd.concat(rejected_parts, ignore_index=True)
        if rejected_parts
        else pd.DataFrame(
            columns=["true_cell_type", "n_rejected", "fraction_of_type_rejected",
                     "threshold"]
        )
    )
    return coverage_df, rejected_df
