"""Post-freeze evaluation metrics.

This module is the ONLY place sealed query labels
(``data/splits/query_eval_labels_<tag>.csv``) are joined to predictions.

Scopes reported for every method:

* ``all``        - every query cell, including true types absent from reference
* ``closed_set`` - query cells whose true type exists in the reference label set

Out-of-reference cells are never silently relabelled; they are counted
separately and always appear in per-class tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)

SCOPES = ("all", "closed_set")


@dataclass
class EvaluationResult:
    summary: pd.DataFrame
    per_class: pd.DataFrame
    confusion: dict[str, pd.DataFrame]
    joined: pd.DataFrame
    out_of_reference_types: list[str]
    reference_labels: list[str]


def join_predictions(predictions: pd.DataFrame, sealed: pd.DataFrame) -> pd.DataFrame:
    required_pred = {"cell_id", "predicted_label", "confidence", "method"}
    required_sealed = {"cell_id", "true_cell_type"}
    if not required_pred.issubset(predictions.columns):
        raise ValueError(f"predictions missing {required_pred - set(predictions.columns)}")
    if not required_sealed.issubset(sealed.columns):
        raise ValueError("sealed labels must contain cell_id/true_cell_type")
    if predictions["cell_id"].duplicated().any():
        raise ValueError("duplicate cell_id in predictions")
    if sealed["cell_id"].duplicated().any():
        raise ValueError("duplicate cell_id in sealed labels")
    joined = predictions.merge(sealed, on="cell_id", how="inner", validate="one_to_one")
    if len(joined) != len(predictions):
        raise ValueError(
            f"Only {len(joined)}/{len(predictions)} predictions matched sealed labels"
        )
    return joined


def _scope_frame(joined: pd.DataFrame, labels: list[str], min_support: int,
                 scope: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_true = joined["true_cell_type"].astype(str).to_numpy()
    y_pred = joined["predicted_label"].astype(str).to_numpy()

    all_labels = sorted(set(y_true) | set(y_pred))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "scope": scope,
            "label": all_labels,
            "precision": p,
            "recall": r,
            "f1": f,
            "support": s.astype(int),
        }
    )

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
        )
    cm_norm_df = pd.DataFrame(cm_norm, index=all_labels, columns=all_labels)

    evaluable = per_class.loc[
        per_class["support"] >= min_support, "label"
    ].tolist()
    macro_f1_evaluable = (
        float(per_class.loc[per_class["label"].isin(evaluable), "f1"].mean())
        if evaluable
        else float("nan")
    )

    summary = {
        "scope": scope,
        "n_query_cells": int(len(joined)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_f1_evaluable": macro_f1_evaluable,
        "evaluable_min_support": int(min_support),
        "n_evaluable_classes": int(len(evaluable)),
        "n_classes_true": int(len(set(y_true))),
    }
    return summary, per_class, cm_df, cm_norm_df


def evaluate_method(
    predictions: pd.DataFrame,
    sealed: pd.DataFrame,
    reference_labels: list[str],
    method: str,
    min_support: int,
) -> EvaluationResult:
    joined = join_predictions(predictions, sealed)
    reference_labels = sorted(map(str, reference_labels))
    true_types = set(joined["true_cell_type"].astype(str).unique())
    oor_types = sorted(true_types - set(reference_labels))
    joined["in_reference"] = joined["true_cell_type"].astype(str).isin(reference_labels)

    summaries: list[dict[str, Any]] = []
    per_class_parts: list[pd.DataFrame] = []
    confusion: dict[str, pd.DataFrame] = {}

    for scope, frame in (
        ("all", joined),
        ("closed_set", joined[joined["in_reference"]]),
    ):
        if frame.empty:
            continue
        summary, pc, cm_df, cm_norm_df = _scope_frame(
            frame, reference_labels, min_support, scope
        )
        summary["method"] = method
        summary["n_out_of_reference_types"] = len(oor_types)
        summary["n_out_of_reference_cells"] = int((~joined["in_reference"]).sum())
        pc.insert(0, "method", method)
        pc["in_reference"] = pc["label"].isin(reference_labels)
        summaries.append(summary)
        per_class_parts.append(pc)
        confusion[f"{method}__{scope}__counts"] = cm_df
        confusion[f"{method}__{scope}__row_normalized"] = cm_norm_df

    summary_df = pd.DataFrame(summaries)[
        [
            "method", "scope", "n_query_cells", "n_out_of_reference_types",
            "n_out_of_reference_cells", "accuracy", "balanced_accuracy",
            "macro_f1", "weighted_f1", "macro_f1_evaluable",
            "n_evaluable_classes", "evaluable_min_support", "n_classes_true",
        ]
    ]
    return EvaluationResult(
        summary=summary_df,
        per_class=pd.concat(per_class_parts, ignore_index=True),
        confusion=confusion,
        joined=joined,
        out_of_reference_types=oor_types,
        reference_labels=reference_labels,
    )
