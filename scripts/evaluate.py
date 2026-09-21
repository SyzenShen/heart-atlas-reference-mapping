#!/usr/bin/env python3
"""Frozen-post-hoc evaluation.

This is the ONLY script allowed to open the sealed query labels
``data/splits/query_eval_labels_<tag>.csv``. It must be run after both
prediction files are frozen. Training/mapping scripts never import from here.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heartmap.baseline import METHOD_NAME as BASELINE_METHOD  # noqa: E402
from heartmap.confidence import confidence_coverage  # noqa: E402
from heartmap.config import load_config  # noqa: E402
from heartmap.metrics import evaluate_method  # noqa: E402
from heartmap.models import METHOD_NAME as SCANVI_METHOD  # noqa: E402
from heartmap.provenance import collect_environment, read_json, write_json  # noqa: E402
from heartmap.split import load_evaluation_labels  # noqa: E402

PREDICTION_FILES = {
    BASELINE_METHOD: "baseline_predictions",
    SCANVI_METHOD: "scanvi_predictions",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tag = cfg.run_tag
    mdir = cfg.results_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)

    # The single permitted read of sealed ground truth, via the sanctioned
    # loader (schema-validated; model-facing scripts never call this).
    sealed = load_evaluation_labels(cfg)
    print(f"Opened sealed labels for evaluation: {len(sealed)} cells")
    manifest = read_json(cfg.split_manifest_path)
    reference_labels = manifest["reference_label_set"]
    min_support = int(cfg["evaluable_class_min_support"])

    summaries, per_class_parts, joined_store = [], [], {}
    for method, stem in PREDICTION_FILES.items():
        path = cfg.results_dir / "predictions" / f"{stem}_{tag}.csv"
        if not path.exists():
            raise SystemExit(f"Missing frozen predictions: {path}")
        preds = pd.read_csv(path)
        if "true_cell_type" in preds.columns:
            raise SystemExit(
                f"{path.name} must not contain true labels before evaluation"
            )
        result = evaluate_method(preds, sealed, reference_labels, method, min_support)
        summaries.append(result.summary)
        per_class_parts.append(result.per_class)
        joined_store[method] = result.joined

        for key, cm in result.confusion.items():
            suffix = "counts" if key.endswith("counts") else "normalized"
            m = key.split("__")[0]
            scope = key.split("__")[1]
            cm.to_csv(
                mdir / f"confusion_{m}_{scope}_{suffix}_{tag}.csv",
                index_label="true_label",
            )

    summary_df = pd.concat(summaries, ignore_index=True)
    per_class_df = pd.concat(per_class_parts, ignore_index=True)
    summary_df.to_csv(mdir / f"summary_{tag}.csv", index=False)
    per_class_df.to_csv(mdir / f"per_class_{tag}.csv", index=False)

    # ---- Confidence-coverage sweep for every method ----------------------
    cov_parts, rej_parts = [], []
    for method, joined in joined_store.items():
        cov, rej = confidence_coverage(
            joined, list(cfg["confidence_thresholds"]), reference_labels
        )
        cov.insert(0, "method", method)
        rej.insert(0, "method", method)
        cov_parts.append(cov)
        rej_parts.append(rej)
    pd.concat(cov_parts, ignore_index=True).to_csv(
        mdir / f"confidence_coverage_{tag}.csv", index=False
    )
    pd.concat(rej_parts, ignore_index=True).to_csv(
        mdir / f"rejected_composition_{tag}.csv", index=False
    )

    # Joined tables for figure generation (truth now attached, eval-only).
    for method, joined in joined_store.items():
        joined.to_csv(
            cfg.results_dir / "predictions" / f"joined_eval_{method}_{tag}.csv",
            index=False,
        )

    scanvi_result_oor = sorted(
        set(joined_store[SCANVI_METHOD]["true_cell_type"]) - set(reference_labels)
    )
    write_json(
        mdir / f"evaluation_metadata_{tag}.json",
        {
            "run_tag": tag,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "query_donor": manifest["query_donor"],
            "n_query_cells": int(len(sealed)),
            "reference_labels": reference_labels,
            "out_of_reference_types": scanvi_result_oor,
            "n_out_of_reference_cells": int(
                joined_store[SCANVI_METHOD]["in_reference"].eq(False).sum()
            ),
            "evaluable_class_min_support": min_support,
            "methods": list(PREDICTION_FILES),
            "confidence_thresholds": list(cfg["confidence_thresholds"]),
            "confidence_definition": (
                "max per-class soft score; a prediction confidence score, "
                "NOT a calibrated probability"
            ),
            "environment": {
                k: collect_environment()[k]
                for k in ("python", "scvi", "scanpy", "sklearn", "git")
            },
        },
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
