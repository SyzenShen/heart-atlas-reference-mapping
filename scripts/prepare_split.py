#!/usr/bin/env python3
"""Deterministic donor-held-out split, label sealing, reference-only HVGs.

Stages
------
1. Load audited dataset and materialise the verified counts layer.
2. Select the query donor with the pre-registered largest-eligible rule.
3. Save reference / sealed-query AnnData, sealed labels CSV and manifest.
4. Select HVGs on the REFERENCE ONLY; save the gene list.
5. Run leakage/sealing assertions (hard failure on any violation).

True query labels are never printed into model artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heartmap import LABELS_KEY  # noqa: E402
from heartmap.config import load_config  # noqa: E402
from heartmap.data import (load_heart_dataset, select_reference_hvgs,
                           validate_counts)  # noqa: E402
from heartmap.provenance import write_json  # noqa: E402
from heartmap.split import (assert_no_overlap, assert_query_sealed,
                            assert_sealed_labels_match, make_split)  # noqa: E402


def load_raw(cfg):
    adata = load_heart_dataset(
        save_path=str(cfg.data_raw_dir),
        remove_nuisance_clusters=bool(cfg["remove_nuisance_clusters"]),
    )
    # The 20k HHCA loader keeps raw counts in X; expose them as layers['counts']
    # so every downstream call uses the explicit, audited count layer.
    validate_counts(adata, None)
    if cfg["counts_layer"] not in adata.layers:
        adata.layers[cfg["counts_layer"]] = adata.X.copy()
    validate_counts(adata, cfg["counts_layer"])
    return adata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    adata = load_raw(cfg)
    reference, query, sealed, manifest, stats, reference_full = make_split(adata, cfg)

    # ---- Reference-only HVG selection ------------------------------------
    # Always fitted on the FULL reference; the smoke cell subsample only
    # shrinks the training data, never the reference-only learned preprocessing.
    hvgs = select_reference_hvgs(reference_full, cfg)
    cfg.hvg_path.write_text("\n".join(hvgs) + "\n")
    print(f"Selected {len(hvgs)} reference-only HVGs -> {cfg.hvg_path}")

    # ---- Hard leakage / sealing checks ------------------------------------
    assert_no_overlap(reference, query, cfg["donor_key"])
    assert_query_sealed(query, cfg)
    sealed_back = assert_sealed_labels_match(query, cfg)

    write_json(
        cfg.splits_dir / f"preprocess_metadata_{cfg.run_tag}.json",
        {
            "n_hvg": len(hvgs),
            "hvg_flavor": cfg.get("hvg_flavor"),
            "hvg_batch_key": cfg["batch_key"],
            "hvg_selected_on": "reference_only",
            "counts_layer": cfg["counts_layer"],
            "query_labels_sealed": True,
            "sealed_labels_n": int(len(sealed_back)),
            "reference_label_set": manifest["reference_label_set"],
        },
    )

    print("Query donor:", manifest["query_donor"])
    print("Reference donors:", ", ".join(manifest["reference_donor_ids"]))
    print(
        f"Reference cells: {manifest['n_reference_cells']} | "
        f"query cells: {manifest['n_query_cells']}"
    )
    print(
        "Query label values in model input:",
        query.obs[LABELS_KEY].astype(str).unique().tolist(),
    )
    print("All leakage/sealing assertions passed.")


if __name__ == "__main__":
    main()
