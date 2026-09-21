#!/usr/bin/env python3
"""scArches query mapping with the frozen scANVI reference.

The query donor's true labels are NOT read by this script. Sealing guards use
only the model-facing query object and the split manifest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heartmap import LABELS_KEY  # noqa: E402
from heartmap.config import load_config  # noqa: E402
from heartmap.data import validate_counts  # noqa: E402
from heartmap.models import (METHOD_NAME, prepare_and_load_query,
                             predict_query, save_history, subset_hvg,
                             train_query_model)  # noqa: E402
from heartmap.provenance import collect_environment, read_json, write_json  # noqa: E402
from heartmap.split import LeakageError, assert_query_sealed, load_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    reference, query, _, manifest = load_split(cfg)
    try:
        assert_query_sealed(query, cfg)
        # Donor isolation check without opening sealed labels:
        ref_donors = set(map(str, manifest["reference_donor_ids"]))
        q_donors = set(query.obs[cfg["donor_key"]].astype(str).unique())
        if ref_donors & q_donors:
            raise LeakageError("query donor appears in reference manifest")
        if manifest["query_donor"] not in q_donors:
            raise LeakageError("query object donor does not match manifest")
    except LeakageError as exc:
        raise SystemExit(f"Leakage guard failed; aborting mapping: {exc}")

    scanvi_path = cfg.models_dir / f"scanvi_reference_{cfg.run_tag}"
    if not scanvi_path.exists():
        raise SystemExit(f"Reference scANVI model missing: {scanvi_path}")
    if not cfg.hvg_path.exists():
        raise SystemExit("HVG list missing; run prepare_split.py first")

    query_hvg = subset_hvg(query, cfg)
    validate_counts(query_hvg, cfg["counts_layer"])

    print("Preparing query genes and loading frozen query model ...")
    query_model, param_info = prepare_and_load_query(
        query_hvg, str(scanvi_path), cfg
    )
    print(
        f"Trainable parameters: {param_info['n_params_trainable']:,} / "
        f"{param_info['n_params_total']:,} total (reference weights frozen)"
    )

    print("scArches query update (weight_decay=0.0) ...")
    query_model, query_summary = train_query_model(query_model, cfg)
    query_model.save(
        str(cfg.models_dir / f"scanvi_query_{cfg.run_tag}"), overwrite=True
    )
    save_history(
        query_model,
        cfg.results_dir / "training" / f"query_mapping_history_{cfg.run_tag}.csv",
    )
    query_summary["params"] = param_info
    query_summary["query_donor"] = manifest["query_donor"]
    query_summary["environment"] = {
        k: collect_environment()[k]
        for k in ("python", "scvi", "torch", "lightning", "git")
    }
    write_json(
        cfg.results_dir / "training" / f"query_mapping_summary_{cfg.run_tag}.json",
        query_summary,
    )

    print("Predicting query labels (soft scores) ...")
    predictions, scores = predict_query(query_model, query_hvg, cfg)
    pred_path = (
        cfg.results_dir / "predictions" / f"scanvi_predictions_{cfg.run_tag}.csv"
    )
    predictions.to_csv(pred_path, index=False)
    scores.to_csv(
        cfg.results_dir / "predictions" / f"scanvi_scores_{cfg.run_tag}.csv",
        index=False,
    )

    # Query latent for joint visualisation; carries predictions, NOT truth.
    query_hvg.obsm["X_scANVI"] = query_model.get_latent_representation()
    q_latent = ad.AnnData(
        X=query_hvg.obsm["X_scANVI"].copy(),
        obs=query_hvg.obs[
            [cfg["donor_key"], LABELS_KEY]
        ].copy(),
    )
    q_latent.obs["split"] = "query"
    q_latent.obs["predicted_label"] = predictions["predicted_label"].to_numpy()
    q_latent.obs["confidence"] = predictions["confidence"].to_numpy()
    q_latent.obs["entropy"] = predictions["entropy"].to_numpy()
    q_latent.write_h5ad(cfg.results_dir / f"query_mapped_{cfg.run_tag}.h5ad")

    # Append stage to run metadata.
    meta_path = cfg.results_dir / f"run_metadata_{cfg.run_tag}.json"
    run_meta = read_json(meta_path) if meta_path.exists() else {}
    run_meta.setdefault("stages", {})["query_mapping"] = query_summary
    write_json(meta_path, run_meta)

    print(f"Wrote {pred_path} ({len(predictions)} cells); method={METHOD_NAME}")


if __name__ == "__main__":
    main()
