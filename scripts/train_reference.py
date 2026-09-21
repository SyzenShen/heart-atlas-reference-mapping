#!/usr/bin/env python3
"""Train scVI then scANVI on the reference donors and save both models.

Pre-training guards
-------------------
* reference/query donor + cell-id disjointness;
* query model input has no true label column and is uniformly Unknown;
* HVG list exists and is reference-derived;
* counts layer verified as non-negative integers.

The sealed query labels file is never opened here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heartmap import LABELS_KEY  # noqa: E402
from heartmap.config import load_config  # noqa: E402
from heartmap.data import validate_counts  # noqa: E402
from heartmap.models import (METHOD_NAME, save_history, subset_hvg,
                             train_scanvi_reference, train_scvi_reference)  # noqa: E402
from heartmap.provenance import collect_environment, read_json, write_json  # noqa: E402
from heartmap.split import (LeakageError, assert_no_overlap, assert_query_sealed,
                            load_model_split)  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    reference, query, manifest = load_model_split(cfg)
    try:
        assert_no_overlap(reference, query, cfg["donor_key"])
        assert_query_sealed(query, cfg)
    except LeakageError as exc:  # hard stop, never train past a leakage failure
        raise SystemExit(f"Leakage guard failed; aborting training: {exc}")

    if manifest["run_tag"] != cfg.run_tag:
        raise SystemExit("Config run_tag does not match frozen split manifest")
    if not cfg.hvg_path.exists():
        raise SystemExit("HVG list missing; run prepare_split.py first")

    # HVG-subset the reference (frozen gene order); keep raw counts untouched.
    ref_hvg = subset_hvg(reference, cfg)
    validate_counts(ref_hvg, cfg["counts_layer"])
    labels = set(ref_hvg.obs[LABELS_KEY].astype(str).unique())
    if cfg["unlabeled_category"] in labels:
        raise SystemExit(
            f"Reference must not contain the unlabeled category "
            f"'{cfg['unlabeled_category']}'"
        )

    env = collect_environment()
    models_root = cfg.models_dir
    models_root.mkdir(exist_ok=True)
    train_dir = cfg.results_dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "run_tag": cfg.run_tag,
        "experiment_name": cfg["experiment_name"],
        "seed": int(cfg["seed"]),
        "query_donor": manifest["query_donor"],
        "reference_donor_ids": manifest["reference_donor_ids"],
        "n_reference_cells": manifest["n_reference_cells"],
        "n_query_cells_sealed": manifest["n_query_cells"],
        "donor_key": cfg["donor_key"],
        "batch_key": cfg["batch_key"],
        "cell_type_key": cfg["cell_type_key"],
        "labels_key": LABELS_KEY,
        "counts_layer": cfg["counts_layer"],
        "n_hvg": int(len(cfg.hvg_path.read_text().splitlines())),
        "hvg_file": cfg.hvg_path.name,
        "config_file": str(cfg.path.name if cfg.path else None),
        "environment": env,
        "stages": {},
        "method": METHOD_NAME,
    }

    # ---- scVI -------------------------------------------------------------
    print("Training scVI reference ...")
    scvi_model, scvi_summary = train_scvi_reference(ref_hvg, cfg)
    scvi_path = models_root / f"scvi_reference_{cfg.run_tag}"
    scvi_model.save(str(scvi_path), overwrite=True)
    save_history(scvi_model, train_dir / f"scvi_history_{cfg.run_tag}.csv")
    write_json(train_dir / f"scvi_summary_{cfg.run_tag}.json", scvi_summary)
    run_meta["stages"]["scvi"] = scvi_summary
    print(f"scVI trained in {scvi_summary['wallclock_seconds']}s -> {scvi_path}")

    ref_hvg.obsm["X_scVI"] = scvi_model.get_latent_representation()
    ref_latent = ad.AnnData(
        X=ref_hvg.obsm["X_scVI"].copy(),
        obs=ref_hvg.obs[
            [cfg["donor_key"], cfg["cell_type_key"], LABELS_KEY]
        ].copy(),
    )
    ref_latent.obs["split"] = "reference"
    ref_latent.write_h5ad(cfg.results_dir / f"reference_latent_{cfg.run_tag}.h5ad")

    # ---- scANVI -----------------------------------------------------------
    print("Training scANVI reference ...")
    scanvi_model, scanvi_summary = train_scanvi_reference(
        scvi_model, ref_hvg, cfg
    )
    scanvi_path = models_root / f"scanvi_reference_{cfg.run_tag}"
    scanvi_model.save(str(scanvi_path), overwrite=True)
    save_history(scanvi_model, train_dir / f"scanvi_history_{cfg.run_tag}.csv")
    write_json(train_dir / f"scanvi_summary_{cfg.run_tag}.json", scanvi_summary)
    run_meta["stages"]["scanvi"] = scanvi_summary
    print(
        f"scANVI trained in {scanvi_summary['wallclock_seconds']}s -> {scanvi_path}"
    )

    ref_hvg.obsm["X_scANVI"] = scanvi_model.get_latent_representation()
    ref_scanvi_latent = ad.AnnData(
        X=ref_hvg.obsm["X_scANVI"].copy(),
        obs=ref_hvg.obs[
            [cfg["donor_key"], cfg["cell_type_key"], LABELS_KEY]
        ].copy(),
    )
    ref_scanvi_latent.obs["split"] = "reference"
    ref_scanvi_latent.write_h5ad(
        cfg.results_dir / f"reference_scanvi_latent_{cfg.run_tag}.h5ad"
    )

    write_json(cfg.results_dir / f"run_metadata_{cfg.run_tag}.json", run_meta)
    print("Reference training complete.")


if __name__ == "__main__":
    main()
