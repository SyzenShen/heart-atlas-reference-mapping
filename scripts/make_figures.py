#!/usr/bin/env python3
"""Generate all figures from frozen predictions + evaluation metrics.

Reads sealed-truth-joined evaluation tables (created by evaluate.py) and
saved latents; never re-trains anything. Smoke figures go to figures/smoke/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heartmap.baseline import METHOD_NAME as BASELINE_METHOD, _normalized_matrix, load_hvgs  # noqa: E402
from heartmap.config import load_config  # noqa: E402
from heartmap.models import METHOD_NAME as SCANVI_METHOD  # noqa: E402
from heartmap.plotting import (joint_umap, plot_confidence_by_cell_type,
                               plot_confidence_correctness, plot_confusion,
                               plot_coverage, plot_f1_vs_abundance,
                               plot_method_comparison, plot_umap_scatter)  # noqa: E402
from heartmap.provenance import sha256_file, write_json  # noqa: E402
from heartmap.split import load_split  # noqa: E402


def _fig_dir(cfg) -> Path:
    d = cfg.figures_dir if not cfg.is_smoke else cfg.figures_dir / "smoke"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_joint_latent(cfg) -> ad.AnnData:
    ref = ad.read_h5ad(cfg.results_dir / f"reference_scanvi_latent_{cfg.run_tag}.h5ad")
    qry = ad.read_h5ad(cfg.results_dir / f"query_mapped_{cfg.run_tag}.h5ad")
    # Attach post-evaluation truth to the query for the evaluation-only panel.
    joined = pd.read_csv(
        cfg.results_dir / "predictions" / f"joined_eval_{SCANVI_METHOD}_{cfg.run_tag}.csv"
    ).set_index("cell_id")
    qry.obs["cell_type_truth_eval"] = qry.obs_names.map(joined["true_cell_type"])
    qry.obs["cell_type_truth_eval"] = qry.obs["cell_type_truth_eval"].astype("category")

    joint = ad.concat([qry, ref], axis=0, join="outer", fill_value=np.nan,
                      keys=["query", "reference"], index_unique="-")
    joint.obsm["X_scANVI"] = np.vstack([qry.X, ref.X])
    # Unified truth column: reference author labels + query post-hoc truth.
    ref_truth = joint.obs[cfg["cell_type_key"]].astype("object")
    q_truth = joint.obs["cell_type_truth_eval"].astype("object")
    joint.obs["truth_eval_only"] = q_truth.combine_first(ref_truth).astype("category")
    return joint


def build_raw_pca_umap(cfg) -> ad.AnnData:
    """Reference-fitted PCA on normalized HVGs (visualisation only)."""
    from sklearn.decomposition import PCA

    reference, query, _, _ = load_split(cfg)
    hvgs = load_hvgs(cfg)
    target_sum = float(cfg["baseline"].get("target_sum", 1e4))
    x_ref, ref_obs = _normalized_matrix(
        reference, hvgs, cfg["counts_layer"], target_sum
    )
    x_qry, q_obs = _normalized_matrix(
        query, hvgs, cfg["counts_layer"], target_sum
    )
    pca = PCA(n_components=int(cfg["baseline"]["n_components"]),
              random_state=int(cfg["seed"]))
    z_ref = pca.fit_transform(x_ref)
    z_qry = pca.transform(x_qry)
    joint = ad.AnnData(
        X=np.zeros((len(z_ref) + len(z_qry), 1)),
        obs=pd.concat(
            [
                q_obs.assign(split="query"),
                ref_obs.assign(split="reference"),
            ]
        ),
    )
    joint.obsm["X_pca_refit_reference"] = np.vstack([z_qry, z_ref])
    joint_umap(joint, "X_pca_refit_reference", cfg)
    return joint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--skip-umap", action="store_true",
                        help="skip the comparatively slow UMAP panels")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tag = cfg.run_tag
    mdir = cfg.results_dir / "metrics"
    figd = _fig_dir(cfg)

    summary = pd.read_csv(mdir / f"summary_{tag}.csv")
    per_class = pd.read_csv(mdir / f"per_class_{tag}.csv")
    coverage = pd.read_csv(mdir / f"confidence_coverage_{tag}.csv")
    meta = pd.io.json.read_json(
        mdir / f"evaluation_metadata_{tag}.json", typ="series"
    )
    query_donor = str(meta["query_donor"])
    n_cells = int(summary[summary["scope"] == "all"]["n_query_cells"].iloc[0])

    # ---- Metric figures ----------------------------------------------------
    plot_method_comparison(
        summary, query_donor, n_cells, figd / "method_comparison.png",
        dpi=int(cfg["figure_dpi"]),
    )
    plot_f1_vs_abundance(
        per_class, query_donor, figd / "f1_vs_cell_abundance.png",
        dpi=int(cfg["figure_dpi"]),
    )
    for method, label in ((SCANVI_METHOD, "scANVI"), (BASELINE_METHOD, "PCA+kNN")):
        cov = coverage[coverage["method"] == method]
        plot_coverage(
            cov, "accuracy_on_retained", "Accuracy on retained cells",
            f"{label}: accuracy vs coverage — donor {query_donor}",
            figd / f"coverage_accuracy_{method}.png", dpi=int(cfg["figure_dpi"]),
        )
        plot_coverage(
            cov, "macro_f1_on_retained", "Macro-F1 on retained cells",
            f"{label}: macro-F1 vs coverage — donor {query_donor}",
            figd / f"coverage_macro_f1_{method}.png", dpi=int(cfg["figure_dpi"]),
        )
    # Canonical names required by the brief point to the scANVI curves.
    import shutil
    shutil.copyfile(
        figd / f"coverage_accuracy_{SCANVI_METHOD}.png",
        figd / "coverage_accuracy.png",
    )
    shutil.copyfile(
        figd / f"coverage_macro_f1_{SCANVI_METHOD}.png",
        figd / "coverage_macro_f1.png",
    )

    for scope in ("closed_set", "all"):
        for method, label in ((SCANVI_METHOD, "scanvi"), (BASELINE_METHOD, "baseline")):
            cm = pd.read_csv(
                mdir / f"confusion_{method}_{scope}_counts_{tag}.csv", index_col=0
            )
            cm_n = pd.read_csv(
                mdir / f"confusion_{method}_{scope}_normalized_{tag}.csv", index_col=0
            )
            plot_confusion(
                cm,
                f"{label} confusion ({scope}, counts) — donor {query_donor}, n={n_cells}",
                figd / f"confusion_{label}_{scope}_counts.png",
                normalized=False, dpi=int(cfg["figure_dpi"]),
            )
            plot_confusion(
                cm_n,
                f"{label} confusion ({scope}, row-normalized) — donor {query_donor}",
                figd / f"confusion_{label}_{scope}_normalized.png",
                normalized=True, dpi=int(cfg["figure_dpi"]),
            )
    shutil.copyfile(
        figd / "confusion_scanvi_closed_set_normalized.png",
        figd / "confusion_scanvi_normalized.png",
    )

    joined_scanvi = pd.read_csv(
        cfg.results_dir / "predictions" / f"joined_eval_{SCANVI_METHOD}_{tag}.csv"
    )
    plot_confidence_correctness(
        joined_scanvi, query_donor,
        figd / "confidence_correct_vs_incorrect.png", dpi=int(cfg["figure_dpi"]),
    )
    plot_confidence_by_cell_type(
        joined_scanvi, query_donor, figd / "confidence_by_cell_type.png",
        dpi=int(cfg["figure_dpi"]),
    )

    written = sorted(p.name for p in figd.glob("*.png"))

    if not args.skip_umap:
        # ---- scANVI joint latent UMAPs ------------------------------------
        joint = build_joint_latent(cfg)
        joint_umap(joint, "X_scANVI", cfg)
        plot_umap_scatter(
            joint, cfg["donor_key"],
            f"scANVI latent by donor — query {query_donor} (n={n_cells})",
            figd / "umap_scanvi_by_donor.png", dpi=int(cfg["figure_dpi"]),
        )
        plot_umap_scatter(
            joint, "split",
            "Reference vs held-out query in scANVI latent",
            figd / "umap_scanvi_by_split.png", dpi=int(cfg["figure_dpi"]),
        )
        plot_umap_scatter(
            joint, "truth_eval_only",
            "True cell types (post-evaluation only; query labels never used in "
            f"training) — donor {query_donor}",
            figd / "umap_scanvi_by_cell_type.png", dpi=int(cfg["figure_dpi"]),
        )
        # Predictions panel: reference faded, query colored by prediction.
        import scanpy as sc
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 7))
        ref_mask = joint.obs["split"].astype(str) == "reference"
        sc.pl.embedding(
            joint[ref_mask].copy(), basis="X_umap", ax=ax, show=False,
            size=6, color=None, frameon=False, alpha=0.25,
            title=f"Query predictions in scANVI latent — donor {query_donor}",
        )
        for coll in ax.collections:
            coll.set_facecolor("lightgrey")
            coll.set_edgecolor("none")
        sc.pl.embedding(
            joint[~ref_mask].copy(), basis="X_umap", ax=ax, show=False,
            size=9, color="predicted_label", frameon=False, alpha=0.9,
        )
        fig.savefig(figd / "umap_scanvi_query_predictions.png",
                    dpi=int(cfg["figure_dpi"]), bbox_inches="tight")
        plt.close(fig)

        # ---- Reference-fitted PCA "raw" UMAP ------------------------------
        raw_joint = build_raw_pca_umap(cfg)
        plot_umap_scatter(
            raw_joint, cfg["donor_key"],
            f"Before integration: reference-fitted PCA UMAP by donor — query "
            f"{query_donor} (visualization only)",
            figd / "umap_raw_by_donor.png", dpi=int(cfg["figure_dpi"]),
        )
        written = sorted(p.name for p in figd.glob("*.png"))

    write_json(
        mdir / f"figures_manifest_{tag}.json",
        {
            "run_tag": tag,
            "figure_dir": str(figd.relative_to(cfg.root)),
            "figures": [
                {"file": n, "sha256": sha256_file(figd / n)} for n in written
            ],
            "derived_from": [
                f"summary_{tag}.csv", f"per_class_{tag}.csv",
                f"confidence_coverage_{tag}.csv",
                f"joined_eval_{SCANVI_METHOD}_{tag}.csv",
            ],
            "note": "UMAPs are visual aids only; quantitative claims use metrics.",
        },
    )
    print(f"Wrote {len(written)} figures to {figd.relative_to(cfg.root)}/")


if __name__ == "__main__":
    main()
