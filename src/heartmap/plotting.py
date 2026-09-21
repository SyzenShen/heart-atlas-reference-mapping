"""Figure generation.

Rules enforced here:
* every quantitative claim is backed by computed metrics / latents;
* UMAPs are visual aids only (fixed seed, model latent or reference-fit PCA),
  never the sole evidence of batch correction or superiority;
* axes are labelled with units, query donor and cell counts are annotated;
* no 3D plots, no truncated axes to exaggerate differences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def _save(fig, path: str | Path, dpi: int = 200) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm: pd.DataFrame, title: str, path: str | Path,
                   normalized: bool, dpi: int = 200) -> None:
    n = len(cm.columns)
    data = cm.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(8.5, 0.95 * n + 2.5),
                                    max(6.0, 0.62 * n + 2.0)))
    if not normalized:
        data = np.nan_to_num(data, nan=0.0).astype(int)
        fmt = "d"
    else:
        fmt = ".2f"
    vmax = 1.0 if normalized else None
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        annot_kws={"size": 9 if normalized else 10},
        cmap="Blues",
        xticklabels=cm.columns,
        yticklabels=cm.index,
        vmin=0,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "row-normalized rate" if normalized else "cells"},
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.tick_params(axis="y", rotation=0)
    _save(fig, path, dpi)


def plot_method_comparison(summary: pd.DataFrame, query_donor: str,
                           n_cells: int, path: str | Path, dpi: int = 200) -> None:
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    df = summary[summary["scope"] == "closed_set"].melt(
        id_vars=["method"], value_vars=metrics, var_name="metric", value_name="value"
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=df, x="metric", y="value", hue="method", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (closed-set, query donor %s)" % query_donor)
    ax.set_xlabel("")
    ax.set_title(f"Method comparison — held-out donor {query_donor} (n={n_cells})")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=9, padding=2)
    ax.legend(title="Method", frameon=True)
    _save(fig, path, dpi)


def plot_f1_vs_abundance(per_class: pd.DataFrame, query_donor: str,
                         path: str | Path, dpi: int = 200) -> None:
    df = per_class[per_class["scope"] == "closed_set"].copy()
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.scatterplot(
        data=df, x="support", y="f1", hue="method", style="method",
        s=130, ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlim(max(df["support"].min() * 0.7, 0.5), df["support"].max() * 1.4)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Query cell-type abundance (n cells, log scale)")
    ax.set_ylabel("Per-class F1 (closed-set)")
    ax.set_title(f"Per-cell-type F1 vs abundance — donor {query_donor}")
    # One label per cell type (both methods share the query support), placed
    # at the better of the two F1 points with alternating vertical offsets to
    # reduce collisions in the dense high-abundance / high-F1 region.
    yoffsets = [-14, 8, -2]
    for i, (label, g) in enumerate(df.groupby("label")):
        row = g.loc[g["f1"].idxmax()]
        ax.annotate(
            str(label), (row["support"], row["f1"]),
            fontsize=8.5, xytext=(6, yoffsets[i % len(yoffsets)]),
            textcoords="offset points",
        )
    ax.axhline(df["f1"].mean(), color="grey", ls="--", lw=1,
               label="macro mean")
    _save(fig, path, dpi)


def plot_coverage(coverage: pd.DataFrame, y_col: str, ylabel: str,
                  title: str, path: str | Path, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.lineplot(data=coverage, x="coverage", y=y_col, marker="o", ax=ax)
    for _, r in coverage.iterrows():
        ax.annotate(f"t={r['threshold']:.1f}",
                    (r["coverage"], r[y_col]), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Coverage (fraction of query cells retained)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save(fig, path, dpi)


def plot_confidence_correctness(joined: pd.DataFrame, query_donor: str,
                                path: str | Path, dpi: int = 200) -> None:
    df = joined.copy()
    df["correct"] = np.where(
        df["predicted_label"].astype(str) == df["true_cell_type"].astype(str),
        "correct", "incorrect",
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.histplot(
        data=df, x="confidence", hue="correct", bins=40, stat="density",
        common_norm=False, alpha=0.55, ax=ax,
    )
    ax.set_xlabel("Prediction confidence score (not a calibrated probability)")
    ax.set_ylabel("Density within group")
    ax.set_title(f"Confidence of correct vs incorrect predictions — {query_donor}")
    _save(fig, path, dpi)


def plot_confidence_by_cell_type(joined: pd.DataFrame, query_donor: str,
                                 path: str | Path, dpi: int = 200) -> None:
    df = joined.copy()
    order = (
        df.groupby("true_cell_type")["confidence"].median().sort_values().index
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(
        data=df, y="true_cell_type", x="confidence", order=order,
        color="#9ecae1", fliersize=2, ax=ax,
    )
    ax.set_xlabel("Prediction confidence score")
    ax.set_ylabel("True cell type")
    ax.set_xlim(0, 1.02)
    ax.set_title(f"Confidence distribution by true cell type — {query_donor}")
    _save(fig, path, dpi)


# --------------------------------------------------------------------------
# UMAP helpers (visualization only; fixed seed)
# --------------------------------------------------------------------------
def joint_umap(adata, rep: str, cfg: Config, key_added: str = "X_umap") -> None:
    """Neighbours + UMAP on a fixed representation, fixed seed."""
    import scanpy as sc

    ucfg = cfg.get("umap", {})
    sc.pp.neighbors(
        adata,
        use_rep=rep,
        n_neighbors=int(ucfg.get("n_neighbors", 15)),
        random_state=int(ucfg.get("seed", 42)),
    )
    sc.tl.leiden(adata, key_added="_leiden_viz",
                 random_state=int(ucfg.get("seed", 42)))
    sc.tl.umap(
        adata,
        min_dist=float(ucfg.get("min_dist", 0.3)),
        random_state=int(ucfg.get("seed", 42)),
    )
    if "X_umap" in adata.obsm and key_added != "X_umap":
        adata.obsm[key_added] = adata.obsm["X_umap"]


def plot_umap_scatter(adata, color: str, title: str, path: str | Path,
                      palette: Any = None, dpi: int = 200,
                      groups_order: Any = None, size: float = 8.0) -> None:
    import scanpy as sc

    fig, ax = plt.subplots(figsize=(8.5, 7))
    sc.pl.embedding(
        adata, basis="X_umap", color=color, ax=ax, show=False, size=size,
        palette=palette, groups=groups_order, title=title, frameon=False,
    )
    _save(fig, path, dpi)
