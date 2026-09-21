"""Data loading, raw-count validation and full AnnData auditing.

No model training may happen before :func:`validate_counts` confirms where the
raw integer UMI counts live (``adata.X`` vs a layer). The loader runs with
``run_setup_anndata=False`` so we audit the untouched object first.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Column-name fragments used only to *report* candidate metadata fields.
# Final key selection is human-reviewed (docs/DATA_DICTIONARY.md) and pinned
# in the YAML config; these heuristics never make scientific choices silently.
DONOR_HINTS = ("donor", "patient", "individual", "subject", "sample")
BATCH_HINTS = ("batch", "donor", "sample", "channel", "experiment", "tech")
LABEL_HINTS = ("cell_type", "celltype", "cell type", "annotation", "ann_",
               "cell_state", "cluster", "identity", "label")
NUISANCE_LABELS = {"doublets", "NotAssigned", "unassigned", "Unknown", ""}


def load_heart_dataset(save_path: str, remove_nuisance_clusters: bool = True):
    """Load the 20k subsampled Human Heart Cell Atlas via scvi-tools.

    ``run_setup_anndata=False`` keeps the AnnData in its on-disk state for the
    audit; model scripts run setup themselves at the appropriate stage.
    """
    import scvi

    # NOTE: scvi-tools 1.3.3's signature is
    # heart_cell_atlas_subsampled(save_path, remove_nuisance_clusters) and it
    # does NOT run setup_anndata. In newer versions (>=1.4, stable docs checked
    # 2026-09-21) there is an additional run_setup_anndata=True parameter;
    # pass run_setup_anndata=False there if upgrading.
    adata = scvi.data.heart_cell_atlas_subsampled(
        save_path=save_path,
        remove_nuisance_clusters=remove_nuisance_clusters,
    )
    return adata


def _matrix_summary(matrix) -> dict[str, Any]:
    import scipy.sparse as sp

    summary: dict[str, Any] = {
        "type": type(matrix).__qualname__,
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "is_sparse": sp.issparse(matrix),
    }
    if sp.issparse(matrix):
        flat = matrix.data
        summary["sparse_format"] = matrix.getformat()
        summary["n_nonzero"] = int(matrix.nnz)
        summary["density"] = float(matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
    else:
        flat = np.asarray(matrix).ravel()
        summary["n_nonzero"] = int(np.count_nonzero(matrix))
    flat = np.asarray(flat)
    if flat.size:
        sample_n = min(50_000, flat.size)
        rng = np.random.default_rng(0)
        idx = rng.choice(flat.size, size=sample_n, replace=False)
        sample = flat[idx].astype(float)
        summary["sampled_n"] = int(sample_n)
        summary["min"] = float(np.nanmin(flat))
        summary["max"] = float(np.nanmax(flat))
        summary["n_negative"] = int(np.sum(flat < 0))
        nz = sample[sample != 0]
        if nz.size:
            max_deviation = float(np.max(np.abs(nz - np.round(nz))))
            frac_non_integer = float(np.mean(np.abs(nz - np.round(nz)) > 1e-4))
        else:
            max_deviation = 0.0
            frac_non_integer = 0.0
        summary["sampled_max_round_deviation"] = max_deviation
        summary["sampled_fraction_non_integer"] = frac_non_integer
        summary["looks_like_integer_counts"] = (
            summary["n_negative"] == 0 and max_deviation < 1e-4
        )
    return summary


def validate_counts(adata, layer: str | None) -> dict[str, Any]:
    """Locate and verify raw counts.

    Raises ``RuntimeError`` when the requested location does not hold
    non-negative, near-integer values. Log-normalised values must never be fed
    to the negative-binomial model.
    """
    if layer is not None:
        if layer not in adata.layers:
            raise RuntimeError(
                f"Requested counts layer '{layer}' not found; available layers: "
                f"{list(adata.layers)}"
            )
        matrix = adata.layers[layer]
        location = f"layers['{layer}']"
    else:
        matrix = adata.X
        location = "X"

    summary = _matrix_summary(matrix)
    summary["location"] = location
    if not summary.get("looks_like_integer_counts", False):
        raise RuntimeError(
            f"Matrix at {location} failed integer/non-negative count validation "
            f"(min={summary.get('min')}, max={summary.get('max')}, "
            f"fraction_non_integer={summary.get('sampled_fraction_non_integer')})."
        )
    return summary


def obs_column_summary(adata) -> pd.DataFrame:
    rows = []
    for col in adata.obs.columns:
        s = adata.obs[col]
        n_unique = s.nunique(dropna=True)
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "n_unique": int(n_unique),
            "n_missing": int(s.isna().sum()),
        }
        if n_unique <= 60:
            row["example_values"] = "|".join(
                sorted(str(v) for v in s.dropna().unique())[:60]
            )
        else:
            row["example_values"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def _value_counts_csv(adata, col, path) -> None:
    adata.obs[col].value_counts(dropna=False).to_csv(path, header=["count"])


def audit_adata(adata) -> dict[str, Any]:
    """Comprehensive structural audit (see instruction section 5)."""
    audit: dict[str, Any] = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "X": _matrix_summary(adata.X),
        "layers": {name: _matrix_summary(mat) for name, mat in adata.layers.items()},
        "raw_present": adata.raw is not None,
        "obsm": {k: list(np.asarray(v).shape) for k, v in adata.obsm.items()},
        "varm": {k: list(np.asarray(v).shape) for k, v in adata.varm.items()},
        "uns_keys": list(adata.uns.keys()),
        "obs_columns": list(adata.obs.columns),
        "var_columns": list(adata.var.columns),
        "obs_missing_total": int(adata.obs.isna().sum().sum()),
        "obs_names_unique": bool(adata.obs_names.is_unique),
        "var_names_unique": bool(adata.var_names.is_unique),
    }
    if adata.raw is not None:
        audit["raw"] = {
            "shape": list(adata.raw.shape),
            "X": _matrix_summary(adata.raw.X),
            "var_columns": list(adata.raw.var.columns),
        }

    # Candidate metadata columns (report only).
    def matches(col, hints):
        c = col.lower()
        return any(h in c for h in hints)

    audit["candidate_donor_columns"] = [
        c for c in adata.obs.columns if matches(c, DONOR_HINTS)
    ]
    audit["candidate_batch_columns"] = [
        c for c in adata.obs.columns if matches(c, BATCH_HINTS)
    ]
    audit["candidate_label_columns"] = [
        c for c in adata.obs.columns if matches(c, LABEL_HINTS)
    ]

    # Per-column detail and donor/cell-type value counts are exposed separately
    # through obs_column_summary / write_audit_tables.
    categorical = {}
    for col in adata.obs.columns:
        n_unique = adata.obs[col].nunique(dropna=True)
        categorical[col] = {"n_unique": int(n_unique)}
        if n_unique <= 100:
            categorical[col]["values"] = {
                str(k): int(v)
                for k, v in adata.obs[col].value_counts(dropna=False).items()
            }
    audit["categorical_columns"] = categorical
    return audit


def write_audit_tables(adata, results_dir) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    obs_column_summary(adata).to_csv(results_dir / "obs_columns.csv", index=False)


def confirm_keys(adata, cfg) -> dict[str, Any]:
    """Validate that the config-selected keys exist and report their support.

    This deliberately fails loudly rather than silently guessing fields.
    """
    info: dict[str, Any] = {}
    for name in ("donor_key", "batch_key", "cell_type_key"):
        key = cfg[name]
        if key not in adata.obs.columns:
            raise RuntimeError(
                f"Configured {name}='{key}' not present in adata.obs "
                f"(columns: {list(adata.obs.columns)})"
            )
        info[name] = {
            "key": key,
            "n_unique": int(adata.obs[key].nunique(dropna=True)),
            "n_missing": int(adata.obs[key].isna().sum()),
        }
    layer = cfg["counts_layer"]
    if layer not in ("counts",) and layer not in adata.layers:
        raise RuntimeError(f"counts_layer '{layer}' not in layers")
    if layer in adata.layers:
        info["counts_layer"] = {"key": layer, "present": True}
    else:
        info["counts_layer"] = {
            "key": None,
            "present": False,
            "note": "Falling back to adata.X, which must itself hold raw counts.",
        }
    return info


def select_reference_hvgs(reference, cfg: Config) -> list[str]:
    """Select HVGs from REFERENCE ONLY (count-based Seurat v3 flavor).

    The query never touches this function. ``batch_key`` aware HVG selection is
    run on the raw count layer. Gene ordering returned here is the single
    feature definition shared by both the baseline and the scVI/scANVI models.
    """
    import scanpy as sc

    layer = cfg["counts_layer"]
    if layer not in reference.layers:
        raise RuntimeError(
            f"counts layer '{layer}' missing on reference when selecting HVGs"
        )
    validate_counts(reference, layer)
    sc.pp.highly_variable_genes(
        reference,
        n_top_genes=int(cfg["n_hvg"]),
        flavor=str(cfg.get("hvg_flavor", "seurat_v3")),
        layer=layer,
        batch_key=cfg["batch_key"],
        subset=False,
    )
    hvgs = reference.var_names[reference.var["highly_variable"]].astype(str).tolist()
    if len(hvgs) == 0:
        raise RuntimeError("HVG selection returned 0 genes")
    return hvgs


def apply_reference_genes(adata, hvgs: list[str], counts_layer: str,
                          fill_missing: bool = False):
    """Subset (and optionally zero-pad) an AnnData to the reference gene set.

    For model mapping we prefer scvi's own ``prepare_query_anndata`` (it pads
    missing genes and validates order). The baseline uses this helper with
    ``fill_missing=True``.
    """
    import anndata as ad
    import scipy.sparse as sp

    present = [g for g in hvgs if g in adata.var_names]
    if fill_missing and len(present) < len(hvgs):
        missing = [g for g in hvgs if g not in adata.var_names]
        n_missing = len(missing)
        zero_counts = sp.csr_matrix(
            (adata.n_obs, n_missing), dtype=adata.X.dtype
        )
        extra = ad.AnnData(
            X=zero_counts,
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=pd.Index(missing)),
        )
        if counts_layer in adata.layers:
            extra.layers[counts_layer] = zero_counts.copy()
        aligned = ad.concat(
            [adata[:, present].copy(), extra], axis=1, join="outer", fill_value=0
        )
        aligned = aligned[:, hvgs].copy()
    else:
        aligned = adata[:, present].copy()
        aligned = aligned[:, hvgs].copy() if set(present) == set(hvgs) else aligned
    return aligned


def check_nuisance_labels_absent(adata, cell_type_key: str) -> dict[str, Any]:
    values = set(adata.obs[cell_type_key].astype(str).unique())
    found = sorted(values & {"doublets", "NotAssigned"})
    return {"nuisance_labels_present": found, "clean": len(found) == 0}
