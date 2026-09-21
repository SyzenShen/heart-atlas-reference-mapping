"""Deterministic donor-held-out split and query-label sealing.

The query donor is chosen by a pre-registered, deterministic rule
(``largest_eligible_donor``). It is selected BEFORE any model exists and can
never be swapped based on model performance.

Sealing design
--------------
* ``query_eval_labels_<tag>.csv`` -- only ``cell_id`` + ``true_cell_type``;
  read by the evaluation stage alone.
* The model-facing query AnnData has the author annotation column removed and
  ``labels_scanvi`` set to the single ``Unknown`` category for every cell.
* The split manifest records stable hashes of both cell-id sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from . import LABELS_KEY
from .config import Config
from .provenance import (dataset_fingerprint, read_json, stable_hash_strings,
                         write_json)

# A donor's "major" cell type: >= this fraction of its cells AND >= this many
# cells. Every major type must also exist among the other donors.
MAJOR_TYPE_MIN_FRACTION = 0.01
MAJOR_TYPE_MIN_CELLS = 10

SEALED_COLUMNS = ["cell_id", "true_cell_type"]
PREDICTION_FREEZE_NOTE = (
    "Query labels were sealed before model training and are read only by "
    "scripts/evaluate.py after predictions are frozen."
)


# --------------------------------------------------------------------------
# Query donor selection
# --------------------------------------------------------------------------
def donor_statistics(adata, cfg: Config) -> pd.DataFrame:
    donor_key = cfg["donor_key"]
    label_key = cfg["cell_type_key"]
    exclude = set(cfg.get("exclude_labels", []))

    df = adata.obs[[donor_key, label_key]].copy()
    df[label_key] = df[label_key].astype(str)
    df = df[~df[label_key].isin(exclude)]
    df = df[df[donor_key].notna()]

    rows = []
    donor_sets = {
        d: set(g[label_key].unique())
        for d, g in df.groupby(donor_key, observed=True)
    }
    for donor, g in df.groupby(donor_key, observed=True):
        counts = g[label_key].value_counts()
        n_cells = int(len(g))
        major = set(
            counts[
                (counts >= MAJOR_TYPE_MIN_CELLS)
                & (counts / n_cells >= MAJOR_TYPE_MIN_FRACTION)
            ].index
        )
        others_union: set[str] = set()
        for other_d, types in donor_sets.items():
            if other_d != donor:
                others_union |= types
        missing_in_reference = sorted(major - others_union)
        rows.append(
            {
                "donor_id": str(donor),
                "n_cells": n_cells,
                "n_cell_types": int(g[label_key].nunique()),
                "major_cell_types": ";".join(sorted(major)),
                "major_types_missing_in_other_donors": ";".join(
                    missing_in_reference
                ),
            }
        )
    stats = pd.DataFrame(rows).sort_values(
        ["n_cells", "donor_id"], ascending=[False, True]
    ).reset_index(drop=True)
    return stats


def select_query_donor(adata, cfg: Config) -> tuple[str, pd.DataFrame]:
    """Apply the deterministic eligibility rule and return (donor, table)."""
    stats = donor_statistics(adata, cfg)
    min_cells = int(cfg["minimum_query_cells"])
    min_types = int(cfg["minimum_query_cell_types"])

    stats["eligible"] = (
        (stats["n_cells"] >= min_cells)
        & (stats["n_cell_types"] >= min_types)
        & (stats["major_types_missing_in_other_donors"] == "")
    )
    eligible = stats[stats["eligible"]]
    if eligible.empty:
        raise RuntimeError(
            "No donor satisfies the eligibility rule. The rule may only be "
            "relaxed by editing the config thresholds before any model run, "
            "never by inspecting model performance."
        )
    # Rule: largest; ties broken by lexicographically smallest donor ID.
    top = eligible.sort_values(
        ["n_cells", "donor_id"], ascending=[False, True]
    ).iloc[0]
    return str(top["donor_id"]), stats


# --------------------------------------------------------------------------
# Split + sealing
# --------------------------------------------------------------------------
def _stratified_sample(obs: pd.DataFrame, n_max: int, label_col: str,
                       seed: int) -> pd.Index:
    if len(obs) <= n_max:
        return obs.index
    # proportional allocation per type, at least 1 cell per type
    frac = min(1.0, n_max / len(obs))
    parts = []
    leftover_budget = n_max
    rng = np.random.default_rng(seed)
    for _, idx in obs.groupby(label_col).groups.items():
        idx = pd.Index(idx)
        take = max(1, int(round(len(idx) * frac)))
        take = min(take, len(idx), leftover_budget)
        chosen = rng.choice(idx.to_numpy(), size=take, replace=False)
        parts.append(pd.Index(chosen))
        leftover_budget -= take
    out = obs.index.join(pd.Index(np.concatenate(parts)), how="inner")
    return out[:n_max]


def make_split(adata, cfg: Config, audit_info: dict[str, Any] | None = None):
    """Create the reference/query objects, seal query labels, save manifest.

    Returns additionally ``reference_full``: the complete (non-subsampled)
    reference. Reference-only learned preprocessing (HVGs) is fitted on the
    full reference even for smoke runs; smoke only subsamples the cells used
    for the fast training code path.
    """
    donor_key = cfg["donor_key"]
    label_key = cfg["cell_type_key"]
    unlabeled = cfg["unlabeled_category"]
    exclude = set(cfg.get("exclude_labels", []))

    query_donor, stats = select_query_donor(adata, cfg)

    work = adata.copy()
    work.obs[label_key] = work.obs[label_key].astype(str)
    keep = ~work.obs[label_key].isin(exclude)
    work = work[keep].copy()

    query_mask = (work.obs[donor_key].astype(str) == query_donor).to_numpy()
    reference_full = work[~query_mask].copy()
    query_full = work[query_mask].copy()
    reference = reference_full.copy()
    query = query_full.copy()

    if cfg.is_smoke:
        ref_cap = int(cfg.get("smoke_reference_max_cells", 600))
        q_cap = int(cfg.get("smoke_query_max_cells", 300))
        ref_idx = _stratified_sample(
            reference.obs, ref_cap, label_key, int(cfg["seed"])
        )
        q_idx = _stratified_sample(
            query.obs, q_cap, label_key, int(cfg["seed"]) + 1
        )
        reference = reference[ref_idx].copy()
        query = query[q_idx].copy()

    # Reference: author annotations copied into the model labels column.
    reference.obs[LABELS_KEY] = reference.obs[label_key].astype(str).values
    # Query: true annotation sealed away; model column is uniformly Unknown.
    sealed = pd.DataFrame(
        {
            "cell_id": query.obs_names.to_numpy(),
            "true_cell_type": query.obs[label_key].astype(str).values,
        }
    )

    query_model = query.copy()
    # Remove the broad annotation column AND any other cell-label-bearing
    # column (e.g. fine-grained 'cell_states') so no easily-misused copy of the
    # truth can travel into the model-facing object. Technical/QC metadata
    # (donor, sample, region, QC scores) is retained but unused by the model.
    from .data import LABEL_HINTS

    def _is_label_column(col: str) -> bool:
        c = col.lower()
        return any(h in c for h in LABEL_HINTS)

    drop_cols = [
        c for c in query_model.obs.columns
        if c != LABELS_KEY and _is_label_column(c)
    ]
    query_model.obs = query_model.obs.drop(columns=drop_cols)
    query_model.obs[LABELS_KEY] = unlabeled  # plain string; scvi encodes it

    # Persist artifacts ----------------------------------------------------
    cfg.data_processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.splits_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    reference.write_h5ad(cfg.reference_path, compression="gzip")
    query_model.write_h5ad(cfg.query_model_input_path, compression="gzip")
    sealed.to_csv(cfg.sealed_labels_path, index=False)
    stats.to_csv(cfg.results_dir / "query_selection.csv", index=False)

    ref_labels = sorted(reference.obs[LABELS_KEY].unique().tolist())
    manifest = {
        "run_tag": cfg.run_tag,
        "query_donor": query_donor,
        "donor_key": donor_key,
        "batch_key": cfg["batch_key"],
        "cell_type_key": label_key,
        "labels_key": LABELS_KEY,
        "counts_layer": cfg["counts_layer"],
        "unlabeled_category": unlabeled,
        "seed": int(cfg["seed"]),
        "selection_rule": cfg["query_selection_rule"],
        "selection_rule_detail": {
            "minimum_query_cells": int(cfg["minimum_query_cells"]),
            "minimum_query_cell_types": int(cfg["minimum_query_cell_types"]),
            "major_type_min_fraction": MAJOR_TYPE_MIN_FRACTION,
            "major_type_min_cells": MAJOR_TYPE_MIN_CELLS,
            "tie_break": "lexicographically smallest donor ID",
        },
        "reference_donor_ids": sorted(
            reference.obs[donor_key].astype(str).unique().tolist()
        ),
        "n_reference_cells": int(reference.n_obs),
        "n_query_cells": int(query_model.n_obs),
        "reference_cell_ids_sha256": stable_hash_strings(reference.obs_names),
        "query_cell_ids_sha256": stable_hash_strings(query_model.obs_names),
        "reference_label_set": ref_labels,
        "dataset": cfg["dataset"],
        "dataset_fingerprint": dataset_fingerprint(adata),
        "label_sealing_note": PREDICTION_FREEZE_NOTE,
        "sealed_labels_file": str(cfg.sealed_labels_path.name),
    }
    write_json(cfg.split_manifest_path, manifest)
    return reference, query_model, sealed, manifest, stats, reference_full


def load_split(cfg: Config):
    """Read a previously frozen split from disk."""
    reference = ad.read_h5ad(cfg.reference_path)
    query = ad.read_h5ad(cfg.query_model_input_path)
    sealed = pd.read_csv(cfg.sealed_labels_path)
    manifest = read_json(cfg.split_manifest_path)
    return reference, query, sealed, manifest


# --------------------------------------------------------------------------
# Sealing / leakage assertions (used by scripts and tests)
# --------------------------------------------------------------------------
class LeakageError(AssertionError):
    pass


def assert_no_overlap(reference, query, donor_key: str) -> None:
    ref_donors = set(reference.obs[donor_key].astype(str).unique())
    q_donors = set(query.obs[donor_key].astype(str).unique())
    if ref_donors & q_donors:
        raise LeakageError(f"Donor overlap between ref/query: {ref_donors & q_donors}")
    overlap = set(reference.obs_names) & set(query.obs_names)
    if overlap:
        raise LeakageError(f"{len(overlap)} overlapping cell IDs between ref/query")


def assert_query_sealed(query, cfg: Config) -> None:
    """The model-facing query object must carry no usable true labels."""
    from .data import LABEL_HINTS

    label_key = cfg["cell_type_key"]
    unlabeled = cfg["unlabeled_category"]
    if label_key in query.obs.columns:
        raise LeakageError(
            f"True label column '{label_key}' must be removed from query model "
            "input."
        )
    leftover = [
        c for c in query.obs.columns
        if c != LABELS_KEY and any(h in c.lower() for h in LABEL_HINTS)
    ]
    if leftover:
        raise LeakageError(
            f"Query model input still contains label-like columns: {leftover}"
        )
    if LABELS_KEY not in query.obs.columns:
        raise LeakageError(f"Query is missing required labels column {LABELS_KEY}")
    values = set(query.obs[LABELS_KEY].astype(str).unique())
    if values != {unlabeled}:
        raise LeakageError(
            f"Query labels must all be '{unlabeled}', found {values}"
        )


def assert_sealed_labels_match(query, cfg: Config) -> pd.DataFrame:
    sealed = pd.read_csv(cfg.sealed_labels_path)
    if list(sealed.columns) != SEALED_COLUMNS:
        raise LeakageError(
            f"Sealed labels file must have columns {SEALED_COLUMNS}, "
            f"got {list(sealed.columns)}"
        )
    if sealed["cell_id"].duplicated().any():
        raise LeakageError("Duplicate cell_id in sealed labels")
    if set(sealed["cell_id"]) != set(query.obs_names):
        raise LeakageError("Sealed label cell IDs do not match query cell IDs")
    return sealed
