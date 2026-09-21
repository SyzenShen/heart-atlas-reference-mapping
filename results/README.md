# results/

Small text/CSV/JSON artifacts are tracked; latent `.h5ad` files and model
weights are git-ignored. Everything carries a `main` or `smoke` tag and the
two are never mixed (`verify_outputs.py` enforces this).

## Top level

- `environment.json`, `data_audit.json` — environment and AnnData audits.
- `obs_columns.csv`, `donor_cell_counts.csv`, `donor_celltype_crosstab.csv`.
- `query_selection.csv` — donor ranking (label-blind `largest_donor_by_cells`
  rule; composition described for documentation only).
- `run_metadata_<tag>.json` — per-run provenance (versions, device, seed,
  timestamps, git info).
- `reference_scanvi_latent_<tag>.h5ad`, `query_mapped_<tag>.h5ad` — latent
  representations for joint UMAPs (git-ignored).

## `predictions/`

- `baseline_predictions_<tag>.csv`, `scanvi_predictions_<tag>.csv` — frozen
  predictions, **no true labels**.
- `scanvi_scores_<tag>.csv` — full per-class soft scores.
- `joined_eval_<method>_<tag>.csv` — predictions joined with sealed truth;
  written only by `evaluate.py` (evaluation-only artifact).

## `training/`

Training histories and summaries (timing, epochs completed, early-stop
status, parameter counts for the query update, PCA variance for the baseline).

## `metrics/`

Written by `scripts/evaluate.py`; schema in `docs/METRICS.md`:

- `summary_<tag>.csv` — accuracy / balanced accuracy / macro- and weighted-F1
  per method × scope;
- `per_class_<tag>.csv` — precision/recall/F1/support for every type;
- `confusion_<method>_<scope>_<counts|normalized>_<tag>.csv`;
- `confidence_coverage_<tag>.csv`, `rejected_composition_<tag>.csv`;
- `evaluation_metadata_<tag>.json`, `figures_manifest_<tag>.json`.
