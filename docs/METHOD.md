# Methods

A concise, reproducible description of the analysis. All numerical parameters
live in `configs/main.yaml` (the `smoke` config only exercises code paths).

## 1. Data

The dataset is the ~20k-cell subsampled **Human Heart Cell Atlas** distributed
through `scvi.data.heart_cell_atlas_subsampled` (scvi-tools 1.3.3,
`remove_nuisance_clusters=True`). After the loader removes doublet /
NotAssigned nuisance clusters, the object contains **18,641 cells × 26,662
genes** from **14 donors** (`D1–D7, D11, H2–H7`).

Audit (`results/data_audit.json`, `docs/DATA_DICTIONARY.md`) established that:

- `adata.X` is a CSR `float32` matrix of raw integer UMI counts
  (non-negative, non-zeros integer, range 1–2693);
- there is no `.raw` and no count layer on download.

The pipeline copies `X` into `layers["counts"]` and uses that layer
exclusively for count models. Labels are the 11 author-provided **broad** cell
types in `cell_type`. The 65-state `cell_states` column is future work and is
removed from query inputs.

## 2. Donor-held-out split

One complete donor is the query; every other donor is the reference. No
cell-level random split is ever performed.

The query donor is fixed by a pre-registered, **label-blind** deterministic
rule (`largest_donor_by_cells`):

1. cells must have a donor ID;
2. choose the donor with the most cells; ties resolve to the lexicographically
   smallest donor ID.

The `cell_type` column is **never read** during selection — the rule is
label-blind. Cell-type composition is computed *after* selection for
description only (see `results/query_selection.csv`). **D6** (3,009 cells,
all 11 types) is the query. The reference comprises the other 13 donors
(15,632 cells). The split, SHA-256 hashes of reference/query cell IDs, keys,
seed and software versions are stored in
`data/splits/split_manifest_main.json`.

## 3. Label sealing

- `data/splits/query_eval_labels_main.csv` holds `cell_id, true_cell_type` and
  is read **only** by `scripts/evaluate.py`, after predictions are frozen.
- The model-facing query AnnData has `cell_type`, `cell_states` and every other
  label-like column removed; `labels_scanvi` is the plain string `"Unknown"`
  for every query cell.
- `src/heartmap/split.py` raises `LeakageError` if a label column survives, if
  labels are not all `Unknown`, or if donor/cell IDs overlap. Tests enforce
  this statically (training scripts cannot import evaluation code or reference
  the sealed file) and dynamically.

## 4. Reference-only preprocessing

- **HVGs:** 2,000 genes with `scanpy.pp.highly_variable_genes(flavor="seurat_v3",
  batch_key="donor")`, computed on raw counts of the **reference only**. The
  ranked list is frozen to `data/processed/hvg_main.txt`. The smoke split
  reuses HVGs from the full reference.
- **Query alignment:** genes are reordered to the reference list; missing
  genes are zero-padded by `SCANVI.prepare_query_anndata`.
- No log-normalised data is ever passed to scVI/scANVI.

## 5. PCA + kNN baseline

The baseline deliberately shares the exact reference HVGs so the comparison
isolates the representation model:

1. library-size normalisation (`target_sum=1e4`) + `log1p`, applied
   independently to reference and query (a per-cell transform; parameters are
   fixed, not fit jointly);
2. PCA with 30 components, **fit on reference only**, query calls
   `transform`;
3. kNN classifier, k = 15, Euclidean distance, distance-weighted vote, fit on
   reference PCA coordinates and reference labels;
4. confidence = the neighbour-weighted vote fraction for the winning class;
   prediction entropy is also stored.

The HVG matrix is densified once for sklearn PCA (~15,632 × 2,000 ≈ 0.125 GB);
this is recorded in `results/training/baseline_summary_main.json`.

## 6. scVI / scANVI reference models

Architecture follows the current official scvi-tools reference-mapping
tutorial (checked for 1.3.x):

- `SCVI(layer="counts", batch_key="donor", n_latent=30, n_hidden=128,
  n_layers=2, dropout_rate=0.2, use_layer_norm="both",
  use_batch_norm="none", encode_covariates=True, gene_likelihood="zinb")`;
- scVI training: up to 400 epochs, 90/10 train/validation split within
  reference cells, early stopping (patience 20, validation every 5 epochs),
  batch size 256;
- `SCANVI.from_scvi_model(..., labels_key="labels_scanvi",
  unlabeled_category="Unknown")`, up to 20 epochs,
  `n_samples_per_label=100`, early stopping patience 10.

The layer-norm / no-batch-norm / encoded-covariate configuration is the one
the scArches workflow requires for surgery on new batches.

## 7. scArches query mapping

1. `SCANVI.prepare_query_anndata(query, ref_model_dir)` aligns features;
2. `SCANVI.load_query_data(query, ref_model_dir)` rebuilds the model with
   frozen reference weights and fresh batch adaptation parameters;
3. query update: up to 100 epochs, `plan_kwargs={"weight_decay": 0.0}`
   (official recommendation; a config validator rejects any non-zero value),
   early stopping patience 10;
4. predictions from `predict(soft=True)`: per-class scores, confidence =
   `max(score)`, entropy = `-Σ p log p`.

The number of trainable vs total parameters is saved in
`query_mapping_summary` and is asserted by tests (trainable < total).

## 8. Joint latent visualisation

Reference and query scANVI latents are concatenated; a single UMAP
(`n_neighbors=15, min_dist=0.3, seed=42`) is fit on the joint `X_scANVI`
embedding. UMAPs are visual diagnostics only, never quantitative evidence.
The true-label panel (query truth revealed post hoc) and the prediction panel
are rendered separately with explicit titles.

## 9. Evaluation

Only `scripts/evaluate.py` opens the sealed labels. Each method is scored on:

- accuracy, balanced accuracy, macro-F1, weighted-F1;
- per-class precision, recall, F1, support;
- raw-count and row-normalised confusion matrices.

Two scopes are always reported:

- **all** — every query cell; cell types absent from the reference label set
  are tagged `out_of_reference` and counted (the main split happens to contain
  none, because all 11 types occur in the reference; this is stated
  explicitly);
- **closed_set** — query cells whose true type belongs to the reference label
  set.

Rare types are never dropped from the per-class table. A pre-registered
secondary macro-F1 is computed over **evaluable classes** with ≥ 20 query
cells (Mesothelial, n = 9 in D6, is excluded from that single aggregate but
retained everywhere else); the all-class macro-F1 is always reported
alongside it.

## 10. Confidence–coverage analysis

For thresholds t ∈ {0, 0.5, 0.6, 0.7, 0.8, 0.9}, retain cells with
confidence ≥ t and record retained/rejected counts, coverage, accuracy,
balanced accuracy, macro-F1, error rate on retained cells, and the true-type
composition of rejected cells.

The maximum soft score is called a **prediction confidence score**. No
calibration analysis is performed, so it must not be described as a calibrated
probability. One held-out donor is a single case study: monotone coverage is a
property of the thresholding rule; any accuracy gain at high confidence is
evidence about *this donor's cells*, not a general claim.

## 11. Reproducibility

Seed 42 controls HVG selection, model initialisation and UMAP. Per-run
metadata (`results/run_metadata_main.json`, training summaries) records
versions, device, wall-clock times, epochs actually completed and early-stop
status. Heavy data and model weights are git-ignored; the committed artifacts
are code, configs, tests, docs, the split manifest, small prediction/metric
CSVs and figures.
