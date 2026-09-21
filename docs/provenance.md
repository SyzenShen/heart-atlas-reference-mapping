# Provenance

This document records the lineage of every decision in the experiment.
Machine-readable copies are emitted per run as `results/run_metadata_<tag>.json`
and per step as `results/training/*_summary_<tag>.json`.

## Dataset

- **Loader:** `scvi.data.heart_cell_atlas_subsampled(save_path=..., remove_nuisance_clusters=True)`
  (scvi-tools 1.3.3). The 1.3.3 signature has **no** `run_setup_anndata`
  argument (that parameter was introduced in scvi-tools 1.4).
- **Source:** subsampled Human Heart Cell Atlas shipped by scvi-tools
  (<https://www.heartcellatlas.org/>; Littmann et al. 2022).
- **Downloaded file:** `data/raw/hca_subsampled_20k.h5ad` (65,713,707 bytes,
  SHA recorded in `results/data_audit.json` if present).
- **After nuisance removal** (doublets / NotAssigned clusters dropped by the
  loader): **18,641 cells × 26,662 genes**.
- Fingerprint (`data/splits/split_manifest_main.json`):
  - `n_obs = 18641`, `n_vars = 26662`
  - `obs_names_sha256 = 097343b5…1d95`
  - `var_names_sha256 = 0b6b43ec…c133`

## Keys (chosen after auditing the AnnData, not from field names)

| role | key |
|---|---|
| donor key | `donor` |
| batch key | `donor` (same column; donor is the technical/biological batch) |
| cell-type key | `cell_type` (11 broad author annotations) |
| counts layer | `counts` (materialised copy of raw `adata.X`) |
| scANVI labels key | `labels_scanvi` (`Unknown` for every query cell) |

`cell_states` (65 fine states) is intentionally out of scope and listed as
future work; it is also removed from the query model input so it cannot act as
a fine-grained label leak.

## Counts verification

- `adata.X` is a CSR `float32` matrix of raw integer UMI counts
  (min 1, max 2693, no negatives, all non-zeros integer).
- No `.raw`, no count-bearing `layers` exist on download; the pipeline writes
  `adata.layers["counts"] = adata.X.copy()` and always passes
  `layer="counts"` to `SCVI.setup_anndata`.
- See `results/data_audit.json` and `docs/DATA_DICTIONARY.md`.

## Query-donor selection (pre-registered, deterministic, label-blind)

Rule name: `largest_donor_by_cells` (see `results/query_selection.csv`).

1. Drop cells without a donor ID.
2. Pick the donor with the largest cell count.
3. Tie-break: lexicographically smallest donor ID.

The `cell_type` column is **never read** during selection — the rule is
label-blind. Cell-type composition is computed *after* selection for
description only. **D6** is the largest donor (3,009 cells, 11 cell types) and
was therefore fixed as the single query donor before any model was trained.
Reference = the other 13 donors, **15,632 cells**.

## Software versions

| component | version |
|---|---|
| Python | 3.10.11 |
| scvi-tools | 1.3.3 (last line supporting Python 3.10) |
| scanpy | 1.11.5 |
| anndata | 0.11.4 |
| PyTorch | 2.2.2 (CPU build) |
| Lightning | 2.6.6 |
| numpy | 1.26.4 |
| numba / llvmlite | 0.60.0 / 0.43.0 |
| jax / jaxlib | 0.4.30 |
| scikit-learn | 1.7.2 |
| scikit-misc | 0.5.2 (seurat_v3 HVG support) |
| igraph / leidenalg | 1.0.0 / 0.12.0 |

Full lockfile: `requirements-lock.txt`. Backend: **CPU on Intel macOS**;
CUDA unavailable, Apple MPS intentionally not used (scvi-tools/Lightning do not
provide reliable MPS support for these models). GPU training runs via the
Colab notebook.

## Official references checked

- scArches paper: Lotfollahi et al., *Nat Biotechnol* 2022,
  <https://doi.org/10.1038/s41587-021-01001-7>
- scANVI paper: Xu et al., *Mol Syst Biol* 2021,
  <https://doi.org/10.15252/msb.20209620>
- Current reference-mapping tutorial used as the API template:
  <https://docs.scvi-tools.org/en/1.3.x/user_guide/notebooks/multimodal/scarches_scvi_tools.html>
  (the old `tutorials/notebooks/scrna/scarches_scvi_tools.html` URL 404s).
- SCVI / SCANVI / dataset API pages for scvi-tools 1.3.3.
- API/tutorial access verified against the installed 1.3.3 package at build
  time (2026-09).

## Model parameters and seeds

All parameters come from `configs/main.yaml`; seed = 42.

- HVGs: 2,000, `seurat_v3`, batch-aware across reference donors, fit on the
  full reference only.
- scVI: `n_latent=30`, `n_hidden=128`, `n_layers=2`, `dropout_rate=0.2`,
  `use_layer_norm="both"`, `use_batch_norm="none"`,
  `encode_covariates=True`, ZINB likelihood; up to 400 epochs with early
  stopping (patience 20).
- scANVI: initialised from scVI, up to 20 epochs,
  `n_samples_per_label=100`, `unlabeled_category="Unknown"`.
- scArches query update: `SCANVI.prepare_query_anndata` + `load_query_data`,
  up to 100 epochs, **`weight_decay=0.0`** (official recommendation),
  reference weights frozen; trainable parameter count is recorded in
  `results/training/query_mapping_summary_<tag>.json`.

### Main run as executed (CPU, seed 42, 2026-09-21)

| stage | epochs actually run | early stop | wall time |
|---|---|---|---|
| scVI | 316 / 400 | yes (validation patience 20) | 1,197.5 s (~20.0 min) |
| scANVI | 11 / 20 | yes (patience 10) | 89.5 s |
| query update | 76 / 100 | yes (patience 10) | 71.8 s |

Query-model parameters: **578,946 trainable / 1,426,188 total** (reference
weights frozen).

- Confidence = maximum soft class score (a *prediction confidence score*, not
  a calibrated probability); entropy over soft scores is also stored.

## Git provenance

`git` is not initialised until the final phase. Run metadata records
branch/commit/remote once a repository exists. No commit is made without
explicit user authorisation.
