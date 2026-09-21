# data/

Large data files are git-ignored; directories are tracked with `.gitkeep`.

## `raw/`

- `hca_subsampled_20k.h5ad` — downloaded by
  `scvi.data.heart_cell_atlas_subsampled(save_path="data/raw",
  remove_nuisance_clusters=True)` (scvi-tools 1.3.3). About 66 MB; after
  nuisance removal 18,641 cells × 26,662 genes. `X` is raw integer UMI counts.

Download manually if needed (this is the exact URL embedded in the
scvi-tools 1.3.3 loader):

```bash
curl -L -o data/raw/hca_subsampled_20k.h5ad \
  "https://github.com/YosefLab/scVI-data/blob/master/hca_subsampled_20k.h5ad?raw=true"
```

## `processed/`

- `reference_<tag>.h5ad` — reference AnnData (13 donors, 15,632 cells, main).
- `query_model_input_<tag>.h5ad` — sealed query AnnData for models
  (no truth columns; `labels_scanvi == "Unknown"`).
- `hvg_<tag>.txt` — 2,000 reference-only seurat_v3 HVGs (one gene per line).
  The smoke split uses HVGs computed on the full reference.

## `splits/`

- `split_manifest_<tag>.json` — query/reference donor IDs, cell-ID SHA-256
  hashes, keys, seed, selection rule, dataset fingerprint.
- `query_eval_labels_<tag>.csv` — **sealed ground truth**
  (`cell_id, true_cell_type`). Read only by `scripts/evaluate.py` after
  predictions are frozen.
- `preprocess_metadata_<tag>.json` — preprocessing decisions and versions.
