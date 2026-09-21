# Leakage-prevention checklist

These controls are implemented in `src/heartmap/split.py`, enforced by
`tests/`, and re-checked by `scripts/verify_outputs.py`.

## Split

- [x] Split is at **donor granularity**; no random cell split exists in code
      or notebooks.
- [x] Query donor chosen by the label-blind `largest_donor_by_cells` rule
      before any training; donor ranking frozen in `results/query_selection.csv`.
      Selection uses only the donor key column — `cell_type` is never read.
- [x] Reference donor IDs and query donor ID are disjoint (13 vs 1).
- [x] Reference and query cell IDs are disjoint; both sets SHA-256 hashed in
      `data/splits/split_manifest_main.json`.

## Labels

- [x] Query truth exists only in
      `data/splits/query_eval_labels_main.csv` with columns
      `cell_id, true_cell_type`.
- [x] The query model input has `cell_type`, `cell_states` and all other
      label-like columns removed (`LABEL_HINTS` list in `split.py`).
- [x] `labels_scanvi` is exactly `"Unknown"` for every query cell.
- [x] No second copy of query labels survives in `obs`, `obsm`, `uns`,
      layers, or `.raw`.

## Process isolation

- [x] Only the evaluation-side scripts — `scripts/evaluate.py` and
      `scripts/verify_outputs.py` (which recompute metrics from the sealed
      labels after predictions are frozen) — open the sealed label file
      via `load_evaluation_labels`. Training/mapping/figure scripts use
      `load_model_split`, which never reads the sealed CSV; a monkeypatch
      test asserts `pd.read_csv(sealed_path)` is never called.
- [x] Static test greps training scripts
      (`run_baseline.py`, `train_reference.py`, `map_query.py`) for
      sealed-file references, `true_cell_type`, `load_split`, and evaluation
      imports — including an AST import check. A further AST test verifies
      every training script imports `load_model_split`, and that only the
      evaluation-side scripts import the sealed-label loaders.
- [x] Hyperparameters are fixed in YAML before evaluation; no threshold or
      epoch is selected using query accuracy.
- [x] Only one query donor exists; there is no "try several donors and keep
      the best one" code path.

## Preprocessing

- [x] HVGs (2,000, seurat_v3, batch-aware) fit on reference only; list frozen
      in `data/processed/hvg_main.txt`.
- [x] Baseline PCA fit on reference only; query calls `transform`.
- [x] kNN classifier fit on reference embeddings and reference labels only.
- [x] Scaling/target-sum values are fixed constants (10,000), never estimated
      jointly.
- [x] Query genes are reordered/zero-padded to the reference set via
      `prepare_query_anndata`.

## Input validity

- [x] scVI/scANVI receive raw integer UMI counts (`layer="counts"`); a
      validator rejects negatives, non-integers, missing layers, and refuses
      log-normalised values.
- [x] UMAP is computed from the model latent and labelled visual-only.

## Verification commands

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/verify_outputs.py --config configs/main.yaml --verify-published-results
```

`--require-complete-local-run` additionally checks git-ignored h5ad/model
artifacts and returns non-zero with an explicit list of missing items unless
a true local run is present.
