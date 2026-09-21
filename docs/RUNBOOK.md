# Local runbook

CPU-only, macOS/Linux. The main deep models train in roughly tens of minutes
on a modern multi-core CPU (T4 GPU via Colab is faster — see
`COLAB_RUNBOOK.md`).

## 1. Create the environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt        # or: pip install -r requirements-lock.txt
pip install -e .
```

On python.org macOS Python with certificate errors, prefix downloads with:

```bash
export SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")
```

## 2. Audit environment and data

```bash
python scripts/check_environment.py
python scripts/audit_data.py --config configs/main.yaml
```

Outputs: `docs/environment_audit.md`, `results/environment.json`,
`results/data_audit.json`, `results/obs_columns.csv`,
`results/donor_cell_counts.csv`, `results/donor_celltype_crosstab.csv`,
`docs/DATA_DICTIONARY.md`.

The dataset downloads automatically to `data/raw/` on first use. If download
fails, place the loader's file at `data/raw/hca_subsampled_20k.h5ad`
manually (same SHA; see `docs/provenance.md`).

**Stop here unless the audit shows raw integer counts and the keys
`donor / donor / cell_type`.**

## 3. Fix the split and seal labels

```bash
python scripts/prepare_split.py --config configs/main.yaml
```

Produces reference/query h5ad files, the frozen HVG list, the split manifest
with cell-ID hashes, and `data/splits/query_eval_labels_main.csv`. Re-run for
smoke with `--config configs/smoke.yaml` (HVGs always come from the full
reference).

## 4. Run the baseline (predictions only, no labels)

```bash
python scripts/run_baseline.py --config configs/main.yaml
# -> results/predictions/baseline_predictions_main.csv
```

## 5. Train reference models and map the query

```bash
python scripts/train_reference.py --config configs/main.yaml
# -> models/scvi_reference_main/, models/scanvi_reference_main/,
#    results/reference_scanvi_latent_main.h5ad, training histories/summaries

python scripts/map_query.py --config configs/main.yaml
# -> models/scanvi_query_main/, results/query_mapped_main.h5ad,
#    results/predictions/scanvi_predictions_main.csv (+ scores)
```

Neither script reads the sealed labels.

## 6. Freeze predictions, then evaluate

Only after both prediction files exist:

```bash
python scripts/evaluate.py --config configs/main.yaml
python scripts/make_figures.py --config configs/main.yaml
# add --skip-umap on machines without igraph/leidenalg (diagnostic only)
```

## 7. Verify

```bash
python -m pytest tests/ -q
python scripts/verify_outputs.py --config configs/main.yaml
python scripts/verify_outputs.py --require-complete-main-run
```

The last command exits 0 only when a genuine main run is complete; smoke
artifacts never satisfy it.

## Smoke path (code-path check only)

```bash
python scripts/prepare_split.py  --config configs/smoke.yaml
python scripts/run_baseline.py   --config configs/smoke.yaml
python scripts/train_reference.py --config configs/smoke.yaml
python scripts/map_query.py      --config configs/smoke.yaml
python scripts/evaluate.py       --config configs/smoke.yaml
python scripts/make_figures.py   --config configs/smoke.yaml
```

Smoke outputs carry the `smoke` tag, land in `figures/smoke/`, and are
explicitly excluded from scientific claims.

## Re-running safely

Re-running a step overwrites same-tag artifacts. Never re-run
`prepare_split.py` with altered selection criteria after looking at results;
the selection rule must stay fixed independently of model performance. Do not
tune any YAML parameter against query accuracy.
