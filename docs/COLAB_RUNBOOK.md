# Colab runbook

Free-T4 path for the GPU stages. Free Colab is interactive: keep the tab open,
do not treat it as an automation backend.

## 1. Prepare the repository

Push this repository to your GitHub account (or any URL Colab can clone).
Nothing needs to be uploaded — large data/model files are downloaded or
created inside the runtime.

## 2. Open the notebook

In Colab: **File → Open notebook → GitHub**, select your fork, choose
`notebooks/heart_mapping_colab.ipynb` (or upload the `.ipynb` directly).

## 3. Select the GPU

**Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**
Cell 1 must print `cuda_available: True` and a GPU name. Factory-reset the
runtime and retry if it does not.

## 4. Run cells in order

| cells | purpose | notes |
|---|---|---|
| 1 | GPU check | restart runtime if CUDA is missing |
| 2 | pinned installs | restart only if Colab prompts; requires Python 3.10/3.11 |
| 3 | `git clone` your fork | edit `REPO_URL` first |
| 4 | download data + count audit | must report integer counts |
| 5 | split + label sealing | prints query donor D6 and counts |
| 6 | PCA+kNN baseline | seconds |
| 7 | scVI + scANVI reference training | minutes on T4; **do not re-run** |
| 8 | scArches query update + predictions | prints frozen/total parameter counts |
| 9 | evaluation (opens sealed labels) | only after predictions exist |
| 10 | figures | UMAPs included |
| 11 | pytest + strict verifier | must end with `--require-complete-main-run` passing |
| 12 | zip + browser download | one zip with all main artifacts |
| 13 | optional Drive copy | commented out by default |

## 5. How training should look

- scVI: `train_loss_epoch` decreases from the mid-500s and plateaus; early
  stopping with patience 20 ends training before 400 epochs if validation
  loss stops improving.
- scANVI: 20 epochs maximum; classification loss on reference validation
  cells decreases.
- Query update: at most 100 epochs with `weight_decay=0.0`; the printed
  trainable parameter count is well below the total.

## 6. CUDA out-of-memory

Factory-reset the runtime, lower `scvi.batch_size` in `configs/main.yaml`
(e.g. 256 → 128, committed change documented in the run metadata), rerun from
cell 7. Do not change model depth, epochs or thresholds to chase query
accuracy.

## 7. What to download and merge back

The zip contains `results/`, `figures/`, `data/splits/`, `data/processed/`
and `models/`. Locally:

```bash
unzip -o ~/Downloads/heart_atlas_main_run.zip -d <your-local-clone>
python scripts/verify_outputs.py --config configs/main.yaml \
    --require-complete-main-run
```

Commit code/configs/tests/docs plus small CSVs, manifests, figures and
training summaries. `.h5ad` data and model weights stay git-ignored per
`.gitignore`.

## 8. Reproducing verification and pushing

Re-run the strict verifier (above); it exits non-zero unless the zip contains
a genuine main run. Then push the verified state yourself — the notebook never
stores tokens or pushes on your behalf.

## 9. Python version caveat

The pinned `scvi-tools==1.3.3` wheels require CPython 3.10/3.11. If a future
Colab default runtime moves to 3.12+, either select an older runtime (if
offered) or coordinate a controlled upgrade to scvi-tools ≥ 1.4 (its dataset
loader adds `run_setup_anndata`); do not silently mix versions inside one run.
