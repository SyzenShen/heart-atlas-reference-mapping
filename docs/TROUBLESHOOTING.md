# Troubleshooting

Issues actually encountered while building this repository on Intel macOS /
Python 3.10, plus common failure modes.

## Installation

**`llvmlite` / `numba` fail to build from source.**
Pin the compatible wheels: `numba==0.60.0`, `llvmlite==0.43.0`,
`jax==0.4.30`, `jaxlib==0.4.30`. This combination has macOS x86_64 wheels.

**scvi-tools installs a release requiring Python ≥ 3.12.**
On Python 3.10 pin `scvi-tools==1.3.3` (the last 1.3.x line). Do not install
1.4+ on 3.10.

**`seurat_v3 HVG fails` (`scikit-misc` / loess missing).**
Install `scikit-misc==0.5.2` (provides the loess estimator
`highly_variable_genes(flavor="seurat_v3")` calls).

**Leiden/UMAP errors about `igraph` or `leidenalg`.**
Install `igraph==1.0.0` and `leidenalg==0.12.0`.

**`SSL: CERTIFICATE_VERIFY_FAILED` when downloading data on python.org Python.**
The framework Python ships without trusted CAs. Either run
`"/Applications/Python 3.10/Install Certificates.command"` or prefix commands
with `SSL_CERT_FILE=$(.venv/bin/python -c "import certifi;print(certifi.where())")`.
The data file can also be fetched with `curl -L -o data/raw/hca_subsampled_20k.h5ad <url>`.

## API differences

**`TypeError: ... unexpected keyword 'run_setup_anndata'`.**
That loader argument belongs to scvi-tools ≥ 1.4. On 1.3.3 call
`scvi.data.heart_cell_atlas_subsampled(save_path=..., remove_nuisance_clusters=True)`
and run `SCVI.setup_anndata(...)` yourself.

**Old tutorial URL 404s.**
For 1.3.x the reference-mapping tutorial lives under
`user_guide/notebooks/multimodal/scarches_scvi_tools.html`.

## Preprocessing

**loess crash on the tiny smoke reference.**
seurat_v3 with many tiny batches can fail inside loess. HVGs are therefore
always selected on the **full** reference, even for the smoke split (HVG is a
reference-only artifact; smoke only subsamples cells downstream).

## Training

**MPS warnings / hangs on Apple Silicon.**
The config never auto-selects MPS (`accelerator: auto` resolves to CPU when
CUDA is absent). Use the Colab notebook for GPU runs.

**CUDA out-of-memory on Colab.**
Restart the runtime, reduce `scvi.batch_size` (e.g. 128), and rerun from the
training cell; do not change any other parameter.

**Early stopping ends training very early on smoke.**
Expected — smoke uses tiny data and 1–2 epochs; its metrics are meaningless
and are never compared with main.

## Evaluation

**`KeyError` / non-unique cell IDs in evaluation.**
Indicates a prediction file was edited or concatenated. Re-run the
corresponding prediction script; do not hand-edit CSVs.

**Mesothelial missing from evaluable macro-F1.**
By design: D6 contains only 9 Mesothelial cells and the pre-registered
evaluable-support threshold is 20. The class stays in the per-class table and
in the all-class macro-F1.

## Verification

**`verify_outputs.py --require-complete-main-run` exits 1.**
It lists every missing artifact. A smoke run never satisfies it. Status must
stay at `BASELINE_COMPLETED` or `READY_FOR_COLAB` until a genuine main run
(both prediction sets, metrics and figures) is present and verified.
