# Scientific notes

## What this experiment is

A single, rigorously sealed **donor-held-out case study** of label transfer:
can a reference scANVI model, updated onto an unseen heart donor with the
scArches surgery workflow, recover the author-annotated broad cell types of
that donor without ever seeing its labels?

The unit being held out is a **donor** — the level at which biological
variation, sample handling and technical effects co-vary. A random cell split
would place sibling cells from the same donor on both sides of the split and
would over-estimate transfer performance.

## What the models represent

- **scVI** learns a donor-conditioned low-dimensional representation of raw
  UMI counts with a ZINB decoder; it integrates batches but has no notion of
  cell-type labels.
- **scANVI** extends the scVI latent space with a semi-supervised classifier,
  turning the representation into one that supports label transfer.
- **scArches** (implemented natively in scvi-tools via
  `prepare_query_anndata` + `load_query_data`) freezes the trained reference
  network and adds new, trainable batch-adaptation parameters for the query
  donor, so the query is *mapped into* the reference space rather than used to
  retrain it.
- The **PCA + kNN baseline** answers whether the generative machinery provides
  value beyond a simple linear representation plus neighbour voting using the
  same 2,000 reference-derived genes.

## Reading the results honestly

- Accuracy is dominated by abundant populations (ventricular cardiomyocytes,
  endothelial, pericytes, fibroblasts). Macro-F1 and per-class recall are the
  place where rare types (mesothelial, adipocytes, neuronal) are visible.
- Confidence thresholding trades coverage against reliability: raising the
  threshold abstains on more cells. If rare/ambiguous populations account for
  most abstentions, that is a coverage-equity caveat. High-confidence errors
  are important and must not be hidden.
- The maximum soft score is not calibrated; a 0.9 score need not imply 90%
  empirical correctness. Reliability/ECE analysis is out of scope here.
- Visual mixing in the joint UMAP is consistent with successful alignment but
  is never proof of biological correctness; the quantitative tables carry the
  conclusion.

## Scope boundaries

This project does **not**:

- reproduce the full scArches paper or its multi-organ benchmark;
- reconstruct the Human Heart Cell Atlas;
- discover new cell types or states (the 65 fine `cell_states` annotations
  are untouched future work);
- claim generalisation to humans at large — one donor is one case, and
  thousands of cells from it are not thousands of replicates.

## Future work

1. Repeat the identical sealed protocol across multiple pre-registered
   held-out donors to obtain a donor-level distribution of metrics.
2. Fine-state transfer (`cell_states`) with stronger support thresholds.
3. Explicit calibration (temperature scaling / reliability diagrams) fit
   without touching the query donor.
4. Novelty-aware scoring (e.g., class-conditional uncertainty) for query
   states genuinely absent from the reference — the current deterministic
   split happens to contain no out-of-reference types in D6.
5. External validation on an independently sampled heart cohort.
