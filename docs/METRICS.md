# Metrics reference

All evaluation outputs are written by `scripts/evaluate.py`, the only script
that reads the sealed labels. Files are tagged `_main` / `_smoke`.

## Prediction files (`results/predictions/`)

Schema for both `baseline_predictions_<tag>.csv` and
`scanvi_predictions_<tag>.csv`:

| column | meaning |
|---|---|
| `cell_id` | query cell barcode; unique; matches sealed label file 1:1 |
| `predicted_label` | reference label predicted for the cell |
| `confidence` | winning vote fraction (baseline) or max soft score (scANVI), in [0, 1] |
| `entropy` | `-Σ p log p` over class scores/votes (`0.0` for unanimous) |
| `query_donor` | held-out donor ID (`D6`) |
| `method` | `pca_knn` or `scanvi_scarches` |
| `status` | `predicted` (or `abstained` in extensions) |

No true-label column is ever present. `scanvi_scores_<tag>.csv` additionally
stores the full per-class soft scores.

## `summary_<tag>.csv`

One row per method × scope (`all`, `closed_set`).

| column | definition |
|---|---|
| `n_query_cells` | cells scored in this scope |
| `n_out_of_reference_types` | distinct true types absent from reference labels |
| `n_out_of_reference_cells` | cells of such types |
| `accuracy` | fraction correct |
| `balanced_accuracy` | mean recall over true types |
| `macro_f1` | mean F1 over **all** true types in scope |
| `weighted_f1` | support-weighted F1 |
| `macro_f1_evaluable` | macro-F1 over types with support ≥ `evaluable_min_support` (20, pre-registered) |
| `n_evaluable_classes` | how many types entered that aggregate |
| `n_classes_true` | distinct true types in scope |

Rare types remain in `per_class_<tag>.csv` regardless of the evaluable
threshold.

## `per_class_<tag>.csv`

`method, scope, label, precision, recall, f1, support, in_reference`.

`in_reference=False` marks out-of-reference types (kept in the `all` scope;
precision/recall/F1 are 0 by construction because the model cannot emit that
label).

## Confusion matrices

`confusion_<method>_<scope>_<counts|normalized>_<tag>.csv` — rows are true
labels, columns predicted labels. Counts matrices sum to the number of query
cells in scope; normalized rows sum to 1.

## `confidence_coverage_<tag>.csv`

One row per method × threshold (thresholds sorted ascending):

`retained_cells`, `rejected_cells`, `coverage`
(= retained / total query cells), `accuracy_on_retained`,
`balanced_accuracy_on_retained`, `macro_f1_on_retained`,
`error_rate_on_retained` (1 − accuracy), `retained_out_of_reference_cells`,
`mean_confidence`.

By construction coverage at t = 0 is 1.0 and coverage is non-increasing in t
(asserted by `verify_outputs.py` and the test suite).

## `rejected_composition_<tag>.csv`

`method, true_cell_type, n_rejected, threshold, fraction_of_type_rejected` —
which true types the abstention rule discards at each threshold. A high
fraction for rare types is a coverage/fairness warning, not a feature.

## Interpretation rules

- Metrics summarise **cells within one held-out donor**; cells are not
  independent biological replicates.
- Accuracy and weighted-F1 are dominated by abundant types; macro-F1 and the
  per-class table show rare-type behaviour.
- `confidence` is a score, not a calibrated probability; increasing accuracy
  over retained cells says high-confidence predictions were more reliable for
  these cells — nothing more.
