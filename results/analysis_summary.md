# Analysis summary

Auto-generated after frozen-prediction evaluation. One held-out donor (**D6**, 3009 cells) vs a 13-donor reference (15632 cells). Cells within the donor are not independent replicates.

## Overall metrics (closed set == all: no out-of-reference types in D6)

| method | accuracy | balanced acc. | macro-F1 | weighted-F1 | macro-F1 (evaluable, support>=20) |
|---|---|---|---|---|---|
| pca_knn | 0.952 | 0.779 | 0.795 | 0.947 | 0.861 |
| scanvi_scarches | 0.967 | 0.927 | 0.920 | 0.967 | 0.928 |

## Weakest cell types under scANVI (by F1)

| type | support | PCA+kNN recall | scANVI recall | PCA+kNN F1 | scANVI F1 |
|---|---|---|---|---|---|
| Neuronal | 24 | 0.17 | 0.67 | 0.29 | 0.74 |
| Smooth_muscle_cells | 127 | 0.62 | 0.88 | 0.73 | 0.81 |
| Mesothelial | 9 | 0.00 | 0.89 | 0.00 | 0.84 |
| Pericytes | 372 | 0.85 | 0.85 | 0.85 | 0.89 |

## Confidence-coverage (coverage / accuracy on retained)

- **pca_knn** — t=0: 1.000/0.952, t=0.5: 0.996/0.955, t=0.6: 0.978/0.964, t=0.7: 0.959/0.971, t=0.8: 0.938/0.977, t=0.9: 0.892/0.984
- **scanvi_scarches** — t=0: 1.000/0.967, t=0.5: 0.999/0.968, t=0.6: 0.990/0.971, t=0.7: 0.984/0.973, t=0.8: 0.976/0.975, t=0.9: 0.959/0.982

Coverage starts at 1.0 and cannot increase with the threshold; retained accuracy rises for both methods (more cells abstained). Confidence is a prediction confidence score, **not a calibrated probability**; these statements concern D6 cells only. Rare mesothelial cells (n=9) are excluded from the evaluable macro-F1 but retained in the all-class table.
