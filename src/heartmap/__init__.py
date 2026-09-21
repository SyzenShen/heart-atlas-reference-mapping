"""heartmap: donor-held-out reference mapping on the Human Heart Cell Atlas.

Educational, reproducible pipeline built around scvi-tools (scVI + scANVI +
scArches query mapping) with a strict donor-level split and label sealing.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Column name used as the scANVI labels key in every model-facing AnnData.
# Reference cells carry their author annotations here; query cells carry the
# single unlabeled category (config: unlabeled_category) and nothing else.
LABELS_KEY = "labels_scanvi"
# Columns added to AnnData to mark the provenance split.
SPLIT_KEY = "split"
QUERY_DONOR_COL = "query_donor"
