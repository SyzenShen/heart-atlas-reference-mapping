#!/usr/bin/env python3
"""PCA + kNN baseline.

Reference-only normalisation/PCA/kNN. The sealed query labels file is never
opened by this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heartmap.baseline import prediction_columns, run_baseline  # noqa: E402
from heartmap.config import load_config  # noqa: E402
from heartmap.provenance import collect_environment, write_json  # noqa: E402
from heartmap.split import (assert_no_overlap, assert_query_sealed,
                            load_split)  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    reference, query, _, manifest = load_split(cfg)
    assert_no_overlap(reference, query, cfg["donor_key"])
    assert_query_sealed(query, cfg)

    predictions, metadata = run_baseline(reference, query, cfg)
    assert list(predictions.columns) == prediction_columns()
    assert not predictions["cell_id"].duplicated().any()
    assert predictions["confidence"].between(0, 1).all()

    out = cfg.results_dir / "predictions" / f"baseline_predictions_{cfg.run_tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)

    metadata["environment"] = {
        k: collect_environment()[k]
        for k in ("python", "scvi", "scanpy", "sklearn", "numpy", "git")
    }
    metadata["query_donor"] = manifest["query_donor"]
    metadata["predictions_file"] = out.name
    write_json(cfg.results_dir / "training" / f"baseline_summary_{cfg.run_tag}.json",
               metadata)
    print(f"Wrote {out} ({len(predictions)} query cells)")


if __name__ == "__main__":
    main()
