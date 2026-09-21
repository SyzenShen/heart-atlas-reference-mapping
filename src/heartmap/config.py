"""Configuration loading and validation.

All experiment parameters come from YAML files (configs/main.yaml or
configs/smoke.yaml). Nothing here is tuned from held-out query performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repository root = two levels up from this file (src/heartmap/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_TOP_LEVEL = [
    "experiment_name",
    "run_tag",
    "seed",
    "dataset",
    "donor_key",
    "batch_key",
    "cell_type_key",
    "counts_layer",
    "unlabeled_category",
    "n_hvg",
    "query_selection_rule",
    "baseline",
    "scvi",
    "scanvi",
    "query_mapping",
    "confidence_thresholds",
]


class ConfigError(ValueError):
    """Raised when a configuration file is missing required structure."""


@dataclass
class Config:
    """Thin wrapper around the validated YAML mapping."""

    raw: dict[str, Any]
    path: Path | None = None

    # ---- convenience accessors -------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def run_tag(self) -> str:
        return self.raw["run_tag"]

    @property
    def is_smoke(self) -> bool:
        return self.raw["run_tag"] == "smoke"

    # ---- paths -------------------------------------------------------------
    @property
    def root(self) -> Path:
        # Tests may redirect the entire artifact tree via raw['_project_root'].
        redirected = self.raw.get("_project_root")
        return Path(redirected) if redirected else PROJECT_ROOT

    @property
    def data_raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def splits_dir(self) -> Path:
        return self.root / "data" / "splits"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def docs_dir(self) -> Path:
        return self.root / "docs"

    # ---- standard artifact paths ------------------------------------------
    @property
    def reference_path(self) -> Path:
        return self.data_processed_dir / f"reference_{self.run_tag}.h5ad"

    @property
    def query_model_input_path(self) -> Path:
        return self.data_processed_dir / f"query_model_input_{self.run_tag}.h5ad"

    @property
    def sealed_labels_path(self) -> Path:
        return self.splits_dir / f"query_eval_labels_{self.run_tag}.csv"

    @property
    def split_manifest_path(self) -> Path:
        return self.splits_dir / f"split_manifest_{self.run_tag}.json"

    @property
    def hvg_path(self) -> Path:
        return self.data_processed_dir / f"hvg_{self.run_tag}.txt"

    # ---- validation --------------------------------------------------------
    def validate(self) -> None:
        missing = [k for k in REQUIRED_TOP_LEVEL if k not in self.raw]
        if missing:
            raise ConfigError(f"Config is missing required keys: {missing}")
        if self.raw["run_tag"] not in ("main", "smoke"):
            raise ConfigError("run_tag must be 'main' or 'smoke'")
        if int(self.raw["n_hvg"]) <= 0:
            raise ConfigError("n_hvg must be positive")
        if self.raw["query_selection_rule"] != "largest_donor_by_cells":
            raise ConfigError(
                "Only the pre-registered 'largest_donor_by_cells' rule is allowed"
            )
        thresholds = list(self.raw["confidence_thresholds"])
        if 0.0 not in [float(t) for t in thresholds]:
            raise ConfigError("confidence_thresholds must include 0.0")
        if any(not (0.0 <= float(t) <= 1.0) for t in thresholds):
            raise ConfigError("confidence thresholds must be within [0, 1]")
        baseline = self.raw["baseline"]
        for k in ("n_components", "n_neighbors", "weights"):
            if k not in baseline:
                raise ConfigError(f"baseline config missing '{k}'")
        if float(self.raw["query_mapping"]["weight_decay"]) != 0.0:
            raise ConfigError(
                "scArches query mapping must use weight_decay=0.0 per official "
                "tutorial recommendation"
            )


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file {path} did not parse to a mapping")
    cfg = Config(raw=raw, path=path)
    cfg.validate()
    return cfg
