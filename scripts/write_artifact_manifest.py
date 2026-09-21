#!/usr/bin/env python3
"""(Re)generate ``results/artifact_manifest_<tag>.json``.

Records SHA-256 of every tracked pipeline artifact (config, HVG list, split
manifest, sealed labels, predictions, scores, metrics, training histories and
summaries, figure manifest) so the published-results verifier can re-check
integrity rather than mere existence.

Provenance semantics: the artifacts were generated before git was initialised
in this repository, so ``generation_git_commit`` is always null. The git
block is a *verification snapshot*: the code state at manifest (re)generation
time. Generation-time git fields are never back-filled.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heartmap.config import load_config  # noqa: E402
from heartmap.provenance import _git_info, sha256_file, write_json  # noqa: E402


def artifact_paths(cfg) -> list[tuple[str, str]]:
    """(manifest key, repo-relative path) for every tracked artifact."""
    tag = cfg.run_tag
    config_name = cfg.path.name if cfg.path else "main.yaml"
    paths: list[tuple[str, str]] = [
        ("config", f"configs/{config_name}"),
        ("hvg_list", f"data/processed/hvg_{tag}.txt"),
        ("split_manifest", f"data/splits/split_manifest_{tag}.json"),
        ("sealed_labels", f"data/splits/query_eval_labels_{tag}.csv"),
        ("baseline_predictions",
         f"results/predictions/baseline_predictions_{tag}.csv"),
        ("scanvi_predictions",
         f"results/predictions/scanvi_predictions_{tag}.csv"),
        ("scanvi_scores", f"results/predictions/scanvi_scores_{tag}.csv"),
        ("joined_eval_pca_knn",
         f"results/predictions/joined_eval_pca_knn_{tag}.csv"),
        ("joined_eval_scanvi_scarches",
         f"results/predictions/joined_eval_scanvi_scarches_{tag}.csv"),
        ("query_selection", "results/query_selection.csv"),
        ("metrics_summary", f"results/metrics/summary_{tag}.csv"),
        ("metrics_per_class", f"results/metrics/per_class_{tag}.csv"),
        ("metrics_confidence_coverage",
         f"results/metrics/confidence_coverage_{tag}.csv"),
        ("metrics_rejected_composition",
         f"results/metrics/rejected_composition_{tag}.csv"),
        ("evaluation_metadata", f"results/metrics/evaluation_metadata_{tag}.json"),
        ("figures_manifest", f"results/metrics/figures_manifest_{tag}.json"),
        ("baseline_summary", f"results/training/baseline_summary_{tag}.json"),
        ("scvi_summary", f"results/training/scvi_summary_{tag}.json"),
        ("scvi_history", f"results/training/scvi_history_{tag}.csv"),
        ("scanvi_summary", f"results/training/scanvi_summary_{tag}.json"),
        ("scanvi_history", f"results/training/scanvi_history_{tag}.csv"),
        ("query_mapping_summary",
         f"results/training/query_mapping_summary_{tag}.json"),
        ("query_mapping_history",
         f"results/training/query_mapping_history_{tag}.csv"),
    ]
    for method in ("pca_knn", "scanvi_scarches"):
        for scope in ("all", "closed_set"):
            for kind in ("counts", "normalized"):
                paths.append(
                    (
                        f"confusion_{method}_{scope}_{kind}",
                        f"results/metrics/confusion_{method}_{scope}_{kind}_{tag}.csv",
                    )
                )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    artifacts: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for name, rel in artifact_paths(cfg):
        path = cfg.root / rel
        if not path.exists():
            missing.append(rel)
            continue
        artifacts[name] = {"path": rel, "sha256": sha256_file(path)}
    for rel in missing:
        print(f"[skip] missing artifact: {rel}")

    git = _git_info()
    payload = {
        "schema": "artifact_manifest_v2",
        "note": (
            "All listed artifacts were generated before git was initialised "
            "in this repository; generation_git_commit is therefore null. "
            "The verification_snapshot block records the code state at "
            "manifest (re)generation time, not at artifact generation time."
        ),
        "generation_git_commit": None,
        "verification_snapshot": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git["commit"],
            "git_branch": git["branch"],
            "git_dirty": git["dirty"],
            "git_remote": git["remote"],
        },
        "artifacts": artifacts,
    }
    out = cfg.results_dir / f"artifact_manifest_{cfg.run_tag}.json"
    write_json(out, payload)
    print(f"Wrote {out} ({len(artifacts)} artifacts hashed)")


if __name__ == "__main__":
    main()
