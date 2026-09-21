#!/usr/bin/env python3
"""Verify pipeline outputs, leakage guards, schemas and status consistency.

Two verification tiers
-----------------------
* ``--verify-published-results`` (default): checks only artifacts tracked in
  git — predictions, metrics, figures, manifests, hashes, sealed-label
  schema, HVG list, query-selection table. Passes on a fresh clone.
* ``--require-complete-local-run``: additionally checks local (git-ignored)
  artifacts — processed h5ad files, model weights, latent h5ad, and
  split-isolation on the h5ad objects. Fails on a fresh clone.

``--require-complete-main-run`` is kept as a backward-compatible alias for
``--require-complete-local-run``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heartmap.baseline import prediction_columns  # noqa: E402
from heartmap.config import PROJECT_ROOT, load_config  # noqa: E402
from heartmap.provenance import read_json  # noqa: E402


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))
        if not ok:
            self.errors.append(f"{name}: {detail}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def exist(self, name: str, path: Path) -> bool:
        ok = path.exists()
        self.check(f"exists:{name}", ok, str(path.relative_to(PROJECT_ROOT)))
        return ok

    def print_report(self) -> None:
        for name, ok, detail in self.checks:
            mark = "PASS" if ok else "FAIL"
            print(f"[{mark}] {name}" + (f" ({detail})" if detail and not ok else ""))
        for w in self.warnings:
            print(f"[WARN] {w}")


# --------------------------------------------------------------------------
# Published checks — tracked artifacts only (pass on fresh clone)
# --------------------------------------------------------------------------
def _verify_published_static(cfg, rep: Report) -> None:
    rep.exist("data audit", cfg.results_dir / "data_audit.json")
    rep.exist("obs columns", cfg.results_dir / "obs_columns.csv")
    rep.exist("query selection", cfg.results_dir / "query_selection.csv")
    rep.exist("donor crosstab", cfg.results_dir / "donor_celltype_crosstab.csv")
    rep.exist("split manifest", cfg.split_manifest_path)
    rep.exist("sealed labels", cfg.sealed_labels_path)
    rep.exist("hvg list", cfg.hvg_path)
    rep.exist("preprocess metadata",
              cfg.splits_dir / f"preprocess_metadata_{cfg.run_tag}.json")
    rep.exist("artifact manifest", cfg.results_dir / "artifact_manifest.json")

    if cfg.split_manifest_path.exists():
        manifest = read_json(cfg.split_manifest_path)
        rep.check(
            "manifest hashes",
            bool(manifest.get("reference_cell_ids_sha256"))
            and bool(manifest.get("query_cell_ids_sha256")),
            "missing cell-id hashes",
        )
        rep.check(
            "manifest selection rule",
            manifest.get("selection_rule") == "largest_donor_by_cells",
            "unexpected selection rule",
        )
        rep.check(
            "manifest label-blind rule",
            manifest.get("selection_rule_detail", {}).get("label_blind") is True,
            "selection rule is not label-blind",
        )

    if cfg.sealed_labels_path.exists():
        sealed = pd.read_csv(cfg.sealed_labels_path)
        rep.check(
            "sealed schema", list(sealed.columns) == ["cell_id", "true_cell_type"],
            str(list(sealed.columns)),
        )
        rep.check("sealed unique ids", not sealed.cell_id.duplicated().any(),
                  "duplicate cell ids")


def _verify_published_predictions(cfg, rep: Report, stem: str) -> None:
    path = cfg.results_dir / "predictions" / f"{stem}_{cfg.run_tag}.csv"
    if not rep.exist(f"predictions:{stem}", path):
        return
    df = pd.read_csv(path)
    rep.check(
        f"{stem} schema", list(df.columns)[:7] == prediction_columns(),
        str(list(df.columns)),
    )
    rep.check(f"{stem} no true labels", "true_cell_type" not in df.columns,
              "true label leaked into predictions")
    rep.check(f"{stem} unique cell ids", not df.cell_id.duplicated().any(),
              "duplicates")
    rep.check(f"{stem} confidence range", df.confidence.between(0, 1).all(),
              "out of [0,1]")
    rep.check(f"{stem} non-empty entropy", df.entropy.notna().all(), "NaN entropy")
    if cfg.sealed_labels_path.exists():
        sealed = pd.read_csv(cfg.sealed_labels_path)
        rep.check(
            f"{stem} matches sealed ids",
            set(df.cell_id) == set(sealed.cell_id),
            "cell-id set differs from sealed labels",
        )


def _verify_published_models_metadata(cfg, rep: Report) -> None:
    """Training summaries and run metadata are tracked; model.pt is not."""
    for name in ("scvi", "scanvi", "query_mapping"):
        stem = "query_mapping" if name == "query_mapping" else name
        rep.exist(
            f"history:{name}",
            cfg.results_dir / "training" / f"{stem}_history_{cfg.run_tag}.csv",
        )
        summary_name = (
            f"{stem}_summary_{cfg.run_tag}.json"
            if name != "scvi"
            else f"scvi_summary_{cfg.run_tag}.json"
        )
        rep.exist(
            f"summary:{name}", cfg.results_dir / "training" / summary_name
        )
    rep.exist("run metadata", cfg.results_dir / f"run_metadata_{cfg.run_tag}.json")


def _verify_published_metrics(cfg, rep: Report) -> None:
    mdir = cfg.results_dir / "metrics"
    summary_path = mdir / f"summary_{cfg.run_tag}.csv"
    if not rep.exist("metrics summary", summary_path):
        return
    summary = pd.read_csv(summary_path)
    need_cols = {
        "method", "scope", "n_query_cells", "accuracy", "balanced_accuracy",
        "macro_f1", "weighted_f1",
    }
    rep.check("summary columns", need_cols.issubset(summary.columns),
              str(list(summary.columns)))
    rep.check("summary two methods",
              set(summary.method) >= {"pca_knn", "scanvi_scarches"},
              str(set(summary.method)))
    rep.check("summary scopes", set(summary.scope) == {"all", "closed_set"},
              str(set(summary.scope)))
    rep.check("metrics bounded",
              summary[["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]]
              .apply(lambda s: s.between(0, 1).all()).all(),
              "metric outside [0,1]")

    sealed_n = None
    if cfg.sealed_labels_path.exists():
        sealed_n = len(pd.read_csv(cfg.sealed_labels_path))
    for method in ("pca_knn", "scanvi_scarches"):
        cm_path = mdir / f"confusion_{method}_all_counts_{cfg.run_tag}.csv"
        if cm_path.exists() and sealed_n is not None:
            cm = pd.read_csv(cm_path, index_col=0)
            rep.check(
                f"confusion total:{method}", int(cm.to_numpy().sum()) == sealed_n,
                f"{int(cm.to_numpy().sum())} != {sealed_n}",
            )
    cov_path = mdir / f"confidence_coverage_{cfg.run_tag}.csv"
    if cov_path.exists():
        cov = pd.read_csv(cov_path)
        for method, g in cov.groupby("method"):
            g = g.sort_values("threshold")
            mono = g.coverage.diff().dropna().le(1e-12).all()
            rep.check(f"coverage monotonic:{method}", bool(mono), "coverage increased")
            row0 = g.sort_values("threshold").iloc[0]
            rep.check(
                f"coverage at t=0:{method}",
                abs(float(row0.coverage) - 1.0) < 1e-9,
                f"coverage={row0.coverage}",
            )
    rep.exist("evaluation metadata",
              mdir / f"evaluation_metadata_{cfg.run_tag}.json")
    rep.exist("per-class table", mdir / f"per_class_{cfg.run_tag}.csv")


def _verify_published_figures(cfg, rep: Report) -> None:
    mdir = cfg.results_dir / "metrics"
    manifest = mdir / f"figures_manifest_{cfg.run_tag}.json"
    if not rep.exist("figures manifest", manifest):
        return
    payload = read_json(manifest)
    missing = [
        f["file"] for f in payload.get("figures", [])
        if not (cfg.figures_dir / f["file"]).exists()
        and not (cfg.figures_dir / "smoke" / f["file"]).exists()
    ]
    rep.check("all listed figures exist", not missing, ", ".join(missing))


# --------------------------------------------------------------------------
# Local-only checks — require git-ignored h5ad / model artifacts
# --------------------------------------------------------------------------
def _verify_local_h5ad(cfg, rep: Report) -> None:
    rep.exist("reference h5ad", cfg.reference_path)
    rep.exist("query model input h5ad", cfg.query_model_input_path)
    rep.exist("reference latent",
              cfg.results_dir / f"reference_scanvi_latent_{cfg.run_tag}.h5ad")
    rep.exist("query latent", cfg.results_dir / f"query_mapped_{cfg.run_tag}.h5ad")


def _verify_local_models(cfg, rep: Report) -> None:
    for name in ("scvi_reference", "scanvi_reference", "scanvi_query"):
        d = cfg.models_dir / f"{name}_{cfg.run_tag}"
        rep.exist(f"model:{name}", d / "model.pt")


def _verify_split_isolation(cfg, rep: Report) -> None:
    if not (cfg.reference_path.exists() and cfg.query_model_input_path.exists()):
        return
    import anndata as ad

    ref = ad.read_h5ad(cfg.reference_path)
    qry = ad.read_h5ad(cfg.query_model_input_path)
    rep.check(
        "donor disjoint",
        not (set(ref.obs[cfg["donor_key"]].astype(str))
             & set(qry.obs[cfg["donor_key"]].astype(str))),
        "shared donor between reference and query",
    )
    rep.check(
        "cell-id disjoint",
        not (set(ref.obs_names) & set(qry.obs_names)),
        "shared cell ids",
    )
    rep.check(
        "query true-label column removed",
        cfg["cell_type_key"] not in qry.obs.columns,
        f"{cfg['cell_type_key']} present in query model input",
    )
    if "labels_scanvi" in qry.obs.columns:
        vals = set(qry.obs["labels_scanvi"].astype(str).unique())
        rep.check(
            "query labels all Unknown", vals == {cfg["unlabeled_category"]},
            str(vals),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument(
        "--verify-published-results", action="store_true",
        help="fail unless all tracked (published) artifacts pass verification",
    )
    parser.add_argument(
        "--require-complete-local-run", action="store_true",
        help="additionally require git-ignored h5ad/model artifacts",
    )
    parser.add_argument(
        "--require-complete-main-run", action="store_true",
        help="(deprecated alias for --require-complete-local-run)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    rep = Report()

    require_local = args.require_complete_local_run or args.require_complete_main_run

    # ---- Published (tracked) checks --------------------------------------
    _verify_published_static(cfg, rep)
    _verify_published_predictions(cfg, rep, "baseline_predictions")
    _verify_published_predictions(cfg, rep, "scanvi_predictions")
    _verify_published_models_metadata(cfg, rep)
    _verify_published_metrics(cfg, rep)
    _verify_published_figures(cfg, rep)

    # ---- Local-only (git-ignored) checks --------------------------------
    if require_local:
        _verify_local_h5ad(cfg, rep)
        _verify_local_models(cfg, rep)
        _verify_split_isolation(cfg, rep)

    if cfg.is_smoke:
        rep.warn("SMOKE artifacts verified; these never count as a main run.")

    # README status consistency (lightweight textual guard).
    readme = PROJECT_ROOT / "README.md"
    if readme.exists():
        txt = readme.read_text()
        if "MAIN_RESULTS_VERIFIED" in txt and rep.errors and args.verify_published_results:
            rep.warn("README mentions MAIN_RESULTS_VERIFIED while verification has errors.")

    rep.print_report()

    print("\n--- Summary ---")
    tier = "published" if not require_local else "complete local run"
    print(f"verification tier: {tier}")
    print(f"checks: {sum(ok for _, ok, _ in rep.checks)}/{len(rep.checks)} passed")

    if require_local and cfg.is_smoke:
        print("FAIL: --require-complete-local-run used with smoke config.")
        return 2

    if require_local:
        required = [
            cfg.models_dir / "scanvi_reference_main" / "model.pt",
            cfg.models_dir / "scanvi_query_main" / "model.pt",
            cfg.results_dir / "predictions" / "baseline_predictions_main.csv",
            cfg.results_dir / "predictions" / "scanvi_predictions_main.csv",
            cfg.results_dir / "metrics" / "summary_main.csv",
            cfg.results_dir / "metrics" / "confidence_coverage_main.csv",
        ]
        missing = [str(p.relative_to(PROJECT_ROOT)) for p in required if not p.exists()]
        if missing or rep.errors:
            print("\nLOCAL RUN INCOMPLETE. Missing/problematic items:")
            for m in missing:
                print(f"  - {m}")
            for e in rep.errors:
                print(f"  - {e}")
            return 1
        print("REQUIRED LOCAL-RUN ARTIFACTS PRESENT (status >= LOCAL_RUN_COMPLETE).")
        return 0

    if args.verify_published_results and rep.errors:
        print("\nPUBLISHED RESULTS VERIFICATION FAILED.")
        for e in rep.errors:
            print(f"  - {e}")
        return 1

    if rep.errors and require_local:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
