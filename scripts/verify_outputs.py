#!/usr/bin/env python3
"""Verify pipeline outputs, leakage guards, schemas and integrity.

Two verification tiers, both STRICT (any failed check exits non-zero):

* Published tier (default / ``--verify-published-results``): checks only
  artifacts tracked in git — predictions, metrics, figures, manifests,
  training histories/summaries. Integrity is verified by recomputing
  SHA-256 hashes from ``results/artifact_manifest_<tag>.json`` and
  ``figures_manifest_<tag>.json``, and by recomputing accuracy / balanced
  accuracy / macro- and weighted-F1 from predictions + sealed labels and
  comparing them to the stored metrics. Passing this tier on a fresh clone
  is what ``MAIN_RESULTS_VERIFIED`` in the README means.
* Local tier (``--require-complete-local-run``): additionally checks
  git-ignored artifacts — processed h5ad files, model weights, latent h5ad —
  and re-runs split-isolation assertions on the h5ad objects. Fails on a
  fresh clone. Missing items are reported once (no duplicates).

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

from heartmap.baseline import METHOD_NAME as BASELINE_METHOD  # noqa: E402
from heartmap.baseline import prediction_columns  # noqa: E402
from heartmap.config import PROJECT_ROOT, load_config  # noqa: E402
from heartmap.metrics import evaluate_method  # noqa: E402
from heartmap.models import METHOD_NAME as SCANVI_METHOD  # noqa: E402
from heartmap.provenance import read_json, sha256_file  # noqa: E402
from heartmap.split import LeakageError, load_evaluation_labels  # noqa: E402

METRIC_TOL = 1e-9
SCORE_TOL = 1e-6


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))
        if not ok:
            self.errors.append(f"{name}: {detail}" if detail else name)

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


def _load_sealed(cfg, rep: Report) -> pd.DataFrame | None:
    """Read sealed labels via the sanctioned loader (schema-validated)."""
    if not cfg.sealed_labels_path.exists():
        return None
    try:
        return load_evaluation_labels(cfg)
    except LeakageError as exc:
        rep.check("sealed schema", False, str(exc))
        return None


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
    rep.exist("artifact manifest",
              cfg.results_dir / f"artifact_manifest_{cfg.run_tag}.json")

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
        detail = manifest.get("selection_rule_detail", {})
        rep.check(
            "manifest label-blind rule", detail.get("label_blind") is True,
            "selection rule is not label-blind",
        )
        rep.check(
            "manifest rule-revision note",
            "rule_revision_note" in detail,
            "missing rule_revision_note",
        )


def _verify_artifact_manifest(cfg, rep: Report) -> None:
    """Recompute SHA-256 of every artifact listed in the manifest."""
    path = cfg.results_dir / f"artifact_manifest_{cfg.run_tag}.json"
    if not path.exists():
        return
    payload = read_json(path)
    rep.check(
        "artifact manifest generation commit null",
        payload.get("generation_git_commit") is None,
        "generation_git_commit must be null (git not initialised at "
        "generation time)",
    )
    snapshot = payload.get("verification_snapshot", {})
    rep.check(
        "artifact manifest verification snapshot",
        bool(snapshot.get("git_commit")),
        "verification_snapshot.git_commit missing",
    )
    for name, entry in payload.get("artifacts", {}).items():
        art_path = PROJECT_ROOT / entry["path"]
        if not art_path.exists():
            rep.check(f"sha256:{name}", False, f"missing file {entry['path']}")
            continue
        actual = sha256_file(art_path)
        rep.check(
            f"sha256:{name}", actual == entry["sha256"],
            f"expected {entry['sha256'][:12]}..., got {actual[:12]}...",
        )


def _verify_published_predictions(cfg, rep: Report, stem: str,
                                  sealed: pd.DataFrame | None) -> None:
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
    if sealed is not None:
        rep.check(
            f"{stem} matches sealed ids",
            set(df.cell_id) == set(sealed.cell_id),
            "cell-id set differs from sealed labels",
        )


def _verify_published_scores(cfg, rep: Report,
                             sealed: pd.DataFrame | None) -> None:
    """scanvi_scores: ids match sealed; argmax == predicted_label; max ==
    confidence."""
    path = cfg.results_dir / "predictions" / f"scanvi_scores_{cfg.run_tag}.csv"
    pred_path = cfg.results_dir / "predictions" / f"scanvi_predictions_{cfg.run_tag}.csv"
    if not rep.exist("scores:scanvi", path) or not pred_path.exists():
        return
    scores = pd.read_csv(path)
    preds = pd.read_csv(pred_path)
    class_cols = [c for c in scores.columns if c != "cell_id"]
    rep.check("scores class columns", len(class_cols) > 0, "no class columns")
    if sealed is not None:
        rep.check(
            "scores match sealed ids",
            set(scores.cell_id) == set(sealed.cell_id),
            "cell-id set differs from sealed labels",
        )

    merged = scores.merge(
        preds[["cell_id", "predicted_label", "confidence"]],
        on="cell_id", how="inner", validate="one_to_one",
    )
    rep.check(
        "scores join coverage", len(merged) == len(scores),
        f"{len(merged)}/{len(scores)} rows matched predictions",
    )
    if merged.empty or not class_cols:
        return

    mat = merged[class_cols].to_numpy(dtype=float)
    row_max = mat.max(axis=1)
    argmax_labels = np.asarray(class_cols, dtype=object)[mat.argmax(axis=1)]
    pred_labels = merged["predicted_label"].astype(str).to_numpy()
    col_index = {c: i for i, c in enumerate(class_cols)}
    pred_idx = np.array(
        [col_index.get(c, -1) for c in pred_labels], dtype=int
    )
    rows = np.arange(len(merged))
    known = pred_idx >= 0
    score_at_pred = np.full(len(merged), np.nan)
    score_at_pred[known] = mat[rows[known], pred_idx[known]]

    rep.check(
        "scores predicted_label in class columns", bool(known.all()),
        f"{int((~known).sum())} predicted labels missing from score columns",
    )
    exact_argmax = argmax_labels == pred_labels
    tied = np.isclose(score_at_pred, row_max, rtol=0, atol=1e-12)
    rep.check(
        "scores argmax == predicted_label",
        bool((exact_argmax | tied).all()),
        f"{int((~(exact_argmax | tied)).sum())} rows disagree",
    )
    rep.check(
        "scores max == confidence",
        bool(np.allclose(row_max, merged["confidence"].to_numpy(dtype=float),
                         rtol=0, atol=SCORE_TOL)),
        "stored confidence differs from max soft score",
    )


def _verify_published_models_metadata(cfg, rep: Report) -> None:
    """Training summaries/histories and run metadata are tracked; model.pt is
    checked only in the local tier. Hash integrity is covered by the artifact
    manifest."""
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
    rep.exist("baseline summary",
              cfg.results_dir / "training" / f"baseline_summary_{cfg.run_tag}.json")
    meta_path = cfg.results_dir / f"run_metadata_{cfg.run_tag}.json"
    if rep.exist("run metadata", meta_path):
        meta = read_json(meta_path)
        git = meta.get("environment", {}).get("git", {}) or {}
        rep.check(
            "run metadata generation git null",
            git.get("commit") is None and git.get("branch") is None,
            "generation-time git fields must stay null; the later commit "
            "belongs in artifact_manifest verification_snapshot only",
        )


def _recompute_metrics(cfg, rep: Report, sealed: pd.DataFrame | None) -> None:
    """Recompute headline metrics from predictions + sealed labels and compare
    against the stored summary CSV."""
    mdir = cfg.results_dir / "metrics"
    summary_path = mdir / f"summary_{cfg.run_tag}.csv"
    if not summary_path.exists() or sealed is None:
        return
    stored = pd.read_csv(summary_path)
    manifest = read_json(cfg.split_manifest_path)
    reference_labels = manifest["reference_label_set"]
    min_support = int(cfg["evaluable_class_min_support"])

    for method, stem in ((BASELINE_METHOD, "baseline_predictions"),
                         (SCANVI_METHOD, "scanvi_predictions")):
        ppath = cfg.results_dir / "predictions" / f"{stem}_{cfg.run_tag}.csv"
        if not ppath.exists():
            continue
        preds = pd.read_csv(ppath)
        result = evaluate_method(preds, sealed, reference_labels, method,
                                 min_support)
        rec = result.summary.set_index(["method", "scope"])
        for _, row in stored[stored["method"] == method].iterrows():
            key = (method, row["scope"])
            if key not in rec.index:
                rep.check(f"recompute:{method}:{row['scope']}", False,
                          "scope missing from recomputation")
                continue
            r = rec.loc[key]
            for metric in ("n_query_cells", "n_out_of_reference_cells",
                           "accuracy", "balanced_accuracy", "macro_f1",
                           "weighted_f1", "macro_f1_evaluable"):
                ok = abs(float(r[metric]) - float(row[metric])) <= METRIC_TOL
                rep.check(
                    f"recompute {metric}:{method}:{row['scope']}", ok,
                    f"stored {row[metric]} vs recomputed {r[metric]}",
                )


def _verify_published_metrics(cfg, rep: Report,
                              sealed: pd.DataFrame | None) -> None:
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
              set(summary.method) >= {BASELINE_METHOD, SCANVI_METHOD},
              str(set(summary.method)))
    rep.check("summary scopes", set(summary.scope) == {"all", "closed_set"},
              str(set(summary.scope)))
    rep.check("metrics bounded",
              summary[["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]]
              .apply(lambda s: s.between(0, 1).all()).all(),
              "metric outside [0,1]")

    if sealed is not None:
        sealed_n = len(sealed)
        for method in (BASELINE_METHOD, SCANVI_METHOD):
            cm_path = mdir / f"confusion_{method}_all_counts_{cfg.run_tag}.csv"
            if cm_path.exists():
                cm = pd.read_csv(cm_path, index_col=0)
                rep.check(
                    f"confusion total:{method}",
                    int(cm.to_numpy().sum()) == sealed_n,
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

    _recompute_metrics(cfg, rep, sealed)


def _verify_published_figures(cfg, rep: Report) -> None:
    """Every listed figure must exist AND match its recorded SHA-256."""
    mdir = cfg.results_dir / "metrics"
    manifest = mdir / f"figures_manifest_{cfg.run_tag}.json"
    if not rep.exist("figures manifest", manifest):
        return
    payload = read_json(manifest)
    for f in payload.get("figures", []):
        main_p = cfg.figures_dir / f["file"]
        smoke_p = cfg.figures_dir / "smoke" / f["file"]
        actual = main_p if main_p.exists() else smoke_p
        if not actual.exists():
            rep.check(f"figure:{f['file']}", False,
                      str(main_p.relative_to(PROJECT_ROOT)))
            continue
        recorded = f.get("sha256")
        if not recorded:
            rep.check(f"figure sha256:{f['file']}", False, "no hash recorded")
            continue
        actual_hash = sha256_file(actual)
        rep.check(
            f"figure sha256:{f['file']}", actual_hash == recorded,
            f"expected {recorded[:12]}..., got {actual_hash[:12]}...",
        )


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
        help="strictly verify all tracked (published) artifacts; the default "
             "mode is identical",
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
    sealed = _load_sealed(cfg, rep)
    _verify_artifact_manifest(cfg, rep)
    _verify_published_predictions(cfg, rep, "baseline_predictions", sealed)
    _verify_published_predictions(cfg, rep, "scanvi_predictions", sealed)
    _verify_published_scores(cfg, rep, sealed)
    _verify_published_models_metadata(cfg, rep)
    _verify_published_metrics(cfg, rep, sealed)
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
        if "MAIN_RESULTS_VERIFIED" in txt and rep.errors:
            rep.warn(
                "README mentions MAIN_RESULTS_VERIFIED while verification "
                "has errors."
            )

    rep.print_report()

    print("\n--- Summary ---")
    tier = "complete local run" if require_local else "published"
    print(f"verification tier: {tier}")
    print(f"checks: {sum(ok for _, ok, _ in rep.checks)}/{len(rep.checks)} passed")

    if require_local:
        if cfg.is_smoke:
            print("FAIL: --require-complete-local-run used with smoke config.")
            return 2
        if rep.errors:
            print("\nLOCAL RUN INCOMPLETE. Problematic items (deduplicated):")
            for e in dict.fromkeys(rep.errors):
                print(f"  - {e}")
            return 1
        print("REQUIRED LOCAL-RUN ARTIFACTS PRESENT (status >= LOCAL_RUN_COMPLETE).")
        return 0

    # Published tier is strict by default: any error -> exit 1.
    if rep.errors:
        print("\nPUBLISHED RESULTS VERIFICATION FAILED. Problematic items:")
        for e in dict.fromkeys(rep.errors):
            print(f"  - {e}")
        return 1
    print("PUBLISHED RESULTS VERIFIED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
