"""Label sealing and training/evaluation separation."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from heartmap.split import (LeakageError, assert_no_overlap,
                            assert_query_sealed,
                            assert_sealed_labels_match, load_model_split,
                            make_split)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TRAIN_SCRIPTS = ["run_baseline.py", "train_reference.py", "map_query.py"]


def test_query_has_no_true_or_fine_labels(synthetic_atlas, synthetic_config):
    _, query, _, _, _, _ = make_split(synthetic_atlas, synthetic_config)
    assert "cell_type" not in query.obs.columns
    assert "cell_states" not in query.obs.columns  # fine labels also removed
    assert set(query.obs["labels_scanvi"].astype(str)) == {"Unknown"}


def test_sealing_assertion_detects_truth_column(synthetic_atlas, synthetic_config):
    _, query, _, _, _, _ = make_split(synthetic_atlas, synthetic_config)
    query.obs["cell_type"] = "something"
    with pytest.raises(LeakageError):
        assert_query_sealed(query, synthetic_config)


def test_sealing_assertion_detects_non_unknown(synthetic_atlas, synthetic_config):
    _, query, _, _, _, _ = make_split(synthetic_atlas, synthetic_config)
    query.obs["labels_scanvi"] = "0"
    with pytest.raises(LeakageError):
        assert_query_sealed(query, synthetic_config)


def test_overlap_assertions(synthetic_atlas, synthetic_config):
    ref, query, _, _, _, _ = make_split(synthetic_atlas, synthetic_config)
    assert_no_overlap(ref, query, "donor")  # should pass silently
    # Inject a shared value on a fake key -> generic overlap guard must trip.
    ref.obs["_probe"] = "shared"
    query.obs["_probe"] = "shared"
    with pytest.raises(LeakageError):
        assert_no_overlap(ref, query, "_probe")


def test_sealed_labels_match_query(synthetic_atlas, synthetic_config):
    _, query, _, _, _, _ = make_split(synthetic_atlas, synthetic_config)
    sealed = assert_sealed_labels_match(query, synthetic_config)
    assert sealed.cell_id.is_unique
    assert (sealed.true_cell_type.astype(str).isin({"0", "1", "2"})).all()


def test_training_scripts_never_read_sealed_labels():
    """Static guard: training/mapping sources cannot touch evaluation inputs."""
    forbidden_substrings = ["query_eval_labels", "heartmap.metrics",
                            "evaluate import", "true_cell_type",
                            "load_split"]
    for name in TRAIN_SCRIPTS:
        src = (SCRIPTS / name).read_text()
        for bad in forbidden_substrings:
            assert bad not in src, f"{name} must not reference {bad}"


def test_training_scripts_do_not_import_evaluate():
    """Parse AST: ensure evaluate.py is never imported by training scripts."""
    for name in TRAIN_SCRIPTS:
        tree = ast.parse((SCRIPTS / name).read_text())
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
            if isinstance(node, ast.Import):
                imported_modules.update(n.name for n in node.names)
        assert not any("evaluate" in m for m in imported_modules), name


def test_training_scripts_use_load_model_split():
    """Parse AST: every training script must import load_model_split."""
    for name in TRAIN_SCRIPTS:
        tree = ast.parse((SCRIPTS / name).read_text())
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported_names.update(a.name for a in node.names)
        assert "load_model_split" in imported_names, (
            f"{name} must import load_model_split, not load_split"
        )


def test_load_model_split_does_not_read_sealed_labels(
    synthetic_atlas, synthetic_config, monkeypatch
):
    """Dynamic guard: load_model_split must never call pd.read_csv on the
    sealed labels file."""
    make_split(synthetic_atlas, synthetic_config)
    sealed_path = str(synthetic_config.sealed_labels_path)
    original = pd.read_csv

    def guard(path, *args, **kwargs):
        if sealed_path in str(path):
            raise AssertionError(
                "load_model_split must not read sealed labels "
                f"({sealed_path})"
            )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guard)
    # Should succeed without triggering the guard.
    reference, query, manifest = load_model_split(synthetic_config)
    assert reference.n_obs + query.n_obs == synthetic_atlas.n_obs
    assert manifest["query_donor"] == "A"
