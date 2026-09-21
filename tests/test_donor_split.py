"""Deterministic donor rule and split integrity."""

from __future__ import annotations

import pandas as pd
import pytest

from heartmap.split import (donor_cell_counts, donor_statistics,
                            load_split, make_split, select_query_donor)


def test_largest_donor_by_cells_selected(synthetic_atlas, synthetic_config):
    donor, stats = select_query_donor(synthetic_atlas, synthetic_config)
    # Fixture: A=240, B=240, C=200; tie between A/B -> lexicographic A.
    assert donor == "A"
    top = stats.sort_values(["n_cells", "donor_id"], ascending=[False, True])
    assert top.iloc[0]["donor_id"] == "A"


def test_selection_is_deterministic(synthetic_atlas, synthetic_config):
    d1, s1 = select_query_donor(synthetic_atlas, synthetic_config)
    d2, s2 = select_query_donor(synthetic_atlas, synthetic_config)
    assert d1 == d2
    pd.testing.assert_frame_equal(s1, s2)


def test_donor_cell_counts_is_label_blind(synthetic_atlas, synthetic_config):
    """donor_cell_counts must not read cell_type — only the donor key."""
    counts = donor_cell_counts(synthetic_atlas, synthetic_config)
    assert list(counts.columns) == ["donor_id", "n_cells"]
    assert counts["n_cells"].sum() == synthetic_atlas.n_obs
    # Largest first, tie-break by donor ID.
    assert counts.iloc[0]["donor_id"] == "A"


def test_selection_works_without_cell_type_column(
    synthetic_atlas, synthetic_config
):
    """Label-blindness: delete the entire cell_type column and selection
    must still pick the same donor."""
    atlas_no_labels = synthetic_atlas.copy()
    del atlas_no_labels.obs["cell_type"]
    assert "cell_type" not in atlas_no_labels.obs.columns

    donor, stats = select_query_donor(atlas_no_labels, synthetic_config)
    assert donor == "A"  # same donor as with labels present
    assert list(stats.columns) == [
        "donor_id", "n_cells", "n_cell_types", "cell_type_composition",
    ]
    assert (stats["cell_type_composition"] == "").all()
    assert (stats["n_cell_types"] == 0).all()
    assert stats["n_cells"].sum() == synthetic_atlas.n_obs


def test_selection_invariant_to_label_values(synthetic_atlas, synthetic_config):
    """Scrambling every label must not change the donor-cell counts."""
    counts_before = donor_cell_counts(synthetic_atlas, synthetic_config)
    scrambled = synthetic_atlas.copy()
    scrambled.obs["cell_type"] = "X"
    counts_after = donor_cell_counts(scrambled, synthetic_config)
    pd.testing.assert_frame_equal(counts_before, counts_after)


def test_make_split_isolates_donor_and_cells(synthetic_atlas, synthetic_config):
    ref, query, sealed, manifest, stats, ref_full = make_split(
        synthetic_atlas, synthetic_config
    )
    assert manifest["query_donor"] == "A"
    assert "A" not in manifest["reference_donor_ids"]
    assert set(ref.obs["donor"].astype(str)) == {"B", "C"}
    assert set(query.obs["donor"].astype(str)) == {"A"}
    assert not (set(ref.obs_names) & set(query.obs_names))
    # sealed labels equal query truth
    assert len(sealed) == query.n_obs
    assert list(sealed.columns) == ["cell_id", "true_cell_type"]
    # hashes exist and are stable
    assert len(manifest["reference_cell_ids_sha256"]) == 64
    assert len(manifest["query_cell_ids_sha256"]) == 64
    # label set recorded
    assert set(manifest["reference_label_set"]) == {"0", "1", "2"}


def test_split_roundtrip(synthetic_atlas, synthetic_config):
    make_split(synthetic_atlas, synthetic_config)
    ref2, query2, sealed2, manifest2 = load_split(synthetic_config)
    assert ref2.n_obs + query2.n_obs == synthetic_atlas.n_obs
    assert set(query2.obs["labels_scanvi"].astype(str)) == {"Unknown"}


def test_statistics_table_columns(synthetic_atlas, synthetic_config):
    stats = donor_statistics(synthetic_atlas, synthetic_config)
    for col in ("donor_id", "n_cells", "n_cell_types"):
        assert col in stats.columns


def test_manifest_records_label_blind_rule(synthetic_atlas, synthetic_config):
    _, _, _, manifest, _, _ = make_split(synthetic_atlas, synthetic_config)
    assert manifest["selection_rule"] == "largest_donor_by_cells"
    assert manifest["selection_rule_detail"]["label_blind"] is True
