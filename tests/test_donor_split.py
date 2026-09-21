"""Deterministic donor rule and split integrity."""

from __future__ import annotations

import pandas as pd
import pytest

from heartmap.split import (make_split, select_query_donor,
                            donor_statistics, load_split)


def test_largest_eligible_donor_selected(synthetic_atlas, synthetic_config):
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


def test_too_high_threshold_means_no_eligible(synthetic_atlas, synthetic_config):
    synthetic_config.raw["minimum_query_cells"] = 10_000
    with pytest.raises(RuntimeError):
        select_query_donor(synthetic_atlas, synthetic_config)


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
