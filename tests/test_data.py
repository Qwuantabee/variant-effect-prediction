"""Tests for the data layer.

These need the ProteinGym files, which src/variant_effect/data.py downloads
on first use (41 MB archive plus two small CSVs).
"""

import pytest

from variant_effect import data

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def reference():
    return data.reference_table()


def test_reference_has_all_assays(reference):
    assert len(reference) == 217
    for col in ("DMS_id", "target_seq", "seq_len", "coarse_selection_type"):
        assert col in reference.columns


def test_metadata_does_not_collide_with_result_columns(reference):
    """Regression test.

    assay_metadata used to return a key called "assay" holding the assay
    description. Benchmark rows are built as {"assay": dms_id, **metadata},
    so that key silently replaced the DMS id with a sentence like "Yeast
    growth", and the comparison against the official per-assay scores
    matched nothing and printed an empty table.
    """
    meta = data.assay_metadata("IF1_ECOLI_Kelsic_2016", reference=reference)
    assert "assay" not in meta
    row = {"assay": "IF1_ECOLI_Kelsic_2016", **meta}
    assert row["assay"] == "IF1_ECOLI_Kelsic_2016"


def test_load_assay_sequence_matches_mutants(reference):
    """Every mutant's wildtype letter must agree with the target sequence.

    This is the check that confirms positions are 1-indexed into target_seq,
    which is the assumption the whole scoring path rests on.
    """
    from variant_effect.mutations import check_against_sequence, parse_variant

    for assay in ("IF1_ECOLI_Kelsic_2016", "BLAT_ECOLX_Stiffler_2015"):
        df, seq = data.load_assay(assay, reference=reference)
        for mutant in df["mutant"].head(200):
            check_against_sequence(parse_variant(mutant), seq)


def test_singles_only_filter(reference):
    df_all, _ = data.load_assay("IF1_ECOLI_Kelsic_2016", reference=reference,
                                singles_only=False)
    df_singles, _ = data.load_assay("IF1_ECOLI_Kelsic_2016", reference=reference)
    assert len(df_singles) <= len(df_all)
    assert not df_singles["mutant"].str.contains(":").any()


def test_default_assays_are_all_in_the_reference(reference):
    known = set(reference["DMS_id"])
    for assay in data.DEFAULT_ASSAYS + data.HARD_ASSAYS:
        assert assay in known, assay


def test_official_scores_cover_our_assays():
    official = data.official_scores()
    assert "ESM2 (650M)" in official.columns
    for assay in data.DEFAULT_ASSAYS:
        assert assay in official.index, assay
