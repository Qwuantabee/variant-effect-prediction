"""The different-quantity registry, and its agreement with the reference table."""

import pytest

from variant_effect import data, quantity
from conftest import require

RASK_A = "RASK_HUMAN_Weng_2022_abundance"
RASK_B = "RASK_HUMAN_Weng_2022_binding-DARPin_K55"
SPG1_A = "SPG1_STRSG_Olson_2014"
SPG1_B = "SPG1_STRSG_Wu_2016"


def test_rask_abundance_vs_binding_is_flagged_as_a_different_quantity():
    st = quantity.pair_status(RASK_A, RASK_B)
    assert st["different_quantity"] is True
    assert st["parallel"] is False
    assert any("DIFFERENT_QUANTITY" in r for r in st["reasons"])


def test_order_of_the_two_assays_does_not_matter():
    assert (quantity.pair_status(RASK_B, RASK_A)
            == quantity.pair_status(RASK_A, RASK_B))


def test_spg1_is_flagged_as_a_different_scale_not_a_different_quantity():
    st = quantity.pair_status(SPG1_A, SPG1_B)
    assert st["different_quantity"] is False
    assert st["different_scale"] is True
    assert st["parallel"] is False
    assert any("lnW" in r for r in st["reasons"])


def test_an_unlisted_pair_is_parallel():
    st = quantity.pair_status("X_A", "X_B")
    assert st["parallel"] is True and st["reasons"] == []
    assert st["reference_coarse_differs"] is None


def test_in_different_quantity_pair_reads_the_same_list():
    assert quantity.in_different_quantity_pair(RASK_A)
    assert not quantity.in_different_quantity_pair(SPG1_A)


def test_topk_script_uses_the_registry_list():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import run_topk_analysis as tk
    assert tk.DIFFERENT_QUANTITY is quantity.DIFFERENT_QUANTITY
    # 16 hand-curated pairs, plus the 5 the reference table marked different
    # and the hand list did not (Mark approved adding them 2026-11-14: the
    # coarse_selection_type differs across each, so they were never parallel
    # measures). The count is asserted exactly so an accidental deletion from
    # either group fails here.
    assert len(quantity.DIFFERENT_QUANTITY) == 21


def test_reference_table_agrees_the_rask_pair_is_expression_vs_binding():
    ref = data.reference_table()
    require(not ref.empty, "reference table not loaded")
    st = quantity.pair_status(RASK_A, RASK_B, reference=ref)
    assert st["reference_coarse_differs"] is True
    assert any("Expression" in r and "Binding" in r for r in st["reasons"])


def test_reference_disagreements_lists_pairs_both_ways():
    """The two directions of the hand-list / reference-table comparison.

    TEST-CHANGE: this asserted that CCDB Adkar_2012/Tripathi_2016 was ABSENT
    from the hand list (it appeared in unlisted_diff). Mark approved adding the
    five reference-flagged pairs on 2026-11-14, so that assertion now encodes
    the opposite of the intended state. The test keeps both directions and
    gains a check that each of the five is genuinely listed.
    """
    ref = data.reference_table()
    require(not ref.empty, "reference table not loaded")
    listed_same, unlisted_diff = quantity.reference_disagreements(ref)

    # direction 1: the hand list holds pairs the coarse label cannot express
    assert ("CAR11_HUMAN_Meitlis_2020_gof",
            "CAR11_HUMAN_Meitlis_2020_lof") in listed_same

    # direction 2: every remaining reference disagreement must be a REAL one --
    # its two assays must differ in coarse type and must not be listed. This
    # can still fail if the registry grows a pair the reference agrees on.
    coarse = ref.set_index("DMS_id")["coarse_selection_type"]
    for a, b in unlisted_diff:
        assert coarse.loc[a] != coarse.loc[b]
        assert frozenset((a, b)) not in quantity.DIFFERENT_QUANTITY

    # the five pairs Mark approved are now listed, so they can never appear as
    # an unlisted disagreement again
    for a, b in [
        ("CCDB_ECOLI_Adkar_2012", "CCDB_ECOLI_Tripathi_2016"),
        ("RL40A_YEAST_Mavor_2016", "RL40A_YEAST_Roscoe_2014"),
        ("RL40A_YEAST_Roscoe_2013", "RL40A_YEAST_Roscoe_2014"),
        ("SRC_HUMAN_Ahler_2019", "SRC_HUMAN_Nguyen_2022"),
        ("SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM",
         "SRC_HUMAN_Nguyen_2022"),
    ]:
        assert frozenset((a, b)) in quantity.DIFFERENT_QUANTITY
        assert (a, b) not in unlisted_diff

    # RASK is consistent between the two sources
    assert (RASK_A, RASK_B) not in listed_same
    assert (RASK_A, RASK_B) not in unlisted_diff
