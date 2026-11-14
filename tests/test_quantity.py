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
    assert len(quantity.DIFFERENT_QUANTITY) == 16


def test_reference_table_agrees_the_rask_pair_is_expression_vs_binding():
    ref = data.reference_table()
    require(not ref.empty, "reference table not loaded")
    st = quantity.pair_status(RASK_A, RASK_B, reference=ref)
    assert st["reference_coarse_differs"] is True
    assert any("Expression" in r and "Binding" in r for r in st["reasons"])


def test_reference_disagreements_lists_pairs_both_ways():
    ref = data.reference_table()
    require(not ref.empty, "reference table not loaded")
    listed_same, unlisted_diff = quantity.reference_disagreements(ref)
    # the hand list holds pairs the coarse label cannot express ...
    assert ("CAR11_HUMAN_Meitlis_2020_gof",
            "CAR11_HUMAN_Meitlis_2020_lof") in listed_same
    # ... and the reference table flags pairs the hand list does not hold
    assert ("CCDB_ECOLI_Adkar_2012", "CCDB_ECOLI_Tripathi_2016") in unlisted_diff
    # RASK is consistent between the two sources
    assert (RASK_A, RASK_B) not in listed_same
    assert (RASK_A, RASK_B) not in unlisted_diff
