"""Mutation parsing. Boring to test and expensive to get wrong -- an
off-by-one here silently scores the residue next door and every downstream
correlation drops by a few points for no visible reason."""

import pytest

from variant_effect import mutations as mut

SEQ = "MKTAYIAKQR"   # 1-indexed: M1 K2 T3 A4 Y5 I6 A7 K8 Q9 R10


def test_parse_single():
    (s,) = mut.parse_variant("K2A")
    assert (s.wt, s.pos, s.mut) == ("K", 2, "A")
    assert s.idx == 1


def test_parse_multiple():
    subs = mut.parse_variant("K2A:Y5F")
    assert len(subs) == 2
    assert str(subs[1]) == "Y5F"


def test_parse_rejects_junk():
    for bad in ["", "2A", "KA", "K-2A", "k2a", "K2"]:
        with pytest.raises(ValueError):
            mut.parse_variant(bad)


def test_check_catches_off_by_one():
    """The whole point of carrying the wildtype letter in the notation."""
    mut.check_against_sequence(mut.parse_variant("K2A"), SEQ)   # correct
    with pytest.raises(ValueError, match="numbering mismatch"):
        mut.check_against_sequence(mut.parse_variant("T2A"), SEQ)


def test_check_catches_past_end():
    with pytest.raises(ValueError, match="past the end"):
        mut.check_against_sequence(mut.parse_variant("R99A"), SEQ)


def test_apply():
    subs = mut.parse_variant("K2A:Y5F")
    assert mut.apply_variant(SEQ, subs) == "MATAFIAKQR"


def test_apply_does_not_mutate_input():
    original = SEQ
    mut.apply_variant(SEQ, mut.parse_variant("K2A"))
    assert SEQ == original


def test_mutated_positions_dedupes_and_sorts():
    variants = ["Y5F", "K2A", "Y5W", "K2R"]
    assert mut.mutated_positions(variants) == [1, 4]


def test_is_single_substitution():
    assert mut.is_single_substitution("K2A")
    assert not mut.is_single_substitution("K2A:Y5F")
