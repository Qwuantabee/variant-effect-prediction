"""Tests for position-cluster resampling and canonical mutant spelling."""

import numpy as np
import pytest

from variant_effect import evaluate
from variant_effect.resampling import (ClusterIndex, canonical_mutant,
                                       canonical_series, expand,
                                       substitution_positions)


def test_canonical_mutant_orders_substitutions_by_position():
    """The SPG1 failure: Olson writes D266W:V265S, Wu writes V265S:D266W."""
    assert canonical_mutant("D266W:V265S") == "V265S:D266W"
    assert canonical_mutant("V265S:D266W") == "V265S:D266W"


def test_canonical_mutant_leaves_singles_and_is_idempotent():
    assert canonical_mutant("A2M") == "A2M"
    once = canonical_mutant("K50R:A2M:L9P")
    assert once == "A2M:L9P:K50R"
    assert canonical_mutant(once) == once


def test_canonical_mutant_does_not_merge_distinct_variants():
    a = canonical_mutant("A2M:K5R")
    b = canonical_mutant("A2M:K5L")
    assert a != b
    assert len(set(canonical_series(["A2M:K5R", "K5R:A2M", "A2M:K5L"]))) == 2


def test_substitution_positions_are_sorted_and_distinct():
    assert substitution_positions("D266W:V265S") == (265, 266)
    assert substitution_positions("A2M") == (2,)


def test_substitution_positions_rejects_garbage():
    with pytest.raises(ValueError):
        substitution_positions("not-a-mutant")


def test_cluster_weights_are_equal_within_a_position_and_product_for_doubles():
    mutants = ["A1V", "A1G", "A1L", "K2R", "L3P", "A1V:K2R", "K2R:L3P"]
    idx = ClusterIndex.from_mutants([canonical_mutant(m) for m in mutants])
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = idx.draw(rng)
        # three singles at position 1 share one multiplicity
        assert w[0] == w[1] == w[2]
        # a double carries the product of its two positions' multiplicities
        assert w[5] == w[0] * w[3]
        assert w[6] == w[3] * w[4]


def test_cluster_weights_have_mean_one_per_variant():
    """The expected weight of every variant is 1, so the replicate has the
    same expected size as the data."""
    mutants = [f"A{p}V" for p in range(1, 41)] + [
        f"A{p}V:A{p + 1}G" for p in range(1, 40)]
    idx = ClusterIndex.from_mutants(mutants)
    rng = np.random.default_rng(1)
    w = np.mean([idx.draw(rng) for _ in range(4000)], axis=0)
    assert np.allclose(w, 1.0, atol=0.12)


def test_expand_repeats_by_weight():
    assert list(expand(np.array([2, 0, 1]))) == [0, 0, 2]


def _clustered_fixture(n_pos=30, per_pos=40, seed=0):
    """Zero within-position variation: every variant at a position shares one
    position effect in truth AND in the prediction, so a variant-level
    bootstrap sees 1,200 'independent' pairs where there are only 30."""
    rng = np.random.default_rng(seed)
    pos_truth = rng.normal(size=n_pos)
    pos_pred = 0.5 * pos_truth + rng.normal(size=n_pos)
    truth = np.repeat(pos_truth, per_pos)
    pred = np.repeat(pos_pred, per_pos)
    groups = np.repeat(np.arange(n_pos), per_pos)
    return pred, truth, groups


def test_cluster_ci_is_much_wider_than_variant_ci_on_clustered_data():
    """The review's C2: on data with zero within-position variation the
    variant-level interval is spuriously tight and the position-cluster
    interval is not."""
    pred, truth, groups = _clustered_fixture()
    lo_v, hi_v = evaluate.spearman_ci(pred, truth, n_boot=300, seed=3)
    lo_c, hi_c = evaluate.spearman_ci(pred, truth, n_boot=300, seed=3,
                                      groups=groups)
    assert (hi_c - lo_c) > 2.0 * (hi_v - lo_v)


def test_cluster_ci_matches_variant_ci_when_every_variant_is_its_own_cluster():
    rng = np.random.default_rng(5)
    truth = rng.normal(size=600)
    pred = truth + rng.normal(size=600)
    lo_v, hi_v = evaluate.spearman_ci(pred, truth, n_boot=400, seed=2)
    lo_c, hi_c = evaluate.spearman_ci(pred, truth, n_boot=400, seed=2,
                                      groups=np.arange(600))
    assert abs((hi_c - lo_c) - (hi_v - lo_v)) < 0.03


def test_cluster_ci_brackets_the_point_estimate():
    pred, truth, groups = _clustered_fixture(n_pos=60, per_pos=10, seed=8)
    rho = evaluate.spearman(pred, truth)
    lo, hi = evaluate.spearman_ci(pred, truth, n_boot=300, seed=1,
                                  groups=groups)
    assert lo <= rho <= hi
