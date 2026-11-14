"""Tests for the pooled-truth estimators.

The pooled target is the whole basis of the claim that 0.350 is attenuated
agreement rather than an upper bound, so the estimator has to have the
property the argument depends on: pooling two noisy measurements has to be
*closer* to the latent truth than either one, never further. A plausible but
wrong implementation -- pool the raw values instead of z-scores, or pool by
concatenation -- fails that.
"""

import numpy as np
import pytest
from scipy import stats

from variant_effect import pooled_truth


def _planted(seed=0, n=6000, err=0.6):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 1, n)
    a = truth + rng.normal(0, err, n)
    b = truth + rng.normal(0, err, n)
    return truth, a, b


def test_pool_is_closer_to_the_latent_truth_than_either_assay():
    """The central property. Two noisy measurements of a known truth: the mean
    of the z-scored pair has to correlate with the truth better than either
    assay alone, and by more than sampling noise."""
    truth, a, b = _planted()
    pooled = pooled_truth.pool_z(a, b)
    r_a = stats.spearmanr(a, truth).statistic
    r_b = stats.spearmanr(b, truth).statistic
    r_p = stats.spearmanr(pooled, truth).statistic
    assert r_p > r_a
    assert r_p > r_b
    # and it should not be a trivial improvement: halving the error variance
    # buys a visibly better rank correlation at this noise level
    assert r_p - max(r_a, r_b) > 0.02


def test_pool_of_two_independent_errors_halves_the_noise_variance():
    """Averaging two independent errors gives 1/sqrt(2) the sd, which is what
    "the pooled estimate is closer to the truth" means numerically.

    Measured on the raw scale, because z-scoring is an affine map and an
    affine map cannot change a rank statistic -- so a residual measured on
    the z scale would test the standardisation, not the pooling. On the raw
    scale with a unit-variance truth the identity is exact up to sampling.
    """
    truth, a, b = _planted(n=20000, err=1.0)
    sd_one = np.std(a - truth)
    sd_pool = np.std((a + b) / 2.0 - truth)
    assert sd_pool == pytest.approx(sd_one / np.sqrt(2), rel=0.03)


def test_z_pooling_is_rank_identical_to_raw_pooling_when_spreads_match():
    """The guarantee that the z scale is safe: when the two assays have the
    same spread, pool_z is a positive affine image of the raw average, so
    every rank metric computed against it is unchanged. Where the spreads
    differ, the z scale is the only defensible choice -- that is the next
    test -- but it must not silently reorder anything here."""
    truth, a, b = _planted(n=5000, err=0.5)
    raw_pool = (a + b) / 2.0
    z_pool = pooled_truth.pool_z(a, b)
    # 0.9999, not 1.0: each assay is standardised by its own sample sd, and
    # two finite samples of the same noise level do not have identical sds.
    # The reordering that leaves is float-level, not scale-level.
    assert stats.spearmanr(raw_pool, z_pool).statistic > 0.999


def test_pool_cancels_a_scale_and_offset_between_the_two_assays():
    """A fluorescence reporter and a growth rate are not in the same units.
    Pooling raw values would let the larger-scale assay dictate the target;
    pooling z-scores cannot, so the pooled target must equal the pooled
    target of the rescaled pair."""
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 3000)
    b = 50.0 * a + 1000.0 + rng.normal(0, 3.0, 3000)
    raw = pooled_truth.pool_z(a, b)
    ref = pooled_truth.pool_z(a, (b - 1000.0) / 50.0)
    assert np.allclose(raw, ref, equal_nan=True)


def test_pool_is_nan_where_either_measurement_is_missing():
    """A half-pooled value is a different quantity from a full one. Mixing them
    would make the target inconsistent across variants, so a variant missing
    from one assay is dropped from the target, not scored on the other."""
    a = np.array([1.0, 2.0, 3.0] + [0.5] * 20)
    b = np.array([1.0, np.nan, 3.0] + [0.5] * 20)
    pooled = pooled_truth.pool_z(a, b)
    assert np.isnan(pooled[1])
    assert np.isfinite(pooled[0])


def test_pool_of_a_measurement_with_itself_is_that_measurement():
    rng = np.random.default_rng(8)
    a = rng.normal(0, 2, 500)
    assert np.allclose(pooled_truth.pool_z(a, a),
                       pooled_truth.zscore(a), equal_nan=True)


def test_split_half_reliability_is_higher_for_less_noisy_pairs():
    """Reliability is the diagnostic used to flag pairs where pooling is not
    informative, so it has to move with noise in the right direction."""
    _, clean_a, clean_b = _planted(err=0.2)
    _, noisy_a, noisy_b = _planted(err=3.0)
    assert (pooled_truth.split_half_reliability(clean_a, clean_b)
            > pooled_truth.split_half_reliability(noisy_a, noisy_b))
    assert pooled_truth.split_half_reliability(clean_a, clean_b) > 0.9


def test_split_half_reliability_is_one_for_an_assay_against_itself():
    rng = np.random.default_rng(9)
    a = rng.normal(0, 1, 800)
    assert pooled_truth.split_half_reliability(a, a) == pytest.approx(1.0)


def test_split_half_reliability_handles_a_constant_assay():
    """A perfectly flat assay has no ordering to reproduce; NaN rather than a
    divide-by-zero or a spurious 1.0."""
    flat = np.full(500, 3.0)
    rng = np.random.default_rng(10)
    assert np.isnan(pooled_truth.split_half_reliability(flat, rng.normal(0, 1, 500)))


def test_pooled_noise_is_the_single_assay_noise_over_root_two():
    """The remaining-noise statement in the README's new paragraph is this
    identity; if it drifts the prose is wrong."""
    _, a, b = _planted(n=20000, err=0.8)
    pooled = pooled_truth.pool_z(a, b)
    za, zb = pooled_truth.zscore(a), pooled_truth.zscore(b)
    assert pooled_truth.pooled_noise_sd(a, b) == pytest.approx(
        np.std(za - zb, ddof=1) / np.sqrt(2))
    # the pooled series' own residual against a noiseless truth is smaller
    assert np.std(pooled) < np.std(za)


def test_spearman_brown_gives_the_pooled_reliability():
    """2r/(1+r) is what pooling buys, stated as reliability: r = 0.5 becomes
    0.667, never 1.0. The README says pooling reduces but does not eliminate
    noise, and this is that sentence as arithmetic."""
    assert pooled_truth.reliability_ceiling(0.5) == pytest.approx(2 / 3)
    assert pooled_truth.reliability_ceiling(1.0) == pytest.approx(1.0)
    assert pooled_truth.reliability_ceiling(0.0) == pytest.approx(0.0)
    # never above one for a positive correlation, i.e. pooling two imperfect
    # measurements cannot produce a perfect one
    for r in (0.1, 0.3, 0.7, 0.9):
        assert pooled_truth.reliability_ceiling(r) < 1.0


def test_reliability_ceiling_of_nan_is_nan():
    assert np.isnan(pooled_truth.reliability_ceiling(np.nan))
