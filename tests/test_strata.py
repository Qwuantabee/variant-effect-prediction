"""Tests for the range-matched control."""

import numpy as np
import pytest

from variant_effect import evaluate, strata


def test_range_matched_rho_removes_a_spread_artefact():
    """A stratum whose truth is spread over a much wider range gets a higher
    rank correlation from the same noise level. Planted: identical noise, but
    the 'double' stratum has 3x the truth spread. The raw gap is positive
    (singles worse); on the shared window it is near zero."""
    rng = np.random.default_rng(0)
    n = 6000
    t_s = rng.normal(0, 1, n)
    t_d = rng.normal(0, 3, n)
    p_s = t_s + rng.normal(0, 1, n)
    p_d = t_d + rng.normal(0, 1, n)
    raw_gap = evaluate.spearman(p_s, t_s) - evaluate.spearman(p_d, t_d)
    assert raw_gap < -0.15          # doubles look much better ...
    rs, rd, ns, nd = strata.range_matched_rho(p_s, t_s, p_d, t_d)
    # ... until the spread is matched: most of the gap goes (it does not all
    # go, because the shapes of the two truth distributions still differ)
    assert abs(rs - rd) < 0.5 * abs(raw_gap)
    assert ns > 50 and nd > 50


def test_range_matched_rho_is_nan_for_a_tiny_stratum():
    x = np.arange(20.0)
    rs, rd, ns, nd = strata.range_matched_rho(x, x, x, x)
    assert np.isnan(rs) and np.isnan(rd)


def test_range_window_is_the_intersection_of_central_quantiles():
    a = np.linspace(-1, 1, 1000)
    b = np.linspace(-3, 3, 1000)
    lo, hi = strata.range_window(a, b, frac=0.9)
    assert lo == pytest.approx(np.quantile(a, 0.05), abs=1e-6)
    assert hi == pytest.approx(np.quantile(a, 0.95), abs=1e-6)


def test_range_window_none_when_the_ranges_do_not_overlap():
    assert strata.range_window(np.linspace(0, 1, 100),
                               np.linspace(5, 6, 100)) is None


def test_range_matched_gap_equals_the_difference_of_rhos():
    rng = np.random.default_rng(2)
    t_s, t_d = rng.normal(0, 1, 800), rng.normal(0, 1, 800)
    p_s, p_d = t_s + rng.normal(0, 1, 800), t_d + rng.normal(0, 2, 800)
    rs, rd, _, _ = strata.range_matched_rho(p_s, t_s, p_d, t_d)
    assert strata.range_matched_gap((p_s, t_s), (p_d, t_d)) == \
        pytest.approx(rs - rd)
