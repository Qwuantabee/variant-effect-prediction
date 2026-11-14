"""Tests for the assay-level statistics behind the top-k analysis.

The numbers these functions produce are the ones the README uses to say the
model is at the assay's ceiling rather than below it, so the estimators need
to be pinned. Each test here checks a property that a plausible but wrong
implementation would violate -- a noise estimate that does not cancel a scale
difference, a reliability that does not fall under range restriction, a robust
check on the pair finding.
"""

import numpy as np
import pandas as pd
import pytest

from variant_effect import assay_stats


def test_noise_sd_recovers_a_planted_error():
    """Two measurements of the same truth with known error sd."""
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 1, 4000)
    a = truth + rng.normal(0, 0.3, 4000)
    b = truth + rng.normal(0, 0.3, 4000)
    # sd of each error is 0.3, so sd(a - b) = 0.3*sqrt(2), and the estimator
    # divides by sqrt(2) to get back one measurement's error
    assert 0.27 < assay_stats.noise_sd(a, b) < 0.33


def test_noise_sd_is_zero_for_identical_measurements():
    a = np.linspace(0, 1, 500)
    assert assay_stats.noise_sd(a, a) == pytest.approx(0.0, abs=1e-12)


def test_zscore_noise_cancels_a_scale_difference_between_assays():
    """The whole point of the z-unit estimator.

    Two assays that are affine rescalings of each other measure the same
    ordering in different units -- a fluorophore read in arbitrary units
    against a growth rate. In raw units sd(a - b) is enormous; in z units the
    disagreement is only the noise.
    """
    rng = np.random.default_rng(1)
    truth = rng.normal(0, 1, 4000)
    a = truth + rng.normal(0, 0.05, 4000)
    b = 12.5 * truth + 300 + rng.normal(0, 0.05 * 12.5, 4000)

    assert assay_stats.noise_sd(a, b) > 5.0          # raw units: nonsense
    z = assay_stats.noise_sd_z(a, b)
    assert z < 0.10, f"z units should cancel the rescaling, got {z}"


def test_zscore_handles_nan():
    v = assay_stats.zscore(np.array([1.0, 2.0, np.nan, 3.0] * 5))
    assert np.isnan(v[2])
    assert np.isfinite(v[0])


def test_reliability_falls_when_the_subset_is_range_restricted():
    """Reliability is 1 - sigma^2/var, so shrinking the subset's variance must
    shrink its reliability. This is the mechanism the top-k claim rests on:
    the top decile spans less of the scale, so it resolves less."""
    rng = np.random.default_rng(2)
    whole = rng.normal(0, 1, 5000)
    subset = whole[whole > 1.3]          # a range-restricted slice
    sigma = 0.5
    assert assay_stats.reliability_z(whole, sigma) > 0.7
    assert assay_stats.reliability_z(subset, sigma) < 0.0


def test_reliability_is_positive_for_a_spread_out_subset():
    rng = np.random.default_rng(3)
    v = rng.normal(0, 1.0, 2000)
    assert assay_stats.reliability_z(v, 0.2) == pytest.approx(0.96, abs=0.01)


def test_top_decile_mask_matches_evaluate_top_k_recall_threshold():
    """The subset this module calls "the top decile" has to be the subset
    evaluate.top_k_recall scores against, or the reliability is attached to
    the wrong group. Both use quantile(1 - frac), inclusive."""
    truth = np.array([0.1, 0.5, 0.9, 1.0, 2.0] * 20)
    mask = assay_stats.top_decile_mask(truth)
    thr = np.quantile(truth, 0.9)
    assert np.array_equal(mask, truth >= thr)


def test_top_decile_mask_includes_ties_at_the_threshold():
    """A DMS assay with many variants sitting on the same score -- common at a
    ceiling -- puts more than 10% of variants inside the decile, and the
    metric has to agree with that rather than dropping them arbitrarily."""
    truth = np.concatenate([np.zeros(900), np.full(100, 5.0)])
    mask = assay_stats.top_decile_mask(truth)
    assert mask.sum() == 100
    # a plateauesque assay: half the variants share the top value
    plateau = np.concatenate([np.zeros(500), np.full(500, 1.0)])
    assert assay_stats.top_decile_mask(plateau).sum() == 500


def test_top_decile_overlap_is_one_for_identical_and_low_for_disjoint():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, 2000)
    assert assay_stats.top_decile_overlap(a, a) == pytest.approx(1.0)
    assert assay_stats.top_decile_overlap(a, -a) < 0.05


def test_paired_ranks_agreement_is_spearman_on_the_shared_part():
    from scipy import stats as sps
    rng = np.random.default_rng(5)
    a = rng.normal(0, 1, 500)
    b = a + rng.normal(0, 1, 500)
    assert assay_stats.paired_ranks_agreement(a, b) == pytest.approx(
        sps.spearmanr(a, b).statistic)


def test_decile_shape_sees_a_plateau():
    """An assay whose top decile is a flat ceiling: same score for everything
    at the top. spread_ratio near zero, at_max_frac large."""
    plateau = np.concatenate([np.linspace(0, 1, 900), np.full(100, 1.0)])
    shape = assay_stats.decile_shape(plateau)
    assert shape["at_max_frac"] >= 0.09
    assert shape["distinct_frac"] < 1.0


def test_decile_shape_on_a_spread_assay():
    rng = np.random.default_rng(6)
    v = rng.normal(0, 1, 5000)
    shape = assay_stats.decile_shape(v)
    assert 0.1 < shape["spread_ratio"] < 0.6
    assert shape["n_top"] >= 400
    assert shape["at_max_frac"] < 0.01


def test_decile_shape_returns_empty_for_a_tiny_assay():
    assert assay_stats.decile_shape(np.arange(10.0)) == {}


def test_ceiling_script_finds_pairs_that_share_enough_variants():
    """The pair finder is the foundation of the cross-assay ceiling; if it
    silently returns pairs with a handful of shared variants the ceiling is
    noise. Checked against the reference table that ships in data/."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import run_topk_analysis as rta

    data_dir = root / "data" / "proteingym"
    if not data_dir.exists() or not any(data_dir.glob("*.csv")):
        pytest.skip("ProteinGym assays not downloaded")

    ref = pd.read_csv(root / "data" / "DMS_substitutions.csv")
    pairs = rta.replicate_pairs(ref)
    assert len(pairs) >= 20
    assert all(p["n_shared"] >= rta.MIN_SHARED for p in pairs)
    assert all(p["a"] != p["b"] for p in pairs)
    # every pair is a pair of the same protein
    for p in pairs:
        same = ref[ref["DMS_id"].isin([p["a"], p["b"]])]["UniProt_ID"].nunique()
        assert same == 1


def test_scope_audit_notation_regex_rejects_indels():
    """The audit's whole purpose is to certify "no indels", so its notation
    pattern has to actually reject them. A pattern that silently accepts
    `A2del` would turn the claim into a tautology."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import run_scope_audit as rsa

    for good in ("A2M", "A2*", "A2M:K5R", "W12Y:A33V:G90S"):
        assert rsa.SUB_RE.match(good), good
    for bad in ("A2del", "A2insG", "A2fs", "A2_3delinsAA", "del2", "", "A2M:K5"):
        assert not rsa.SUB_RE.match(bad), bad


def test_scope_audit_summary_is_consistent_with_the_reference_table():
    """If the audit has run, its own count of multi-substitution variants must
    agree with ProteinGym's reference column. Two independent counts that
    disagree mean the scan is not reading what it claims to read."""
    import json
    from pathlib import Path
    path = (Path(__file__).resolve().parents[1]
            / "results" / "scope" / "summary.json")
    if not path.exists():
        pytest.skip("scope audit has not been run")
    s = json.loads(path.read_text())
    assert s["variants_scanned"] == s["reference_total_mutants"]
    assert s["multi_substitution_variants"] == s["reference_multiple_mutants"]
    assert s["variants_not_substitution_notation"] == 0
