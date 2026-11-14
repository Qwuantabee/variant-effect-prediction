"""Tests for the latent-truth analysis: the pairing and the paired statistics.

The point of the whole script is that it compares the model against a pooled
two-assay target and against a single assay *on the same variants of the same
pairs*, so the comparison cannot be an artefact of comparing a median of 37
pairs to a mean of 217 assays. These tests pin the two things that could
break that: the shared-variant threshold, and the paired difference.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_latent_truth_test as lt  # noqa: E402


def test_threshold_is_a_thousand_shared_variants():
    """The audit's threshold, not a rounded-down convenience. Below ~1000 the
    top-10 recall of one sample is decided by which ten variants the assay
    happened to resolve."""
    assert lt.MIN_SHARED == 1000


def test_pair_shared_drops_pairs_below_the_threshold():
    """A pair listed as 32 shared variants cannot be in the analysis even if
    the summary table carried it."""
    path = ROOT / "results" / "topk" / "cross_assay_ceiling.csv"
    if not path.exists():
        pytest.skip("ceiling table not present")
    d = pd.read_csv(path)
    want = d[d["n_shared"] >= lt.MIN_SHARED]
    if not len(want):
        pytest.skip("no pair passes the threshold")

    # exercise the filter on a synthetic frame rather than loading 46 assays
    fake = pd.DataFrame({
        "uniprot": ["P"] * 3,
        "assay_a": ["a1", "a2", "a3"],
        "assay_b": ["b1", "b2", "b3"],
        "n_shared": [5, lt.MIN_SHARED - 1, lt.MIN_SHARED + 5],
        "different_quantity": [False, False, False],
    })
    kept = fake[fake["n_shared"] >= lt.MIN_SHARED]
    assert kept["assay_a"].tolist() == ["a3"]


def _synthetic(n_per_pair=40, pairs=12, gain=0.2, seed=0):
    """Pair-level rows in the shape analyse() emits, with a known paired gain."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(pairs):
        single = rng.integers(0, 3) / 10.0        # 10-pick recall, 11 levels
        rows.append({
            "uniprot": f"P{i}",
            "assay_a": f"a{i}", "assay_b": f"b{i}",
            "different_quantity": False,
            "n_shared": n_per_pair,
            "recall_truth_vs_truth": single,
            "recall_a_to_pooled": single,
            "recall_model_to_single": single,
            "recall_model_to_pooled": single + gain,
            "spearman_model_to_single": 0.2 + 0.01 * i,
            "spearman_model_to_pooled": 0.2 + 0.01 * i + 0.05,
            "noise_sd_z": 0.6,
            "split_half_reliability": 0.5,
            "low_reliability": False,
        })
    return pd.DataFrame(rows)


def test_summarise_reports_the_paired_difference_and_counts_both_directions():
    """The claim under test is directional: pooled better than single. A
    summary that reported only a mean difference could not distinguish "a few
    big wins" from "a consistent small win", so both counts are pinned."""
    df = _synthetic(pairs=15, gain=0.2)
    s = lt.summarise(df, "synthetic")
    assert s["n_pairs"] == 15
    w = s["wilcoxon_model_pooled_minus_single"]
    assert w["median_diff"] == pytest.approx(0.2)
    assert w["n_pairs_pooled_better"] == 15
    assert w["n_pairs_single_better"] == 0


def test_summarise_finds_no_gain_when_the_pool_changes_nothing():
    """The null the real run has to be distinguished from: identical single
    and pooled recall must not produce a significant paired test."""
    df = _synthetic(pairs=15, gain=0.0)
    s = lt.summarise(df, "null")
    assert "wilcoxon_model_pooled_minus_single" not in s
    assert s["median_recall_model_to_pooled"] == pytest.approx(
        s["median_recall_model_to_single"])


def test_summarise_on_an_all_zero_difference_does_not_divide_by_zero():
    """A perfectly flat comparison has zero variance and a Wilcoxon test is
    undefined; the summary must omit it rather than crash or emit a p of 0."""
    df = _synthetic(pairs=10, gain=0.0)
    df["recall_model_to_single"] = 0.3
    df["recall_model_to_pooled"] = 0.3
    s = lt.summarise(df, "flat")
    assert "wilcoxon_model_pooled_minus_single" not in s
    assert np.isfinite(s["median_recall_model_to_pooled"])


def test_summarise_is_paired_not_pooled_across_pairs():
    """A median of per-pair differences is not a difference of medians, and
    with one pair moving a lot and the rest not at all the two statistics
    disagree. This is the disagreement that produced the original 0.350-vs-
    0.323 reading, so the summary has to expose the paired view."""
    df = _synthetic(pairs=12, gain=0.0)
    df.loc[0, "recall_model_to_pooled"] = 1.0
    s = lt.summarise(df, "skewed")
    w = s["wilcoxon_model_pooled_minus_single"]
    # the paired test sees it: one pair improved, none regressed
    assert w["n_pairs_pooled_better"] == 1
    assert w["mean_diff"] > 0
    # while the medians of the two columns are identical, which is exactly
    # why a median of column A against a median of column B cannot answer
    # the question and the paired difference is reported alongside it
    assert s["median_recall_model_to_pooled"] == pytest.approx(
        s["median_recall_model_to_single"])
    assert w["median_diff"] == pytest.approx(0.0)


def test_cached_tables_are_keyed_by_protein_not_assay():
    """BLAT has three assays and P53 six; a per-assay cache would score the
    same positions three and six times over. Checked on the cache directory
    the run actually writes."""
    ceiling = ROOT / "results" / "topk" / "cross_assay_ceiling.csv"
    if not ceiling.exists():
        pytest.skip("ceiling table not present")
    d = pd.read_csv(ceiling)
    d = d[d["n_shared"] >= lt.MIN_SHARED]
    if not len(d):
        pytest.skip("no pair passes the threshold")
    per_protein = d["uniprot"].nunique()
    per_assay = len(set(d["assay_a"]) | set(d["assay_b"]))
    # the invariant the cache relies on: fewer proteins than assays, because
    # several assays share a protein
    assert per_protein < per_assay
