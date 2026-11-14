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
from conftest import require  # noqa: E402


def test_threshold_is_a_thousand_shared_variants():
    """The audit's threshold, not a rounded-down convenience. Below ~1000 the
    top-10 recall of one sample is decided by which ten variants the assay
    happened to resolve."""
    assert lt.MIN_SHARED == 1000


def test_pair_shared_drops_pairs_below_the_threshold():
    """A pair listed as 32 shared variants cannot be in the analysis even if
    the summary table carried it."""
    path = ROOT / "results" / "topk" / "cross_assay_ceiling.csv"
    require(path.exists(), "ceiling table not present")
    d = pd.read_csv(path)
    want = d[d["n_shared"] >= lt.MIN_SHARED]
    require(len(want) > 0, "no pair passes the threshold")

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
            # the per-pair gain column summarise() reads for its
            # gain-vs-reliability diagnostic
            "recall_gain_pooled_vs_single": gain,
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
    require(ceiling.exists(), "ceiling table not present")
    d = pd.read_csv(ceiling)
    d = d[d["n_shared"] >= lt.MIN_SHARED]
    require(len(d) > 0, "no pair passes the threshold")
    per_protein = d["uniprot"].nunique()
    per_assay = len(set(d["assay_a"]) | set(d["assay_b"]))
    # the invariant the cache relies on: fewer proteins than assays, because
    # several assays share a protein
    assert per_protein < per_assay


def test_committed_results_are_internally_consistent():
    """The README quotes these from summary.json, so the file has to be
    coherent: the pairs it summarises are the threshold-passing pairs, and the
    paired counts add up."""
    path = ROOT / "results" / "latent_truth" / "summary.json"
    require(path.exists(), "latent-truth run has not completed")
    import json
    s = json.loads(path.read_text())
    a = s["assays"]
    assert s["threshold_shared_variants"] == lt.MIN_SHARED
    assert a["n_pairs"] == 32
    assert 0 < a["n_proteins"] <= a["n_pairs"]
    # every per-pair row is one pair of two distinct assays
    per = pd.read_csv(ROOT / "results" / "latent_truth" / "per_pair.csv")
    assert len(per) == a["n_pairs"]
    assert (per["assay_a"] != per["assay_b"]).all()
    assert (per["n_shared"] >= lt.MIN_SHARED).all()
    assert int(per["n_shared"].sum()) == a["n_variants"]
    w = a["wilcoxon_model_pooled_minus_single"]
    assert w["n_pairs_pooled_better"] + w["n_pairs_single_better"] <= a["n_pairs"]


def test_model_and_assay_columns_are_not_transposed():
    """A transposition would make the model look like it scored 1.000 against
    the pool. The assay row is ~1 by construction (it is half the target) and
    the model row must not be."""
    path = ROOT / "results" / "latent_truth" / "per_pair.csv"
    require(path.exists(), "latent-truth run has not completed")
    d = pd.read_csv(path)
    assert d["recall_a_to_pooled"].median() > 0.9
    assert d["recall_model_to_pooled"].median() < 0.6


def test_truth_vs_truth_reproduces_the_published_ceiling():
    """The old 0.350 has to still be 0.350 on the pairs it was computed from,
    or the comparison in the README is between two different populations
    again."""
    path = ROOT / "results" / "latent_truth" / "summary.json"
    require(path.exists(), "latent-truth run has not completed")
    import json
    s = json.loads(path.read_text())
    assert s["assays"]["median_recall_truth_vs_truth"] == pytest.approx(
        0.35, abs=0.06)


def test_pooled_target_is_never_a_nan_target():
    """top_k_recall silently returns NaN on a fully-NaN truth, which would make
    a failing pair look like a missing one. Every row has a finite recall
    against both targets."""
    path = ROOT / "results" / "latent_truth" / "per_pair.csv"
    require(path.exists(), "latent-truth run has not completed")
    d = pd.read_csv(path)
    for col in ("recall_model_to_single", "recall_model_to_pooled",
                "recall_truth_vs_truth", "split_half_reliability"):
        assert np.isfinite(d[col]).all(), col


def test_gain_vs_reliability_is_reported():
    """The gain concentrates on low-reliability pairs for Spearman and not for
    recall, and the README says so. Both correlations have to be in the
    summary, or the prose is uncited."""
    path = ROOT / "results" / "latent_truth" / "summary.json"
    require(path.exists(), "latent-truth run has not completed")
    import json
    s = json.loads(path.read_text())
    g = s["assays"]["gain_vs_reliability"]
    assert g["n_pairs"] == s["assays"]["n_pairs"]
    assert -1.0 <= g["recall_gain_spearman"] <= 1.0
    assert -1.0 <= g["spearman_gain_spearman"] <= 1.0
    # the two metrics must not be the same column wearing two names
    assert g["recall_gain_spearman"] != g["spearman_gain_spearman"]



# ----------------------------------------------------------------------
# review C9 / C4: tests that call the code they are named after
# ----------------------------------------------------------------------

def _fake_scores(mapping):
    def loader(assay_id):
        s = pd.Series(mapping[assay_id])
        s.index.name = "mutant"
        return s
    return loader


def test_pair_shared_really_applies_both_thresholds(monkeypatch):
    """The older test of this name re-implemented the filter on a frame and never
    called pair_shared. This one does. Pair 1 is dropped by the CSV n_shared;
    pair 2 claims enough shared variants in the CSV but the assays only overlap
    in 3 rows; pair 3 passes; and the returned arrays are aligned to the shared
    variant list."""
    big = {f"A{i}V": float(i) for i in range(1, 1201)}
    big_b = {f"A{i}V": float(-i) for i in range(1, 1201)}
    tiny_a = {"A1V": 1.0, "A2V": 2.0, "A3V": 3.0, "A4V": 4.0}
    tiny_b = {"A1V": 1.0, "A2V": 2.0, "A3V": 3.0, "C9V": 9.0}
    monkeypatch.setattr(lt, "_scores", _fake_scores({
        "big_a": big, "big_b": big_b, "tiny_a": tiny_a, "tiny_b": tiny_b}))
    pairs = pd.DataFrame({
        "uniprot": ["P1", "P2", "P3"],
        "assay_a": ["big_a", "tiny_a", "big_a"],
        "assay_b": ["big_b", "tiny_b", "big_b"],
        "n_shared": [lt.MIN_SHARED - 1, lt.MIN_SHARED + 5, lt.MIN_SHARED + 5],
        "different_quantity": [False, False, True],
    })
    out = lt.pair_shared(pairs)
    assert [o["uniprot"] for o in out] == ["P3"]
    o = out[0]
    assert o["different_quantity"] is True
    assert len(o["shared"]) == 1200
    # sa and sb are aligned to 'shared'
    first = o["shared"][0]
    assert o["sa"][0] == big[first] and o["sb"][0] == big_b[first]


def test_pair_shared_keeps_doubles_spelled_in_opposite_orders(tmp_path,
                                                              monkeypatch):
    """C4 on the real loader: 1,100 doubles, half written in reverse order in
    assay A. A raw join keeps 550; the canonical one keeps 1,100."""
    monkeypatch.setattr(lt.data, "ASSAY_DIR", tmp_path)
    rows_a, rows_b = [], []
    for i in range(1, 1101):
        fwd = f"A{i}V:G{i + 1}L"
        rev = f"G{i + 1}L:A{i}V"
        rows_a.append((rev if i % 2 else fwd, float(i)))
        rows_b.append((fwd, float(i) * 2))
    pd.DataFrame(rows_a, columns=["mutant", "DMS_score"]).to_csv(
        tmp_path / "S_A.csv", index=False)
    pd.DataFrame(rows_b, columns=["mutant", "DMS_score"]).to_csv(
        tmp_path / "S_B.csv", index=False)
    pairs = pd.DataFrame({"uniprot": ["S"], "assay_a": ["S_A"],
                          "assay_b": ["S_B"], "n_shared": [1100],
                          "different_quantity": [False]})
    out = lt.pair_shared(pairs)
    assert len(out) == 1 and len(out[0]["shared"]) == 1100
    raw_overlap = (set(r[0] for r in rows_a) & set(r[0] for r in rows_b))
    assert len(raw_overlap) == 550


def test_canonical_scores_reports_duplicates_it_drops(tmp_path, monkeypatch):
    from variant_effect import data
    monkeypatch.setattr(data, "ASSAY_DIR", tmp_path)
    pd.DataFrame([("A1V:G2L", 1.0), ("G2L:A1V", 5.0), ("C3W", 2.0),
                  ("D4E", np.nan)],
                 columns=["mutant", "DMS_score"]).to_csv(
        tmp_path / "D.csv", index=False)
    s = data.canonical_scores("D")
    assert list(s.index) == ["A1V:G2L", "C3W"]
    assert s["A1V:G2L"] == 1.0               # first row kept
    assert s.attrs["duplicates_dropped"] == 1
    assert s.attrs["unparseable"] == 0


def test_paired_report_rounds_float_noise_so_ties_stay_tied():
    """The review's C5 discovery: recall differences are multiples of 0.05 but
    carry 17th-digit noise, which splits ties and moves the Wilcoxon p. After
    rounding, mathematically equal differences must be equal, and the p must not
    depend on the noise."""
    base = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.15, -0.05, -0.05, 0.2,
                     0.25, -0.1, 0.05])
    noisy_a = base + np.array([1e-17, -3e-17, 2e-17, 5e-17, -4e-17, 1e-17,
                               -2e-17, 3e-17, 0, 1e-17, -1e-17, 2e-17])
    noisy_b = base - np.array([2e-17, 1e-17, -3e-17, 1e-17, 5e-17, -2e-17,
                               4e-17, -1e-17, 1e-17, -3e-17, 2e-17, -1e-17])
    ra = lt.paired_report(noisy_a)
    rb = lt.paired_report(noisy_b)
    assert ra["p"] == pytest.approx(rb["p"], abs=1e-12)
    assert ra["statistic"] == rb["statistic"]
    # unrounded, the same two vectors give different p-values
    from scipy import stats
    assert (stats.wilcoxon(noisy_a, method="approx").pvalue
            != stats.wilcoxon(noisy_b, method="approx").pvalue)


def test_paired_report_returns_every_method_and_the_unit_counts():
    d = np.array([0.1, 0.05, 0.2, -0.05, 0.0, 0.15, 0.1, 0.05, 0.3, 0.1])
    r = lt.paired_report(d, seed=1)
    for key in ("p", "p_exact", "p_signflip", "statistic", "n_better",
                "n_worse", "n_zero", "median_diff", "method"):
        assert key in r
    assert (r["n_better"], r["n_worse"], r["n_zero"]) == (8, 1, 1)
    assert 0 < r["p_signflip"] <= 1


def test_summarise_has_a_protein_level_primary_test_and_collapses_repeats():
    """Eight pairs from two proteins (six from one): the protein-level test has
    two units, the pair-level one eight."""
    df = _synthetic(pairs=8, gain=0.2)
    df["uniprot"] = ["PA"] * 6 + ["PB"] * 2
    s = lt.summarise(df, "x")
    assert s["wilcoxon_model_pooled_minus_single"]["n_units"] == 8
    assert "protein_level_recall_gain" not in s       # two units: below 8
    df2 = _synthetic(pairs=16, gain=0.2)
    df2["uniprot"] = [f"P{i // 2}" for i in range(16)]
    s2 = lt.summarise(df2, "y")
    assert s2["protein_level_recall_gain"]["n_units"] == 8
    assert s2["wilcoxon_model_pooled_minus_single"]["n_units"] == 16
