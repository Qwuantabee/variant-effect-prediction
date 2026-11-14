"""Tests for the replicated-epistasis script.

The claim this script exists to support is narrow: on a *pooled two-assay
target of the same protein*, the independent-site scorer's accuracy on single
mutants is compared with the same scorer's accuracy on double mutants of the
same shared variant list. These tests pin the pieces that could silently make
that comparison wrong.

The failure modes worth testing are the ones that produce a plausible number
rather than an exception:

  * pooling the raw scores instead of the z-scores, which lets the assay with
    the larger scale dominate the target;
  * resampling the two strata *paired*, which estimates a different quantity
    and would narrow the interval;
  * matching variants loosely, so one variant is counted twice;
  * computing the top-decile recall against a size-dependent k, which makes a
    20,000-variant stratum and a 3,000-variant one non-comparable;
  * reporting degradation when the interval spans zero.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_epistasis_replicated as rep  # noqa: E402


def _planted(seed=0, n_single=2000, n_double=2000, gap=0.0):
    rng = np.random.default_rng(seed)
    truth_s = rng.normal(0, 1, n_single)
    truth_d = rng.normal(0, 1, n_double)
    # model sees the truth plus noise; a positive gap means the double stratum
    # is predicted worse
    model_s = truth_s + rng.normal(0, 0.7, n_single)
    model_d = truth_d + rng.normal(0, 0.7 + gap, n_double)
    return (model_s, truth_s), (model_d, truth_d)


# ----------------------------------------------------------------------
# stratification and the pooling frame
# ----------------------------------------------------------------------

def test_n_substitutions_counts_colon_separated_sites():
    assert rep.n_substitutions("A2M") == 1
    assert rep.n_substitutions("A2M:K5R") == 2
    assert rep.n_substitutions("A2M:K5R:L9P") == 3


def test_pooled_frame_uses_z_scores_not_raw_scores():
    """If pooling were done on the raw columns, an assay reported over a wider
    range would dominate the average. Planted: assay B is assay A times 50, so
    the z-scored pool must be identical to either assay's z-score."""
    import pandas as pd
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 400)
    n = 0.5 * rng.normal(0, 1, 400)
    df = pd.DataFrame({
        "mutant": [f"A{i + 1}V" for i in range(400)],
        "n_sub": 1,
        "score_A": a,
        "score_B": 50.0 * (a + n),
    })
    out = rep.pooled_frame(df, "A", "B")
    expected = (
        (a - a.mean()) / a.std(ddof=1)
        + (50.0 * (a + n) - (50.0 * (a + n)).mean()) / (50.0 * (a + n)).std(ddof=1)
    ) / 2.0
    assert np.allclose(out["pooled"].to_numpy(), expected)


def test_pooled_frame_is_nan_where_either_assay_is_missing():
    """A variant missing from one assay must not enter the target: a half-pooled
    value is a different quantity and would make the target inconsistent.

    The fixture is 20 rows because ``pooled_truth.zscore`` refuses to
    standardise fewer than 10 finite values, so a 3-row frame is all-NaN for a
    reason unrelated to what this test is about.
    """
    import pandas as pd
    n = 20
    df = pd.DataFrame({
        "mutant": [f"A{i + 1}V" for i in range(n)],
        "n_sub": 1,
        "score_A": np.linspace(0.0, 1.0, n),
        "score_B": np.linspace(0.1, 0.9, n),
    })
    df.loc[n - 1, "score_A"] = np.nan
    out = rep.pooled_frame(df, "A", "B")
    assert np.isnan(out["pooled"].iloc[n - 1])
    assert np.isfinite(out["pooled"].iloc[0])
    assert int(out["pooled"].notna().sum()) == n - 1


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------

def test_top_decile_recall_is_scale_invariant_and_size_free():
    """The metric has to be comparable between a 20k and a 3k stratum, so it is
    a share of the measured top decile, not a count."""
    rng = np.random.default_rng(2)
    n = 5000
    truth = rng.normal(0, 1, n)
    pred = truth + rng.normal(0, 0.8, n)
    r_small = rep._top_decile(pred[:500], truth[:500])
    r_big = rep._top_decile(pred, truth)
    assert 0.0 <= r_small <= 1.0
    assert 0.0 <= r_big <= 1.0
    assert abs(r_small - r_big) < 0.35


def test_top_decile_recall_is_high_for_a_perfect_predictor():
    rng = np.random.default_rng(3)
    truth = rng.normal(0, 1, 1000)
    assert rep._top_decile(truth, truth) > 0.99


def test_top_decile_recall_returns_nan_when_stratum_is_tiny():
    truth = np.arange(10.0)
    assert np.isnan(rep._top_decile(truth, truth))


# ----------------------------------------------------------------------
# the gap and its interval
# ----------------------------------------------------------------------

def test_gap_is_zero_when_the_two_strata_are_equally_predicted():
    s, d = _planted(seed=4, gap=0.0)
    point, lo, hi = rep.bootstrap_gap(s, d, n_boot=200)
    assert abs(point) < 0.05
    assert lo < 0 < hi


def test_gap_detects_a_planted_degradation_on_doubles():
    """The test the script exists to make: if doubles really are predicted
    worse, the interval must exclude zero on the positive side."""
    s, d = _planted(seed=5, gap=2.0)
    point, lo, hi = rep.bootstrap_gap(s, d, n_boot=300)
    assert point > 0.10
    assert lo > 0


def test_gap_interval_is_not_narrowed_by_pairing_the_strata():
    """The strata are disjoint variant sets. A paired bootstrap over a shared
    index would reuse the same truth values in both arms and produce a
    materially narrower interval; that would be the wrong quantity."""
    s, d = _planted(seed=6, gap=0.5)
    _, lo, hi = rep.bootstrap_gap(s, d, n_boot=400, seed=1)
    assert hi - lo > 0.02


def test_bootstrap_gap_refuses_tiny_strata():
    s, d = _planted(seed=7, n_single=10, n_double=10)
    point, lo, hi = rep.bootstrap_gap(s, d, n_boot=100)
    assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)


# ----------------------------------------------------------------------
# the verdict wording, which is what the README repeats
# ----------------------------------------------------------------------

def _gap_row(point, lo, hi, n_s=2000, n_d=2000):
    return {
        "protein": "TEST",
        "n_singles": n_s,
        "n_doubles": n_d,
        "gap_spearman": point,
        "gap_spearman_lo": lo,
        "gap_spearman_hi": hi,
    }


def _rel(pooled):
    return [{"protein": "TEST", "stratum": "doubles",
             "pooled_reliability": pooled}]


def test_verdict_says_not_distinguishable_when_the_interval_spans_zero():
    """A point estimate on either side of zero with an interval spanning zero
    must never be reported as degradation or as improvement."""
    for point in (0.03, -0.03):
        v = rep.verdict_text([], [_gap_row(point, -0.09, 0.14)], _rel(0.9))
        assert v["TEST"]["state"] == "not_distinguishable"


def test_verdict_says_degrades_only_when_the_interval_excludes_zero():
    v = rep.verdict_text([], [_gap_row(0.2, 0.05, 0.35)], _rel(0.9))
    assert v["TEST"]["state"] == "degrades"


def test_verdict_uses_the_reliability_of_the_right_protein():
    """The reliability rows carry a protein column; picking the first doubles
    row regardless of protein silently reports one assay pair's noise next to
    another protein's gap. RASK (0.52) and SPG1 (0.94) differ enough here that
    the mix-up is visible."""
    rel = [{"protein": "A", "stratum": "doubles", "pooled_reliability": 0.2},
           {"protein": "B", "stratum": "doubles", "pooled_reliability": 0.95}]
    rows = [{**_gap_row(0.0, -0.1, 0.1), "protein": "A"},
            {**_gap_row(0.0, -0.1, 0.1), "protein": "B"}]
    v = rep.verdict_text([], rows, rel)
    assert "uninformative" in v["A"]["target_note"]
    assert "uninformative" not in v["B"]["target_note"]


def test_verdict_says_better_on_doubles_for_a_negative_interval():
    v = rep.verdict_text([], [_gap_row(-0.2, -0.35, -0.05)], _rel(0.9))
    assert v["TEST"]["state"] == "better_on_doubles"


def test_verdict_flags_an_uninformative_pool():
    """A noisy target attenuates every correlation towards zero, so a null
    result against it cannot be read as evidence of no degradation."""
    v = rep.verdict_text([], [_gap_row(0.0, -0.1, 0.1)], _rel(0.25))
    assert "uninformative" in v["TEST"]["target_note"]


def test_verdict_calls_a_tiny_stratum_insufficient():
    v = rep.verdict_text([], [_gap_row(0.1, -0.5, 0.9, n_s=20, n_d=30)],
                         _rel(0.9))
    assert v["TEST"]["state"] == "insufficient"


# ----------------------------------------------------------------------
# the loader: real data, cheap assertions
# ----------------------------------------------------------------------

@pytest.mark.parametrize("key,protein,min_shared", [
    ("rask", "RASK_HUMAN", 20000),
    ("spg1", "SPG1_STRSG", 1000),
])
def test_loader_finds_the_shared_variant_set(key, protein, min_shared):
    """These are the counts the whole design rests on, so they are re-derived
    from the CSV rather than trusted from a manifest."""
    pair = rep.PAIRS[key]
    assert pair["protein"] == protein
    df = rep.load_pair(pair["assay_a"], pair["assay_b"])
    assert len(df) >= min_shared
    assert df["mutant"].is_unique
    assert set(df["n_sub"].unique()) <= {1, 2}
    assert df["n_sub"].value_counts().get(2, 0) > 0


def test_loader_never_emits_triples():
    """Wu_2016 carries 147k triples and quadruples. They must be filtered out:
    the comparison is doubles against singles, and a triple in the double
    stratum would conflate k with the target."""
    pair = rep.PAIRS["spg1"]
    df = rep.load_pair(pair["assay_a"], pair["assay_b"])
    assert (df["n_sub"] <= 2).all()


def test_shared_sets_are_disjoint_between_proteins():
    """The two proteins are reported separately and never pooled; sharing a
    variant string between them would mean the pooling boundary was wrong."""
    a = set(rep.load_pair(**{k: rep.PAIRS["rask"][k]
                             for k in ("assay_a", "assay_b")})["mutant"])
    b = set(rep.load_pair(**{k: rep.PAIRS["spg1"][k]
                             for k in ("assay_a", "assay_b")})["mutant"])
    assert not (a & b)


def test_report_tables_renderers_read_the_run_outputs():
    """The README numbers are rendered from the CSVs, not typed. This pins the
    renderer to the run's actual output so a stale or hand-edited table is
    caught here rather than in the prose."""
    import importlib.util
    path = ROOT / "scripts" / "report_tables.py"
    spec = importlib.util.spec_from_file_location("report_tables", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    body = mod.epistasis_replicated_table()
    if body is None:
        pytest.skip("run the script first: results are not present")

    per = pd.read_csv(ROOT / "results" / "epistasis_replicated"
                      / "per_stratum.csv")
    gaps = pd.read_csv(ROOT / "results" / "epistasis_replicated"
                       / "gaps.csv")
    for _, r in per.iterrows():
        assert f"{r['spearman']:+.3f}" in body
    for _, g in gaps.iterrows():
        assert f"{g['gap_spearman']:+.3f}" in body


def test_readme_epistasis_replicated_block_matches_the_run():
    """A generated block that drifted from results is worse than no block:
    the prose around it quotes these numbers."""
    readme = (ROOT / "README.md").read_text()
    begin = "<!-- BEGIN GENERATED: epistasis_replicated -->"
    end = "<!-- END GENERATED: epistasis_replicated -->"
    if begin not in readme:
        pytest.skip("marker not present")
    block = readme.split(begin, 1)[1].split(end, 1)[0]
    gaps_path = ROOT / "results" / "epistasis_replicated" / "gaps.csv"
    if not gaps_path.exists():
        pytest.skip("results not present")
    gaps = pd.read_csv(gaps_path)
    for _, g in gaps.iterrows():
        assert g["protein"] in block
        assert f"{g['gap_spearman']:+.3f}" in block


def test_f7ybw8_pair_is_declared_skipped_with_a_reason():
    assert "F7YBW8_MESOW" in rep.SKIPPED
    assert "zero shared" in rep.SKIPPED["F7YBW8_MESOW"]


def test_score_from_table_is_the_sum_of_independent_site_log_odds():
    """The whole point is that the scorer is unchanged -- a sum of per-site
    log-odds, no interaction term. A mutation at a site the table was not
    built for must raise rather than silently score zero."""
    table = {}
    for pos in range(1, 5):
        row = np.zeros(20)
        row[rep.ORDER["V"]] = -1.0
        row[rep.ORDER["A"]] = 0.0
        table[pos] = row
    single = rep.score_from_table(table, ["A2V"])
    double = rep.score_from_table(table, ["A2V:A3V"])
    assert single[0] == pytest.approx(-1.0)
    assert double[0] == pytest.approx(-2.0)
    with pytest.raises(KeyError):
        rep.score_from_table(table, ["A9V"])
