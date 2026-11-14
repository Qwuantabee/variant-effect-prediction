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
from conftest import require  # noqa: E402


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
    require(body is not None, "run the script first: results are not present")

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
    require(begin in readme, "marker not present")
    block = readme.split(begin, 1)[1].split(end, 1)[0]
    gaps_path = ROOT / "results" / "epistasis_replicated" / "gaps.csv"
    require(gaps_path.exists(), "results not present")
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



# ======================================================================
# Review fixes (C1 C2 C3 C4 C8 C9): tests that fail on the old code
# ======================================================================

def _write_assay(dirpath, name, rows):
    pd.DataFrame(rows, columns=["mutant", "DMS_score"]).to_csv(
        dirpath / f"{name}.csv", index=False)


@pytest.fixture
def reversed_order_assays(tmp_path, monkeypatch):
    """Two assays of one protein: the same 40 doubles, but assay A writes half
    of them in reverse position order (Olson 2014) and assay B always writes
    them position-ordered (Wu 2016). 30 singles written identically."""
    rng = np.random.default_rng(0)
    a_rows, b_rows = [], []
    for p in range(1, 31):
        m = f"A{p}V"
        a_rows.append((m, rng.normal()))
        b_rows.append((m, rng.normal()))
    for i in range(40):
        p = i + 1
        fwd = f"A{p}V:G{p + 1}L"
        rev = f"G{p + 1}L:A{p}V"
        a_rows.append((rev if i % 2 else fwd, rng.normal()))
        b_rows.append((fwd, rng.normal()))
    monkeypatch.setattr(rep.data, "ASSAY_DIR", tmp_path)
    _write_assay(tmp_path, "X_A", a_rows)
    _write_assay(tmp_path, "X_B", b_rows)
    return "X_A", "X_B"


def test_load_pair_keeps_doubles_spelled_in_opposite_orders(
        reversed_order_assays):
    """The SPG1 bug (C4): a join on the raw string drops every double the two
    labs wrote in different substitution orders. 20 of the 40 doubles are
    reversed in assay A."""
    a, b = reversed_order_assays
    merged = rep.load_pair(a, b)
    assert int((merged.n_sub == 2).sum()) == 40
    assert int((merged.n_sub == 1).sum()) == 30
    assert merged["mutant"].is_unique


def test_join_report_exact_is_strictly_smaller_than_canonical_on_reversal(
        reversed_order_assays):
    a, b = reversed_order_assays
    j = rep.join_report(a, b)
    assert j["canonical_intersection"] == 70
    assert j["exact_string_intersection"] == 50
    assert j["dropped_by_exact_string_join"] == 20
    assert j["exact_is_subset_of_canonical"] is True


def test_load_pair_counts_duplicates_instead_of_hiding_them(tmp_path,
                                                            monkeypatch):
    """C12: a variant written twice (here as two orderings) keeps its first
    row, and the count is reported rather than silent."""
    rows_a = [("A1V", 1.0), ("A2V:G3L", 1.0), ("G3L:A2V", 9.0)]
    rows_b = [("A1V", 1.0), ("A2V:G3L", 2.0)]
    monkeypatch.setattr(rep.data, "ASSAY_DIR", tmp_path)
    _write_assay(tmp_path, "D_A", rows_a)
    _write_assay(tmp_path, "D_B", rows_b)
    merged = rep.load_pair("D_A", "D_B")
    assert len(merged) == 2
    row = merged[merged["mutant"] == "A2V:G3L"].iloc[0]
    assert row["score_D_A"] == 1.0                 # the first row kept
    assert rep.join_report("D_A", "D_B")["D_A"]["duplicates_dropped_first_kept"] == 1


def _read_raw(assay):
    d = pd.read_csv(rep.data.ASSAY_DIR / f"{assay}.csv",
                    usecols=["mutant", "DMS_score"])
    d = d[np.isfinite(d["DMS_score"])]
    d = d[d["mutant"].str.count(":") <= 1]
    return d


def _independent_canonical(m):
    """Written separately from resampling.canonical_mutant on purpose."""
    return ":".join(sorted(m.split(":"), key=lambda s: int(s[1:-1])))


def test_spg1_canonical_join_matches_an_independent_recomputation(require_data):
    """On the real SPG1 pair the exact-string join loses doubles. The expected
    set is recomputed here with a separate implementation, and the loader's
    docstring claim (exact == canonical) is checked, not trusted."""
    pair = rep.PAIRS["spg1"]
    require_data((rep.data.ASSAY_DIR / f"{pair['assay_a']}.csv").exists(),
                 "SPG1 assays are not downloaded (run src/variant_effect/data.py)")
    a, b = _read_raw(pair["assay_a"]), _read_raw(pair["assay_b"])
    exact = set(a["mutant"]) & set(b["mutant"])
    canon = ({_independent_canonical(m) for m in a["mutant"]}
             & {_independent_canonical(m) for m in b["mutant"]})
    merged = rep.load_pair(pair["assay_a"], pair["assay_b"])
    assert set(merged["mutant"]) == canon
    assert len(canon) > len(exact) > 0
    j = rep.join_report(pair["assay_a"], pair["assay_b"])
    assert j["exact_string_intersection"] == len(exact)
    assert j["canonical_intersection"] == len(canon)


def test_rask_exact_and_canonical_joins_agree(require_data):
    """RASK never needed the canonical join; recorded so that is a measurement."""
    pair = rep.PAIRS["rask"]
    require_data((rep.data.ASSAY_DIR / f"{pair['assay_a']}.csv").exists(),
                 "RASK assays are not downloaded")
    j = rep.join_report(pair["assay_a"], pair["assay_b"])
    assert j["exact_string_intersection"] == j["canonical_intersection"]


def test_f7ybw8_pair_is_declared_skipped_because_the_data_say_so(require_data):
    """The skip reason is re-derived from the CSVs on the canonical key, not
    read back from the constant that states it."""
    a, b = "F7YBW8_MESOW_Aakre_2015", "F7YBW8_MESOW_Ding_2023"
    require_data((rep.data.ASSAY_DIR / f"{a}.csv").exists(),
                 "F7YBW8 assays are not downloaded")
    j = rep.join_report(a, b)
    assert j["canonical_intersection"] == 0
    assert j["exact_string_intersection"] == 0
    assert "F7YBW8_MESOW" in rep.SKIPPED


def test_both_pairs_are_tagged_not_parallel_by_the_registry():
    for key in ("rask", "spg1"):
        p = rep.PAIRS[key]
        st = rep.quantity.pair_status(p["assay_a"], p["assay_b"])
        assert st["parallel"] is False, key
    assert rep.quantity.pair_status(
        rep.PAIRS["rask"]["assay_a"],
        rep.PAIRS["rask"]["assay_b"])["different_quantity"] is True


def _synthetic_two_assay_frame(n_pos=40, per_double=25, seed=0):
    """Singles + doubles on shared positions, with a per-position-PAIR effect
    in the doubles' truth and no within-pair variation (the clustered case)."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(1, n_pos + 1):
        t = rng.normal()
        rows.append((f"A{p}V", 1, t + rng.normal(0, .2), t + rng.normal(0, .2),
                     t + rng.normal(0, 0.8)))
    for p in range(1, n_pos):
        eff = rng.normal()
        pe = 0.5 * eff + rng.normal()
        for _ in range(per_double):
            rows.append((f"A{p}V:G{p + 1}L", 2, eff + rng.normal(0, .02),
                         eff + rng.normal(0, .02), pe))
    # the doubles rows have repeated mutant strings by construction; make them
    # unique with a distinct third-site-free spelling per row
    df = pd.DataFrame(rows, columns=["mutant", "n_sub", "A", "B", "model"])
    df["mutant"] = [f"{m}" for m in df["mutant"]]
    return df


def test_stratified_reports_four_targets_with_position_clustered_intervals():
    """Shape of the output, and the review's central point (C2): the interval
    comes from a position bootstrap, so it is wider than the variant-level one
    on a frame whose doubles carry only a per-pair effect."""
    df = _synthetic_two_assay_frame()
    df = df.rename(columns={"A": "score_AA", "B": "score_BB"})
    df = rep.pooled_frame(df, "AA", "BB")
    pair = {"protein": "SYN", "role": "test", "assay_a": "AA", "assay_b": "BB"}
    rows, gaps = rep.stratified(pair, df, df["model"].to_numpy(), 40,
                                n_boot=120)
    assert {g["target"] for g in gaps} == set(rep.TARGETS)
    assert {r["target"] for r in rows} == set(rep.TARGETS)
    assert len(rows) == 2 * len(rep.TARGETS)
    g = next(x for x in gaps if x["target"] == rep.PRIMARY_TARGET)
    assert g["is_primary_target"] and g["n_position_clusters"] == 40
    assert np.isfinite(g["range_matched_gap"])

    s = df[df.n_sub == 1]
    d = df[df.n_sub == 2]
    kw = dict(n_boot=120, seed=0)
    pt_v, lo_v, hi_v = rep.bootstrap_gap((s["model"], s["pooled_rank"]),
                                         (d["model"], d["pooled_rank"]), **kw)
    pt_c, lo_c, hi_c = rep.bootstrap_gap(
        (s["model"], s["pooled_rank"]), (d["model"], d["pooled_rank"]),
        mutants_s=s["mutant"].tolist(), mutants_d=d["mutant"].tolist(), **kw)
    assert pt_v == pytest.approx(pt_c)
    assert (hi_c - lo_c) > 1.5 * (hi_v - lo_v)
    # and stratified() used the clustered one
    assert (g["gap_spearman_hi"] - g["gap_spearman_lo"]) > 1.5 * (hi_v - lo_v)


def test_bootstrap_gap_with_mutants_is_a_position_bootstrap_not_variant_level():
    """Zero within-position variation: identical rows per position. The
    variant-level interval is spuriously tight."""
    n_pos, per = 30, 40
    rng = np.random.default_rng(3)
    tr = np.repeat(rng.normal(size=n_pos), per)
    pr = np.repeat(rng.normal(size=n_pos) * 0.3, per) + 0.3 * tr
    muts_d = [f"A{p}V:G{p + 1}L" for p in range(1, n_pos + 1) for _ in range(per)]
    muts_s = [f"A{p}V" for p in range(1, 41)]
    ts_ = rng.normal(size=40)
    ps_ = ts_ + rng.normal(size=40)
    _, lo_v, hi_v = rep.bootstrap_gap((ps_, ts_), (pr, tr), n_boot=150)
    _, lo_c, hi_c = rep.bootstrap_gap((ps_, ts_), (pr, tr), n_boot=150,
                                      mutants_s=muts_s, mutants_d=muts_d)
    assert (hi_c - lo_c) > 1.5 * (hi_v - lo_v)


def test_reliability_report_withholds_spearman_brown_for_a_non_parallel_pair():
    df = _synthetic_two_assay_frame(per_double=3)
    df = df.rename(columns={"A": "score_AA", "B": "score_BB"})
    df = rep.pooled_frame(df, "AA", "BB")
    par = rep.reliability_report(df, "AA", "BB", parallel=True)
    non = rep.reliability_report(df, "AA", "BB", parallel=False)
    assert np.isfinite(par[0]["pooled_reliability"])
    assert all(np.isnan(r["pooled_reliability"]) for r in non)
    assert all(np.isfinite(r["split_half_spearman"]) for r in non)
    assert {"pearson_z_a_z_b", "corr_pooled_z_with_a"} <= set(non[0])


def test_verdict_prints_a_loud_note_for_a_non_parallel_pair_and_the_controlled_state():
    g = {**_gap_row(0.1, -0.2, 0.3), "target": "pooled_rank",
         "pair_parallel": False, "pair_reasons": "hand-curated list",
         "range_matched_gap": 0.01, "range_matched_gap_lo": -0.1,
         "range_matched_gap_hi": 0.12}
    rel = [{"protein": "TEST", "stratum": "doubles",
            "pooled_reliability": np.nan, "parallel_pair": False}]
    v = rep.verdict_text([], [g], rel)["TEST"]
    assert v["pair_note"].startswith("NOT A REPLICATE PAIR")
    assert "not parallel" in v["target_note"]
    assert v["range_matched_state"] == "not_distinguishable"


def test_verdict_keys_non_primary_targets_separately():
    base = _gap_row(0.0, -0.1, 0.1)
    v = rep.verdict_text([], [{**base, "target": "assay_a"},
                              {**base, "target": rep.PRIMARY_TARGET}], _rel(0.9))
    assert set(v) == {"TEST", "TEST|assay_a"}


def test_pooled_frame_adds_the_rank_pool_and_it_differs_from_the_z_pool():
    rng = np.random.default_rng(4)
    a = rng.normal(size=300)
    b = np.exp(2 * (a + rng.normal(size=300)))
    df = pd.DataFrame({"mutant": [f"A{i + 1}V" for i in range(300)],
                       "n_sub": 1, "score_A": a, "score_B": b})
    out = rep.pooled_frame(df, "A", "B")
    assert "pooled_rank" in out and np.isfinite(out["pooled_rank"]).all()
    assert not np.allclose(out["pooled_rank"], out["pooled"])
    # rank pool is invariant to the monotone transform of assay B
    out2 = rep.pooled_frame(df.assign(score_B=np.log(b)), "A", "B")
    assert np.allclose(out["pooled_rank"], out2["pooled_rank"])


def test_masked_table_does_not_load_the_model_when_everything_is_cached(
        tmp_path):
    """Rescoring from the cache must not need ESM-2 (and must not need a GPU)."""
    positions = [3, 5]
    arr = np.arange(40, dtype=np.float32).reshape(2, 20)
    path = tmp_path / "P.npz"
    np.savez(path, positions=np.array(positions), table=arr)
    table = rep.masked_table(None, "ACDEFGHIK", positions, path)
    assert set(table) == {3, 5}
    assert np.allclose(table[5], arr[1])


def test_verdict_refuses_to_read_an_interval_resampled_from_a_handful_of_positions():
    """SPG1 sits on 4 positions. A position bootstrap over 4 clusters is not an
    interval for anything, so the state must not be 'degrades' even when the
    percentile interval happens to exclude zero."""
    g = {**_gap_row(0.2, 0.05, 0.35), "n_position_clusters": 4,
         "range_matched_gap": 0.2, "range_matched_gap_lo": 0.1,
         "range_matched_gap_hi": 0.3}
    v = rep.verdict_text([], [g], _rel(0.9))["TEST"]
    assert v["state"] == "insufficient_clusters"
    assert v["range_matched_state"] == "insufficient_clusters"
    g["n_position_clusters"] = 185
    v = rep.verdict_text([], [g], _rel(0.9))["TEST"]
    assert v["state"] == "degrades"
