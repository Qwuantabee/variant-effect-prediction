"""Tests for the findings figures in scripts/make_figures.py.

A figure here is a claim a reader takes in faster than the prose, so the checks
pin the part that could silently draw the wrong thing:

  * every plotted number is read from a committed results/ file, and each data
    function stops if its own arithmetic (a count, a median, a recomputed U)
    disagrees with the stored summary;
  * the figures say what the README says -- 32 of 32 pairs gain Spearman, no
    top-10 recall test crosses p = 0.05, 17 of 32 pairs sit below
    truth-vs-truth, family-level p = 0.0213 -- and a regenerated artefact that
    moved one of them fails here instead of leaving a stale picture;
  * the superseded variant-level RASK interval kept for figure d is not a
    hand-typed number: it is reproduced by resampling the committed scored
    frame by variant, with the same function that draws it by position.

Nothing here loads a model or re-runs the latent-truth or epistasis analyses.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import make_figures as mf  # noqa: E402
from conftest import require  # noqa: E402


def _have(*paths):
    for p in paths:
        require(Path(p).exists(), f"{Path(p).relative_to(ROOT)} missing "
                                  "(it is committed under results/)")


# ---- a) rank gain ----------------------------------------------------------

def test_rank_gain_every_pair_and_protein_gains():
    _have(mf.LT / "per_pair.csv", mf.LT / "summary.json")
    per, prot, summ = mf.rank_gain_data()
    assert len(per) == 32 and int((per["gain"] > 0).sum()) == 32
    assert len(prot) == 20 and int((prot > 0).sum()) == 20
    assert round(per["gain"].median(), 3) == 0.045
    assert round(prot.median(), 3) == 0.051
    same = per[~per["different_quantity"]]
    assert len(same) == 13 and (same["gain"] > 0).all()
    assert round(same["gain"].median(), 3) == 0.038


def test_rank_gain_stops_when_the_summary_disagrees(tmp_path, monkeypatch):
    """The data function is a check, not just a loader: a per_pair.csv edited
    after summary.json was written must stop the figure."""
    _have(mf.LT / "per_pair.csv", mf.LT / "summary.json")
    d = tmp_path / "latent_truth"
    d.mkdir()
    (d / "summary.json").write_text((mf.LT / "summary.json").read_text())
    per = pd.read_csv(mf.LT / "per_pair.csv")
    per.loc[0, "spearman_model_to_pooled"] = per.loc[0, "spearman_model_to_single"] - 0.2
    per.to_csv(d / "per_pair.csv", index=False)
    monkeypatch.setattr(mf, "LT", d)
    with pytest.raises(SystemExit, match="disagrees"):
        mf.rank_gain_data()


# ---- b) recall vs rank p-values -------------------------------------------

def test_pvalues_recall_never_crosses_rank_always_does():
    _have(mf.LT / "summary.json")
    rows = mf.pvalue_data()
    assert len(rows) == 6                       # 3 subsets x 2 levels
    rec = [p for r in rows for p in r["recall"]]
    spe = [p for r in rows for p in r["spearman"]]
    assert len(rec) == len(spe) == 18
    assert min(rec) >= 0.05 and round(min(rec), 3) == 0.059
    assert max(spe) < 0.012
    primary = {(r["level"]): r for r in rows if r["primary_subset"]}
    assert [round(p, 3) for p in primary["pair"]["recall"]] == [0.066, 0.071, 0.102]
    assert [round(p, 3) for p in primary["protein"]["recall"]] == [0.102, 0.117, 0.167]


# ---- c) model vs truth-vs-truth --------------------------------------------

def test_ceiling_counts_use_the_rounded_tie_rule():
    _have(mf.LT / "per_pair.csv", mf.LT / "summary.json")
    d = mf.ceiling_data()
    s, p = d["single"], d["pooled"]
    assert (s["below"], s["on"], s["above"]) == (17, 1, 14)
    assert (p["below"], p["on"], p["above"]) == (16, 1, 15)
    assert s["below"] + s["on"] + s["above"] == d["n"] == 32
    # recall is a multiple of 0.05: float subtraction must not invent a gap
    assert round(d["mean_truth"], 3) == 0.408
    assert round(s["mean_model"], 3) == 0.292


# ---- d) resampling unit -----------------------------------------------------

def test_resampling_unit_same_estimate_different_verdict():
    _have(mf.REP / "gaps.csv", mf.SUPERSEDED_GAPS)
    d = mf.resampling_unit_data()
    sp = d["spearman"]
    assert round(sp["point"], 3) == -0.088
    v_lo, v_hi = sp["variant"]
    p_lo, p_hi = sp["position"]
    assert v_hi < 0 < p_hi                     # variant excludes zero, position does not
    assert (p_hi - p_lo) / (v_hi - v_lo) > 3   # about three times wider
    assert d["n_doubles"] == 20124 and d["positions"] == 187
    rc = d["recall"]
    assert rc["variant"][1] < 0 and rc["position"][1] < 0   # recall: excludes zero both ways


def test_superseded_variant_level_interval_is_reproducible():
    """Figure d draws the first run's variant-level interval from a verbatim
    copy of results/epistasis_replicated/gaps.csv at 081e21b. Reproduce it:
    resample the 20,124 doubles and 2,948 singles of the committed scored frame
    as independent variants (no position labels) with the repo's own
    gap_bootstrap, same B and seed as the script. Position labels would give
    the current interval; leaving them out gives the superseded one."""
    scored = mf.REP / "scored_RASK_HUMAN.csv"
    _have(scored, mf.SUPERSEDED_GAPS, mf.REP / "gaps.csv", mf.REP / "manifest.json")
    import json

    from variant_effect import evaluate, resampling
    n_boot = json.loads((mf.REP / "manifest.json").read_text())["n_boot"]
    df = pd.read_csv(scored)
    s, d = df[df["n_sub"] == 1], df[df["n_sub"] == 2]

    def rho(pred, truth):
        return evaluate.spearman(np.asarray(pred, float), np.asarray(truth, float))

    single = (s["model"].to_numpy(), s["pooled"].to_numpy())
    double = (d["model"].to_numpy(), d["pooled"].to_numpy())
    got = resampling.gap_bootstrap(rho, single, double, n_boot=n_boot, seed=0)
    old = pd.read_csv(mf.SUPERSEDED_GAPS)
    old = old[old["protein"] == "RASK_HUMAN"].iloc[0]
    point, lo, hi = got[0], got[1], got[2]
    assert point == pytest.approx(old["gap_spearman"], abs=1e-6)
    assert lo == pytest.approx(old["gap_spearman_lo"], abs=1e-6)
    assert hi == pytest.approx(old["gap_spearman_hi"], abs=1e-6)


# ---- e) range matching ------------------------------------------------------

def test_range_match_means_and_rask():
    _have(mf.EPI / "paired_gaps.csv", mf.REP / "gaps.csv", mf.REP / "manifest.json")
    d = mf.range_match_data()
    assert round(d["mean_raw"], 3) == -0.135 and round(d["mean_rm"], 3) == -0.054
    assert round(d["rask_raw"], 3) == -0.088 and round(d["rask_rm"], 3) == 0.012
    assert len(d["left"]) == 5 and len(d["right"]) == 8
    # the figure's title says range matching widens GRB2: check the data does
    by = {r[0].split()[0]: r for r in d["left"]}
    assert by["GRB2"][4] > by["GRB2"][1] > 0
    assert by["HIS7"][4] > by["HIS7"][1]       # HIS7 moves toward zero from -0.51


# ---- f) family test ---------------------------------------------------------

def test_family_data_reproduces_the_stored_tests():
    _have(mf.FAM / "per_method_viral_gap.csv", mf.FAM / "family_viral_gap.csv",
          mf.FAM / "family_separation.csv")
    d = mf.family_data()
    sep = d["sep"]
    assert (sep["n_no_alignment"], sep["n_alignment"]) == (23, 19)
    assert (sep["n_families_no_alignment"], sep["n_families_alignment"]) == (6, 8)
    assert round(sep["p_family_level_exact"], 4) == 0.0213
    assert round(sep["p_no_alignment_worse"], 5) == 0.00047
    assert len(d["ck"]) == 42 and d["ck"]["family"].nunique() == 14


# ---- g) residual vs total variance -----------------------------------------

def test_only_his7_residual_exceeds_the_total():
    _have(mf.EPI / "per_stratum_all.csv")
    a = mf.residual_data()
    flagged = [i for i, r in a.iterrows() if r["residual_exceeds_total"]]
    assert len(flagged) == 1 and flagged[0].startswith("HIS7")
    his = a.loc[flagged[0]]
    assert his["nonadditive_residual_sd_z"] > his["double_total_sd_z"]
    assert round(a["nonadditive_residual_var_share"].mean(), 3) == 0.586


# ---- h) scaling by split ---------------------------------------------------

def test_scaling_split_matches_the_published_columns():
    _have(mf.RESULTS / "esm2_scaling" / "subgroups.csv")
    d = mf.scaling_split_data()
    assert len(d) == 10 and d.index[-1] == "Taxa_Virus"   # sorted by the 15B change
    assert round(d.loc["Taxa_Virus", "d_3B"], 3) == 0.033
    assert round(d.loc["Taxa_Virus", "d_15B"], 3) == 0.072
    worse = ["Function_Stability", "Function_Binding", "Taxa_Human"]
    better = ["Taxa_Virus", "Function_OrganismalFitness"]
    assert (d.loc[worse, ["d_3B", "d_15B"]] < 0).all().all()
    assert (d.loc[better, ["d_3B", "d_15B"]] > 0).all().all()


# ---- every figure draws -----------------------------------------------------

@pytest.mark.parametrize("name", sorted(mf.FINDINGS))
def test_each_findings_figure_writes_a_readable_png(name, tmp_path, monkeypatch):
    """Run each figure function end to end into a temp directory: a PNG at
    150 dpi, wide enough to read (the single-panel figure is 8.6 in wide, the
    multi-panel ones 11 in), not a blank canvas."""
    _have(mf.LT / "per_pair.csv", mf.REP / "gaps.csv", mf.SUPERSEDED_GAPS,
          mf.EPI / "paired_gaps.csv", mf.FAM / "family_separation.csv",
          mf.RESULTS / "esm2_scaling" / "subgroups.csv")
    monkeypatch.setattr(mf, "FIG", tmp_path)
    mf.FINDINGS[name]()
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1
    import matplotlib.image as mpimg
    img = mpimg.imread(pngs[0])
    h, w = img.shape[:2]
    assert w >= 1200 and h >= 650                  # 8 x 4.3 in or more at 150 dpi
    assert float(img[..., :3].std()) > 0.05        # not a blank canvas


# ---- the README says what the figures say ------------------------------------

README = ROOT / "README.md"
FINDINGS_PNGS = ("pooled_target_rank_gain.png", "recall_vs_rank_significance.png",
                 "model_vs_truth_vs_truth.png", "rask_resampling_unit.png",
                 "range_matching_gap.png", "family_test_unit.png",
                 "his7_residual_vs_variance.png", "esm2_scaling_by_split.png",
                 "at_a_glance.png")


def _glance():
    text = README.read_text()
    i = text.index("## At a glance")
    j = text.index("\n## ", i + 5)
    return text[i:j]


def test_readme_embeds_every_findings_figure_and_links_the_results_page():
    require(README.exists(), "README.md missing")
    text = README.read_text()
    for name in FINDINGS_PNGS:
        assert (mf.FIG / name).exists(), f"figures/{name} is not committed"
        assert f"figures/{name}" in text, f"README does not embed figures/{name}"
    assert "RESULTS.md" in text
    # the relocated tables live on the page, not in two places that can disagree
    for name in ("topk_ceiling", "latent_truth", "sim_controls", "epistasis",
                 "family_viral_gap", "scaling_paired", "scaling_subgroups"):
        assert f"<!-- BEGIN GENERATED: {name} -->" not in text


def test_readme_at_a_glance_numbers_come_from_the_artefacts():
    """The four-bullet summary at the top is typed prose, so pin it: every
    number in it is recomputed here from the committed results files."""
    _have(mf.LT / "per_pair.csv", mf.LT / "summary.json", mf.REP / "gaps.csv",
          mf.SUPERSEDED_GAPS)
    g = _glance().replace("\u2212", "-")
    per, prot, summ = mf.rank_gain_data()
    n_pairs, n_prot = len(per), len(prot)
    assert f"{int((per['gain'] > 0).sum())} of {n_pairs}" in g
    assert f"{int((prot > 0).sum())} of {n_prot}" in g
    assert f"{per['gain'].median():+.3f}" in g and f"{prot.median():+.3f}" in g
    rows = {(r["level"]): r for r in mf.pvalue_data() if r["primary_subset"]}
    # the README's convention: four decimals below 0.1, three above
    fmt = lambda ps: " / ".join(f"{p:.4f}" if p < 0.1 else f"{p:.3f}" for p in ps)
    assert fmt(rows["pair"]["recall"]) in g and fmt(rows["protein"]["recall"]) in g
    allp = [p for r in mf.pvalue_data() for p in r["recall"]]
    assert f"{min(allp):.3f}" in g
    c = mf.ceiling_data()["single"]
    assert f"{c['below']} of 32" in g and f"on {c['above']} above" in g
    u = mf.resampling_unit_data()["spearman"]
    s = lambda x: f"{x:+.3f}"
    assert s(u["point"]) in g
    assert f"[{s(u['variant'][0])}, {s(u['variant'][1])}]" in g
    assert f"[{s(u['position'][0])}, {s(u['position'][1])}]" in g
    assert "20,124" in g and "187" in g
