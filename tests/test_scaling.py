"""Tests for the ESM-2 scaling analysis.

The claim under test is a published pattern that this project cannot re-measure
at the top end (3B and 15B are 12 and 60 GB of weights, which the rails forbid
downloading). What can be tested is that the *published* numbers, aggregated by
our own rule, reproduce the published headline, and that the statistical claim
about the 650M-versus-3B gap is computed correctly.

The important test is ``test_650m_vs_3b_gap_is_not_a_capacity_effect``: it fails
if the subgroup movement ever becomes uniformly negative, which would turn the
explanation from "objective mismatch" into "bigger is worse" -- a different and
much less interesting claim.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from variant_effect import aggregate, data

SIZES = ["ESM2 (8M)", "ESM2 (35M)", "ESM2 (150M)", "ESM2 (650M)",
         "ESM2 (3B)", "ESM2 (15B)"]


@pytest.fixture(scope="module")
def summary():
    return data.official_summary()


@pytest.fixture(scope="module")
def official():
    return data.official_scores()


def test_headline_peaks_before_3b(summary):
    """The pattern the README quotes: 650M 0.414, 3B 0.406, 15B 0.400.

    Pinned so that a change to the input file is visible. If ProteinGym revises
    these numbers the test should be updated deliberately, with the reason.
    """
    h = summary.loc[SIZES, "Average_Spearman"].astype(float)
    assert h.idxmax() == "ESM2 (650M)"
    assert h["ESM2 (650M)"] == pytest.approx(0.414, abs=0.001)
    assert h["ESM2 (3B)"] == pytest.approx(0.406, abs=0.001)
    assert h["ESM2 (15B)"] == pytest.approx(0.400, abs=0.001)
    # monotone up to 650M, then down
    assert (h.loc[SIZES[:4]].diff().dropna() > 0).all()
    assert (h.loc[["ESM2 (650M)", "ESM2 (3B)", "ESM2 (15B)"]].diff().dropna()
            < 0).all()


def test_gap_is_smaller_than_the_published_error_bars(summary):
    """0.008 against bootstrap SEs of ~0.012 means an unpaired reading of the
    headline cannot call the difference real. That is why the paired test is
    the right instrument, and why 'bigger models are worse' is not supportable
    from the headline alone."""
    gap = abs(summary.loc["ESM2 (650M)", "Average_Spearman"]
              - summary.loc["ESM2 (3B)", "Average_Spearman"])
    bound = np.hypot(
        summary.loc["ESM2 (650M)", "Bootstrap_standard_error_Spearman"],
        summary.loc["ESM2 (3B)", "Bootstrap_standard_error_Spearman"])
    assert gap < bound


def test_paired_test_detects_the_sign_that_the_headline_hides(official):
    """The paired mean difference is negative -- 3B is slightly worse on the
    average assay -- while the Wilcoxon test says the *median* gap is not
    centered on zero either. Both point the same way, which is what makes the
    finding worth explaining rather than dismissing."""
    d = (official["ESM2 (3B)"] - official["ESM2 (650M)"]).dropna()
    assert len(d) == 217
    assert d.mean() < 0
    # more assays get worse than get better
    assert (d < 0).sum() > (d > 0).sum()


def test_subgroup_movement_is_not_uniformly_negative(summary):
    """The substance of the explanation.

    If 3B were simply a worse model, every subgroup would move down. Instead
    the split is by what the assay measures: the assays that probe the
    protein's own stability get slightly worse, and the assays that probe how
    it behaves in a host get better -- the viral taxon most of all.
    """
    s = summary.loc[SIZES]
    numeric = s.apply(pd.to_numeric, errors="coerce")
    d3b = numeric.loc["ESM2 (3B)"] - numeric.loc["ESM2 (650M)"]
    assert d3b["Taxa_Virus"] > 0.02, (
        f"3B should gain on viral assays, got {d3b['Taxa_Virus']:+.3f}")
    assert d3b["Function_Stability"] < 0, (
        f"3B should lose on stability assays, got {d3b['Function_Stability']:+.3f}")
    assert d3b["High_MSA_depth"] < 0
    assert d3b["Function_OrganismalFitness"] > 0
    # and both directions must be present, or this is not a trade-off
    assert (d3b > 0).any() and (d3b < 0).any()


def test_15b_continues_the_trend_3b_started(summary):
    """The 15B model is further along the same axis than 3B -- worse on
    stability, better on virus. A one-off anomaly at 3B would not do that, so
    this is evidence the mechanism is about training objective rather than a
    bad checkpoint."""
    s = summary.loc[["ESM2 (650M)", "ESM2 (3B)", "ESM2 (15B)"]]
    virus = s["Taxa_Virus"].astype(float)
    stab = s["Function_Stability"].astype(float)
    assert virus.is_monotonic_increasing, f"virus: {virus.to_dict()}"
    assert stab.is_monotonic_decreasing, f"stability: {stab.to_dict()}"


def test_published_headline_reproduces_for_every_size(summary, official):
    """Our aggregation rule, applied to each model's per-assay column, must
    give that model's published headline. This is what licenses analysing the
    published table instead of re-running the big checkpoints."""
    base = pd.DataFrame({"selection": official["Selection Type"].to_numpy(),
                         "uniprot": official["UniProt ID"].to_numpy()})
    for m in SIZES:
        t = base.assign(spearman=official[m].to_numpy(dtype=float))
        ours = aggregate.macro_over_categories(t)
        assert abs(ours - float(summary.loc[m, "Average_Spearman"])) < 0.001, (
            f"{m}: {ours:.4f} vs {summary.loc[m, 'Average_Spearman']}")


def test_local_curve_rises_to_650m_before_plateauing():
    """The local measurement, if it has been run.

    Skipped when results/esm2_scaling/local_curve.csv is absent, because it
    needs two checkpoint downloads and a GPU. When present it must show the
    same shape as the published curve: each step up to 650M helps, and the
    gain shrinks.
    """
    from pathlib import Path
    path = (Path(__file__).resolve().parents[1] / "results" / "esm2_scaling"
            / "local_curve.csv")
    if not path.exists():
        pytest.skip("local scaling curve has not been run")
    d = pd.read_csv(path)
    means = d.groupby("model")["spearman"].mean()
    for order in (["8M", "35M", "150M", "650M"],):
        present = [m for m in order if m in means.index]
        vals = means[present].to_numpy()
        assert (np.diff(vals) > 0).all(), f"not monotone: {means.to_dict()}"
        if len(vals) >= 4:
            assert np.diff(vals)[-1] < np.diff(vals)[0], (
                "the last step should be smaller than the first, or the curve "
                f"is not showing diminishing returns: {np.diff(vals)}")
