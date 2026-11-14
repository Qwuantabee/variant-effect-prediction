"""Tests for the cross-family viral-gap analysis.

The README's claim that the viral split is not an ESM-2 artefact rests on a
comparison against published numbers for other model families. These tests
pin the parts that could silently rot: that the quoted headlines match the
published summary table, that the gap is computed as virus minus everything
else with the right sign, and that the family table does not quietly drop a
family whose columns are missing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from variant_effect import data  # noqa: E402


# A missing published table is a FAILURE here, not a skip (tests/conftest.py):
# a skip turned a deleted data file into a green run.
requires_published = pytest.mark.usefixtures("published_scores_present")


@requires_published
def test_quoted_headlines_match_the_published_summary():
    """Every headline the README quotes has to be the published one. A typo
    here would put a number in the prose that came from nowhere, which is the
    failure mode the generated tables exist to prevent."""
    import run_family_analysis as rfa

    summary = data.official_summary()
    # official_summary is indexed by model name, and the index is what the
    # script reads; a name missing from it means the README quotes a model
    # the benchmark does not publish
    for name, quoted in rfa.QUOTED.items():
        assert name in summary.index, f"{name} not in the published summary"
        published = float(summary.loc[name, "Average_Spearman"])
        assert quoted == pytest.approx(published, abs=5e-4), name


@requires_published
def test_viral_gap_has_the_right_sign_and_is_not_a_headline():
    """The gap is virus minus non-virus on the *per-assay* column. ESM-2 650M
    is the known case: strongly negative. If the sign flipped or the wrong
    column were read, this would catch it."""
    import run_family_analysis as rfa

    per_method = rfa.per_method_gap(data.official_scores())
    row = per_method.set_index("method").loc["ESM2 (650M)"]
    assert row["gap"] < -0.15
    assert row["virus_mean"] < row["nonviral_mean"]
    # the per-assay mean must be the higher number, not the two-stage headline
    assert row["nonviral_mean"] + row["virus_mean"] > 0.6

    esmc = per_method.set_index("method").loc["ESM-C (600M)"]
    assert esmc["gap"] < -0.15


@requires_published
def test_family_table_keeps_every_quotable_family():
    """A family silently dropped because its columns were renamed would make
    the separation test compare fewer methods than the README says."""
    import run_family_analysis as rfa

    per_method = rfa.per_method_gap(data.official_scores())
    families = rfa.family_table(per_method)
    assert "ESM-C (seq LM, diverse training set)" in set(families["family"])
    assert "ESM-2 (seq LM, UniRef)" in set(families["family"])
    assert int(families["methods"].sum()) == len(
        set(sum(rfa.FAMILIES.values(), [])) & set(per_method["method"]))


@requires_published
def test_no_alignment_families_have_a_larger_gap_than_alignment_families():
    """The measured claim in the README, at a threshold loose enough not to
    break on a small data revision but tight enough to fail if it reverses."""
    import run_family_analysis as rfa

    per_method = rfa.per_method_gap(data.official_scores())
    families = rfa.family_table(per_method)
    sep = rfa.gap_separation(per_method, families)
    assert sep["median_gap_no_alignment"] < sep["median_gap_alignment"] - 0.05
    assert sep["p_no_alignment_worse"] < 0.01


def test_separation_needs_both_groups():
    """A guard against the test passing on an empty comparison, which is what
    a mis-keyed family name would produce."""
    import run_family_analysis as rfa

    empty = pd.DataFrame({"method": [], "virus_mean": [], "nonviral_mean": [],
                          "gap": [], "gap_se": [], "gap_t": [],
                          "n_assays": []})
    assert rfa.gap_separation(empty, pd.DataFrame()) == {}



# ----------------------------------------------------------------------
# family-level inference (review C6): the unit that varies is the family
# ----------------------------------------------------------------------

def _synthetic_methods_and_families(no_gaps, yes_gaps, per_family=4, seed=0):
    """Build per_method / family frames in the shapes gap_separation reads,
    over the real FAMILIES keys so the NO_ALIGNMENT membership is the repo's."""
    import run_family_analysis as rfa
    rng = np.random.default_rng(seed)
    no_fams = [f for f in rfa.FAMILIES if f in rfa.NO_ALIGNMENT]
    yes_fams = [f for f in rfa.FAMILIES if f not in rfa.NO_ALIGNMENT]
    assert len(no_fams) == len(no_gaps) and len(yes_fams) == len(yes_gaps)
    method_rows, fam_rows = [], []
    for fams, gaps in ((no_fams, no_gaps), (yes_fams, yes_gaps)):
        for fam, g in zip(fams, gaps):
            members = rfa.FAMILIES[fam]
            vals = g + rng.normal(0, 0.001, len(members))
            for c, v in zip(members, vals):
                method_rows.append({"method": c, "gap": v})
            fam_rows.append({"family": fam, "gap": float(np.mean(vals)),
                             "reads_alignment_of_the_family":
                             fam not in rfa.NO_ALIGNMENT})
    return pd.DataFrame(method_rows), pd.DataFrame(fam_rows)


def _brute_force_exact_p(no, yes):
    """P(U <= observed) over every labelling of the pooled families; written
    independently of scipy."""
    import itertools
    pooled = list(no) + list(yes)
    n_no = len(no)

    def u_of(group_idx):
        grp = [pooled[i] for i in group_idx]
        rest = [pooled[i] for i in range(len(pooled)) if i not in group_idx]
        return sum((a > b) + 0.5 * (a == b) for a in grp for b in rest)

    obs = u_of(tuple(range(n_no)))
    count = total = 0
    for idx in itertools.combinations(range(len(pooled)), n_no):
        total += 1
        count += u_of(idx) <= obs + 1e-12
    return count / total


def test_family_level_p_is_an_exact_test_over_family_means_not_checkpoints():
    """The p of record uses 6 vs 8 FAMILIES. The checkpoint-level p (23 vs 19)
    is returned only as a legacy key and is much smaller on the same data."""
    import run_family_analysis as rfa
    no = [-0.20, -0.15, -0.12, -0.08, -0.05, 0.01]
    yes = [-0.10, -0.04, -0.02, -0.01, 0.0, 0.02, 0.03, 0.05]
    per_method, family = _synthetic_methods_and_families(no, yes)
    sep = rfa.gap_separation(per_method, family)
    assert sep["n_families_no_alignment"] == 6
    assert sep["n_families_alignment"] == 8
    expected = _brute_force_exact_p(
        family.loc[~family["reads_alignment_of_the_family"], "gap"].tolist(),
        family.loc[family["reads_alignment_of_the_family"], "gap"].tolist())
    assert sep["p_family_level_exact"] == pytest.approx(expected, abs=1e-9)
    # pseudo-replication: counting checkpoints makes the same data look far
    # stronger than counting families does
    assert sep["p_no_alignment_worse"] < sep["p_family_level_exact"] / 5


def test_family_level_test_needs_three_families_per_group():
    import run_family_analysis as rfa
    per_method, family = _synthetic_methods_and_families(
        [-0.2] * 6, [-0.01] * 8)
    short = family.iloc[:2]            # two no-alignment families only
    sep = rfa.gap_separation(per_method, short)
    assert "p_family_level_exact" not in sep
    assert "p_no_alignment_worse" in sep      # checkpoint level still returned


@requires_published
def test_family_level_p_on_the_published_table_is_the_one_the_readme_quotes():
    """On the real per-assay table: 6 vs 8 families, exact one-sided U = 8. The
    p is recomputed from the table and compared with the artefact the README
    renders, so a drift in either is caught."""
    import run_family_analysis as rfa
    per_method = rfa.per_method_gap(data.official_scores())
    family = rfa.family_table(per_method)
    sep = rfa.gap_separation(per_method, family)
    assert (sep["n_families_no_alignment"], sep["n_families_alignment"]) == (6, 8)
    assert sep["mannwhitney_u_family_level"] == 8.0
    assert 0.01 < sep["p_family_level_exact"] < 0.05
    art = pd.read_csv(ROOT / "results" / "families" / "family_separation.csv")
    assert art["p_family_level_exact"].iloc[0] == pytest.approx(
        sep["p_family_level_exact"])
