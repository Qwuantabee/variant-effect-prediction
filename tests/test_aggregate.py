"""Tests for the ProteinGym aggregation rule.

The rule in aggregate.py is not a style choice -- it is what makes a local
result comparable to the published headline. These tests pin it against the
official summary table, which is the only authority on what "0.414" means. If
a refactor changes the aggregation, the headline moves and the comparison to
the literature silently becomes an apples-to-oranges one.

The strongest test here is ``test_reproduces_published_subgroup_values``: it
takes the published per-assay table (someone else's numbers, not ours) and
checks that applying our aggregation to it reproduces the published subgroup
averages. That test cannot pass by construction -- it would fail if the
aggregation were a plain mean, a UniProt-only average, or a category-only
average.
"""

import numpy as np
import pandas as pd
import pytest

from variant_effect import aggregate, data

# the published subgroup columns and the reference-table columns they split on
SUBGROUP_CHECKS = [
    ("selection", "Activity", "Function_Activity"),
    ("selection", "Binding", "Function_Binding"),
    ("selection", "Expression", "Function_Expression"),
    ("selection", "OrganismalFitness", "Function_OrganismalFitness"),
    ("selection", "Stability", "Function_Stability"),
    ("taxon", "Human", "Taxa_Human"),
    ("taxon", "Eukaryote", "Taxa_Other_Eukaryote"),
    ("taxon", "Prokaryote", "Taxa_Prokaryote"),
    ("taxon", "Virus", "Taxa_Virus"),
    ("msa_depth", "Low", "Low_MSA_depth"),
    ("msa_depth", "Medium", "Medium_MSA_depth"),
    ("msa_depth", "High", "High_MSA_depth"),
]


@pytest.fixture(scope="module")
def official_table():
    """The published per-assay table with category and UniProt columns."""
    o = data.official_scores()
    return pd.DataFrame({
        "assay": o.index,
        "spearman": o["ESM2 (650M)"].to_numpy(dtype=float),
        "selection": o["Selection Type"].to_numpy(),
        "taxon": o["Taxon"].to_numpy(),
        "msa_depth": o["MSA_Neff_L_category"].to_numpy(),
        "uniprot": o["UniProt ID"].to_numpy(),
    })


@pytest.fixture(scope="module")
def summary():
    return data.official_summary()


def test_two_stage_matches_hand_computation():
    """Two assays on one protein must count once, not twice."""
    values = [0.2, 0.8, 0.6]
    uniprot = ["A", "A", "B"]
    # protein A contributes 0.5, protein B contributes 0.6, so the answer is
    # 0.55 -- a plain mean would give 0.533
    assert aggregate.two_stage(values, uniprot) == pytest.approx(0.55)


def test_two_stage_ignores_nan():
    """A model that could not score one assay is not penalised twice."""
    values = [0.4, np.nan, 0.6]
    uniprot = ["A", "B", "B"]
    assert aggregate.two_stage(values, uniprot) == pytest.approx(0.5)


def test_two_stage_all_nan_is_nan():
    assert np.isnan(aggregate.two_stage([np.nan, np.nan], ["A", "B"]))


def test_macro_over_categories_weights_categories_equally():
    """A category with two assays and a category with one must count equally,
    which is what makes the headline a *balanced* average."""
    table = pd.DataFrame({
        "spearman": [0.1, 0.3, 0.9],
        "selection": ["Stability", "Stability", "Binding"],
        "uniprot": ["A", "B", "C"],
    })
    # stability 0.2, binding 0.9 -> 0.55; a per-assay mean would be 0.433
    assert aggregate.macro_over_categories(table) == pytest.approx(0.55)


def test_macro_over_categories_handles_missing_category():
    """Some benchmarks lack a category entirely; the average must renormalise
    rather than treat the absent category as zero."""
    table = pd.DataFrame({
        "spearman": [0.4, 0.6],
        "selection": ["Binding", "Binding"],
        "uniprot": ["A", "B"],
    })
    assert aggregate.macro_over_categories(table) == pytest.approx(0.5)


def test_reproduces_published_headline(official_table, summary):
    """The headline check.

    Applying the aggregation to the published per-assay ESM-2 650M values must
    give the published headline of 0.414. A plain mean gives 0.438, so this
    test distinguishes the correct rule from the obvious wrong one.
    """
    ours = aggregate.macro_over_categories(official_table)
    published = float(summary.loc["ESM2 (650M)", "Average_Spearman"])
    assert abs(ours - published) < 0.001, (
        f"aggregation gives {ours:.4f}, published {published:.4f}")


@pytest.mark.parametrize("by,level,published_col", SUBGROUP_CHECKS)
def test_reproduces_published_subgroup_values(official_table, summary,
                                              by, level, published_col):
    """Every published subgroup column, reproduced from the per-assay table.

    These twelve checks are what establish the rule: a wrong stage order or a
    missing UniProt stage fails at least one of them, and the failing subgroup
    identifies which stage is wrong. The OrganismalFitness and Low-MSA-depth
    splits are the sensitive ones -- a plain mean misses them by 0.023.
    """
    col = by
    sub = official_table[official_table[col] == level]
    ours = aggregate.two_stage(sub["spearman"], sub["uniprot"])
    published = float(summary.loc["ESM2 (650M)", published_col])
    assert abs(ours - published) < 0.001, (
        f"{by}={level}: aggregation gives {ours:.4f}, published {published:.4f}")


def test_reproduces_published_headline_across_models(official_table, summary):
    """The rule is not tuned to ESM-2 650M.

    Eight predictors with published headlines spanning 0.226 to 0.455 must all
    land within 0.001. A rule fitted to one model would not survive this.
    """
    o = data.official_scores()
    for model in ["ESM2 (650M)", "ESM2 (3B)", "ESM2 (150M)", "ESM2 (8M)",
                  "ESM-1v (single)", "GEMME", "Site-Independent", "EVmutation"]:
        table = official_table.assign(spearman=o[model].to_numpy(dtype=float))
        ours = aggregate.macro_over_categories(table)
        published = float(summary.loc[model, "Average_Spearman"])
        assert abs(ours - published) < 0.001, (
            f"{model}: {ours:.4f} vs published {published:.4f}")


def test_summarise_reports_both_averages(official_table):
    """Both the headline and the plain per-assay mean must be reported.

    The plain mean is 0.024 higher than the headline on ESM-2 650M. Anyone
    computing it by hand will get that number, so a report that shows only the
    headline looks wrong; showing both, with the reason for the gap, is the
    honest presentation.
    """
    s = aggregate.summarise(official_table)
    assert s["n_assays"] == 217
    assert s["n_uniprot"] == 186
    assert s["mean_per_assay"] > s["headline"], (
        "on ProteinGym the unbalanced per-assay mean runs above the balanced "
        "headline; if this flips, the aggregation or the input changed")


def test_subgroup_table_covers_every_level(official_table):
    """Every subgroup level must be present and finite.

    The column holding the levels is named explicitly per subgroup because the
    reference table and the published summary disagree on header spelling
    ('Selection Type' vs 'selection'), and the fixture here uses the short
    names from this project's own per-assay tables.
    """
    level_cols = {"category": "selection", "taxon": "taxon",
                  "msa_depth": "msa_depth"}
    for by, levels in aggregate.SUBGROUPS.items():
        tab = aggregate.subgroup_table(official_table, by=by,
                                       level_col=level_cols[by],
                                       uniprot_col="uniprot")
        assert list(tab["level"]) == list(levels)
        assert tab["spearman"].notna().all()
        assert (tab["n_assays"] > 0).all()
