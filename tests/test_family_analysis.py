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


def _have_published():
    return (ROOT / "data" / "official_spearman_dms_level.csv").exists()


requires_published = pytest.mark.skipif(
    not _have_published(), reason="published per-assay table not downloaded")


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
