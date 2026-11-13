"""Aggregating per-assay Spearman the way ProteinGym does.

ProteinGym publishes a single headline number per model (ESM-2 650M: 0.414)
alongside per-assay Spearman values for 217 assays. Recomputing that headline
from the per-assay table is not a plain mean, and getting it wrong makes a
correct implementation look like a broken one: the plain mean of the published
per-assay ESM-2 650M values is 0.438, not 0.414.

The rule, recovered by testing candidate aggregations against the published
subgroup columns (Activity / Binding / Expression / OrganismalFitness /
Stability, and the Human / Eukaryote / Prokaryote / Virus and Low / Medium /
High MSA-depth splits) is two-staged:

  1. within a subgroup, average the per-assay Spearman across assays that
     share a UniProt id -- otherwise a protein with three assays (BLAT_ECOLX
     has three, SPIKE_SARS2 has two) counts three times;
  2. then average across subgroups.

With that, every published subgroup column reproduces to within 0.0005 (see
``tests/test_aggregate.py``), and the headline is the unweighted mean of the
five functional categories -- their macro average is 0.4138 against the
published 0.414.

The first stage matters because the benchmark is not balanced: 96 of 217
assays are human, and stability assays are easier than binding ones. Both
distort a plain mean, in opposite directions for different subgroups.
"""

import numpy as np
import pandas as pd

# the five functional categories ProteinGym averages over for the headline
CATEGORIES = ("Activity", "Binding", "Expression",
              "OrganismalFitness", "Stability")

# every subgroup the published summary table reports, keyed by a short name
# mapping onto the column that carries its levels. Used to validate the
# aggregation rather than to guess -- see tests/test_aggregate.py.
SUBGROUPS = {
    "category": CATEGORIES,
    "taxon": ("Human", "Eukaryote", "Prokaryote", "Virus"),
    "msa_depth": ("Low", "Medium", "High"),
}


def two_stage(per_assay, uniprot):
    """Mean within UniProt id, then mean across UniProt ids.

    per_assay: per-assay Spearman values.
    uniprot: the UniProt id of each assay, same length.

    Assays whose Spearman is NaN are dropped before either stage, so a model
    that could not score one assay is not penalised twice for it.
    """
    df = pd.DataFrame({"rho": np.asarray(per_assay, dtype=float),
                       "uniprot": list(uniprot)}).dropna(subset=["rho"])
    if df.empty:
        return float("nan")
    return float(df.groupby("uniprot")["rho"].mean().mean())


def macro_over_categories(table, score_col="spearman",
                          category_col="selection", uniprot_col="uniprot"):
    """ProteinGym's headline: two-stage within each category, then average the
    five category values with equal weight.

    table: one row per assay, with a Spearman column, a functional category
    and a UniProt id.
    """
    values = []
    for cat in CATEGORIES:
        sub = table[table[category_col] == cat]
        if sub.empty:
            continue
        values.append(two_stage(sub[score_col], sub[uniprot_col]))
    if not values:
        return float("nan")
    return float(np.mean(values))


def summarise(table, score_col="spearman", category_col="selection",
              uniprot_col="uniprot"):
    """All of the aggregations worth reporting, with the caveat attached.

    The plain mean is included because it is what a reader would compute by
    hand and would otherwise silently disagree with the headline by 0.024.
    """
    return {
        "n_assays": int(len(table)),
        "n_uniprot": int(table[uniprot_col].nunique()),
        "headline": macro_over_categories(table, score_col, category_col,
                                          uniprot_col),
        "mean_per_assay": float(table[score_col].mean()),
        "mean_per_uniprot": two_stage(table[score_col], table[uniprot_col]),
    }


def subgroup_table(table, model_col="spearman", by="category",
                   level_col=None, uniprot_col="uniprot"):
    """Two-stage value for each subgroup level, for one model.

    level_col is the column holding the level labels, which the caller names
    because the reference table and the published summary use different header
    spellings ('Selection Type' vs 'selection'). Defaults to the short name in
    SUBGROUPS, which is what the per-assay tables in this project use.

    Returns a long dataframe: subgroup, level, n_assays, n_uniprot, spearman.
    This is the shape the published summary table exposes, which is what lets
    the aggregation be checked rather than asserted.
    """
    levels = SUBGROUPS[by]
    col = level_col or by
    rows = []
    for level in levels:
        sub = table[table[col] == level]
        rows.append({
            "subgroup": by,
            "level": level,
            "n_assays": int(len(sub)),
            "n_uniprot": int(sub[uniprot_col].nunique()),
            "spearman": two_stage(sub[model_col], sub[uniprot_col]),
        })
    return pd.DataFrame(rows)
