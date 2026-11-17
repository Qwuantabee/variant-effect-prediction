"""Is the viral split about ESM-2, or about the objective any of these models
is trained on?

The README's viral section explains the failure as an objective mismatch:
ESM-2 learns conservation from UniRef, viral proteins vary across strains
under immune selection, so the model reads escape as tolerance. That story
makes a prediction beyond ESM-2 -- if it is about reading conservation from
UniRef rather than about this particular architecture, then a *different*
protein language model trained on the same kind of data should fail on
viruses in the same way, and a method that reads an alignment of the family
should not.

ProteinGym publishes per-assay Spearman for 97 methods over the same 217
assays, which makes this testable without downloading anything. This script
does two things to the published table:

  1. **The viral gap, per method.** mean(virus) - mean(non-virus) for every
     published method. ESM-C is the headline comparison: a different
     architecture, a different (larger, more diverse) training set, and it
     should still be asked the same question.
  2. **The family pattern.** The same gap averaged per method family. If the
     gap tracks the *training objective* it should be large and negative for
     every sequence-only language model and near zero for the alignment and
     retrieval methods -- and that is a pattern no single model's number can
     show.

A caveat that has to travel with the numbers: the gaps are computed on the
published per-assay column, so they are unweighted means over assays, not
ProteinGym's two-stage headline. That is the right thing here -- the question
is about a subgroup difference within each method, and the two-stage rule
would put every method through the same reweighting, changing the level but
not the comparison between methods.

Run:
    python scripts/run_family_analysis.py


Reading the family table (moved from the README)
-------------------------------------------------
ESM-C is the direct test the open-questions list asked for, and it fails harder
than ESM-2 rather than differently: 0.238 on virus against 0.465 elsewhere, a gap
of -0.227 where ESM-2 650M's is -0.198. It is a different architecture (a newer
ESM lineage with a substantially different, more diverse training set), offered
at 300M and 600M -- one below and one at the size of the 650M checkpoint -- and
both behave the same way (0.433 and 0.434 as per-assay means, 0.406 and 0.405 as
headlines). Same failure, same size, same direction.

The retrieval-augmented numbers: the figures usually quoted for them (~0.44 for
ESM-C, ~0.50 for the retrieval methods) are per-assay means, while the figures in
the published summary table are two-stage headlines, and for these methods the two
differ by about 0.03. Both are reported. On the per-assay column, AIDO Protein-RAG
(16B) averages 0.542 and Protriever 0.499; as headlines they are 0.518 and 0.479.
What matters for the viral question is neither level: AIDO's viral mean (0.521) is
barely below its non-viral mean (0.545), a gap of -0.024, and Protriever is
flatter still at +0.018 over the 200 assays where it is reported. Both retrieval
methods have nearly closed a gap that costs ESM-2 650M 0.198 of its 0.466
non-viral per-assay mean, about 42% of it. (An earlier version said 48%, dividing
by the 0.414 headline, which is a two-stage aggregate over all 217 assays and not
a non-viral mean; 650M's non-viral per-assay mean in per_method_viral_gap.csv is
0.4665, the column the gap is computed on.)

The one row that does not fit: RITA, an autoregressive sequence model, has a
slightly positive gap (+0.021). It is also the lowest-scoring sequence-only
family on non-viral assays here (0.355, against 0.393 to 0.481 for the others; the
lowest figure in the whole table, 0.346, belongs to Tranception without retrieval,
which reads an alignment and so is not the relevant comparison), so its flatness
is more likely a ceiling effect than a genuine viral strength -- a model that
averages 0.355 where the other sequence-only families average 0.39 to 0.47 has
less to lose. The pattern is reported with that row in it rather than trimmed,
because trimming a single inconvenient family is how a trend becomes a story.

ProteinGym's two-stage headlines are the numbers a reader will find in the paper.
The plain per-assay means are higher -- 0.438 for ESM-2 650M -- and quoting one
against the other is the mistake the aggregation section of the README warns
about, so the comparison uses each method's published headline only for the values
it lists as headlines, and the gap table uses the per-assay column for every
method identically.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_effect import data  # noqa: E402

OUT = ROOT / "results" / "families"

# Families, chosen so each row is a claim about a *training objective* rather
# than about a vendor. A family with one member is labelled as such in the
# table; the aggregate row is the mean over members, so a family with six
# checkpoints does not outvote one with two.
FAMILIES = {
    "ESM-2 (seq LM, UniRef)": [
        "ESM2 (8M)", "ESM2 (35M)", "ESM2 (150M)", "ESM2 (650M)",
        "ESM2 (3B)", "ESM2 (15B)"],
    "ESM-C (seq LM, diverse training set)": [
        "ESM-C (300M)", "ESM-C (600M)"],
    "ESM-1 (seq LM, UniRef)": [
        "ESM-1b", "ESM-1v (single)", "ESM-1v (ensemble)"],
    "RITA (seq LM, autoregressive)": [
        "RITA S", "RITA M", "RITA L", "RITA XL"],
    "Progen2 (seq LM, autoregressive)": [
        "Progen2 S", "Progen2 M", "Progen2 Base", "Progen2 L", "Progen2 XL"],
    "ProtSSN (structure-aware)": [
        "ProtSSN (k=10, h=512)", "ProtSSN (k=20, h=768)",
        "ProtSSN (ensemble)"],
    "Tranception, no retrieval (MSA)": [
        "Tranception S no retrieval", "Tranception M no retrieval",
        "Tranception L no retrieval"],
    "Tranception, retrieval (MSA)": [
        "Tranception S", "Tranception M", "Tranception L"],
    "TranceptEVE (MSA + retrieval)": [
        "TranceptEVE S", "TranceptEVE M", "TranceptEVE L"],
    "EVE / EVmutation (MSA only)": [
        "EVE (single)", "EVE (ensemble)", "EVmutation"],
    "Site-Independent (MSA only)": ["Site-Independent"],
    "GEMME (MSA only)": ["GEMME"],
    "VESPA (MSA + score priors)": ["VESPA", "VESPAl", "VespaG"],
    "Protein-RAG (retrieval-augmented)": [
        "AIDO Protein-RAG (16B)", "Protriever"],
}

# Which of the above read an alignment of the assayed family. The prediction
# under test is that the gap is concentrated in the ones that do not.
NO_ALIGNMENT = {
    "ESM-2 (seq LM, UniRef)", "ESM-C (seq LM, diverse training set)",
    "ESM-1 (seq LM, UniRef)", "RITA (seq LM, autoregressive)",
    "Progen2 (seq LM, autoregressive)", "ProtSSN (structure-aware)",
}

# Headline numbers the README quotes, so the comparison is against the value
# a reader would look up rather than against a recomputation. Checked in
# tests/test_family_analysis.py against the published summary table.
#
# These are ProteinGym's *headline* (Average_Spearman: two-stage within
# UniProt then macro over the five categories). The plain per-assay mean is a
# different, higher number -- 0.4382 for ESM-2 650M against a headline of
# 0.414 -- and mixing the two is exactly the error the aggregation section of
# the README warns about, so the headline is the one quoted here.
QUOTED = {
    "ESM2 (650M)": 0.414,
    "ESM-C (300M)": 0.406,
    "ESM-C (600M)": 0.405,
    "AIDO Protein-RAG (16B)": 0.518,
    "Protriever": 0.479,
}


def per_method_gap(official, uniprot_col="UniProt ID", taxon_col="Taxon"):
    """mean(virus) - mean(non-virus) for every published method column."""
    taxon = official[taxon_col]
    is_virus = taxon == "Virus"
    rows = []
    for col in official.columns:
        if col in ("Number of Mutants", "Selection Type", uniprot_col,
                   taxon_col, "MSA_Neff_L_category"):
            continue
        s = pd.to_numeric(official[col], errors="coerce")
        n_ok = int(s.notna().sum())
        if n_ok < 150:          # skip columns that are mostly empty
            continue
        v = s[is_virus]
        nv = s[~is_virus]
        # paired test over assays is not right here (different assays), so a
        # two-sample difference with its own standard error is reported
        gap = float(v.mean() - nv.mean())
        se = float(np.sqrt(v.var(ddof=1) / v.notna().sum()
                           + nv.var(ddof=1) / nv.notna().sum()))
        rows.append({
            "method": col,
            "n_assays": int(v.notna().sum() + nv.notna().sum()),
            "virus_mean": float(v.mean()),
            "nonviral_mean": float(nv.mean()),
            "gap": gap,
            "gap_se": se,
            "gap_t": gap / se if se > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("gap")


def family_table(per_method):
    """The gap averaged per family, plus the overall split."""
    by_method = per_method.set_index("method")
    rows = []
    for fam, cols in FAMILIES.items():
        cols = [c for c in cols if c in by_method.index]
        if not cols:
            continue
        sub = by_method.loc[cols]
        rows.append({
            "family": fam,
            "methods": len(cols),
            "reads_alignment_of_the_family": fam not in NO_ALIGNMENT,
            "virus": float(sub["virus_mean"].mean()),
            "nonviral": float(sub["nonviral_mean"].mean()),
            "gap": float(sub["gap"].mean()),
            "worst_gap": float(sub["gap"].min()),
        })
    df = pd.DataFrame(rows).sort_values("gap")
    return df


def gap_separation(per_method, family):
    """Do the no-alignment families and the alignment families differ in gap?

    UNIT OF INFERENCE = THE FAMILY. Checkpoints of one family (ESM-2 8M ...
    15B, Progen2 S ... XL) are the same architecture and training data at
    different sizes; they are not independent draws of a method, and a test over
    them counts one family many times. The test of record is therefore an EXACT
    one-sided Mann-Whitney over family-mean gaps (family_table's gap column):
    p_family_level_exact, with n_families_* and the family-level medians. With
    6 versus 8 families the exact null enumerates all 3,003 labellings, so no
    normal approximation is involved.

    The checkpoint-level test (23 versus 19 checkpoints) is still returned, as
    p_no_alignment_worse with the checkpoint counts and medians, ONLY for
    continuity with earlier output: it is pseudo-replicated and its p overstates
    the evidence by a large factor. Do not quote it as a significance.
    """
    by_method = per_method.set_index("method")["gap"]
    no_align, align = [], []
    for fam, cols in FAMILIES.items():
        vals = [by_method[c] for c in cols if c in by_method.index]
        (no_align if fam in NO_ALIGNMENT else align).extend(vals)
    if len(no_align) < 3 or len(align) < 3:
        return {}
    u, p = stats.mannwhitneyu(no_align, align, alternative="less")
    out = {
        "n_no_alignment": len(no_align),
        "n_alignment": len(align),
        "median_gap_no_alignment": float(np.median(no_align)),
        "median_gap_alignment": float(np.median(align)),
        "mannwhitney_u": float(u),
        "p_no_alignment_worse": float(p),
    }
    if (family is not None and len(family)
            and "reads_alignment_of_the_family" in family.columns):
        fam_no = family.loc[~family["reads_alignment_of_the_family"], "gap"]
        fam_yes = family.loc[family["reads_alignment_of_the_family"], "gap"]
        if len(fam_no) >= 3 and len(fam_yes) >= 3:
            uf, pf = stats.mannwhitneyu(fam_no, fam_yes, alternative="less",
                                        method="exact")
            out.update({
                "n_families_no_alignment": int(len(fam_no)),
                "n_families_alignment": int(len(fam_yes)),
                "median_family_gap_no_alignment": float(fam_no.median()),
                "median_family_gap_alignment": float(fam_yes.median()),
                "mannwhitney_u_family_level": float(uf),
                "p_family_level_exact": float(pf),
            })
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    official = data.official_scores()
    summary = data.official_summary()

    # the quoted headlines must match the published summary table, or the
    # README is quoting a number from nowhere
    print("quoted vs published summary")
    quote_rows = []
    for name, quoted in QUOTED.items():
        published = None
        if name in summary.index:
            published = float(summary.loc[name, "Average_Spearman"])
        else:
            hit = summary[summary["Model_name"] == name]
            if len(hit):
                published = float(hit["Average_Spearman"].iloc[0])
        diff = (quoted - published) if published is not None else np.nan
        quote_rows.append({"model": name, "quoted": quoted,
                           "published_headline": published,
                           "difference": diff})
        status = ("absent from summary table" if published is None
                  else f"published {published:.4f}, diff {diff:+.4f}")
        print(f"  {name:<26} {status}")
    quotes = pd.DataFrame(quote_rows)
    quotes.to_csv(OUT / "quoted_vs_published.csv", index=False)
    if quotes["difference"].abs().max() > 0.001:
        raise SystemExit(
            "quoted headline does not match the published summary table -- "
            "fix QUOTED rather than the test")

    per_method = per_method_gap(official)
    per_method.to_csv(OUT / "per_method_viral_gap.csv", index=False)

    fam = family_table(per_method)
    fam.to_csv(OUT / "family_viral_gap.csv", index=False)

    sep = gap_separation(per_method, fam)
    pd.DataFrame([sep]).to_csv(OUT / "family_separation.csv", index=False)

    print("\nviral gap by family (mean over the family's checkpoints)")
    print(fam.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\nper-method extremes")
    print(f"  worst 3: {', '.join(per_method['method'].head(3))}")
    print(f"  best  3: {', '.join(per_method['method'].tail(3))}")

    print(f"\nmethods with a gap > +0.02: "
          f"{int((per_method['gap'] > 0.02).sum())} of {len(per_method)}")
    print(f"methods with a gap < -0.15: "
          f"{int((per_method['gap'] < -0.15).sum())} of {len(per_method)}")

    if sep and "p_family_level_exact" in sep:
        print(f"\nfamily level ({sep['n_families_no_alignment']} vs "
              f"{sep['n_families_alignment']} families): median family gap "
              f"{sep['median_family_gap_no_alignment']:+.3f} vs "
              f"{sep['median_family_gap_alignment']:+.3f}, exact one-sided "
              f"Mann-Whitney U={sep['mannwhitney_u_family_level']:.0f} "
              f"p={sep['p_family_level_exact']:.4f}")
        print(f"(checkpoint level, pseudo-replicated, NOT a significance: "
              f"{sep['n_no_alignment']} vs {sep['n_alignment']} checkpoints, "
              f"p={sep['p_no_alignment_worse']:.3g})")
    elif sep:
        print(f"\nno-alignment checkpoints median gap "
              f"{sep['median_gap_no_alignment']:+.3f} vs alignment "
              f"{sep['median_gap_alignment']:+.3f} (no family-level test)")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
