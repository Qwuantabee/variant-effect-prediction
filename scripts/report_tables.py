"""Generate the README's result tables from the result files, so no number is
hand-copied.

Every table in the README that reports a measured value is written by this
script from CSVs that other scripts produced. That is the only way to keep the
prose honest: a number in the README that was typed by hand can drift from the
file it came from, and a reader has no way to tell.

Run:
    python scripts/report_tables.py            # print
    python scripts/report_tables.py --write    # rewrite the generated blocks

Markers delimit the generated regions:
    <!-- BEGIN GENERATED: <name> -->  ...  <!-- END GENERATED: <name> -->
Anything outside those markers is prose and is left alone. The bulky tables
(PAGE_TABLES) live in RESULTS.md so the README stays readable; the README says
which table each section refers to. Every other table is in the README itself.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_effect import aggregate, data  # noqa: E402

README = ROOT / "README.md"
RESULTS_PAGE = ROOT / "RESULTS.md"
# Tables that are written to RESULTS.md instead of the README. The replicated-
# epistasis table is NOT here: a test compares it with the README, byte for byte.
PAGE_TABLES = ("topk_ceiling", "latent_truth", "sim_controls", "epistasis",
               "family_viral_gap")
BEGIN = "<!-- BEGIN GENERATED: {} -->"
END = "<!-- END GENERATED: {} -->\n"


def fmt_table(df, floatfmt="{:.3f}"):
    """Markdown table, numbers formatted, no index."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("n/a" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. headline on the full benchmark
# ---------------------------------------------------------------------------
def full_benchmark_table():
    path = ROOT / "results" / "full" / "esm2_t33_650M_UR50D_per_assay.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    esm = d[d["method"] == "esm2"].dropna(subset=["spearman"])
    if esm.empty:
        return None

    rows = []
    ours = aggregate.summarise(esm.rename(columns={"selection": "selection"}))
    off = data.official_scores()
    off_table = pd.DataFrame({
        "spearman": off["ESM2 (650M)"].to_numpy(dtype=float),
        "selection": off["Selection Type"].to_numpy(),
        "uniprot": off["UniProt ID"].to_numpy(),
    })
    off_head = aggregate.macro_over_categories(off_table)

    rows.append({"": "headline (macro over 5 categories)",
                 "this implementation": ours["headline"],
                 "ProteinGym published": off_head,
                 "difference": ours["headline"] - off_head})
    rows.append({"": "per-assay mean (unbalanced)",
                 "this implementation": ours["mean_per_assay"],
                 "ProteinGym published": off_table["spearman"].mean(),
                 "difference": ours["mean_per_assay"] - off_table["spearman"].mean()})
    return fmt_table(pd.DataFrame(rows), floatfmt="{:+.3f}")


def full_benchmark_coverage():
    path = ROOT / "results" / "full" / "esm2_t33_650M_UR50D_per_assay.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    esm = d[d["method"] == "esm2"].dropna(subset=["spearman"])
    off = data.official_scores()
    both = esm["assay"].isin(off.index)
    if not both.any():
        return (f"{len(esm)} of {len(off)} assays scored; none of them has a "
                "published per-assay value to compare against.")
    diff = (esm.set_index("assay").loc[
        esm.loc[both, "assay"], "spearman"].to_numpy()
        - off.loc[esm.loc[both, "assay"], "ESM2 (650M)"].to_numpy())
    text = (
        f"{len(esm)} of {len(off)} assays scored. "
        f"Against the published per-assay column: mean difference "
        f"{diff.mean():+.4f}, median absolute {np.abs(diff).min() if len(diff) == 0 else np.median(np.abs(diff)):.4f}, "
        f"maximum absolute {np.abs(diff).max():.4f}. "
        f"{int((np.abs(diff) < 0.01).sum())} of {len(diff)} assays within 0.01."
    )
    return text


# ---------------------------------------------------------------------------
# 2. viral alignment comparison
# ---------------------------------------------------------------------------
def viral_table():
    path = ROOT / "results" / "viral" / "viral.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path).dropna(subset=["spearman"])
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_viral_alignment import classify
    d["virus_class"] = d["organism"].map(classify)

    rows = []
    for cls, label in (("animal_virus", "animal viruses"),
                       ("phage", "bacteriophages")):
        sub = d[d["virus_class"] == cls]
        if sub.empty:
            continue
        rows.append({
            "group": label,
            "assays": len(sub),
            "alignment (ours)": sub["spearman"].mean(),
            "alignment (published)": sub["official_site_independent"].mean(),
            "ESM-2 650M (published)": sub["official_esm2_650m"].mean(),
            "alignment beats ESM-2": f"{int((sub['spearman'] > sub['official_esm2_650m']).sum())}/{len(sub)}",
        })
    rows.append({
        "group": "all viral assays",
        "assays": len(d),
        "alignment (ours)": d["spearman"].mean(),
        "alignment (published)": d["official_site_independent"].mean(),
        "ESM-2 650M (published)": d["official_esm2_650m"].mean(),
        "alignment beats ESM-2": f"{int((d['spearman'] > d['official_esm2_650m']).sum())}/{len(d)}",
    })
    return fmt_table(pd.DataFrame(rows), floatfmt="{:+.3f}")


# ---------------------------------------------------------------------------
# 3. ESM-2 scaling
# ---------------------------------------------------------------------------
def scaling_table():
    path = ROOT / "results" / "esm2_scaling" / "subgroups.csv"
    if not path.exists():
        return None
    mov = pd.read_csv(path, index_col=0)
    want = ["Function_Stability", "High_MSA_depth", "Function_Binding",
            "Function_Activity", "Taxa_Human", "Taxa_Prokaryote",
            "Function_Expression", "Medium_MSA_depth",
            "Function_OrganismalFitness", "Taxa_Virus"]
    mov = mov.loc[[w for w in want if w in mov.index]]
    out = pd.DataFrame({
        "split": mov.index,
        "650M": mov["650M"].to_numpy(),
        "3B": mov["3B"].to_numpy(),
        "15B": mov["15B"].to_numpy(),
        "3B − 650M": mov["d_3B"].to_numpy(),
        "15B − 650M": mov["d_15B"].to_numpy(),
    })
    return fmt_table(out, floatfmt="{:+.3f}")


def scaling_headline_table():
    path = ROOT / "results" / "esm2_scaling" / "scaling.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d["params"] = d["params"].map(lambda p: f"{p/1e9:.0f}B" if p >= 1e9 else f"{p/1e6:.0f}M")
    out = d[["model", "params", "headline", "published_se"]]
    return fmt_table(out, floatfmt="{:.3f}")


def paired_table():
    path = ROOT / "results" / "esm2_scaling" / "paired_tests.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    out = pd.DataFrame({
        "comparison": d["model_b"] + " − " + d["model_a"],
        "mean diff": d["mean_diff_b_minus_a"],
        "median diff": d["median_diff"],
        "SE": d["se_diff"],
        "paired t p": [("<0.0001" if p < 1e-4 else f"{p:.4f}")
                       for p in d["p_paired_t"]],
        "Wilcoxon p": [("<0.0001" if p < 1e-4 else f"{p:.4f}")
                       for p in d["p_wilcoxon"]],
        "W/L": d["b_wins"].astype(str) + "/" + d["a_wins"].astype(str),
    })
    return fmt_table(out, floatfmt="{:+.4f}")


def local_curve_table():
    path = ROOT / "results" / "esm2_scaling" / "local_curve.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    g = d.groupby("model")["spearman"].agg(["mean", "std", "count"])
    order = [m for m in ["8M", "35M", "150M", "650M"] if m in g.index]
    g = g.loc[order]
    out = pd.DataFrame({
        "model": g.index,
        "assays": g["count"].astype(int).to_numpy(),
        "mean Spearman": g["mean"].to_numpy(),
        "sd": g["std"].to_numpy(),
    })
    return fmt_table(out, floatfmt="{:.3f}")


# ---------------------------------------------------------------------------
# 4. structure
# ---------------------------------------------------------------------------
def burial_table():
    path = ROOT / "results" / "structure" / "burial_groups.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    esm = d[d["method"] == "esm"]
    if esm.empty:
        return None
    out = esm[["class", "n", "spearman", "top10_recall"]].rename(
        columns={"class": "buried/exposed",
                 "spearman": "Spearman",
                 "top10_recall": "top-10 recall"})
    return fmt_table(out, floatfmt="{:+.3f}")


def structure_heldout_text():
    path = ROOT / "results" / "structure" / "full_heldout.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    if d.empty:
        return None
    from scipy import stats
    diff = d["adjusted"] - d["plain"]
    t, p = stats.ttest_rel(d["adjusted"], d["plain"])
    return (
        f"On the {len(d)} held-out assays, plain ESM-2 scores "
        f"{d['plain'].mean():+.4f} and the burial-aware rescoring "
        f"{d['adjusted'].mean():+.4f}, a change of {diff.mean():+.4f} "
        f"(paired t={t:.2f}, p={p:.3g}, better on {int((diff > 0).sum())} of "
        f"{len(d)})."
    )


def scope_audit_table():
    """What the benchmark actually contains, from results/scope/summary.json."""
    path = ROOT / "results" / "scope" / "summary.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    if not s.get("variants_scanned"):
        return None
    total = s["variants_scanned"]
    rows = [
        ("assays in the benchmark", f"{s['assays_scored']}"),
        ("assays scored on disk", f"{s['assays_on_disk']}"),
        ("variants scanned", f"{total:,}"),
        ("variants not in substitution notation",
         f"{s['variants_not_substitution_notation']}"),
        ("variants carrying a stop codon", f"{s['variants_with_stop_codon']}"),
        ("multi-substitution variants",
         f"{s['multi_substitution_variants']:,} "
         f"({100 * s['multi_substitution_variants'] / total:.1f}%)"),
        ("assays that are single-substitution only",
         f"{s['assays_all_single_substitution']} of {s['assays_scored']}"),
        ("indel archive on disk", "yes" if s["indel_archive_present"] else "no"),
        ("clinical labels on disk", f"{s['clinical_labels_on_disk']}"),
    ]
    out = pd.DataFrame(rows, columns=["what", "measured"])
    return fmt_table(out, floatfmt="{:.3f}")


def family_gap_table():
    """Viral gap by model family, from the published per-assay table."""
    path = ROOT / "results" / "families" / "family_viral_gap.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    out = pd.DataFrame({
        "family": d["family"],
        "checkpoints": d["methods"].astype(int),
        "reads an alignment": d["reads_alignment_of_the_family"].map(
            {True: "yes", False: "no"}),
        "virus": d["virus"],
        "non-viral": d["nonviral"],
        "gap": d["gap"],
    })
    return fmt_table(out, floatfmt="{:+.3f}")


def family_separation_text():
    """The family-level exact test, with the checkpoint-level number labelled
    as the descriptive, pseudo-replicated thing it is."""
    path = ROOT / "results" / "families" / "family_separation.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    if d.empty or "p_family_level_exact" not in d.columns \
            or not np.isfinite(d["p_family_level_exact"].iloc[0]):
        return None
    r = d.iloc[0]
    return (
        f"Tested at the level of the family, which is the unit that varies "
        f"(checkpoints within a family share an architecture and a training "
        f"set): the {int(r['n_families_no_alignment'])} families that do not "
        f"read an alignment have a median viral gap of "
        f"{r['median_family_gap_no_alignment']:+.3f}, the "
        f"{int(r['n_families_alignment'])} that do have "
        f"{r['median_family_gap_alignment']:+.3f}; the exact one-sided "
        f"Mann-Whitney test gives U = {r['mannwhitney_u_family_level']:.0f}, "
        f"p = {r['p_family_level_exact']:.3f}. Counted per checkpoint "
        f"({int(r['n_no_alignment'])} versus {int(r['n_alignment'])}) the same "
        f"test returns p = {r['p_no_alignment_worse']:.2g}, but that number "
        f"counts one family many times and is shown only so it is not mistaken "
        f"for the evidence."
    )


def quoted_vs_published_text():
    path = ROOT / "results" / "families" / "quoted_vs_published.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    parts = [f"{r['model']} {r['published_headline']:.3f}" for _, r in d.iterrows()]
    return "ProteinGym headlines, read from the published summary table: " + \
        ", ".join(parts) + "."


def epistasis_table():
    """The singles-vs-multis test, from results/epistasis.

    The README once said epistasis could not be tested because the replicate
    subset is single-substitution only. This table is the test that framing
    missed: the full local protein set has multi-mutants, and the scorer's
    independent-site summation is exactly the hypothesis under test. Every
    number below is read from the run's CSVs so none is hand-copied.
    """
    per_path = ROOT / "results" / "epistasis" / "per_stratum_all.csv"
    gaps_path = ROOT / "results" / "epistasis" / "paired_gaps.csv"
    if not per_path.exists() or not gaps_path.exists():
        return None

    per = pd.read_csv(per_path)
    gaps = pd.read_csv(gaps_path)

    def row(r):
        return {
            "assay": r["assay"],
            "n": f"{int(r['n']):,}",
            "spearman": ("n/a" if not np.isfinite(r["spearman"])
                         else f"{r['spearman']:+.3f}"),
            "top-decile recall": ("n/a" if not np.isfinite(r["top_decile_recall"])
                                  else f"{r['top_decile_recall']:.3f}"),
        }

    s = per[per["stratum"] == "single"]
    m = per[per["stratum"] == "multi"]
    combined = pd.DataFrame(
        [dict(row(r), stratum="single") for _, r in s.iterrows()] +
        [dict(row(r), stratum="multi") for _, r in m.iterrows()])
    combined = combined[["assay", "stratum", "n", "spearman",
                         "top-decile recall"]]
    combined = combined.sort_values(["assay", "stratum"])

    pooled_rows = []
    for stratum, label in (("single", "single mutants"),
                           ("multi", "all multi-mutants"),
                           ("nsub2", "doubles only"),
                           ("nsub3", "triples only"),
                           ("nsub5plus", "5+ substitutions")):
        sub = per[per["stratum"] == stratum]
        if sub.empty:
            continue
        pooled_rows.append({
            "stratum": label,
            "assays": f"{len(sub)}",
            "variants scored": f"{int(sub['n'].sum()):,}",
            "mean spearman": f"{sub['spearman'].mean():+.3f}",
            "mean top-decile recall": f"{sub['top_decile_recall'].mean():.3f}",
        })
    pooled = pd.DataFrame(pooled_rows)

    gap_rows = pd.DataFrame({
        "assay": gaps["assay"],
        "spearman, single": [f"{v:+.3f}" for v in gaps["rho_single"]],
        "spearman, multi": [f"{v:+.3f}" for v in gaps["rho_multi"]],
        "gap (single - multi)": [f"{v:+.3f}" for v in gaps["gap_rho"]],
        "range-matched gap": [f"{v:+.3f}"
                              for v in gaps["range_matched_gap"]],
        "non-additive residual share (includes noise)": [
            f"{v:.3f}" for v in
            per.drop_duplicates("assay").set_index("assay")
            .loc[gaps["assay"], "nonadditive_residual_var_share"]],
    })

    return (fmt_table(combined, floatfmt="{:+.3f}") + "\n\n" +
            fmt_table(pooled, floatfmt="{:+.3f}") + "\n\n" +
            fmt_table(gap_rows, floatfmt="{:+.3f}"))


def epistasis_replicated_table():
    """Singles vs doubles against pooled two-assay targets, from
    results/epistasis_replicated.

    Separate from epistasis_table on purpose. That one compares two different
    variant sets inside one assay, so a stratum difference can be a difference
    in the measured quantity. This one intersects two assays of the same
    protein and compares singles against doubles on the same shared variant
    list. Neither pair is a clean replicate pair (the first table says why),
    so four targets are shown side by side and every interval is resampled by
    POSITION. Every number is read from the run's CSVs and manifest.
    """
    import json
    base = ROOT / "results" / "epistasis_replicated"
    per_path = base / "per_stratum.csv"
    rel_path = base / "reliability.csv"
    gap_path = base / "gaps.csv"
    man_path = base / "manifest.json"
    if not (per_path.exists() and rel_path.exists() and gap_path.exists()
            and man_path.exists()):
        return None

    per = pd.read_csv(per_path)
    rel = pd.read_csv(rel_path)
    gaps = pd.read_csv(gap_path)
    man = json.loads(man_path.read_text())

    def ci(v, lo, hi):
        return (f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]" if np.isfinite(v)
                else "n/a")

    pair_rows = []
    for key, p in man["pairs"].items():
        st, jn = p["pair_status"], p["join"]
        pair_rows.append({
            "protein": p["protein"],
            "pair": f"{p['assay_a'].split('_', 2)[2]} vs "
                    f"{p['assay_b'].split('_', 2)[2]}",
            "parallel measures": "yes" if st["parallel"] else "NO",
            "why not": "; ".join(st["reasons"]) or "-",
            "shared singles / doubles": (f"{p['shared_singles']:,} / "
                                         f"{p['shared_doubles']:,}"),
            "exact-string join": f"{jn['exact_string_intersection']:,}",
            "canonical join": f"{jn['canonical_intersection']:,}",
        })
    table_pairs = fmt_table(pd.DataFrame(pair_rows))

    rows = per[["protein", "target", "stratum", "n"]].copy()
    rows["spearman [position CI]"] = [
        ci(v, lo, hi) for v, lo, hi in zip(per["spearman"], per["spearman_lo"],
                                           per["spearman_hi"])]
    rows["top-decile recall"] = [
        "n/a" if not np.isfinite(v) else f"{v:.3f}"
        for v in per["top_decile_recall"]]
    rows["n"] = [f"{int(v):,}" for v in rows["n"]]
    table_a = fmt_table(rows)

    rr = rel[["protein", "stratum", "n", "split_half_spearman",
              "pearson_z_a_z_b", "pooled_reliability",
              "corr_pooled_z_with_a", "corr_pooled_z_with_b"]].copy()
    rr["n"] = [f"{int(v):,}" for v in rr["n"]]
    for col in ("split_half_spearman", "pearson_z_a_z_b",
                "pooled_reliability", "corr_pooled_z_with_a",
                "corr_pooled_z_with_b"):
        rr[col] = ["n/a" if not np.isfinite(v) else f"{v:.3f}" for v in rr[col]]
    rr = rr.rename(columns={
        "split_half_spearman": "spearman(A,B)",
        "pearson_z_a_z_b": "pearson(zA,zB)",
        "pooled_reliability": "Spearman-Brown (parallel pairs only)",
        "corr_pooled_z_with_a": "corr(z-pool, A)",
        "corr_pooled_z_with_b": "corr(z-pool, B)"})

    g = gaps.copy()
    out = pd.DataFrame({
        "protein": g["protein"],
        "target": [t + (" (primary)" if prim else "")
                   for t, prim in zip(g["target"], g["is_primary_target"])],
        "singles / doubles": [f"{int(a):,} / {int(b):,}"
                              for a, b in zip(g["n_singles"], g["n_doubles"])],
        "positions": [f"{int(v)}" for v in g["n_position_clusters"]],
        "gap rho (single - double)": [
            ci(v, lo, hi) for v, lo, hi in zip(
                g["gap_spearman"], g["gap_spearman_lo"], g["gap_spearman_hi"])],
        "range-matched gap": [
            ci(v, lo, hi) for v, lo, hi in zip(
                g["range_matched_gap"], g["range_matched_gap_lo"],
                g["range_matched_gap_hi"])],
        "top-decile gap": [
            ci(v, lo, hi) for v, lo, hi in zip(
                g["gap_recall"], g["gap_recall_lo"], g["gap_recall_hi"])],
    })

    return (table_pairs + "\n\n" + table_a + "\n\n" + fmt_table(rr)
            + "\n\n" + fmt_table(out))


def topk_ceiling_table():
    """The cross-assay ceiling, from results/topk.

    The prose around this table argues from these numbers, so they are
    rendered rather than typed: a hand-copied 0.35 that drifts to 0.30 would
    still read as a sentence.
    """
    path = ROOT / "results" / "topk" / "cross_assay_ceiling.csv"
    shape_path = ROOT / "results" / "topk" / "shape_vs_recall.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    per_assay = (ROOT / "results" / "full"
                 / "esm2_t33_650M_UR50D_per_assay.csv")
    model_recall = np.nan
    if per_assay.exists():
        pa = pd.read_csv(per_assay)
        esm = pa[pa["method"] == "esm2"]
        if len(esm):
            model_recall = float(esm["top10_recall"].mean())

    both = d["noise_exceeds_top_spread_a"] & d["noise_exceeds_top_spread_b"]
    n_model_assays = 0
    if per_assay.exists() and len(esm):
        n_model_assays = int(esm["top10_recall"].notna().sum())
    rows = [
        ("replicate assay pairs", f"{len(d)}"),
        ("proteins covered", f"{d['uniprot'].nunique()}"),
        ("median top-10 recall, truth against truth",
         f"{d['top10_recall_mean'].median():.3f}"),
        ("median Spearman over the whole shared set",
         f"{d['spearman_whole'].median():.3f}"),
        ("median Jaccard overlap of the two top deciles",
         f"{d['jaccard_top_decile'].median():.3f}"),
        ("median per-measurement error (sd units)",
         f"{d['noise_sd_z'].median():.3f}"),
        ("median top-decile internal spread (sd units)",
         f"{d['top_decile_sd_a'].median():.3f}"),
        ("pairs whose decile spread is smaller than one measurement's "
         "error, on both sides", f"{int(both.sum())} of {len(d)}"),
        ("pairs whose top-decile 'reliability' formula is at or below zero "
         "(decile selected on the same assay: see the simulated control)",
         f"{int((d['reliability_top_decile'] <= 0).sum())} of {len(d)}"),
        (f"the model's mean top-10 recall, {n_model_assays} assays",
         "n/a" if not np.isfinite(model_recall) else f"{model_recall:.3f}"),
    ]
    out = pd.DataFrame(rows, columns=["measurement", "value"])

    if shape_path.exists():
        s = pd.read_csv(shape_path)
        s = s.rename(columns={"predictor": "decile-shape predictor",
                              "n_assays": "assays",
                              "p": "p"})
        s = s[["decile-shape predictor", "assays", "spearman", "p"]]
        return (fmt_table(out, floatfmt="{:.3f}") + "\n\n" +
                "Does the shape of the top decile predict the model's recall?"
                "\n\n" + fmt_table(s, floatfmt="{:+.3f}"))
    return fmt_table(out, floatfmt="{:.3f}")


def latent_truth_table():
    """The pooled-truth test: is 0.350 an upper bound, or attenuated agreement?

    Read from results/latent_truth/summary.json, which
    scripts/run_latent_truth_test.py writes. Everything here is a paired
    quantity: the same variants of the same pairs, one target swapped for
    another. Nothing compares a median of 37 pairs to a mean of 217 assays,
    which is what made the old 0.350-vs-0.323 sentence unsound.
    """
    path = ROOT / "results" / "latent_truth" / "summary.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    allp = s["assays"]
    same = s.get("same_quantity", {})
    rel = s.get("reliable_only", {})
    bench = s.get("benchmark_reference", {})

    def f(x, spec="{:+.3f}"):
        if x is None or not np.isfinite(x):
            return "n/a"
        return spec.format(x)

    def pv(x):
        """A p of 4e-10 must not print as 0.0000: 'p = 0' reads as a failure
        of the test, not an extreme result."""
        if x is None or not np.isfinite(x):
            return "n/a"
        return "< 0.0001" if x < 1e-4 else f"{x:.4f}"

    w = allp.get("wilcoxon_model_pooled_minus_single") or {}
    w_r = allp.get("wilcoxon_spearman_pooled_minus_single") or {}
    w_t = allp.get("wilcoxon_assay_to_pooled_minus_truth_vs_truth") or {}
    pw = allp.get("protein_level_recall_gain") or {}
    pw_r = allp.get("protein_level_spearman_gain") or {}

    def three(rep):
        """Wilcoxon (tie-corrected approx) / exact / sign-flip permutation, all
        shown so no method is picked for the p it gives."""
        if not rep:
            return "n/a"
        return (f"{pv(rep.get('p'))} / {pv(rep.get('p_exact'))} / "
                f"{pv(rep.get('p_signflip'))}")

    rows = [
        ("replicate pairs with >= {} shared variants".format(
            s["threshold_shared_variants"]), f"{allp['n_pairs']}"),
        ("proteins covered", f"{allp['n_proteins']}"),
        ("shared variant measurements", f"{allp['n_variants']}"),
        ("median truth-vs-truth top-10 recall (the old 0.350 definition)",
         f(allp["median_recall_truth_vs_truth"], "{:.3f}")),
        ("mean truth-vs-truth top-10 recall, same pairs",
         f(allp["mean_recall_truth_vs_truth"], "{:.3f}")),
        ("median top-10 recall, assay against the pooled target",
         f(allp["median_recall_a_to_pooled"], "{:.3f}")),
        ("median top-10 recall, model against one assay",
         f(allp["median_recall_model_to_single"], "{:.3f}")),
        ("median top-10 recall, model against the pooled target",
         f(allp["median_recall_model_to_pooled"], "{:.3f}")),
        ("PROTEIN level (primary): proteins",
         f"{pw.get('n_units', 'n/a')}"),
        ("protein-level median paired gain, top-10 recall",
         f(pw.get("median_diff"))),
        ("protein-level proteins better / worse / tied",
         f"{pw.get('n_better', 'n/a')} / {pw.get('n_worse', 'n/a')} / "
         f"{pw.get('n_zero', 'n/a')}"),
        ("protein-level p, top-10 recall (Wilcoxon / exact / sign-flip)",
         three(pw)),
        ("pair level (secondary): median paired gain, top-10 recall",
         f(w.get("median_diff"))),
        ("pair-level pairs better / worse / tied",
         f"{w.get('n_better', 0)} / {w.get('n_worse', 0)} / "
         f"{w.get('n_zero', 0)} of {allp['n_pairs']}"),
        ("pair-level p, top-10 recall (Wilcoxon / exact / sign-flip)",
         three(w)),
        ("median Spearman, model against one assay",
         f(allp["median_spearman_model_to_single"], "{:.3f}")),
        ("median Spearman, model against the pooled target",
         f(allp["median_spearman_model_to_pooled"], "{:.3f}")),
        ("protein-level median paired gain, Spearman",
         f(pw_r.get("median_diff"))),
        ("protein-level proteins better / worse",
         f"{pw_r.get('n_better', 'n/a')} / {pw_r.get('n_worse', 'n/a')}"),
        ("protein-level p, Spearman (Wilcoxon / exact / sign-flip)",
         three(pw_r)),
        ("pair-level median paired gain, Spearman",
         f(w_r.get("median_diff"))),
        ("pair-level pairs better / worse",
         f"{w_r.get('n_better', 0)} / {w_r.get('n_worse', 0)}"),
        ("pair-level p, Spearman (Wilcoxon / exact / sign-flip)",
         three(w_r)),
        ("median inter-assay rank agreement of a pair (a reliability only "
         "for same-quantity pairs)",
         f(allp["median_split_half_reliability"], "{:.3f}")),
        ("pairs at or below 0.3 agreement, pooling uninformative",
         f"{allp['n_low_reliability']} of {allp['n_pairs']}"),
    ]
    if same:
        rows += [
            ("same-quantity pairs only: n", f"{same['n_pairs']}"),
            ("same-quantity median paired gain, top-10 recall",
             f((same.get("wilcoxon_model_pooled_minus_single")
                or {}).get("median_diff"))),
            ("same-quantity pair-level p, top-10 recall "
             "(Wilcoxon / exact / sign-flip)",
             three(same.get("wilcoxon_model_pooled_minus_single"))),
            ("same-quantity proteins",
             f"{(same.get('protein_level_recall_gain') or {}).get('n_units', 'n/a')}"),
            ("same-quantity protein-level p, top-10 recall "
             "(Wilcoxon / exact / sign-flip)",
             three(same.get("protein_level_recall_gain"))),
            ("same-quantity protein-level p, Spearman "
             "(Wilcoxon / exact / sign-flip)",
             three(same.get("protein_level_spearman_gain"))),
        ]
    if rel:
        rows += [
            ("reliable pairs only (split-half > 0.3): n", f"{rel['n_pairs']}"),
            ("reliable-pair median paired gain, top-10 recall",
             f((rel.get("wilcoxon_model_pooled_minus_single")
                or {}).get("median_diff"))),
            ("reliable-pair protein-level p, top-10 recall "
             "(Wilcoxon / exact / sign-flip)",
             three(rel.get("protein_level_recall_gain"))),
        ]
    if bench:
        rows += [
            (f"model mean top-10 recall, all {bench['n_all_assays']} assays",
             f(bench["mean_recall_all_assays"], "{:.3f}")),
            (f"model mean top-10 recall, the {bench['n_paired_assays']} "
             "assays that have a replicate",
             f(bench["mean_recall_paired_assays"], "{:.3f}")),
        ]
    out = pd.DataFrame(rows, columns=["pooled-truth measurement", "value"])
    tail = ""
    if w_t:
        # The assay row is ~1 by construction: the assay is HALF of the target
        # it is scored against, so its own noise counts as signal. The gain is
        # therefore mostly self-inclusion, and says nothing about attenuation
        # of 0.350 (scripts/sim_pooling_controls.py measures the effect on
        # simulated assays).
        tail = ("\n\nSelf-inclusion control, same pairs: an assay scored "
                "against a pooled target that contains it gains "
                f"{f(w_t.get('median_diff'))} over truth-against-truth "
                f"(Wilcoxon p {pv(w_t.get('p'))}). That gain is self-inclusion "
                "(the assay is half of its own target), not a measure of how "
                "much attenuation 0.350 contained: see the simulated control "
                "below.")
    return fmt_table(out) + tail


def sim_controls_table():
    """The simulated self-inclusion and decile-reliability controls, from
    results/sim_controls/summary.json (scripts/sim_pooling_controls.py)."""
    path = ROOT / "results" / "sim_controls" / "summary.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    si, dr = s["selfinclusion"], s["decile_reliability"]
    rows = []
    for label, key in (("two assays with NO shared signal", "no_shared_signal"),
                       (f"two assays of one latent value, noise sd "
                        f"{si['noise_sd_z']:.3f} (the real median)",
                        "shared_latent")):
        v = si[key]
        rows.append({
            "simulation": label,
            "A vs B": f"{v['a_to_b']:.3f}",
            "A vs pool(A,B)": f"{v['a_to_pooled']:.3f}",
            "B vs pool(A,B)": f"{v['b_to_pooled']:.3f}",
            "latent vs A": f"{v['latent_to_single_a']:.3f}",
            "latent vs pool": f"{v['latent_to_pooled']:.3f}",
        })
    t1 = fmt_table(pd.DataFrame(rows))
    t2 = fmt_table(pd.DataFrame([{
        "reps": dr["reps"],
        "sigma_hat": f"{dr['sigma_hat']:.3f}",
        "variance of A in its top decile": f"{dr['var_observed_in_subset']:.3f}",
        "reliability formula": f"{dr['formula_reliability']:+.3f}",
        "reps with negative formula": f"{dr['share_of_reps_with_negative_formula']:.0%}",
        "true latent variance in that decile": f"{dr['var_latent_in_subset']:.3f}",
        "corr(A, B) in that decile": f"{dr['corr_a_b_in_subset']:+.3f}",
        "corr(latent, A) in that decile": f"{dr['corr_latent_a_in_subset']:+.3f}",
    }]))
    return (f"Top-10 recall, mean over {si['reps']} simulations of "
            f"{si['n_variants']} variants:\n\n" + t1 + "\n\nDecile "
            "reliability, formula against truth, top decile of A selected on A "
            "itself:\n\n" + t2)


TABLES = {
    "full_benchmark": full_benchmark_table,
    "full_benchmark_coverage": full_benchmark_coverage,
    "viral_comparison": viral_table,
    "scaling_headline": scaling_headline_table,
    "scaling_paired": paired_table,
    "scaling_local": local_curve_table,
    "scaling_subgroups": scaling_table,
    "structure_burial": burial_table,
    "structure_heldout": structure_heldout_text,
    "scope_audit": scope_audit_table,
    "topk_ceiling": topk_ceiling_table,
    "latent_truth": latent_truth_table,
    "sim_controls": sim_controls_table,
    "epistasis": epistasis_table,
    "epistasis_replicated": epistasis_replicated_table,
    "family_viral_gap": family_gap_table,
    "family_separation": family_separation_text,
    "quoted_headlines": quoted_vs_published_text,
}


def render(name):
    fn = TABLES[name]
    out = fn()
    if out is None:
        return f"_({name}: results not present; run the corresponding script)_"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    docs = {p: p.read_text() for p in (README, RESULTS_PAGE) if p.exists()}
    for name in TABLES:
        body = render(name)
        home = RESULTS_PAGE if name in PAGE_TABLES else README
        begin, end = BEGIN.format(name), END.format(name)
        if begin not in docs.get(home, ""):
            print(f"(no marker for {name} in {home.name})")
            continue
        pre, rest = docs[home].split(begin, 1)
        _, post = rest.split(end, 1)
        docs[home] = pre + begin + "\n" + body + "\n" + end + post
        print(f"--- {name} ---")
        print(body)
        print()

    if args.write:
        for p, t in docs.items():
            p.write_text(t)
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
