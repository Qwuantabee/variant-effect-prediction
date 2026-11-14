"""Generate the README's result tables from the result files, so no number is
hand-copied.

Every table in the README that reports a measured value is written by this
script from CSVs that other scripts produced. That is the only way to keep the
prose honest: a number in the README that was typed by hand can drift from the
file it came from, and a reader has no way to tell.

Run:
    python scripts/report_tables.py            # print
    python scripts/report_tables.py --write    # rewrite the generated blocks

Markers in the README delimit the generated regions:
    <!-- BEGIN GENERATED: <name> -->  ...  <!-- END GENERATED: <name> -->
Anything outside those markers is prose and is left alone.
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
        "paired t p": d["p_paired_t"],
        "Wilcoxon p": d["p_wilcoxon"],
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
    path = ROOT / "results" / "families" / "family_separation.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    if d.empty or not np.isfinite(d["p_no_alignment_worse"].iloc[0]):
        return None
    r = d.iloc[0]
    return (
        f"Across the {int(r['n_no_alignment'])} checkpoints in families that do "
        f"not read an alignment, the median viral gap is "
        f"{r['median_gap_no_alignment']:+.3f}; across the "
        f"{int(r['n_alignment'])} that do, it is "
        f"{r['median_gap_alignment']:+.3f} "
        f"(Mann-Whitney one-sided p={r['p_no_alignment_worse']:.2g})."
    )


def quoted_vs_published_text():
    path = ROOT / "results" / "families" / "quoted_vs_published.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    parts = [f"{r['model']} {r['published_headline']:.3f}" for _, r in d.iterrows()]
    return "ProteinGym headlines, read from the published summary table: " + \
        ", ".join(parts) + "."


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

    text = README.read_text()
    updated = text
    for name in TABLES:
        body = render(name)
        begin, end = BEGIN.format(name), END.format(name)
        if begin not in updated:
            print(f"(no marker for {name} in README)")
            continue
        pre, rest = updated.split(begin, 1)
        _, post = rest.split(end, 1)
        updated = pre + begin + "\n" + body + "\n" + end + post
        print(f"--- {name} ---")
        print(body)
        print()

    if args.write:
        README.write_text(updated)
        print(f"wrote {README}")


if __name__ == "__main__":
    main()
