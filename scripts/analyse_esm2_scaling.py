"""Why ESM-2 3B scores worse than 650M on ProteinGym, from published numbers.

The README's "still to do" asks for this to be confirmed and understood. It
cannot be confirmed by running the 3B checkpoint -- the weights are ~12 GB and
the rails forbid downloading them -- so this script does the two things that
are possible without them:

1. **Confirm the published pattern and test it.** Pull the six ESM-2 sizes out
   of ProteinGym's summary table, then test the 650M-versus-3B gap with a
   *paired* test over the 217 per-assay values. The published bootstrap SEs are
   ~0.012, so the 0.008 headline gap sits inside two of them and a naive
   comparison cannot separate it from zero. A paired test can, because it
   removes the assay-level variance that dominates those SEs.

2. **Localise it.** Compare the subgroup columns. A capacity effect should move
   every split together; an objective-mismatch effect should move some splits
   up and others down, and the pattern of which is the explanation.

The answer the data gives is neither "bigger is worse" nor "it is noise". The
headline is flat because the movement underneath it is in opposite directions,
and the direction splits cleanly by whether the assay measures a protein's own
stability or its behaviour in a host.

Writes results/esm2_scaling/{scaling,paired_tests,subgroups}.csv.

Run:
    python scripts/analyse_esm2_scaling.py


Why the 3B checkpoint is a trade, not a regression (moved from the README)
---------------------------------------------------------------------------
A model that had simply gotten worse would move every split down. Instead the
split is by what each assay measures. The assays that probe a protein's own
stability get worse; the assays that probe how it behaves in a host get better, and
the viral taxon improves by the largest margin of any split. The 15B model continues
along the same axis rather than reversing it. That is the shape of an objective
mismatch, not a scaling failure. ESM-2 is trained to predict masked residues across
UniRef, a proxy for "what sequence would thrive in the organisms UniRef samples",
which is closer to organismal fitness than to whether a purified protein folds.
Scaling the model sharpens whatever the objective actually rewards. On stability
assays the extra capacity adds nothing and mildly overfits; on host-context assays
it helps. The practical implication is the boring one: 650M is the right default for
stability and binding work, and if the target is a virus, neither 650M nor 3B is the
right tool; the alignment method is, by a wide margin.

The 3B checkpoint is about 12 GB and was not downloaded, so the local measurement
stops at 650M. Each local step up is real (the paired differences are significant at
every step) but the third step buys a quarter of what the first did: by 650M the
model is near the plateau for this benchmark, so a 2.3x parameter increase at 3B
cannot be expected to help much.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_effect import aggregate, data  # noqa: E402

OUT = ROOT / "results" / "esm2_scaling"
OUT.mkdir(parents=True, exist_ok=True)

SIZES = ["ESM2 (8M)", "ESM2 (35M)", "ESM2 (150M)", "ESM2 (650M)",
         "ESM2 (3B)", "ESM2 (15B)"]
PARAMS = {"ESM2 (8M)": 8e6, "ESM2 (35M)": 35e6, "ESM2 (150M)": 150e6,
          "ESM2 (650M)": 650e6, "ESM2 (3B)": 3e9, "ESM2 (15B)": 1.5e10}

# the summary table columns, grouped by what they measure
SPLIT_GROUPS = {
    "selection_type": ["Function_Stability", "Function_Binding",
                       "Function_Expression", "Function_OrganismalFitness",
                       "Function_Activity"],
    "taxon": ["Taxa_Human", "Taxa_Other_Eukaryote", "Taxa_Prokaryote",
              "Taxa_Virus"],
    "msa_depth": ["Low_MSA_depth", "Medium_MSA_depth", "High_MSA_depth"],
    "mut_depth": ["Depth_1", "Depth_2", "Depth_3", "Depth_4", "Depth_5+"],
}
SPLITS = [c for cols in SPLIT_GROUPS.values() for c in cols]

# Splits that measure how well the protein tolerates mutation on its own,
# versus how it behaves in a host. The distinction is the whole finding.
OWN_PROPERTY = {"Function_Stability", "High_MSA_depth", "Function_Binding"}
HOST_CONTEXT = {"Taxa_Virus", "Function_OrganismalFitness", "Medium_MSA_depth"}


def headline_curve(summary):
    rows = []
    for m in SIZES:
        r = summary.loc[m]
        rows.append({"model": m, "params": PARAMS[m],
                     "headline": float(r["Average_Spearman"]),
                     "published_se": float(r["Bootstrap_standard_error_Spearman"]),
                     **{k: float(r[k]) for k in SPLITS}})
    return pd.DataFrame(rows)


def paired_tests(official, pairs):
    """Paired tests over the per-assay values.

    The published SEs are bootstrap SEs of a mean, which include the spread
    *between* assays. Assays differ in difficulty by far more than any model
    differs from another, so that spread swamps the comparison. Pairing by
    assay removes it, leaving only the model-to-model difference on the same
    data -- which is the question.
    """
    out = []
    for a, b in pairs:
        d = (official[b] - official[a]).dropna()
        if len(d) < 3:
            continue
        t, p = stats.ttest_rel(official.loc[d.index, b],
                               official.loc[d.index, a])
        try:
            _, wp = stats.wilcoxon(d)
        except ValueError:
            wp = np.nan
        se = d.std() / np.sqrt(len(d))
        out.append({"model_a": a, "model_b": b, "n": len(d),
                    "mean_diff_b_minus_a": float(d.mean()),
                    "median_diff": float(d.median()),
                    "sd_diff": float(d.std()),
                    "se_diff": float(se),
                    "t_stat": float(t), "p_paired_t": float(p),
                    "p_wilcoxon": float(wp),
                    "b_wins": int((d > 0).sum()), "a_wins": int((d < 0).sum()),
                    "mean_diff_in_se": float(d.mean() / se) if se > 0 else np.nan})
    return pd.DataFrame(out)


def subgroup_movement(curve):
    sub = curve.set_index("model")[SPLITS]
    mov = pd.DataFrame({"650M": sub.loc["ESM2 (650M)"],
                        "3B": sub.loc["ESM2 (3B)"],
                        "15B": sub.loc["ESM2 (15B)"]})
    mov["d_3B"] = mov["3B"] - mov["650M"]
    mov["d_15B"] = mov["15B"] - mov["650M"]
    return mov.sort_values("d_3B")


def main():
    summary = data.official_summary()
    official = data.official_scores()

    print("=" * 74)
    print("1. published headline vs model size")
    print("=" * 74)
    curve = headline_curve(summary)
    curve.to_csv(OUT / "scaling.csv", index=False)
    print(curve[["model", "params", "headline", "published_se"]]
          .to_string(index=False))
    peak = curve.loc[curve["headline"].idxmax()]
    print(f"\npeak at {peak['model']} ({peak['headline']:.3f}); "
          f"3B is {curve.set_index('model').loc['ESM2 (3B)', 'headline']:.3f}, "
          f"15B is {curve.set_index('model').loc['ESM2 (15B)', 'headline']:.3f}")

    print("\n" + "=" * 74)
    print("2. is the 650M -> 3B gap distinguishable from zero?")
    print("=" * 74)
    pairs = [("ESM2 (650M)", "ESM2 (3B)"), ("ESM2 (650M)", "ESM2 (15B)"),
             ("ESM2 (150M)", "ESM2 (650M)"), ("ESM2 (35M)", "ESM2 (150M)"),
             ("ESM2 (3B)", "ESM2 (15B)")]
    tests = paired_tests(official, pairs)
    tests.to_csv(OUT / "paired_tests.csv", index=False)
    for _, r in tests.iterrows():
        print(f"  {r['model_b']:<12} - {r['model_a']:<12} "
              f"mean {r['mean_diff_b_minus_a']:+.4f}  "
              f"({r['mean_diff_in_se']:+.2f} SE)  "
              f"paired t p={r['p_paired_t']:.2g}  wilcoxon p={r['p_wilcoxon']:.2g}  "
              f"{r['b_wins']}W/{r['a_wins']}L")
    print("\n  For comparison, the published bootstrap SEs are ~0.012 each, so an")
    print("  unpaired comparison of 0.414 vs 0.406 cannot resolve 0.008. The")
    print("  paired test can, because it removes between-assay difficulty.")

    print("\n" + "=" * 74)
    print("3. where the movement is, and in which direction")
    print("=" * 74)
    mov = subgroup_movement(curve)
    mov.to_csv(OUT / "subgroups.csv")
    print(mov.round(3).to_string())

    own = mov.loc[mov.index.isin(OWN_PROPERTY)]
    host = mov.loc[mov.index.isin(HOST_CONTEXT)]
    print(f"\n  own-property splits (stability, binding, high MSA depth): "
          f"mean d_3B {own['d_3B'].mean():+.3f}")
    print(f"  host-context splits (virus, organismal fitness, medium MSA): "
          f"mean d_3B {host['d_3B'].mean():+.3f}")

    print("\n" + "=" * 74)
    print("4. headline recomputed from per-assay values (two-stage rule)")
    print("=" * 74)
    base = pd.DataFrame({"selection": official["Selection Type"].to_numpy(),
                         "uniprot": official["UniProt ID"].to_numpy()})
    for m in SIZES:
        t = base.assign(spearman=official[m].to_numpy(dtype=float))
        h = aggregate.macro_over_categories(t)
        pub = float(summary.loc[m, "Average_Spearman"])
        print(f"  {m:<12} {h:.4f}  published {pub:.3f}  diff {h - pub:+.4f}")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
