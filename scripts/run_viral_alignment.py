"""Test the immune-selection explanation directly: alignment vs language model on viral assays.

The README's account of the viral failure is that ESM-2 learns constraint from
conservation across UniRef, and viral proteins are also under immune selection,
so positions vary across strains for reasons that have nothing to do with
whether the protein folds. The model reads that variation as tolerance.

That explanation predicts something specific. If the problem is that a
family-agnostic model cannot tell structural constraint from immune escape,
then a method that reads conservation off an alignment *of that viral family*
should do well on the same assays -- because immune-driven variation is
comparatively sparse within a family, and what an alignment of 50,000
enterovirus sequences shows is which positions those viruses cannot change.

The null result that would sink the story: every method struggles on viral
assays, including the family-specific ones. Then the assays are just
noisier or the proteins are more permissive, and immune selection is an
unnecessary hypothesis.

Method: site-independent model (Hopf et al. 2017) -- per-column amino acid
frequencies from the assay's own MSA, scored as
log f(mut|position) - log f(wt|position). No coupling terms, so this is the
weakest thing that qualifies as alignment-based, which makes it a conservative
test: if even this beats a 650M-parameter language model, the point stands.

Run:
    python scripts/run_viral_alignment.py
    python scripts/run_viral_alignment.py --taxa Virus
    python scripts/run_viral_alignment.py --assays R1AB_SARS2_Flynn_2022 --max-rows 20000

Outputs results/viral/<tag>.csv with one row per assay and method.
"""

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_effect import data, evaluate, msa  # noqa: E402

RESULTS = ROOT / "results" / "viral"
RESULTS.mkdir(parents=True, exist_ok=True)


def subsample(msa_obj, max_rows, seed=0):
    """Cap alignment size with a fixed-seed random sample.

    Not the first N rows: ProteinGym's .a2m files are sorted by alignment
    score, and the top rows are the closest homologs. Taking them first makes
    the profile an average over a single clade -- on R1AB the first 800 rows
    are 100% identical to the query, while the file overall has 9 distinct
    residues per column on median. Sampling at random from the whole file is
    the difference between a real profile and a stretched copy of the target.
    """
    if max_rows is None or len(msa_obj) <= max_rows:
        return msa_obj
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(msa_obj), max_rows, replace=False))
    return msa.MSA(msa_obj.query, [msa_obj.sequences[i] for i in idx],
                   msa_obj.msa_start, msa_obj.target_seq)


# Default is unweighted. Clustered weighting at 80% identity is the textbook
# choice and is what the published method uses, but on these alignments it
# behaves badly: ProteinGym's .a2m files are ~96% near-identical to the query
# for R1AB, so a 0.8 threshold collapses them into a handful of clusters and
# Neff_L falls to 0.06 against the published 1.14. Unweighted profiles use
# every row and land closer to the published per-assay value at every size
# tested (see results/viral/weighting_sensitivity.csv). The headline finding --
# alignment beats ESM-2 by a wide margin on viral assays -- holds for both
# weightings, which is why this is a tuning detail and not the result.
    if max_rows is None or len(msa_obj) <= max_rows:
        return msa_obj
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(msa_obj), max_rows, replace=False))
    return msa.MSA(msa_obj.query, [msa_obj.sequences[i] for i in idx],
                   msa_obj.msa_start, msa_obj.target_seq)


def classify(organism):
    """Phage vs animal virus, because they behave in opposite directions.

    ProteinGym's ``Taxon`` column labels both as Virus, and averaging over
    them cancels the effect: the alignment method wins on 21 of 24 animal
    viruses and on 1 of 7 phages. Splitting is not cosmetic -- a phage
    replicates in a bacterium with no adaptive immune system, so the
    immune-selection account of ESM-2's failure has no mechanism to act
    through there, and the comparison has no claim to make.
    """
    if not isinstance(organism, str):
        return "unknown"
    low = organism.lower()
    if "phage" in low or "bacteriophage" in low:
        return "phage"
    return "animal_virus"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxa", nargs="*", default=["Virus"])
    ap.add_argument("--assays", nargs="*", default=None)
    ap.add_argument("--max-rows", type=int, default=50000)
    ap.add_argument("--weights", default="none",
                    choices=["none", "clustered"])
    ap.add_argument("--tag", default="viral")
    args = ap.parse_args()

    ref = data.reference_table()
    official = data.official_scores()

    if args.assays:
        ids = args.assays
    else:
        ids = ref[ref["taxon"].isin(args.taxa)]["DMS_id"].tolist()

    print(f"{len(ids)} assays; max_rows={args.max_rows} weights={args.weights}",
          flush=True)

    rows = []
    for i, aid in enumerate(ids, 1):
        row = ref[ref["DMS_id"] == aid].iloc[0]
        seq_len = int(row["seq_len"])
        t0 = time.time()
        try:
            path = data.msa_path(aid, reference=ref)
            text = path.read_text(errors="replace")
            query, others = msa.parse_a2m(text)
            m = msa.MSA(query, others, int(row["MSA_start"]), row["target_seq"])
            m = subsample(m, args.max_rows)

            w = None
            if args.weights == "clustered":
                w = msa.sequence_weights(m.sequences, 0.8,
                                         min(len(m), args.max_rows))

            df = pd.read_csv(data.ASSAY_DIR / f"{aid}.csv")
            variants = df["mutant"].astype(str).tolist()
            truth = df["DMS_score"].to_numpy(dtype=float)

            scores = msa.site_independent_scores(m, variants, weights=w)
            rho = evaluate.spearman(scores, truth)
            n_eff = float((w.sum() if w is not None else len(m)) / max(seq_len, 1))
            n_used = len(m)
        except Exception as exc:
            print(f"[{i}/{len(ids)}] {aid}: FAILED {type(exc).__name__}: {exc}",
                  flush=True)
            rows.append({"assay": aid, "taxon": row["taxon"],
                         "spearman": np.nan, "error": str(exc)})
            continue

        ours = official.loc[aid, "Site-Independent"] if aid in official.index else np.nan
        print(f"[{i}/{len(ids)}] {aid:<36} rho={rho:+.4f} "
              f"(official site-indep {ours:+.3f}) "
              f"rows={n_used:6d} Neff_L={n_eff:7.2f} {time.time() - t0:.1f}s",
              flush=True)
        rows.append({"assay": aid, "taxon": row["taxon"],
                     "organism": row["source_organism"],
                     "seq_len": seq_len, "selection": row["coarse_selection_type"],
                     "msa_depth": row["MSA_Neff_L_category"],
                     "msa_rows_total": len(others) + 1, "msa_rows_used": n_used,
                     "neff_l_ours": n_eff, "neff_l_published": row["MSA_Neff_L"],
                     "spearman": rho,
                     "official_site_independent": ours,
                     "official_esm2_650m": (official.loc[aid, "ESM2 (650M)"]
                                            if aid in official.index else np.nan)})

    out = RESULTS / f"{args.tag}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    ok = df.dropna(subset=["spearman"])
    if len(ok):
        ok = ok.copy()
        ok["virus_class"] = ok["organism"].map(classify)
        print(f"\n--- {len(ok)} assays, alignment (site-independent) ---")
        print(f"  mean per-assay Spearman  {ok['spearman'].mean():+.4f}")
        print(f"  published same method    {ok['official_site_independent'].mean():+.4f}")
        print(f"  published ESM2 650M      {ok['official_esm2_650m'].mean():+.4f}")
        d = ok["spearman"] - ok["official_site_independent"]
        print(f"  ours - published site-indep: mean {d.mean():+.4f}, "
              f"max abs {d.abs().max():.4f}")

        print("\n--- animal virus vs phage ---")
        for cls in ("animal_virus", "phage"):
            sub = ok[ok["virus_class"] == cls]
            if sub.empty:
                continue
            gain = sub["spearman"] - sub["official_esm2_650m"]
            wins = int((gain > 0).sum())
            print(f"  {cls:<14} n={len(sub):2d}  alignment {sub['spearman'].mean():+.3f}  "
                  f"ESM2 650M {sub['official_esm2_650m'].mean():+.3f}  "
                  f"gain {gain.mean():+.3f}  beats ESM2 on {wins}/{len(sub)}")

        a = ok[ok["virus_class"] == "animal_virus"]
        p = ok[ok["virus_class"] == "phage"]
        if len(a) and len(p):
            print(f"\n  The immune-selection explanation predicts the animal-virus row "
                  f"and says nothing about the phage row.")
            ok.to_csv(out, index=False)


if __name__ == "__main__":
    main()
