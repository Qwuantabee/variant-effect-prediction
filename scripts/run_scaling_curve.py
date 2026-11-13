"""Local ESM-2 scaling curve: does our implementation show the same plateau?

The published per-assay table can be analysed without any weights, but the
scaling story is only *our* result if we measure it. This scores the same
subset of assays with each locally available checkpoint and reports the paired
difference, so the claim "650M > 150M > 35M" is checked by this code rather
than quoted.

Only checkpoints already in the HF cache are used. 3B and 15B are not
downloaded (12 GB and 60 GB) and are analysed from the published table in
scripts/analyse_esm2_scaling.py instead.

Run:
    HF_HUB_OFFLINE=1 python scripts/run_scaling_curve.py
    HF_HUB_OFFLINE=1 python scripts/run_scaling_curve.py --assays A B C

Writes results/esm2_scaling/local_curve.csv and prints the paired comparison.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_effect import data, evaluate  # noqa: E402

OUT = ROOT / "results" / "esm2_scaling"
OUT.mkdir(parents=True, exist_ok=True)

# smallest to largest, all of which fit comfortably in 16 GB at fp16
MODELS = [
    ("8M", "facebook/esm2_t6_8M_UR50D"),
    ("35M", "facebook/esm2_t12_35M_UR50D"),
    ("150M", "facebook/esm2_t30_150M_UR50D"),
    ("650M", "facebook/esm2_t33_650M_UR50D"),
]

# a spread of sizes and taxa, long enough to separate the checkpoints but
# short enough to run in minutes. Deliberately includes the viral assay.
ASSAYS = [
    "IF1_ECOLI_Kelsic_2016",
    "CCDB_ECOLI_Tripathi_2016",
    "SUMO1_HUMAN_Weile_2017",
    "RL40A_YEAST_Roscoe_2013",
    "DYR_ECOLI_Thompson_2019",
    "NUD15_HUMAN_Suiter_2020",
    "TPMT_HUMAN_Matreyek_2018",
    "BLAT_ECOLX_Stiffler_2015",
    "R1AB_SARS2_Flynn_2022",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assays", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    assays = args.assays or ASSAYS
    ref = data.reference_table()
    data.ensure_assays(assays)

    rows = []
    for label, model in MODELS:
        from variant_effect.scoring import ESMScorer
        print(f"\n=== {label} ({model}) ===", flush=True)
        try:
            scorer = ESMScorer(model)
        except Exception as exc:
            print(f"  unavailable: {type(exc).__name__}: {exc}", flush=True)
            continue
        print(f"  {scorer}", flush=True)
        for aid in assays:
            row = ref[ref["DMS_id"] == aid].iloc[0]
            df = pd.read_csv(data.ASSAY_DIR / f"{aid}.csv")
            variants = df["mutant"].astype(str).tolist()
            truth = df["DMS_score"].to_numpy(dtype=float)
            t0 = time.time()
            if int(row["seq_len"]) > 1022:
                sc = scorer.score_variants_windowed(
                    row["target_seq"], variants, batch_size=args.batch_size)
            else:
                sc = scorer.score_variants(
                    row["target_seq"], variants, batch_size=args.batch_size)
            rho = evaluate.spearman(sc, truth)
            print(f"  {aid:<32} rho={rho:+.4f}  ({time.time() - t0:.0f}s)",
                  flush=True)
            rows.append({"model": label, "assay": aid, "spearman": rho,
                         "uniprot": row["UniProt_ID"],
                         "taxon": row["taxon"],
                         "selection": row["coarse_selection_type"]})
        # the previous checkpoint's memory has to go before the next loads
        del scorer
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    d = pd.DataFrame(rows)
    if d.empty:
        print("no results")
        return
    d.to_csv(OUT / "local_curve.csv", index=False)

    print("\n--- local mean Spearman by checkpoint ---")
    print(d.groupby("model")["spearman"].agg(["mean", "std", "count"])
          .round(4).to_string())

    pivot = d.pivot_table(index="assay", columns="model", values="spearman")
    order = [m for m, _ in MODELS if m in pivot.columns]
    print("\n--- paired differences between adjacent checkpoints ---")
    from scipy import stats
    for a, b in zip(order, order[1:]):
        diff = (pivot[b] - pivot[a]).dropna()
        if len(diff) < 3:
            continue
        t, p = stats.ttest_rel(pivot.loc[diff.index, b],
                               pivot.loc[diff.index, a])
        print(f"  {b:<6} - {a:<6} mean {diff.mean():+.4f}  "
              f"sd {diff.std():.4f}  paired t p={p:.3g}  n={len(diff)}")


if __name__ == "__main__":
    main()
