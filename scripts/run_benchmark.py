"""Benchmark ESM-2 and the cheap baselines across a set of DMS assays.

Run:
    python scripts/run_benchmark.py                      # default 8 assays, 650M
    python scripts/run_benchmark.py --model facebook/esm2_t12_35M_UR50D
    python scripts/run_benchmark.py --assays BLAT_ECOLX_Stiffler_2015
    python scripts/run_benchmark.py --scheme wt          # cheap, worse

Writes results/benchmark_<model>.csv and prints a comparison against the
official ProteinGym numbers for the same model.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_effect import baselines, data, evaluate  # noqa: E402
from variant_effect.scoring import ESMScorer  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# maps our --model argument onto the column name in ProteinGym's published
# per-assay results, where one exists
OFFICIAL_COLUMN = {
    "facebook/esm2_t33_650M_UR50D": "ESM2 (650M)",
    "facebook/esm2_t36_3B_UR50D": "ESM2 (3B)",
    "facebook/esm2_t30_150M_UR50D": "ESM2 (150M)",
}


def run_assay(assay_id, scorer, reference, scheme="masked", batch_size=8):
    df, sequence = data.load_assay(assay_id, reference=reference)
    meta = data.assay_metadata(assay_id, reference=reference)
    variants = df["mutant"].tolist()
    truth = df["DMS_score"].to_numpy()
    truth_bin = df["DMS_score_bin"].to_numpy()

    print(f"\n{assay_id}", flush=True)
    print(f"  {meta['organism']} | {meta['seq_len']} aa | "
          f"{len(variants)} variants | {meta['selection']}", flush=True)

    t0 = time.time()
    if meta["seq_len"] > 1022:
        esm_scores = scorer.score_variants_windowed(
            sequence, variants, batch_size=batch_size)
    else:
        esm_scores = scorer.score_variants(
            sequence, variants, scheme=scheme, batch_size=batch_size)
    elapsed = time.time() - t0

    rows = []
    esm_metrics = evaluate.evaluate(esm_scores, truth, truth_bin)
    rows.append({"assay": assay_id, "method": "esm2", **esm_metrics,
                 "seconds": round(elapsed, 1), **meta})
    print(f"  esm2        rho={esm_metrics['spearman']:+.3f} "
          f"[{esm_metrics['spearman_lo']:+.3f},{esm_metrics['spearman_hi']:+.3f}] "
          f"auc={esm_metrics.get('auc', float('nan')):.3f}  ({elapsed:.0f}s)",
          flush=True)

    for name, fn in baselines.BASELINES.items():
        scores = fn(sequence, variants)
        m = evaluate.evaluate(scores, truth, truth_bin)
        rows.append({"assay": assay_id, "method": name, **m, **meta})
        print(f"  {name:<11} rho={m['spearman']:+.3f}", flush=True)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--assays", nargs="*", default=None)
    ap.add_argument("--scheme", default="masked", choices=["masked", "wt"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--include-hard", action="store_true",
                    help="also run the known-difficult viral assay")
    args = ap.parse_args()

    assays = args.assays or list(data.DEFAULT_ASSAYS)
    if args.include_hard:
        assays += data.HARD_ASSAYS

    reference = data.reference_table()
    data.ensure_assays(assays)

    print(f"loading {args.model}", flush=True)
    scorer = ESMScorer(args.model)
    print(f"  {scorer}", flush=True)

    rows = []
    for assay in assays:
        rows += run_assay(assay, scorer, reference, scheme=args.scheme,
                          batch_size=args.batch_size)

    out = pd.DataFrame(rows)
    tag = args.model.split("/")[-1] + ("" if args.scheme == "masked" else "_wt")
    path = RESULTS / f"benchmark_{tag}.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")

    summarise(out)
    compare_to_official(out, args.model)


def summarise(df):
    print("\n--- mean Spearman across assays ---")
    agg = (df.groupby("method")["spearman"]
             .agg(["mean", "std", "count"])
             .sort_values("mean", ascending=False))
    for method, r in agg.iterrows():
        print(f"  {method:<12} {r['mean']:+.3f}  (sd {r['std']:.3f}, "
              f"n={int(r['count'])})")


def compare_to_official(df, model):
    """The check that matters: does our implementation reproduce ProteinGym's
    published per-assay numbers for the same model?

    A systematic gap here means a bug -- most often an off-by-one in the
    position indexing, or scoring the wrong token because of the BOS offset.
    """
    col = OFFICIAL_COLUMN.get(model)
    if col is None:
        print(f"\n(no official column known for {model}, skipping comparison)")
        return

    official = data.official_scores()
    if col not in official.columns:
        print(f"\n(column {col!r} not in the official results file)")
        return

    ours = df[df["method"] == "esm2"].set_index("assay")["spearman"]
    print(f"\n--- vs official ProteinGym {col} ---")
    print(f"{'assay':<32} {'ours':>7} {'official':>9} {'diff':>7}")

    diffs = []
    for assay, val in ours.items():
        if assay not in official.index:
            continue
        ref = float(official.loc[assay, col])
        diffs.append(val - ref)
        print(f"{assay:<32} {val:+.3f} {ref:+9.3f} {val - ref:+7.3f}")

    if diffs:
        diffs = np.array(diffs)
        print(f"\nmean difference {diffs.mean():+.4f}, "
              f"max absolute {np.abs(diffs).max():.4f}")
        if np.abs(diffs).max() < 0.02:
            print("reproduces the official numbers")
        else:
            print("DOES NOT match -- check position indexing and tokenisation")


if __name__ == "__main__":
    main()
