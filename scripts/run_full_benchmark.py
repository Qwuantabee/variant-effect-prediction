"""The full ProteinGym substitution benchmark: all 217 assays, all mutants.

Why this exists next to ``run_benchmark.py``: the nine-assay subset was chosen
to be small and fast, and it is not a sample of the benchmark -- it is eight
small single-mutant assays plus one viral failure case. Its mean Spearman is
therefore not comparable to any published number. This script scores the whole
benchmark and aggregates it the way ProteinGym does, so the headline can be
compared to theirs directly.

Run:
    HF_HUB_OFFLINE=1 python scripts/run_full_benchmark.py
    HF_HUB_OFFLINE=1 python scripts/run_full_benchmark.py --assays BLAT_ECOLX_Stiffler_2015
    python scripts/run_full_benchmark.py --baselines-only

Resumable: each assay's result is appended to results/full/<model>.jsonl as
soon as it finishes, so an interrupted run does not start over. Delete that
file to force a clean run.

Outputs:
    results/full/<model>.jsonl          per-assay rows, one JSON object per line
    results/full/<model>_per_assay.csv  same, as a table
    results/full/<model>_summary.csv    aggregation + the official comparison
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

from variant_effect import aggregate, baselines, data, evaluate  # noqa: E402

RESULTS = ROOT / "results" / "full"
RESULTS.mkdir(parents=True, exist_ok=True)

OFFICIAL_COLUMN = {
    "facebook/esm2_t33_650M_UR50D": "ESM2 (650M)",
    "facebook/esm2_t36_3B_UR50D": "ESM2 (3B)",
    "facebook/esm2_t30_150M_UR50D": "ESM2 (150M)",
    "facebook/esm2_t12_35M_UR50D": "ESM2 (35M)",
    "facebook/esm2_t6_8M_UR50D": "ESM2 (8M)",
}


def assay_rows(assay_id, reference, scorer=None, baselines_map=None,
               batch_size=32):
    """Score one assay with the model and with each cheap baseline.

    Every mutant is scored, not just single substitutions: the published
    per-assay values are computed over the full assay. On A4_HUMAN_Seuma_2022
    the model scores 0.431 over all 14811 mutants but 0.498 over the 796
    singles, and ProteinGym reports 0.431 -- so restricting to singles would
    disagree with the official table by 0.07 on that assay alone.
    """
    df = pd.read_csv(data.ASSAY_DIR / f"{assay_id}.csv")
    row = reference[reference["DMS_id"] == assay_id].iloc[0]
    sequence = row["target_seq"]
    variants = df["mutant"].astype(str).tolist()
    truth = df["DMS_score"].to_numpy(dtype=float)
    truth_bin = (df["DMS_score_bin"].to_numpy(dtype=float)
                 if "DMS_score_bin" in df.columns else None)

    meta = {
        "assay": assay_id,
        "uniprot": row["UniProt_ID"],
        "taxon": row["taxon"],
        "selection": row["coarse_selection_type"],
        "msa_depth": row["MSA_Neff_L_category"],
        "seq_len": int(row["seq_len"]),
        "n_variants": len(df),
        "n_multi": int(df["mutant"].astype(str).str.contains(":").sum()),
    }

    rows = []
    if scorer is not None:
        t0 = time.time()
        if meta["seq_len"] > 1022:
            scores = scorer.score_variants_windowed(
                sequence, variants, batch_size=batch_size)
        else:
            scores = scorer.score_variants(
                sequence, variants, scheme="masked", batch_size=batch_size)
        elapsed = time.time() - t0
        m = evaluate.evaluate(scores, truth, truth_bin)
        rows.append({**meta, "method": "esm2", "seconds": round(elapsed, 1),
                     **m})
        print(f"  esm2        rho={m['spearman']:+.4f}  ({elapsed:.0f}s)",
              flush=True)

    for name, fn in (baselines_map or {}).items():
        t0 = time.time()
        scores = fn(sequence, variants)
        m = evaluate.evaluate(scores, truth, truth_bin)
        rows.append({**meta, "method": name, "seconds": round(time.time() - t0, 1),
                     **m})
        print(f"  {name:<11} rho={m['spearman']:+.4f}", flush=True)

    return rows


def read_jsonl(path):
    if not Path(path).exists():
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # a killed run can leave a partial final line
                    continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--assays", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop cleanly after this many seconds (resumable)")
    args = ap.parse_args()

    reference = data.reference_table()
    official = data.official_scores()
    assays = args.assays or reference["DMS_id"].tolist()

    tag = args.model.split("/")[-1]
    jsonl = RESULTS / f"{tag}.jsonl"

    done = {(r["assay"], r["method"]) for r in read_jsonl(jsonl)}
    if done:
        print(f"resuming: {len(done)} (assay, method) rows already present",
              flush=True)

    scorer = None
    if not args.baselines_only:
        from variant_effect.scoring import ESMScorer
        print(f"loading {args.model}", flush=True)
        scorer = ESMScorer(args.model)
        print(f"  {scorer}", flush=True)

    bmap = {} if args.no_baselines else dict(baselines.BASELINES)

    t_start = time.time()
    n_new = 0
    for i, assay in enumerate(assays, 1):
        wanted = set()
        if scorer is not None:
            wanted.add("esm2")
        wanted |= set(bmap)
        if wanted <= {m for (a, m) in done if a == assay}:
            continue

        if (args.max_seconds is not None
                and time.time() - t_start > args.max_seconds):
            print(f"\nstopping after {n_new} assays (--max-seconds); "
                  f"rerun to continue", flush=True)
            break

        print(f"\n[{i}/{len(assays)}] {assay}", flush=True)
        try:
            rows = assay_rows(assay, reference, scorer=scorer,
                              baselines_map=bmap, batch_size=args.batch_size)
        except Exception as exc:                      # keep the sweep alive
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            rows = [{**{k: None for k in
                        ("uniprot", "taxon", "selection", "msa_depth",
                         "seq_len", "n_variants", "n_multi")},
                     "assay": assay, "method": "esm2", "spearman": np.nan,
                     "error": f"{type(exc).__name__}: {exc}"}]

        with open(jsonl, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=float) + "\n")
        n_new += 1

    rows = read_jsonl(jsonl)
    if not rows:
        print("no results")
        return
    table = pd.DataFrame(rows)
    per_assay = RESULTS / f"{tag}_per_assay.csv"
    table.to_csv(per_assay, index=False)
    print(f"\nwrote {per_assay} ({len(table)} rows)")

    report(table, tag, official)


def report(table, tag, official):
    """Aggregate every method and compare the model to the published table."""
    out = []
    print("\n--- aggregation ---")
    for method in table["method"].dropna().unique():
        sub = table[table["method"] == method]
        s = aggregate.summarise(sub)
        out.append({"method": method, **s})
        print(f"  {method:<11} headline {s['headline']:.4f}  "
              f"(per-assay mean {s['mean_per_assay']:.4f}, "
              f"per-UniProt {s['mean_per_uniprot']:.4f}, "
              f"n={s['n_assays']} assays / {s['n_uniprot']} proteins)")

    model_tag = next((m for m in table["method"].dropna().unique()
                      if m == "esm2"), None)
    col = OFFICIAL_COLUMN.get("facebook/" + tag)
    if model_tag and col and col in official.columns:
        sub = table[table["method"] == model_tag].set_index("assay")
        ours = sub["spearman"].astype(float)
        both = ours.index.intersection(official.index)
        diff = ours[both] - official.loc[both, col]
        print(f"\n--- per-assay check against ProteinGym {col} ---")
        print(f"  assays compared      {len(both)}")
        print(f"  mean difference      {diff.mean():+.4f}")
        print(f"  median abs difference {diff.abs().median():.4f}")
        print(f"  max abs difference   {diff.abs().max():.4f}")
        print(f"  assays within 0.01   {int((diff.abs() < 0.01).sum())}/{len(both)}")

        head_ours = aggregate.macro_over_categories(
            table[table["method"] == model_tag])
        off_table = pd.DataFrame({
            "spearman": official[col].values,
            "selection": official["Selection Type"].values,
            "uniprot": official["UniProt ID"].values,
        })
        head_off = aggregate.macro_over_categories(off_table)
        print(f"\n  headline ours {head_ours:.4f}  official {head_off:.4f}  "
              f"diff {head_ours - head_off:+.4f}")

    pd.DataFrame(out).to_csv(RESULTS / f"{tag}_summary.csv", index=False)


if __name__ == "__main__":
    main()
