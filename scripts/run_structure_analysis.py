"""Does the top-k failure come from treating buried and exposed positions alike?

The README's fourth open item. The suspicion is that the respectable Spearman
and the poor top-10 recall are two views of the same weakness: a buried residue
is packed against the core, so nearly every substitution there is damaging and
the ranking among those variants is compressed; an exposed residue trades with
solvent, so more substitutions are tolerated and the ordering carries more
information. If the model cannot tell which regime a position is in, it will
rank well overall and pick badly at the top.

That hypothesis makes three predictions, each testable with scores already
computed plus structures ProteinGym distributes:

  P1  Spearman on buried-only variants is lower than on exposed-only variants.
  P2  Top-k recall is worse on buried-only subsets than exposed-only ones.
  P3  The model's top-ranked variants are biased toward exposed positions.

If P1-P3 hold, burial is a lever worth pulling. If they do not, the top-k
failure has a different cause and adding structure would be wasted work -- so
this script reports the answer either way rather than assuming it.

Then the constructive half: combine ESM-2 with structure and see whether the
combination beats either alone. Two named schemes, both simple enough to state:

  burial-aware   ESM score shrunk toward the buried/unexposed prior:
                 score - lambda * max(0, 0.25 - rsa), i.e. penalise mutations
                 at buried positions, with lambda fitted on a *training* split
                 of assays and evaluated on the held-out ones.
  structure-only negative relative accessibility of the mutated position.

lambda is fitted on half the assays and scored on the other half. Fitting on
everything would make the improvement meaningless.

Run:
    python scripts/run_structure_analysis.py
    python scripts/run_structure_analysis.py --assays BLAT_ECOLX_Stiffler_2015
    python scripts/run_structure_analysis.py --taxa Virus Human

Writes results/structure/<tag>.csv (per assay) and burial_groups.csv.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_effect import baselines, data, evaluate, structure  # noqa: E402

RESULTS = ROOT / "results" / "structure"
RESULTS.mkdir(parents=True, exist_ok=True)


def assay_frame(aid, reference, scorer=None):
    """Per-variant frame: truth, ESM score, burial, class."""
    row = reference[reference["DMS_id"] == aid].iloc[0]
    seq = row["target_seq"]
    df = pd.read_csv(data.ASSAY_DIR / f"{aid}.csv")

    try:
        pdb = data.structure_path(aid, reference=reference)
        rsa = structure.relative_sasa(pdb, seq, row["pdb_range"])
    except Exception as exc:
        print(f"  structure unavailable: {type(exc).__name__}: {exc}",
              flush=True)
        rsa = np.full(len(seq), np.nan)

    variants = df["mutant"].astype(str).tolist()
    truth = df["DMS_score"].to_numpy(dtype=float)

    out = pd.DataFrame({
        "assay": aid,
        "variant": variants,
        "truth": truth,
        "rsa": structure.position_burial(variants, rsa),
        "n_atoms_modelled": int(np.isfinite(rsa).sum()),
    })
    out["burial"] = structure.burial_classes(
        np.where(np.isfinite(out["rsa"]), out["rsa"], np.nan))
    # position_burial returns NaN for multi-mutants; classify those separately
    out.loc[out["rsa"].isna(), "burial"] = "multi_or_unmodelled"

    if scorer is not None:
        seq_len = int(row["seq_len"])
        if seq_len > 1022:
            out["esm"] = scorer.score_variants_windowed(
                seq, variants, batch_size=32)
        else:
            out["esm"] = scorer.score_variants(
                seq, variants, batch_size=32)
    else:
        out["esm"] = np.nan

    out["blosum62"] = baselines.blosum62_score(seq, variants)
    out["structure_only"] = -out["rsa"]
    return out


def group_stats(frame, method, min_n=20):
    """Spearman and top-10 recall per burial class."""
    rows = []
    for cls in ("buried", "intermediate", "exposed"):
        sub = frame[(frame["burial"] == cls) & frame[method].notna()
                    & frame["truth"].notna()]
        if len(sub) < min_n:
            continue
        rows.append({
            "class": cls, "n": len(sub),
            "spearman": evaluate.spearman(sub[method].to_numpy(),
                                          sub["truth"].to_numpy()),
            "top10_recall": evaluate.top_k_recall(sub[method].to_numpy(),
                                                  sub["truth"].to_numpy()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assays", nargs="*", default=None)
    ap.add_argument("--taxa", nargs="*", default=None)
    ap.add_argument("--max-assays", type=int, default=None)
    ap.add_argument("--tag", default="structure")
    ap.add_argument("--no-model", action="store_true")
    args = ap.parse_args()

    ref = data.reference_table()
    if args.assays:
        ids = args.assays
    elif args.taxa:
        ids = ref[ref["taxon"].isin(args.taxa)]["DMS_id"].tolist()
    else:
        ids = ref["DMS_id"].tolist()
    if args.max_assays:
        ids = ids[:args.max_assays]

    scorer = None
    if not args.no_model:
        from variant_effect.scoring import ESMScorer
        print("loading model", flush=True)
        scorer = ESMScorer()

    frames = []
    t0 = time.time()
    for i, aid in enumerate(ids, 1):
        try:
            f = assay_frame(aid, ref, scorer=scorer)
            frames.append(f)
            n_cov = int(np.isfinite(f["rsa"]).mean() * 100)
            print(f"[{i}/{len(ids)}] {aid:<36} rsa coverage {n_cov:3d}%  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        except Exception as exc:
            print(f"[{i}/{len(ids)}] {aid:<36} FAILED {type(exc).__name__}: "
                  f"{exc}", flush=True)

    if not frames:
        print("no assays scored")
        return
    allv = pd.concat(frames, ignore_index=True)
    allv.to_parquet(RESULTS / f"{args.tag}_variants.parquet") \
        if False else allv.to_csv(RESULTS / f"{args.tag}_variants.csv",
                                  index=False)

    # ---- aggregate per assay, pooled across assays -----------------------
    print("\n--- pooled by burial class (all assays, per-variant) ---")
    pooled = []
    for method in ("esm", "blosum62", "structure_only"):
        if allv[method].notna().sum() == 0:
            continue
        for r in group_stats(allv, method):
            pooled.append({"method": method, **r})
    pooled = pd.DataFrame(pooled)
    print(pooled.round(3).to_string(index=False))
    pooled.to_csv(RESULTS / "burial_groups.csv", index=False)

    # ---- ranking of top picks by burial ----------------------------------
    if allv["esm"].notna().any():
        print("\n--- are the model's top picks biased toward exposed positions? ---")
        rows = []
        for aid, g in allv.groupby("assay"):
            g = g.dropna(subset=["esm", "rsa"])
            if len(g) < 50:
                continue
            top = g.nlargest(10, "esm")
            bottom = g.nsmallest(10, "esm")
            rows.append({
                "assay": aid,
                "mean_rsa_all": g["rsa"].mean(),
                "mean_rsa_top10": top["rsa"].mean(),
                "mean_rsa_bottom10": bottom["rsa"].mean(),
            })
        rk = pd.DataFrame(rows)
        rk["top_minus_all"] = rk["mean_rsa_top10"] - rk["mean_rsa_all"]
        print(f"  {len(rk)} assays with both scores and structure")
        print(f"  mean RSA of top-10 picks minus assay mean: "
              f"{rk['top_minus_all'].mean():+.4f}")
        print(f"  assays where top picks are more exposed: "
              f"{int((rk['top_minus_all'] > 0).sum())}/{len(rk)}")
        from scipy import stats
        t, p = stats.ttest_1samp(rk["top_minus_all"].dropna(), 0)
        print(f"  one-sample t-test vs 0: t={t:.2f}, p={p:.3g}")
        rk.to_csv(RESULTS / f"{args.tag}_topk_rsa.csv", index=False)

    # ---- does adding structure help, on held-out assays? -----------------
    if allv["esm"].notna().any():
        print("\n--- burial-aware rescoring, lambda fitted on half the assays ---")
        assays = sorted(allv["assay"].unique())
        rng = np.random.default_rng(0)
        rng.shuffle(assays)
        half = len(assays) // 2
        train = set(assays[:half])
        test = set(assays[half:])

        def score_lambda(lam, subset):
            rows = []
            for aid, g in allv[allv["assay"].isin(subset)].groupby("assay"):
                g = g.dropna(subset=["esm", "rsa", "truth"])
                if len(g) < 50:
                    continue
                adj = (g["esm"].to_numpy()
                       - lam * np.clip(0.25 - g["rsa"].to_numpy(), 0, None))
                rows.append({
                    "assay": aid,
                    "plain": evaluate.spearman(g["esm"].to_numpy(),
                                               g["truth"].to_numpy()),
                    "adjusted": evaluate.spearman(adj, g["truth"].to_numpy()),
                })
            return pd.DataFrame(rows)

        grid = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
        fit = [(lam, score_lambda(lam, train)) for lam in grid]
        best_lam, best = max(fit, key=lambda x: x[1]["plain"].mean()
                             if len(x[1]) else -9)
        # choose lambda by the *adjusted* score on the training half
        best_lam = max(fit, key=lambda x: x[1]["adjusted"].mean()
                       if len(x[1]) else -9)[0]
        print(f"  lambda grid {grid}, chosen on training half: {best_lam}")
        held = score_lambda(best_lam, test)
        if len(held):
            print(f"  held-out assays: {len(held)}")
            print(f"    plain ESM      {held['plain'].mean():+.4f}")
            print(f"    burial-aware   {held['adjusted'].mean():+.4f}")
            print(f"    change         "
                  f"{(held['adjusted'] - held['plain']).mean():+.4f}")
            from scipy import stats
            d = held["adjusted"] - held["plain"]
            if len(d) >= 3:
                t, p = stats.ttest_rel(held["adjusted"], held["plain"])
                print(f"    paired t={t:.2f}, p={p:.3g}, "
                      f"{int((d > 0).sum())}W/{int((d < 0).sum())}L")
            held.to_csv(RESULTS / f"{args.tag}_heldout.csv", index=False)


if __name__ == "__main__":
    main()
