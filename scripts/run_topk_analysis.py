"""Where the top-10 recall failure comes from: the assay, not the model.

The README section "The top-k problem is not about burial" rules out burial
and leaves three candidates: epistasis, the heterogeneity of what "top 10%"
means, and the measurement noise floor. This script tests the last two with
data already on disk and no model forward pass.

The lever is that ProteinGym contains 24 proteins assayed more than once --
55 assays over 24 UniProt ids. Two assays of the same protein, restricted to
the variants they share, are two measurements. That gives three quantities
that no predictor can beat:

  1. **The cross-assay ceiling.** Score assay A's variants (or simply use A's
     own measurements) as a "predictor" of assay B's top 10%. The best possible
     predictor of an assay is the assay. If the truth-vs-truth top-10 recall is
     0.35, then a model at 0.32 is not failing -- it is at the ceiling.

  2. **The noise floor.** sigma = sd(a - b)/sqrt(2) estimates one
     measurement's error from a pair. Combined with the observed variance of a
     subset, it gives that subset's reliability, `1 - sigma^2/var`. That is the
     attenuation limit on a correlation inside a subset chosen INDEPENDENTLY of
     the noise. For the top decile of the same assay it is not a limit at all:
     selection on a noisy y shrinks the subset's variance, so the formula can be
     negative while the latent value in the subset is still rankable
     (scripts/sim_pooling_controls.py). Read it as "the decile spread compared
     with one measurement's noise", not as "cannot be ranked by anything".

  3. **The decile's shape.** sd(top 10%) / sd(whole assay), and the share of
     the assay's range the decile spans. A decile that spans a third of the
     assay's range but only a tenth of its variance is a plateau.

Every number this script produces lands in results/topk/ and is rendered into
the README by scripts/report_tables.py, so nothing is hand-copied.

Run:
    python scripts/run_topk_analysis.py


Decile spread against measurement error (moved from the README)
----------------------------------------------------------------
Per-measurement error, standardised so two assays on different scales are
comparable, has a median of 0.592 in standard-deviation units. The top decile's
entire internal spread has a median of 0.294. On 29 of the 37 pairs, on both
sides, the whole range of the decile is smaller than one measurement's error,
and the "reliability" of the decile (one minus that error variance over the
decile's own variance) is at or below zero on 32 of the 37. That number compares
the decile's spread with one measurement's error and must not be read as "cannot
be ranked by anything": the decile is selected on the very assay whose error is
being subtracted, and selecting on a noisy measurement shrinks the selected
variances (regression to the mean) while the noise estimate, made on the whole
assay, stays put. scripts/sim_pooling_controls.py reproduces exactly this: at the
real median noise level the formula averages -1.14 over 400 simulated assays
(negative in all 400) while the latent value inside the decile has variance
0.298 in the same units, and the two assays still correlate at +0.323 inside it.
A decile that is hard to rank is a fair description of the real data; "zero
reliability, so unrankable" is not. The Jaccard overlap of two assays' top
deciles is 0.161: the sets barely coincide even where the rankings agree at 0.61
over the whole assay.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_effect import assay_stats, data, evaluate, quantity  # noqa: E402

OUT = ROOT / "results" / "topk"

# Pairs whose two assays measure deliberately different quantities -- a gof
# and a lof assay of the same protein, or expression against activity -- can
# disagree for reasons that are biology rather than error, and they are the
# pairs most likely to understate the ceiling. They are kept: they make the
# ceiling a lower bound, which is the conservative direction for the claim
# that the model is already at it. Flagged so the reading is explicit.
# The list itself lives in variant_effect.quantity, where pooled_truth callers can
# consult it; this name is kept so existing imports and the README renderer work.
DIFFERENT_QUANTITY = quantity.DIFFERENT_QUANTITY

MIN_SHARED = 200


def _scores(assay_id):
    """DMS_score per mutant for one assay, or None if it will not load."""
    try:
        return data.canonical_scores(assay_id)
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError):
        return None


def replicate_pairs(reference):
    """Every same-protein assay pair whose shared variant set is large enough
    to measure anything."""
    by_protein = reference.groupby("UniProt_ID")["DMS_id"].apply(list)
    pairs = []
    for uniprot, ids in by_protein.items():
        if len(ids) < 2:
            continue
        loaded = {i: _scores(i) for i in sorted(ids)}
        loaded = {k: v for k, v in loaded.items() if v is not None}
        for a, b in itertools.combinations(sorted(loaded), 2):
            shared = loaded[a].index.intersection(loaded[b].index)
            if len(shared) < MIN_SHARED:
                continue
            pairs.append({
                "uniprot": uniprot,
                "a": a,
                "b": b,
                "n_shared": int(len(shared)),
                "different_quantity": frozenset((a, b)) in DIFFERENT_QUANTITY,
                "sa": loaded[a].loc[shared].to_numpy(dtype=float),
                "sb": loaded[b].loc[shared].to_numpy(dtype=float),
            })
    return pairs


def ceiling_table(pairs):
    """The truth-vs-truth numbers: how well one assay predicts another's
    top 10%, how much they agree inside the top decile, and how much of the
    assay's total variance is real signal at the top."""
    rows = []
    for p in pairs:
        a, b = p["sa"], p["sb"]
        sigma = assay_stats.noise_sd_z(a, b)
        whole = assay_stats.paired_ranks_agreement(a, b)
        # take each assay as the truth in turn, the other as the predictor
        rec_ab = evaluate.top_k_recall(b, a)
        rec_ba = evaluate.top_k_recall(a, b)
        za, zb = assay_stats.zscore(a), assay_stats.zscore(b)
        ma, mb = assay_stats.top_decile_mask(a), assay_stats.top_decile_mask(b)
        top_rho = (assay_stats.paired_ranks_agreement(a[ma], b[ma])
                   if ma.sum() >= 20 else np.nan)
        top_rho_b = (assay_stats.paired_ranks_agreement(a[mb], b[mb])
                     if mb.sum() >= 20 else np.nan)
        top_spread_a = (float(np.nanstd(za[ma], ddof=1)) if ma.sum() >= 20
                        else np.nan)
        top_spread_b = (float(np.nanstd(zb[mb], ddof=1)) if mb.sum() >= 20
                        else np.nan)
        rows.append({
            "uniprot": p["uniprot"],
            "assay_a": p["a"],
            "assay_b": p["b"],
            "n_shared": p["n_shared"],
            "different_quantity": p["different_quantity"],
            "spearman_whole": whole,
            "spearman_a_top_decile": top_rho,
            "spearman_b_top_decile": top_rho_b,
            "top10_recall_a_to_b": rec_ab,
            "top10_recall_b_to_a": rec_ba,
            "top10_recall_mean": np.nanmean([rec_ab, rec_ba]),
            "jaccard_top_decile": assay_stats.top_decile_overlap(a, b),
            "noise_sd_z": sigma,
            "sd_a": float(np.nanstd(a, ddof=1)),
            "sd_b": float(np.nanstd(b, ddof=1)),
            "reliability_whole": assay_stats.reliability_z(za, sigma),
            "reliability_top_decile": assay_stats.reliability_z(za[ma], sigma),
            "reliability_top_decile_b": assay_stats.reliability_z(zb[mb], sigma),
            "top_decile_sd_a": top_spread_a,
            "top_decile_sd_b": top_spread_b,
            # the direct statement: is the decile's whole internal spread
            # smaller than one measurement's error?
            "noise_exceeds_top_spread_a": bool(sigma > top_spread_a),
            "noise_exceeds_top_spread_b": bool(sigma > top_spread_b),
            "noise_over_top_spread": (sigma / top_spread_a
                                      if top_spread_a and np.isfinite(top_spread_a)
                                      else np.nan),
        })
    return pd.DataFrame(rows)


def assay_shape(reference, per_assay):
    """Per-assay decile shape, joined to the model's top-10 recall.

    Tests the "heterogeneity of the top 10%" candidate directly: if the model
    does badly on assays whose top decile is a plateau, then spread_ratio or
    range_share should predict recall across the 217 assays. Assays with no
    replicate partner are included here -- the shape does not need one.
    """
    rows = []
    for assay_id in per_assay.index:
        s = _scores(assay_id)
        if s is None:
            continue
        shape = assay_stats.decile_shape(s.to_numpy(dtype=float))
        if not shape:
            continue
        meta = reference[reference["DMS_id"] == assay_id]
        rows.append({
            "assay": assay_id,
            "uniprot": meta["UniProt_ID"].iloc[0] if len(meta) else "",
            "taxon": meta["taxon"].iloc[0] if len(meta) else "",
            "selection": meta["selection_type"].iloc[0] if len(meta) else "",
            **shape,
            "model_spearman": per_assay.loc[assay_id, "spearman"],
            "model_top10_recall": per_assay.loc[assay_id, "top10_recall"],
            "has_replicate": False,
        })
    df = pd.DataFrame(rows)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference = data.reference_table()

    per_assay = pd.read_csv(
        ROOT / "results" / "full" / "esm2_t33_650M_UR50D_per_assay.csv")
    esm = per_assay[per_assay["method"] == "esm2"].set_index("assay")

    print("finding replicate assay pairs ...", flush=True)
    pairs = replicate_pairs(reference)
    print(f"  {len(pairs)} pairs over "
          f"{len({p['uniprot'] for p in pairs})} proteins", flush=True)

    ceil = ceiling_table(pairs)
    ceil.to_csv(OUT / "cross_assay_ceiling.csv", index=False)

    shape = assay_shape(reference, esm)
    shape["has_replicate"] = shape["assay"].isin(
        set(ceil["assay_a"]) | set(ceil["assay_b"]))
    shape.to_csv(OUT / "assay_shape.csv", index=False)

    print(f"\ncross-assay ceiling over {len(ceil)} pairs")
    print(f"  median top-10 recall (truth vs truth): "
          f"{ceil['top10_recall_mean'].median():.3f}")
    print(f"  median Spearman over the whole shared set: "
          f"{ceil['spearman_whole'].median():.3f}")
    print(f"  median top-decile reliability: "
          f"{ceil['reliability_top_decile'].median():.3f}")
    both = ceil["noise_exceeds_top_spread_a"] & ceil["noise_exceeds_top_spread_b"]
    print(f"  median noise sd, z units: {ceil['noise_sd_z'].median():.3f}")
    print(f"  median top-decile sd, z units: "
          f"{ceil['top_decile_sd_a'].median():.3f}")
    print(f"  pairs where the top decile's whole internal spread is smaller "
          f"than one measurement's error, on both sides: "
          f"{int(both.sum())}/{len(ceil)}")
    print(f"\nmodel mean top-10 recall over 217 assays: "
          f"{esm['top10_recall'].mean():.3f}")

    # Does the *shape* of the decile predict the model's recall? Tests the
    # "top 10% is heterogeneous" candidate across all 217 assays, including
    # the 195 with no replicate partner.
    print("\nshape of the top decile vs the model's recall, 217 assays")
    shape_tests = []
    for col in ("spread_ratio", "range_share", "distinct_frac", "at_max_frac"):
        m = shape[[col, "model_top10_recall"]].dropna()
        if len(m) < 20:
            continue
        r, pv = stats.spearmanr(m[col], m["model_top10_recall"])
        shape_tests.append({"predictor": col, "n_assays": len(m),
                            "spearman": float(r), "p": float(pv)})
        print(f"  {col:<14} rho={r:+.3f} p={pv:.3g} n={len(m)}")
    pd.DataFrame(shape_tests).to_csv(OUT / "shape_vs_recall.csv", index=False)

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
