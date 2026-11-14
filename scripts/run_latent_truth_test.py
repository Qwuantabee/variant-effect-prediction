"""Is 0.350 a ceiling, or just two noisy assays agreeing?

The README's top-k section says the model is at the ceiling because
truth-vs-truth top-10 recall between two replicate assays is 0.350 against the
model's 0.323. An audit of that paragraph raised two objections and both are
fair:

  1. 0.350 is a *median over 37 replicate pairs*; 0.323 is a *mean over 217
     assays*. A median and a mean of different populations are not a
     comparison, however close the two numbers happen to land.

  2. More importantly, *another noisy assay is not an upper bound*. Two
     assays of one protein are two measurements of the same latent quantity,
     each carrying its own error. Their agreement is attenuated by that error,
     so a predictor that tracked the latent truth closely could beat their
     mutual agreement without contradicting anything. "The best predictor is
     the assay" is true only when the assay is noiseless.

The test
--------

Averaging two noisy measurements reduces the noise. For a pair of assays A and
B on the variants they share, with each standardised to zero mean and unit
spread over that shared set,

    pooled = (z_A + z_B) / 2

has half the measurement variance of either assay *if* the two errors are
independent and of equal variance -- the same assumption the existing noise
estimator ``sd(a - b)/sqrt(2)`` already makes. Pooled is therefore a closer
estimate of the latent truth than either assay alone, and it is the better
target to score a predictor against.

For every pair with at least 1000 shared variants this script reports, on the
same shared variants:

  * top-10 recall and Spearman of assay A, assay B, and the pooled estimate
    against the pooled target;
  * the same two metrics for the ESM-2 model score;
  * split-half reliability, ``spearman(z_A, z_B)``, the share of each assay's
    ranking that the other reproduces.

The key comparison is *paired*: the model's recall against the pooled target
minus its recall against a single assay, on the same variants of the same
pair, with a Wilcoxon signed-rank test over pairs. No median is compared to a
mean anywhere.

What the pooled number is and is not
-----------------------------------

Pooling removes half the *independent* measurement variance. It does not
remove the other half, and it cannot remove error the two assays share (a
common reporter, a common growth-rate readout, the same variant library
synthesised once). Two consequences, and they point in opposite directions:

  * pooled is still a noisy target, so any recall measured against it is a
    *lower* bound on the recall the same predictor would show against the
    true latent quantity. The real headroom is at least what is measured
    here, not at most;
  * pooled is *not* a platonic truth either. Where the two assays measure
    deliberately different quantities (a gain-of-function against a
    loss-of-function assay, expression against activity) their average is a
    pseudo-quantity that no cell ever computed, so a recall against it can be
    optimistic in the sense of measuring something that is not biology.

The pool is reported for all pairs and separately for the pairs that are the
same assay condition, so the second caveat is visible rather than argued.

One further asymmetry matters for reading the table. Assay A is not an
independent "predictor" of the pooled target: A is half of it, so A-vs-pooled
recall is inflated by A's own noise component and is a diagnostic, not a
benchmark. The model score is independent of both assays, which is why the
model row is the one that answers the question.

Run:
    python scripts/run_latent_truth_test.py --scores-only   # fill the cache
    python scripts/run_latent_truth_test.py                 # analyse
    python scripts/run_latent_truth_test.py --score-budget 300
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_effect import data, evaluate, pooled_truth  # noqa: E402
from variant_effect.mutations import parse_variant  # noqa: E402

OUT = ROOT / "results" / "latent_truth"
CEILING = ROOT / "results" / "topk" / "cross_assay_ceiling.csv"
PER_ASSAY = ROOT / "results" / "full" / "esm2_t33_650M_UR50D_per_assay.csv"
CACHE = OUT / "_cache"

# 1000 shared variants is the threshold the audit asked for: below it the
# top-10 recall of a single sample is dominated by which ten variants the
# assay happened to resolve, not by the assay's resolvable ordering.
MIN_SHARED = 1000
TOP_FRAC = 0.1
K = 10

# A pair whose split-half reliability is at or below this is flagged: the two
# assays reproduce so little of each other's ordering that their average is
# dominated by whatever they do not share, and pooling it is not informative
# about a latent truth -- it is a third measurement of the disagreement.
LOW_RELIABILITY = 0.3


# ----------------------------------------------------------------------
# assay and model scores
# ----------------------------------------------------------------------

def _scores(assay_id):
    """DMS_score per variant, keyed by the canonical mutant string (see
    data.canonical_scores): a raw-string key drops doubles that two labs wrote
    in different substitution orders."""
    return data.canonical_scores(assay_id)


def pair_shared(pairs, min_shared=MIN_SHARED):
    """Pairs, their shared variants, and both measurements on those variants."""
    out = []
    for _, r in pairs.iterrows():
        if r["n_shared"] < min_shared:
            continue
        a = _scores(r["assay_a"])
        b = _scores(r["assay_b"])
        shared = a.index.intersection(b.index)
        if len(shared) < min_shared:
            continue
        out.append({
            "uniprot": r["uniprot"],
            "assay_a": r["assay_a"],
            "assay_b": r["assay_b"],
            "different_quantity": bool(r["different_quantity"]),
            "shared": list(shared),
            "sa": a.loc[shared].to_numpy(dtype=float),
            "sb": b.loc[shared].to_numpy(dtype=float),
        })
    return out


def mutated_positions(variants):
    pos = set()
    for m in variants:
        for s in parse_variant(str(m)):
            pos.add(s.idx)
    return np.array(sorted(pos), dtype=np.int64)


_REF = None


def sequence_for(protein):
    """The wildtype sequence, from any assay row of that UniProt id."""
    global _REF
    if _REF is None:
        _REF = data.reference_table()
    row = _REF[_REF["UniProt_ID"] == protein]
    if row.empty:
        raise KeyError(f"no reference row for {protein}")
    return row.iloc[0]["target_seq"]


def protein_tables(proteins, pairs, budget_seconds=None, threads=8):
    """Masked-marginal log-probability tables, one per protein, cached.

    One table per *protein* rather than one per assay: the three BLAT assays
    and the six P53 assays all need the same positions, and a per-assay table
    would run them three and six times over.

    ``budget_seconds`` stops the run cleanly between proteins, so a long
    fill can be split across several commands and resumed -- a table already
    on disk is never recomputed.
    """
    import torch
    from variant_effect.scoring import ESMScorer

    torch.set_num_threads(threads)
    CACHE.mkdir(parents=True, exist_ok=True)
    scorer = None
    started = time.time()
    sequences = {}

    for protein in proteins:
        path = CACHE / f"{protein}.npz"
        pos_needed = set()
        for p in pairs:
            if p["uniprot"] == protein:
                pos_needed |= set(mutated_positions(p["shared"]).tolist())
        pos_needed = np.array(sorted(pos_needed), dtype=np.int64)

        have = {}
        if path.exists():
            with np.load(path) as z:
                have = {int(p): z["table"][i]
                        for i, p in enumerate(z["positions"])}
        todo = [int(p) for p in pos_needed if int(p) not in have]
        if not todo:
            print(f"  {protein}: cached ({len(pos_needed)} positions)",
                  flush=True)
            continue
        if (budget_seconds is not None
                and time.time() - started > budget_seconds):
            print(f"  budget reached before {protein}; rerun to resume",
                  flush=True)
            return False

        if scorer is None:
            print("loading ESM-2 ...", flush=True)
            scorer = ESMScorer(device="cpu")
        seq = sequence_for(protein)
        print(f"  {protein}: {len(todo)} of {len(pos_needed)} positions "
              f"(len {len(seq)}) ...", flush=True)
        t0 = time.time()
        for start in range(0, len(todo), 32):
            block = todo[start:start + 32]
            tab = scorer.masked_marginal_logprobs(seq, block, batch_size=8)
            for p in block:
                have[p] = tab[p]
            saved = np.array(sorted(have), dtype=np.int64)
            np.savez(path, positions=saved,
                     table=np.stack([have[int(p)] for p in saved]))
            print(f"    {min(start + 32, len(todo))}/{len(todo)} "
                  f"({(time.time() - t0) / (start + len(block)):.2f} s/pos)",
                  flush=True)
    return True


def model_scores(protein, variants, tables):
    """Independent-site ESM-2 score: sum of per-site masked log-odds."""
    table = tables[protein]
    aa = "ACDEFGHIKLMNPQRSTVWY"
    order = {a: i for i, a in enumerate(aa)}
    out = np.zeros(len(variants), dtype=float)
    for i, m in enumerate(variants):
        total = 0.0
        for s in parse_variant(str(m)):
            row = table[s.idx]
            total += row[order[s.mut]] - row[order[s.wt]]
        out[i] = total
    return out


def load_tables(proteins, pairs):
    """Tables and sequences for every protein, from the cache on disk."""
    sequences, tables = {}, {}
    missing = []
    for protein in proteins:
        sequences[protein] = sequence_for(protein)
        path = CACHE / f"{protein}.npz"
        need = set()
        for p in pairs:
            if p["uniprot"] == protein:
                need |= set(mutated_positions(p["shared"]).tolist())
        if not path.exists():
            missing.append(protein)
            continue
        with np.load(path) as z:
            tab = {int(p): z["table"][i]
                   for i, p in enumerate(z["positions"])}
        if not need.issubset(tab):
            missing.append(protein)
            continue
        tables[protein] = tab
    return sequences, tables, missing


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------

def analyse(pairs, tables):
    """Per-pair metrics against three targets: A, B, and the pooled estimate."""
    rows = []
    for p in pairs:
        sa, sb = p["sa"], p["sb"]
        za, zb = pooled_truth.zscore(sa), pooled_truth.zscore(sb)
        pooled = pooled_truth.pool_z(sa, sb)
        model = model_scores(p["uniprot"], p["shared"], tables)

        rel = pooled_truth.split_half_reliability(sa, sb)
        # recall of each *assay* against the pooled target is a diagnostic,
        # not a benchmark: A and B are each half of the target.
        rec_a_pool = evaluate.top_k_recall(za, pooled, k=K, top_frac=TOP_FRAC)
        rec_b_pool = evaluate.top_k_recall(zb, pooled, k=K, top_frac=TOP_FRAC)
        # the old ceiling, unchanged in definition: A's top decile scored by B
        rec_ab = evaluate.top_k_recall(za, sb, k=K, top_frac=TOP_FRAC)
        rec_ba = evaluate.top_k_recall(zb, sa, k=K, top_frac=TOP_FRAC)
        # the model against each target, always on the same shared variants
        rec_m_pool = evaluate.top_k_recall(model, pooled, k=K, top_frac=TOP_FRAC)
        rec_m_a = evaluate.top_k_recall(model, sa, k=K, top_frac=TOP_FRAC)
        rec_m_b = evaluate.top_k_recall(model, sb, k=K, top_frac=TOP_FRAC)

        rows.append({
            "uniprot": p["uniprot"],
            "assay_a": p["assay_a"],
            "assay_b": p["assay_b"],
            "different_quantity": p["different_quantity"],
            "n_shared": len(p["shared"]),
            "split_half_reliability": rel,
            "low_reliability": bool(np.isfinite(rel) and rel <= LOW_RELIABILITY),
            # truth-vs-truth, the number 0.350 came from
            "recall_a_to_b": rec_ab,
            "recall_b_to_a": rec_ba,
            "recall_truth_vs_truth": np.nanmean([rec_ab, rec_ba]),
            # assay against the pooled target (diagnostic; inflated by
            # A's or B's own noise component)
            "recall_a_to_pooled": rec_a_pool,
            "recall_b_to_pooled": rec_b_pool,
            # the model, against a single assay and against the pool
            "recall_model_to_a": rec_m_a,
            "recall_model_to_b": rec_m_b,
            "recall_model_to_single": np.nanmean([rec_m_a, rec_m_b]),
            "recall_model_to_pooled": rec_m_pool,
            "recall_gain_pooled_vs_single": rec_m_pool - np.nanmean(
                [rec_m_a, rec_m_b]),
            # rank agreement, same three targets
            "spearman_truth_vs_truth": pooled_truth.split_half_reliability(
                sa, sb),
            "spearman_a_to_pooled": evaluate.spearman(za, pooled),
            "spearman_model_to_single": np.nanmean([
                evaluate.spearman(model, sa), evaluate.spearman(model, sb)]),
            "spearman_model_to_pooled": evaluate.spearman(model, pooled),
            # single-assay error, for the noise budget
            "noise_sd_z": float(np.nanstd(za - zb, ddof=1) / np.sqrt(2.0)),
        })
    return pd.DataFrame(rows)


TIE_DECIMALS = 9
N_SIGNFLIP = 200000


def paired_report(diff, seed=0):
    """Paired test of a per-unit difference, with every method named.

    The input is one difference per unit (a pair, or a protein). Recall at k =
    10 averaged over two targets takes only multiples of 0.05, so the
    differences are heavily TIED, but they are computed by subtracting floats
    and differ in the 17th digit (0.30000000000000004 vs 0.3). Left alone, that
    noise decides which tied differences get which rank, and the Wilcoxon p
    moves with it: on this repo's 32 pairs the unrounded differences give
    p = 0.063 or 0.049 depending on the scipy version and the order the floats
    were produced in, while the properly tied differences give 0.066. So the
    differences are ROUNDED TO 9 DECIMALS before any rank is taken.

    Reported, none chosen for the number it gives:
      p                  scipy.stats.wilcoxon, zero_method='wilcox' (zeros
                         dropped), method='approx' (the normal approximation
                         with scipy's tie correction, which is the one valid
                         with tied ranks)
      p_exact            same, method='exact' (assumes no ties; shown because a
                         reader will try it)
      p_signflip         Monte Carlo sign-flip permutation test on the MEAN
                         difference over the non-zero units (two-sided, N_SIGNFLIP
                         flips, seeded); exact in its null whatever the ties
    """
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    d = np.round(d, TIE_DECIMALS)
    nzd = d[d != 0]
    out = {
        "n_units": int(len(d)),
        "n_zero": int((d == 0).sum()),
        "n_better": int((d > 0).sum()),
        "n_worse": int((d < 0).sum()),
        "median_diff": float(np.median(d)) if len(d) else float("nan"),
        "mean_diff": float(np.mean(d)) if len(d) else float("nan"),
        "tie_rounding_decimals": TIE_DECIMALS,
        "method": "scipy.stats.wilcoxon(zero_method='wilcox', "
                  "method='approx'); differences rounded to "
                  f"{TIE_DECIMALS} decimals so float noise does not split ties",
    }
    if len(d) >= 8 and len(nzd) >= 1:
        w = stats.wilcoxon(d, zero_method="wilcox", method="approx")
        out["statistic"] = float(w.statistic)
        out["p"] = float(w.pvalue)
        try:
            out["p_exact"] = float(stats.wilcoxon(
                d, zero_method="wilcox", method="exact").pvalue)
        except Exception:
            out["p_exact"] = float("nan")
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(N_SIGNFLIP, len(nzd)))
        obs = abs(nzd.mean())
        flips = np.abs((signs * nzd).mean(axis=1))
        out["p_signflip"] = float((np.sum(flips >= obs - 1e-12) + 1)
                                  / (N_SIGNFLIP + 1))
    return out


def protein_level(d, col):
    """One value per protein: the mean of col over that protein's pairs.

    The 32 pairs cover 20 proteins and 14 of the 46 assays appear in more than
    one pair (P53 has 6 pairs), so pairs are not independent. Averaging within
    a protein gives one unit per protein."""
    return d.groupby("uniprot")[col].mean()


def summarise(df, subset_name):
    """Paired statistics. Every comparison here is on the same variants of the
    same pairs -- no median is set against a mean of a different population."""
    d = df.dropna(subset=["recall_model_to_pooled", "recall_model_to_single"])
    out = {
        "subset": subset_name,
        "n_pairs": int(len(d)),
        "n_proteins": int(d["uniprot"].nunique()),
        "n_variants": int(d["n_shared"].sum()),
        # descriptive, per-pair; medians reported because a 10-variant recall
        # takes 11 discrete values and means of it are misleading
        "median_recall_truth_vs_truth": float(
            d["recall_truth_vs_truth"].median()),
        "mean_recall_truth_vs_truth": float(d["recall_truth_vs_truth"].mean()),
        "median_recall_model_to_single": float(
            d["recall_model_to_single"].median()),
        "median_recall_model_to_pooled": float(
            d["recall_model_to_pooled"].median()),
        "median_recall_a_to_pooled": float(d["recall_a_to_pooled"].median()),
        "median_split_half_reliability": float(
            d["split_half_reliability"].median()),
        "median_spearman_model_to_single": float(
            d["spearman_model_to_single"].median()),
        "median_spearman_model_to_pooled": float(
            d["spearman_model_to_pooled"].median()),
        "median_noise_sd_z": float(d["noise_sd_z"].median()),
        "n_low_reliability": int(d["low_reliability"].sum()),
    }
    # Paired tests. PROTEIN-level is the primary one (one unit per protein);
    # PAIR-level is secondary because the pairs share proteins and assays.
    diff = (d["recall_model_to_pooled"] - d["recall_model_to_single"])
    d = d.assign(_recall_gain=diff.to_numpy())
    pair_rep = paired_report(diff.to_numpy())
    if "p" in pair_rep:
        out["wilcoxon_model_pooled_minus_single"] = {
            **pair_rep,
            "level": "pair (secondary)",
            "n_pairs_pooled_better": pair_rep["n_better"],
            "n_pairs_single_better": pair_rep["n_worse"],
        }
    prot_rep = paired_report(protein_level(d, "_recall_gain").to_numpy())
    if "p" in prot_rep:
        out["protein_level_recall_gain"] = {**prot_rep,
                                            "level": "protein (primary)"}
    # and the same comparison on rank correlation, which is not bounded by 11
    diff_r = (d["spearman_model_to_pooled"]
              - d["spearman_model_to_single"])
    d = d.assign(_spearman_gain=diff_r.to_numpy())
    rep_r = paired_report(diff_r.to_numpy())
    if "p" in rep_r:
        out["wilcoxon_spearman_pooled_minus_single"] = {
            **rep_r,
            "level": "pair (secondary)",
            "n_pairs_pooled_better": rep_r["n_better"],
            "n_pairs_single_better": rep_r["n_worse"],
        }
    prot_r = paired_report(protein_level(d, "_spearman_gain").to_numpy())
    if "p" in prot_r:
        out["protein_level_spearman_gain"] = {**prot_r,
                                              "level": "protein (primary)"}
    # does the gain concentrate where the two assays agree least? if it does,
    # the pooled target is least like a latent truth exactly where it helps
    # the model most, which limits how the gain may be read
    m = d[["split_half_reliability", "recall_gain_pooled_vs_single",
           "spearman_model_to_pooled", "spearman_model_to_single"]].dropna()
    if len(m) >= 8 and m["recall_gain_pooled_vs_single"].nunique() > 1 \
            and m["split_half_reliability"].nunique() > 1:
        r_gain, p_gain = stats.spearmanr(
            m["split_half_reliability"], m["recall_gain_pooled_vs_single"])
        spr_gain = m["spearman_model_to_pooled"] - m["spearman_model_to_single"]
        if spr_gain.nunique() > 1:
            r_spr, p_spr = stats.spearmanr(
                m["split_half_reliability"], spr_gain)
        else:
            r_spr, p_spr = np.nan, np.nan
        out["gain_vs_reliability"] = {
            "recall_gain_spearman": float(r_gain),
            "recall_gain_p": float(p_gain),
            "spearman_gain_spearman": float(r_spr),
            "spearman_gain_p": float(p_spr),
            "n_pairs": int(len(m)),
        }

    # truth-vs-truth also rises against the pool: the ceiling itself was
    # attenuated, which is the audit's point stated as a measurement
    # NOTE (review C13): assay A is HALF of the pooled target, so scoring A
    # against the pool counts A's own measurement noise as signal. This gain is
    # therefore partly SELF-INCLUSION, a diagnostic that says nothing about
    # whether 0.350 could be exceeded by a predictor independent of both
    # assays. scripts/sim_pooling_controls.py measures how large the
    # self-inclusion effect is on simulated assays (with no shared signal at
    # all it already lifts A-vs-pool recall far above A-vs-B recall).
    diff_t = (d["recall_a_to_pooled"]
              - d["recall_truth_vs_truth"]).to_numpy()
    rep_t = paired_report(diff_t)
    if "p" in rep_t:
        out["wilcoxon_assay_to_pooled_minus_truth_vs_truth"] = {
            "median_diff": rep_t["median_diff"], "p": rep_t["p"],
            "note": "assay is half of the pooled target: self-inclusion, "
                    "diagnostic only"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-only", action="store_true",
                    help="fill the masked-marginal cache and stop")
    ap.add_argument("--score-budget", type=float, default=None,
                    help="seconds of scoring before stopping cleanly")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--proteins", default=None,
                    help="comma-separated UniProt ids to score; lets several "
                         "processes fill different shards of the cache at "
                         "once, since each protein's table is independent")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ceiling = pd.read_csv(CEILING)
    pairs = pair_shared(ceiling)
    proteins = sorted({p["uniprot"] for p in pairs})
    print(f"{len(pairs)} pairs with >= {MIN_SHARED} shared variants over "
          f"{len(proteins)} proteins", flush=True)

    if args.proteins:
        keep = set(args.proteins.split(","))
        pairs = [p for p in pairs if p["uniprot"] in keep]
        proteins = [p for p in proteins if p in keep]
        print(f"shard: {len(proteins)} proteins", flush=True)

    if args.scores_only or args.score_budget is not None:
        done = protein_tables(proteins, pairs, budget_seconds=args.score_budget,
                              threads=args.threads)
        print("scores complete" if done else "partial", flush=True)
        return

    sequences, tables, missing = load_tables(proteins, pairs)
    if missing:
        raise SystemExit(
            f"missing model tables for {missing}; run with --scores-only")

    df = analyse(pairs, tables)
    df.to_csv(OUT / "per_pair.csv", index=False)

    allp = summarise(df, "all pairs")
    same = summarise(df[~df["different_quantity"]],
                     "same-quantity pairs only")
    reliable = summarise(df[~df["low_reliability"]],
                         "pairs with split-half reliability > "
                         f"{LOW_RELIABILITY}")
    summary = {"threshold_shared_variants": MIN_SHARED,
               "top_frac": TOP_FRAC, "k": K,
               "assays": allp, "same_quantity": same,
               "reliable_only": reliable}
    # the mean over the 217-assay benchmark, on the assays that have a pair,
    # so the README can stop comparing a median of 37 with a mean of 217
    if PER_ASSAY.exists():
        pa = pd.read_csv(PER_ASSAY)
        esm = pa[pa["method"] == "esm2"]
        paired_ids = sorted(set(df["assay_a"]) | set(df["assay_b"]))
        sub = esm[esm["assay"].isin(paired_ids)]
        summary["benchmark_reference"] = {
            "n_all_assays": int(len(esm)),
            "mean_recall_all_assays": float(esm["top10_recall"].mean()),
            "n_paired_assays": int(len(sub)),
            "mean_recall_paired_assays": float(sub["top10_recall"].mean()),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{allp['n_pairs']} pairs, {allp['n_proteins']} proteins, "
          f"{allp['n_variants']} variant measurements")
    print(f"  truth vs truth (the old 0.350):        "
          f"median {allp['median_recall_truth_vs_truth']:.3f}  "
          f"mean {allp['mean_recall_truth_vs_truth']:.3f}")
    print(f"  assay vs pooled target:                 "
          f"median {allp['median_recall_a_to_pooled']:.3f}")
    print(f"  model vs single assay:                  "
          f"median {allp['median_recall_model_to_single']:.3f}")
    print(f"  model vs pooled target:                 "
          f"median {allp['median_recall_model_to_pooled']:.3f}")
    w = allp.get("wilcoxon_model_pooled_minus_single")
    if w:
        print(f"  paired difference (pooled - single):    "
              f"{w['median_diff']:+.3f} median, "
              f"{w['n_pairs_pooled_better']}/{allp['n_pairs']} pairs better, "
              f"p={w['p']:.4f}")
    print(f"  median split-half reliability:          "
          f"{allp['median_split_half_reliability']:.3f} "
          f"({allp['n_low_reliability']} pairs at or below {LOW_RELIABILITY})")


if __name__ == "__main__":
    main()
